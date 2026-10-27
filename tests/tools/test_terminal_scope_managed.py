"""Managed scope pins the per-turn terminal policy the way it pins the standalone bridge.

A sandbox launcher pins ``terminal.backend`` through managed scope and withholds the profile's .env.
The per-turn scope must take the pinned backend over the profile's own ``config.yaml``, and an
unreadable .env must block turns only while nothing pins the backend.
"""

import json
import logging
import textwrap
from pathlib import Path

import pytest

from tools.terminal_scope import TerminalPolicyUnavailable, build_profile_terminal_scope, default_terminal_scope


@pytest.fixture
def managed(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    managed_scope.invalidate_managed_cache()
    yield md
    managed_scope.invalidate_managed_cache()


def _write_managed(md: Path, body: str) -> None:
    from hermes_cli import managed_scope

    (md / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    managed_scope.invalidate_managed_cache()


def _profile(tmp_path: Path, *, config_yaml: str = "", dotenv: str = "") -> Path:
    home = tmp_path / "profiles" / "work"
    home.mkdir(parents=True)
    if config_yaml:
        (home / "config.yaml").write_text(textwrap.dedent(config_yaml), encoding="utf-8")
    if dotenv:
        (home / ".env").write_text(dotenv, encoding="utf-8")
    return home


def _deny_reads(monkeypatch, denied: Path) -> None:
    real_read_bytes = Path.read_bytes

    def read_bytes(self):
        if self == denied:
            raise PermissionError(13, "Access is denied", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)


def test_managed_backend_beats_profile_config(tmp_path, managed):
    _write_managed(managed, "terminal:\n  backend: busybox\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: local\n  timeout: 77\n")

    scope = build_profile_terminal_scope(home)

    assert scope["TERMINAL_ENV"] == "busybox"
    assert scope["TERMINAL_TIMEOUT"] == "77"


def test_unreadable_env_builds_when_backend_is_pinned(tmp_path, managed, monkeypatch):
    _write_managed(managed, "terminal:\n  backend: busybox\n")
    home = _profile(tmp_path, dotenv="TERMINAL_ENV=docker\n")
    _deny_reads(monkeypatch, home / ".env")

    scope = build_profile_terminal_scope(home, env_overlay={"TERMINAL_TIMEOUT": "42"})

    assert scope["TERMINAL_ENV"] == "busybox"
    assert scope["TERMINAL_TIMEOUT"] == "42"


def test_unreadable_env_fails_closed_without_a_pinned_backend(tmp_path, managed, monkeypatch):
    _write_managed(managed, "terminal:\n  timeout: 5\n")
    home = _profile(tmp_path, dotenv="TERMINAL_ENV=docker\n")
    _deny_reads(monkeypatch, home / ".env")

    with pytest.raises(TerminalPolicyUnavailable):
        build_profile_terminal_scope(home)


def test_managed_backend_beats_env_and_launch_overlay(tmp_path, managed):
    _write_managed(managed, "terminal:\n  backend: busybox\n")
    home = _profile(tmp_path, dotenv="TERMINAL_ENV=docker\n")

    scope = build_profile_terminal_scope(home, env_overlay={"TERMINAL_ENV": "ssh"})

    assert scope["TERMINAL_ENV"] == "busybox"


def test_managed_values_expand_against_the_process_environment(tmp_path, managed, monkeypatch):
    monkeypatch.setenv("LAB_PINNED_BACKEND", "busybox")
    _write_managed(managed, "terminal:\n  backend: ${LAB_PINNED_BACKEND}\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: local\n")

    assert build_profile_terminal_scope(home)["TERMINAL_ENV"] == "busybox"


def test_no_managed_scope_keeps_the_profile_backend(tmp_path, managed):
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: local\n")

    assert build_profile_terminal_scope(home)["TERMINAL_ENV"] == "local"


@pytest.mark.parametrize("key, env_var", [
    ("docker_env", "TERMINAL_DOCKER_ENV"),
    ("docker_volumes", "TERMINAL_DOCKER_VOLUMES"),
])
def test_managed_mapping_merges_with_the_profile_mapping(tmp_path, managed, key, env_var):
    """Managed pins the entries it names; the profile keeps the rest, as in load_config()."""
    _write_managed(managed, f"terminal:\n  {key}:\n    A: managed\n")
    home = _profile(tmp_path, config_yaml=f"terminal:\n  {key}:\n    A: profile\n    B: keep\n")

    scope = build_profile_terminal_scope(home)

    assert json.loads(scope[env_var]) == {"A": "managed", "B": "keep"}


def test_managed_list_replaces_the_profile_list(tmp_path, managed):
    _write_managed(managed, "terminal:\n  docker_forward_env: [MANAGED]\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  docker_forward_env: [PROFILE_A, PROFILE_B]\n")

    assert json.loads(build_profile_terminal_scope(home)["TERMINAL_DOCKER_FORWARD_ENV"]) == ["MANAGED"]


@pytest.mark.parametrize("backend_yaml", ["", "''", "'  '", "null"])
def test_empty_managed_backend_is_not_a_pin(tmp_path, managed, monkeypatch, backend_yaml):
    """An empty backend names nothing to run, so a withheld .env still fails closed."""
    _write_managed(managed, f"terminal:\n  backend: {backend_yaml}\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: local\n", dotenv="TERMINAL_ENV=docker\n")
    _deny_reads(monkeypatch, home / ".env")

    with pytest.raises(TerminalPolicyUnavailable):
        build_profile_terminal_scope(home)


def test_null_managed_values_leave_the_profile_values(tmp_path, managed):
    _write_managed(managed, "terminal:\n  backend:\n  docker_image:\n")
    home = _profile(tmp_path, config_yaml="terminal:\n  backend: docker\n")

    scope = build_profile_terminal_scope(home)

    assert scope["TERMINAL_ENV"] == "docker"
    assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "0"


def test_managed_docker_image_pins_the_image(tmp_path, managed):
    _write_managed(managed, "terminal:\n  docker_image: lab/image:1\n")
    home = _profile(tmp_path)

    scope = build_profile_terminal_scope(home)

    assert scope["TERMINAL_DOCKER_IMAGE"] == "lab/image:1"
    assert scope["TERMINAL_DOCKER_IMAGE_PINNED"] == "1"


def test_withheld_env_terminal_keys_fall_back_to_defaults(tmp_path, managed, monkeypatch, caplog):
    """A withheld .env supplies nothing: its TERMINAL_* keys take their defaults unless managed scope
    (or the trusted launch overlay) sets them. The warning names the file once."""
    _write_managed(managed, "terminal:\n  backend: busybox\n  docker_network: false\n")
    home = _profile(tmp_path, dotenv="TERMINAL_TIMEOUT=999\nTERMINAL_DOCKER_NETWORK=false\n")
    _deny_reads(monkeypatch, home / ".env")
    defaults = default_terminal_scope()

    with caplog.at_level(logging.WARNING, logger="tools.terminal_scope"):
        scope = build_profile_terminal_scope(home)
        build_profile_terminal_scope(home)

    assert scope["TERMINAL_TIMEOUT"] == defaults["TERMINAL_TIMEOUT"] != "999"
    assert scope["TERMINAL_DOCKER_NETWORK"] == "False"
    named = [r for r in caplog.records if str(home / ".env") in r.getMessage()]
    assert len(named) == 1


def test_managed_ref_resolves_from_the_process_env_on_a_routed_turn(tmp_path, managed, monkeypatch):
    """A routed turn binds the secondary profile's secret scope before building its terminal policy;
    a managed ${VAR} still reads the launch process, whatever that profile's .env holds."""
    from tools.terminal_scope import get_terminal_scope
    from tui_gateway import server

    monkeypatch.setenv("LAB_PINNED_BACKEND", "busybox")
    _write_managed(managed, "terminal:\n  backend: ${LAB_PINNED_BACKEND}\n")
    for dotenv in ("OTHER_KEY=x\n", "LAB_PINNED_BACKEND=local\n"):
        home = tmp_path / dotenv[:5]
        home.mkdir()
        (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
        (home / ".env").write_text(dotenv, encoding="utf-8")

        scopes = server._profile_runtime_scope_tokens(home, hydrate_secrets=False)
        try:
            assert get_terminal_scope()["TERMINAL_ENV"] == "busybox"
        finally:
            server._release_profile_runtime_scope_tokens(scopes)
