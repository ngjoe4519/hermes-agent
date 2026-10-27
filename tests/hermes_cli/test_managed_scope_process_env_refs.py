"""Managed ``${VAR}`` refs expand against the launch process's environment only.

A routed turn binds the secondary profile's secret scope before it loads config. User refs must
resolve through that scope, but a managed ref is administrator (or sandbox launcher) policy: a profile
whose .env lacks the variable must not get the literal ``${VAR}``, and one that defines it must not
override the pin.
"""

import textwrap

import pytest

from agent import secret_scope as ss


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    _invalidate()
    return home, managed


def _invalidate() -> None:
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


def _write(path, body: str) -> None:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    _invalidate()


@pytest.fixture
def routed_scope(tmp_path):
    """Bind a secret scope stamped with a secondary profile's home, as a routed turn does."""
    tokens = []

    def bind(mapping: dict) -> None:
        tokens.append(ss.set_secret_scope(mapping, profile_home=str(tmp_path / "profiles" / "b")))
        _invalidate()

    yield bind
    for token in reversed(tokens):
        ss.reset_secret_scope(token)


@pytest.mark.parametrize("profile_secrets", [{}, {"LAB_PINNED_BACKEND": "local"}])
def test_managed_overlay_ref_ignores_the_routed_profile(homes, monkeypatch, routed_scope, profile_secrets):
    from hermes_cli import managed_scope

    _home, managed = homes
    monkeypatch.setenv("LAB_PINNED_BACKEND", "busybox")
    _write(managed / "config.yaml", "terminal:\n  backend: ${LAB_PINNED_BACKEND}\n")
    routed_scope(profile_secrets)

    assert managed_scope.apply_managed_overlay({})["terminal"]["backend"] == "busybox"
    assert managed_scope.pinned_terminal_backend() == "busybox"


@pytest.mark.parametrize("profile_secrets", [{}, {"LAB_PINNED_BACKEND": "local"}])
def test_load_config_managed_ref_ignores_the_routed_profile(homes, monkeypatch, routed_scope, profile_secrets):
    from hermes_cli.config import cfg_get, load_config

    home, managed = homes
    monkeypatch.setenv("LAB_PINNED_BACKEND", "busybox")
    _write(home / "config.yaml", "terminal:\n  backend: local\n")
    _write(managed / "config.yaml", "terminal:\n  backend: ${LAB_PINNED_BACKEND}\n")
    routed_scope(profile_secrets)

    assert cfg_get(load_config(), "terminal", "backend") == "busybox"


def test_user_refs_still_resolve_through_the_routed_profile(homes, monkeypatch, routed_scope):
    from hermes_cli.config import cfg_get, load_config

    home, managed = homes
    monkeypatch.setenv("LAB_USER_MODEL", "launch/model")
    monkeypatch.setenv("LAB_PINNED_BACKEND", "busybox")
    _write(home / "config.yaml", "model:\n  default: ${LAB_USER_MODEL}\n")
    _write(managed / "config.yaml", "terminal:\n  backend: ${LAB_PINNED_BACKEND}\n")
    routed_scope({"LAB_USER_MODEL": "profile/model"})

    config = load_config()

    assert cfg_get(config, "model", "default") == "profile/model"
    assert cfg_get(config, "terminal", "backend") == "busybox"


def test_expansion_leaves_the_bound_scope_in_place(homes, monkeypatch, routed_scope):
    from hermes_cli import managed_scope

    _home, managed = homes
    monkeypatch.setenv("LAB_PINNED_BACKEND", "busybox")
    _write(managed / "config.yaml", "terminal:\n  backend: ${LAB_PINNED_BACKEND}\n")
    routed_scope({"PROFILE_ONLY": "x"})

    managed_scope.apply_managed_overlay({})

    assert ss.current_secret_scope() == {"PROFILE_ONLY": "x"}
