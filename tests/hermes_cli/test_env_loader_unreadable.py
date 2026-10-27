"""A present but unreadable profile .env loads like a missing one while managed scope pins the backend.

A sandbox launcher denies the profile's .env to the sandboxed backend, passes its values in as
environment and pins ``terminal.backend`` through managed scope. The backend must still start, and the
launcher's values must survive the load, a later ``/reload`` and a project ``.env``. Without that pin
the file may be the only place the terminal backend is set, so an unreadable file still fails the load.
"""

import builtins
import io
import logging
import os
from pathlib import Path

import pytest

from hermes_cli import managed_scope
from hermes_cli.env_loader import load_hermes_dotenv


def _deny_reads(monkeypatch, *denied: Path) -> None:
    """Make each *denied* file exist but refuse reads, as Windows does for a file a sandbox denies."""
    blocked = {os.path.abspath(path) for path in denied}
    real_open = io.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and os.path.abspath(file) in blocked:
            raise PermissionError(13, "Access is denied", os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)


@pytest.fixture
def managed_dir(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    managed_scope.invalidate_managed_cache()
    yield md
    managed_scope.invalidate_managed_cache()


def _write_managed(md: Path, body: str) -> None:
    (md / "config.yaml").write_text(body, encoding="utf-8")
    managed_scope.invalidate_managed_cache()


@pytest.fixture
def pinned(managed_dir):
    _write_managed(managed_dir, "terminal:\n  backend: busybox\n")
    return managed_dir


def _profile(tmp_path: Path) -> tuple[Path, Path]:
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    return home, env_file


def _unreadable_warnings(caplog, path: Path) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "not readable" in r.getMessage() and str(path) in r.getMessage()]


def test_unreadable_user_env_loads_as_absent(tmp_path, monkeypatch, caplog, pinned):
    home, env_file = _profile(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-launcher")
    _deny_reads(monkeypatch, env_file)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        assert load_hermes_dotenv(hermes_home=home) == []

    assert os.environ["OPENAI_API_KEY"] == "from-launcher"
    assert _unreadable_warnings(caplog, env_file)


def test_unreadable_user_env_keeps_inherited_provider_keys(tmp_path, monkeypatch, pinned):
    """The keys a readable .env clears when it does not define them are the launcher's here."""
    home, env_file = _profile(tmp_path)
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "from-launcher")
    _deny_reads(monkeypatch, env_file)

    load_hermes_dotenv(hermes_home=home)

    assert os.environ["HERMES_ACP_AUTH_METHOD"] == "from-launcher"


@pytest.mark.parametrize("managed_yaml", [
    None,
    "terminal:\n  timeout: 5\n",
    "terminal:\n  backend:\n",
    "terminal:\n  backend: ''\n",
    "terminal:\n  backend: '   '\n",
])
def test_unreadable_user_env_raises_without_a_pinned_backend(tmp_path, monkeypatch, managed_dir, managed_yaml):
    """The file may be the only place the backend is set; skipping it would run commands on the host."""
    if managed_yaml is not None:
        _write_managed(managed_dir, managed_yaml)
    home, env_file = _profile(tmp_path)
    _deny_reads(monkeypatch, env_file)

    with pytest.raises(PermissionError):
        load_hermes_dotenv(hermes_home=home)


def test_unreadable_user_env_warns_once_per_path(tmp_path, monkeypatch, caplog, pinned):
    """Gateways reload per turn; the warning names the file once, while every load still retries it."""
    home, env_file = _profile(tmp_path)
    _deny_reads(monkeypatch, env_file)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        for _ in range(3):
            assert load_hermes_dotenv(hermes_home=home) == []

    assert len(_unreadable_warnings(caplog, env_file)) == 1


def test_project_env_fills_gaps_without_replacing_launcher_keys(tmp_path, monkeypatch, pinned):
    """A withheld profile .env still exists, so a project .env only fills keys nothing else set."""
    home, env_file = _profile(tmp_path)
    project_env = tmp_path / "project.env"
    project_env.write_text("OPENAI_API_KEY=from-project\nLAB_PROJECT_ONLY=from-project\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-launcher")
    monkeypatch.setenv("LAB_PROJECT_ONLY", "placeholder")
    monkeypatch.delenv("LAB_PROJECT_ONLY")
    _deny_reads(monkeypatch, env_file)

    assert load_hermes_dotenv(hermes_home=home, project_env=project_env) == [project_env]

    assert os.environ["OPENAI_API_KEY"] == "from-launcher"
    assert os.environ["LAB_PROJECT_ONLY"] == "from-project"


def test_unreadable_op_env_loads_as_absent(tmp_path, monkeypatch, caplog, pinned):
    """The launcher withholds every profile credential file, the 1Password bootstrap included."""
    home, env_file = _profile(tmp_path)
    op_env = home / ".op.env"
    op_env.write_text("OP_SERVICE_ACCOUNT_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    _deny_reads(monkeypatch, env_file, op_env)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        assert load_hermes_dotenv(hermes_home=home) == []

    assert "OP_SERVICE_ACCOUNT_TOKEN" not in os.environ
    assert _unreadable_warnings(caplog, op_env)


def test_unreadable_op_env_raises_without_a_pinned_backend(tmp_path, monkeypatch, managed_dir):
    home, _env_file = _profile(tmp_path)
    op_env = home / ".op.env"
    op_env.write_text("OP_SERVICE_ACCOUNT_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    _deny_reads(monkeypatch, op_env)

    with pytest.raises(PermissionError):
        load_hermes_dotenv(hermes_home=home)


def test_readable_user_env_still_loads(tmp_path, monkeypatch):
    home, env_file = _profile(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-launcher")

    assert load_hermes_dotenv(hermes_home=home) == [env_file]
    assert os.environ["OPENAI_API_KEY"] == "from-file"


def test_reload_keeps_launcher_keys_when_user_env_is_unreadable(monkeypatch):
    """``/reload`` reads an unreadable .env as empty; it must not delete every key on that basis."""
    from hermes_cli.config import get_env_path, reload_env

    env_file = get_env_path()
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-launcher")
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "from-launcher")
    _deny_reads(monkeypatch, env_file)

    assert reload_env() == 0

    assert os.environ["OPENAI_API_KEY"] == "from-launcher"
    assert os.environ["HERMES_ACP_AUTH_METHOD"] == "from-launcher"


def test_reload_still_removes_keys_a_readable_env_dropped(monkeypatch):
    from hermes_cli.config import get_env_path, reload_env

    get_env_path().write_text("ANTHROPIC_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "stale")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale")

    assert reload_env() == 2

    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"


def test_doctor_reports_an_unreadable_env_file(tmp_path, monkeypatch, capsys):
    from hermes_cli import doctor as doctor_mod
    from hermes_cli.doctor_config import _check_env_file

    home, env_file = _profile(tmp_path)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    _deny_reads(monkeypatch, env_file)

    finding = _check_env_file(False)

    out = capsys.readouterr().out
    assert ".env is not readable" in out
    assert "No API key found" not in out
    assert finding.issues == []


def test_acp_does_not_report_a_withheld_env_as_missing(tmp_path, monkeypatch, caplog, pinned):
    from acp_adapter import entry

    home, env_file = _profile(tmp_path)
    monkeypatch.setattr(entry, "get_hermes_home", lambda: home)
    _deny_reads(monkeypatch, env_file)

    with caplog.at_level(logging.INFO, logger=entry.__name__):
        entry._load_env()

    messages = [r.getMessage() for r in caplog.records if r.name == entry.__name__]
    assert messages
    assert not any("No .env found" in m for m in messages)
