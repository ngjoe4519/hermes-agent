"""The Webapp serves this host's desktop plugins to the browser's plugin loader.

Native Desktop reads ``<root>/desktop-plugins`` from its own disk. A browser tab's only disk is the
server, so ``/api/desktop-plugins/*`` answers the same three reads the loader makes (the root, its
folders, a folder's ``plugin.js`` and package marker), whole, and nothing else. The renderer half is
pinned in ``apps/desktop/src/lib/browser-desktop-plugins.test.ts``.
"""
import json

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


def _client(monkeypatch, surface):
    for key, value in (("auth_required", False), ("bound_host", None), ("ui_surface", surface)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


@pytest.fixture
def hermes_root(monkeypatch, tmp_path):
    """A custom Hermes root, served through one of its named profiles."""
    root = tmp_path / "hermes"
    (root / "profiles" / "work").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "work"))
    return root


@pytest.fixture
def webapp(monkeypatch, hermes_root):
    client = _client(monkeypatch, "webapp")
    yield client
    client.close()


def _install(root, name, source, marker=None):
    folder = root / "desktop-plugins" / name
    folder.mkdir(parents=True)
    (folder / "plugin.js").write_text(source, encoding="utf-8")
    if marker is not None:
        (folder / ".hermes-package.json").write_text(json.dumps(marker), encoding="utf-8")
    return folder


def test_webapp_loads_the_app_level_plugins_folder_whole(webapp, hermes_root):
    # Larger than the 512 KiB /api/fs preview cap, which would hand the loader half a module.
    source = "export default { id: 'big' }\n" + "// pad\n" * 120_000
    folder = _install(hermes_root, "big", source, marker={"package": "big-pkg"})

    root = webapp.get("/api/desktop-plugins/root").json()["root"]
    # App-level like Electron's: the root's folder, not the served profile's.
    assert root == str((hermes_root / "desktop-plugins").resolve())

    listed = webapp.get("/api/desktop-plugins/list", params={"path": root}).json()["entries"]
    assert listed == [{"name": "big", "path": str(folder.resolve()), "isDirectory": True}]

    names = {e["name"] for e in webapp.get("/api/desktop-plugins/list", params={"path": listed[0]["path"]}).json()["entries"]}
    assert names == {"plugin.js", ".hermes-package.json"}

    read = webapp.get("/api/desktop-plugins/read", params={"path": str(folder.resolve() / "plugin.js")}).json()
    assert (read["text"], read["truncated"]) == (source, False)
    marker = webapp.get("/api/desktop-plugins/read", params={"path": str(folder.resolve() / ".hermes-package.json")})
    assert json.loads(marker.json()["text"]) == {"package": "big-pkg"}


def test_webapp_reads_nothing_but_plugin_entries_and_markers(webapp, hermes_root):
    folder = _install(hermes_root, "notes", "export default {}")
    (folder / "secrets.txt").write_text("token", encoding="utf-8")
    (folder / "lib").mkdir()
    (folder / "lib" / "plugin.js").write_text("export default {}", encoding="utf-8")
    (hermes_root / ".env").write_text("OPENAI_API_KEY=sk-test", encoding="utf-8")
    (hermes_root / "desktop-plugins" / "plugin.js").write_text("loose", encoding="utf-8")
    plugins = (hermes_root / "desktop-plugins").resolve()

    for path in (
        folder.resolve() / "secrets.txt",
        folder.resolve() / "lib" / "plugin.js",
        plugins / "plugin.js",
        plugins / ".." / ".env",
        hermes_root.resolve() / ".env",
    ):
        assert webapp.get("/api/desktop-plugins/read", params={"path": str(path)}).status_code == 404, path

    for path in (folder.resolve() / "lib", plugins / ".."):
        assert webapp.get("/api/desktop-plugins/list", params={"path": str(path)}).status_code == 404, path


@pytest.mark.parametrize("surface", ["gateway", "dashboard", "serve"])
def test_only_the_webapp_serves_desktop_plugins(monkeypatch, hermes_root, surface):
    _install(hermes_root, "notes", "export default {}")
    client = _client(monkeypatch, surface)
    try:
        assert client.get("/api/desktop-plugins/root").status_code == 404
    finally:
        client.close()
