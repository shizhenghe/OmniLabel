"""A pull/restart must never serve a previous checkout's ignored dist bundle."""
import importlib.util
import os
from pathlib import Path


spec = importlib.util.spec_from_file_location("frontend_build", Path(__file__).parents[2] / "scripts/frontend_build.py")
frontend_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frontend_build)


def fixture(root):
    (root / "src").mkdir()
    (root / "src" / "App.tsx").write_text("old source")
    (root / "package.json").write_text('{}')
    (root / "dist" / "assets").mkdir(parents=True)
    (root / "dist" / "index.html").write_text('<script src="/assets/app.js"></script>')
    (root / "dist" / "assets" / "app.js").write_text("old bundle")


def test_source_content_changes_with_identical_mtime_require_rebuild(tmp_path):
    fixture(tmp_path)
    frontend_build.write_stamp(tmp_path)
    assert frontend_build.build_is_current(tmp_path)
    path = tmp_path / "src" / "App.tsx"
    previous = path.stat()
    path.write_text("new source")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert not frontend_build.build_is_current(tmp_path)


def test_missing_bundle_asset_and_configuration_change_are_detected(tmp_path):
    fixture(tmp_path)
    frontend_build.write_stamp(tmp_path)
    (tmp_path / "dist" / "assets" / "app.js").unlink()
    assert not frontend_build.build_is_current(tmp_path)
    (tmp_path / "dist" / "assets" / "app.js").write_text("new bundle")
    frontend_build.write_stamp(tmp_path)
    (tmp_path / "vite.config.ts").write_text("updated config")
    assert not frontend_build.build_is_current(tmp_path)


def test_unchanged_source_does_not_rebuild_and_ignores_data_changes(tmp_path, monkeypatch):
    fixture(tmp_path)
    frontend_build.write_stamp(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "labels.json").write_text("new annotation")
    monkeypatch.setattr(frontend_build.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unnecessary build")))
    frontend_build.ensure_build(tmp_path)
