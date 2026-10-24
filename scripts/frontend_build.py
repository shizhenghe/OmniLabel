"""Detect stale frontend bundles using source contents, including restored checkouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAMP = ".source-build.json"


def source_fingerprint(root: Path) -> str:
    files = {root / "index.html", root / "package.json", root / "package-lock.json",
             root / "scripts" / "frontend_build.py"}
    files.update(root.glob("tsconfig*.json"))
    files.update(root.glob("vite.config.*"))
    for folder in ("src", "public"):
        files.update(path for path in (root / folder).rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def build_is_current(root: Path) -> bool:
    try:
        stamp = json.loads((root / "dist" / STAMP).read_text())
        if not isinstance(stamp, dict):
            return False
        if stamp.get("source") != source_fingerprint(root) or not (root / "dist" / "index.html").is_file():
            return False
        assets = stamp.get("assets")
        if not isinstance(assets, dict) or not assets:
            return False
        return all((root / "dist" / name).is_file() and
                   hashlib.sha256((root / "dist" / name).read_bytes()).hexdigest() == digest
                   for name, digest in assets.items())
    except (OSError, ValueError, TypeError):
        return False


def write_stamp(root: Path) -> None:
    dist = root / "dist"
    if not (dist / "index.html").is_file():
        raise RuntimeError("Frontend build did not produce dist/index.html")
    assets = {p.relative_to(dist).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(dist.rglob("*")) if p.is_file() and p.name != STAMP}
    (dist / STAMP).write_text(json.dumps({"source": source_fingerprint(root), "assets": assets}))


def ensure_build(root: Path) -> None:
    if build_is_current(root):
        return
    print("Frontend source changed or bundle is missing; building the current checkout...", flush=True)
    try:
        subprocess.run(["npm", "run", "build"], cwd=root, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("npm is required to rebuild this checkout. Install Node.js, then run bash scripts/setup.sh.") from exc
    # Also stamp when a downstream package script omits the postbuild hook.
    write_stamp(root)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) == 2 else ""
    if command == "stamp":
        write_stamp(ROOT)
    elif command == "ensure":
        ensure_build(ROOT)
    elif command == "check":
        raise SystemExit(0 if build_is_current(ROOT) else 1)
    else:
        raise SystemExit("Usage: python scripts/frontend_build.py stamp|check|ensure")
