#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt --cache-dir .cache/pip
npm ci --no-audit --no-fund --cache .cache/npm
npm run build
PYTHONPATH=backend .venv/bin/python -m pytest
