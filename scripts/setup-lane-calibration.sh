#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -x .venv/bin/python || { echo 'Run bash scripts/setup.sh first.' >&2; exit 1; }
.venv/bin/python -m pip install -r requirements-lane-calibration.txt --cache-dir .cache/pip
.venv/bin/python -m pip check
