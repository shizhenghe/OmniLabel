#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f dist/index.html || ! -x .venv/bin/python ]]; then
  echo 'Run bash scripts/setup.sh before starting OmniLabel.' >&2
  exit 1
fi
export PYTHONPATH="$PWD/backend${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m uvicorn omnilabel.main:app \
  --host "${OMNILABEL_HOST:-127.0.0.1}" --port "${OMNILABEL_PORT:-8000}"
