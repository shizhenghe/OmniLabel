#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
worker_python="${OMNILABEL_MODEL_WORKER_PYTHON:-python}"
exec "$worker_python" -m uvicorn model_worker.app:app \
  --host "${OMNILABEL_WORKER_HOST:-127.0.0.1}" \
  --port "${OMNILABEL_WORKER_PORT:-9000}"
