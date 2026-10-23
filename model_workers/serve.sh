#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${OMNILABEL_WORKER_MODEL:?Set OMNILABEL_WORKER_MODEL=sam2,mmdetection3d,or openoccupancy}"
exec python -m uvicorn model_workers.app:app \
  --host "${MODEL_WORKER_HOST:-127.0.0.1}" --port "${MODEL_WORKER_PORT:-9101}" --workers 1
