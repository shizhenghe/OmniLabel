#!/usr/bin/env bash
# Use an existing LATR/OpenMMLab environment as an isolated OmniLabel worker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export LATR_REPO="${LATR_REPO:-/data2/szh_data/LATR-main}"
export LATR_ONCE_CHECKPOINT="${LATR_ONCE_CHECKPOINT:-${LATR_CHECKPOINT:-$LATR_REPO/ckpt/once.pth}}"
export LATR_ONCE_CONFIG="${LATR_ONCE_CONFIG:-${LATR_CONFIG:-$LATR_REPO/config/release_iccv/once.py}}"
export LATR_OPENLANE_CHECKPOINT="${LATR_OPENLANE_CHECKPOINT:-$LATR_REPO/ckpt/openlane.pth}"
export LATR_OPENLANE_CONFIG="${LATR_OPENLANE_CONFIG:-$LATR_REPO/config/release_iccv/latr_1000_baseline.py}"
# Backward-compatible ONCE bindings used by older deployments.
export LATR_CHECKPOINT="$LATR_ONCE_CHECKPOINT"
export LATR_CONFIG="$LATR_ONCE_CONFIG"
export LATR_DEFAULT_VARIANT="${LATR_DEFAULT_VARIANT:-once}"
export OMNILABEL_MODEL_DEVICE="${OMNILABEL_MODEL_DEVICE:-cuda:0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
default_latr_python=python
if [[ -x /opt/anaconda3/envs/Latr/bin/python ]]; then
  default_latr_python=/opt/anaconda3/envs/Latr/bin/python
fi
latr_python="${LATR_PYTHON:-$default_latr_python}"

usage() {
  cat <<'USAGE'
Usage: bash model_workers/latr-local.sh prepare|check|serve
  prepare  Add OmniLabel worker packages, fvcore and geffnet to the active LATR Python environment.
  check    Verify source, ONCE/OpenLane checkpoints, configs, imports and selected CUDA device.
  serve    Check both variants, load the default, then serve /predict and /models on 127.0.0.1:9104.
Options: LATR_PYTHON, LATR_REPO, LATR_ONCE_CHECKPOINT, LATR_ONCE_CONFIG,
         LATR_OPENLANE_CHECKPOINT, LATR_OPENLANE_CONFIG, LATR_DEFAULT_VARIANT,
         OMNILABEL_MODEL_DEVICE, CUDA_VISIBLE_DEVICES, MODEL_WORKER_HOST,
         MODEL_WORKER_PORT, MODEL_WORKER_TOKEN, LATR_VISIBILITY_THRESHOLD.
USAGE
}

check_runtime() {
  PYTHONPATH="$PWD:$LATR_REPO${PYTHONPATH:+:$PYTHONPATH}" "$latr_python" - <<'PY'
import hashlib
import os
from pathlib import Path

repo = Path(os.environ['LATR_REPO'])
variants = {
    'once': (Path(os.environ['LATR_ONCE_CHECKPOINT']), Path(os.environ['LATR_ONCE_CONFIG']),
             '65a6958c162e3c7be0960bceb3f54650', 2),
    'openlane': (Path(os.environ['LATR_OPENLANE_CHECKPOINT']), Path(os.environ['LATR_OPENLANE_CONFIG']),
                 'd8ecb900c34fd23a9e7af840aff00843', 21),
}
for path, description in ((repo / 'models' / 'latr.py', 'LATR source'),):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(description + ' missing or empty: ' + str(path))
for name, (checkpoint, config, expected, _) in variants.items():
    for path, description in ((checkpoint, name + ' checkpoint'), (config, name + ' config')):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(description + ' missing or empty: ' + str(path))
    checksum = hashlib.md5()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    digest = checksum.hexdigest()
    if digest != expected:
        raise SystemExit('LATR ' + name + ' checkpoint MD5 mismatch: ' + digest)
import torch
import torchvision
import cv2
import fvcore
import scipy
import matplotlib
import geffnet
from mmcv.utils import Config
import mmdet
import mmdet3d
import mmseg
from models.latr import LATR

device = torch.device(os.environ['OMNILABEL_MODEL_DEVICE'])
if device.type == 'cuda':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Check the NVIDIA driver, GPU visibility and this LATR torch build.')
    torch.empty(1, device=device)
    print('GPU:', torch.cuda.get_device_name(device))
for name, (_, config, expected_digest, categories) in variants.items():
    cfg = Config.fromfile(str(config))
    if int(cfg.num_category) != categories:
        raise SystemExit('LATR ' + name + ' config does not match its checkpoint.')
    print(name + ' checkpoint MD5:', expected_digest)
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('Preflight passed for ONCE and OpenLane. /models becomes available:true after the default model loads.')
PY
}

case "${1:---help}" in
  -h|--help|help) usage ;;
  prepare)
    export PATH="$(dirname "$(command -v "$latr_python")"):$PATH"
    bash model_workers/setup.sh latr
    ;;
  check) check_runtime ;;
  serve)
    check_runtime
    export OMNILABEL_WORKER_MODEL=latr
    export MODEL_WORKER_PORT="${MODEL_WORKER_PORT:-9104}"
    export PYTHONPATH="$PWD:$LATR_REPO${PYTHONPATH:+:$PYTHONPATH}"
    exec "$latr_python" -m uvicorn model_workers.app:app \
      --host "${MODEL_WORKER_HOST:-127.0.0.1}" --port "$MODEL_WORKER_PORT" --workers 1
    ;;
  *) usage >&2; exit 2 ;;
esac
