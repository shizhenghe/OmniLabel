#!/usr/bin/env bash
# Standalone SAM2.1 tiny worker for Linux NVIDIA servers. Main app stays separate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="$PWD"
worker_env="${SAM2_VENV_DIR:-$repo/.model-envs/sam2}"
[[ "$worker_env" = /* ]] || worker_env="$repo/$worker_env"
tiny_checkpoint="$repo/model_weights/sam2/sam2.1_hiera_tiny.pt"
python_bin="${SAM2_PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

usage() {
  cat <<'USAGE'
Usage: bash model_workers/sam2-local.sh install|check|serve
  install  Create isolated environment, install CUDA 12.1 torch and SAM2, get tiny weights.
  check    Check environment, CUDA device, checkpoint and packaged model config.
  serve    Check then start the worker at 127.0.0.1:9101 (one process).
Options: SAM2_PYTHON, SAM2_VENV_DIR, SAM2_CHECKPOINT, SAM2_CONFIG,
         SAM2_PRECISION, SAM2_POINTS_PER_BATCH, CUDA_VISIBLE_DEVICES,
         OMNILABEL_MODEL_DEVICE, MODEL_WORKER_HOST, MODEL_WORKER_PORT.
See docs/sam2-local.md for GPU compatibility and cloud/private-network connections.
USAGE
}

check_runtime() {
  if [[ ! -x "$worker_env/bin/python" ]]; then
    echo 'SAM2 environment missing. Run: bash model_workers/sam2-local.sh install' >&2
    exit 1
  fi
  export SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$tiny_checkpoint}"
  export SAM2_CONFIG="${SAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_t.yaml}"
  export OMNILABEL_MODEL_DEVICE="${OMNILABEL_MODEL_DEVICE:-cuda:0}"
  export SAM2_PRECISION="${SAM2_PRECISION:-auto}"
  export SAM2_POINTS_PER_BATCH="${SAM2_POINTS_PER_BATCH:-16}"
  "$worker_env/bin/python" - <<'PY'
import os
from importlib import resources
from pathlib import Path
import torch
import torchvision
import sam2
from model_workers.adapters import resolve_sam2_precision

checkpoint = Path(os.environ['SAM2_CHECKPOINT'])
if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
    raise SystemExit('Checkpoint missing or empty: set SAM2_CHECKPOINT to trusted SAM2.1 weights.')
config = os.environ['SAM2_CONFIG']
if not resources.files('sam2').joinpath(config).is_file():
    raise SystemExit('SAM2_CONFIG must be a packaged config such as configs/sam2.1/sam2.1_hiera_t.yaml.')
device = os.environ['OMNILABEL_MODEL_DEVICE']
if device.startswith('cuda'):
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Check NVIDIA driver, GPU visibility and the installed torch CUDA build.')
    torch.empty(1, device=device)  # Verify the selected ordinal, not just another visible GPU.
    print('GPU:', torch.cuda.get_device_name(torch.device(device)), 'capability:', torch.cuda.get_device_capability(device))
precision = resolve_sam2_precision(torch, device, os.environ['SAM2_PRECISION'])
print('SAM2 inference precision:', precision, 'automatic-mask points per batch:', os.environ['SAM2_POINTS_PER_BATCH'])
print('PyTorch:', torch.__version__, 'torchvision:', torchvision.__version__, 'CUDA runtime:', torch.version.cuda)
print('Preflight passed. Start the service; /models must report available:true after actual model loading.')
PY
}

case "${1:---help}" in
  -h|--help|help) usage ;;
  install)
    command -v git >/dev/null
    command -v curl >/dev/null
    command -v nvidia-smi >/dev/null || { echo 'Install/check your NVIDIA driver first (nvidia-smi is missing).' >&2; exit 1; }
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    "$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "This pinned torch profile requires Python 3.10–3.12"'
    if [[ -n "${SAM2_CHECKPOINT:-}" && ! -s "$SAM2_CHECKPOINT" ]]; then
      echo 'Custom SAM2_CHECKPOINT must already exist and be nonempty; provide the matching trusted weights first.' >&2
      exit 1
    fi
    if [[ -e "$worker_env" && ! -f "$worker_env/pyvenv.cfg" ]]; then
      echo 'SAM2_VENV_DIR exists but is not a Python venv; choose an empty dedicated directory.' >&2
      exit 1
    fi
    "$python_bin" -m venv "$worker_env"
    "$worker_env/bin/python" -m pip install --upgrade pip setuptools wheel
    "$worker_env/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    # The platform adapter disables SAM2's optional CUDA post-processing anyway;
    # skipping that extension avoids requiring nvcc or a second toolkit install.
    PATH="$worker_env/bin:$PATH" SAM2_BUILD_CUDA=0 bash model_workers/setup.sh sam2
    if [[ -z "${SAM2_CHECKPOINT:-}" && ! -s "$tiny_checkpoint" ]]; then
      mkdir -p "$(dirname "$tiny_checkpoint")"
      checkpoint_part="$(mktemp "${tiny_checkpoint}.download.XXXXXX")"
      trap 'rm -f -- "$checkpoint_part"' EXIT
      curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 --connect-timeout 20 \
        --output "$checkpoint_part" https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
      mv -- "$checkpoint_part" "$tiny_checkpoint"
      trap - EXIT
    fi
    check_runtime
    echo 'Installed. Start: bash model_workers/sam2-local.sh serve'
    ;;
  check) check_runtime ;;
  serve)
    check_runtime
    export OMNILABEL_WORKER_MODEL=sam2
    export MODEL_WORKER_PORT="${MODEL_WORKER_PORT:-9101}"
    export PATH="$worker_env/bin:$PATH"
    exec bash model_workers/serve.sh
    ;;
  *) usage >&2; exit 2 ;;
esac
