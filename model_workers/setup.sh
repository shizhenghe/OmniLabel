#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${1:-}"
case "$model" in
  sam2|latr|mmdetection3d|openoccupancy) ;;
  *) echo 'Usage: bash model_workers/setup.sh sam2|latr|mmdetection3d|openoccupancy' >&2; exit 2 ;;
esac
# Run within the selected model's existing GPU Python environment. Do not
# replace CUDA/PyTorch versions or combine incompatible OpenMMLab generations.
python -c 'import torch; print("Existing PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'
python -m pip install -r model_workers/requirements-worker.txt
case "$model" in
  sam2)
    python -c 'import sys; assert sys.version_info >= (3,10), "SAM2 requires Python 3.10+"'
    python -m pip install 'setuptools>=64' wheel
    # Use the already selected torch build for model compilation. Build isolation
    # otherwise downloads another torch runtime from SAM2's build requirements.
    python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@${SAM2_SOURCE_REF:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
    ;;
  latr)
    : "${LATR_REPO:?Set LATR_REPO to the trusted LATR source checkout}"
    python -m pip install fvcore geffnet
    python -c 'from mmcv.utils import Config; import mmdet, mmdet3d, mmseg, cv2, scipy, matplotlib, geffnet, fvcore; print("LATR legacy OpenMMLab runtime is importable")'
    ;;
  mmdetection3d)
    python -m pip install openmim
    python -m mim install 'mmengine>=0.8.0,<1.0.0' 'mmcv==2.1.0' 'mmdet==3.3.0'
    python -m pip install 'mmdet3d==1.4.0'
    ;;
  openoccupancy)
    : "${OPENOCCUPANCY_REPO:?Set OPENOCCUPANCY_REPO to the official source checkout with its supported legacy environment}"
    python -c 'from mmcv import Config; from mmcv.runner import load_checkpoint; import mmdet3d; print("Legacy MMDetection3D:", mmdet3d.__version__)'
    python -m pip install -e "$OPENOCCUPANCY_REPO"
    ;;
esac
echo 'Runtime installed. Configure the matching checkpoint/config and start model_workers/serve.sh.'
