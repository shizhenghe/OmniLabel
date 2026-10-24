# OmniLabel GPU model worker

This optional service turns real SAM 3 or MMDetection3D inference into the
review-only proposal schema consumed by OmniLabel. It does not bundle model
code, checkpoints, Hugging Face credentials, or CUDA libraries.

Run SAM 3 and MMDetection3D in separate environments when their PyTorch/CUDA
dependency sets differ. Both workers can use the same source tree.

## SAM 3 worker

Follow the official SAM 3 installation guide and obtain access to its gated
checkpoint. Then install the small HTTP/polygon layer:

```bash
python -m pip install -r requirements-model-worker.txt
export OMNILABEL_SAM3_DEVICE=cuda
# Optional when not using the authenticated Hugging Face download/cache:
export OMNILABEL_SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
export OMNILABEL_SAM3_MODEL_VERSION='facebook/sam3:your-checkpoint-id'
export OMNILABEL_WORKER_PORT=9001
export OMNILABEL_MODEL_WORKER_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/serve-model-worker.sh
```

`prompt` is a comma/Chinese-comma/newline-separated label list, for example
`vehicle, pedestrian, road`. By default every camera in the current frame is
processed. Limit it with `OMNILABEL_SAM3_CAMERAS=front,front_wide`. Optional
controls are `OMNILABEL_SAM3_SCORE_THRESHOLD` (default `0.5`),
`OMNILABEL_SAM3_MIN_AREA` (default `64` pixels),
`OMNILABEL_SAM3_POLYGON_EPSILON` (default `0.002`), and
`OMNILABEL_SAM3_MAX_POLYGONS` (default `1000`). Holes are not representable in
the current OmniLabel polygon schema; the adapter exports simplified external
contours as review candidates.

## MMDetection3D worker

Create an environment matching the official MMEngine/MMCV/MMDetection/
MMDetection3D compatibility table, install this repository's base worker
requirements, and provide a config/checkpoint pair:

```bash
python -m pip install -r requirements-model-worker.txt
export OMNILABEL_MMDET3D_CONFIG=/absolute/path/to/model-config.py
export OMNILABEL_MMDET3D_CHECKPOINT=/absolute/path/to/checkpoint.pth
export OMNILABEL_MMDET3D_DEVICE=cuda:0
export OMNILABEL_MMDET3D_TASK=detection
export OMNILABEL_MMDET3D_POINT_DIM=4
export OMNILABEL_MMDET3D_SCORE_THRESHOLD=0.3
export OMNILABEL_MMDET3D_MODEL_VERSION='project/checkpoint-sha256:...'
export OMNILABEL_WORKER_PORT=9002
export OMNILABEL_MODEL_WORKER_PYTHON="$CONDA_PREFIX/bin/python"
bash scripts/serve-model-worker.sh
```

For a nuScenes-style model that expects five point features, set
`OMNILABEL_MMDET3D_POINT_DIM=5`; the adapter pads missing features with zero.
This makes tensor dimensions compatible but does not make a checkpoint trained
on another LiDAR, taxonomy, intensity normalization, or sweep pattern accurate.
Use a checkpoint validated or fine-tuned for the actual sensor data.

The adapter accepts only `LiDARInstance3DBoxes`, uses their gravity center and
`(length, width, height, yaw)` representation, and rejects camera/depth boxes.
Map checkpoint classes to annotation labels with JSON, for example:

```bash
export OMNILABEL_MMDET3D_CLASS_MAP='{"car":"vehicle","truck":"truck","pedestrian":"pedestrian"}'
```

Set `OMNILABEL_MMDET3D_TASK=segmentation` for a point segmentor. The result is
accepted only when it contains exactly one semantic class prediction per input
point, preserving OmniLabel's original point indices.

## Connect both workers to OmniLabel

In the shell that starts the main OmniLabel service:

```bash
export OMNILABEL_INFERENCE_URLS='{"sam3":"http://127.0.0.1:9001/v1/predict","mmdetection3d":"http://127.0.0.1:9002/v1/predict"}'
export OMNILABEL_INFERENCE_MODELS='sam3,mmdetection3d'
bash scripts/serve.sh
```

If both adapters share one compatible environment, run one worker and use the
existing `OMNILABEL_INFERENCE_URL=http://127.0.0.1:9000/v1/predict` instead.
Set the same secret in `OMNILABEL_WORKER_TOKEN` on each worker and
`OMNILABEL_INFERENCE_TOKEN` on OmniLabel to require bearer authentication.

Check readiness before opening the UI:

```bash
curl http://127.0.0.1:9001/health
curl http://127.0.0.1:9002/health
```

`available: true` means the package and configured files are visible. Models
load lazily on the first prediction, so the first request can still fail due to
GPU memory, checkpoint access, or binary/CUDA incompatibility.
