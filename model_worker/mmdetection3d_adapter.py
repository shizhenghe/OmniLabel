from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .common import WorkerError, input_points, merged_annotations, parse_labels


class MMDetection3DAdapter:
    model_id = "mmdetection3d"
    name = "MMDetection3D"
    tasks = ["3d-object-detection", "point-semantic-segmentation"]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None

    def _paths(self) -> tuple[Path | None, Path | None]:
        config = os.environ.get("OMNILABEL_MMDET3D_CONFIG", "")
        checkpoint = os.environ.get("OMNILABEL_MMDET3D_CHECKPOINT", "")
        return (Path(config) if config else None, Path(checkpoint) if checkpoint else None)

    def status(self) -> dict[str, Any]:
        config, checkpoint = self._paths()
        installed = importlib.util.find_spec("mmdet3d") is not None and importlib.util.find_spec("torch") is not None
        configured = bool(installed and config and config.is_file() and checkpoint and checkpoint.is_file())
        return {
            "id": self.model_id,
            "name": self.name,
            "tasks": self.tasks,
            "available": configured,
            "loaded": self._model is not None,
            "detail": "ready for lazy loading" if configured else "install MMDetection3D and set its config and checkpoint paths",
        }

    def _load(self):
        if self._model is not None:
            return self._model
        config, checkpoint = self._paths()
        if not config or not checkpoint or not config.is_file() or not checkpoint.is_file():
            raise WorkerError("MMDetection3D config or checkpoint is not configured")
        try:
            import torch
            from mmdet3d.apis import init_model
        except ImportError:
            raise WorkerError("MMDetection3D runtime is not installed in the worker environment") from None
        device = os.environ.get("OMNILABEL_MMDET3D_DEVICE", "cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise WorkerError("MMDetection3D is configured for CUDA but no CUDA device is available")
        try:
            self._model = init_model(str(config), str(checkpoint), device=device)
        except Exception:
            raise WorkerError("MMDetection3D could not load; check config/checkpoint compatibility and worker logs") from None
        return self._model

    @staticmethod
    def _class_map() -> dict[str, str]:
        raw = os.environ.get("OMNILABEL_MMDET3D_CLASS_MAP", "{}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            raise WorkerError("OMNILABEL_MMDET3D_CLASS_MAP must be a JSON object") from None
        if not isinstance(value, dict) or any(not isinstance(key, str) or not isinstance(label, str) for key, label in value.items()):
            raise WorkerError("OMNILABEL_MMDET3D_CLASS_MAP must map strings to strings")
        return value

    @staticmethod
    def _identifier(frame_id: str, label: str, values: list[float]) -> str:
        digest = hashlib.sha1(json.dumps([label, *values], separators=(",", ":")).encode()).hexdigest()[:12]
        return f"mmd3d-{frame_id[:40]}-{digest}"

    def _detect(self, model, points: np.ndarray, request: dict[str, Any], annotations: dict[str, Any]) -> None:
        from mmdet3d.apis import inference_detector

        result = inference_detector(model, points)
        if isinstance(result, tuple):
            result = result[0]
        predictions = getattr(result, "pred_instances_3d", None)
        if predictions is None:
            raise WorkerError("MMDetection3D detector returned no pred_instances_3d")
        boxes = predictions.bboxes_3d
        if type(boxes).__name__ != "LiDARInstance3DBoxes":
            raise WorkerError("MMDetection3D output boxes are not in LiDAR coordinates")
        centers = boxes.gravity_center.detach().cpu().numpy()
        dimensions = boxes.dims.detach().cpu().numpy()
        yaws = boxes.yaw.detach().cpu().numpy()
        scores = predictions.scores_3d.detach().cpu().numpy()
        indices = predictions.labels_3d.detach().cpu().numpy()
        classes = list(getattr(model, "dataset_meta", {}).get("classes", []))
        if not classes:
            raise WorkerError("MMDetection3D checkpoint does not provide class metadata")
        threshold = float(os.environ.get("OMNILABEL_MMDET3D_SCORE_THRESHOLD", "0.3"))
        mapping = self._class_map()
        selected = set(parse_labels(str(request.get("prompt", ""))))
        frame_id = str(request["frame"]["id"])
        for center, size, yaw, score, class_index in zip(centers, dimensions, yaws, scores, indices):
            if float(score) < threshold or not 0 <= int(class_index) < len(classes):
                continue
            raw_label = str(classes[int(class_index)])
            label = mapping.get(raw_label, raw_label)
            if selected and raw_label not in selected and label not in selected:
                continue
            values = [float(item) for item in (*center, *size, yaw)]
            if not all(math.isfinite(item) for item in values) or any(item <= 0 for item in values[3:6]):
                continue
            identifier = self._identifier(frame_id, label, values)
            annotations["boxes"].append({
                "id": identifier,
                "track_id": identifier,
                "label": label,
                "center": values[:3],
                "size": values[3:6],
                "yaw": values[6],
            })

    def _segment(self, model, points: np.ndarray, annotations: dict[str, Any]) -> None:
        from mmdet3d.apis import inference_segmentor

        with tempfile.NamedTemporaryFile(suffix=".bin") as stream:
            points.tofile(stream)
            stream.flush()
            result = inference_segmentor(model, stream.name)
        if isinstance(result, tuple):
            result = result[0]
        prediction = getattr(getattr(result, "pred_pts_seg", None), "pts_semantic_mask", None)
        if prediction is None:
            raise WorkerError("MMDetection3D segmentor returned no point semantic mask")
        prediction = prediction.detach().cpu().numpy().reshape(-1)
        if len(prediction) != len(points):
            raise WorkerError("MMDetection3D semantic output does not preserve input point indices")
        classes = list(getattr(model, "dataset_meta", {}).get("classes", []))
        mapping = self._class_map()
        labels = {}
        for index, class_index in enumerate(prediction):
            if 0 <= int(class_index) < len(classes):
                raw_label = str(classes[int(class_index)])
                labels[str(index)] = mapping.get(raw_label, raw_label)
        annotations["point_labels"] = labels

    def predict(self, request: dict[str, Any]) -> tuple[dict[str, Any], str]:
        task = os.environ.get("OMNILABEL_MMDET3D_TASK", "detection")
        if task not in ("detection", "segmentation"):
            raise WorkerError("OMNILABEL_MMDET3D_TASK must be detection or segmentation")
        target_dim = int(os.environ.get("OMNILABEL_MMDET3D_POINT_DIM", "4"))
        points = input_points(request.get("frame", {}), target_dim=target_dim)
        annotations = merged_annotations(request, replace_prefix="mmd3d-")
        with self._lock:
            model = self._load()
            if task == "detection":
                self._detect(model, points, request, annotations)
            else:
                self._segment(model, points, annotations)
        version = os.environ.get("OMNILABEL_MMDET3D_MODEL_VERSION")
        if not version:
            _, checkpoint = self._paths()
            version = f"mmdetection3d:{checkpoint.name if checkpoint else 'unknown'}"
        return annotations, version

