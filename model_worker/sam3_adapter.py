from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .common import WorkerError, decode_data_image, merged_annotations, parse_labels


class Sam3Adapter:
    model_id = "sam3"
    name = "SAM 3"
    tasks = ["image-instance-segmentation", "text-prompt-segmentation"]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processor = None

    def status(self) -> dict[str, Any]:
        installed = importlib.util.find_spec("sam3") is not None and importlib.util.find_spec("torch") is not None
        checkpoint = os.environ.get("OMNILABEL_SAM3_CHECKPOINT", "")
        configured = installed and (not checkpoint or Path(checkpoint).is_file())
        return {
            "id": self.model_id,
            "name": self.name,
            "tasks": self.tasks,
            "available": configured,
            "loaded": self._processor is not None,
            "detail": "ready for lazy loading" if configured else "install SAM 3 and configure an accessible checkpoint or Hugging Face login",
        }

    def _load(self):
        if self._processor is not None:
            return self._processor
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError:
            raise WorkerError("SAM 3 runtime is not installed in the worker environment") from None
        device = os.environ.get("OMNILABEL_SAM3_DEVICE", "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise WorkerError("SAM 3 is configured for CUDA but no CUDA device is available")
        checkpoint = os.environ.get("OMNILABEL_SAM3_CHECKPOINT") or None
        if checkpoint and not Path(checkpoint).is_file():
            raise WorkerError("configured SAM 3 checkpoint is unavailable")
        try:
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=checkpoint,
                load_from_HF=checkpoint is None,
            )
            threshold = float(os.environ.get("OMNILABEL_SAM3_SCORE_THRESHOLD", "0.5"))
            self._processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
        except Exception:
            raise WorkerError("SAM 3 could not load; check checkpoint access, CUDA, and worker logs") from None
        return self._processor

    @staticmethod
    def _polygons(mask: np.ndarray) -> list[list[list[float]]]:
        try:
            import cv2
        except ImportError:
            raise WorkerError("SAM 3 polygon conversion requires opencv-python-headless") from None
        binary = np.asarray(mask).squeeze().astype(np.uint8)
        if binary.ndim != 2:
            raise WorkerError("SAM 3 returned a mask with an unsupported shape")
        minimum_area = float(os.environ.get("OMNILABEL_SAM3_MIN_AREA", "64"))
        epsilon_ratio = float(os.environ.get("OMNILABEL_SAM3_POLYGON_EPSILON", "0.002"))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                continue
            approximation = cv2.approxPolyDP(contour, epsilon_ratio * cv2.arcLength(contour, True), True)
            points = [[float(point[0][0]), float(point[0][1])] for point in approximation]
            if len(points) >= 3:
                polygons.append(points)
        return polygons

    def predict(self, request: dict[str, Any]) -> tuple[dict[str, Any], str]:
        labels = parse_labels(str(request.get("prompt", "")))
        if not labels:
            raise WorkerError("SAM 3 requires one or more comma-separated text labels")
        clip = request.get("clip", {})
        frame = request.get("frame", {})
        cameras = {camera.get("id"): camera for camera in clip.get("cameras", []) if isinstance(camera, dict)}
        images = frame.get("images", {})
        if not isinstance(images, dict) or not images:
            raise WorkerError("SAM 3 requires at least one camera image")
        selected = {item.strip() for item in os.environ.get("OMNILABEL_SAM3_CAMERAS", "").split(",") if item.strip()}
        annotations = merged_annotations(request, replace_prefix="sam3-")
        maximum = int(os.environ.get("OMNILABEL_SAM3_MAX_POLYGONS", "1000"))

        with self._lock:
            processor = self._load()
            counter = 0
            for camera_id, reference in images.items():
                if selected and camera_id not in selected:
                    continue
                camera = cameras.get(camera_id)
                if camera is None or not isinstance(reference, str):
                    raise WorkerError("SAM 3 image references an unknown camera")
                image = decode_data_image(reference, (int(camera["width"]), int(camera["height"])))
                state = processor.set_image(image)
                for label in labels:
                    processor.reset_all_prompts(state)
                    output = processor.set_text_prompt(prompt=label, state=state)
                    masks = output["masks"].detach().cpu().numpy()
                    for mask in masks:
                        for points in self._polygons(mask):
                            counter += 1
                            if counter > maximum:
                                raise WorkerError("SAM 3 proposal exceeds the configured polygon limit")
                            annotations["image_polygons"].append({
                                "id": f"sam3-{camera_id}-{counter:04d}",
                                "camera_id": camera_id,
                                "label": label,
                                "points": points,
                            })
        version = os.environ.get("OMNILABEL_SAM3_MODEL_VERSION", "facebook/sam3")
        return annotations, version

