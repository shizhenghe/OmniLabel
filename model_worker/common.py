from __future__ import annotations

import base64
import binascii
import copy
import io
import re
from typing import Any

import numpy as np
from PIL import Image


class WorkerError(RuntimeError):
    """A safe error that can be returned to the worker caller."""


def parse_labels(prompt: str) -> list[str]:
    labels = []
    for value in re.split(r"[,，;；\n]+", prompt):
        label = value.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def decode_data_image(reference: str, expected_size: tuple[int, int]) -> Image.Image:
    try:
        header, encoded = reference.split(",", 1)
        if header not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
            raise ValueError
        pixels = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(pixels))
        image.load()
    except (ValueError, OSError, binascii.Error):
        raise WorkerError("request contains an invalid embedded camera image") from None
    if image.size != expected_size:
        raise WorkerError("embedded camera image dimensions do not match calibration")
    return image.convert("RGB")


def input_points(frame: dict[str, Any], target_dim: int = 4) -> np.ndarray:
    try:
        points = np.asarray(frame["points"], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        raise WorkerError("frame points must be a numeric Nx3 or Nx4 array") from None
    if points.ndim != 2 or points.shape[1] not in (3, 4) or not np.isfinite(points).all():
        raise WorkerError("frame points must be a finite Nx3 or Nx4 array")
    if points.shape[1] == 3:
        points = np.column_stack((points, np.zeros(len(points), dtype=np.float32)))
    if target_dim < 4:
        raise WorkerError("MMDetection3D point dimension must be at least four")
    if target_dim > points.shape[1]:
        points = np.column_stack((points, np.zeros((len(points), target_dim - points.shape[1]), dtype=np.float32)))
    return np.ascontiguousarray(points)


def merged_annotations(request: dict[str, Any], *, replace_prefix: str) -> dict[str, Any]:
    source = request.get("annotations")
    if not isinstance(source, dict):
        raise WorkerError("annotations must be an object")
    result = copy.deepcopy(source)
    result.setdefault("boxes", [])
    result.setdefault("vectors", [])
    result.setdefault("image_polygons", [])
    result.setdefault("point_labels", {})
    for key in ("boxes", "vectors", "image_polygons"):
        if not isinstance(result[key], list):
            raise WorkerError(f"annotations.{key} must be a list")
        result[key] = [item for item in result[key] if not str(item.get("id", "")).startswith(replace_prefix)]
    if not isinstance(result["point_labels"], dict):
        raise WorkerError("annotations.point_labels must be an object")
    result.pop("revision", None)
    return result

