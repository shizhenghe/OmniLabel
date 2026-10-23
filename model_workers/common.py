"""Model-independent validation and conversion, without importing GPU packages."""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import hashlib
import io
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image


class ModelUnavailable(RuntimeError):
    pass


class InvalidInput(ValueError):
    pass


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise InvalidInput("Model returned non-finite values")
    return result


def points_array(frame):
    points = np.asarray(frame.get("points", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 4) or not 1 <= len(points) <= 500000:
        raise InvalidInput("A model frame requires 1–500000 original XYZ/XYZI points")
    if not np.isfinite(points).all():
        raise InvalidInput("Point coordinates must be finite")
    if points.shape[1] == 3:
        points = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
    return points


def image_from_payload(payload, camera_id):
    camera = next((c for c in payload["clip"]["cameras"] if c["id"] == camera_id), None)
    if camera is None:
        raise InvalidInput("Choose a camera present in this clip")
    reference = payload["frame"].get("images", {}).get(camera_id, "")
    try:
        header, encoded = reference.split(",", 1)
        if header not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
            raise ValueError
        if len(encoded) > 32 * 1024 * 1024:
            raise ValueError
        pixels = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(pixels)) as original:
            if original.width * original.height > 40000000 or original.size != (camera["width"], camera["height"]):
                raise ValueError
            image = original.convert("RGB")
    except (ValueError, binascii.Error, OSError, Image.DecompressionBombError):
        raise InvalidInput("A valid embedded image matching its calibration is required") from None
    return image, camera


def current_annotations(payload):
    value = deepcopy(payload.get("annotations", {}))
    for key, default in (("boxes", []), ("vectors", []), ("image_polygons", []), ("point_labels", {}), ("revision", 0)):
        value.setdefault(key, deepcopy(default))
    return value


def model_version(model_id, checkpoint):
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return model_id + "/sha256:" + digest.hexdigest()


def new_id(prefix):
    return prefix + "-" + uuid4().hex[:12]


def validate_options(options):
    if not isinstance(options, dict):
        raise InvalidInput("options must be an object")
    threshold = options.get("score_threshold", 0.3)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise InvalidInput("score_threshold must be between 0 and 1")
    label = options.get("label", "object")
    if not isinstance(label, str) or not label.strip() or len(label) > 80:
        raise InvalidInput("label must contain 1–80 characters")
    return float(threshold), label.strip()


def _simplify_closed(vertices, epsilon=0.7):
    """Remove collinear points and bounded-error RDP simplification."""
    clean = []
    for index, b in enumerate(vertices):
        a, c = vertices[index - 1], vertices[(index + 1) % len(vertices)]
        if (b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0]):
            clean.append(b)
    if len(clean) <= 256:
        return clean
    # Bound output complexity; the warning accompanying every SAM2 result
    # states that polygon conversion is an approximation, not a raster mask.
    stride = int(math.ceil(len(clean) / 256))
    return clean[::stride]


def mask_to_polygons(mask, minimum_area=16):
    """Trace exterior pixel boundaries, preserving disconnected components.

    Holes cannot be represented by the application's simple polygon schema;
    they are counted explicitly in conversion metadata rather than concealed.
    A bounded 768-pixel working image limits checkerboard boundary memory.
    """
    original = np.asarray(mask, dtype=bool)
    if original.ndim != 2 or not original.size:
        raise InvalidInput("Model masks must be nonempty two-dimensional arrays")
    original_h, original_w = original.shape
    factor = min(1.0, 768.0 / max(original_h, original_w))
    if factor < 1:
        original_image = Image.fromarray(original.astype(np.uint8))
        work = np.asarray(original_image.resize((max(1, round(original_w * factor)), max(1, round(original_h * factor))), Image.Resampling.NEAREST)) > 0
    else:
        work = original
    h, w = work.shape
    adjacent = {}
    def edge(start, end):
        adjacent.setdefault(start, []).append(end)
    top = work & ~np.pad(work[:-1], ((1, 0), (0, 0)))
    right = work & ~np.pad(work[:, 1:], ((0, 0), (0, 1)))
    bottom = work & ~np.pad(work[1:], ((0, 1), (0, 0)))
    left = work & ~np.pad(work[:, :-1], ((0, 0), (1, 0)))
    for y, x in zip(*np.nonzero(top)): edge((int(x), int(y)), (int(x + 1), int(y)))
    for y, x in zip(*np.nonzero(right)): edge((int(x + 1), int(y)), (int(x + 1), int(y + 1)))
    for y, x in zip(*np.nonzero(bottom)): edge((int(x + 1), int(y + 1)), (int(x), int(y + 1)))
    for y, x in zip(*np.nonzero(left)): edge((int(x), int(y + 1)), (int(x), int(y)))
    polygons, holes = [], 0
    directions = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    while adjacent:
        start = next(iter(adjacent))
        current, previous_direction, contour = start, 0, []
        while True:
            contour.append(current)
            choices = adjacent[current]
            # Right turn first separates components touching at one pixel corner.
            ordering = [(previous_direction + 1) % 4, previous_direction, (previous_direction - 1) % 4, (previous_direction + 2) % 4]
            target = min(choices, key=lambda end: ordering.index(directions[(end[0] - current[0], end[1] - current[1])]))
            choices.remove(target)
            if not choices: del adjacent[current]
            previous_direction = directions[(target[0] - current[0], target[1] - current[1])]
            current = target
            if current == start: break
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(contour, contour[1:] + contour[:1])) / 2
        if area < 0:
            holes += 1
        elif area / (factor * factor) >= minimum_area:
            simplified = _simplify_closed(contour)
            if len(simplified) >= 3:
                polygons.append([[min(float(x) * original_w / w, original_w), min(float(y) * original_h / h, original_h)] for x, y in simplified])
    return polygons[:128], {"hole_count": holes, "polygon_limit": 128, "working_scale": factor,
                            "approximate_polygon_conversion": True}
