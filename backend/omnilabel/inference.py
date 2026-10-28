"""Validated, review-only adapter for an administrator-configured model worker.

The application bundles neither checkpoints nor synthetic model predictions.
Credentials and endpoint selection are server-side; client payloads cannot
choose a destination. Worker responses are proposals, never saved labels.
"""
from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from .models import Annotations, Camera, Frame

MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 24 * 1024 * 1024
TIMEOUT = httpx.Timeout(connect=10, read=120, write=60, pool=5)
MONOCULAR_FRAME_MODELS = frozenset({"latr"})


class InferenceError(RuntimeError):
    """Safe-to-display worker failure without URLs, tokens or raw responses."""


def validate_worker_url(value: str) -> str:
    if not value:
        raise InferenceError("Model worker is not configured. Set OMNILABEL_INFERENCE_URL on the server.")
    try:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError
        parsed = urlsplit(value)
        if not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise ValueError
        # Accessing .port also validates malformed and out-of-range ports.
        _ = parsed.port
        host = parsed.hostname.lower()
        if host == "localhost":
            loopback = True
        else:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise ValueError
        httpx.URL(value)
    except (ValueError, httpx.InvalidURL):
        raise InferenceError(
            "Invalid model worker endpoint. Use HTTPS or HTTP on a literal loopback address/localhost, without URL credentials or fragments."
        ) from None
    return value


def _worker_url(model_id: str | None = None, endpoint: str | None = None) -> str:
    names = {"sam2": "SAM2", "latr": "LATR", "mmdetection3d": "MMDET3D", "openoccupancy": "OPENOCCUPANCY"}
    if endpoint is None:
        prefix = names.get(model_id)
        endpoint = os.environ.get(f"OMNILABEL_{prefix}_URL", "") if prefix else ""
        endpoint = endpoint or os.environ.get("OMNILABEL_INFERENCE_URL", "")
    return validate_worker_url(endpoint)


def worker_configured(model_id: str | None = None, endpoint: str | None = None) -> bool:
    """Whether an endpoint is validly configured, not whether a model is healthy."""
    try:
        _worker_url(model_id, endpoint)
        return True
    except InferenceError:
        return False


def _outbound_images(frame: dict, assets_root: Path | None) -> dict:
    """Embed trusted local image bytes so a remote GPU worker receives pixels."""
    if assets_root is None:
        data_root = Path(os.environ.get(
            "OMNILABEL_DATA_DIR", Path(__file__).resolve().parents[2] / "data"
        ))
        root = (data_root / "assets").resolve()
    else:
        root = Path(assets_root).resolve()
    images = {}
    total = 0
    for camera_id, reference in frame.get("images", {}).items():
        if reference.startswith("data:image/"):
            try:
                header, encoded = reference.split(",", 1)
                if header not in ("data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"):
                    raise ValueError
                if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
                    raise ValueError
                if len(base64.b64decode(encoded, validate=True)) > MAX_IMAGE_BYTES:
                    raise ValueError
            except (ValueError, binascii.Error):
                raise InferenceError("Worker image data URI is invalid or exceeds 24 MiB.") from None
            images[camera_id] = reference
            total += len(reference.encode("utf-8"))
        elif reference.startswith("/api/assets/"):
            path = (root / reference.removeprefix("/api/assets/")).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise InferenceError("A camera image is unavailable in the configured asset directory.")
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix.lower())
            if mime is None or path.stat().st_size > MAX_IMAGE_BYTES:
                raise InferenceError("Worker images must be PNG, JPEG or WebP files under 24 MiB each.")
            # Read at most the bound even if another process changes the file.
            with path.open("rb") as stream:
                pixels = stream.read(MAX_IMAGE_BYTES + 1)
            if len(pixels) > MAX_IMAGE_BYTES:
                raise InferenceError("Worker image exceeds 24 MiB.")
            images[camera_id] = f"data:{mime};base64," + base64.b64encode(pixels).decode("ascii")
            total += len(images[camera_id])
        else:
            raise InferenceError("Worker camera images must use imported local assets or image data URIs.")
        if total > MAX_REQUEST_BYTES:
            raise InferenceError("Worker request exceeds 64 MiB; resize images or split the frame.")
    return images


def infer_proposal(
    clip: dict, frame: dict, annotations: dict, model_id: str, prompt: str,
    *, assets_root: Path | None = None, options: dict | None = None,
    endpoint: str | None = None, token: str | None = None,
    scene_context: dict | None = None,
) -> dict:
    """Request actual inference and return a validated human-review proposal.

    The configured URL is the exact POST endpoint. Redirects are forbidden,
    HTTPS certificate verification remains enabled, and responses are streamed
    with a decompressed byte bound. No annotation persistence occurs here.
    """
    endpoint = _worker_url(model_id, endpoint)
    if not isinstance(model_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", model_id):
        raise InferenceError("model_id must be a short alphanumeric model identifier.")
    if not isinstance(prompt, str) or len(prompt) > 2000:
        raise InferenceError("Worker prompt must be text of at most 2000 characters.")
    try:
        frame_model = Frame.model_validate(frame)
        cameras = [Camera.model_validate(camera) for camera in clip["cameras"]]
        current = Annotations.model_validate(annotations).validate_for_frame(frame_model, cameras)
    except (ValidationError, ValueError, KeyError, TypeError):
        raise InferenceError("Current frame or annotation data is invalid for model inference.") from None
    outbound_frame = frame_model.model_dump()
    outbound_cameras = cameras
    if model_id in MONOCULAR_FRAME_MODELS:
        camera_id = (options or {}).get("camera_id")
        if model_id == "latr" and not camera_id:
            raise InferenceError("LATR requires one selected camera image.")
        if camera_id:
            selected_camera = next((camera for camera in cameras if camera.id == camera_id), None)
            if selected_camera is None or camera_id not in frame_model.images:
                raise InferenceError("The selected camera image is unavailable for model inference.")
            # Monocular workers need only the active image and its calibration.
            # Do this before base64 embedding: sending every camera can inflate a
            # seven-camera frame beyond the worker's bounded request size.
            outbound_cameras = [selected_camera]
            outbound_frame["images"] = {camera_id: frame_model.images[camera_id]}
            for field in ("annotation_camera_from_lidar", "projection_camera_from_lidar"):
                outbound_frame[field] = ({camera_id: outbound_frame[field][camera_id]}
                                         if camera_id in outbound_frame[field] else {})
        # Image-only inference must never serialize a materialized LiDAR scan.
        outbound_frame["points"] = []
        outbound_frame["pointcloud_file"] = None
    try:
        outbound_frame["images"] = _outbound_images(outbound_frame, assets_root)
    except OSError:
        raise InferenceError("A camera image could not be read for model inference.") from None
    payload = {
        "schema_version": "1.0", "model_id": model_id, "prompt": prompt,
        "clip": {"id": clip["id"], "cameras": [camera.model_dump() for camera in outbound_cameras]},
        "frame": outbound_frame, "annotations": current.model_dump(),
    }
    if options:
        payload["options"] = options
    if scene_context:
        payload["scene_context"] = scene_context
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        raise InferenceError("Current frame cannot be serialized for model inference.") from None
    if len(body) > MAX_REQUEST_BYTES:
        raise InferenceError("Worker request exceeds 64 MiB; resize images or split the frame.")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token is None:
        names = {"sam2": "SAM2", "latr": "LATR", "mmdetection3d": "MMDET3D", "openoccupancy": "OPENOCCUPANCY"}
        prefix = names.get(model_id)
        token = os.environ.get(f"OMNILABEL_{prefix}_TOKEN", os.environ.get("OMNILABEL_INFERENCE_TOKEN", "")) if prefix else os.environ.get("OMNILABEL_INFERENCE_TOKEN", "")
    if token:
        if "\r" in token or "\n" in token:
            raise InferenceError("Invalid model worker authentication configuration.")
        headers["Authorization"] = "Bearer " + token
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=False, verify=True) as client:
            with client.stream("POST", endpoint, content=body, headers=headers) as response:
                if response.status_code != 200:
                    raise InferenceError(f"Model worker returned HTTP {response.status_code}; no annotations were changed.")
                length = response.headers.get("content-length")
                if length and length.isdigit() and int(length) > MAX_RESPONSE_BYTES:
                    raise InferenceError("Model worker response exceeds 32 MiB.")
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_RESPONSE_BYTES:
                        raise InferenceError("Model worker response exceeds 32 MiB.")
    except httpx.TimeoutException:
        raise InferenceError("Model worker timed out; no annotations were changed.") from None
    except (httpx.HTTPError, ValueError, UnicodeError):
        # HTTP client errors can contain the full URL or authentication headers.
        raise InferenceError("Model worker request failed; check server connectivity and worker configuration.") from None
    try:
        result = json.loads(chunks)
        proposed = result["annotations"]
        if not isinstance(proposed, dict):
            raise ValueError
        model_version = result["model_version"]
        if not isinstance(model_version, str) or not model_version.strip() or len(model_version) > 200:
            raise ValueError
        # Worker-supplied revisions have no authority over application storage.
        proposed["revision"] = current.revision
        validated = Annotations.model_validate(proposed).validate_for_frame(frame_model, cameras)
    except (ValidationError, ValueError, KeyError, TypeError):
        raise InferenceError("Model worker returned invalid annotations or model metadata; no annotations were changed.") from None
    output = {
        "proposal": validated.model_dump(), "model_id": model_id,
        "model_version": model_version, "review_required": True,
        "source": "external-worker",
    }
    weight_variant = result.get("weight_variant")
    if weight_variant in ("once", "openlane"):
        output["weight_variant"] = weight_variant
    if "occupancy" in result:
        output["occupancy"] = validate_occupancy(result["occupancy"])
    warnings = result.get("warnings", [])
    if isinstance(warnings, list) and len(warnings) <= 20 and all(isinstance(w, str) and len(w) <= 2000 for w in warnings):
        output["warnings"] = warnings
    confidence = result.get("confidence", {})
    if isinstance(confidence, dict) and len(confidence) <= 20000:
        from math import isfinite
        if all(isinstance(k, str) and isinstance(v, (float, int)) and isfinite(v) and 0 <= v <= 1 for k, v in confidence.items()):
            output["confidence"] = confidence
    conversion = result.get("conversion")
    if isinstance(conversion, list) and len(conversion) <= 256:
        output["conversion"] = conversion
    return output


def validate_occupancy(value: dict) -> dict:
    """Validate learned occupancy geometry before it reaches the WebGL view."""
    try:
        from math import isfinite
        if not isinstance(value, dict) or value.get("coordinate_frame") != "lidar":
            raise ValueError
        size = float(value["voxel_size"])
        if not isfinite(size) or not 0.02 <= size <= 10:
            raise ValueError
        voxels = value["voxels"]
        if not isinstance(voxels, list) or len(voxels) > 300_000:
            raise ValueError
        seen = set()
        dimensions = value.get("voxel_dimensions")
        if dimensions is not None and (not isinstance(dimensions, list) or len(dimensions) != 3 or any(not isfinite(float(d)) or not 0 < float(d) <= 100 for d in dimensions)):
            raise ValueError
        for voxel in voxels:
            index, center = voxel["index"], voxel["center"]
            if len(index) != 3 or any(type(i) is not int for i in index) or tuple(index) in seen:
                raise ValueError
            seen.add(tuple(index))
            if len(center) != 3 or any(not isfinite(float(x)) for x in center):
                raise ValueError
            if voxel["state"] not in ("occupied", "free") or not isinstance(voxel["label"], str) or not 0 < len(voxel["label"]) <= 80:
                raise ValueError
        # Preserve range/origin metadata for custom model grids; indices are
        # grid-relative and need not have an origin at the LiDAR sensor.
        return value
    except (ValueError, KeyError, TypeError, OverflowError):
        raise InferenceError("Model worker returned invalid occupancy geometry; no annotations were changed.") from None
