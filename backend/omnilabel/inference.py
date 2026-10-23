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


class InferenceError(RuntimeError):
    """Safe-to-display worker failure without URLs, tokens or raw responses."""


def _worker_url() -> str:
    value = os.environ.get("OMNILABEL_INFERENCE_URL", "")
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


def worker_configured() -> bool:
    """Whether an endpoint is validly configured, not whether a model is healthy."""
    try:
        _worker_url()
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
    *, assets_root: Path | None = None,
) -> dict:
    """Request actual inference and return a validated human-review proposal.

    The configured URL is the exact POST endpoint. Redirects are forbidden,
    HTTPS certificate verification remains enabled, and responses are streamed
    with a decompressed byte bound. No annotation persistence occurs here.
    """
    endpoint = _worker_url()
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
    try:
        outbound_frame["images"] = _outbound_images(outbound_frame, assets_root)
    except OSError:
        raise InferenceError("A camera image could not be read for model inference.") from None
    payload = {
        "schema_version": "1.0", "model_id": model_id, "prompt": prompt,
        "clip": {"id": clip["id"], "cameras": [camera.model_dump() for camera in cameras]},
        "frame": outbound_frame, "annotations": current.model_dump(),
    }
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError):
        raise InferenceError("Current frame cannot be serialized for model inference.") from None
    if len(body) > MAX_REQUEST_BYTES:
        raise InferenceError("Worker request exceeds 64 MiB; resize images or split the frame.")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get("OMNILABEL_INFERENCE_TOKEN", "")
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
    return {
        "proposal": validated.model_dump(), "model_id": model_id,
        "model_version": model_version, "review_required": True,
        "source": "external-worker",
    }
