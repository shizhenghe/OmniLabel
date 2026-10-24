"""Lossless portable clip import. ZIP members are never extracted blindly."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .models import Annotations, Clip

# Optional deployment quotas are bytes/members, never source-point quotas.
# Zero means unrestricted. UploadFile and ZIP assets stream through disk so a
# large archive need not be copied into one giant request bytes object.
MAX_UPLOAD = int(os.environ.get("OMNILABEL_MAX_UPLOAD_BYTES", "0"))
MAX_EXPANDED = int(os.environ.get("OMNILABEL_MAX_EXPANDED_BYTES", "0"))
MAX_FILES = int(os.environ.get("OMNILABEL_MAX_ARCHIVE_MEMBERS", "0"))


def read_member(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> bytes:
    try:
        return archive.read(entry)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError) as exc:
        raise ValueError("ZIP member failed integrity or compression validation") from exc


def member_path(name: str) -> str:
    value = PurePosixPath(name)
    if "\\" in name or value.is_absolute() or ".." in value.parts or not value.parts or ":" in name or "\x00" in name:
        raise ValueError("unsafe ZIP or asset path")
    return str(value)


def checked_point_array(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] not in (3, 4) or points.dtype.kind not in "fiu":
        raise ValueError("point file must contain a real numeric Nx3 or Nx4 array")
    for start in range(0, len(points), 100_000):
        if not np.isfinite(points[start:start + 100_000]).all():
            raise ValueError("point file must contain finite values")
    return points


def decode_point_array(data: bytes, name: str) -> np.ndarray:
    extension = Path(name).suffix.lower()
    if extension == ".bin":
        if len(data) % 16:
            raise ValueError(".bin points must contain little-endian float32 XYZI records")
        points = np.frombuffer(data, dtype="<f4").reshape(-1, 4)
    elif extension == ".npy":
        stream = io.BytesIO(data)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("NPY format versions 1 and 2 are supported")
        if len(shape) != 2 or shape[1] not in (3, 4) or shape[0] < 0 or dtype.kind not in "fiu":
            raise ValueError("NPY must contain a real numeric Nx3 or Nx4 array")
        expected_size = int(shape[0]) * int(shape[1]) * dtype.itemsize
        if expected_size > len(data) - stream.tell():
            raise ValueError("NPY data is shorter than its declared shape")
        points = np.frombuffer(stream.read(expected_size), dtype=dtype).reshape(shape, order="F" if fortran else "C")
    elif extension == ".pcd":
        text = data.decode("ascii")
        lines = text.splitlines()
        index = next((i for i, line in enumerate(lines) if line.strip().upper().startswith("DATA ")), None)
        if index is None or lines[index].strip().lower() != "data ascii":
            raise ValueError("only ASCII PCD is supported; convert binary PCD to .bin or .npy")
        headers = {line.split()[0].upper(): line.split()[1:] for line in lines[:index] if line.strip() and not line.startswith("#")}
        fields = headers.get("FIELDS", [])
        if not all(key in fields for key in ("x", "y", "z")):
            raise ValueError("PCD must have x y z fields")
        counts = headers.get("COUNT", ["1"] * len(fields))
        if any(value != "1" for value in counts):
            raise ValueError("PCD multi-count fields are not supported")
        raw = np.loadtxt(io.StringIO("\n".join(lines[index + 1:])), ndmin=2)
        if raw.shape[1] != len(fields):
            raise ValueError("PCD field count does not match point records")
        selected = [fields.index(key) for key in ("x", "y", "z")]
        if "intensity" in fields:
            selected.append(fields.index("intensity"))
        points = raw[:, selected]
    else:
        raise ValueError("point files must be .bin, .npy or ASCII .pcd")
    return checked_point_array(points)


def decode_points(data: bytes, name: str):
    """Compatibility helper for callers needing JSON rows; no count truncation."""
    return decode_point_array(data, name).astype(float).tolist()


def point_array_from_file(path: Path, name: str) -> np.ndarray:
    """Validate a complete binary scan without loading another complete copy."""
    extension = Path(name).suffix.lower()
    try:
        if extension == ".bin":
            size = path.stat().st_size
            if size % 16:
                raise ValueError(".bin points must contain little-endian float32 XYZI records")
            points = np.memmap(path, mode="r", dtype="<f4", shape=(size // 16, 4)) if size else np.empty((0, 4), dtype="<f4")
        elif extension == ".npy":
            points = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            return decode_point_array(path.read_bytes(), name)
        return checked_point_array(points)
    except (OSError, EOFError) as exc:
        raise ValueError("point file data is missing or shorter than its declared shape") from exc


def stage_point_asset(archive, entry, name, staging: Path, namespace: str, frame: dict):
    point_stage = staging / "_pointclouds"
    point_stage.mkdir(exist_ok=True)
    source = point_stage / (uuid.uuid4().hex + ".source")
    try:
        try:
            with archive.open(entry) as incoming, source.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError) as exc:
            raise ValueError("ZIP member failed integrity or compression validation") from exc
        points = point_array_from_file(source, name)
        if frame.get("point_count") is not None and frame["point_count"] != len(points):
            raise ValueError("point_count does not match the point asset")
        target = uuid.uuid4().hex + ".npy"
        if Path(name).suffix.lower() == ".npy":
            shutil.copyfile(source, point_stage / target)
        else:
            np.save(point_stage / target, points, allow_pickle=False)
        frame["points"] = []
        frame["pointcloud_file"] = f"{namespace}/{target}"
        frame["point_count"] = len(points)
    finally:
        source.unlink(missing_ok=True)


def validate_image(data: bytes, expected_size: tuple[int, int] | None = None) -> str:
    if len(data) > 24 * 1024 * 1024:
        raise ValueError("individual images may not exceed 24 MiB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError("images must be JPEG, PNG or WebP")
            if image.width > 16384 or image.height > 16384 or image.width * image.height > 40_000_000:
                raise ValueError("image dimensions exceed interactive limit")
            if expected_size is not None and image.size != expected_size:
                raise ValueError(f"image size {image.size} does not match calibrated camera dimensions {expected_size}")
            suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[image.format]
            image.verify()
            return suffix
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("invalid image asset") from exc


def parse_import(payload, filename: str, assets_root: Path, *, include_scene: bool = False):
    """Return validated clip, labels, staged asset directory, final asset namespace.

    Caller publishes the directory and clip transactionally; on failure it cleans
    the staging path. The namespace is random, so imported files cannot overwrite
    files belonging to an existing clip.
    """
    incoming = io.BytesIO(payload) if isinstance(payload, bytes) else payload
    incoming.seek(0, io.SEEK_END)
    payload_size = incoming.tell()
    incoming.seek(0)
    if MAX_UPLOAD and payload_size > MAX_UPLOAD:
        raise ValueError("upload exceeds configured byte limit")
    archive = None
    names = {}
    if Path(filename).suffix.lower() == ".zip":
        try:
            archive = zipfile.ZipFile(incoming)
            entries = archive.infolist()
            if ((MAX_FILES and len(entries) > MAX_FILES)
                    or (MAX_EXPANDED and sum(e.file_size for e in entries) > MAX_EXPANDED)):
                raise ValueError("ZIP exceeds configured member or expanded-byte limit")
            for entry in entries:
                name = member_path(entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("ZIP symlinks are not permitted")
                if entry.flag_bits & 1:
                    raise ValueError("encrypted ZIP members are not supported")
                if not entry.is_dir():
                    if name in names:
                        raise ValueError("ZIP contains duplicate members")
                    names[name] = entry
            if "manifest.json" not in names:
                raise ValueError("ZIP must contain manifest.json at its root")
            raw = json.loads(read_member(archive, names["manifest.json"]))
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError("invalid ZIP archive") from exc
    elif Path(filename).suffix.lower() == ".json":
        raw = json.load(incoming)
    else:
        raise ValueError("upload a .json manifest or a .zip clip archive")

    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    annotations_raw = raw.get("annotations", {})
    if not isinstance(annotations_raw, dict):
        raise ValueError("annotations must be an object keyed by frame ID")
    clip_raw = raw.get("clip", raw)
    if not isinstance(clip_raw, dict):
        raise ValueError("clip must be an object")
    # Work on a copy; remove envelope fields when a manifest uses top-level clip fields.
    scene_fields = ("scene_annotations", "scene_map", "scene_registration")
    scene_bundle = {key: raw[key] for key in scene_fields if key in raw}
    clip_raw = {key: value for key, value in clip_raw.items() if key not in ("annotations", "schema_version", *scene_fields)}
    namespace = f"import-{uuid.uuid4().hex}"
    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=assets_root))
    try:
        camera_sizes = {camera.get("id"): (camera.get("width"), camera.get("height"))
                        for camera in clip_raw.get("cameras", []) if isinstance(camera, dict)}
        for frame in clip_raw.get("frames", []):
            if not isinstance(frame, dict):
                raise ValueError("frames must be objects")
            if "points_path" in frame:
                if frame.get("points") or frame.get("pointcloud_file"):
                    raise ValueError("a frame cannot contain both points_path and another point source")
                path = member_path(frame.pop("points_path"))
                if archive is None or path not in names:
                    raise ValueError(f"missing point asset: {path}")
                stage_point_asset(archive, names[path], path, staging, namespace, frame)
            images = frame.get("images", {})
            if not isinstance(images, dict):
                raise ValueError("frame images must be an object")
            for camera_id, path in list(images.items()):
                if not isinstance(path, str):
                    raise ValueError("image paths must be strings")
                if path.startswith("/api/assets/"):
                    relative = member_path(path.removeprefix("/api/assets/"))
                    asset = (assets_root / relative).resolve()
                    if not asset.is_relative_to(assets_root.resolve()) or not asset.is_file():
                        raise ValueError("referenced server image asset is missing; use a ZIP with image files")
                    validate_image(asset.read_bytes(), camera_sizes.get(camera_id))
                    continue
                if path.startswith("data:image/"):
                    try:
                        header, encoded = path.split(",", 1)
                        if ";base64" not in header:
                            raise ValueError("only base64 data images are supported")
                        image_bytes = base64.b64decode(encoded, validate=True)
                    except Exception as exc:
                        raise ValueError("invalid base64 image") from exc
                else:
                    relative = member_path(path)
                    if archive is None or relative not in names:
                        raise ValueError(f"missing image asset: {relative}; package images with manifest.json in ZIP")
                    image_bytes = read_member(archive, names[relative])
                suffix = validate_image(image_bytes, camera_sizes.get(camera_id))
                target = uuid.uuid4().hex + suffix
                (staging / target).write_bytes(image_bytes)
                images[camera_id] = f"/api/assets/{namespace}/{target}"
        clip = Clip.model_validate(clip_raw)
        # JSON manifests may reference scans already present in this single-user
        # store, but may never create dangling or escaping source references.
        point_root = (assets_root.parent / "pointclouds").resolve()
        for frame in clip.frames:
            if frame.pointcloud_file and not frame.pointcloud_file.startswith(namespace + "/"):
                source = (point_root / frame.pointcloud_file).resolve()
                if not source.is_relative_to(point_root) or not source.is_file():
                    raise ValueError("referenced point asset is missing; use points_path in a portable ZIP")
                if len(point_array_from_file(source, source.name)) != frame.point_count:
                    raise ValueError("point_count does not match the referenced point asset")
        frame_map = {frame.id: frame for frame in clip.frames}
        if set(annotations_raw) - set(frame_map):
            raise ValueError("annotations reference an unknown frame")
        annotations = {frame_id: Annotations.model_validate(value).validate_for_frame(frame_map[frame_id], clip.cameras)
                       for frame_id, value in annotations_raw.items()}
        if scene_bundle:
            from .scene import validate_scene_import
            validate_scene_import(clip, scene_bundle.get("scene_annotations"), scene_bundle.get("scene_map"), scene_bundle.get("scene_registration"))
        base = (clip, annotations, staging, namespace)
        return (*base, scene_bundle) if include_scene else base
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if archive:
            archive.close()
