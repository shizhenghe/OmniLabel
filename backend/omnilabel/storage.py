"""Small durable single-machine store with optimistic annotation revisions."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .models import Annotations, Clip, Frame


LOGGER = logging.getLogger(__name__)

# Increment whenever the summary-building contract changes.  File mtimes alone
# cannot invalidate a cache produced by an older server that rejected fields a
# newer Clip schema accepts.
CLIP_SUMMARY_CACHE_VERSION = 2


class ConflictError(Exception):
    pass


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value):
        raise ValueError("invalid identifier")
    return value


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.lock = threading.RLock()
        self._point_validations = {}
        for folder in ("clips", "annotations", "jobs", "assets", "pointclouds"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

    def recover_interrupted_jobs(self):
        """Recover jobs only when the single service process explicitly starts.

        Merely importing the API module or opening a Store for inspection must
        never mutate jobs that may belong to a currently running service.
        """
        with self.lock:
            for path in (self.root / "jobs").glob("*.json"):
                job = json.loads(path.read_text())
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", error="Server restarted before the job completed. Submit it again.")
                    atomic_json(path, job)

    def clip(self, clip_id: str) -> Clip:
        from .projection_calibration import overlay_projection_calibrations
        return overlay_projection_calibrations(self, self.source_clip(clip_id))

    def source_clip(self, clip_id: str) -> Clip:
        """Read the source manifest without materializing local calibration overlays."""
        path = self.root / "clips" / f"{safe_id(clip_id)}.json"
        if not path.is_file():
            raise KeyError("clip not found")
        return Clip.model_validate_json(path.read_text())

    def pointcloud_path(self, frame: Frame) -> Path:
        if frame.pointcloud_file is None:
            raise ValueError("this frame stores its points inline")
        points_root = (self.root / "pointclouds").resolve()
        source = (points_root / frame.pointcloud_file).resolve()
        if not source.is_relative_to(points_root) or not source.is_file():
            raise ValueError("referenced point cloud is missing or outside the pointcloud store")
        return source

    def calibration_pointcloud_path(self, frame: Frame) -> Path:
        if frame.calibration_pointcloud_file is None:
            raise ValueError("this frame has no synchronized calibration point cloud")
        points_root = (self.root / "pointclouds").resolve()
        source = (points_root / frame.calibration_pointcloud_file).resolve()
        if not source.is_relative_to(points_root) or not source.is_file():
            raise ValueError("referenced calibration point cloud is missing or outside the pointcloud store")
        return source

    def _stored_point_array(self, source: Path, expected_count: int | None, *, label: str) -> np.ndarray:
        try:
            points = np.load(source, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise ValueError(f"referenced {label} is not a valid NPY array") from exc
        if (points.ndim != 2 or points.shape[1] not in (3, 4)
                or len(points) != expected_count or points.dtype.kind not in "fiu"):
            raise ValueError(f"referenced {label} has invalid shape, count, type, or values")
        stat = source.stat()
        key = (str(source), stat.st_size, stat.st_mtime_ns, expected_count)
        if key not in self._point_validations:
            for start in range(0, len(points), 100_000):
                if not np.isfinite(points[start:start + 100_000]).all():
                    raise ValueError(f"referenced {label} contains non-finite values")
            self._point_validations[key] = True
        if points.shape[1] == 3:
            points = np.column_stack((points, np.zeros(len(points), dtype=points.dtype)))
        return points

    def frame_point_array(self, frame: Frame) -> np.ndarray:
        if frame.pointcloud_file is None:
            points = np.asarray(frame.points, dtype=float)
            return points if points.size else np.empty((0, 4), dtype=float)
        source = self.pointcloud_path(frame)
        return self._stored_point_array(source, frame.point_count, label="point cloud")

    def calibration_point_array(self, frame: Frame) -> np.ndarray:
        source = self.calibration_pointcloud_path(frame)
        return self._stored_point_array(source, frame.calibration_point_count,
                                        label="calibration point cloud")

    def frame_points(self, frame: Frame) -> list[list[float]]:
        return self.frame_point_array(frame).tolist()

    def materialize_frame(self, frame: Frame) -> Frame:
        if frame.pointcloud_file is None:
            return frame
        value = frame.model_dump()
        value["points"] = self.frame_points(frame)
        value["pointcloud_file"] = None
        return Frame.model_validate(value)

    def summaries(self):
        clip_paths = sorted((self.root / "clips").glob("*.json"), key=lambda path: path.stem)
        files = {path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in clip_paths}
        cache_path = self.root / "clip_summaries.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (cached.get("version") == CLIP_SUMMARY_CACHE_VERSION
                    and cached.get("files") == files
                    and isinstance(cached.get("summaries"), list)):
                return cached["summaries"]
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            pass
        summaries = []
        for path in clip_paths:
            try:
                clip = Clip.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValidationError) as exc:
                LOGGER.warning("Ignoring invalid clip manifest %s: %s", path, exc)
                continue
            summaries.append({"id": clip.id, "name": clip.name, "description": clip.description,
                              "fps": clip.fps, "frame_count": len(clip.frames), "camera_count": len(clip.cameras)})
        atomic_json(cache_path, {"version": CLIP_SUMMARY_CACHE_VERSION,
                                 "files": files, "summaries": summaries})
        return summaries

    def add_clip(self, clip: Clip, annotations: dict[str, Annotations] | None = None):
        with self.lock:
            destination = self.root / "clips" / f"{safe_id(clip.id)}.json"
            if destination.exists():
                raise ConflictError("A clip with this ID already exists. Give the imported clip a unique ID.")
            # Publish the clip last so list/read never sees incomplete initial labels.
            for frame_id, value in (annotations or {}).items():
                path = self.root / "annotations" / clip.id / f"{safe_id(frame_id)}.json"
                atomic_json(path, value.model_dump())
            atomic_json(destination, clip.model_dump())

    def annotations(self, clip_id: str, frame_id: str) -> Annotations:
        path = self.root / "annotations" / safe_id(clip_id) / f"{safe_id(frame_id)}.json"
        return Annotations.model_validate_json(path.read_text()) if path.exists() else Annotations()

    def save_annotations(self, clip_id: str, frame_id: str, value: Annotations) -> Annotations:
        with self.lock:
            previous = self.annotations(clip_id, frame_id)
            if value.revision != previous.revision:
                raise ConflictError(f"Revision conflict: expected {previous.revision}, received {value.revision}. Reload before saving.")
            updated = value.model_copy(update={"revision": previous.revision + 1})
            atomic_json(self.root / "annotations" / safe_id(clip_id) / f"{safe_id(frame_id)}.json", updated.model_dump())
            return updated

    def all_annotations(self, clip: Clip) -> dict[str, dict]:
        with self.lock:
            return {frame.id: self.annotations(clip.id, frame.id).model_dump() for frame in clip.frames}

    def save_job(self, job: dict):
        with self.lock:
            atomic_json(self.root / "jobs" / f"{safe_id(job['id'])}.json", job)

    def job(self, job_id: str):
        path = self.root / "jobs" / f"{safe_id(job_id)}.json"
        if not path.exists():
            raise KeyError("job not found")
        with self.lock:
            return json.loads(path.read_text())
