"""Small durable single-machine store with optimistic annotation revisions."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path

from .models import Annotations, Clip


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
        for folder in ("clips", "annotations", "jobs", "assets"):
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
        path = self.root / "clips" / f"{safe_id(clip_id)}.json"
        if not path.is_file():
            raise KeyError("clip not found")
        return Clip.model_validate_json(path.read_text())

    def summaries(self):
        summaries = []
        for path in sorted((self.root / "clips").glob("*.json")):
            clip = Clip.model_validate_json(path.read_text())
            summaries.append({"id": clip.id, "name": clip.name, "description": clip.description,
                              "fps": clip.fps, "frame_count": len(clip.frames), "camera_count": len(clip.cameras)})
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
