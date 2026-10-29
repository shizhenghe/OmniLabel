"""Durable, serial LATR batches that only produce world-space proposals.

No point cloud is materialized and no frame/scene annotation is written here.
Each successful frame has its own result file, while the pollable job remains
small. Publication uses the existing scene save/revision workflow.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import threading
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import Field, model_validator

from .model_connections import ModelConnections
from .models import Annotations, Identifier, Model, Vector
from .scene import _scene_path, registered_poses
from .scene_models import SceneAnnotations
from .storage import ConflictError, Store, atomic_json, safe_id

TERMINAL = {"completed", "cancelled", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def _file_token(path: Path):
    if not path.is_file():
        return None
    info = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"inode": info.st_ino, "size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "sha256": digest.hexdigest()}


def _manifest_token(store: Store, clip_id: str):
    # A new atomic import/global calibration is a new source document, even if
    # it happens to preserve the same selected camera's numeric parameters.
    info = (store.root / "clips" / f"{safe_id(clip_id)}.json").stat()
    return [info.st_ino, info.st_size, info.st_mtime_ns]


def _scene_revision(store: Store, clip_id: str) -> int:
    path = _scene_path(store, clip_id)
    return SceneAnnotations.model_validate_json(path.read_text()).revision if path.exists() else 0


def _frame_stub(frame, camera_id: str) -> dict:
    value = frame.model_dump(exclude={"points"})
    # The image-only worker does not need the source scan or its labels. A
    # validated empty frame avoids loading binary scans merely to serialize it.
    value.update(points=[], point_count=0, pointcloud_file=None,
                 images={camera_id: frame.images[camera_id]} if camera_id in frame.images else {})
    for name in ("annotation_camera_from_lidar", "projection_camera_from_lidar"):
        value[name] = {camera_id: value[name][camera_id]} if camera_id in value[name] else {}
    return value


def _input_token(store: Store, clip_id: str, frame, camera, matrix) -> str:
    reference = frame.images.get(camera.id)
    image_token = None
    if reference and reference.startswith("/api/assets/"):
        root = (store.root / "assets").resolve()
        path = (root / reference.removeprefix("/api/assets/")).resolve()
        if not path.is_relative_to(root):
            raise ValueError("相机图像路径超出平台资源目录。")
        image_token = _file_token(path)
    elif reference:
        image_token = _hash(reference)
    document = store.root / "projection_calibrations" / clip_id / frame.id / f"{camera.id}.json"
    return _hash({"camera": camera.model_dump(), "frame": _frame_stub(frame, camera.id),
                  "world_from_lidar": np.asarray(matrix, dtype=float).tolist(),
                  "image_token": image_token, "projection_document": _file_token(document)})


class LatrBatchRequest(Model):
    camera_id: Identifier
    frame_ids: list[Identifier] | None = Field(default=None, min_length=1, max_length=2000)
    label: str = Field(default="lane", min_length=1, max_length=80)
    score_threshold: float = Field(default=.3, ge=0, le=1)
    weight_variant: Literal["once", "openlane"] = "once"

    @model_validator(mode="after")
    def unique_frames(self):
        if self.frame_ids is not None and len(self.frame_ids) != len(set(self.frame_ids)):
            raise ValueError("批量预标注的帧 ID 不得重复。")
        if not self.label.strip():
            raise ValueError("车道线类别不能为空。")
        return self


class LatrBatchManager:
    def __init__(self, store: Store, executor: ThreadPoolExecutor, connections: ModelConnections):
        self.store, self.executor, self.connections = store, executor, connections
        self.directory = store.root / "latr_batches"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, str] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._inference_lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.directory / f"{safe_id(job_id)}.json"

    def _folder(self, job_id: str) -> Path:
        return self.directory / safe_id(job_id)

    def _save(self, job: dict):
        with self.store.lock:
            items = job["items"]
            event = self._cancel.get(job["id"])
            job.update(updated_at=_now(),
                       processed=sum(item["status"] not in {"pending", "running"} for item in items),
                       succeeded=sum(item["status"] == "succeeded" for item in items),
                       failed=sum(item["status"] == "failed" for item in items),
                       skipped=sum(item["status"] == "skipped" for item in items),
                       vector_count=sum(item.get("vector_count", 0) for item in items),
                       cancel_requested=event.is_set() if event else job.get("cancel_requested", False))
            atomic_json(self._path(job["id"]), job)

    def get(self, clip_id: str, job_id: str) -> dict:
        safe_id(clip_id)
        with self.store.lock:
            path = self._path(job_id)
            if not path.is_file():
                raise KeyError("未找到批量 LATR 任务。")
            job = json.loads(path.read_text())
            if job["clip_id"] != clip_id:
                raise KeyError("该批量任务不属于当前 clip。")
            return job

    def status(self, clip_id: str) -> dict:
        with self.store.lock:
            self.store.source_clip(clip_id)
            jobs = [json.loads(path.read_text()) for path in self.directory.glob("*.json")]
            jobs = sorted((job for job in jobs if job["clip_id"] == clip_id),
                          key=lambda job: job["created_at"], reverse=True)
            latest = next((job for job in jobs if job["status"] not in TERMINAL), jobs[0] if jobs else None)
            return {"latest_job": latest, "jobs": [{key: value for key, value in job.items() if key != "items"}
                                                    for job in jobs]}

    def start(self, clip_id: str, request: LatrBatchRequest) -> dict:
        from .inference import worker_configured
        endpoint, _ = self.connections.endpoint("latr")
        if not worker_configured("latr", endpoint):
            raise HTTPException(503, "请先配置有效的 LATR 推理服务地址。")
        with self.store.lock:
            if clip_id in self._active:
                raise ConflictError("当前 clip 已有批量 LATR 任务，请等待完成或取消后再提交。")
            clip = self.store.clip(clip_id)
            camera = next((camera for camera in clip.cameras if camera.id == request.camera_id), None)
            if camera is None:
                raise ValueError("所选相机不属于当前 clip。")
            wanted = set(request.frame_ids) if request.frame_ids is not None else {frame.id for frame in clip.frames}
            if not wanted <= {frame.id for frame in clip.frames}:
                raise ValueError("所选帧不属于当前 clip。")
            frames = [frame for frame in clip.frames if frame.id in wanted]
            poses = registered_poses(self.store, clip)
            revision = _scene_revision(self.store, clip_id)
            job_id = uuid.uuid4().hex
            snapshot = {"clip": {"id": clip.id, "cameras": [camera.model_dump()]},
                        "manifest_token": _manifest_token(self.store, clip.id),
                        "frames": [_frame_stub(frame, camera.id) for frame in frames],
                        "tokens": {frame.id: _input_token(self.store, clip.id, frame, camera, poses[frame.id]) for frame in frames},
                        "frame_revisions": {frame.id: self.store.annotations(clip.id, frame.id).revision for frame in frames},
                        "world_from_lidar": {frame.id: poses[frame.id].tolist() for frame in frames}}
            job = {"id": job_id, "clip_id": clip.id, "camera_id": camera.id, "status": "queued",
                   "created_at": _now(), "scene_revision": revision, "parameters": request.model_dump(),
                   "total": len(frames), "cancel_requested": False,
                   "items": [{"frame_id": frame.id, "status": "pending", "vector_count": 0} for frame in frames]}
            atomic_json(self._folder(job_id) / "snapshot.json", snapshot)
            self._cancel[job_id] = threading.Event()
            self._active[clip_id] = job_id
            self._save(job)
            try:
                self.executor.submit(self._run, job_id, endpoint, self.connections.token("latr"))
            except Exception:
                self._active.pop(clip_id, None)
                self._cancel.pop(job_id, None)
                job.update(status="failed", error="无法启动批量任务，请重新提交。")
                self._save(job)
                raise
            return dict(job)

    def _run(self, job_id: str, endpoint: str, token: str):
        from .inference import infer_proposal
        job = None
        try:
            with self.store.lock:
                job = json.loads(self._path(job_id).read_text())
                snapshot = json.loads((self._folder(job_id) / "snapshot.json").read_text())
                event = self._cancel[job_id]
                job["status"] = "running"
                self._save(job)
            options = {name: job["parameters"][name] for name in ("camera_id", "label", "score_threshold", "weight_variant")}
            for ordinal, (item, frame) in enumerate(zip(job["items"], snapshot["frames"])):
                if event.is_set():
                    break
                if item["status"] in {"succeeded", "skipped"}:
                    continue
                if job["camera_id"] not in frame["images"]:
                    item.update(status="skipped", error="该帧缺少所选相机图像。", retryable=False)
                    self._save(job)
                    continue
                item["status"] = "running"
                self._save(job)
                try:
                    with self._inference_lock:
                        if event.is_set():
                            item["status"] = "pending"
                            break
                        result = infer_proposal(snapshot["clip"], frame,
                                                Annotations(revision=snapshot["frame_revisions"][frame["id"]]).model_dump(),
                                                "latr", options["label"], assets_root=self.store.root / "assets",
                                                options=options, endpoint=endpoint, token=token)
                    proposed = Annotations.model_validate(result["proposal"])
                    matrix = np.asarray(snapshot["world_from_lidar"][frame["id"]], dtype=np.float64)
                    vectors = []
                    for index, vector in enumerate(proposed.vectors):
                        points = np.asarray(vector.points, dtype=np.float64)
                        world = points @ matrix[:3, :3].T + matrix[:3, 3]
                        vectors.append(vector.model_copy(update={"id": f"batch-{job_id}-{ordinal}-{index}",
                                                                 "points": [tuple(point) for point in world.tolist()]}).model_dump())
                    atomic_json(self._folder(job_id) / "results" / f"{safe_id(frame['id'])}.json",
                                {"frame_id": frame["id"], "vectors": vectors,
                                 "model_version": result.get("model_version"), "warnings": result.get("warnings", [])})
                    item.update(status="succeeded", vector_count=len(vectors), retryable=False)
                except Exception as exc:
                    # infer_proposal sanitizes worker URLs/credentials before
                    # propagating inference errors. Keep failure local to frame.
                    item.update(status="failed", error=str(exc)[:1500], retryable=True)
                self._save(job)
            job["status"] = "cancelled" if event.is_set() else "completed"
            if event.is_set():
                for item in job["items"]:
                    if item["status"] in {"pending", "running"}:
                        item.update(status="skipped", error="用户取消，尚未执行此帧。", retryable=True)
            self._save(job)
        except Exception as exc:
            if job is not None:
                job.update(status="failed", error=str(exc)[:1500])
                for item in job["items"]:
                    if item["status"] in {"pending", "running"}:
                        item.update(status="skipped", error="任务中断，尚未执行此帧。", retryable=True)
                self._save(job)
        finally:
            with self.store.lock:
                if job is not None:
                    if self._active.get(job["clip_id"]) == job_id:
                        self._active.pop(job["clip_id"], None)
                self._cancel.pop(job_id, None)

    def cancel(self, clip_id: str, job_id: str) -> dict:
        with self.store.lock:
            job = self.get(clip_id, job_id)
            if job["status"] in TERMINAL:
                return job
            event = self._cancel.get(job_id)
            if event is not None:
                event.set()
            # Do not write an old copy of the items while the worker updates it.
            job["cancel_requested"] = True
            atomic_json(self._path(job_id), job)
            return job

    def _validate_snapshot(self, job: dict, snapshot: dict):
        clip_id = job["clip_id"]
        clip = self.store.clip(clip_id)
        if (_manifest_token(self.store, clip_id) != snapshot["manifest_token"]
                or _scene_revision(self.store, clip_id) != job["scene_revision"]):
            raise ConflictError("clip 或场景标注已改变，批量候选未应用，请重新生成。")
        camera = next((value for value in clip.cameras if value.id == job["camera_id"]), None)
        frames = {frame.id: frame for frame in clip.frames}
        poses = registered_poses(self.store, clip)
        for frame_id, fingerprint in snapshot["tokens"].items():
            frame = frames.get(frame_id)
            if (frame is None or camera is None
                    or self.store.annotations(clip_id, frame_id).revision != snapshot["frame_revisions"][frame_id]
                    or _input_token(self.store, clip_id, frame, camera, poses[frame_id]) != fingerprint):
                raise ConflictError("图像、标定、位姿或源帧标注已改变，批量候选未应用，请重新生成。")

    def retry(self, clip_id: str, job_id: str) -> dict:
        """Resume failed/unprocessed frames in place, retaining successful ones."""
        from .inference import worker_configured
        with self.store.lock:
            job = self.get(clip_id, job_id)
            if job["status"] not in TERMINAL or clip_id in self._active:
                raise ConflictError("请等待当前批量任务结束后再重试。")
            snapshot = json.loads((self._folder(job_id) / "snapshot.json").read_text())
            self._validate_snapshot(job, snapshot)
            targets = [item for item in job["items"] if item["status"] == "failed" or item.get("retryable")]
            if not targets:
                raise ValueError("没有可重试的失败或未执行帧。")
            endpoint, _ = self.connections.endpoint("latr")
            if not worker_configured("latr", endpoint):
                raise HTTPException(503, "请先配置有效的 LATR 推理服务地址。")
            for item in targets:
                item.update(status="pending", vector_count=0, retryable=False)
                item.pop("error", None)
            job.update(status="queued", cancel_requested=False)
            job.pop("error", None)
            self._cancel[job_id] = threading.Event()
            self._active[clip_id] = job_id
            self._save(job)
            try:
                self.executor.submit(self._run, job_id, endpoint, self.connections.token("latr"))
            except Exception:
                self._active.pop(clip_id, None)
                self._cancel.pop(job_id, None)
                job.update(status="failed", error="无法重试任务；已完成候选保留。")
                self._save(job)
                raise
            return dict(job)

    def proposal(self, clip_id: str, job_id: str) -> dict:
        with self.store.lock:
            job = self.get(clip_id, job_id)
            if job["status"] not in TERMINAL:
                raise ConflictError("请等待批量任务完成，或取消后再读取已完成候选。")
            snapshot = json.loads((self._folder(job_id) / "snapshot.json").read_text())
            self._validate_snapshot(job, snapshot)
            vectors, sources, warnings = [], [], []
            for item in job["items"]:
                if item["status"] != "succeeded":
                    continue
                result = json.loads((self._folder(job_id) / "results" / f"{safe_id(item['frame_id'])}.json").read_text())
                vectors.extend(Vector.model_validate(vector).model_dump() for vector in result["vectors"])
                sources.append(item["frame_id"])
                warnings.extend(f"{item['frame_id']}: {warning}" for warning in result.get("warnings", []))
            if job["failed"] or job["skipped"]:
                warnings.insert(0, f"成功 {job['succeeded']} 帧，失败 {job['failed']} 帧，跳过 {job['skipped']} 帧；仅包含成功帧的候选。")
            return {"job_id": job_id, "clip_id": clip_id, "camera_id": job["camera_id"],
                    "coordinate_frame": "world", "scene_revision": job["scene_revision"],
                    "vectors": vectors, "source_frame_ids": sources, "warnings": warnings}

    def recover(self):
        with self.store.lock:
            for path in self.directory.glob("*.json"):
                job = json.loads(path.read_text())
                if job["status"] in TERMINAL:
                    continue
                for item in job["items"]:
                    if item["status"] in {"pending", "running"}:
                        # A result atomically published immediately before a
                        # restart remains recoverable, even if metadata lagged.
                        result_path = self._folder(job["id"]) / "results" / f"{safe_id(item['frame_id'])}.json"
                        if result_path.is_file():
                            result = json.loads(result_path.read_text())
                            item.update(status="succeeded", vector_count=len(result["vectors"]))
                        else:
                            item.update(status="skipped", error="服务重启中断此帧，已有结果保留。", retryable=True)
                job.update(status="failed", error="服务重启中断任务；已完成候选保留，可重试未完成帧。")
                self._save(job)
            self._active.clear()
            self._cancel.clear()

    def shutdown(self):
        with self.store.lock:
            for event in self._cancel.values():
                event.set()


def mount_latr_batch_routes(app: FastAPI, manager: LatrBatchManager):
    route = "/api/clips/{clip_id}/latr-batches"

    def respond(operation, *args):
        try:
            return operation(*args)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'")) from exc
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get(route)
    def overview(clip_id: str):
        return respond(manager.status, clip_id)

    @app.post(route, status_code=202)
    def start(clip_id: str, request: LatrBatchRequest):
        return respond(manager.start, clip_id, request)

    @app.get(route + "/{job_id}")
    def read(clip_id: str, job_id: str):
        return respond(manager.get, clip_id, job_id)

    @app.post(route + "/{job_id}/cancel")
    def cancel(clip_id: str, job_id: str):
        return respond(manager.cancel, clip_id, job_id)

    @app.post(route + "/{job_id}/retry", status_code=202)
    def retry(clip_id: str, job_id: str):
        return respond(manager.retry, clip_id, job_id)

    @app.get(route + "/{job_id}/proposal")
    def proposal(clip_id: str, job_id: str):
        return respond(manager.proposal, clip_id, job_id)
