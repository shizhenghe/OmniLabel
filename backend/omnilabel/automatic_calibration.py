"""Durable, cancellable all-frame/all-camera calibration proposals.

Source scans, annotations and map registration are never rewritten. A job
records an immutable candidate and source/revision tokens for each pair;
publication rechecks both tokens and recomputes the quality assessment.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

from .calibration_input import calibration_input
from .models import Identifier, Model
from .projection_calibration import (
    ProjectionUpdate, _check_revision, _fingerprint, _image_path, _read_record,
    document_for, save_projection_calibration,
)
from .storage import ConflictError, Store, atomic_json, safe_id

LOGGER = logging.getLogger(__name__)
TERMINAL = {"completed", "cancelled", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AutomaticCalibrationRequest(Model):
    engine: Literal["koide", "direct_nid"] = "koide"
    overwrite_saved: bool = False
    frame_ids: list[Identifier] | None = None
    camera_ids: list[Identifier] | None = None


def engines() -> list[dict]:
    from .automatic_projection import automatic_status
    from .koide_runtime import koide_status
    return [{"id": "koide", "name": "Koide 精准配准", **koide_status()},
            {"id": "direct_nid", "name": "内置 CPU 局部配准", **automatic_status()}]


def _stat(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "inode": stat.st_ino, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def input_fingerprint(store: Store, frame, camera) -> str:
    """Catch changed images/scans in addition to the projection revision."""
    points = {"primary": (_stat(store.pointcloud_path(frame)) if frame.pointcloud_file
                          else _fingerprint(frame.points))}
    if frame.calibration_pointcloud_file:
        points["calibration"] = {
            **_stat(store.calibration_pointcloud_path(frame)),
            "lidar_to_ego": frame.calibration_lidar_to_ego,
            "timestamp": frame.calibration_timestamp,
        }
    return _fingerprint({"points": points, "image": _stat(_image_path(store, frame, camera.id)),
                         "camera": camera.model_dump(), "lidar_to_ego": frame.lidar_to_ego})


def solve_pair(engine: str, points: np.ndarray, image_path: Path, camera,
               initial: np.ndarray, work_dir: Path) -> dict:
    from .automatic_projection import automatic_refine, evaluate_automatic_candidate
    if engine == "direct_nid":
        return automatic_refine(points, image_path, camera, initial)
    from .automatic_projection import _options, _prepare, _quality
    from .koide_runtime import run_koide_calibration
    # Exclude a fixed spatial subset of points from Koide's training cloud.
    # The complete image remains available to the registration, just as for
    # the CPU engine; this is point holdout rather than an image holdout.
    prepared = _prepare(points, image_path, camera, initial, _options(None))
    metrics, reasons = _quality(prepared, np.eye(4), _options(None))
    reasons = [reason for reason in reasons if not reason.startswith(("独立留出区域", "图像与点云强度相关性"))]
    if reasons:
        return {"matrix": initial.tolist(), "method": "koide", "usable": False,
                "metrics": metrics, "rejection_reasons": reasons}
    training = ~prepared["holdout"]
    # Use the current camera estimate as Koide's input frame and optimize a
    # small correction around identity. Reconstructed world maps may otherwise
    # introduce translations of hundreds of metres into the optimizer.
    training_points = np.column_stack((prepared["xyz"][training], prepared["intensity"][training]))
    result = run_koide_calibration(points=training_points, image_path=image_path, camera=camera,
                                  initial_camera_from_lidar=np.eye(4), work_dir=work_dir)
    candidate = np.asarray(result["matrix"], dtype=float) @ initial
    assessment = evaluate_automatic_candidate(points, image_path, camera, initial, candidate)
    assessment.update(method="koide", artifacts=result.get("artifacts", {}))
    assessment["metrics"].update(engine="koide", validation="fixed_spatial_point_holdout",
                                 optimization_frame="initial_camera_coordinates")
    return assessment


class AutomaticCalibrationManager:
    def __init__(self, store: Store, executor: ThreadPoolExecutor):
        self.store = store
        self.executor = executor
        self.directory = store.root / "automatic_calibration_jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._active: set[str] = set()
        self._applying: set[str] = set()
        self._cancel: dict[str, threading.Event] = {}
        self._source_cache = None

    def _source(self, clip_id: str):
        # Inline scans can make a manifest large. Reuse its validated snapshot
        # until an atomic import/global-calibration write changes its identity.
        path = self.store.root / "clips" / f"{safe_id(clip_id)}.json"
        stat = path.stat()
        token = (clip_id, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self._source_cache is None or self._source_cache[0] != token:
            self._source_cache = (token, self.store.source_clip(clip_id))
        return self._source_cache[1]

    def _path(self, job_id: str) -> Path:
        return self.directory / f"{safe_id(job_id)}.json"

    def _save(self, job: dict):
        items = job["items"]
        if job["id"] in self._cancel:
            job["cancel_requested"] = self._cancel[job["id"]].is_set()
        job.update(updated_at=_now(), processed=sum(item["status"] not in {"pending", "running"} for item in items),
                   accepted=sum(item["status"] == "accepted" for item in items),
                   rejected=sum(item["status"] == "rejected" for item in items),
                   failed=sum(item["status"] in {"failed", "conflict"} for item in items),
                   skipped=sum(item["status"] == "skipped" for item in items),
                   saved=sum(item["status"] == "saved" for item in items))
        atomic_json(self._path(job["id"]), job)

    def get(self, clip_id: str, job_id: str) -> dict:
        safe_id(clip_id)
        with self.store.lock:
            path = self._path(job_id)
            if not path.is_file():
                raise KeyError("未找到自动标定任务。")
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["clip_id"] != clip_id:
                raise KeyError("该自动标定任务不属于当前片段。")
            return job

    def recover(self):
        with self.store.lock:
            for path in self.directory.glob("*.json"):
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["status"] in TERMINAL:
                    continue
                for item in job["items"]:
                    if item["status"] in {"pending", "running"}:
                        item.update(status="skipped", reason="服务重启，尚未完成此图像与点云的标定。")
                job.update(status="failed", error="服务重启中断任务；已有候选保留，可保存或重新运行。")
                self._save(job)

    def shutdown(self):
        # Stop after the current pair, rather than blocking service shutdown
        # until an entire clip has been processed.
        with self.store.lock:
            for event in self._cancel.values():
                event.set()

    def status(self, clip_id: str) -> dict:
        with self.store.lock:
            clip = self.store.source_clip(clip_id)
            paths = sorted(self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
            latest = None
            for path in paths:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["clip_id"] == clip_id:
                    latest = job
                    break
            effective = self.store.clip(clip_id)
            saved_pairs = 0
            for frame in effective.frames:
                for camera_id in frame.images:
                    record_path = self.store.root / "projection_calibrations" / clip.id / frame.id / f"{camera_id}.json"
                    if record_path.is_file():
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                        if record.get("matrix") == frame.projection_camera_from_lidar.get(camera_id) and record.get("matrix") is not None:
                            saved_pairs += 1
            return {"engines": engines(), "latest_job": latest,
                    "total_pairs": len(clip.frames) * len(clip.cameras), "saved_pairs": saved_pairs}

    def start(self, clip_id: str, request: AutomaticCalibrationRequest) -> dict:
        with self.store.lock:
            clip = self.store.source_clip(clip_id)
            if self._active or self._applying:
                raise ConflictError("已有自动标定任务正在运行，请等待完成或取消。")
            engine = next(item for item in engines() if item["id"] == request.engine)
            if not engine["available"]:
                raise ValueError(engine["detail"])
            if not clip.cameras:
                raise ValueError("当前片段没有相机，无法自动标定。")
            requested_frames = set(request.frame_ids or [frame.id for frame in clip.frames])
            requested_cameras = set(request.camera_ids or [camera.id for camera in clip.cameras])
            if request.frame_ids is not None and not request.frame_ids:
                raise ValueError("至少选择一帧。")
            if request.camera_ids is not None and not request.camera_ids:
                raise ValueError("至少选择一路相机。")
            unknown_frames = requested_frames - {frame.id for frame in clip.frames}
            unknown_cameras = requested_cameras - {camera.id for camera in clip.cameras}
            if unknown_frames:
                raise ValueError(f"片段中不存在这些帧：{', '.join(sorted(unknown_frames))}")
            if unknown_cameras:
                raise ValueError(f"片段中不存在这些相机：{', '.join(sorted(unknown_cameras))}")
            selected_frames = [frame for frame in clip.frames if frame.id in requested_frames]
            selected_cameras = [camera for camera in clip.cameras if camera.id in requested_cameras]
            items = [{"frame_id": frame.id, "camera_id": camera.id, "status": "pending"}
                     for frame in selected_frames for camera in selected_cameras]
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "clip_id": clip_id, "engine": request.engine,
                   "status": "queued", "created_at": _now(), "updated_at": _now(),
                   "overwrite_saved": request.overwrite_saved, "cancel_requested": False,
                   "selection": {"frame_ids": [frame.id for frame in selected_frames],
                                 "camera_ids": [camera.id for camera in selected_cameras]},
                   "total": len(items), "items": items}
            self._save(job)
            self._active.add(job_id)
            self._cancel[job_id] = threading.Event()
            try:
                self.executor.submit(self._run, job_id, clip_id)
            except RuntimeError:
                self._active.discard(job_id)
                self._cancel.pop(job_id, None)
                job.update(status="failed", error="服务正在关闭，请重启后重新运行。")
                self._save(job)
                raise
            return job

    def _run(self, job_id: str, clip_id: str):
        job = self.get(clip_id, job_id)
        event = self._cancel[job_id]
        try:
            with self.store.lock:
                job["status"] = "running"
                self._save(job)
            for item in job["items"]:
                if event.is_set():
                    break
                try:
                    with self.store.lock:
                        # Fresh source context protects jobs from concurrent
                        # global calibration edits between individual pairs.
                        clip = self._source(clip_id)
                        frame = next(frame for frame in clip.frames if frame.id == item["frame_id"])
                        camera = next(camera for camera in clip.cameras if camera.id == item["camera_id"])
                        document = document_for(self.store, clip_id, frame.id, camera.id, context=(clip, frame, camera))
                        if camera.id not in frame.images:
                            item.update(status="skipped", reason="此帧缺少该相机图像。")
                            self._save(job)
                            continue
                        if not (frame.calibration_point_count or frame.point_count or frame.points):
                            item.update(status="skipped", reason="此帧没有点云。")
                            self._save(job)
                            continue
                        if document["saved_matrix"] is not None and not job["overwrite_saved"]:
                            item.update(status="skipped", reason="保留此帧相机已保存的校准参数。")
                            self._save(job)
                            continue
                        image_path = _image_path(self.store, frame, camera.id)
                        token = input_fingerprint(self.store, frame, camera)
                        registration = calibration_input(
                            self.store, frame, np.asarray(document["effective_matrix"], dtype=float))
                        item.update(status="running", revision=document["revision"], input_fingerprint=token)
                        self._save(job)
                    work_dir = self.store.root / "automatic_calibration_artifacts" / job_id / frame.id / camera.id
                    result = solve_pair(job["engine"], registration.points, image_path, camera,
                                        registration.initial_camera_from_input, work_dir)
                    result["matrix"] = registration.camera_from_primary(
                        np.asarray(result["matrix"], dtype=float)).tolist()
                    result.setdefault("metrics", {}).update(input_source=registration.source,
                                                              input_point_count=len(registration.points))
                    with self.store.lock:
                        updated_clip = self._source(clip_id)
                        updated_frame = next(frame for frame in updated_clip.frames if frame.id == item["frame_id"])
                        updated_camera = next(camera for camera in updated_clip.cameras if camera.id == item["camera_id"])
                        current = document_for(self.store, clip_id, frame.id, camera.id,
                                               context=(updated_clip, updated_frame, updated_camera))
                        _check_revision(current, document["revision"])
                        if input_fingerprint(self.store, updated_frame, updated_camera) != token:
                            raise ConflictError("求解期间源图像或点云已变化，请重新运行。")
                        item.update(status="accepted" if result["usable"] else "rejected",
                                    matrix=result["matrix"], metrics=result.get("metrics"),
                                    reason="；".join(result.get("rejection_reasons", [])),
                                    method=job["engine"], artifacts=result.get("artifacts", {}))
                except ConflictError as exc:
                    item.update(status="conflict", reason=str(exc))
                except Exception as exc:
                    LOGGER.exception("Calibration pair failed: %s/%s/%s", clip_id, item["frame_id"], item["camera_id"])
                    item.update(status="failed", reason=str(exc)[:1500],
                                artifacts=getattr(exc, "artifacts", {}), error_code=getattr(exc, "code", None))
                with self.store.lock:
                    self._save(job)
            with self.store.lock:
                for item in job["items"]:
                    if item["status"] == "pending":
                        item.update(status="skipped", reason="任务已取消，未计算此项。")
                job.update(status="cancelled" if event.is_set() else "completed", cancel_requested=event.is_set())
                self._save(job)
        except Exception as exc:
            LOGGER.exception("Calibration job failed: %s", job_id)
            with self.store.lock:
                job.update(status="failed", error=str(exc)[:1500])
                self._save(job)
        finally:
            with self.store.lock:
                self._active.discard(job_id)
                self._cancel.pop(job_id, None)
                self._source_cache = None

    def cancel(self, clip_id: str, job_id: str) -> dict:
        with self.store.lock:
            job = self.get(clip_id, job_id)
            event = self._cancel.get(job_id)
            if job["status"] not in TERMINAL and event:
                event.set()
                job["cancel_requested"] = True
                self._save(job)
            return job

    def apply(self, clip_id: str, job_id: str) -> dict:
        with self.store.lock:
            job = self.get(clip_id, job_id)
            if job["status"] not in TERMINAL:
                raise ConflictError("请等待计算结束或取消后再保存候选。")
            if job_id in self._applying:
                raise ConflictError("此任务正在保存，请稍候。")
            self._applying.add(job_id)
        try:
            for item in job["items"]:
                if item["status"] != "accepted":
                    continue
                with self.store.lock:
                    try:
                        clip = self._source(clip_id)
                        frame = next(frame for frame in clip.frames if frame.id == item["frame_id"])
                        camera = next(camera for camera in clip.cameras if camera.id == item["camera_id"])
                        if input_fingerprint(self.store, frame, camera) != item["input_fingerprint"]:
                            raise ConflictError("源图像或点云已变化，请重新运行。")
                        context = (clip, frame, camera)
                        current = document_for(self.store, clip_id, frame.id, camera.id, context=context)
                        record = _read_record(self.store, clip_id, frame.id, camera.id) or {}
                        # Publication and the job checkpoint are separate atomic
                        # files. Recover a publication made just before a crash.
                        published = (record.get("source_job_id") == job_id
                                     and record.get("source_input_fingerprint") == item["input_fingerprint"]
                                     and current["saved_matrix"] == item["matrix"])
                        document = current if published else save_projection_calibration(
                            self.store, clip_id, frame.id, camera.id,
                            ProjectionUpdate(revision=item["revision"], matrix=item["matrix"], method=job["engine"]),
                            context=context, source_job_id=job_id,
                            source_input_fingerprint=item["input_fingerprint"])
                        item.update(status="saved", saved_at=document["updated_at"])
                    except ConflictError as exc:
                        item.update(status="conflict", reason=str(exc))
                    except (ValueError, KeyError, OSError, StopIteration) as exc:
                        item.update(status="failed", reason=str(exc))
                    self._save(job)
            return job
        finally:
            with self.store.lock:
                self._applying.discard(job_id)
                self._source_cache = None

    def export(self, clip_id: str) -> dict:
        with self.store.lock:
            clip = self.store.source_clip(clip_id)
            frames = []
            for frame in clip.frames:
                cameras = {}
                for camera in clip.cameras:
                    document = document_for(self.store, clip_id, frame.id, camera.id, context=(clip, frame, camera))
                    cameras[camera.id] = {"camera_from_lidar": document["effective_matrix"],
                                          "saved": document["saved_matrix"] is not None,
                                          "has_image": camera.id in frame.images,
                                          "method": document["method"], "metrics": document["metrics"],
                                          "updated_at": document["updated_at"], "stale": document["stale"]}
                frames.append({"frame_id": frame.id, "timestamp": frame.timestamp,
                               "lidar_to_ego": frame.lidar_to_ego, "cameras": cameras})
            return {"schema_version": "1.0", "clip_id": clip.id, "exported_at": _now(),
                    "convention": "column vectors; camera_from_lidar maps original frame LiDAR points into optical camera (x right, y down, z forward)",
                    "units": {"translation": "metres", "timestamp": "seconds"},
                    "intrinsics_note": "相机内参和畸变沿用导入值；自动标定优化图像与点云之间的刚体外参。",
                    "cameras": [camera.model_dump() for camera in clip.cameras], "frames": frames}


def mount_automatic_calibration_routes(app: FastAPI, manager: AutomaticCalibrationManager):
    route = "/api/clips/{clip_id}/automatic-calibration"

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
    def read_status(clip_id: str):
        return respond(manager.status, clip_id)

    @app.post(route + "/jobs", status_code=202)
    def start_job(clip_id: str, value: AutomaticCalibrationRequest):
        return respond(manager.start, clip_id, value)

    @app.get(route + "/jobs/{job_id}")
    def read_job(clip_id: str, job_id: str):
        return respond(manager.get, clip_id, job_id)

    @app.post(route + "/jobs/{job_id}/cancel")
    def cancel_job(clip_id: str, job_id: str):
        return respond(manager.cancel, clip_id, job_id)

    @app.post(route + "/jobs/{job_id}/apply")
    def apply_job(clip_id: str, job_id: str):
        return respond(manager.apply, clip_id, job_id)

    @app.get(route + "/export")
    def export_parameters(clip_id: str):
        value = respond(manager.export, clip_id)
        return JSONResponse(value, headers={"Content-Disposition": f'attachment; filename="{safe_id(clip_id)}-camera-lidar-calibration.json"'})
