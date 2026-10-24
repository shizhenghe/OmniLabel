"""Mount 4D scene endpoints without coupling them to model worker routes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
import numpy as np

from .models import Clip
from .scene import (
    SCENE_PREVIEW_POINTS, frame_annotations, get_scene, get_scene_map, reconstruct_scene,
    registered_poses, save_reconstruction, save_scene, source_fingerprint,
)
from .scene_models import ReconstructionRequest, SceneAnnotations
from .storage import Store

LOGGER = logging.getLogger(__name__)


def mount_scene_routes(app: FastAPI, store: Store, executor: ThreadPoolExecutor):
    def clip_for(clip_id: str) -> Clip:
        try:
            return store.clip(clip_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "Clip not found") from exc

    @app.get("/api/clips/{clip_id}/scene/annotations", response_model=SceneAnnotations)
    def read_scene(clip_id: str):
        return get_scene(store, clip_for(clip_id))

    @app.put("/api/clips/{clip_id}/scene/annotations", response_model=SceneAnnotations)
    def write_scene(clip_id: str, value: SceneAnnotations):
        clip = clip_for(clip_id)
        try:
            return save_scene(store, clip, value)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/clips/{clip_id}/scene/frame/{frame_id}")
    def evaluated_frame(clip_id: str, frame_id: str):
        clip = clip_for(clip_id)
        frame = next((f for f in clip.frames if f.id == frame_id), None)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        return frame_annotations(get_scene(store, clip), clip, frame, registered_poses(store, clip),
                                 store.annotations(clip.id, frame_id))

    @app.get("/api/clips/{clip_id}/scene/map")
    def cached_map(clip_id: str):
        try:
            return get_scene_map(store, clip_for(clip_id), max_points=SCENE_PREVIEW_POINTS)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    def reconstruct(job: dict, clip: Clip, request: ReconstructionRequest):
        try:
            job.update(status="running", started_at=time.time())
            store.save_job(job)
            with store.lock:
                scene = get_scene(store, clip)
                poses = registered_poses(store, clip)
                legacy = store.all_annotations(clip)
                annotations = {
                    frame.id: frame_annotations(scene, clip, frame, poses, legacy[frame.id]).model_dump()
                    for frame in clip.frames
                }
                static_ids = {track.id for track in scene.tracks if track.mode == "static"}
                for value in annotations.values():
                    value["_static_track_ids"] = list(static_ids)
                fingerprint = source_fingerprint(store, clip)
            # Imported/cached ICP registration is part of the scene coordinate
            # system. Rebuilding after a label edit must not reset it to the
            # original unregistered source poses.
            reconstruction_clip = clip.model_copy(deep=True)
            for frame in reconstruction_clip.frames:
                frame.ego_to_world = (poses[frame.id] @ np.linalg.inv(np.asarray(frame.lidar_to_ego, dtype=float))).tolist()
            source_frames = {frame.id: frame for frame in clip.frames}
            result = reconstruct_scene(
                reconstruction_clip,
                annotations,
                request,
                point_loader=lambda value: store.frame_point_array(source_frames[value["id"]]),
            )
            result = save_reconstruction(store, clip, result, fingerprint, scene, poses)
            preview = get_scene_map(store, clip, max_points=SCENE_PREVIEW_POINTS)
            job.update(status="completed", result=preview, finished_at=time.time())
        except Exception as exc:
            LOGGER.exception("Scene reconstruction %s failed", job["id"])
            job.update(status="failed", error=str(exc)[:1500], finished_at=time.time())
        store.save_job(job)

    @app.post("/api/clips/{clip_id}/scene/reconstruct", status_code=202)
    def submit_reconstruction(clip_id: str, request: ReconstructionRequest):
        clip = clip_for(clip_id)
        job = {"id": uuid.uuid4().hex, "clip_id": clip.id, "kind": "reconstruct", "status": "queued",
               "created_at": time.time(), "parameters": request.model_dump()}
        store.save_job(job)
        response = dict(job)
        executor.submit(reconstruct, job, clip, request)
        return response
