"""Revision-checked extrinsic editing without rewriting scans or annotations."""
from __future__ import annotations

from hashlib import sha256
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import field_validator

from .models import Clip, Identifier, Model, rigid_transform
from .storage import ConflictError, Store, atomic_json, safe_id


class CalibrationUpdate(Model):
    revision: str
    cameras: dict[Identifier, list[list[float]]]
    lidar_to_ego: dict[Identifier, list[list[float]]]

    @field_validator("cameras", "lidar_to_ego")
    @classmethod
    def valid_matrices(cls, value):
        for matrix in value.values():
            rigid_transform(matrix)
        return value


def _extrinsics(clip: Clip) -> dict:
    return {"cameras": {camera.id: camera.camera_from_ego for camera in clip.cameras},
            "lidar_to_ego": {frame.id: frame.lidar_to_ego for frame in clip.frames}}


def _revision(clip: Clip) -> str:
    value = {"clip_id": clip.id, **_extrinsics(clip)}
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibration_document(store: Store, clip: Clip) -> dict:
    from .projection_calibration import projection_calibration_scopes
    from .scene import registered_poses
    # The editor receives an overlaid clip elsewhere in the API. A physical
    # mount preview must instead start from source projections, just like the
    # save path, since local projection records become stale after that edit.
    source = store.source_clip(clip.id)
    return {"schema_version": "1.0", "clip_id": clip.id, "revision": _revision(clip),
            "units": {"translation": "metres", "rotation": "matrix"},
            "convention": "column vectors; camera_from_ego maps ego to optical camera; lidar_to_ego maps LiDAR to ego",
            **_extrinsics(clip),
            "frame_projection_overrides": {frame.id: {
                "annotation_camera_from_lidar": frame.annotation_camera_from_lidar,
                "projection_camera_from_lidar": frame.projection_camera_from_lidar,
            } for frame in source.frames},
            "frame_projection_scopes": projection_calibration_scopes(store, source),
            "world_from_lidar": {fid: pose.tolist() for fid, pose in registered_poses(store, clip).items()}}


def save_calibration(store: Store, clip_id: str, value: CalibrationUpdate) -> dict:
    from .scene import (_clip_fingerprint, get_scene_map, invalidate_scene_map,
                        is_supplied_scene_map, registered_poses, source_fingerprint)
    with store.lock:
        # Local projection documents are overlays, not source extrinsics.
        # Rewriting an overlaid clip would bake those overrides into the
        # manifest and prevent a changed base calibration from staling them.
        clip = store.source_clip(clip_id)
        if value.revision != _revision(clip):
            raise ConflictError("Calibration revision conflict. Reload the extrinsics before saving.")
        if set(value.cameras) != {camera.id for camera in clip.cameras}:
            raise ValueError("Calibration must contain exactly the clip's camera IDs")
        if set(value.lidar_to_ego) != {frame.id for frame in clip.frames}:
            raise ValueError("Calibration must contain exactly the clip's frame IDs")
        before = calibration_document(store, clip)
        old_poses = registered_poses(store, clip)
        supplied_map = False
        try:
            map_metadata = get_scene_map(store, clip, include_points=False)
            valid_map = True
            supplied_map = is_supplied_scene_map(map_metadata)
        except KeyError:
            valid_map = False
        changed_camera = value.cameras != before["cameras"]
        changed_lidar = value.lidar_to_ego != before["lidar_to_ego"]
        if not changed_camera and not changed_lidar:
            return {**before, "lidar_changed": False, "map_rebuild_required": False}
        raw = clip.model_dump()
        old_cameras = {camera.id: camera for camera in clip.cameras}
        old_frames = {frame.id: frame for frame in clip.frames}
        for camera in raw["cameras"]:
            camera["camera_from_ego"] = value.cameras[camera["id"]]
        for frame in raw["frames"]:
            original = old_frames[frame["id"]]
            new_lidar = value.lidar_to_ego[frame["id"]]
            for field in ("annotation_camera_from_lidar", "projection_camera_from_lidar"):
                for camera_id, matrix in frame[field].items():
                    old_camera = old_cameras[camera_id].camera_from_ego
                    new_camera = value.cameras[camera_id]
                    if old_camera == new_camera and original.lidar_to_ego == new_lidar:
                        continue
                    # Imported per-frame corrections are relative to the
                    # source mounts. Transport them through the edited mounts
                    # instead of letting an old direct matrix mask the edit.
                    # Local calibration documents remain separate overlays
                    # and become stale through their base fingerprint.
                    frame[field][camera_id] = (np.asarray(new_camera) @ np.linalg.inv(np.asarray(old_camera))
                        @ np.asarray(matrix) @ np.linalg.inv(np.asarray(original.lidar_to_ego))
                        @ np.asarray(new_lidar)).tolist()
            frame["lidar_to_ego"] = value.lidar_to_ego[frame["id"]]
        updated = Clip.model_validate(raw)
        poses = {frame.id: old_poses[frame.id] if frame.lidar_to_ego == value.lidar_to_ego[frame.id] else
                 old_poses[frame.id] @ np.linalg.inv(np.asarray(frame.lidar_to_ego))
                 @ np.asarray(value.lidar_to_ego[frame.id]) for frame in clip.frames}
        history = store.root / "calibration_history" / safe_id(clip.id)
        # This small original-calibration snapshot contains no scan arrays. The
        # source point files and all frame/scene annotations remain untouched.
        if not (history / "original.json").exists():
            atomic_json(history / "original.json", before)
        atomic_json(history / f"{before['revision']}.json", before)
        atomic_json(store.root / "clips" / f"{safe_id(clip.id)}.json", updated.model_dump())
        # A camera-only change must not discard an earlier ICP registration.
        # For a LiDAR mount change, keep its estimated ego pose and compose the
        # new mount; world annotations remain fixed and need visual review.
        atomic_json(store.root / "scene_poses" / f"{safe_id(clip.id)}.json",
                    {"clip_fingerprint": _clip_fingerprint(store, updated),
                     "world_from_lidar": {fid: pose.tolist() for fid, pose in poses.items()}})
        rebuild_required = changed_lidar and not supplied_map
        if rebuild_required:
            invalidate_scene_map(store, clip.id)
        elif valid_map:
            # Camera edits and supplied world-map mount edits leave the map
            # geometry fixed. Refresh its source token and pose metadata.
            path = store.root / "scene_maps" / f"{safe_id(clip.id)}.json"
            cached = json.loads(path.read_text())
            cached["source_fingerprint"] = source_fingerprint(store, updated)
            cached["world_from_lidar"] = {fid: pose.tolist() for fid, pose in poses.items()}
            atomic_json(path, cached)
        return {**calibration_document(store, updated), "lidar_changed": changed_lidar,
                "map_rebuild_required": rebuild_required}


def mount_calibration_routes(app: FastAPI, store: Store):
    def get_clip(clip_id: str) -> Clip:
        try:
            return store.clip(clip_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "Clip not found") from exc

    @app.get("/api/clips/{clip_id}/calibration")
    def read_calibration(clip_id: str):
        with store.lock:
            return calibration_document(store, get_clip(clip_id))

    @app.put("/api/clips/{clip_id}/calibration")
    def write_calibration(clip_id: str, value: CalibrationUpdate):
        get_clip(clip_id)
        try:
            return save_calibration(store, clip_id, value)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/clips/{clip_id}/calibration/export")
    def export_calibration(clip_id: str):
        with store.lock:
            return JSONResponse(calibration_document(store, get_clip(clip_id)),
                                headers={"Content-Disposition": f'attachment; filename="{safe_id(clip_id)}-calibration.json"'})
