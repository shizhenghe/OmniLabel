"""Persistent scene geometry and genuine pose/ICP whole-clip reconstruction.

The scene is the editing coordinate system, not an overlay of independent frames.
Tracks carry world-space temporal keyframes and vectors live once per clip.
Source scans/calibration remain untouched. ICP is bounded local registration,
not SLAM or a claim of global loop-closure/map accuracy.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from math import atan2, cos, pi, sin
from pathlib import Path

import numpy as np

from .geometry import DYNAMIC_LABELS, _inside_box, _matrix, _points, _transform, _voxel_indices, fuse_map
from .models import Annotations, Clip, Frame
from .scene_models import ReconstructionRequest, SceneAnnotations
from .storage import ConflictError, Store, atomic_json, safe_id

MAX_ICP_POINTS = 3000
MAX_ICP_ITERATIONS = 35
MAX_ICP_PAIR_DISTANCE = 3.0


def _scene_path(store: Store, clip_id: str) -> Path:
    return store.root / "scenes" / f"{safe_id(clip_id)}.json"


def _map_path(store: Store, clip_id: str) -> Path:
    return store.root / "scene_maps" / f"{safe_id(clip_id)}.json"


def _poses_path(store: Store, clip_id: str) -> Path:
    return store.root / "scene_poses" / f"{safe_id(clip_id)}.json"


def source_poses(clip: Clip) -> dict[str, np.ndarray]:
    return {f.id: _matrix(f.ego_to_world, "ego_to_world") @ _matrix(f.lidar_to_ego, "lidar_to_ego")
            for f in clip.frames}


def registered_poses(store: Store, clip: Clip) -> dict[str, np.ndarray]:
    """Keep a registered coordinate system even when annotation changes stale the map."""
    path = _poses_path(store, clip.id)
    if path.exists():
        raw = json.loads(path.read_text())
        if raw.get("clip_fingerprint") == _clip_fingerprint(store, clip):
            matrices = raw.get("world_from_lidar", {})
            if set(matrices) == {f.id for f in clip.frames}:
                return {fid: _matrix(value, "registered world_from_lidar") for fid, value in matrices.items()}
    return source_poses(clip)


def _clip_fingerprint(store: Store, clip: Clip) -> str:
    path = store.root / "clips" / f"{safe_id(clip.id)}.json"
    if path.exists():
        stat = path.stat()
        token = [clip.id, stat.st_size, stat.st_mtime_ns]
    else:
        token = clip.model_dump()
    return sha256(json.dumps(token, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_fingerprint(store: Store, clip: Clip) -> str:
    """Cheap on repeated reads; saved scans are immutable and annotations atomic."""
    tokens = [_clip_fingerprint(store, clip)]
    for path in [_scene_path(store, clip.id)] + [store.root / "annotations" / clip.id / f"{f.id}.json" for f in clip.frames]:
        if path.exists():
            stat = path.stat()
            tokens.append([path.name, stat.st_size, stat.st_mtime_ns])
        else:
            tokens.append([path.name, None])
    return sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _heading(yaw: float, transform: np.ndarray) -> float:
    direction = transform[:3, :3] @ np.array([cos(yaw), sin(yaw), 0.0])
    if np.linalg.norm(direction[:2]) < 1e-8:
        raise ValueError("A box heading becomes vertical in this coordinate frame; yaw-only cuboids cannot represent it")
    return float(atan2(direction[1], direction[0]))


def transform_box(box: dict, transform: np.ndarray) -> dict:
    result = deepcopy(box)
    result["center"] = _transform(np.array([box["center"]], dtype=float), transform)[0].tolist()
    result["yaw"] = _heading(box["yaw"], transform)
    return result


def _unique_id(preferred: str, used: set[str], suffix: str) -> str:
    if preferred not in used:
        used.add(preferred)
        return preferred
    digest = sha256(f"{preferred}:{suffix}".encode()).hexdigest()[:10]
    value = f"{preferred[:80]}-{digest}"
    while value in used:
        value = f"{preferred[:72]}-{sha256((value + suffix).encode()).hexdigest()[:16]}"
    used.add(value)
    return value


def initialize_scene(clip: Clip, annotations: dict[str, dict], poses: dict[str, np.ndarray]) -> SceneAnnotations:
    """Migrate legacy labels nondestructively, retaining all temporal observations."""
    tracks: dict[str, dict] = {}
    vectors = []
    geometries = set()
    used = {box["track_id"] for ann in annotations.values() for box in ann.get("boxes", [])}
    for frame in clip.frames:
        value = annotations.get(frame.id, {})
        matrix = poses[frame.id]
        for box in value.get("boxes", []):
            world_box = transform_box(box, matrix)
            track = tracks.setdefault(box["track_id"], {"id": box["track_id"], "label": box["label"],
                                                       "mode": "dynamic", "keyframes": []})
            track["keyframes"].append({"frame_id": frame.id, **{k: world_box[k] for k in ("center", "size", "yaw")}})
        for vector in value.get("vectors", []):
            world = deepcopy(vector)
            world["points"] = _transform(np.asarray(vector["points"], dtype=float), matrix).tolist()
            geometry = json.dumps([world["label"], world["closed"], np.round(world["points"], 3).tolist()], separators=(",", ":"))
            if geometry in geometries:
                continue
            geometries.add(geometry)
            world["id"] = _unique_id(world["id"], used, frame.id)
            vectors.append(world)
    for track in tracks.values():
        centers = np.asarray([key["center"] for key in track["keyframes"]])
        yaw_changes = [abs((key["yaw"] - track["keyframes"][0]["yaw"] + pi) % (2 * pi) - pi)
                       for key in track["keyframes"]]
        if track["label"].lower() not in DYNAMIC_LABELS or (len(centers) > 1 and np.max(np.linalg.norm(centers - centers[0], axis=1)) < 0.10 and max(yaw_changes) < 0.1):
            track["mode"] = "static"
    return SceneAnnotations.model_validate({"revision": 0, "coordinate_frame": "world", "vectors": vectors,
                                            "tracks": list(tracks.values())}).validate_for_clip(clip)


def get_scene(store: Store, clip: Clip) -> SceneAnnotations:
    with store.lock:
        path = _scene_path(store, clip.id)
        if path.exists():
            return SceneAnnotations.model_validate_json(path.read_text()).validate_for_clip(clip)
        scene = initialize_scene(clip, store.all_annotations(clip), registered_poses(store, clip))
        atomic_json(path, scene.model_dump())
        return scene


def save_scene(store: Store, clip: Clip, value: SceneAnnotations) -> SceneAnnotations:
    value.validate_for_clip(clip)
    with store.lock:
        previous = get_scene(store, clip)
        if previous.revision != value.revision:
            raise ConflictError(f"Scene revision conflict: expected {previous.revision}, received {value.revision}. Reload the scene before saving.")
        saved = value.model_copy(update={"revision": previous.revision + 1})
        atomic_json(_scene_path(store, clip.id), saved.model_dump())
        return saved


def sample_track(track: dict, clip: Clip, frame_id: str) -> dict:
    times = {f.id: f.timestamp for f in clip.frames}
    keys = sorted(track["keyframes"], key=lambda k: times[k["frame_id"]])
    time = times[frame_id]
    if track["mode"] == "static" or time <= times[keys[0]["frame_id"]]:
        selected = keys[0]
    elif time >= times[keys[-1]["frame_id"]]:
        selected = keys[-1]
    else:
        right = next(i for i, key in enumerate(keys) if times[key["frame_id"]] >= time)
        a, b = keys[right - 1], keys[right]
        alpha = (time - times[a["frame_id"]]) / (times[b["frame_id"]] - times[a["frame_id"]])
        delta = (b["yaw"] - a["yaw"] + pi) % (2 * pi) - pi
        selected = {"center": (np.asarray(a["center"]) * (1 - alpha) + np.asarray(b["center"]) * alpha).tolist(),
                    "size": (np.asarray(a["size"]) * (1 - alpha) + np.asarray(b["size"]) * alpha).tolist(),
                    "yaw": float(a["yaw"] + delta * alpha)}
    return {"id": track["id"], "track_id": track["id"], "label": track["label"],
            **{key: deepcopy(selected[key]) for key in ("center", "size", "yaw")}}


def frame_annotations(scene: SceneAnnotations, clip: Clip, frame: Frame,
                      poses: dict[str, np.ndarray], legacy: Annotations | dict | None = None) -> Annotations:
    inverse = np.linalg.inv(poses[frame.id])
    raw = legacy.model_dump() if isinstance(legacy, Annotations) else (legacy or {})
    boxes = [transform_box(sample_track(track.model_dump(), clip, frame.id), inverse) for track in scene.tracks]
    vectors = []
    for vector in scene.vectors:
        value = vector.model_dump()
        value["points"] = _transform(np.asarray(value["points"], dtype=float), inverse).tolist()
        vectors.append(value)
    # Scene track IDs may overlap legacy polygon IDs. Preserve polygon content
    # while giving colliding IDs stable alternatives in this derived response.
    polygons = deepcopy(raw.get("image_polygons", []))
    used = {b["id"] for b in boxes} | {v["id"] for v in vectors}
    for polygon in polygons:
        polygon["id"] = _unique_id(polygon["id"], used, frame.id)
    return Annotations.model_validate({"boxes": boxes, "vectors": vectors, "image_polygons": polygons,
                                       "point_labels": raw.get("point_labels", {}), "revision": raw.get("revision", 0)})


# Stable convenience entry points for model jobs and root API integration.
load_scene = get_scene


def evaluate_scene(store: Store, clip: Clip, frame_id: str, scene: SceneAnnotations | None = None) -> dict:
    frame = next((value for value in clip.frames if value.id == frame_id), None)
    if frame is None:
        raise ValueError("Frame not found in scene clip")
    return frame_annotations(scene or get_scene(store, clip), clip, frame, registered_poses(store, clip),
                             store.annotations(clip.id, frame_id)).model_dump()


def _fusion_annotations(frame: Frame, annotations: dict, exclude_dynamic: bool) -> dict:
    """Keep explicitly static vehicles; semantic class alone is not motion."""
    value = deepcopy(annotations)
    static_ids = set(value.pop("_static_track_ids", []))
    static_boxes = [box for box in value.get("boxes", []) if box.get("track_id") in static_ids]
    if not exclude_dynamic:
        value["boxes"] = []
        value["point_labels"] = {k: f"scene-static:{label}" if label.lower() in DYNAMIC_LABELS else label
                                 for k, label in value.get("point_labels", {}).items()}
        return value
    value["boxes"] = [box for box in value.get("boxes", []) if box.get("track_id") not in static_ids]
    if static_boxes:
        cloud = _points(frame.model_dump())
        if len(cloud) * len(static_boxes) > 50_000_000:
            raise ValueError("Too many static target/point comparisons for reconstruction")
        protected = np.zeros(len(cloud), dtype=bool)
        for box in static_boxes:
            protected |= _inside_box(cloud, box)
        for index, label in list(value.get("point_labels", {}).items()):
            if label.lower() in DYNAMIC_LABELS and protected[int(index)]:
                value["point_labels"][index] = f"scene-static:{label}"
    return value


def _static_scan(frame: Frame, annotations: dict, exclude_dynamic: bool) -> np.ndarray:
    cloud = _points(frame.model_dump())
    if not exclude_dynamic:
        return cloud
    keep = np.ones(len(cloud), dtype=bool)
    if len(cloud) * len(annotations.get("boxes", [])) > 50_000_000:
        raise ValueError("Too many target/point comparisons for scene reconstruction")
    for box in annotations.get("boxes", []):
        if box["label"].lower() in DYNAMIC_LABELS:
            keep &= ~_inside_box(cloud, box)
    for index, label in annotations.get("point_labels", {}).items():
        if label.lower() in DYNAMIC_LABELS and 0 <= int(index) < len(cloud):
            keep[int(index)] = False
    return cloud[keep]


def _registration_sample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3))
    xyz = points[:, :3]
    _, indices = np.unique(_voxel_indices(xyz, max(voxel_size, 0.2)), axis=0, return_index=True)
    xyz = xyz[np.sort(indices)]
    if len(xyz) > MAX_ICP_POINTS:
        xyz = xyz[np.linspace(0, len(xyz) - 1, MAX_ICP_POINTS, dtype=int)]
    return xyz


def _spatial_support(points: np.ndarray) -> bool:
    if len(points) < 30:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    # A plane is valid; an isolated point/line cannot constrain rigid registration.
    return len(singular) >= 2 and singular[1] > max(1e-5, singular[0] * 1e-4)


def _nearest(source: np.ndarray, target: np.ndarray):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(target)
        return lambda query: tree.query(query, k=1, workers=1)
    except ImportError:
        # Bounded NumPy fallback keeps the route functional in minimal installs.
        def query(values):
            distances, indices = [], []
            for start in range(0, len(values), 128):
                chunk = values[start:start + 128]
                squared = ((chunk[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
                index = np.argmin(squared, axis=1)
                distances.extend(np.sqrt(squared[np.arange(len(chunk)), index]))
                indices.extend(index)
            return np.asarray(distances), np.asarray(indices)
        return query


def register_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray,
                 max_distance: float = MAX_ICP_PAIR_DISTANCE) -> tuple[np.ndarray, dict]:
    """Trimmed, point-to-point SVD ICP with explicit overlap/degeneracy checks."""
    if not _spatial_support(source) or not _spatial_support(target):
        raise ValueError("ICP needs at least 30 non-collinear static points in both scans")
    query = _nearest(source, target)
    matrix = np.array(initial, dtype=float, copy=True)
    iterations = 0
    converged = False
    for iteration in range(MAX_ICP_ITERATIONS):
        transformed = _transform(source, matrix)
        distances, indices = query(transformed)
        valid = np.flatnonzero(distances <= max_distance)
        if len(valid) < 30:
            raise ValueError("ICP scan overlap is insufficient (fewer than 30 nearby correspondences)")
        # Trim long correspondences instead of fitting dynamic/outlier geometry.
        order = valid[np.argsort(distances[valid])]
        keep = order[:max(30, int(len(order) * 0.85))]
        a, b = transformed[keep], target[indices[keep]]
        if not _spatial_support(a) or not _spatial_support(b):
            raise ValueError("ICP correspondence geometry is degenerate")
        center_a, center_b = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - center_a).T @ (b - center_b))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = center_b - rotation @ center_a
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = rotation, translation
        matrix = delta @ matrix
        iterations = iteration + 1
        angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
        if np.linalg.norm(translation) < 1e-4 and angle < 1e-4:
            converged = True
            break
    distances, _ = query(_transform(source, matrix))
    inliers = distances <= max_distance
    fitness = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(distances[inliers] ** 2))) if inliers.any() else float("inf")
    if fitness < 0.25 or not np.isfinite(rmse) or rmse > max_distance * 0.5:
        raise ValueError(f"ICP quality rejected: overlap={fitness:.3f}, inlier RMSE={rmse:.3f}m. Supply better poses or split the clip.")
    return matrix, {"fitness": fitness, "rmse": rmse, "iterations": iterations, "converged": converged,
                    "correspondence_distance": max_distance, "source_points": len(source), "target_points": len(target)}


def reconstruct_scene(clip: Clip, annotations: dict[str, dict], request: ReconstructionRequest) -> dict:
    """Build one world-space map, retaining an explicit scan trajectory."""
    poses = source_poses(clip)
    fusion_annotations = {frame.id: _fusion_annotations(frame, annotations.get(frame.id, {}), request.exclude_dynamic)
                          for frame in clip.frames}
    reports = []
    if request.registration == "icp":
        previous_frame, previous_scan = None, None
        original_poses = source_poses(clip)
        for frame in clip.frames:
            scan = _registration_sample(_static_scan(frame, fusion_annotations.get(frame.id, {}), request.exclude_dynamic), request.voxel_size)
            if previous_frame is None:
                if not _spatial_support(scan):
                    raise ValueError("ICP reference scan needs at least 30 non-collinear static points")
            else:
                initial = poses[previous_frame.id] @ np.linalg.inv(original_poses[previous_frame.id]) @ original_poses[frame.id]
                target = _transform(previous_scan, poses[previous_frame.id])
                poses[frame.id], report = register_icp(scan, target, initial)
                reports.append({"frame_id": frame.id, "reference_frame_id": previous_frame.id, **report})
            previous_frame, previous_scan = frame, scan
    corrected = clip.model_dump()
    for frame in corrected["frames"]:
        frame["ego_to_world"] = (poses[frame["id"]] @ np.linalg.inv(np.asarray(frame["lidar_to_ego"], dtype=float))).tolist()
    result = fuse_map(corrected, fusion_annotations, request.voxel_size)
    result["point_labels"] = {index: label.removeprefix("scene-static:") for index, label in result.get("point_labels", {}).items()}
    points = np.asarray(result["points"], dtype=float)
    if not len(points):
        raise ValueError("Reconstruction produced no static map points; inspect scan contents and dynamic filtering")
    result.update(
        trajectory=[{"frame_id": f.id, "timestamp": f.timestamp, "position": poses[f.id][:3, 3].tolist()} for f in clip.frames],
        world_from_lidar={fid: matrix.tolist() for fid, matrix in poses.items()},
        bounds={"min": points[:, :3].min(axis=0).tolist(), "max": points[:, :3].max(axis=0).tolist()},
        registration={"mode": request.registration, "frames": reports, "loop_closure": False,
                      "note": "Uses supplied poses" if request.registration == "poses" else "Local sequential point-to-point ICP; no loop closure or global SLAM accuracy guarantee"},
        method="pose-aligned whole-clip static reconstruction" if request.registration == "poses" else "sequential ICP whole-clip static reconstruction",
        exclude_dynamic=request.exclude_dynamic,
    )
    return result


def rebase_scene(scene: SceneAnnotations, old_poses: dict[str, np.ndarray], new_poses: dict[str, np.ndarray]) -> SceneAnnotations:
    raw = scene.model_dump()
    changed = False
    for track in raw["tracks"]:
        for key in track["keyframes"]:
            delta = new_poses[key["frame_id"]] @ np.linalg.inv(old_poses[key["frame_id"]])
            if not np.allclose(delta, np.eye(4), atol=1e-8):
                updated = transform_box(key, delta)
                key.update(center=updated["center"], yaw=updated["yaw"])
                changed = True
    if changed:
        raw["revision"] += 1
    return SceneAnnotations.model_validate(raw)


def get_scene_map(store: Store, clip: Clip) -> dict:
    path = _map_path(store, clip.id)
    with store.lock:
        if not path.exists():
            raise KeyError("Scene map has not been reconstructed")
        value = json.loads(path.read_text())
        if value.get("source_fingerprint") != source_fingerprint(store, clip):
            raise KeyError("Scene source annotations changed; reconstruct the scene map to refresh dynamic filtering")
        return value


def save_reconstruction(store: Store, clip: Clip, result: dict, fingerprint: str,
                        scene_snapshot: SceneAnnotations, old_poses: dict[str, np.ndarray]) -> dict:
    with store.lock:
        if fingerprint != source_fingerprint(store, clip):
            raise ConflictError("Scene or per-frame labels changed during reconstruction; retry using the latest labels")
        poses = {fid: _matrix(value, "reconstructed world_from_lidar") for fid, value in result["world_from_lidar"].items()}
        rebased = rebase_scene(scene_snapshot, old_poses, poses)
        if rebased.revision != scene_snapshot.revision:
            atomic_json(_scene_path(store, clip.id), rebased.model_dump())
        atomic_json(_poses_path(store, clip.id), {"clip_fingerprint": _clip_fingerprint(store, clip),
                                                 "world_from_lidar": result["world_from_lidar"]})
        result.update(scene_revision=rebased.revision, source_fingerprint=source_fingerprint(store, clip))
        atomic_json(_map_path(store, clip.id), result)
        return result


def invalidate_scene_map(store: Store, clip_id: str):
    """Optional explicit invalidation; fingerprints also detect all saved edits."""
    with store.lock:
        path = _map_path(store, clip_id)
        if path.exists():
            value = json.loads(path.read_text())
            value["source_fingerprint"] = "stale"
            atomic_json(path, value)


def scene_export(store: Store, clip: Clip) -> dict:
    scene = get_scene(store, clip)
    value = {"scene_annotations": scene.model_dump()}
    # Export poses even if map points are omitted: this is the scene coordinate
    # contract and is required for exact cross-server ICP round trips.
    value["scene_registration"] = {"world_from_lidar": {fid: matrix.tolist() for fid, matrix in registered_poses(store, clip).items()}}
    try:
        cached = get_scene_map(store, clip)
        value["scene_map"] = {key: deepcopy(cached[key]) for key in (
            "coordinate_frame", "frame_count", "point_count", "trajectory", "bounds", "method", "voxel_size", "registration", "exclude_dynamic") if key in cached}
    except KeyError:
        pass
    return value


def validate_scene_import(clip: Clip, scene_raw: dict | None, scene_map: dict | None = None,
                          scene_registration: dict | None = None) -> dict | None:
    """Validate a complete scene bundle before publishing any imported files."""
    if scene_raw is None:
        if scene_registration is not None or scene_map is not None:
            raise ValueError("scene map/registration metadata requires scene_annotations")
        return None
    scene = SceneAnnotations.model_validate(scene_raw).validate_for_clip(clip)
    poses = None
    if scene_registration is not None:
        if not isinstance(scene_registration, dict) or not isinstance(scene_registration.get("world_from_lidar"), dict):
            raise ValueError("scene registration must be an object with world_from_lidar transforms")
        raw = scene_registration.get("world_from_lidar", {})
        if set(raw) != {f.id for f in clip.frames}:
            raise ValueError("scene registration must supply exactly the clip frame IDs")
        poses = {fid: _matrix(matrix, "scene world_from_lidar").tolist() for fid, matrix in raw.items()}
    if scene_map is not None:
        # Cached point geometry is deliberately not restored. Only small
        # diagnostic metadata exported by scene_export is accepted.
        if not isinstance(scene_map, dict):
            raise ValueError("scene map metadata must be an object")
        if scene_map.get("coordinate_frame", "world") != "world":
            raise ValueError("scene map metadata must use world coordinates")
        try:
            json.dumps(scene_map, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("scene map metadata must contain finite JSON values") from exc
    return {"scene_annotations": scene.model_dump(),
            "scene_registration": {"world_from_lidar": poses} if poses is not None else None}


def restore_scene(store: Store, clip: Clip, scene_raw: dict | None,
                  scene_map: dict | None = None, scene_registration: dict | None = None):
    """Restore validated world labels and corrected poses, never trust cached points."""
    bundle = validate_scene_import(clip, scene_raw, scene_map, scene_registration)
    if bundle is None:
        return
    with store.lock:
        atomic_json(_scene_path(store, clip.id), bundle["scene_annotations"])
        if bundle["scene_registration"] is not None:
            atomic_json(_poses_path(store, clip.id), {"clip_fingerprint": _clip_fingerprint(store, clip),
                                                     "world_from_lidar": bundle["scene_registration"]["world_from_lidar"]})
