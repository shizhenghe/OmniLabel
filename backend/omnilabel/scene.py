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
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from .geometry import DYNAMIC_LABELS, _inside_box, _matrix, _points, _static_chunks, _transform, _voxel_indices, fuse_map
from .models import Annotations, Clip, Frame
from .scene_models import ReconstructionRequest, SceneAnnotations
from .registration import register_multiscale, spatial_sample
from .storage import ConflictError, Store, atomic_json, safe_id

MAX_ICP_POINTS = 15000  # Solver sample budget only; source/fused map is never truncated.
MAX_ICP_ITERATIONS = 45
MAX_ICP_PAIR_DISTANCE = 3.0
FUSION_VERSION = 3
SCENE_PREVIEW_POINTS = None  # No implicit map preview truncation.


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
        point_root = (store.root / "pointclouds").resolve()
        for frame in clip.frames:
            if frame.pointcloud_file:
                source = (point_root / frame.pointcloud_file).resolve()
                if not source.is_relative_to(point_root):
                    raise ValueError("Referenced point cloud is outside the pointcloud store")
                if source.is_file():
                    info = source.stat()
                    token.append([frame.pointcloud_file, info.st_size, info.st_mtime_ns])
                else:
                    token.append([frame.pointcloud_file, None])
    else:
        token = clip.model_dump()
    return sha256(json.dumps(token, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_fingerprint(store: Store, clip: Clip, *, include_inode: bool) -> str:
    tokens = [_clip_fingerprint(store, clip)]
    for path in [_scene_path(store, clip.id)] + [store.root / "annotations" / clip.id / f"{f.id}.json" for f in clip.frames]:
        if path.exists():
            stat = path.stat()
            # Atomic publication replaces the inode. Equal-sized edits can
            # share a timestamp on some filesystems and must still conflict.
            tokens.append([path.name, *([stat.st_ino] if include_inode else []), stat.st_size, stat.st_mtime_ns])
        else:
            tokens.append([path.name, None])
    return sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def legacy_source_fingerprint(store: Store, clip: Clip) -> str:
    """Validate maps saved before atomic annotation identity was included."""
    return _source_fingerprint(store, clip, include_inode=False)


def source_fingerprint(store: Store, clip: Clip) -> str:
    """Cheap on repeated reads; saved scans are immutable and annotations atomic."""
    return "v2:" + _source_fingerprint(store, clip, include_inode=True)


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


def _fusion_annotations(frame: Frame, annotations: dict, exclude_dynamic: bool, point_loader=None) -> dict:
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
        # Test only the annotated candidate rows, in batches. Loading/converting
        # the whole scan to protect a handful of static labels wastes RAM.
        raw = point_loader(frame.model_dump()) if point_loader is not None else frame.points
        candidates = (index for index, label in value.get("point_labels", {}).items()
                      if label.lower() in DYNAMIC_LABELS)
        pending = []

        def protect(indices):
            points = np.asarray([raw[int(index)][:3] for index in indices], dtype=float)
            protected = np.zeros(len(points), dtype=bool)
            for box in static_boxes:
                protected |= _inside_box(points, box)
            for index, keep in zip(indices, protected):
                if keep:
                    value["point_labels"][index] = f"scene-static:{value['point_labels'][index]}"

        for index in candidates:
            pending.append(index)
            if len(pending) == 100_000:
                protect(pending)
                pending = []
        if pending:
            protect(pending)
    return value


def _static_scan(frame: Frame, annotations: dict, exclude_dynamic: bool, point_loader=None) -> np.ndarray:
    cloud = _points(frame.model_dump(), point_loader)
    if not exclude_dynamic:
        return cloud
    keep = np.ones(len(cloud), dtype=bool)
    for box in annotations.get("boxes", []):
        if box["label"].lower() in DYNAMIC_LABELS:
            keep &= ~_inside_box(cloud, box)
    for index, label in annotations.get("point_labels", {}).items():
        if label.lower() in DYNAMIC_LABELS and 0 <= int(index) < len(cloud):
            keep[int(index)] = False
    return cloud[keep]


def _registration_sample(points: np.ndarray, voxel_size: float, max_points: int = 15000) -> np.ndarray:
    return spatial_sample(points, voxel_size, max_points)


def _registration_scan(frame: Frame, annotations: dict, request: ReconstructionRequest, point_loader=None) -> np.ndarray:
    # Keep registration in the sensor coordinate frame while streaming all
    # packets. A bounded spatial reservoir avoids materialising a huge scan.
    raw = frame.model_dump()
    raw["ego_to_world"] = np.eye(4).tolist()
    raw["lidar_to_ego"] = np.eye(4).tolist()
    candidate = np.empty((0, 3))
    voxel = request.registration_voxel_size * .5
    for static, _, _, _, _ in _static_chunks({"frames": [raw]}, {frame.id: annotations}, point_loader):
        if not len(static):
            continue
        sample = spatial_sample(static[:, :3], voxel, request.registration_sample_size)
        candidate = spatial_sample(np.concatenate((candidate, sample)), voxel, request.registration_sample_size)
    return candidate


def _spatial_support(points: np.ndarray) -> bool:
    if len(points) < 30:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    # A plane is valid; an isolated point/line cannot constrain rigid registration.
    return len(singular) >= 2 and singular[1] > max(1e-5, singular[0] * 1e-4)


def register_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray,
                 max_distance: float = MAX_ICP_PAIR_DISTANCE) -> tuple[np.ndarray, dict]:
    """Compatibility entry point backed by the robust multiscale solver."""
    return register_multiscale(source, target, initial, max_distance=max_distance)


def reconstruct_scene(clip: Clip, annotations: dict[str, dict], request: ReconstructionRequest,
                      point_loader=None, *, array_result: bool = False) -> dict:
    """Build one world-space map, retaining an explicit scan trajectory."""
    poses = source_poses(clip)
    fusion_annotations = {frame.id: _fusion_annotations(frame, annotations.get(frame.id, {}), request.exclude_dynamic,
                                                        point_loader)
                          for frame in clip.frames}
    reports = []
    if request.registration == "icp":
        history = []
        previous_frame = None
        original_poses = source_poses(clip)
        for frame in clip.frames:
            # Sampling is only for the registration solver. Final fusion reads
            # every original point again, independently of this solver budget.
            scan = _registration_scan(frame, fusion_annotations.get(frame.id, {}), request, point_loader)
            if previous_frame is None:
                if not _spatial_support(scan):
                    raise ValueError("ICP reference scan needs at least 30 non-collinear static points")
            else:
                initial = poses[previous_frame.id] @ np.linalg.inv(original_poses[previous_frame.id]) @ original_poses[frame.id]
                # Register to several already-aligned scans. This retains useful
                # static surfaces when adjacent packets have partial occlusion.
                target = np.concatenate([entry[1] for entry in history[-request.submap_frames:]])
                poses[frame.id], report = register_multiscale(
                    scan, target, initial, base_voxel=request.registration_voxel_size,
                    max_points=request.registration_sample_size)
                reports.append({"frame_id": frame.id, "reference_frame_id": previous_frame.id,
                                "reference_frame_ids": [entry[0] for entry in history[-request.submap_frames:]], **report})
            history.append((frame.id, _transform(scan, poses[frame.id])))
            if len(history) > request.submap_frames:
                history.pop(0)
            previous_frame = frame
    corrected = clip.model_dump()
    for frame in corrected["frames"]:
        frame["ego_to_world"] = (poses[frame["id"]] @ np.linalg.inv(np.asarray(frame["lidar_to_ego"], dtype=float))).tolist()
    result = fuse_map(corrected, fusion_annotations, request.voxel_size,
                      preserve_detail=request.preserve_detail, detail_voxel_size=request.detail_voxel_size,
                      point_loader=point_loader, fusion_mode=request.fusion_mode,
                      outlier_filter=request.outlier_filter, outlier_neighbors=request.outlier_neighbors,
                      outlier_std_ratio=request.outlier_std_ratio, return_array=array_result)
    result["point_labels"] = {index: label.removeprefix("scene-static:") for index, label in result.get("point_labels", {}).items()}
    points = np.asarray(result["points"], dtype=float)
    if not len(points):
        raise ValueError("Reconstruction produced no static map points; inspect scan contents and dynamic filtering")
    result.update(
        trajectory=[{"frame_id": f.id, "timestamp": f.timestamp, "position": poses[f.id][:3, 3].tolist()} for f in clip.frames],
        world_from_lidar={fid: matrix.tolist() for fid, matrix in poses.items()},
        bounds={"min": points[:, :3].min(axis=0).tolist(), "max": points[:, :3].max(axis=0).tolist()},
        registration={"mode": request.registration, "frames": reports, "loop_closure": False,
                      "note": "Uses supplied poses" if request.registration == "poses" else "Multiscale trimmed/Huber scan-to-local-submap ICP; no loop closure or global SLAM accuracy guarantee"},
        method="pose-aligned whole-clip static reconstruction" if request.registration == "poses" else "multiscale local-submap ICP whole-clip static reconstruction",
        exclude_dynamic=request.exclude_dynamic,
        fusion_version=FUSION_VERSION,
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


def _stored_map_points(store: Store, value: dict, *, check_values: bool = True) -> np.ndarray:
    relative = value.get("pointcloud_file")
    if not isinstance(relative, str):
        return np.asarray(value.get("points", []), dtype=float)
    root = (store.root / "scene_pointclouds").resolve()
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or source.suffix.lower() != ".npy" or not source.is_file():
        raise ValueError("Scene map point cloud is missing or outside the pointcloud store")
    try:
        points = np.load(source, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("Scene map point cloud is not a valid NPY array") from exc
    if (points.ndim != 2 or points.shape[1] != 4 or len(points) != value.get("point_count")
            or not np.issubdtype(points.dtype, np.number)):
        raise ValueError("Scene map point cloud has invalid shape, count, or type")
    if check_values:
        for start in range(0, len(points), 100_000):
            if not np.isfinite(points[start:start + 100_000]).all():
                raise ValueError("Scene map point cloud has nonfinite values")
    return points


def is_supplied_scene_map(value: dict) -> bool:
    return (value.get("source_map") == "supplied" or
            value.get("method") == "supplied reconstructed local-world map")


def get_scene_map(store: Store, clip: Clip, *, max_points: int | None = None,
                  include_points: bool = True) -> dict:
    path = _map_path(store, clip.id)
    with store.lock:
        if not path.exists():
            raise KeyError("Scene map has not been reconstructed")
        value = json.loads(path.read_text())
        if value.get("fusion_version") != FUSION_VERSION:
            raise KeyError("Scene fusion profile changed; reconstruct the scene map to retain measured surface detail")
        # A supplied world map survives annotation/mount edits. Legacy
        # calibration saves wrote a stale source token even though that
        # immutable world geometry remained valid. A separate explicit
        # geometry invalidation still prevents serving genuinely stale points.
        if value.get("geometry_invalidated") is True:
            raise KeyError("Scene map was explicitly invalidated; reconstruct or restore the scene map")
        supplied_map = is_supplied_scene_map(value)
        if supplied_map:
            # Cached pose metadata is not the scene coordinate authority:
            # calibration can change it without changing any world-map point.
            value["world_from_lidar"] = {fid: pose.tolist() for fid, pose in registered_poses(store, clip).items()}
        if not supplied_map:
            saved_token = value.get("source_fingerprint")
            current_token = source_fingerprint(store, clip)
            legacy_match = (isinstance(saved_token, str) and not saved_token.startswith("v2:")
                            and saved_token == legacy_source_fingerprint(store, clip))
            if saved_token != current_token and not legacy_match:
                raise KeyError("Scene source annotations changed; reconstruct the scene map to refresh dynamic filtering")
        if not include_points:
            value.pop("points", None)
            value.pop("pointcloud_file", None)
            return value
        points = _stored_map_points(store, value)
        if max_points is not None and max_points <= 0:
            raise ValueError("An explicit preview max_points must be positive")
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64) if max_points is not None and len(points) > max_points else None
        selected = points if indices is None else points[indices]
        value["points"] = np.asarray(selected, dtype=float).tolist()
        labels = value.get("point_labels", {})
        if indices is not None and labels:
            value["point_labels"] = {
                str(output): labels[str(source)] for output, source in enumerate(indices) if str(source) in labels
            }
        value["rendered_point_count"] = len(selected)
        value["preview_truncated"] = indices is not None
        value.pop("pointcloud_file", None)
        return value


def scene_map_page(store: Store, clip: Clip, offset: int = 0, limit: int | None = None) -> dict:
    """Contiguous transport pages, with global indices; no points are sampled."""
    if offset < 0 or (limit is not None and limit <= 0):
        raise ValueError("Map point offset must be nonnegative and limit positive")
    with store.lock:
        metadata = get_scene_map(store, clip, include_points=False)
        stored = json.loads(_map_path(store, clip.id).read_text())
        points = _stored_map_points(store, stored, check_values=False)
        end = len(points) if limit is None else min(len(points), offset + limit)
        selected = points[offset:end]
        if not np.isfinite(selected).all():
            raise ValueError("Scene map point page contains nonfinite values")
        labels = stored.get("point_labels", {})
        return {"points": np.asarray(selected).tolist(), "point_offset": offset,
                "point_count": len(points), "returned_point_count": len(selected),
                "point_labels": {str(index): labels[str(index)] for index in range(offset, end) if str(index) in labels},
                "next_offset": end if end < len(points) else None, "coordinate_frame": "world",
                "source_fingerprint": metadata["source_fingerprint"], "sampling": False}


def scene_map_file(store: Store, clip: Clip) -> Path:
    with store.lock:
        get_scene_map(store, clip, include_points=False)
        value = json.loads(_map_path(store, clip.id).read_text())
        _stored_map_points(store, value, check_values=False)
        relative = value.get("pointcloud_file")
        if not isinstance(relative, str):
            raise KeyError("Reconstruct this scene to enable full-resolution NPY export")
        return store.root / "scene_pointclouds" / relative


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
        points = np.asarray(result.get("points", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 4 or len(points) != result.get("point_count"):
            raise ValueError("Reconstructed scene points do not match point_count")
        points_root = store.root / "scene_pointclouds" / safe_id(clip.id)
        points_root.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid4().hex}.npy"
        destination = points_root / filename
        temporary = points_root / f".{filename}.tmp"
        previous_file = None
        map_path = _map_path(store, clip.id)
        if map_path.exists() and map_path.stat().st_size < 16 * 1024 * 1024:
            try:
                previous_file = json.loads(map_path.read_text()).get("pointcloud_file")
            except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
                pass
        try:
            with temporary.open("wb") as stream:
                np.save(stream, points, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            stored = {key: value for key, value in result.items() if key != "points"}
            stored["pointcloud_file"] = f"{safe_id(clip.id)}/{filename}"
            atomic_json(map_path, stored)
        except Exception:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        if previous_file and previous_file != stored["pointcloud_file"]:
            previous_path = (store.root / "scene_pointclouds" / previous_file).resolve()
            root = (store.root / "scene_pointclouds").resolve()
            if previous_path.is_relative_to(root):
                previous_path.unlink(missing_ok=True)
        return result


def invalidate_scene_map(store: Store, clip_id: str):
    """Optional explicit invalidation; fingerprints also detect all saved edits."""
    with store.lock:
        path = _map_path(store, clip_id)
        if path.exists():
            value = json.loads(path.read_text())
            value["source_fingerprint"] = "stale"
            value["geometry_invalidated"] = True
            atomic_json(path, value)


def scene_export(store: Store, clip: Clip) -> dict:
    scene = get_scene(store, clip)
    value = {"scene_annotations": scene.model_dump()}
    # Export poses even if map points are omitted: this is the scene coordinate
    # contract and is required for exact cross-server ICP round trips.
    value["scene_registration"] = {"world_from_lidar": {fid: matrix.tolist() for fid, matrix in registered_poses(store, clip).items()}}
    try:
        cached = get_scene_map(store, clip, include_points=False)
        value["scene_map"] = {key: deepcopy(cached[key]) for key in (
            "coordinate_frame", "frame_count", "point_count", "trajectory", "bounds", "method", "voxel_size", "registration", "exclude_dynamic",
            "fusion_version", "preserve_detail", "detail_voxel_size", "effective_voxel_size", "point_reduction",
            "intensity_available", "intensity_has_contrast", "intensity_range", "fusion_mode",
            "input_point_count", "excluded_dynamic_points", "excluded_outlier_points", "outlier_filter",
            "input_truncated", "output_truncated") if key in cached}
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
