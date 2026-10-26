#!/usr/bin/env python3
"""Install the reconstructed 4D clip fixture as an OmniLabel scene."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from omnilabel.models import Annotations, Clip  # noqa: E402
from omnilabel.scene import FUSION_VERSION, source_fingerprint  # noqa: E402
from omnilabel.scene_models import SceneAnnotations  # noqa: E402
from omnilabel.storage import Store, atomic_json, safe_id  # noqa: E402


CAMERAS = (
    ("front", "Front", "CAM_FRONT"),
    ("front_short", "Front short", "CAM_FRONT_SHORT"),
    ("front_wide", "Front wide", "CAM_FRONT_WIDE"),
    ("back_left", "Back left", "CAM_BACK_LEFT"),
    ("left_wide", "Left wide", "CAM_LEFT_WIDE"),
    ("back_right", "Back right", "CAM_BACK_RIGHT"),
    ("right_wide", "Right wide", "CAM_RIGHT_WIDE"),
)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def quaternion_transform(rotation, translation) -> np.ndarray:
    q = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError("pose must contain finite wxyz rotation and xyz translation")
    q /= np.linalg.norm(q)
    w, x, y, z = q
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = t
    return matrix


def correction_transform(raw: dict) -> np.ndarray:
    """Interpret source projection corrections as ego-local cm/degrees."""
    roll, pitch, yaw = np.deg2rad([
        float(raw.get("droll", 0)), float(raw.get("dpitch", 0)), float(raw.get("dyaw", 0)),
    ])
    rx = np.asarray([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.asarray([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.asarray([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = np.asarray([raw.get("dx", 0), raw.get("dy", 0), raw.get("dz", 0)], dtype=float) / 100.0
    if not np.isfinite(result).all():
        raise ValueError("projection correction contains a non-finite value")
    return result


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return np.round(matrix, 12).tolist()


def read_pcd_header(path: Path) -> tuple[int, int]:
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: incomplete PCD header")
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: DATA declaration is missing") from exc
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                offset = stream.tell()
                break
    expected = {
        "FIELDS": ["x", "y", "z", "intensity"],
        "SIZE": ["8", "8", "8", "4"],
        "TYPE": ["F", "F", "F", "F"],
        "COUNT": ["1", "1", "1", "1"],
        "DATA": ["binary"],
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise ValueError(f"{path}: unsupported PCD {key} {header.get(key)!r}")
    count = int(header.get("POINTS", ["0"])[0])
    if count <= 0 or path.stat().st_size != offset + count * 28:
        raise ValueError(f"{path}: PCD point count or byte size is inconsistent")
    return offset, count


def convert_pcd_preview(source: Path, destination: Path, max_points: int,
                        detail_intensity: float = 60.0) -> dict:
    offset, count = read_pcd_header(source)
    stride = max(1, math.ceil(count / max_points))
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f4")])
    records = np.memmap(source, dtype=dtype, mode="r", offset=offset, shape=(count,))
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    intensity_min, intensity_max = np.inf, -np.inf
    detail_indices = []
    for start in range(0, count, 250_000):
        stop = min(count, start + 250_000)
        chunk = records[start:stop]
        xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"]))
        intensities = np.asarray(chunk["intensity"])
        if not np.isfinite(xyz).all() or not np.isfinite(intensities).all():
            raise ValueError(f"{source}: point cloud contains non-finite values")
        minimum = np.minimum(minimum, xyz.min(axis=0))
        maximum = np.maximum(maximum, xyz.max(axis=0))
        intensity_min = min(intensity_min, float(intensities.min()))
        intensity_max = max(intensity_max, float(intensities.max()))
        detail_indices.append(np.flatnonzero(intensities >= detail_intensity).astype(np.int64) + start)
    base_indices = np.arange(0, count, stride, dtype=np.int64)
    detail = np.concatenate(detail_indices) if detail_indices else np.empty(0, dtype=np.int64)
    selected = np.union1d(base_indices, detail)
    output_count = len(selected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    points = np.lib.format.open_memmap(destination, mode="w+", dtype="<f4", shape=(output_count, 4))
    for start in range(0, output_count, 250_000):
        indices = selected[start:start + 250_000]
        chunk = records[indices]
        points[start:start + len(indices), 0] = chunk["x"]
        points[start:start + len(indices), 1] = chunk["y"]
        points[start:start + len(indices), 2] = chunk["z"]
        points[start:start + len(indices), 3] = chunk["intensity"]
    points.flush()
    return {
        "source_count": count,
        "point_count": output_count,
        "stride": stride,
        "detail_intensity": detail_intensity,
        "detail_point_count": len(detail),
        "bounds": {"min": minimum.tolist(), "max": maximum.tolist()},
        "intensity_range": [intensity_min, intensity_max],
    }


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def annotations_for_frames(clip: Clip, vectors: list[dict], tracks: list[dict]) -> dict[str, Annotations]:
    raw = {
        "revision": 0,
        "boxes": [{
            "id": track["id"], "track_id": track["id"], "label": track["label"],
            "center": track["keyframes"][0]["center"], "size": track["keyframes"][0]["size"],
            "yaw": track["keyframes"][0]["yaw"],
        } for track in tracks],
        "vectors": vectors, "image_polygons": [], "point_labels": {},
    }
    return {
        frame.id: Annotations.model_validate(raw).validate_for_frame(frame, clip.cameras)
        for frame in clip.frames
    }


def sync_existing_frame_annotations(data_root: Path, clip_id: str) -> dict:
    """Mirror world labels for an already installed clip without overwriting edits."""
    store = Store(data_root)
    clip = store.clip(clip_id)
    scene_path = data_root / "scenes" / f"{clip_id}.json"
    scene = SceneAnnotations.model_validate_json(scene_path.read_text(encoding="utf-8")).validate_for_clip(clip)
    annotations = annotations_for_frames(
        clip, [item.model_dump() for item in scene.vectors], [item.model_dump() for item in scene.tracks],
    )
    destination = data_root / "annotations" / clip_id
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"frame annotations already exist for {clip_id}; refusing to overwrite edits")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{clip_id}-labels-", dir=destination.parent))
    try:
        for frame_id, value in annotations.items():
            atomic_json(staging / f"{frame_id}.json", value.model_dump())
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    map_path = data_root / "scene_maps" / f"{clip_id}.json"
    if map_path.is_file():
        metadata = load_json(map_path)
        metadata["source_fingerprint"] = source_fingerprint(store, clip)
        atomic_json(map_path, metadata)
    return {"clip_id": clip_id, "frames_updated": len(annotations),
            "boxes_per_frame": len(scene.tracks), "vectors_per_frame": len(scene.vectors)}


def refresh_existing_point_preview(source: Path, data_root: Path, clip_id: str,
                                   max_points: int, detail_intensity: float) -> dict:
    """Replace a sparse installed preview with a lane-detail-preserving one."""
    store = Store(data_root)
    clip = store.clip(clip_id)
    pcd_files = sorted((source / "lidar_point_cloud").glob("*.pcd"))
    if len(pcd_files) != 1:
        raise ValueError("lidar_point_cloud must contain exactly one PCD file")
    relative = clip.frames[0].pointcloud_file
    if not relative or any(frame.pointcloud_file != relative for frame in clip.frames):
        raise ValueError("installed clip does not use one shared reconstructed-map point file")
    point_path = data_root / "pointclouds" / relative
    scene_path = data_root / "scene_pointclouds" / relative
    temporary = point_path.with_name(f".{point_path.name}.detail.tmp")
    temporary_scene = scene_path.with_name(f".{scene_path.name}.detail.tmp")
    try:
        cloud = convert_pcd_preview(pcd_files[0], temporary, max_points, detail_intensity)
        os.replace(temporary, point_path)
        link_or_copy(point_path, temporary_scene)
        os.replace(temporary_scene, scene_path)

        raw_clip = clip.model_dump()
        for frame in raw_clip["frames"]:
            frame["point_count"] = cloud["point_count"]
        raw_clip["description"] = (
            "50-frame, seven-camera 4D clip imported from tests/4D_clip_DATA_01_20260808160513v12. "
            "Global detection/map labels use 147 per-frame projection corrections. "
            f"Interactive cloud keeps every {cloud['stride']}th background point plus all "
            f"{cloud['detail_point_count']:,} points with intensity >= {detail_intensity:g}, preserving lane-marking detail "
            f"from the supplied {cloud['source_count']:,}-point reconstructed map."
        )
        updated = Clip.model_validate(raw_clip)
        atomic_json(data_root / "clips" / f"{clip_id}.json", updated.model_dump())

        from omnilabel.scene import _clip_fingerprint
        poses_path = data_root / "scene_poses" / f"{clip_id}.json"
        poses = load_json(poses_path)
        poses["clip_fingerprint"] = _clip_fingerprint(store, updated)
        atomic_json(poses_path, poses)

        map_path = data_root / "scene_maps" / f"{clip_id}.json"
        metadata = load_json(map_path)
        metadata.update(
            point_count=cloud["point_count"], bounds=cloud["bounds"],
            point_reduction=(f"all intensity >= {detail_intensity:g} plus deterministic source-order "
                             f"background stride {cloud['stride']}"),
            intensity_range=cloud["intensity_range"], input_point_count=cloud["source_count"],
            output_truncated=cloud["point_count"] < cloud["source_count"],
        )
        metadata["source_fingerprint"] = source_fingerprint(store, updated)
        atomic_json(map_path, metadata)
        return {"clip_id": clip_id, **cloud}
    finally:
        temporary.unlink(missing_ok=True)
        temporary_scene.unlink(missing_ok=True)


def build_scene(source: Path, clip_id: str, data_root: Path, max_points: int,
                detail_intensity: float) -> dict:
    pose_files = sorted((source / "global_pose").glob("*.json"))
    if len(pose_files) != 1:
        raise ValueError("global_pose must contain exactly one JSON file")
    pose_data = load_json(pose_files[0])
    frame_items = pose_data.get("frame_items")
    if not isinstance(frame_items, list) or not frame_items:
        raise ValueError("global pose has no frame_items")
    if pose_data.get("camera_channels") != [item[2] for item in CAMERAS]:
        raise ValueError("camera channel order does not match the known seven-camera layout")

    image_dirs = [source / f"camera_image_{index}" for index in range(len(CAMERAS))]
    config_dir = source / "camera_config"
    config_by_stem = {path.stem: load_json(path) for path in sorted(config_dir.glob("*.json"))}
    if set(config_by_stem) != {item["target_frame_stem"] for item in frame_items}:
        raise ValueError("camera_config files do not exactly match global pose frames")

    staged_root = Path(tempfile.mkdtemp(prefix=f".{clip_id}-", dir=data_root))
    staged_assets = staged_root / "assets" / clip_id
    staged_points = staged_root / "pointclouds" / clip_id
    staged_scene_points = staged_root / "scene_pointclouds" / clip_id
    installed = []
    published_clip = False
    try:
        pcd_files = sorted((source / "lidar_point_cloud").glob("*.pcd"))
        if len(pcd_files) != 1:
            raise ValueError("lidar_point_cloud must contain exactly one PCD file")
        point_name = "reconstructed-map-preview.npy"
        cloud = convert_pcd_preview(pcd_files[0], staged_points / point_name, max_points, detail_intensity)
        link_or_copy(staged_points / point_name, staged_scene_points / point_name)

        poses, frames, correction_count = {}, [], 0
        timestamps = [int(item["lidar_timestamp_us"]) for item in frame_items]
        if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("frame timestamps are not strictly increasing")
        first_timestamp = timestamps[0]
        static_extrinsics = []

        for frame_index, item in enumerate(frame_items):
            stem = item["target_frame_stem"]
            frame_id = f"frame-{frame_index:03d}"
            pose = quaternion_transform(item["ego_pose_local"]["rotation"], item["ego_pose_local"]["translation"])
            poses[frame_id] = np.eye(4).tolist()
            configs = config_by_stem[stem]
            if not isinstance(configs, list) or len(configs) != len(CAMERAS):
                raise ValueError(f"{stem}: expected seven camera configurations")
            images, annotation_transforms = {}, {}
            for camera_index, (camera_id, _, _) in enumerate(CAMERAS):
                image = image_dirs[camera_index] / f"{stem}.jpg"
                if not image.is_file():
                    raise ValueError(f"missing image {image}")
                destination = staged_assets / camera_id / f"{frame_id}.jpg"
                link_or_copy(image, destination)
                images[camera_id] = f"/api/assets/{clip_id}/{camera_id}/{frame_id}.jpg"

                external = np.asarray(configs[camera_index]["camera_external"], dtype=np.float64).reshape(4, 4)
                if frame_index == 0:
                    static_extrinsics.append(external @ pose)
                correction_path = source / "projection_corrections" / f"camera_image_{camera_index}" / f"{stem}.json"
                corrected = external
                if correction_path.is_file():
                    correction = load_json(correction_path)
                    if correction.get("camera") != f"camera_image_{camera_index}" or correction.get("frame") != stem:
                        raise ValueError(f"correction identity mismatch in {correction_path}")
                    # Source labels/map are in local-world coordinates. Conjugate
                    # the ego-local adjustment into that coordinate system.
                    corrected = external @ pose @ correction_transform(correction) @ np.linalg.inv(pose)
                    correction_count += 1
                annotation_transforms[camera_id] = matrix_list(corrected)

            frames.append({
                "id": frame_id,
                "timestamp": (timestamps[frame_index] - first_timestamp) / 1_000_000,
                "ego_to_world": matrix_list(pose),
                # The supplied map and labels already use local-world axes.
                "lidar_to_ego": matrix_list(np.linalg.inv(pose)),
                "points": [],
                "pointcloud_file": f"{clip_id}/{point_name}",
                "point_count": cloud["point_count"],
                "images": images,
                "annotation_camera_from_lidar": annotation_transforms,
            })

        for frame_index, item in enumerate(frame_items):
            pose = quaternion_transform(item["ego_pose_local"]["rotation"], item["ego_pose_local"]["translation"])
            configs = config_by_stem[item["target_frame_stem"]]
            for camera_index in range(len(CAMERAS)):
                external = np.asarray(configs[camera_index]["camera_external"], dtype=np.float64).reshape(4, 4)
                if not np.allclose(external @ pose, static_extrinsics[camera_index], atol=2e-5):
                    raise ValueError("camera external matrices are not rigidly attached to ego across frames")

        first_configs = config_by_stem[frame_items[0]["target_frame_stem"]]
        cameras = []
        for index, (camera_id, name, _) in enumerate(CAMERAS):
            config = first_configs[index]
            intrinsic = config["camera_internal"]
            k = config.get("distortionK", [0, 0, 0])
            p = config.get("distortionP", [0, 0])
            cameras.append({
                "id": camera_id, "name": name, "width": 1920, "height": 1536,
                "intrinsics": [[intrinsic["fx"], 0, intrinsic["cx"]], [0, intrinsic["fy"], intrinsic["cy"]], [0, 0, 1]],
                "camera_from_ego": matrix_list(static_extrinsics[index]),
                "distortion": [k[0], k[1], p[0], p[1], k[2]],
            })

        diffs = np.diff(np.asarray(timestamps, dtype=np.float64)) / 1_000_000
        fps = float(1 / np.median(diffs)) if len(diffs) else 1.0
        clip = Clip.model_validate({
            "id": clip_id,
            "name": "4D DATA_01 20260808160513 v12",
            "description": (
                "50-frame, seven-camera 4D clip imported from tests/4D_clip_DATA_01_20260808160513v12. "
                f"Global detection/map labels use {correction_count} per-frame projection corrections. "
                f"Interactive point-cloud preview retains every {cloud['stride']}th background point and all "
                f"intensity >= {detail_intensity:g} detail points from the supplied "
                f"{cloud['source_count']:,}-point reconstructed map."
            ),
            "fps": fps, "cameras": cameras, "frames": frames, "source": "4d-clip-import",
        })

        objects = load_json(source / "label" / "objects.json")
        map_objects = load_json(source / "label" / "map.json")
        vectors = [{
            "id": f"map-{item['obj_id']}",
            "label": str(item["obj_type"]).lower().replace("_", "-"),
            "points": [[point["x"], point["y"], point["z"]] for point in item["points"]],
            "closed": item.get("ann_type") == "polygon",
        } for item in map_objects]
        tracks = [{
            "id": f"object-{item['obj_id']}",
            "label": str(item["obj_type"]).lower().replace("_", "-"),
            "mode": "static",
            "keyframes": [{
                "frame_id": "frame-000",
                "center": [item["psr"]["position"][axis] for axis in ("x", "y", "z")],
                "size": [item["psr"]["scale"][axis] for axis in ("x", "y", "z")],
                "yaw": item["psr"]["rotation"]["z"],
            }],
        } for item in objects]
        scene = SceneAnnotations.model_validate({
            "revision": 0, "coordinate_frame": "world", "vectors": vectors, "tracks": tracks,
        }).validate_for_clip(clip)
        # The supplied cloud, boxes and map vectors all use the same local-world
        # coordinates. Mirror them into legacy per-frame annotations as well so
        # image overlays work in both "whole scene" and "single frame" modes.
        frame_annotations = annotations_for_frames(clip, vectors, tracks)

        targets = [
            (staged_assets, data_root / "assets" / clip_id),
            (staged_points, data_root / "pointclouds" / clip_id),
            (staged_scene_points, data_root / "scene_pointclouds" / clip_id),
        ]
        other_targets = [
            data_root / "clips" / f"{clip_id}.json",
            data_root / "scenes" / f"{clip_id}.json",
            data_root / "scene_poses" / f"{clip_id}.json",
            data_root / "scene_maps" / f"{clip_id}.json",
            data_root / "annotations" / clip_id,
        ]
        if any(destination.exists() for _, destination in targets) or any(path.exists() for path in other_targets):
            raise FileExistsError(f"clip output already exists for {clip_id}")
        for staged, destination in targets:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)
            installed.append(destination)

        store = Store(data_root)
        store.add_clip(clip, frame_annotations)
        published_clip = True
        atomic_json(data_root / "scenes" / f"{clip_id}.json", scene.model_dump())
        identity_poses = {frame.id: np.eye(4).tolist() for frame in clip.frames}
        from omnilabel.scene import _clip_fingerprint  # Imported only for the persisted registration contract.
        atomic_json(data_root / "scene_poses" / f"{clip_id}.json", {
            "clip_fingerprint": _clip_fingerprint(store, clip), "world_from_lidar": identity_poses,
        })
        map_metadata = {
            "coordinate_frame": "world", "frame_count": len(frames), "point_count": cloud["point_count"],
            "trajectory": [],
            "bounds": cloud["bounds"], "method": "supplied reconstructed local-world map",
            "voxel_size": float(pose_data.get("map_voxel_m", 0.1)),
            "registration": {"method": "supplied-global-poses", "successful_frames": len(frames)},
            "exclude_dynamic": True, "fusion_version": FUSION_VERSION, "preserve_detail": True,
            "detail_voxel_size": float(pose_data.get("map_voxel_m", 0.1)),
            "effective_voxel_size": float(pose_data.get("map_voxel_m", 0.1)),
            "point_reduction": (f"all intensity >= {detail_intensity:g} plus deterministic source-order "
                                f"background stride {cloud['stride']}"),
            "intensity_available": True,
            "intensity_has_contrast": cloud["intensity_range"][1] > cloud["intensity_range"][0],
            "intensity_range": cloud["intensity_range"], "fusion_mode": "full",
            "input_point_count": cloud["source_count"], "excluded_dynamic_points": 0,
            "excluded_outlier_points": 0, "outlier_filter": False,
            "input_truncated": False, "output_truncated": cloud["point_count"] < cloud["source_count"],
            "scene_revision": scene.revision, "source_fingerprint": source_fingerprint(store, clip),
            "world_from_lidar": identity_poses,
            "pointcloud_file": f"{clip_id}/{point_name}", "point_labels": {},
        }
        # Build the trajectory separately to keep its position representation explicit.
        map_metadata["trajectory"] = [{
            "frame_id": frame["id"], "timestamp": frame["timestamp"],
            "position": list(frame_items[index]["ego_pose_local"]["translation"]),
        } for index, frame in enumerate(frames)]
        atomic_json(data_root / "scene_maps" / f"{clip_id}.json", map_metadata)
        return {
            "clip_id": clip.id, "frames": len(clip.frames), "cameras": len(clip.cameras),
            "objects": len(tracks), "map_vectors": len(vectors), "corrections": correction_count,
            **cloud,
        }
    except Exception:
        if published_clip:
            for path in (
                data_root / "clips" / f"{clip_id}.json", data_root / "scenes" / f"{clip_id}.json",
                data_root / "scene_poses" / f"{clip_id}.json", data_root / "scene_maps" / f"{clip_id}.json",
            ):
                path.unlink(missing_ok=True)
            shutil.rmtree(data_root / "annotations" / clip_id, ignore_errors=True)
        for path in reversed(installed):
            shutil.rmtree(path, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staged_root, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--clip-id", default="4d-clip-data-01-20260808160513-v12")
    parser.add_argument("--max-preview-points", type=int, default=750_000)
    parser.add_argument("--detail-intensity", type=float, default=60.0,
                        help="retain every point at or above this intensity in addition to the background preview")
    parser.add_argument("--sync-frame-annotations", action="store_true",
                        help="mirror scene labels into empty per-frame annotations of an installed clip")
    parser.add_argument("--refresh-point-preview", action="store_true",
                        help="replace an installed sparse preview with a lane-detail-preserving preview")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_preview_points <= 0:
        raise SystemExit("--max-preview-points must be positive")
    clip_id = safe_id(args.clip_id)
    if args.sync_frame_annotations and args.refresh_point_preview:
        raise SystemExit("choose only one maintenance action")
    if not np.isfinite(args.detail_intensity):
        raise SystemExit("--detail-intensity must be finite")
    result = (sync_existing_frame_annotations(args.data_root.resolve(), clip_id)
              if args.sync_frame_annotations else
              refresh_existing_point_preview(args.source.resolve(), args.data_root.resolve(), clip_id,
                                             args.max_preview_points, args.detail_intensity)
              if args.refresh_point_preview else
              build_scene(args.source.resolve(), clip_id, args.data_root.resolve(), args.max_preview_points,
                          args.detail_intensity))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
