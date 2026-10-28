#!/usr/bin/env python3
"""Install a user's 4D label/projection drafts as an independent scene.

The existing clip supplies immutable image/NPY references only. Calibration and
labels are read from the source fixture's camera_config, global_pose,
label_draft/<user> and projection_corrections_draft/<user> directories. Published
labels, published projection corrections and local calibration overlays are
never read. No point cloud or image is copied, filtered or regenerated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from import_4d_clip import (  # noqa: E402
    CAMERAS, annotations_for_frames, correction_transform, load_json,
    matrix_list, quaternion_transform, read_pcd_header,
)
from omnilabel.models import Clip  # noqa: E402
from omnilabel.scene import FUSION_VERSION, _clip_fingerprint, source_fingerprint  # noqa: E402
from omnilabel.scene_models import SceneAnnotations  # noqa: E402
from omnilabel.storage import atomic_json, safe_id  # noqa: E402


DEFAULT_BASE_CLIP_ID = "4d-clip-data-01-20260808160513-v12"
DEFAULT_CLIP_ID = DEFAULT_BASE_CLIP_ID + "-draft-manager1"
DEFAULT_NAME = "4D DATA_01 v12 · manager1 草稿标注"
DELTA_KEYS = ("dx", "dy", "dz", "dyaw", "dpitch", "droll")


def _output_paths(data_root: Path, clip_id: str) -> dict[str, Path]:
    return {
        "clip": data_root / "clips" / f"{clip_id}.json",
        "annotations": data_root / "annotations" / clip_id,
        "scene": data_root / "scenes" / f"{clip_id}.json",
        "poses": data_root / "scene_poses" / f"{clip_id}.json",
        "map": data_root / "scene_maps" / f"{clip_id}.json",
        "report": data_root / "import_reports" / f"{clip_id}.json",
    }


def _check_destination(data_root: Path, clip_id: str):
    targets = list(_output_paths(data_root, clip_id).values()) + [
        data_root / "projection_calibrations" / clip_id,
        data_root / "projection_calibration_history" / clip_id,
        data_root / "calibration_history" / clip_id,
    ]
    present = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if present:
        raise FileExistsError("refusing to overwrite an existing draft scene: " + ", ".join(present))


def _stored_path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise ValueError(f"missing or invalid shared asset: {relative}")
    return path


def _npy_array(path: Path, count: int) -> np.ndarray:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid shared point cloud {path}") from exc
    if (array.shape != (count, 4) or array.dtype.kind not in "fiu"):
        raise ValueError(f"shared point cloud {path} does not retain all {count} source points")
    for start in range(0, count, 250_000):
        if not np.isfinite(array[start:start + 250_000]).all():
            raise ValueError(f"non-finite values in shared point cloud {path}")
    return array


def _load_corrections(directory: Path, stems: set[str]) -> tuple[dict, list[Path]]:
    if not directory.is_dir():
        raise ValueError(f"draft projection directory is missing: {directory}")
    files = sorted(directory.rglob("*.json"))
    if not files:
        raise ValueError(f"draft projection directory contains no corrections: {directory}")
    corrections = {}
    camera_names = {f"camera_image_{index}" for index in range(len(CAMERAS))}
    for path in files:
        relative = path.relative_to(directory)
        if len(relative.parts) != 2 or relative.parts[0] not in camera_names or path.stem not in stems:
            raise ValueError(f"draft correction does not match a source camera/frame: {path}")
        raw = load_json(path)
        if (not isinstance(raw, dict) or raw.get("camera") != path.parent.name
                or raw.get("frame") != path.stem or any(key not in raw for key in DELTA_KEYS)):
            raise ValueError(f"draft correction identity or parameters are invalid: {path}")
        if any(isinstance(raw[key], bool) or not isinstance(raw[key], (int, float))
               or not np.isfinite(raw[key]) for key in DELTA_KEYS):
            raise ValueError(f"draft correction parameters must be finite numbers: {path}")
        correction_transform(raw)  # Validate the same metres/degrees convention as the original importer.
        corrections[(path.parent.name, path.stem)] = raw
    return corrections, files


def prepare_draft(source: Path, data_root: Path, *, base_clip_id: str = DEFAULT_BASE_CLIP_ID,
                  clip_id: str = DEFAULT_CLIP_ID, draft_user: str = "manager1",
                  name: str = DEFAULT_NAME,
                  apply_projection_corrections: bool = True) -> dict:
    """Validate/build all small scene documents without writing to the data store."""
    source, data_root = source.resolve(), data_root.resolve()
    base_clip_id, clip_id, draft_user = map(safe_id, (base_clip_id, clip_id, draft_user))
    if base_clip_id == clip_id:
        raise ValueError("the draft must use a different clip ID from its shared asset source")
    _check_destination(data_root, clip_id)
    base_clip = Clip.model_validate(load_json(data_root / "clips" / f"{base_clip_id}.json"))
    if base_clip.id != base_clip_id:
        raise ValueError("base clip identity does not match its manifest filename")
    base_map = load_json(data_root / "scene_maps" / f"{base_clip_id}.json")
    supplied_map = (base_map.get("source_map") == "supplied"
                    or base_map.get("method") == "supplied reconstructed local-world map")
    if not supplied_map or base_map.get("fusion_version") != FUSION_VERSION:
        raise ValueError("base clip must have the complete supplied reconstructed map")

    pose_files = sorted((source / "global_pose").glob("*.json"))
    pcd_files = sorted((source / "lidar_point_cloud").glob("*.pcd"))
    if len(pose_files) != 1 or len(pcd_files) != 1:
        raise ValueError("source must contain exactly one global pose JSON and one reconstructed PCD")
    pose_data = load_json(pose_files[0])
    frame_items = pose_data.get("frame_items")
    if (not isinstance(frame_items, list) or not frame_items
            or len(frame_items) != len(base_clip.frames)
            or pose_data.get("num_frames") != len(frame_items)):
        raise ValueError("source global poses do not match the installed clip frame count")
    if pose_data.get("camera_channels") != [item[2] for item in CAMERAS]:
        raise ValueError("source camera channel order does not match the seven-camera layout")
    camera_ids = [item[0] for item in CAMERAS]
    if {camera.id for camera in base_clip.cameras} != set(camera_ids):
        raise ValueError("installed clip does not have the seven source cameras")
    source_stems = {item["target_frame_stem"] for item in frame_items}
    if len(source_stems) != len(frame_items):
        raise ValueError("source global poses contain duplicate frame names")
    config_files = sorted((source / "camera_config").glob("*.json"))
    configs_by_stem = {path.stem: load_json(path) for path in config_files}
    if set(configs_by_stem) != source_stems:
        raise ValueError("camera_config files do not exactly match source global poses")

    _, point_count = read_pcd_header(pcd_files[0])
    point_file = base_clip.frames[0].pointcloud_file
    if (point_file is None or any(frame.pointcloud_file != point_file or frame.point_count != point_count
                                  for frame in base_clip.frames)):
        raise ValueError("installed clip does not retain the complete shared source point cloud")
    if (base_map.get("pointcloud_file") != point_file or base_map.get("point_count") != point_count
            or base_map.get("input_point_count") != point_count or base_map.get("output_truncated")
            or base_map.get("input_truncated") or base_map.get("frame_count") != len(frame_items)):
        raise ValueError("installed scene map count or shared point-cloud mapping does not match the source")
    frame_points = _stored_path(data_root / "pointclouds", point_file)
    scene_points = _stored_path(data_root / "scene_pointclouds", point_file)
    point_array = _npy_array(frame_points, point_count)
    if not os.path.samefile(frame_points, scene_points):
        scene_array = _npy_array(scene_points, point_count)
        for start in range(0, point_count, 250_000):
            if not np.array_equal(point_array[start:start + 250_000], scene_array[start:start + 250_000]):
                raise ValueError("shared per-frame and scene point clouds do not contain identical source points")

    labels_directory = source / "label_draft" / draft_user
    correction_directory = source / "projection_corrections_draft" / draft_user
    object_path, map_path = labels_directory / "objects.json", labels_directory / "map.json"
    objects, map_objects = load_json(object_path), load_json(map_path)
    if not isinstance(objects, list) or not isinstance(map_objects, list):
        raise ValueError("draft object/map labels must be lists")
    if apply_projection_corrections:
        corrections, correction_files = _load_corrections(correction_directory, source_stems)
    else:
        # A raw-calibration comparison must not read either draft corrections
        # or locally published projection_calibrations from the base clip.
        corrections, correction_files = {}, []
    timestamps = [int(item["lidar_timestamp_us"]) for item in frame_items]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("source frame timestamps must be strictly increasing")
    frames, cameras, frame_mapping = [], [], []
    static_extrinsics = {}
    camera_by_id = {camera.id: camera for camera in base_clip.cameras}
    corrected_cameras = {camera_id: 0 for camera_id in camera_ids}
    identity = np.eye(4)
    for frame_index, (item, base_frame) in enumerate(zip(frame_items, base_clip.frames)):
        frame_id, stem = f"frame-{frame_index:03d}", item["target_frame_stem"]
        timestamp = (timestamps[frame_index] - timestamps[0]) / 1_000_000
        pose = quaternion_transform(item["ego_pose_local"]["rotation"], item["ego_pose_local"]["translation"])
        inverse_pose = np.linalg.inv(pose)
        if (base_frame.id != frame_id or not np.isclose(base_frame.timestamp, timestamp, atol=1e-8)
                or not np.allclose(base_frame.ego_to_world, pose, atol=1e-8)
                or not np.allclose(base_frame.lidar_to_ego, inverse_pose, atol=1e-8)):
            raise ValueError(f"installed frame {base_frame.id} does not map to source frame {stem}")
        if (set(base_map.get("world_from_lidar", {})) != {f.id for f in base_clip.frames}
                or not np.allclose(base_map["world_from_lidar"][frame_id], identity, atol=1e-8)):
            raise ValueError("base supplied map must use the source labels' local-world coordinate system")
        configs = configs_by_stem[stem]
        if not isinstance(configs, list) or len(configs) != len(CAMERAS):
            raise ValueError(f"{stem}: expected seven camera configurations")
        if set(base_frame.images) != set(camera_ids):
            raise ValueError(f"{frame_id}: installed image cameras do not match the source")
        transforms = {}
        for camera_index, (camera_id, camera_name, _) in enumerate(CAMERAS):
            image = source / f"camera_image_{camera_index}" / f"{stem}.jpg"
            reference = base_frame.images[camera_id]
            if not reference.startswith("/api/assets/"):
                raise ValueError(f"unsupported shared image reference: {reference}")
            stored_image = _stored_path(data_root / "assets", reference.removeprefix("/api/assets/"))
            if not image.is_file() or image.stat().st_size != stored_image.stat().st_size:
                raise ValueError(f"shared image does not match source camera/frame: {image}")
            config = configs[camera_index]
            external = np.asarray(config["camera_external"], dtype=float).reshape(4, 4)
            intrinsic = config["camera_internal"]
            intrinsics = [[intrinsic["fx"], 0, intrinsic["cx"]],
                          [0, intrinsic["fy"], intrinsic["cy"]], [0, 0, 1]]
            k, p = config.get("distortionK", [0, 0, 0]), config.get("distortionP", [0, 0])
            distortion = [k[0], k[1], p[0], p[1], k[2]]
            if (not np.allclose(camera_by_id[camera_id].intrinsics, intrinsics, atol=1e-8)
                    or not np.allclose(camera_by_id[camera_id].distortion, distortion, atol=1e-8)):
                raise ValueError(f"{stem}/{camera_id}: source intrinsics do not match the shared asset camera")
            if frame_index == 0:
                static_extrinsics[camera_id] = external @ pose
                cameras.append({
                    "id": camera_id, "name": camera_name,
                    "width": camera_by_id[camera_id].width, "height": camera_by_id[camera_id].height,
                    "intrinsics": intrinsics, "distortion": distortion,
                    "camera_from_ego": matrix_list(static_extrinsics[camera_id]),
                })
            elif not np.allclose(external @ pose, static_extrinsics[camera_id], atol=2e-5):
                raise ValueError(f"source camera {camera_id} is not rigidly attached to the ego poses")
            correction = corrections.get((f"camera_image_{camera_index}", stem))
            # Never fall back to projection_corrections (published results).
            corrected = external
            if correction is not None:
                corrected = external @ pose @ correction_transform(correction) @ inverse_pose
                corrected_cameras[camera_id] += 1
            transforms[camera_id] = matrix_list(corrected)
        frames.append({
            "id": frame_id, "timestamp": timestamp, "ego_to_world": matrix_list(pose),
            "lidar_to_ego": matrix_list(inverse_pose), "points": [],
            "pointcloud_file": point_file, "point_count": point_count, "images": dict(base_frame.images),
            "annotation_camera_from_lidar": transforms,
            "projection_camera_from_lidar": deepcopy(transforms),
        })
        frame_mapping.append({"frame_id": frame_id, "source_frame": stem, "images": dict(base_frame.images)})

    correction_count = len(corrections)
    fps = float(1_000_000 / np.median(np.diff(timestamps))) if len(timestamps) > 1 else base_clip.fps
    projection_description = (
        f"投影参数：{correction_directory}，应用 {correction_count} 份逐帧投影修正。"
        if apply_projection_corrections else
        "投影参数：仅使用原始 camera_config，未应用草稿修正或平台 Koide 覆盖。"
    )
    clip = Clip.model_validate({
        "id": clip_id, "name": name, "fps": fps, "cameras": cameras, "frames": frames,
        # Keep the supplied-map loading path: all 50 frames share one cloud.
        "source": "4d-clip-import",
        "description": (
            f"{draft_user} 草稿：{labels_directory}；{projection_description}"
            f"{len(frames)} 帧、{len(cameras)} 路相机。"
            "缺少草稿修正的相机使用原始 camera_config；点云和标注使用同一有效投影矩阵。"
            f"复用原场景全部 {point_count:,} 个点和图像；独立保存草稿标注与参数。"
            f"来源清单：data/import_reports/{clip_id}.json。"
        ),
    })
    vectors = [{
        "id": f"map-{item['obj_id']}", "label": str(item["obj_type"]).lower().replace("_", "-"),
        "points": [[point["x"], point["y"], point["z"]] for point in item["points"]],
        "closed": item.get("ann_type") == "polygon",
    } for item in map_objects]
    tracks = [{
        "id": f"object-{item['obj_id']}", "label": str(item["obj_type"]).lower().replace("_", "-"),
        "mode": "static", "keyframes": [{
            "frame_id": frames[0]["id"],
            "center": [item["psr"]["position"][axis] for axis in ("x", "y", "z")],
            "size": [item["psr"]["scale"][axis] for axis in ("x", "y", "z")],
            "yaw": item["psr"]["rotation"]["z"],
        }],
    } for item in objects]
    scene = SceneAnnotations.model_validate({"revision": 0, "coordinate_frame": "world",
                                             "vectors": vectors, "tracks": tracks}).validate_for_clip(clip)
    annotations = annotations_for_frames(clip, vectors, tracks)
    identity_poses = {frame.id: identity.tolist() for frame in clip.frames}
    metadata = deepcopy(base_map)
    metadata.pop("source_fingerprint", None)
    metadata.update(
        frame_count=len(frames), scene_revision=0, source_map="supplied",
        world_from_lidar=identity_poses, point_labels={},
        trajectory=[{"frame_id": frame["id"], "timestamp": frame["timestamp"],
                     "position": [row[3] for row in frame["ego_to_world"][:3]]}
                    for frame in frames],
    )
    inputs = [pose_files[0], object_path, map_path, *config_files, *correction_files]
    report = {
        "format": "omnilabel-4d-draft-import-v1", "clip_id": clip_id, "base_clip_id": base_clip_id,
        "draft_user": draft_user, "source_directory": str(source),
        "label_directory": str(labels_directory),
        "projection_directory": str(correction_directory) if apply_projection_corrections else None,
        "counts": {"frames": len(frames), "cameras": len(cameras), "objects": len(tracks),
                   "map_vectors": len(vectors), "projection_corrections": correction_count, "points": point_count},
        "corrections_by_camera": corrected_cameras, "frames": frame_mapping,
        "projection_convention": {
            "mode": "draft_corrections" if apply_projection_corrections else "raw_camera_config",
            "units": {"translation": "ego-local metres", "rotation": "degrees"},
            "rotation_order": "Rz(dyaw) @ Ry(dpitch) @ Rx(droll)",
            "effective_matrix": ("camera_external @ ego_to_world @ correction @ inverse(ego_to_world)"
                                 if apply_projection_corrections else "camera_external"),
            "missing_draft_correction": "original camera_config camera_external; no published fallback",
            "matrix_storage": "both projection_camera_from_lidar and annotation_camera_from_lidar in clip manifest",
        },
        "shared_assets": {"pointcloud_file": point_file, "scene_pointcloud_file": point_file,
                          "source_pcd": str(pcd_files[0]), "source_point_count": point_count,
                          "copied_points": 0, "copied_images": 0},
        "input_sha256": {str(path.relative_to(source)): sha256(path.read_bytes()).hexdigest() for path in inputs},
        "source_projection_corrections": [deepcopy(corrections[key]) for key in sorted(corrections)],
        "source_object_labels": objects, "source_map_labels": map_objects,
    }
    return {"clip": clip, "scene": scene, "annotations": annotations, "map": metadata,
            "poses": {"world_from_lidar": identity_poses}, "report": report}


def install_draft(source: Path, data_root: Path, **options) -> dict:
    """Publish an independent draft clip, refusing all existing output paths."""
    data_root = data_root.resolve()
    bundle = prepare_draft(source, data_root, **options)
    clip = bundle["clip"]
    _check_destination(data_root, clip.id)
    staging = Path(tempfile.mkdtemp(prefix=f".{clip.id}-", dir=data_root))
    outputs = _output_paths(data_root, clip.id)
    published, annotation_directory_created = [], False
    try:
        staged = _output_paths(staging, clip.id)
        atomic_json(staged["clip"], clip.model_dump())
        atomic_json(staged["scene"], bundle["scene"].model_dump())
        for frame_id, value in bundle["annotations"].items():
            atomic_json(staged["annotations"] / f"{frame_id}.json", value.model_dump())
        # Fingerprints use file metadata. Hard links retain that metadata during
        # publication; this temporary symlink exposes the already shared NPYs.
        (staging / "pointclouds").symlink_to(data_root / "pointclouds", target_is_directory=True)
        fingerprint_store = SimpleNamespace(root=staging)
        bundle["poses"]["clip_fingerprint"] = _clip_fingerprint(fingerprint_store, clip)
        bundle["map"]["source_fingerprint"] = source_fingerprint(fingerprint_store, clip)
        bundle["report"]["imported_at"] = datetime.now(timezone.utc).isoformat()
        for key in ("poses", "map", "report"):
            atomic_json(staged[key], bundle[key])
        for path in outputs.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        outputs["annotations"].mkdir()  # Exclusive reservation: never overwrite saved annotations.
        annotation_directory_created = True
        for path in sorted(staged["annotations"].glob("*.json")):
            destination = outputs["annotations"] / path.name
            os.link(path, destination)
            published.append(destination)
        # The clip is visible only after its scene, parameters and labels exist.
        for key in ("scene", "poses", "map", "report", "clip"):
            os.link(staged[key], outputs[key])
            published.append(outputs[key])
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        if annotation_directory_created:
            outputs["annotations"].rmdir()
        raise
    finally:
        shutil.rmtree(staging)
    return {"clip_id": clip.id, "name": clip.name, **bundle["report"]["counts"],
            "corrections_by_camera": bundle["report"]["corrections_by_camera"],
            "shared_pointcloud_file": clip.frames[0].pointcloud_file,
            "report": str(outputs["report"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="parent 4D clip directory containing camera_config/global_pose")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--base-clip-id", default=DEFAULT_BASE_CLIP_ID)
    parser.add_argument("--clip-id", default=DEFAULT_CLIP_ID)
    parser.add_argument("--draft-user", default="manager1")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--no-projection-corrections", action="store_true",
                        help="use raw camera_config extrinsics; ignore draft and published corrections")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing any data")
    args = parser.parse_args()
    options = {key: getattr(args, key) for key in ("base_clip_id", "clip_id", "draft_user", "name")}
    options["apply_projection_corrections"] = not args.no_projection_corrections
    if args.dry_run:
        bundle = prepare_draft(args.source, args.data_root, **options)
        result = {"dry_run": True, "clip_id": bundle["clip"].id, **bundle["report"]["counts"],
                  "corrections_by_camera": bundle["report"]["corrections_by_camera"],
                  "shared_assets": bundle["report"]["shared_assets"]}
    else:
        result = install_draft(args.source, args.data_root, **options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
