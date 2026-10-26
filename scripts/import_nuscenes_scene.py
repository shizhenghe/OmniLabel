#!/usr/bin/env python3
"""Import an exported nuScenes LiDAR scene as one lazy OmniLabel clip."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
def quaternion_matrix(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0:
        raise ValueError("quaternion cannot be zero")
    w, x, y, z = quaternion / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def transform(rotation, translation) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(rotation)
    vector = np.asarray(translation, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("translation must contain three finite values")
    matrix[:3, 3] = vector
    return matrix.tolist()


def object_record(path: Path) -> dict:
    value = np.load(path, allow_pickle=True)
    if value.shape != () or not isinstance(value.item(), dict):
        raise ValueError(f"{path}: expected one exported metadata object")
    return value.item()


def numeric_array(path: Path, columns: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if (value.ndim != 2 or value.shape[1] != columns
            or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()):
        raise ValueError(f"{path}: expected a finite numeric Nx{columns} array")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "tests" / "nuscenes_scene_0056")
    parser.add_argument("--clip-id", default="nuscenes-scene-0056")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    manifest_path = source / "export_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}")
    export = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_meta = sorted(export.get("frames", []), key=lambda item: item["timestamp"])
    if not frames_meta or len(frames_meta) != export.get("exported_keyframes"):
        raise ValueError("export manifest frame count is missing or inconsistent")

    sys.path.insert(0, str(ROOT / "backend"))
    from omnilabel.models import Annotations, Clip, Frame
    from omnilabel.storage import atomic_json, safe_id

    clip_id = safe_id(args.clip_id)
    clip_path = DATA / "clips" / f"{clip_id}.json"
    points_path = DATA / "pointclouds" / clip_id
    annotations_path = DATA / "annotations" / clip_id
    if clip_path.exists() or points_path.exists() or annotations_path.exists():
        raise SystemExit(f"Output for {clip_id} already exists; refusing to overwrite it")

    point_parent = points_path.parent
    annotation_parent = annotations_path.parent
    point_parent.mkdir(parents=True, exist_ok=True)
    annotation_parent.mkdir(parents=True, exist_ok=True)
    point_stage = Path(tempfile.mkdtemp(prefix=f".{clip_id}-", dir=point_parent))
    annotation_stage = Path(tempfile.mkdtemp(prefix=f".{clip_id}-", dir=annotation_parent))
    published_points = published_annotations = False
    total_points = total_boxes = total_labels = 0
    try:
        timestamps = [int(item["timestamp"]) for item in frames_meta]
        if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("export timestamps must be strictly increasing")
        intervals = sorted((b - a) / 1_000_000 for a, b in zip(timestamps, timestamps[1:]))
        fps = 1.0 / intervals[len(intervals) // 2] if intervals else 1.0
        first_timestamp = timestamps[0]
        semantic_names = {int(key): value for key, value in export["occ_class_names"].items()}
        frames = []
        reference_calibration = None

        for position, item in enumerate(frames_meta):
            index = int(item["export_index"])
            if index != position:
                raise ValueError("export frame indexes must be contiguous from zero")
            frame_id = f"frame-{index:03d}"
            source_points = source / "pc" / f"pc{index}.npy"
            points = numeric_array(source_points, 4)
            semantic = numeric_array(source / "pc_seman" / f"pc_seman_{index}.npy", 4)
            if len(points) != int(item["num_points"]) or semantic.shape != points.shape:
                raise ValueError(f"frame {index}: point counts disagree with export manifest")
            if not np.allclose(points[:, :3], semantic[:, :3], rtol=0, atol=1e-5):
                raise ValueError(f"frame {index}: semantic points do not align with source points")
            classes = semantic[:, 3]
            if not np.equal(classes, np.floor(classes)).all():
                raise ValueError(f"frame {index}: semantic classes must be integers")
            class_ids = classes.astype(np.int64)
            unknown_classes = set(np.unique(class_ids)) - set(semantic_names)
            if unknown_classes:
                raise ValueError(f"frame {index}: unknown semantic classes {sorted(unknown_classes)}")

            pose = object_record(source / "pose" / f"lidar_ego_pose{index}.npy")
            calibration = object_record(source / "calib" / f"lidar_calibrated_sensor{index}.npy")
            if int(pose["timestamp"]) != int(item["timestamp"]):
                raise ValueError(f"frame {index}: pose timestamp does not match manifest")
            calibration_value = (calibration["rotation"], calibration["translation"])
            if reference_calibration is None:
                reference_calibration = calibration_value
            elif not (np.allclose(calibration_value[0], reference_calibration[0])
                      and np.allclose(calibration_value[1], reference_calibration[1])):
                raise ValueError("LiDAR calibration changes within the scene")

            boxes = numeric_array(source / "bbox" / f"bbox{index}.npy", 7)
            tokens = np.load(source / "bbox" / f"boxes_token{index}.npy", allow_pickle=False)
            categories = np.load(source / "bbox" / f"object_category{index}.npy", allow_pickle=False)
            if len(boxes) != int(item["num_boxes"]) or len(tokens) != len(boxes) or len(categories) != len(boxes):
                raise ValueError(f"frame {index}: box metadata counts disagree")
            box_values = []
            for box, token, category in zip(boxes, tokens, categories):
                category_id = int(category)
                if category_id not in semantic_names:
                    raise ValueError(f"frame {index}: unsupported detection class {category_id}")
                track_id = str(token)
                box_values.append({
                    "id": f"box-{track_id}",
                    "track_id": track_id,
                    "label": semantic_names[category_id],
                    "center": [float(value) for value in box[:3]],
                    # Exported nuScenes boxes use width, length, height; OmniLabel
                    # uses local x-length, y-width, z-height.
                    "size": [float(box[4]), float(box[3]), float(box[5])],
                    "yaw": float(box[6]),
                })
            point_labels = {
                str(point_index): semantic_names[int(category)]
                for point_index, category in enumerate(class_ids) if int(category) != 0
            }

            destination_name = f"{frame_id}.npy"
            shutil.copy2(source_points, point_stage / destination_name)
            frame = {
                "id": frame_id,
                "timestamp": (int(item["timestamp"]) - first_timestamp) / 1_000_000,
                "ego_to_world": transform(pose["rotation"], pose["translation"]),
                "lidar_to_ego": transform(calibration["rotation"], calibration["translation"]),
                "pointcloud_file": f"{clip_id}/{destination_name}",
                "point_count": len(points),
                "images": {},
            }
            annotations = Annotations(boxes=box_values, point_labels=point_labels).validate_for_frame(
                Frame.model_validate(frame), []
            )
            atomic_json(annotation_stage / f"{frame_id}.json", annotations.model_dump())
            frames.append(frame)
            total_points += len(points)
            total_boxes += len(box_values)
            total_labels += len(point_labels)

        clip = Clip.model_validate({
            "id": clip_id,
            "name": f"nuScenes {export['scene_name']} · {len(frames)} frames",
            "description": export.get("scene_description", "Exported nuScenes LiDAR scene"),
            "fps": fps,
            "cameras": [],
            "frames": frames,
            "source": "nuscenes",
        })
        point_stage.rename(points_path)
        published_points = True
        annotation_stage.rename(annotations_path)
        published_annotations = True
        atomic_json(clip_path, clip.model_dump())
        print(f"Created {clip_path}")
        print(f"Frames: {len(frames)}; points: {total_points:,}; boxes: {total_boxes}; semantic labels: {total_labels:,}")
        print(f"Point clouds: {points_path}")
        print(f"Annotations: {annotations_path}")
    except Exception:
        if point_stage.exists():
            shutil.rmtree(point_stage)
        if annotation_stage.exists():
            shutil.rmtree(annotation_stage)
        if published_points:
            shutil.rmtree(points_path, ignore_errors=True)
        if published_annotations:
            shutil.rmtree(annotations_path, ignore_errors=True)
        clip_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
