#!/usr/bin/env python3
"""Package one standard nuScenes scene export for OmniLabel."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAMERA_CHANNELS = {
    "CAM_FRONT": ("front", "Front"),
    "CAM_FRONT_LEFT": ("front_left", "Front left"),
    "CAM_FRONT_RIGHT": ("front_right", "Front right"),
    "CAM_BACK": ("back", "Back"),
    "CAM_BACK_LEFT": ("back_left", "Back left"),
    "CAM_BACK_RIGHT": ("back_right", "Back right"),
}
LIDAR_CHANNEL = "LIDAR_TOP"

# Official nuScenes lidarseg index -> 16-class learning taxonomy used by the
# existing nuScenes fixture and understood by OmniLabel's dynamic filtering.
# None is the official ignore class. A single-scene export commonly filters
# category.json, so this mapping cannot be reconstructed from that table alone.
LIDARSEG_CLASSES = {
    0: None,
    1: None,
    2: "pedestrian",
    3: "pedestrian",
    4: "pedestrian",
    5: "pedestrian",
    6: "pedestrian",
    7: "pedestrian",
    8: "pedestrian",
    9: "barrier",
    10: None,
    11: None,
    12: "traffic_cone",
    13: None,
    14: "bicycle",
    15: "bus",
    16: "bus",
    17: "car",
    18: "construction_vehicle",
    19: "car",
    20: "car",
    21: "motorcycle",
    22: "trailer",
    23: "truck",
    24: "driveable_surface",
    25: "other_flat",
    26: "sidewalk",
    27: "terrain",
    28: "manmade",
    29: None,
    30: "vegetation",
    31: None,
}


def detection_label(category: str) -> str:
    exact = {
        "movable_object.barrier": "barrier",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.bicycle": "bicycle",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    if category.startswith("human.pedestrian."):
        return "pedestrian"
    if category.startswith("vehicle.bus."):
        return "bus"
    if category.startswith("vehicle.emergency."):
        return "car"
    return exact.get(category, category)


def load_table(metadata: Path, name: str) -> list[dict]:
    path = metadata / f"{name}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing nuScenes metadata table: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}: expected a JSON array of objects")
    return value


def index_by(items: list[dict], field: str, table: str) -> dict[str, dict]:
    result = {}
    for item in items:
        value = item.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{table}: missing or duplicate {field}")
        result[value] = item
    return result


def quaternion_matrix(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(quaternion))
    if norm == 0:
        raise ValueError("quaternion cannot be zero")
    w, x, y, z = quaternion / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform(rotation, translation) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(rotation)
    vector = np.asarray(translation, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("translation must contain three finite values")
    matrix[:3, 3] = vector
    return matrix


def inverse_transform(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def read_lidar(path: Path, axis_rotation: np.ndarray) -> np.ndarray:
    """Read nuScenes XYZI+ring data and normalize axes to ego orientation."""
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.fromfile(path, dtype="<f4")
    if len(raw) % 5:
        raise ValueError(f"{path}: expected float32 XYZI+ring records")
    raw = raw.reshape(-1, 5)
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: point cloud contains non-finite values")
    points = np.empty((len(raw), 4), dtype="<f4")
    # nuScenes raw LiDAR axes are rotated relative to ego axes. Rotating XYZ
    # while retaining the sensor origin makes OmniLabel's local axes x-forward,
    # y-left, z-up; the remaining lidar_to_ego transform is translation only.
    points[:, :3] = raw[:, :3] @ axis_rotation.T
    points[:, 3] = raw[:, 3]
    return points


def npy_bytes(points: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, points, allow_pickle=False)
    return output.getvalue()


def build_dataset(source: Path, clip_id: str | None = None) -> tuple[dict, list[dict]]:
    metadata = source / "v1.0-trainval"
    scenes = load_table(metadata, "scene")
    if len(scenes) != 1:
        raise ValueError(f"expected a single scene, found {len(scenes)}")
    scene = scenes[0]
    scene_name = scene.get("name")
    if not isinstance(scene_name, str) or not scene_name:
        raise ValueError("scene metadata is missing its name")
    output_id = clip_id or f"nuscenes-{scene_name}"

    sensors = index_by(load_table(metadata, "sensor"), "token", "sensor")
    calibrations = index_by(load_table(metadata, "calibrated_sensor"), "token", "calibrated_sensor")
    poses = index_by(load_table(metadata, "ego_pose"), "token", "ego_pose")
    categories = index_by(load_table(metadata, "category"), "token", "category")
    instances = index_by(load_table(metadata, "instance"), "token", "instance")
    samples = sorted(load_table(metadata, "sample"), key=lambda item: item["timestamp"])
    if not samples or len(samples) != scene.get("nbr_samples"):
        raise ValueError("scene sample count is missing or inconsistent")
    if any(item.get("scene_token") != scene.get("token") for item in samples):
        raise ValueError("sample table contains data from another scene")
    timestamps = [int(item["timestamp"]) for item in samples]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("sample timestamps must be strictly increasing")

    required_channels = {LIDAR_CHANNEL, *CAMERA_CHANNELS}
    keyframes_by_sample: dict[str, dict[str, dict]] = {}
    calibrations_by_channel: dict[str, dict] = {}
    for item in load_table(metadata, "sample_data"):
        if not item.get("is_key_frame"):
            continue
        calibration = calibrations.get(item.get("calibrated_sensor_token"))
        if calibration is None:
            raise ValueError("sample_data references an unknown calibration")
        sensor = sensors.get(calibration.get("sensor_token"))
        if sensor is None:
            raise ValueError("calibration references an unknown sensor")
        channel = sensor.get("channel")
        if channel not in required_channels:
            continue
        sample_data = keyframes_by_sample.setdefault(item.get("sample_token"), {})
        if channel in sample_data:
            raise ValueError(f"duplicate keyframe sensor {channel} in one sample")
        sample_data[channel] = item
        existing = calibrations_by_channel.setdefault(channel, calibration)
        if existing != calibration:
            raise ValueError(f"calibration changes within the scene for {channel}")

    for sample in samples:
        channels = set(keyframes_by_sample.get(sample["token"], {}))
        if channels != required_channels:
            raise ValueError(
                f"sample {sample['token']} has an incomplete sensor set; "
                f"missing={sorted(required_channels - channels)}"
            )

    cameras = []
    camera_sizes = {}
    for channel, (camera_id, name) in CAMERA_CHANNELS.items():
        calibration = calibrations_by_channel[channel]
        sensor_to_ego = transform(calibration["rotation"], calibration["translation"])
        first_image = keyframes_by_sample[samples[0]["token"]][channel]
        width, height = int(first_image["width"]), int(first_image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image dimensions for {channel}")
        intrinsic = calibration.get("camera_intrinsic")
        if np.asarray(intrinsic).shape != (3, 3):
            raise ValueError(f"missing camera intrinsics for {channel}")
        cameras.append({
            "id": camera_id,
            "name": name,
            "width": width,
            "height": height,
            "intrinsics": intrinsic,
            "camera_from_ego": matrix_list(inverse_transform(sensor_to_ego)),
            "distortion": [0.0] * 5,
        })
        camera_sizes[channel] = (width, height)

    lidar_calibration = calibrations_by_channel[LIDAR_CHANNEL]
    lidar_sensor_to_ego = transform(lidar_calibration["rotation"], lidar_calibration["translation"])
    lidar_axis_rotation = lidar_sensor_to_ego[:3, :3]
    normalized_lidar_to_ego = np.eye(4, dtype=np.float64)
    normalized_lidar_to_ego[:3, 3] = lidar_sensor_to_ego[:3, 3]

    annotations_by_sample: dict[str, list[dict]] = {}
    for annotation in load_table(metadata, "sample_annotation"):
        annotations_by_sample.setdefault(annotation.get("sample_token"), []).append(annotation)

    lidarseg_by_sample_data = {}
    for item in load_table(metadata, "lidarseg"):
        token = item.get("sample_data_token")
        if not isinstance(token, str) or token in lidarseg_by_sample_data:
            raise ValueError("lidarseg has a missing or duplicate sample_data_token")
        lidarseg_by_sample_data[token] = item

    records = []
    frames = []
    annotations = {}
    first_timestamp = timestamps[0]
    for index, sample in enumerate(samples):
        frame_id = f"frame-{index:03d}"
        sensor_data = keyframes_by_sample[sample["token"]]
        lidar = sensor_data[LIDAR_CHANNEL]
        if int(lidar["timestamp"]) != int(sample["timestamp"]):
            raise ValueError(f"LiDAR timestamp differs from sample {sample['token']}")
        pose = poses.get(lidar.get("ego_pose_token"))
        if pose is None or int(pose["timestamp"]) != int(lidar["timestamp"]):
            raise ValueError(f"missing timestamp-aligned LiDAR ego pose for {sample['token']}")
        ego_to_world = transform(pose["rotation"], pose["translation"])
        world_to_ego = inverse_transform(ego_to_world)

        lidar_source = source / lidar["filename"]
        points = read_lidar(lidar_source, lidar_axis_rotation)
        lidar_archive_path = f"lidar/{frame_id}.npy"

        semantic_record = lidarseg_by_sample_data.get(lidar["token"])
        if semantic_record is None:
            raise ValueError(f"missing lidarseg labels for {sample['token']}")
        semantic_path = source / semantic_record["filename"]
        semantic = np.fromfile(semantic_path, dtype=np.uint8)
        if len(semantic) != len(points):
            raise ValueError(
                f"{semantic_path}: {len(semantic)} labels do not match {len(points)} LiDAR points"
            )
        unknown = sorted(set(int(value) for value in np.unique(semantic)) - set(LIDARSEG_CLASSES))
        if unknown:
            raise ValueError(f"{semantic_path}: unknown lidarseg classes {unknown}")
        point_labels = {
            str(point_index): LIDARSEG_CLASSES[int(class_index)]
            for point_index, class_index in enumerate(semantic)
            if LIDARSEG_CLASSES[int(class_index)] is not None
        }

        frame_images = {}
        image_assets = {}
        for channel, (camera_id, _) in CAMERA_CHANNELS.items():
            image = sensor_data[channel]
            if (int(image["width"]), int(image["height"])) != camera_sizes[channel]:
                raise ValueError(f"image dimensions change within the scene for {channel}")
            source_path = source / image["filename"]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            archive_path = f"images/{camera_id}/{frame_id}{source_path.suffix.lower()}"
            frame_images[camera_id] = archive_path
            image_assets[archive_path] = source_path

        box_values = []
        for annotation in annotations_by_sample.get(sample["token"], []):
            instance = instances.get(annotation.get("instance_token"))
            if instance is None:
                raise ValueError("sample_annotation references an unknown instance")
            category = categories.get(instance.get("category_token"))
            if category is None:
                raise ValueError("instance references an unknown category")
            center_world = np.asarray([*annotation["translation"], 1.0], dtype=np.float64)
            center_ego = (world_to_ego @ center_world)[:3]
            center_lidar = center_ego - lidar_sensor_to_ego[:3, 3]
            box_world_rotation = quaternion_matrix(annotation["rotation"])
            heading_lidar = world_to_ego[:3, :3] @ box_world_rotation[:, 0]
            size = np.asarray(annotation["size"], dtype=np.float64)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError("sample_annotation contains an invalid box size")
            box_values.append({
                "id": f"box-{annotation['token']}",
                "track_id": annotation["instance_token"],
                "label": detection_label(category["name"]),
                "center": [float(value) for value in center_lidar],
                # nuScenes stores width, length, height. OmniLabel stores the
                # local x length, local y width and z height.
                "size": [float(size[1]), float(size[0]), float(size[2])],
                "yaw": math.atan2(float(heading_lidar[1]), float(heading_lidar[0])),
            })

        frame = {
            "id": frame_id,
            "timestamp": (int(sample["timestamp"]) - first_timestamp) / 1_000_000.0,
            "ego_to_world": matrix_list(ego_to_world),
            "lidar_to_ego": matrix_list(normalized_lidar_to_ego),
            "points_path": lidar_archive_path,
            "point_count": len(points),
            "images": frame_images,
        }
        frames.append(frame)
        annotations[frame_id] = {
            "revision": 0,
            "boxes": box_values,
            "vectors": [],
            "image_polygons": [],
            "point_labels": point_labels,
        }
        records.append({
            "point_archive_path": lidar_archive_path,
            "point_payload": npy_bytes(points),
            "images": image_assets,
            "point_count": len(points),
            "box_count": len(box_values),
            "label_count": len(point_labels),
        })

    intervals = sorted(
        (right - left) / 1_000_000.0 for left, right in zip(timestamps, timestamps[1:])
    )
    fps = 1.0 / intervals[len(intervals) // 2] if intervals else 1.0
    manifest = {
        "schema_version": "1.0",
        "clip": {
            "id": output_id,
            "name": f"nuScenes {scene_name} · {len(frames)} keyframes",
            "description": (
                f"{scene.get('description', '').strip()} Converted from a standard nuScenes "
                "single-scene export; includes six cameras, LiDAR, 3D boxes and lidarseg labels."
            ).strip(),
            "fps": fps,
            "source": "nuscenes",
            "cameras": cameras,
            "frames": frames,
        },
        "annotations": annotations,
    }
    return manifest, records


def validate_manifest(manifest: dict) -> None:
    sys.path.insert(0, str(ROOT / "backend"))
    from omnilabel.models import Annotations, Clip

    # points_path is the portable import field. Model validation normally runs
    # after the importer has staged it as an internal pointcloud_file.
    clip_value = copy.deepcopy(manifest["clip"])
    for frame in clip_value["frames"]:
        point_path = frame.pop("points_path")
        frame["pointcloud_file"] = f"validation/{Path(point_path).name}"
    clip = Clip.model_validate(clip_value)
    frames = {frame.id: frame for frame in clip.frames}
    for frame_id, value in manifest["annotations"].items():
        Annotations.model_validate(value).validate_for_frame(frames[frame_id], clip.cameras)


def write_archive(source: Path, output: Path, clip_id: str | None, force: bool) -> Path:
    manifest, records = build_dataset(source, clip_id)
    validate_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".zip", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True
        ) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            )
            for record in records:
                archive.writestr(record["point_archive_path"], record["point_payload"])
                for archive_path, source_path in record["images"].items():
                    archive.write(source_path, archive_path)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Created {output}")
    print(
        f"Frames: {len(records)}; points: {sum(item['point_count'] for item in records):,}; "
        f"boxes: {sum(item['box_count'] for item in records):,}; "
        f"semantic labels: {sum(item['label_count'] for item in records):,}"
    )
    print(f"Archive size: {output.stat().st_size / 1024 / 1024:.1f} MiB")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "tests" / "nuscenes_scene-0058"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "nuscenes_scene-0058" / "omnilabel_packages" / "nuscenes-scene-0058.zip",
    )
    parser.add_argument("--clip-id", help="Override the default nuscenes-<scene-name> ID")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_archive(args.source.resolve(), args.output.resolve(), args.clip_id, args.force)


if __name__ == "__main__":
    main()
