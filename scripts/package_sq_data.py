#!/usr/bin/env python3
"""Convert tests/sq_data's nuScenes-style fixture into OmniLabel ZIP clips."""

from __future__ import annotations

import argparse
import json
import math
import struct
import zipfile
from pathlib import Path


CHANNELS = {
    "CAM_FRONT": ("front", "Front"),
    "CAM_FRONT_SHORT": ("front_short", "Front short"),
    "CAM_FRONT_WIDE": ("front_wide", "Front wide"),
    "CAM_BACK_LEFT": ("back_left", "Back left"),
    "CAM_LEFT_WIDE": ("left_wide", "Left wide"),
    "CAM_BACK_RIGHT": ("back_right", "Back right"),
    "CAM_RIGHT_WIDE": ("right_wide", "Right wide"),
}
LIDAR_CHANNEL = "LIDAR_TOP"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def quaternion_matrix(value: list[float]) -> list[list[float]]:
    """Return a 3x3 rotation matrix for a wxyz quaternion."""
    w, x, y, z = value
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("invalid zero or non-finite quaternion")
    w, x, y, z = (item / norm for item in (w, x, y, z))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform(rotation: list[float], translation: list[float]) -> list[list[float]]:
    matrix = quaternion_matrix(rotation)
    return [
        [*matrix[0], translation[0]],
        [*matrix[1], translation[1]],
        [*matrix[2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def inverse_transform(matrix: list[list[float]]) -> list[list[float]]:
    rotation = [row[:3] for row in matrix[:3]]
    transposed = [[rotation[column][row] for column in range(3)] for row in range(3)]
    translation = [matrix[row][3] for row in range(3)]
    inverse_translation = [
        -sum(transposed[row][column] * translation[column] for column in range(3))
        for row in range(3)
    ]
    return [
        [*transposed[0], inverse_translation[0]],
        [*transposed[1], inverse_translation[1]],
        [*transposed[2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def read_binary_xyzi_pcd(path: Path) -> tuple[int, bytes]:
    """Read the fixture's binary float32 XYZI PCD and return its raw records."""
    header: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: missing DATA line")
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: binary data started before DATA line") from exc
            if text and not text.startswith("#"):
                parts = text.split()
                header[parts[0].upper()] = parts[1:]
            if text.upper().startswith("DATA "):
                payload = stream.read()
                break

    expected = {
        "FIELDS": ["x", "y", "z", "intensity"],
        "SIZE": ["4", "4", "4", "4"],
        "TYPE": ["F", "F", "F", "F"],
        "COUNT": ["1", "1", "1", "1"],
        "DATA": ["binary"],
    }
    for key, value in expected.items():
        actual = header.get(key, [])
        if key == "TYPE":
            actual = [item.upper() for item in actual]
        if key == "DATA":
            actual = [item.lower() for item in actual]
        if actual != value:
            raise ValueError(f"{path}: expected {key} {' '.join(value)}, got {' '.join(actual)}")

    points = int(header.get("POINTS", ["-1"])[0])
    if points < 0 or len(payload) != points * 16:
        raise ValueError(f"{path}: payload size does not match POINTS")
    for record in struct.iter_unpack("<ffff", payload):
        if not all(math.isfinite(item) for item in record):
            raise ValueError(f"{path}: point cloud contains non-finite values")
    return points, payload


def build_dataset(source: Path):
    metadata = source / "v1.0-trainval"
    sensors = {item["token"]: item for item in load_json(metadata / "sensor.json")}
    calibrations = {item["token"]: item for item in load_json(metadata / "calibrated_sensor_refined.json")}
    poses = {item["token"]: item for item in load_json(metadata / "ego_pose_refined.json")}
    samples = sorted(load_json(metadata / "sample.json"), key=lambda item: item["timestamp"])
    sample_data = load_json(metadata / "sample_data.json")

    by_sample: dict[str, dict[str, dict]] = {}
    calibration_by_channel: dict[str, dict] = {}
    for item in sample_data:
        calibration = calibrations[item["calibrated_sensor_token"]]
        channel = sensors[calibration["sensor_token"]]["channel"]
        by_sample.setdefault(item["sample_token"], {})[channel] = item
        existing = calibration_by_channel.setdefault(channel, calibration)
        if existing != calibration:
            raise ValueError(f"multiple calibrations found for {channel}")

    required = {LIDAR_CHANNEL, *CHANNELS}
    if set(calibration_by_channel) != required:
        missing = sorted(required - set(calibration_by_channel))
        extra = sorted(set(calibration_by_channel) - required)
        raise ValueError(f"unexpected sensor set; missing={missing}, extra={extra}")

    cameras = []
    for channel, (camera_id, name) in CHANNELS.items():
        calibration = calibration_by_channel[channel]
        camera_from_ego = inverse_transform(transform(calibration["rotation"], calibration["translation"]))
        first = by_sample[samples[0]["token"]][channel]
        cameras.append({
            "id": camera_id,
            "name": name,
            "width": first["width"],
            "height": first["height"],
            "intrinsics": calibration["camera_intrinsic"],
            "camera_from_ego": camera_from_ego,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        })

    lidar_calibration = calibration_by_channel[LIDAR_CHANNEL]
    lidar_to_ego = transform(lidar_calibration["rotation"], lidar_calibration["translation"])
    first_timestamp = samples[0]["timestamp"]
    records = []
    for index, sample in enumerate(samples):
        sensor_data = by_sample.get(sample["token"], {})
        if set(sensor_data) != required:
            raise ValueError(f"sample {sample['token']} does not contain all eight sensors")
        lidar = sensor_data[LIDAR_CHANNEL]
        lidar_path = source / lidar["filename"]
        points, payload = read_binary_xyzi_pcd(lidar_path)
        pose = poses[lidar["ego_pose_token"]]
        if pose["timestamp"] != sample["timestamp"] or lidar["timestamp"] != sample["timestamp"]:
            raise ValueError(f"pose/LiDAR timestamp mismatch in sample {sample['token']}")

        images = {}
        image_paths = {}
        for channel, (camera_id, _) in CHANNELS.items():
            image = sensor_data[channel]
            if image["timestamp"] != sample["timestamp"]:
                raise ValueError(f"camera timestamp mismatch for {channel} in sample {sample['token']}")
            source_path = source / image["filename"]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            archive_path = f"images/{camera_id}/{source_path.name}"
            images[camera_id] = archive_path
            image_paths[archive_path] = source_path

        bin_name = f"lidar/{lidar_path.stem}.bin"
        frame = {
            "id": f"frame-{index:03d}",
            "timestamp": (sample["timestamp"] - first_timestamp) / 1_000_000.0,
            "ego_to_world": transform(pose["rotation"], pose["translation"]),
            "lidar_to_ego": lidar_to_ego,
            "points_path": bin_name,
            "images": images,
        }
        asset_size = len(payload) + sum(path.stat().st_size for path in image_paths.values())
        records.append({
            "frame": frame,
            "points": points,
            "point_payload": payload,
            "point_archive_path": bin_name,
            "images": image_paths,
            "asset_size": asset_size,
        })

    intervals = [records[i + 1]["frame"]["timestamp"] - records[i]["frame"]["timestamp"] for i in range(len(records) - 1)]
    intervals.sort()
    fps = 1.0 / intervals[len(intervals) // 2] if intervals else 1.0
    return cameras, records, fps


def chunks(records: list[dict], max_frames: int, max_asset_bytes: int = 0):
    current = []
    point_count = 0
    asset_bytes = 0
    for record in records:
        would_exceed = current and (
            len(current) >= max_frames
            or (max_asset_bytes > 0 and asset_bytes + record["asset_size"] > max_asset_bytes)
        )
        if would_exceed:
            yield current
            current = []
            point_count = 0
            asset_bytes = 0
        if max_asset_bytes > 0 and record["asset_size"] > max_asset_bytes:
            raise ValueError(f"single frame {record['frame']['id']} exceeds the requested asset-byte limit")
        current.append(record)
        point_count += record["points"]
        asset_bytes += record["asset_size"]
    if current:
        yield current


def write_archives(source: Path, output: Path, max_frames: int, force: bool, max_asset_bytes: int = 0) -> list[Path]:
    cameras, records, fps = build_dataset(source)
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(output)
        if not force:
            raise FileExistsError(f"{output} already exists; pass --force to replace it")
        existing = list(output.iterdir())
        unexpected = [path for path in existing if not (path.is_file() and path.name.startswith("sq-data-part-") and path.suffix == ".zip")]
        if unexpected:
            raise ValueError(f"refusing to clean output directory containing unexpected files: {unexpected[0]}")
        for path in existing:
            path.unlink()
    else:
        output.mkdir(parents=True)

    # Partition only when the operator explicitly requests it. All LiDAR
    # records are preserved; point count never changes scene membership.
    groups = list(chunks(records, max_frames=max_frames, max_asset_bytes=max_asset_bytes))
    results = []
    for part, group in enumerate(groups, start=1):
        clip_id = f"sq-data-part-{part:02d}"
        manifest = {
            "schema_version": "1.0",
            "clip": {
                "id": clip_id,
                "name": f"SQ test data · part {part:02d}/{len(groups):02d}",
                "description": "Converted from tests/sq_data using refined calibration and ego poses.",
                "fps": fps,
                "source": "sq_data",
                "cameras": cameras,
                "frames": [record["frame"] for record in group],
            },
            "annotations": {},
        }
        destination = output / f"{clip_id}.zip"
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for record in group:
                archive.writestr(record["point_archive_path"], record["point_payload"])
                for archive_path, source_path in record["images"].items():
                    archive.write(source_path, archive_path)
        results.append(destination)
        print(
            f"wrote {destination} "
            f"({len(group)} frames, {sum(record['points'] for record in group):,} points, "
            f"{destination.stat().st_size / 1024 / 1024:.1f} MiB)"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("tests/sq_data"))
    parser.add_argument("--output", type=Path, default=Path("tests/sq_data/omnilabel_packages"))
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--max-asset-bytes", type=int, default=0, help="Optional archive source-byte budget; 0 keeps all points without a byte quota")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 2000:
        parser.error("--max-frames must be between 1 and 2000")
    if args.max_asset_bytes < 0:
        parser.error("--max-asset-bytes must be nonnegative")
    write_archives(args.source.resolve(), args.output.resolve(), args.max_frames, args.force, args.max_asset_bytes)


if __name__ == "__main__":
    main()
