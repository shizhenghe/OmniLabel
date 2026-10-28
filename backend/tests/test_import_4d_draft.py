"""Small supplied-map fixtures for independent draft imports; no API/server needed."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from import_4d_draft import (  # noqa: E402
    CAMERAS, DEFAULT_BASE_CLIP_ID, DEFAULT_CLIP_ID, DEFAULT_NAME,
    correction_transform, install_draft, matrix_list, prepare_draft, quaternion_transform,
)
from omnilabel.models import Clip  # noqa: E402
from omnilabel.scene import FUSION_VERSION, _clip_fingerprint, source_fingerprint  # noqa: E402


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def snapshot(root: Path):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def source_and_store(tmp_path):
    source, data = tmp_path / "source", tmp_path / "data"
    point_file = f"{DEFAULT_BASE_CLIP_ID}/map.npy"
    point_dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f4")])
    records = np.zeros(3, dtype=point_dtype)
    records[0], records[1], records[2] = (1, 2, 3, 10), (7, 8, 9, 200), (-5, 6, 8, 255)
    pcd = source / "lidar_point_cloud" / "map.pcd"
    pcd.parent.mkdir(parents=True)
    pcd.write_bytes(b"FIELDS x y z intensity\nSIZE 8 8 8 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
                    b"WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA binary\n" + records.tobytes())
    points = np.column_stack([records[axis] for axis in ("x", "y", "z", "intensity")]).astype("<f4")
    for directory in ("pointclouds", "scene_pointclouds"):
        path = data / directory / point_file
        path.parent.mkdir(parents=True)
        np.save(path, points)
    frames, frame_items = [], []
    transforms = []
    identity = np.eye(4).tolist()
    camera_intrinsics = [[800, 0, 960], [0, 810, 768], [0, 0, 1]]
    for index in range(2):
        yaw = np.deg2rad(20 + index * 8)
        pose_data = {"rotation": [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                     "translation": [10 + index * 2, 3, 0.2]}
        pose = quaternion_transform(pose_data["rotation"], pose_data["translation"])
        frame_id, stem = f"frame-{index:03d}", f"source_{index + 1:03d}"
        frame_items.append({"target_frame_stem": stem, "ego_pose_local": pose_data,
                            "lidar_timestamp_us": 1_000_000 + index * 500_000})
        external = matrix_list(np.linalg.inv(pose))
        transforms.append(external)
        configs, images = [], {}
        for camera_index, (camera_id, _, _) in enumerate(CAMERAS):
            configs.append({"camera_internal": {"fx": 800, "fy": 810, "cx": 960, "cy": 768},
                            "camera_external": np.asarray(external).flatten().tolist(),
                            "distortionK": [0, 0, 0], "distortionP": [0, 0]})
            image_bytes = f"fixture image {camera_id} {stem}".encode()
            image = source / f"camera_image_{camera_index}" / f"{stem}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(image_bytes)
            installed = data / "assets" / DEFAULT_BASE_CLIP_ID / camera_id / f"{frame_id}.jpg"
            installed.parent.mkdir(parents=True, exist_ok=True)
            installed.write_bytes(image_bytes)
            images[camera_id] = f"/api/assets/{DEFAULT_BASE_CLIP_ID}/{camera_id}/{frame_id}.jpg"
        write_json(source / "camera_config" / f"{stem}.json", configs)
        wrong_published = np.eye(4)
        wrong_published[0, 3] = 999
        frames.append({"id": frame_id, "timestamp": index * .5, "ego_to_world": matrix_list(pose),
                       "lidar_to_ego": external, "points": [], "pointcloud_file": point_file,
                       "point_count": 3, "images": images,
                       "annotation_camera_from_lidar": {camera[0]: wrong_published.tolist() for camera in CAMERAS},
                       "projection_camera_from_lidar": {camera[0]: wrong_published.tolist() for camera in CAMERAS}})
    write_json(source / "global_pose" / "clip.json", {
        "num_frames": 2, "camera_channels": [camera[2] for camera in CAMERAS], "frame_items": frame_items,
    })
    write_json(data / "clips" / f"{DEFAULT_BASE_CLIP_ID}.json", {
        "id": DEFAULT_BASE_CLIP_ID, "name": "Existing edited source clip", "fps": 2,
        "source": "4d-clip-import", "frames": frames,
        "cameras": [{"id": camera_id, "name": name, "width": 1920, "height": 1536,
                     "intrinsics": camera_intrinsics, "camera_from_ego": identity,
                     "distortion": [0, 0, 0, 0, 0]} for camera_id, name, _ in CAMERAS],
    })
    write_json(data / "scene_maps" / f"{DEFAULT_BASE_CLIP_ID}.json", {
        "method": "supplied reconstructed local-world map", "fusion_version": FUSION_VERSION,
        "pointcloud_file": point_file, "point_count": 3, "input_point_count": 3,
        "frame_count": 2, "world_from_lidar": {frame["id"]: identity for frame in frames},
        "point_labels": {"2": "an existing user point label"}, "bounds": {"min": [-5, 2, 3], "max": [7, 8, 9]},
    })
    draft_object = {"obj_id": "9", "obj_type": "Traffic_sign", "obj_attr": "visible",
                    "psr": {"position": {"x": 1, "y": 2, "z": 3},
                            "scale": {"x": 2, "y": 3, "z": 1},
                            "rotation": {"x": 0, "y": 0, "z": .2}}}
    draft_vector = {"obj_id": "3", "obj_type": "Divider", "ann_type": "polyline", "obj_attr": "white",
                    "points": [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 5, "z": 6}]}
    write_json(source / "label_draft" / "manager1" / "objects.json", [draft_object])
    write_json(source / "label_draft" / "manager1" / "map.json", [draft_vector])
    # Intentionally incompatible published geometry/parameters must never be used.
    write_json(source / "label" / "objects.json", [{"unrelated": "published"}])
    write_json(source / "label" / "map.json", [{"unrelated": "published"}])
    correction = {"camera": "camera_image_0", "frame": "source_001", "dx": 1.5, "dy": -.2,
                  "dz": .1, "dyaw": -2, "dpitch": .2, "droll": -.1}
    write_json(source / "projection_corrections_draft" / "manager1" / "camera_image_0" / "source_001.json", correction)
    write_json(source / "projection_corrections" / "camera_image_1" / "source_001.json",
               {**correction, "camera": "camera_image_1", "dx": 999})
    write_json(data / "projection_calibrations" / DEFAULT_BASE_CLIP_ID / "frame-000" / "front.json",
               {"matrix": "intentionally invalid existing calibration"})
    return source, data, correction, transforms


def test_draft_geometry_and_effective_matrices_ignore_published_results(source_and_store):
    source, data, correction, transforms = source_and_store
    before = snapshot(data)
    bundle = prepare_draft(source, data)
    assert snapshot(data) == before  # A dry build writes no files or directories.
    clip = bundle["clip"]
    assert clip.id == DEFAULT_CLIP_ID and clip.name == DEFAULT_NAME and clip.source == "4d-clip-import"
    assert all(frame.pointcloud_file == clip.frames[0].pointcloud_file for frame in clip.frames)
    assert bundle["scene"].tracks[0].keyframes[0].center == (1, 2, 3)
    assert bundle["scene"].vectors[0].points[-1] == (4, 5, 6)
    pose = np.asarray(clip.frames[0].ego_to_world)
    expected = np.asarray(transforms[0]) @ pose @ correction_transform(correction) @ np.linalg.inv(pose)
    np.testing.assert_allclose(clip.frames[0].projection_camera_from_lidar["front"], expected, atol=1e-10)
    # Published camera_image_1 has a correction, but this draft does not.
    np.testing.assert_allclose(clip.frames[0].projection_camera_from_lidar["front_short"], transforms[0], atol=1e-10)
    np.testing.assert_allclose(clip.frames[1].projection_camera_from_lidar["front"], transforms[1], atol=1e-10)
    for frame in clip.frames:
        assert frame.projection_camera_from_lidar == frame.annotation_camera_from_lidar
        assert len(bundle["annotations"][frame.id].boxes) == len(bundle["annotations"][frame.id].vectors) == 1
    assert bundle["map"]["point_labels"] == {}
    assert bundle["report"]["source_projection_corrections"] == [correction]
    assert bundle["report"]["source_object_labels"][0]["obj_attr"] == "visible"


def test_install_preserves_existing_assets_and_annotations_and_refuses_overwrite(source_and_store):
    source, data, _, _ = source_and_store
    before = snapshot(data)
    result = install_draft(source, data)
    assert result["projection_corrections"] == 1 and result["points"] == 3
    after = snapshot(data)
    assert {path: after[path] for path in before} == before
    new_files = set(after) - set(before)
    assert len(new_files) == 7  # Clip + scene + map + poses + report + two per-frame annotation files.
    assert all(not path.startswith(("assets/", "pointclouds/", "scene_pointclouds/")) for path in new_files)
    clip = Clip.model_validate_json((data / "clips" / f"{DEFAULT_CLIP_ID}.json").read_text())
    store = SimpleNamespace(root=data)
    poses = json.loads((data / "scene_poses" / f"{DEFAULT_CLIP_ID}.json").read_text())
    metadata = json.loads((data / "scene_maps" / f"{DEFAULT_CLIP_ID}.json").read_text())
    assert poses["clip_fingerprint"] == _clip_fingerprint(store, clip)
    assert metadata["source_fingerprint"] == source_fingerprint(store, clip)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        install_draft(source, data)
    assert snapshot(data) == after


def test_raw_comparison_ignores_all_projection_corrections(source_and_store):
    source, data, _, transforms = source_and_store
    bundle = prepare_draft(
        source, data, clip_id="raw-draft-comparison",
        apply_projection_corrections=False,
    )
    assert bundle["report"]["counts"]["projection_corrections"] == 0
    assert bundle["report"]["projection_directory"] is None
    assert bundle["report"]["projection_convention"]["mode"] == "raw_camera_config"
    assert bundle["report"]["source_projection_corrections"] == []
    for index, frame in enumerate(bundle["clip"].frames):
        expected = np.asarray(transforms[index])
        for camera_id, _, _ in CAMERAS:
            np.testing.assert_allclose(frame.projection_camera_from_lidar[camera_id], expected)
            np.testing.assert_allclose(frame.annotation_camera_from_lidar[camera_id], expected)


def test_bad_draft_identity_or_sparse_asset_is_rejected_without_publication(source_and_store):
    source, data, correction, _ = source_and_store
    before = snapshot(data)
    path = source / "projection_corrections_draft" / "manager1" / "camera_image_0" / "source_001.json"
    write_json(path, {**correction, "frame": "wrong-frame"})
    with pytest.raises(ValueError, match="identity"):
        install_draft(source, data)
    assert snapshot(data) == before
    write_json(path, correction)
    metadata_path = data / "scene_maps" / f"{DEFAULT_BASE_CLIP_ID}.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["point_count"] = 2
    write_json(metadata_path, metadata)
    before = snapshot(data)
    with pytest.raises(ValueError, match="count or shared"):
        install_draft(source, data)
    assert snapshot(data) == before
