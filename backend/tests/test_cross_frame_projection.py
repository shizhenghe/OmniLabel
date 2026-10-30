"""World-map lanes retain one geometry while active-frame projection changes."""
from copy import deepcopy
import json

import numpy as np
import pytest

from omnilabel.calibration import CalibrationUpdate, calibration_document, save_calibration
from omnilabel.geometry import project_points
from omnilabel.models import Clip
from omnilabel.projection_calibration import _base_fingerprint, _base_matrix, document_for
from omnilabel.scene import (
    FUSION_VERSION, _clip_fingerprint, frame_annotations, get_scene, get_scene_map,
    invalidate_scene_map, registered_poses,
)
from omnilabel.storage import Store, atomic_json


def pose(x=0., y=0., yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 0], [0, 0, 0, 1.]])


@pytest.fixture
def supplied(tmp_path):
    store = Store(tmp_path)
    camera = np.array([[0, -1, 0, .1], [0, 0, -1, 1.5], [1, 0, 0, -.2], [0, 0, 0, 1.]])
    frames = []
    for index, ego in enumerate((pose(), pose(2., .3, .08))):
        primary_to_ego = np.linalg.inv(ego)
        imported_correction = pose(.02 * index, -.01 * index, .003 * index)
        annotation = imported_correction @ camera @ primary_to_ego
        # Deliberately preserve a distinct imported physical projection as well
        # as the lower-priority annotation fallback.
        projection = pose(.03, 0, .002) @ annotation
        frames.append({"id": f"f{index}", "timestamp": index * .1,
                       "ego_to_world": ego.tolist(), "lidar_to_ego": primary_to_ego.tolist(),
                       "pointcloud_file": "supplied/map.npy", "point_count": 2, "images": {},
                       "annotation_camera_from_lidar": {"front": annotation.tolist()},
                       "projection_camera_from_lidar": {"front": projection.tolist()}})
    clip = Clip.model_validate({"id": "supplied", "name": "Supplied world map", "fps": 10,
        "source": "4d-clip-import", "frames": frames,
        "cameras": [{"id": "front", "name": "Front", "width": 640, "height": 360,
                     "intrinsics": [[300, 0, 320], [0, 300, 180], [0, 0, 1]],
                     "camera_from_ego": camera.tolist()}]})
    store.add_clip(clip, {})
    cloud = tmp_path / "pointclouds" / "supplied" / "map.npy"
    cloud.parent.mkdir(parents=True)
    points = [[10., -1., 0., .4], [22., -1., 0., .8]]
    np.save(cloud, np.asarray(points, dtype=np.float64))
    scene_cloud = tmp_path / "scene_pointclouds" / "supplied" / "map.npy"
    scene_cloud.parent.mkdir(parents=True)
    np.save(scene_cloud, np.asarray(points, dtype=np.float64))
    identity_poses = {frame.id: np.eye(4).tolist() for frame in clip.frames}
    atomic_json(tmp_path / "scene_poses" / "supplied.json",
                {"clip_fingerprint": _clip_fingerprint(store, clip), "world_from_lidar": identity_poses})
    atomic_json(tmp_path / "scene_maps" / "supplied.json",
                {"source_map": "supplied", "fusion_version": FUSION_VERSION,
                 "source_fingerprint": "legacy-token", "world_from_lidar": identity_poses,
                 "point_count": 2, "pointcloud_file": "supplied/map.npy"})
    atomic_json(tmp_path / "scenes" / "supplied.json",
                {"revision": 3, "coordinate_frame": "world", "tracks": [],
                 "vectors": [{"id": "lane", "label": "lane", "closed": False,
                              "points": [point[:3] for point in points]}]})
    return store, cloud


def update_for(store):
    document = calibration_document(store, store.source_clip("supplied"))
    return deepcopy({key: document[key] for key in ("revision", "cameras", "lidar_to_ego")})


@pytest.mark.parametrize("change", ["camera", "lidar", "both"])
def test_mount_edit_transports_imported_overrides_and_world_lane_pixels(supplied, change):
    store, cloud = supplied
    original = store.source_clip("supplied")
    before_files = {path: path.read_bytes() for path in [cloud,
        store.root / "scene_pointclouds" / "supplied" / "map.npy",
        store.root / "scenes" / "supplied.json",
        *sorted((store.root / "annotations" / "supplied").glob("*.json"))]}
    update = update_for(store)
    if change in {"camera", "both"}:
        update["cameras"]["front"] = (pose(.12, -.03, .015) @ np.asarray(update["cameras"]["front"])).tolist()
    if change in {"lidar", "both"}:
        for index, frame in enumerate(original.frames):
            update["lidar_to_ego"][frame.id] = (pose(.2 + index * .1, .15, .01 * (index + 1))
                                                    @ np.asarray(frame.lidar_to_ego)).tolist()
    result = save_calibration(store, original.id, CalibrationUpdate.model_validate(update))
    assert not result["map_rebuild_required"]
    assert result["lidar_changed"] == (change != "camera")
    changed = store.clip(original.id)
    registered = registered_poses(store, changed)
    metadata = get_scene_map(store, changed, include_points=False)
    assert metadata["world_from_lidar"] == {key: value.tolist() for key, value in registered.items()}
    assert get_scene_map(store, changed)["points"] == [[10., -1., 0., .4], [22., -1., 0., .8]]
    camera_delta = np.asarray(update["cameras"]["front"]) @ np.linalg.inv(np.asarray(original.cameras[0].camera_from_ego))
    scene = get_scene(store, changed)
    pixels = []
    for old_frame, frame in zip(original.frames, changed.frames):
        lidar_delta = np.linalg.inv(np.asarray(old_frame.lidar_to_ego)) @ np.asarray(frame.lidar_to_ego)
        for field in ("annotation_camera_from_lidar", "projection_camera_from_lidar"):
            np.testing.assert_allclose(getattr(frame, field)["front"],
                                       camera_delta @ np.asarray(getattr(old_frame, field)["front"]) @ lidar_delta,
                                       atol=1e-12)
        local = frame_annotations(scene, changed, frame, registered)
        raw = frame.model_dump()
        raw["points"] = local.vectors[0].points
        projected = project_points(raw, changed.cameras[0].model_dump())["points"]
        assert len(projected) == 2
        # Independent optical landmark calculation: each image has its own
        # source camera pose; the edited primary coordinate convention cancels.
        world = np.column_stack((np.asarray(scene.vectors[0].points), np.ones(2)))
        optical = world @ (camera_delta @ np.asarray(old_frame.projection_camera_from_lidar["front"])).T
        expected = optical[:, :2] / optical[:, 2:3] * 300 + [320, 180]
        actual = [[point["u"], point["v"]] for point in projected]
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        pixels.append(actual)
    assert not np.allclose(pixels[0], pixels[1])
    for path, content in before_files.items():
        assert path.read_bytes() == content


def test_legacy_supplied_stale_metadata_uses_authoritative_registration(supplied):
    store, cloud = supplied
    clip = store.source_clip("supplied")
    expected = {frame.id: pose(index * .2, index * .1, .01 * index).tolist()
                for index, frame in enumerate(clip.frames)}
    atomic_json(store.root / "scene_poses" / "supplied.json",
                {"clip_fingerprint": _clip_fingerprint(store, clip), "world_from_lidar": expected})
    path = store.root / "scene_maps" / "supplied.json"
    cached = json.loads(path.read_text())
    cached["source_fingerprint"] = "stale"  # Old calibration writer used this token.
    atomic_json(path, cached)
    before_map, before_cloud = path.read_bytes(), cloud.read_bytes()
    for include_points in (False, True):
        assert get_scene_map(store, clip, include_points=include_points)["world_from_lidar"] == expected
    assert path.read_bytes() == before_map and cloud.read_bytes() == before_cloud


def test_explicit_supplied_geometry_invalidation_is_respected(supplied):
    store, _ = supplied
    invalidate_scene_map(store, "supplied")
    with pytest.raises(KeyError, match="explicitly invalidated"):
        get_scene_map(store, store.clip("supplied"))


def test_noop_calibration_preserves_source_matrices_exactly(supplied):
    store, _ = supplied
    path = store.root / "clips" / "supplied.json"
    before = path.read_bytes()
    result = save_calibration(store, "supplied", CalibrationUpdate.model_validate(update_for(store)))
    assert not result["lidar_changed"] and not result["map_rebuild_required"]
    assert path.read_bytes() == before


def test_changed_mount_stales_local_record_without_baking_overlay(supplied):
    store, cloud = supplied
    source = store.source_clip("supplied")
    record = store.root / "projection_calibrations" / "supplied" / "f0" / "front.json"
    overlay = pose(.7) @ np.asarray(_base_matrix(source.frames[0], source.cameras[0]))
    atomic_json(record, {"base_fingerprint": _base_fingerprint(source.frames[0], source.cameras[0]),
                         "matrix": overlay.tolist(), "method": "manual", "sequence": 1})
    before_record, before_cloud = record.read_bytes(), cloud.read_bytes()
    np.testing.assert_allclose(store.clip("supplied").frames[0].projection_camera_from_lidar["front"], overlay)
    update = update_for(store)
    update["cameras"]["front"][0][3] += .2
    save_calibration(store, "supplied", CalibrationUpdate.model_validate(update))
    document = document_for(store, "supplied", "f0", "front")
    assert document["stale"] and document["saved_matrix"] is None
    expected = np.asarray(source.frames[0].projection_camera_from_lidar["front"]).copy()
    expected[0, 3] += .2
    np.testing.assert_allclose(store.clip("supplied").frames[0].projection_camera_from_lidar["front"], expected)
    assert record.read_bytes() == before_record and cloud.read_bytes() == before_cloud
