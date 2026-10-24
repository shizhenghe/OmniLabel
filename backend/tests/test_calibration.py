"""Calibration persistence, projection transforms and original-data preservation."""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from omnilabel.calibration import mount_calibration_routes
from omnilabel.main import create_app
from omnilabel.models import Annotations, Clip
from omnilabel.scene import FUSION_VERSION, _clip_fingerprint, get_scene_map, registered_poses, source_fingerprint
from omnilabel.storage import atomic_json


@pytest.fixture
def calibration_client(tmp_path):
    identity = np.eye(4).tolist()
    clip = Clip.model_validate({"id": "calibration", "name": "Calibrated source", "fps": 10,
        "cameras": [{"id": "front", "name": "Front", "width": 200, "height": 100,
                     "intrinsics": [[100, 0, 100], [0, 100, 50], [0, 0, 1]], "camera_from_ego": identity}],
        "frames": [{"id": "f0", "timestamp": 0, "ego_to_world": identity, "lidar_to_ego": identity,
                    "pointcloud_file": "calibration/f0.npy", "point_count": 2, "images": {}},
                   {"id": "f1", "timestamp": .1, "ego_to_world": identity, "lidar_to_ego": identity,
                    "points": [[1, 0, 5, .4]], "images": {}}]})
    app = create_app(tmp_path, seed_demo=False)
    store = app.state.store
    if not any(route.path == "/api/clips/{clip_id}/calibration" for route in app.routes):
        before_routes = list(app.routes)
        mount_calibration_routes(app, store)
        # The feature is independently testable before main.py is wired. Its
        # routes must precede the SPA fallback, just as they do in production.
        app.router.routes[:] = app.routes[len(before_routes):] + before_routes
    store.add_clip(clip, {"f0": Annotations(point_labels={"1": "manual-road"})})
    cloud = tmp_path / "pointclouds" / "calibration" / "f0.npy"
    cloud.parent.mkdir(parents=True)
    np.save(cloud, np.array([[1, 0, 5, .4], [2, 1, 5, .8]], dtype=np.float32))
    world = np.eye(4); world[0, 3] = 10
    poses = {frame.id: world.tolist() for frame in clip.frames}
    atomic_json(tmp_path / "scene_poses" / "calibration.json",
                {"clip_fingerprint": _clip_fingerprint(store, clip), "world_from_lidar": poses})
    atomic_json(tmp_path / "scenes" / "calibration.json", {"revision": 2, "coordinate_frame": "world",
                "tracks": [], "vectors": [{"id": "manual-lane", "label": "lane", "closed": False,
                                            "points": [[10, 0, 0], [20, 0, 0]]}]})
    atomic_json(tmp_path / "scene_maps" / "calibration.json", {"fusion_version": FUSION_VERSION,
                "source_fingerprint": source_fingerprint(store, clip), "point_count": 1,
                "points": [[11, 0, 5, .4]], "world_from_lidar": poses})
    with TestClient(app) as client:
        yield client, store, cloud


def update_from(document):
    return {key: deepcopy(document[key]) for key in ("revision", "cameras", "lidar_to_ego")}


def test_camera_correction_persists_exports_and_preserves_scans_labels_and_icp(calibration_client):
    client, store, cloud = calibration_client
    source_bytes = cloud.read_bytes()
    label_path = store.root / "annotations" / "calibration" / "f0.json"
    scene_path = store.root / "scenes" / "calibration.json"
    label_bytes, scene_bytes = label_path.read_bytes(), scene_path.read_bytes()
    previous = client.get("/api/clips/calibration/calibration").json()
    update = update_from(previous)
    update["cameras"]["front"][0][3] = .25
    response = client.put("/api/clips/calibration/calibration", json=update)
    assert response.status_code == 200, response.text
    current = response.json()
    assert current["revision"] != previous["revision"]
    assert current["cameras"]["front"][0][3] == .25
    assert not current["lidar_changed"] and not current["map_rebuild_required"]
    assert store.clip("calibration").frames[0].pointcloud_file == "calibration/f0.npy"
    assert store.clip("calibration").frames[0].point_count == 2
    assert store.clip("calibration").frames[1].points == [[1, 0, 5, .4]]
    assert cloud.read_bytes() == source_bytes
    assert label_path.read_bytes() == label_bytes and scene_path.read_bytes() == scene_bytes
    assert registered_poses(store, store.clip("calibration"))["f0"][0, 3] == 10
    assert get_scene_map(store, store.clip("calibration"))["points"] == [[11, 0, 5, .4]]
    original = json.loads((store.root / "calibration_history" / "calibration" / "original.json").read_text())
    assert original["cameras"] == previous["cameras"]
    exported = client.get("/api/clips/calibration/calibration/export")
    assert exported.status_code == 200 and "calibration.json" in exported.headers["content-disposition"]
    assert exported.json()["cameras"] == current["cameras"]
    assert client.get("/api/clips/calibration/calibration").json()["revision"] == current["revision"]
    assert client.put("/api/clips/calibration/calibration", json=update).status_code == 409


def test_lidar_mount_changes_preserve_estimated_ego_but_invalidate_map(calibration_client):
    client, store, _ = calibration_client
    scene_path = store.root / "scenes" / "calibration.json"
    scene_before = scene_path.read_bytes()
    previous = client.get("/api/clips/calibration/calibration").json()
    update = update_from(previous)
    update["lidar_to_ego"]["f0"][1][3] = .3
    saved = client.put("/api/clips/calibration/calibration", json=update)
    assert saved.status_code == 200, saved.text
    assert saved.json()["lidar_changed"] and saved.json()["map_rebuild_required"]
    poses = registered_poses(store, store.clip("calibration"))
    np.testing.assert_allclose(poses["f0"][:3, 3], [10, .3, 0])
    np.testing.assert_allclose(poses["f1"][:3, 3], [10, 0, 0])
    assert scene_path.read_bytes() == scene_before
    with pytest.raises(KeyError):
        get_scene_map(store, store.clip("calibration"))


@pytest.mark.parametrize("invalid", ["camera-id", "missing-frame", "nonrigid", "bad-row"])
def test_invalid_calibration_never_publishes_partial_change(calibration_client, invalid):
    client, store, _ = calibration_client
    before = client.get("/api/clips/calibration/calibration").json()
    original_clip = (store.root / "clips" / "calibration.json").read_bytes()
    update = update_from(before)
    if invalid == "camera-id": update["cameras"]["wrong"] = update["cameras"].pop("front")
    elif invalid == "missing-frame": update["lidar_to_ego"].pop("f1")
    elif invalid == "nonrigid": update["cameras"]["front"][0][0] = 2
    else: update["lidar_to_ego"]["f0"][3][0] = 1
    assert client.put("/api/clips/calibration/calibration", json=update).status_code == 422
    assert (store.root / "clips" / "calibration.json").read_bytes() == original_clip
    assert client.get("/api/clips/calibration/calibration").json()["revision"] == before["revision"]


def test_noop_calibration_does_not_stale_existing_map(calibration_client):
    client, store, _ = calibration_client
    doc = client.get("/api/clips/calibration/calibration").json()
    before = (store.root / "clips" / "calibration.json").read_bytes()
    assert client.put("/api/clips/calibration/calibration", json=update_from(doc)).status_code == 200
    assert (store.root / "clips" / "calibration.json").read_bytes() == before
    assert get_scene_map(store, store.clip("calibration"))["point_count"] == 1
