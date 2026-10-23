import json
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from omnilabel.main import create_app


def bundle():
    identity = np.eye(4).tolist()
    clip = {"id": "scene-exchange", "name": "World scene exchange", "fps": 10,
            "cameras": [{"id": "front", "name": "Front", "width": 100, "height": 100,
                         "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]], "camera_from_ego": identity}],
            "frames": [{"id": fid, "timestamp": i * .1, "ego_to_world": identity, "lidar_to_ego": identity,
                        "points": [[1, 0, 0, .5], [2, 0, 0, .5]], "images": {}} for i, fid in enumerate(("f0", "f1"))]}
    poses = {}
    for i, fid in enumerate(("f0", "f1")):
        pose = np.eye(4); pose[0, 3] = 10 + i * 2
        poses[fid] = pose.tolist()
    scene = {"revision": 2, "coordinate_frame": "world", "vectors": [
        {"id": "lane", "label": "lane", "closed": False, "points": [[10, 2, 0], [20, 2, 0]]}],
        "tracks": [{"id": "vehicle-1", "label": "car", "mode": "dynamic", "keyframes": [
            {"frame_id": "f0", "center": [14, 2, 1], "size": [2, 1, 1], "yaw": 0},
            {"frame_id": "f1", "center": [15, 2, 1], "size": [2, 1, 1], "yaw": .1}]}]}
    return {"schema_version": "1.1", "clip": clip, "scene_annotations": scene,
            "scene_registration": {"world_from_lidar": poses}}


def test_portable_scene_registered_poses_survive_full_api_import_export_and_rebuild(tmp_path):
    payload = bundle()
    with TestClient(create_app(tmp_path / "first", seed_demo=False)) as first:
        imported = first.post("/api/clips/import", files={"file": ("scene.json", json.dumps(payload))})
        assert imported.status_code == 201, imported.text
        evaluated = first.get("/api/clips/scene-exchange/scene/frame/f0").json()
        assert evaluated["boxes"][0]["center"] == [4, 2, 1]
        archive = first.get("/api/clips/scene-exchange/export.zip")
        assert archive.status_code == 200
    with TestClient(create_app(tmp_path / "second", seed_demo=False)) as second:
        restored = second.post("/api/clips/import", files={"file": ("scene.zip", archive.content)})
        assert restored.status_code == 201, restored.text
        assert second.get("/api/clips/scene-exchange/scene/frame/f0").json() == evaluated
        started = second.post("/api/clips/scene-exchange/scene/reconstruct", json={"registration": "poses", "exclude_dynamic": True, "voxel_size": .2}).json()
        for _ in range(100):
            job = second.get(f"/api/jobs/{started['id']}").json()
            if job["status"] in ("completed", "failed"): break
            time.sleep(.01)
        assert job["status"] == "completed", job
        assert job["result"]["world_from_lidar"] == payload["scene_registration"]["world_from_lidar"]
        assert second.get("/api/clips/scene-exchange/scene/annotations").json() == payload["scene_annotations"]


@pytest.mark.parametrize("bad_registration", ["bad", {}, {"world_from_lidar": []}, {"world_from_lidar": {"f0": []}}])
def test_invalid_scene_bundle_is_rejected_before_clip_publication(tmp_path, bad_registration):
    payload = bundle(); payload["scene_registration"] = bad_registration
    with TestClient(create_app(tmp_path, seed_demo=False)) as client:
        result = client.post("/api/clips/import", files={"file": ("scene.json", json.dumps(payload))})
        assert result.status_code == 422
        assert client.get("/api/clips").json() == []
