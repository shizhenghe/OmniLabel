from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import omnilabel.demo as demo
from omnilabel.main import create_app
from omnilabel.models import Annotations, Clip
from omnilabel.scene import reconstruct_scene
from omnilabel.scene_models import ReconstructionRequest
from omnilabel.storage import Store


def test_synthetic_lane_and_arrow_have_measured_contrast_after_world_fusion(tmp_path, monkeypatch):
    # Rendering uses the same paint polygons; this test concerns LiDAR data, not
    # JPEG rasterization, and avoids writing 84 unrelated camera fixtures.
    monkeypatch.setattr(demo, "_render", lambda *args: None)
    raw, annotations = demo.build_demo(tmp_path, clip_id="demo-ground-detail")
    assert raw["source"] == "synthetic"
    assert raw["frames"][0]["images"]["front"].startswith("/api/assets/demo-ground-detail/")
    clip = Clip.model_validate(raw)
    result = reconstruct_scene(clip, annotations, ReconstructionRequest())
    cloud = np.array(result["points"])
    bright = cloud[cloud[:, 3] > .9]
    assert len(bright) > 3000
    # A dash and an arrow tip at known world positions remain bright. Ordinary
    # asphalt around them remains dim rather than becoming painted by averaging.
    for target in ([1, 3.6, .02], [13.8, 1.8, .02]):
        assert np.linalg.norm(bright[:, :3] - target, axis=1).min() < .13
    assert np.any((cloud[:, 3] < .4) & (np.linalg.norm(cloud[:, :2] - [6, 1.8], axis=1) < .6))
    assert result["intensity_has_contrast"] and result["point_reduction"] == "brightest-measured-point"


def test_xyz_normalisation_does_not_claim_measured_reflectance():
    identity = np.eye(4).tolist()
    clip = Clip.model_validate({"id": "xyz-only", "name": "XYZ input", "fps": 10,
        "cameras": [{"id": "front", "name": "Front", "width": 100, "height": 100,
                     "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]], "camera_from_ego": identity}],
        "frames": [{"id": "f0", "timestamp": 0, "ego_to_world": identity, "lidar_to_ego": identity,
                    "points": [[1, 0, 0], [2, 1, 0]], "images": {}}]})
    result = reconstruct_scene(clip, {}, ReconstructionRequest())
    assert not result["intensity_available"] and not result["intensity_has_contrast"]
    assert all(point[3] == 0 for point in result["points"])


def test_ground_detail_sample_is_seeded_additively_and_never_replaces_saved_demo(tmp_path, monkeypatch):
    identity = np.eye(4).tolist()
    raw = {"id": "demo", "name": "User edited sample", "fps": 10, "cameras": [{
        "id": "front", "name": "Front", "width": 100, "height": 100,
        "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]], "camera_from_ego": identity}],
        "frames": [{"id": "f0", "timestamp": 0, "ego_to_world": identity,
                    "lidar_to_ego": identity, "points": [[1, 0, 0, .3]], "images": {}}]}
    store = Store(tmp_path)
    store.add_clip(Clip.model_validate(raw), {"f0": Annotations(point_labels={"0": "manual-road"})})
    original_clip = (tmp_path / "clips" / "demo.json").read_bytes()
    original_labels = (tmp_path / "annotations" / "demo" / "f0.json").read_bytes()
    built = []

    def build_fixture(asset_dir, clip_id="demo"):
        built.append(clip_id)
        return {**raw, "id": clip_id, "source": "synthetic"}, {}

    monkeypatch.setattr(demo, "build_demo", build_fixture)
    for _ in range(2):
        with TestClient(create_app(tmp_path)) as client:
            ids = {clip["id"] for clip in client.get("/api/clips").json()}
            assert {"demo", "demo-ground-detail"}.issubset(ids)
    assert built == ["demo-ground-detail"]
    assert (tmp_path / "clips" / "demo.json").read_bytes() == original_clip
    assert (tmp_path / "annotations" / "demo" / "f0.json").read_bytes() == original_labels
