"""Complete scans, index-stable paging and portable external point assets."""
from __future__ import annotations

import io
import json
import zipfile

from fastapi.testclient import TestClient
import numpy as np

from omnilabel.main import create_app


def manifest(count: int):
    identity = np.eye(4).tolist()
    return {"id": "full-scan", "name": "Complete source scan", "fps": 10,
            "cameras": [{"id": "front", "name": "Front", "width": 100, "height": 100,
                         "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]], "camera_from_ego": identity}],
            "frames": [{"id": "f0", "timestamp": 0, "ego_to_world": identity, "lidar_to_ego": identity,
                        "points_path": "scans/source.npy", "point_count": count, "images": {}}]}


def archive_for(points: np.ndarray):
    cloud, output = io.BytesIO(), io.BytesIO()
    np.save(cloud, points, allow_pickle=False)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest(len(points))))
        archive.writestr("scans/source.npy", cloud.getvalue())
    return output.getvalue()


def test_scan_above_previous_limit_preserves_every_point_tail_label_and_portable_exports(tmp_path):
    # The old importer rejected this file outright, and old full-map previews
    # would discard its last records. Half a million rows is only 8 MB on disk.
    source = np.zeros((500_003, 4), dtype="<f4")
    source[:, 0] = np.arange(len(source), dtype=np.float32) * .01
    source[:, 3] = .25
    source[-1] = [125.5, 3.25, -.75, .9375]
    route = "/api/clips/full-scan/frames/f0"
    with TestClient(create_app(tmp_path / "first", seed_demo=False)) as client:
        result = client.post("/api/clips/import", files={"file": ("full.zip", archive_for(source))})
        assert result.status_code == 201, result.text
        frame = client.app.state.store.clip("full-scan").frames[0]
        assert frame.points == [] and frame.point_count == len(source)
        stored = client.app.state.store.frame_point_array(frame)
        assert isinstance(stored, np.memmap) and stored.dtype == source.dtype
        np.testing.assert_array_equal(stored[-1], source[-1])
        metadata = client.get("/api/clips/full-scan").json()["frames"][0]
        assert metadata["points"] == [] and metadata["point_count"] == len(source)
        tail = client.get(route + "/points", params={"offset": 500_000, "count": 20}).json()
        assert tail["point_count"] == len(source) and tail["returned_point_count"] == 3
        assert tail["offset"] == 500_000 and tail["points"][-1] == source[-1].tolist()
        complete = client.get(route + "/points").json()
        assert len(complete["points"]) == len(source) and complete["points"][-1] == source[-1].tolist()
        labels = client.get(route + "/annotations").json()
        labels["point_labels"] = {str(len(source) - 1): "lane_marking"}
        saved = client.put(route + "/annotations", json=labels)
        assert saved.status_code == 200
        invalid = saved.json()
        invalid["point_labels"][str(len(source))] = "outside"
        assert client.put(route + "/annotations", json=invalid).status_code == 422
        portable = client.get("/api/clips/full-scan/export.zip")
        assert portable.status_code == 200
        with zipfile.ZipFile(io.BytesIO(portable.content)) as archive:
            frame_raw = json.loads(archive.read("manifest.json"))["clip"]["frames"][0]
            assert "pointcloud_file" not in frame_raw and "points_path" in frame_raw
            exported_cloud = np.load(io.BytesIO(archive.read(frame_raw["points_path"])), allow_pickle=False)
            np.testing.assert_array_equal(exported_cloud, source)
        # JSON export is self-contained for point data, rather than referring to
        # a filename that exists only in the first server's pointcloud directory.
        exported = client.get("/api/clips/full-scan/export").json()
        exported_frame = exported["clip"]["frames"][0]
        assert "pointcloud_file" not in exported_frame and len(exported_frame["points"]) == len(source)
        np.testing.assert_array_equal(exported_frame["points"][-1], source[-1])
    with TestClient(create_app(tmp_path / "fresh", seed_demo=False)) as fresh:
        restored = fresh.post("/api/clips/import", files={"file": ("copy.zip", portable.content)})
        assert restored.status_code == 201, restored.text
        assert fresh.get(route + "/annotations").json() == saved.json()
        np.testing.assert_array_equal(fresh.app.state.store.frame_point_array(
            fresh.app.state.store.clip("full-scan").frames[0]), source)


def test_inline_frame_above_previous_limit_is_valid_without_point_count_quota():
    from omnilabel.models import Frame
    value = manifest(0)["frames"][0]
    value.pop("points_path")
    value.pop("point_count")
    value["points"] = [[0, 0, 0, .5]] * 500_001
    frame = Frame.model_validate(value)
    assert len(frame.points) == frame.point_count == 500_001


def test_invalid_tail_is_not_hidden_by_sampling_and_failed_import_cleans_staging(tmp_path):
    source = np.zeros((100_005, 4), dtype=np.float32)
    source[-1, 0] = np.nan
    with TestClient(create_app(tmp_path, seed_demo=False)) as client:
        result = client.post("/api/clips/import", files={"file": ("bad.zip", archive_for(source))})
        assert result.status_code == 422 and "finite" in result.json()["detail"]
        assert client.get("/api/clips").json() == []
    assert list((tmp_path / "pointclouds").iterdir()) == []
    assert list((tmp_path / "assets").iterdir()) == []


def test_sq_packaging_does_not_split_a_scene_based_on_point_count_or_default_byte_quota():
    from scripts.package_sq_data import chunks
    records = [{"points": 3_000_000, "asset_size": 300 * 1024 * 1024, "frame": {"id": "f0"}},
               {"points": 4_000_000, "asset_size": 400 * 1024 * 1024, "frame": {"id": "f1"}}]
    assert list(chunks(records, max_frames=2000)) == [records]
    assert list(chunks(records, max_frames=1)) == [[records[0]], [records[1]]]


def test_model_input_validator_keeps_large_complete_source_array():
    from model_workers.common import points_array
    source = np.zeros((500_001, 4), dtype=np.float32)
    source[-1] = [1, 2, 3, .9]
    validated = points_array({"points": source})
    assert len(validated) == len(source)
    np.testing.assert_array_equal(validated[-1], source[-1])
