from __future__ import annotations

import copy
import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from omnilabel.importer import decode_points
from omnilabel.main import create_app
from omnilabel.models import Clip
from omnilabel.storage import Store


def manifest():
    identity = np.eye(4).tolist()
    camera_from_ego = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    frame = {"id": "frame-0", "timestamp": 0, "ego_to_world": identity,
             "lidar_to_ego": identity, "points": [[2, 0, 0, 0.5], [3, 0.5, -1, 0.2]], "images": {}}
    frame2 = copy.deepcopy(frame)
    frame2.update(id="frame-1", timestamp=0.1)
    return {"id": "test-clip", "name": "Test clip", "description": "Calibrated test", "fps": 10,
            "cameras": [{"id": "front", "name": "Front", "width": 100, "height": 100,
                         "intrinsics": [[50, 0, 50], [0, 50, 50], [0, 0, 1]], "camera_from_ego": camera_from_ego}],
            "frames": [frame, frame2]}


def import_json(client, data):
    return client.post("/api/clips/import", files={"file": ("clip.json", json.dumps(data).encode(), "application/json")})


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "data", seed_demo=False)) as connection:
        yield connection


def test_import_projection_revision_export_and_persistence(client, tmp_path):
    assert import_json(client, manifest()).status_code == 201
    assert len(client.get("/api/clips").json()) == 1
    route = "/api/clips/test-clip/frames/frame-0/annotations"
    annotations = client.get(route).json()
    annotations["point_labels"] = {"0": "vehicle"}
    annotations["boxes"] = [{"id": "box-1", "track_id": "track-1", "label": "vehicle", "center": [2, 0, 0], "size": [4, 2, 1.5], "yaw": 0}]
    saved = client.put(route, json=annotations)
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    assert client.put(route, json=annotations).status_code == 409
    projection = client.post("/api/clips/test-clip/project", json={"frame_id": "frame-0", "camera_id": "front"})
    assert projection.status_code == 200
    point = projection.json()["points"][0]
    assert point["index"] == 0 and point["u"] == 50 and point["v"] == 50
    assert point["label"] == "vehicle"
    exported = client.get("/api/clips/test-clip/export").json()
    assert exported["annotations"]["frame-0"]["point_labels"] == {"0": "vehicle"}
    with TestClient(create_app(tmp_path / "data", seed_demo=False)) as reopened:
        assert reopened.get(route).json() == saved.json()


def zip_clip(data, members=None):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(data))
        for name, contents in (members or {}).items():
            archive.writestr(name, contents)
    return result.getvalue()


def test_portable_archive_roundtrip_on_fresh_server(client, tmp_path):
    data = manifest()
    data["frames"][0]["images"] = {"front": "images/front.png"}
    image = io.BytesIO()
    Image.new("RGB", (100, 100), (25, 100, 180)).save(image, "PNG")
    imported = client.post("/api/clips/import", files={"file": ("clip.zip", zip_clip(data, {"images/front.png": image.getvalue()}))})
    assert imported.status_code == 201, imported.text
    labels = client.get("/api/clips/test-clip/frames/frame-0/annotations").json()
    labels["image_polygons"] = [{"id": "polygon-1", "camera_id": "front", "label": "road", "points": [[10, 10], [80, 10], [80, 80], [10, 80]]}]
    labels["point_labels"] = {"1": "road"}
    saved = client.put("/api/clips/test-clip/frames/frame-0/annotations", json=labels).json()
    response = client.get("/api/clips/test-clip/export.zip")
    assert response.status_code == 200
    assert not list((tmp_path / "data").glob(".export-*"))
    with TestClient(create_app(tmp_path / "fresh", seed_demo=False)) as fresh:
        restored = fresh.post("/api/clips/import", files={"file": ("restored.zip", response.content)})
        assert restored.status_code == 201, restored.text
        image_url = restored.json()["frames"][0]["images"]["front"]
        assert fresh.get(image_url).content == image.getvalue()
        assert fresh.get("/api/clips/test-clip/frames/frame-0/annotations").json() == saved
        metadata = fresh.get("/api/clips/test-clip/frames/frame-0/masks/front.json").json()
        assert metadata["labels"]["1"] == "road" and metadata["width"] == 100
        mask = Image.open(io.BytesIO(fresh.get("/api/clips/test-clip/frames/frame-0/masks/front.png").content))
        assert mask.size == (100, 100) and mask.getpixel((50, 50)) == 1 and mask.getpixel((0, 0)) == 0


@pytest.mark.parametrize("path", ["../escape.txt", "/absolute.txt", "images/../../escape", "x\\..\\escape", "C:/evil"])
def test_zip_traversal_is_rejected(client, path):
    response = client.post("/api/clips/import", files={"file": ("bad.zip", zip_clip(manifest(), {path: b"evil"}))})
    assert response.status_code == 422
    assert client.get("/api/clips").json() == []


def test_image_dimension_mismatch_is_rejected(client):
    data = manifest()
    data["frames"][0]["images"] = {"front": "front.png"}
    image = io.BytesIO()
    Image.new("RGB", (10, 10)).save(image, "PNG")
    response = client.post("/api/clips/import", files={"file": ("bad.zip", zip_clip(data, {"front.png": image.getvalue()}))})
    assert response.status_code == 422 and "camera dimensions" in response.json()["detail"]


def test_invalid_transform_labels_and_duplicate_import(client):
    data = manifest()
    broken = copy.deepcopy(data)
    broken["frames"][0]["ego_to_world"][0][0] = 2
    assert import_json(client, broken).status_code == 422
    assert import_json(client, data).status_code == 201
    assert import_json(client, data).status_code == 409
    route = "/api/clips/test-clip/frames/frame-0/annotations"
    value = client.get(route).json()
    value["point_labels"] = {"2": "road"}
    assert client.put(route, json=value).status_code == 422
    value["point_labels"] = {"-1": "road"}
    assert client.put(route, json=value).status_code == 422
    assert client.get("/api/assets/%2e%2e/clips/test-clip.json").status_code == 404


def test_mixed_xyz_xyzi_normalized_and_npy_rejects_complex():
    data = manifest()
    data["frames"][0]["points"] = [[1, 2, 3], [2, 3, 4, 0.8]]
    assert Clip.model_validate(data).frames[0].points[0] == [1, 2, 3, 0]
    contents = io.BytesIO()
    np.save(contents, np.ones((2, 4), dtype=complex))
    with pytest.raises(ValueError, match="real numeric"):
        decode_points(contents.getvalue(), "points.npy")
    with pytest.raises(ValueError):
        decode_points(zip_clip({}), "points.npy")


def test_npy_forged_shape_rejected_before_allocating():
    contents = io.BytesIO()
    np.lib.format.write_array_header_1_0(contents, {"descr": "<f4", "fortran_order": False, "shape": (2**40, 4)})
    with pytest.raises(ValueError, match="500000"):
        decode_points(contents.getvalue(), "points.npy")


def test_corrupt_zip_member_is_a_validation_error(client):
    data = manifest()
    data["frames"][0].pop("points")
    data["frames"][0]["points_path"] = "points.bin"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(data))
        archive.writestr("points.bin", np.zeros((2, 4), dtype="<f4").tobytes())
    payload = bytearray(output.getvalue())
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = archive.getinfo("points.bin")
        offset = member.header_offset + 30 + len(member.filename.encode()) + len(member.extra)
    payload[offset] ^= 1
    response = client.post("/api/clips/import", files={"file": ("corrupt.zip", bytes(payload))})
    assert response.status_code == 422 and "integrity" in response.json()["detail"]


def wait_job(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_map_occupancy_and_segmentation_jobs(client):
    assert import_json(client, manifest()).status_code == 201
    for kind in ("map", "occupancy", "ground-segmentation"):
        response = client.post("/api/clips/test-clip/jobs", json={"kind": kind, "frame_id": "frame-0", "voxel_size": 0.5})
        assert response.status_code == 202
        job = wait_job(client, response.json()["id"])
        assert job["status"] == "completed", job
        if kind == "map":
            assert len(job["result"]["points"]) > 0 and job["result"]["frame_count"] == 2
        elif kind == "occupancy":
            assert any(v["state"] == "occupied" for v in job["result"]["voxels"])
        else:
            assert job["result"]["updated_frames"] == 1
    assert client.get("/api/clips/test-clip/frames/frame-0/annotations").json()["revision"] == 1


def test_mask_zip_has_consistent_snapshot_and_reserved_background(client, monkeypatch):
    assert import_json(client, manifest()).status_code == 201
    route = "/api/clips/test-clip/frames/frame-0/annotations"
    value = client.get(route).json()
    value["image_polygons"] = [
        {"id": "road-polygon", "camera_id": "front", "label": "road", "points": [[10, 10], [80, 10], [80, 80], [10, 80]]},
        {"id": "custom-polygon", "camera_id": "front", "label": "custom-class", "points": [[20, 20], [40, 20], [40, 40], [20, 40]]},
        {"id": "erase-polygon", "camera_id": "front", "label": "unlabeled", "points": [[25, 25], [30, 25], [30, 30], [25, 30]]},
    ]
    assert client.put(route, json=value).status_code == 200
    store = client.app.state.store
    original = store.all_annotations
    snapshots = []

    def snapshot(clip):
        snapshots.append(clip.id)
        return original(clip)

    monkeypatch.setattr(store, "all_annotations", snapshot)
    response = client.get("/api/clips/test-clip/frames/frame-0/masks/front.zip")
    assert response.status_code == 200 and snapshots == ["test-clip"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"mask.png", "labels.json"}
        metadata = json.loads(archive.read("labels.json"))
        mask = Image.open(io.BytesIO(archive.read("mask.png")))
        assert [key for key, label in metadata["labels"].items() if label == "unlabeled"] == ["0"]
        assert metadata["labels"][str(mask.getpixel((22, 22)))] == "custom-class"
        assert mask.getpixel((27, 27)) == 0 and mask.getpixel((50, 50)) == 1
        assert metadata["revision"] == int(response.headers["x-annotation-revision"]) == 1


def test_mask_rejects_oversize_canvas_before_allocation(client):
    data = manifest()
    data["cameras"][0].update(width=16384, height=16384)
    assert import_json(client, data).status_code == 201
    response = client.get("/api/clips/test-clip/frames/frame-0/masks/front.zip")
    assert response.status_code == 422 and "40-million-pixel" in response.json()["detail"]


def test_archive_export_checks_member_limit(client, monkeypatch):
    import omnilabel.main as api
    data = manifest()
    data["frames"][0]["images"] = {"front": "front.png"}
    image = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image, "PNG")
    response = client.post("/api/clips/import", files={"file": ("clip.zip", zip_clip(data, {"front.png": image.getvalue()}))})
    assert response.status_code == 201
    monkeypatch.setattr(api, "MAX_FILES", 1)
    response = client.get("/api/clips/test-clip/export.zip")
    assert response.status_code == 422 and "member" in response.json()["detail"]


def test_store_inspection_preserves_live_jobs_but_service_start_recovers(tmp_path):
    root = tmp_path / "store"
    store = Store(root)
    store.save_job({"id": "running-job", "status": "running", "kind": "map"})
    inspected = Store(root)
    assert inspected.job("running-job")["status"] == "running"
    application = create_app(root, seed_demo=False)
    assert inspected.job("running-job")["status"] == "running"
    with TestClient(application) as client:
        recovered = client.get("/api/jobs/running-job").json()
        assert recovered["status"] == "failed" and "restarted" in recovered["error"]
