"""Batch calibration lifecycle, independent publication and source conflicts.

The registration engine is controlled here; its numerical/runtime behavior has
separate tests. These checks exercise real durable records and HTTP endpoints.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from omnilabel import automatic_calibration as batch
from omnilabel import automatic_projection
from omnilabel.calibration_input import calibration_input
from omnilabel.main import create_app
from omnilabel.models import Annotations, Camera, Clip
from omnilabel.projection_calibration import ProjectionUpdate, document_for, save_projection_calibration
from omnilabel.storage import Store, atomic_json


ROUTE = "/api/clips/batch/automatic-calibration"


def accepted(initial, *, usable=True):
    matrix = np.asarray(initial).copy()
    matrix[0, 3] += .1
    return {"matrix": matrix.tolist(), "usable": usable,
            "metrics": {"validation_nid": .2},
            "rejection_reasons": [] if usable else ["Independent validation did not improve."]}


@pytest.fixture
def calibration_api(tmp_path, monkeypatch):
    identity = np.eye(4).tolist()
    cameras = [{"id": name, "name": name.title(), "width": 100, "height": 100,
                "intrinsics": [[50, 0, 50], [0, 50, 50], [0, 0, 1]],
                "camera_from_ego": identity} for name in ("front", "side")]
    frames = [{"id": "f0", "timestamp": 0, "ego_to_world": identity, "lidar_to_ego": identity,
               "pointcloud_file": "batch/f0.npy", "point_count": 2,
               "images": {name: f"/api/assets/batch/f0-{name}.png" for name in ("front", "side")}},
              {"id": "f1", "timestamp": .1, "ego_to_world": identity, "lidar_to_ego": identity,
               "points": [[0, 0, 5, .2], [1, 0, 6, .8]],
               "images": {"front": "/api/assets/batch/f1-front.png"}},
              {"id": "f2", "timestamp": .2, "ego_to_world": identity, "lidar_to_ego": identity,
               "points": [], "images": {name: f"/api/assets/batch/f2-{name}.png" for name in ("front", "side")}}]
    app = create_app(tmp_path, seed_demo=False)
    store = app.state.store
    clip = Clip.model_validate({"id": "batch", "name": "Batch", "fps": 10, "frames": frames, "cameras": cameras})
    store.add_clip(clip, {"f0": Annotations(point_labels={"1": "lane"})})
    second = clip.model_copy(update={"id": "other"})
    store.add_clip(second)
    cloud = tmp_path / "pointclouds" / "batch" / "f0.npy"
    cloud.parent.mkdir(parents=True)
    np.save(cloud, np.array([[0, 0, 5, .2], [1, 0, 6, .8]], dtype=np.float32))
    for frame in clip.frames:
        for image in frame.images.values():
            path = tmp_path / "assets" / image.removeprefix("/api/assets/")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.arange(10000, dtype=np.uint8).reshape(100, 100)).save(path)
    monkeypatch.setattr(batch, "engines", lambda: [
        {"id": "koide", "name": "Koide", "available": True, "detail": "Test runtime"},
        {"id": "direct_nid", "name": "CPU", "available": True, "detail": "Test runtime"}])
    monkeypatch.setattr(batch, "solve_pair", lambda engine, points, image_path, camera, initial, work_dir: accepted(initial))
    monkeypatch.setattr(automatic_projection, "evaluate_automatic_candidate", lambda *args, **kwargs:
                        {"usable": True, "metrics": {"validation_nid": .25}, "rejection_reasons": []})
    with TestClient(app) as client:
        yield client, store, app.state.automatic_calibration


def test_manual_calibration_is_independent_and_survives_reload(calibration_api):
    client, store, _ = calibration_api
    source = store.source_clip("batch").model_dump()
    route = "/api/clips/batch/frames/{}/projection-calibration/{}"
    for frame_id, shift in [("f0", .12), ("f1", -.08)]:
        path = route.format(frame_id, "front")
        document = client.get(path).json()
        matrix = np.asarray(document["effective_matrix"])
        matrix[0, 3] = shift
        result = client.put(path, json={"revision": document["revision"], "matrix": matrix.tolist(),
                                      "method": "manual", "scope": "frame"})
        assert result.status_code == 200, result.text
        assert result.json()["metrics"]["validation"] == "manual_review"
        assert client.put(path, json={"revision": document["revision"], "matrix": matrix.tolist(),
                                     "method": "manual"}).status_code == 409
    effective = Store(store.root).clip("batch")
    assert effective.frames[0].projection_camera_from_lidar["front"][0][3] == .12
    assert effective.frames[1].projection_camera_from_lidar["front"][0][3] == -.08
    assert "side" not in effective.frames[0].projection_camera_from_lidar
    assert store.source_clip("batch").model_dump() == source
    document = client.get(route.format("f0", "front")).json()
    denied = client.put(route.format("f0", "front"), json={"revision": document["revision"],
        "matrix": document["saved_matrix"], "method": "manual", "scope": "clip"})
    assert denied.status_code == 422
    forged = client.put(route.format("f0", "front"), json={"revision": document["revision"],
        "matrix": document["saved_matrix"], "method": "pjlab",
        "metrics": {"line_overlap_before": .1, "line_overlap_after": 1}})
    assert forged.status_code == 422


def test_pjlab_mask_is_bound_to_pair_and_original_image(calibration_api, monkeypatch):
    from io import BytesIO
    client, store, manager = calibration_api
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[20:25, 10:90] = 255
    buffer = BytesIO(); Image.fromarray(pixels).save(buffer, format="PNG")
    path = ROUTE + "/masks/f0/front"
    assert client.put(path, content=b"invalid").status_code == 422
    assert client.put(path, content=buffer.getvalue()).status_code == 200
    clip = store.source_clip("batch")
    token = manager._pair_fingerprint("batch", clip.frames[0], clip.cameras[0], "pjlab")
    assert token
    with pytest.raises(ValueError, match="先上传"):
        manager._pair_fingerprint("batch", clip.frames[0], clip.cameras[1], "pjlab")
    image = store.root / "assets/batch/f0-front.png"
    image.write_bytes(image.read_bytes())
    with pytest.raises(ValueError, match="原始图像已变化"):
        manager._pair_fingerprint("batch", clip.frames[0], clip.cameras[0], "pjlab")
    assert client.put(path, content=buffer.getvalue()).status_code == 200
    assert manager._pair_fingerprint("batch", clip.frames[0], clip.cameras[0], "pjlab") != token
    manager._active.add("running")
    assert client.put(path, content=buffer.getvalue()).status_code == 409
    manager._active.clear()


def test_pjlab_batch_keeps_distinct_frame_results(calibration_api, monkeypatch):
    from io import BytesIO
    client, store, _ = calibration_api
    pixels = np.zeros((100, 100, 3), dtype=np.uint8); pixels[20:25, 10:90] = 255
    buffer = BytesIO(); Image.fromarray(pixels).save(buffer, format="PNG")
    for frame_id in ["f0", "f1"]:
        assert client.put(ROUTE + f"/masks/{frame_id}/front", content=buffer.getvalue()).status_code == 200
    monkeypatch.setattr(batch, "engines", lambda: [{"id": "pjlab", "available": True, "detail": "CLI fixture"}])
    def solve(engine, points, image, camera, initial, work_dir, *, mask_path):
        assert engine == "pjlab" and mask_path.is_file()
        result = accepted(initial)
        result["matrix"][0][3] = .1 if mask_path.parent.name == "f0" else -.2
        result["metrics"].update(line_overlap_before=.6, line_overlap_after=.8)
        return result
    monkeypatch.setattr(batch, "solve_pair", solve)
    response = client.post(ROUTE + "/jobs", json={"engine": "pjlab", "frame_ids": ["f0", "f1"], "camera_ids": ["front"]})
    assert response.status_code == 202, response.text
    job = finished(client, response.json()["id"])
    assert job["accepted"] == 2
    saved = client.post(ROUTE + f'/jobs/{job["id"]}/apply')
    assert saved.status_code == 200 and saved.json()["saved"] == 2
    clip = store.clip("batch")
    assert clip.frames[0].projection_camera_from_lidar["front"][0][3] == .1
    assert clip.frames[1].projection_camera_from_lidar["front"][0][3] == -.2


def start(client, **options):
    response = client.post(ROUTE + "/jobs", json={"engine": "koide", **options})
    assert response.status_code == 202, response.text
    return response.json()


def finished(client, job_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        response = client.get(ROUTE + f"/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in batch.TERMINAL:
            return job
        time.sleep(.01)
    pytest.fail("Batch registration did not finish")


def modify_input(store, kind):
    frame = store.source_clip("batch").frames[0]
    if kind == "image":
        path = store.root / "assets" / "batch" / "f0-front.png"
        before = path.stat()
        Image.new("L", (100, 100), 77).save(path)
    elif kind == "cloud":
        path = store.pointcloud_path(frame)
        before = path.stat()
        np.save(path, np.array([[.4, 0, 5, .2], [1, 0, 6, .8]], dtype=np.float32))
    else:
        path = store.root / "clips" / "batch.json"
        raw = json.loads(path.read_text())
        raw["frames"][0]["points"] = [[2, 0, 5, .2], [1, 0, 6, .8]]
        raw["frames"][0]["pointcloud_file"] = None
        atomic_json(path, raw)
        return
    # Make the identity change deterministic even on a coarse timestamp FS.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))


def test_koide_solves_a_local_delta_instead_of_large_world_transform(tmp_path, monkeypatch):
    camera = Camera(id="front", name="Front", width=100, height=100,
                    intrinsics=[[50, 0, 50], [0, 50, 50], [0, 0, 1]],
                    camera_from_ego=np.eye(4).tolist())
    initial = np.eye(4); initial[:3, 3] = [-291, -16, 3]
    xyz = np.array([[0, 0, 10], [1, 0, 11], [0, 1, 12], [1, 1, 13]], dtype=float)
    prepared = {"xyz": xyz, "intensity": np.array([.1, .2, .3, .4]),
                "holdout": np.array([False, True, False, False])}
    monkeypatch.setattr(automatic_projection, "_prepare", lambda *args: prepared)
    monkeypatch.setattr(automatic_projection, "_quality", lambda *args: ({}, []))
    captured = {}

    def evaluate(points, image_path, value_camera, baseline, candidate):
        captured["candidate"] = candidate
        return {"matrix": candidate.tolist(), "usable": True, "metrics": {}, "rejection_reasons": []}

    monkeypatch.setattr(automatic_projection, "evaluate_automatic_candidate", evaluate)
    from omnilabel import koide_runtime
    delta = np.eye(4); delta[0, 3] = .02

    def run(**kwargs):
        np.testing.assert_allclose(kwargs["initial_camera_from_lidar"], np.eye(4))
        np.testing.assert_allclose(kwargs["points"][:, :3], xyz[[0, 2, 3]])
        return {"matrix": delta.tolist(), "artifacts": {}}

    monkeypatch.setattr(koide_runtime, "run_koide_calibration", run)
    result = batch.solve_pair("koide", np.ones((4, 4)), tmp_path / "image.png", camera, initial, tmp_path / "work")
    np.testing.assert_allclose(captured["candidate"], delta @ initial)
    np.testing.assert_allclose(result["matrix"], delta @ initial)
    assert result["metrics"]["optimization_frame"] == "initial_camera_coordinates"


def test_synchronized_scan_is_converted_back_to_primary_map_coordinates(tmp_path):
    store = Store(tmp_path)
    scan_path = tmp_path / "pointclouds" / "clip" / "calibration" / "f0.npy"
    scan_path.parent.mkdir(parents=True)
    scan = np.array([[1, 2, 3, .4], [4, 5, 6, .8]], dtype=np.float32)
    np.save(scan_path, scan)
    ego_to_world = np.eye(4); ego_to_world[0, 3] = 100
    world_to_ego = np.linalg.inv(ego_to_world)
    frame = Clip.model_validate({"id": "clip", "name": "Clip", "fps": 10, "cameras": [], "frames": [{
        "id": "f0", "timestamp": 0, "ego_to_world": ego_to_world.tolist(),
        "lidar_to_ego": world_to_ego.tolist(), "points": [[0, 0, 0, 1]],
        "calibration_pointcloud_file": "clip/calibration/f0.npy", "calibration_point_count": 2,
        "calibration_lidar_to_ego": np.eye(4).tolist(), "calibration_timestamp": 123.0,
        "images": {},
    }]}).frames[0]
    camera_from_world = np.eye(4); camera_from_world[0, 3] = -100
    value = calibration_input(store, frame, camera_from_world)
    np.testing.assert_array_equal(value.points, scan)
    np.testing.assert_allclose(value.primary_from_input, ego_to_world)
    np.testing.assert_allclose(value.initial_camera_from_input, np.eye(4))
    candidate_from_input = np.eye(4); candidate_from_input[1, 3] = .05
    np.testing.assert_allclose(value.camera_from_primary(candidate_from_input),
                               candidate_from_input @ camera_from_world)


def test_all_pairs_report_individual_quality_and_save_only_accepted(calibration_api, monkeypatch):
    client, store, _ = calibration_api
    original_files = [store.root / "clips" / "batch.json", store.root / "pointclouds" / "batch" / "f0.npy",
                      store.root / "annotations" / "batch" / "f0.json"]
    original_bytes = [path.read_bytes() for path in original_files]

    def solve(engine, points, image_path, camera, initial, work_dir):
        if image_path.name == "f1-front.png":
            raise ValueError("Insufficient intensity information")
        return accepted(initial, usable=camera.id == "front")

    monkeypatch.setattr(batch, "solve_pair", solve)
    job = finished(client, start(client)["id"])
    assert job["status"] == "completed" and job["total"] == job["processed"] == 6
    assert (job["accepted"], job["rejected"], job["failed"], job["skipped"], job["saved"]) == (1, 1, 1, 3, 0)
    assert not (store.root / "projection_calibrations").exists()
    assert not store.clip("batch").frames[0].projection_camera_from_lidar
    saved = client.post(ROUTE + f"/jobs/{job['id']}/apply")
    assert saved.status_code == 200, saved.text
    assert saved.json()["saved"] == 1 and saved.json()["accepted"] == 0
    loaded = Store(store.root).clip("batch")
    assert set(loaded.frames[0].projection_camera_from_lidar) == {"front"}
    assert not loaded.frames[1].projection_camera_from_lidar
    # Save re-evaluates numerical quality instead of trusting job metrics.
    record = json.loads((store.root / "projection_calibrations" / "batch" / "f0" / "front.json").read_text())
    assert record["metrics"] == {"validation_nid": .25, "input_point_count": 2,
                                 "input_source": "primary_point_cloud"}
    assert [path.read_bytes() for path in original_files] == original_bytes
    projection = client.post("/api/clips/batch/project", json={"frame_id": "f0", "camera_id": "front"})
    assert projection.status_code == 200, projection.text
    assert projection.json()["points"][0]["u"] == pytest.approx(51)
    status = client.get(ROUTE).json()
    assert status["saved_pairs"] == 1 and status["total_pairs"] == 6
    exported = client.get(ROUTE + "/export")
    assert exported.status_code == 200 and "calibration.json" in exported.headers["content-disposition"]
    value = exported.json()
    assert [frame["frame_id"] for frame in value["frames"]] == ["f0", "f1", "f2"]
    assert all(set(frame["cameras"]) == {"front", "side"} for frame in value["frames"])
    assert value["frames"][0]["cameras"]["front"]["camera_from_lidar"] == record["matrix"]
    assert not value["frames"][2]["cameras"]["front"]["saved"]


def test_job_scope_can_target_current_pair_or_current_camera(calibration_api):
    client, _, _ = calibration_api
    pair = finished(client, start(client, frame_ids=["f0"], camera_ids=["front"])["id"])
    assert pair["total"] == 1
    assert pair["selection"] == {"frame_ids": ["f0"], "camera_ids": ["front"]}
    assert [(item["frame_id"], item["camera_id"]) for item in pair["items"]] == [("f0", "front")]
    camera = finished(client, start(client, camera_ids=["side"])["id"])
    assert camera["total"] == 3
    assert all(item["camera_id"] == "side" for item in camera["items"])
    for payload in ({"frame_ids": []}, {"camera_ids": []}, {"frame_ids": ["missing"]},
                    {"camera_ids": ["missing"]}):
        response = client.post(ROUTE + "/jobs", json={"engine": "koide", **payload})
        assert response.status_code == 422, response.text


def test_running_job_exclusion_cancellation_and_clip_isolation(calibration_api, monkeypatch):
    client, _, _ = calibration_api
    entered, release = threading.Event(), threading.Event()

    def solve(engine, points, image_path, camera, initial, work_dir):
        entered.set()
        assert release.wait(8)
        return accepted(initial)

    monkeypatch.setattr(batch, "solve_pair", solve)
    job = start(client)
    try:
        assert entered.wait(3)
        assert client.post(ROUTE + "/jobs", json={"engine": "direct_nid"}).status_code == 409
        assert client.post(ROUTE + f"/jobs/{job['id']}/apply").status_code == 409
        other = ROUTE.replace("/batch/", "/other/") + f"/jobs/{job['id']}"
        assert client.get(other).status_code == 404
        assert client.post(other + "/cancel").status_code == 404
        assert client.post(other + "/apply").status_code == 404
        cancelled = client.post(ROUTE + f"/jobs/{job['id']}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["cancel_requested"]
    finally:
        release.set()
    job = finished(client, job["id"])
    assert job["status"] == "cancelled"
    assert job["accepted"] == 1 and job["skipped"] == 5 and job["processed"] == 6
    assert finished(client, start(client)["id"])["status"] == "completed"


@pytest.mark.parametrize("kind", ["image", "cloud", "inline-cloud"])
def test_changed_input_after_solve_blocks_publication(calibration_api, kind):
    client, store, _ = calibration_api
    job = finished(client, start(client)["id"])
    modify_input(store, kind)
    response = client.post(ROUTE + f"/jobs/{job['id']}/apply")
    assert response.status_code == 200, response.text
    result = response.json()
    front = next(item for item in result["items"] if (item["frame_id"], item["camera_id"]) == ("f0", "front"))
    assert front["status"] == "conflict" and front["reason"]
    assert not (store.root / "projection_calibrations" / "batch" / "f0" / "front.json").exists()
    assert result["saved"] == (2 if kind == "image" else 1)


@pytest.mark.parametrize("kind", ["image", "cloud"])
def test_changed_input_during_solve_rejects_candidate(calibration_api, monkeypatch, kind):
    client, store, _ = calibration_api
    changed = False

    def solve(engine, points, image_path, camera, initial, work_dir):
        nonlocal changed
        if not changed:
            changed = True
            modify_input(store, kind)
        return accepted(initial)

    monkeypatch.setattr(batch, "solve_pair", solve)
    job = finished(client, start(client)["id"])
    assert job["items"][0]["status"] == "conflict"
    assert job["failed"] == 1 and job["accepted"] == 2
    assert not (store.root / "projection_calibrations").exists()


def test_newer_projection_revision_survives_old_batch_apply(calibration_api):
    client, store, _ = calibration_api
    job = finished(client, start(client)["id"])
    document = document_for(store, "batch", "f0", "front")
    matrix = np.eye(4); matrix[0, 3] = .3
    save_projection_calibration(store, "batch", "f0", "front",
        ProjectionUpdate(revision=document["revision"], matrix=matrix.tolist(), method="direct_nid"))
    response = client.post(ROUTE + f"/jobs/{job['id']}/apply")
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["status"] == "conflict"
    assert store.clip("batch").frames[0].projection_camera_from_lidar["front"] == matrix.tolist()


def test_apply_is_idempotent_and_preserves_saved_pairs_by_default(calibration_api):
    client, store, _ = calibration_api
    job = finished(client, start(client)["id"])
    route = ROUTE + f"/jobs/{job['id']}/apply"
    first = client.post(route)
    assert first.status_code == 200 and first.json()["saved"] == 3
    paths = sorted((store.root / "projection_calibrations").rglob("*.json"))
    saved_bytes = [path.read_bytes() for path in paths]
    second = client.post(route)
    assert second.status_code == 200 and second.json() == first.json()
    assert [path.read_bytes() for path in paths] == saved_bytes
    preserved = finished(client, start(client)["id"])
    assert preserved["skipped"] == 6 and preserved["accepted"] == 0
    overwritten = finished(client, start(client, overwrite_saved=True)["id"])
    assert overwritten["accepted"] == 3 and overwritten["skipped"] == 3
    assert [path.read_bytes() for path in paths] == saved_bytes


def test_inherited_clip_projection_is_preserved_unless_overwrite_is_requested(calibration_api):
    client, store, _ = calibration_api
    current = document_for(store, "batch", "f0", "front")
    matrix = np.eye(4)
    matrix[0, 3] = .05
    save_projection_calibration(store, "batch", "f0", "front", ProjectionUpdate(
        revision=current["revision"], matrix=matrix.tolist(), method="koide", scope="clip"))
    inherited = document_for(store, "batch", "f1", "front")
    assert inherited["saved_matrix"] is None and inherited["scope"] == "clip"
    exported = client.get(ROUTE + "/export").json()["frames"][1]["cameras"]["front"]
    assert exported["saved"] and exported["scope"] == "clip"
    assert exported["clip_anchor_frame_id"] == "f0" and not exported["clip_stale"]
    np.testing.assert_allclose(exported["camera_from_lidar"], inherited["effective_matrix"])
    record = store.root / "clip_projection_calibrations" / "batch" / "front.json"
    before = record.read_bytes()
    protected = finished(client, start(client, frame_ids=["f1"], camera_ids=["front"])["id"])
    assert protected["skipped"] == 1 and protected["accepted"] == 0
    assert "已保存" in protected["items"][0]["reason"]
    overwrite = finished(client, start(client, frame_ids=["f1"], camera_ids=["front"], overwrite_saved=True)["id"])
    assert overwrite["accepted"] == 1 and overwrite["skipped"] == 0
    assert record.read_bytes() == before
    assert not (store.root / "projection_calibrations" / "batch" / "f1" / "front.json").exists()


def test_apply_revalidates_engine_quality(calibration_api, monkeypatch):
    client, store, _ = calibration_api
    job = finished(client, start(client)["id"])
    monkeypatch.setattr(automatic_projection, "evaluate_automatic_candidate", lambda *args, **kwargs:
                        {"usable": False, "metrics": {}, "rejection_reasons": ["Quality no longer passes"]})
    response = client.post(ROUTE + f"/jobs/{job['id']}/apply")
    assert response.status_code == 200
    assert response.json()["saved"] == 0 and response.json()["failed"] == 3
    assert not (store.root / "projection_calibrations").exists()


@pytest.mark.parametrize("kind", ["image", "cloud"])
def test_saved_projection_is_disabled_when_original_sensor_input_changes(calibration_api, kind):
    client, store, _ = calibration_api
    job = finished(client, start(client)["id"])
    saved = client.post(ROUTE + f"/jobs/{job['id']}/apply")
    assert saved.status_code == 200 and saved.json()["saved"] == 3
    assert store.clip("batch").frames[0].projection_camera_from_lidar["front"]
    modify_input(store, kind)
    document = document_for(store, "batch", "f0", "front")
    assert document["stale"] and document["saved_matrix"] is None
    assert "front" not in Store(store.root).clip("batch").frames[0].projection_camera_from_lidar
    exported = client.get(ROUTE + "/export").json()
    front = exported["frames"][0]["cameras"]["front"]
    assert front["stale"] and not front["saved"]


def test_restart_recovers_interrupted_job_and_retains_computed_candidates(calibration_api):
    client, store, manager = calibration_api
    job = finished(client, start(client)["id"])
    job.update(status="running")
    job["items"][1] = {"frame_id": "f0", "camera_id": "side", "status": "running"}
    job["items"][2] = {"frame_id": "f1", "camera_id": "front", "status": "pending"}
    atomic_json(manager.directory / f"{job['id']}.json", job)
    with ThreadPoolExecutor(max_workers=1) as executor:
        restored = batch.AutomaticCalibrationManager(Store(store.root), executor)
        restored.recover()
        recovered = restored.get("batch", job["id"])
        assert recovered["status"] == "failed" and recovered["accepted"] == 1
        assert recovered["skipped"] == 5
        saved = restored.apply("batch", job["id"])
        assert saved["saved"] == 1 and saved["accepted"] == 0
    assert Store(store.root).clip("batch").frames[0].projection_camera_from_lidar["front"]


def test_apply_retry_after_record_publication_before_job_checkpoint(calibration_api, monkeypatch):
    client, store, manager = calibration_api
    job = finished(client, start(client)["id"])
    save_checkpoint = manager._save

    def interrupted_checkpoint(value):
        raise OSError("Simulated interruption after projection publication")

    monkeypatch.setattr(manager, "_save", interrupted_checkpoint)
    with pytest.raises(OSError, match="Simulated interruption"):
        manager.apply("batch", job["id"])
    first_record = store.root / "projection_calibrations" / "batch" / "f0" / "front.json"
    assert first_record.is_file()
    original = first_record.read_bytes()
    monkeypatch.setattr(manager, "_save", save_checkpoint)
    retried = manager.apply("batch", job["id"])
    assert retried["saved"] == 3 and retried["failed"] == 0
    assert first_record.read_bytes() == original


def test_invalid_scope_engine_and_unavailable_runtime_are_clear_errors(calibration_api, monkeypatch):
    client, _, _ = calibration_api
    assert client.get(ROUTE.replace("/batch/", "/absent/")).status_code == 404
    assert client.post(ROUTE + "/jobs", json={"engine": "unknown"}).status_code == 422
    monkeypatch.setattr(batch, "engines", lambda: [{"id": "koide", "available": False, "detail": "Koide unavailable"}])
    response = client.post(ROUTE + "/jobs", json={"engine": "koide"})
    assert response.status_code == 422 and "Koide unavailable" in response.json()["detail"]


def sam2_job(store):
    clip = store.source_clip('batch')
    proposal = store.annotations('batch', 'f0').model_dump()
    proposal['image_polygons'] = [
        {'id': 'stripe', 'camera_id': 'front', 'label': 'lane', 'points': [[10, 20], [85, 20], [85, 24], [10, 24]]},
        {'id': 'road', 'camera_id': 'front', 'label': 'lane', 'points': [[0, 30], [99, 30], [99, 99], [0, 99]]},
        {'id': 'side', 'camera_id': 'side', 'label': 'lane', 'points': [[5, 5], [10, 5], [10, 10]]},
    ]
    value = {'id': 'sam-mask', 'clip_id': 'batch', 'kind': 'prelabel', 'status': 'completed',
             'parameters': {'model_id': 'sam2'}, 'result': {'model_id': 'sam2', 'frame_id': 'f0',
                'source_camera_id': 'front', 'source_image_token': batch.mask_image_token(store, clip.frames[0], clip.cameras[0]),
                'sam2_candidate_ids': ['stripe', 'road', 'side'], 'proposal': proposal}}
    store.save_job(value)
    return value


def test_selected_sam2_regions_create_original_size_binary_mask_only(calibration_api):
    client, store, manager = calibration_api
    sam2_job(store)
    original = store.annotations('batch', 'f0').model_dump()
    response = client.post(ROUTE + '/masks/f0/front/from-sam2', json={'job_id': 'sam-mask', 'polygon_ids': ['stripe']})
    assert response.status_code == 200, response.text
    assert response.json()['selected_count'] == 1
    path = manager._mask_path('batch', 'f0', 'front')
    with Image.open(path) as mask:
        assert mask.size == (100, 100)
        pixels = np.asarray(mask.convert('L'))
    assert set(np.unique(pixels)) == {0, 255}
    assert pixels[22, 40] == 255 and pixels[60, 40] == 0
    assert store.annotations('batch', 'f0').model_dump() == original
    assert not manager._mask_path('batch', 'f1', 'front').exists()
    assert not manager._mask_path('batch', 'f0', 'side').exists()
    assert json.loads(path.with_suffix('.json').read_text())['source'] == 'sam2'
    clip = store.source_clip('batch')
    assert manager._pair_fingerprint('batch', clip.frames[0], clip.cameras[0], 'pjlab')


@pytest.mark.parametrize('ids', [[], ['missing'], ['stripe', 'stripe'], ['side']])
def test_sam2_mask_rejects_empty_unknown_duplicate_and_other_camera(calibration_api, ids):
    client, store, manager = calibration_api
    sam2_job(store)
    response = client.post(ROUTE + '/masks/f0/front/from-sam2', json={'job_id': 'sam-mask', 'polygon_ids': ids})
    assert response.status_code == 422
    assert not manager._mask_path('batch', 'f0', 'front').exists()


@pytest.mark.parametrize('change', ['image', 'pair', 'clip', 'model', 'unfinished', 'busy', 'no_token'])
def test_sam2_mask_rejects_stale_or_incompatible_job(calibration_api, change):
    client, store, manager = calibration_api
    job = sam2_job(store)
    expected = 422
    if change == 'image':
        image = store.root / 'assets/batch/f0-front.png'
        image.write_bytes(image.read_bytes())
        expected = 409
    elif change == 'pair': job['result']['frame_id'] = 'f1'
    elif change == 'clip': job['clip_id'] = 'other'
    elif change == 'model': job['parameters']['model_id'] = 'latr'
    elif change == 'unfinished': job['status'] = 'running'
    elif change == 'no_token':
        del job['result']['source_image_token']
        expected = 409
    elif change == 'busy':
        manager._active.add('running'); expected = 409
    store.save_job(job)
    response = client.post(ROUTE + '/masks/f0/front/from-sam2', json={'job_id': 'sam-mask', 'polygon_ids': ['stripe']})
    assert response.status_code == expected, response.text
    assert not manager._mask_path('batch', 'f0', 'front').exists()
    manager._active.clear()


def test_sam2_job_records_source_version_and_new_regions_without_loading_cloud(calibration_api, monkeypatch):
    from omnilabel import inference
    client, store, _ = calibration_api
    monkeypatch.setenv('OMNILABEL_SAM2_URL', 'http://127.0.0.1:9101/predict')
    def worker(clip, frame, annotations, *args, **kwargs):
        proposed = json.loads(json.dumps(annotations))
        proposed['image_polygons'].append({'id': 'generated', 'camera_id': 'front', 'label': 'lane',
                                          'points': [[10, 20], [80, 20], [80, 23], [10, 23]]})
        return {'model_id': 'sam2', 'proposal': proposed}
    monkeypatch.setattr(inference, 'infer_proposal', worker)
    monkeypatch.setattr(store, 'materialize_frame', lambda *a: pytest.fail('Image-only SAM2 must not load LiDAR'))
    response = client.post('/api/clips/batch/jobs', json={'kind': 'prelabel', 'frame_id': 'f0', 'model_id': 'sam2',
                                                       'options': {'camera_id': 'front', 'label': 'lane'}})
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + response.json()['id']).json()
        if job['status'] not in ('queued', 'running'): break
        time.sleep(.01)
    assert job['status'] == 'completed', job
    assert job['result']['sam2_candidate_ids'] == ['generated']
    assert job['result']['source_image_token']
    saved = client.post(ROUTE + '/masks/f0/front/from-sam2', json={'job_id': job['id'], 'polygon_ids': ['generated']})
    assert saved.status_code == 200, saved.text


def test_rejected_candidate_preview_is_read_only_and_preserves_failed_quality(calibration_api, monkeypatch):
    client, store, _ = calibration_api
    def solve(*args, **kwargs):
        result = accepted(args[4], usable=False)
        result['matrix'][0][3] += 1
        result['rejection_reasons'] = ['修正超过 0.75 米或 3 度，请先手动校准初值并检查同步情况。']
        result['metrics'].update(line_overlap_before=.8321, line_overlap_after=.8841,
                                 translation_delta_m=1.1, rotation_delta_degrees=0)
        return result
    monkeypatch.setattr(batch, 'solve_pair', solve)
    original = client.get('/api/clips/batch/frames/f0/projection-calibration/front').json()
    labels = store.annotations('batch', 'f0').model_dump()
    job = finished(client, start(client, frame_ids=['f0'], camera_ids=['front'])['id'])
    assert job['rejected'] == 1
    path = ROUTE + f"/jobs/{job['id']}/preview/f0/front"
    response = client.get(path)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value['read_only'] and not value['usable'] and value['candidate_status'] == 'rejected'
    assert value['candidate_matrix'] == job['items'][0]['matrix']
    assert value['comparison_matrix'] == original['effective_matrix']
    assert value['comparison_source'] == 'solver_initial'
    assert value['candidate_matrix'] != value['comparison_matrix']
    assert value['metrics']['line_overlap_after'] == .8841
    assert value['rejection_reasons']
    assert client.get('/api/clips/batch/frames/f0/projection-calibration/front').json() == original
    assert store.annotations('batch', 'f0').model_dump() == labels
    # Preview never converts a rejected result into a publishable one.
    saved = client.post(ROUTE + f"/jobs/{job['id']}/apply").json()
    assert saved['saved'] == 0 and saved['rejected'] == 1
    assert client.get('/api/clips/batch/frames/f0/projection-calibration/front').json() == original


@pytest.mark.parametrize('change', ['image', 'projection', 'missing_pair', 'other_clip', 'no_matrix', 'invalid_matrix', 'running'])
def test_candidate_preview_rejects_stale_or_invalid_context(calibration_api, change):
    client, store, manager = calibration_api
    job = finished(client, start(client, frame_ids=['f0'], camera_ids=['front'])['id'])
    path = ROUTE + f"/jobs/{job['id']}/preview/f0/front"
    expected = 409
    if change == 'image':
        image = store.root / 'assets/batch/f0-front.png'
        image.write_bytes(image.read_bytes())
    elif change == 'projection':
        route = '/api/clips/batch/frames/f0/projection-calibration/front'
        doc = client.get(route).json()
        matrix = np.asarray(doc['effective_matrix']); matrix[0, 3] += .01
        assert client.put(route, json={'revision': doc['revision'], 'matrix': matrix.tolist(), 'method': 'manual'}).status_code == 200
    elif change == 'missing_pair': path = ROUTE + f"/jobs/{job['id']}/preview/f0/side"; expected = 404
    elif change == 'other_clip': path = '/api/clips/other/automatic-calibration' + f"/jobs/{job['id']}/preview/f0/front"; expected = 404
    elif change == 'no_matrix': job['items'][0]['matrix'] = None; expected = 422
    elif change == 'invalid_matrix': job['items'][0]['matrix'] = [[0]*4]*4; expected = 422
    elif change == 'running': job['status'] = 'running'
    atomic_json(manager._path(job['id']), job)
    response = client.get(path)
    assert response.status_code == expected, response.text


def test_saved_preview_compares_to_original_solver_initial(calibration_api):
    client, _, _ = calibration_api
    job = finished(client, start(client, frame_ids=['f0'], camera_ids=['front'])['id'])
    assert client.post(ROUTE + f"/jobs/{job['id']}/apply").json()['saved'] == 1
    value = client.get(ROUTE + f"/jobs/{job['id']}/preview/f0/front").json()
    assert value['candidate_status'] == 'saved' and value['usable']
    assert value['candidate_matrix'] != value['comparison_matrix']
    assert value['candidate_matrix'] == value['saved_matrix']


def test_older_task_without_initial_matrix_can_preview_current_saved_baseline(calibration_api):
    client, _, manager = calibration_api
    job = finished(client, start(client, frame_ids=['f0'], camera_ids=['front'])['id'])
    del job['items'][0]['initial_matrix']
    atomic_json(manager._path(job['id']), job)
    response = client.get(ROUTE + f"/jobs/{job['id']}/preview/f0/front")
    assert response.status_code == 200, response.text
    value = response.json()
    assert value['comparison_source'] == 'current_saved'
    assert value['comparison_matrix'] == value['effective_matrix']


def test_reviewed_rejected_candidate_requires_ack_and_preserves_automatic_verdict(calibration_api, monkeypatch):
    client, store, _ = calibration_api
    def solve(*args, **kwargs):
        result = accepted(args[4], usable=False)
        result['matrix'][0][3] += 1
        result['rejection_reasons'] = ['修正超过 0.75 米或 3 度']
        result['metrics'].update(translation_delta_m=1.1, line_overlap_before=.8321, line_overlap_after=.8841)
        return result
    monkeypatch.setattr(batch, 'solve_pair', solve)
    projection = '/api/clips/batch/frames/f0/projection-calibration/front'
    other = '/api/clips/batch/frames/f1/projection-calibration/front'
    side = '/api/clips/batch/frames/f0/projection-calibration/side'
    original = client.get(projection).json()
    unrelated = [client.get(other).json(), client.get(side).json(), store.annotations('batch', 'f0').model_dump()]
    job = finished(client, start(client, frame_ids=['f0'], camera_ids=['front'])['id'])
    route = ROUTE + f"/jobs/{job['id']}/apply-reviewed/f0/front"
    assert client.post(route, json={'revision': original['revision']}).status_code == 422
    assert client.get(projection).json() == original
    assert client.post(route, json={'revision': 'stale', 'acknowledge_rejected': True}).status_code == 409
    response = client.post(route, json={'revision': original['revision'], 'acknowledge_rejected': True})
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved['saved_matrix'] == job['items'][0]['matrix'] and saved['method'] == 'manual'
    assert saved['metrics']['validation'] == 'manual_review' and saved['metrics']['manual_override']
    audit = saved['metrics']['automatic_review']
    assert audit['job_id'] == job['id'] and audit['automatic_status'] == 'rejected'
    assert audit['automatic_usable'] is False and audit['rejection_reasons'] == ['修正超过 0.75 米或 3 度']
    assert audit['automatic_metrics']['line_overlap_after'] == .8841 and audit['confirmed_at']
    updated_job = client.get(ROUTE + f"/jobs/{job['id']}").json()
    assert updated_job['rejected'] == 1 and updated_job['accepted'] == 0 and updated_job['saved'] == 0
    assert updated_job['items'][0]['manual_applied']
    preview = client.get(ROUTE + f"/jobs/{job['id']}/preview/f0/front").json()
    assert not preview['usable'] and preview['manual_applied'] and preview['already_applied']
    # Refresh and repeated application cannot silently replace a newer revision.
    assert client.post(route, json={'revision': original['revision'], 'acknowledge_rejected': True}).status_code == 409
    assert client.get(projection).json() == saved
    assert client.get(other).json() == unrelated[0] and client.get(side).json() == unrelated[1]
    assert store.annotations('batch', 'f0').model_dump() == unrelated[2]
    assert store.clip('batch').frames[0].projection_camera_from_lidar['front'] == saved['saved_matrix']


def test_reviewed_accepted_candidate_applies_only_one_pair(calibration_api):
    client, _, _ = calibration_api
    job = finished(client, start(client, frame_ids=['f0', 'f1'], camera_ids=['front'])['id'])
    assert job['accepted'] == 2
    doc = client.get('/api/clips/batch/frames/f0/projection-calibration/front').json()
    route = ROUTE + f"/jobs/{job['id']}/apply-reviewed/f0/front"
    response = client.post(route, json={'revision': doc['revision']})
    assert response.status_code == 200, response.text
    assert response.json()['saved_matrix'] == job['items'][0]['matrix']
    assert response.json()['method'] == 'koide'
    assert not response.json()['metrics'].get('manual_override')
    current = client.get(ROUTE + f"/jobs/{job['id']}").json()
    assert current['saved'] == 1 and current['accepted'] == 1
    assert client.get('/api/clips/batch/frames/f1/projection-calibration/front').json()['saved_matrix'] is None


@pytest.mark.parametrize('change', ['image', 'busy', 'matrix_injection'])
def test_reviewed_apply_cannot_bypass_source_or_operation_checks(calibration_api, change):
    client, store, manager = calibration_api
    job = finished(client, start(client, frame_ids=['f0'], camera_ids=['front'])['id'])
    doc = client.get('/api/clips/batch/frames/f0/projection-calibration/front').json()
    payload = {'revision': doc['revision'], 'acknowledge_rejected': True}
    expected = 409
    if change == 'image':
        image = store.root / 'assets/batch/f0-front.png'; image.write_bytes(image.read_bytes())
    elif change == 'busy': manager._active.add('running')
    else: payload['matrix'] = np.eye(4).tolist(); expected = 422
    response = client.post(ROUTE + f"/jobs/{job['id']}/apply-reviewed/f0/front", json=payload)
    assert response.status_code == expected, response.text
    assert client.get('/api/clips/batch/frames/f0/projection-calibration/front').json()['saved_matrix'] is None
    manager._active.clear()
