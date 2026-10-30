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
