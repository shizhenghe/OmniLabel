"""Batch LATR uses one camera and durable proposals without editing labels."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from omnilabel import inference
from omnilabel.main import create_app
from omnilabel.models import Annotations, Clip
from omnilabel.scene import _clip_fingerprint
from omnilabel.storage import Store, atomic_json


ROUTE = "/api/clips/batch/latr-batches"
TERMINAL = {"completed", "cancelled", "failed"}


def prediction(frame, *, identifier="worker-lane", offset=0):
    return {"proposal": {"revision": 0, "boxes": [], "image_polygons": [], "point_labels": {},
                         "vectors": [{"id": identifier, "label": "lane", "closed": False,
                                      "points": [[offset + .01, 0, -.1], [offset + 5.01, 0, .1]]}]},
            "model_id": "latr", "model_version": "browser-independent-test-worker", "warnings": []}


@pytest.fixture
def latr_api(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNILABEL_LATR_URL", "http://127.0.0.1:9909/predict")
    identity = np.eye(4).tolist()
    cameras = [{"id": name, "name": name.title(), "width": 100, "height": 100,
                "intrinsics": [[50, 0, 50], [0, 50, 50], [0, 0, 1]],
                "camera_from_ego": identity} for name in ("front", "side")]
    frames = []
    for i in range(3):
        pose = np.eye(4); pose[:3, 3] = [500000 + i * 10, 3500000, 1.7]
        points = {"pointcloud_file": "batch/f0.npy", "point_count": 2} if not i else {
            "points": [[1, 0, 0, .2], [2, 1, 0, .8]]}
        frames.append({"id": f"f{i}", "timestamp": i * .1, "ego_to_world": pose.tolist(),
                       "lidar_to_ego": identity, **points,
                       "images": {camera: f"/api/assets/batch/f{i}-{camera}.png"
                                  for camera in (("front", "side") if i < 2 else ("side",))}})
    app = create_app(tmp_path, seed_demo=False)
    store = app.state.store
    clip = Clip.model_validate({"id": "batch", "name": "Batch lanes", "fps": 10,
                                "frames": frames, "cameras": cameras})
    store.add_clip(clip, {frame.id: Annotations(point_labels={"0": "road"}) for frame in clip.frames})
    store.add_clip(clip.model_copy(update={"id": "other"}))
    cloud = tmp_path / "pointclouds" / "batch" / "f0.npy"
    cloud.parent.mkdir(parents=True)
    np.save(cloud, np.asarray([[1, 0, 0, .2], [2, 1, 0, .8]], dtype=np.float32))
    for frame in clip.frames:
        for source in frame.images.values():
            path = tmp_path / "assets" / source.removeprefix("/api/assets/")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 100), (30, 60, 90)).save(path)
    # A registered map may differ from input ego poses; proposals must use the
    # actual scene coordinate system, retaining centimetres at UTM magnitudes.
    poses = {}
    for i, frame in enumerate(clip.frames):
        matrix = np.asarray(frame.ego_to_world).copy()
        matrix[0, 3] += 1000.125
        poses[frame.id] = matrix.tolist()
    atomic_json(tmp_path / "scene_poses" / "batch.json", {
        "clip_fingerprint": _clip_fingerprint(store, clip), "world_from_lidar": poses})
    calls = []

    def worker(clip_data, frame, annotations, model_id, prompt, **kwargs):
        calls.append({"frame": frame, "annotations": annotations, "model_id": model_id,
                      "prompt": prompt, **kwargs})
        return prediction(frame)

    monkeypatch.setattr(inference, "infer_proposal", worker)
    with TestClient(app) as client:
        yield client, store, app.state.latr_batches, calls, poses


def submit(client, **options):
    response = client.post(ROUTE, json={"camera_id": "front", "label": "lane",
                                      "score_threshold": .35, "weight_variant": "once", **options})
    assert response.status_code == 202, response.text
    return response.json()


def finished(client, job_id, route=ROUTE):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        response = client.get(f"{route}/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in TERMINAL:
            return job
        time.sleep(.01)
    pytest.fail("Batch LATR job did not finish")


def originals(store):
    folders = ("annotations", "scenes", "clips", "pointclouds", "assets", "projection_calibrations", "scene_poses")
    return {str(path.relative_to(store.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in folders for path in (store.root / folder).rglob("*") if path.is_file()}


def test_selected_camera_batch_uses_registered_world_coordinates_and_leaves_sources_untouched(latr_api, monkeypatch):
    client, store, manager, calls, poses = latr_api
    before = originals(store)
    monkeypatch.setattr(Store, "materialize_frame", lambda *args: pytest.fail("LATR must not materialize LiDAR"))
    monkeypatch.setattr(Store, "frame_point_array", lambda *args: pytest.fail("LATR must not load source pointclouds"))
    job = finished(client, submit(client, camera_id="side", frame_ids=["f0", "f1"])["id"])
    assert job["status"] == "completed", job
    assert (job["total"], job["processed"], job["succeeded"], job["failed"], job["skipped"], job["vector_count"]) == (2, 2, 2, 0, 0, 2)
    assert [call["frame"]["id"] for call in calls] == ["f0", "f1"]
    assert all(call["options"]["camera_id"] == "side" for call in calls)
    assert all(call["model_id"] == "latr" and call["frame"]["points"] == [] for call in calls)
    assert all(call["options"]["weight_variant"] == "once" and call["options"]["score_threshold"] == .35 for call in calls)
    response = client.get(f"{ROUTE}/{job['id']}/proposal")
    assert response.status_code == 200, response.text
    proposal = response.json()
    assert proposal["coordinate_frame"] == "world"
    assert proposal["source_frame_ids"] == ["f0", "f1"]
    assert len({v["id"] for v in proposal["vectors"]}) == 2
    for frame_id, vector in zip(proposal["source_frame_ids"], proposal["vectors"]):
        expected = np.asarray([.01, 0, -.1]) + np.asarray(poses[frame_id])[:3, 3]
        np.testing.assert_allclose(vector["points"][0], expected, atol=1e-8, rtol=0)
    assert originals(store) == before
    # A fresh reader sees durable job/result files, without a Python task reference.
    assert client.get(ROUTE).json()["latest_job"]["id"] == job["id"]
    assert manager.get("batch", job["id"])["vector_count"] == 2
    assert client.get(f"/api/clips/other/latr-batches/{job['id']}").status_code == 404


def test_partial_failure_and_missing_selected_image_preserve_successful_proposals(latr_api, monkeypatch):
    client, store, manager, calls, poses = latr_api
    attempted = []

    def worker(clip, frame, *args, **kwargs):
        attempted.append(frame["id"])
        if frame["id"] == "f1":
            raise inference.InferenceError("Fixture worker failed for this image")
        return prediction(frame)

    monkeypatch.setattr(inference, "infer_proposal", worker)
    before = originals(store)
    job = finished(client, submit(client)["id"])
    assert attempted == ["f0", "f1"]
    assert (job["succeeded"], job["failed"], job["skipped"], job["vector_count"]) == (1, 1, 1, 1)
    assert [item["status"] for item in job["items"]] == ["succeeded", "failed", "skipped"]
    response = client.get(f"{ROUTE}/{job['id']}/proposal")
    assert response.status_code == 200, response.text
    assert response.json()["source_frame_ids"] == ["f0"]
    assert len(response.json()["vectors"]) == 1
    assert response.json()["warnings"]
    assert originals(store) == before


def test_cancellation_stops_after_current_image_and_blocks_overlapping_batch(latr_api, monkeypatch):
    client, store, manager, calls, poses = latr_api
    entered, release = threading.Event(), threading.Event()
    attempted = []

    def worker(clip, frame, *args, **kwargs):
        attempted.append(frame["id"])
        entered.set()
        assert release.wait(4)
        return prediction(frame)

    monkeypatch.setattr(inference, "infer_proposal", worker)
    started = submit(client, camera_id="side")
    try:
        assert entered.wait(3)
        assert client.get(f"{ROUTE}/{started['id']}/proposal").status_code == 409
        assert client.post(ROUTE, json={"camera_id": "front"}).status_code == 409
        cancelled = client.post(f"{ROUTE}/{started['id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["cancel_requested"]
    finally:
        release.set()
    job = finished(client, started["id"])
    assert job["status"] == "cancelled"
    assert attempted == ["f0"]
    assert job["succeeded"] == 1
    response = client.get(f"{ROUTE}/{job['id']}/proposal")
    assert response.status_code == 200, response.text
    assert response.json()["source_frame_ids"] == ["f0"]


@pytest.mark.parametrize("change", ["scene", "frame", "image", "calibration", "registration"])
def test_saved_changes_invalidate_batch_proposal_before_application(latr_api, change):
    client, store, manager, calls, poses = latr_api
    job = finished(client, submit(client, frame_ids=["f0"])["id"])
    proposal_url = f"{ROUTE}/{job['id']}/proposal"
    assert client.get(proposal_url).status_code == 200
    if change == "scene":
        scene_url = "/api/clips/batch/scene/annotations"
        scene = client.get(scene_url).json()
        assert client.put(scene_url, json=scene).status_code == 200
    elif change == "frame":
        value = store.annotations("batch", "f0")
        store.save_annotations("batch", "f0", value)
    elif change == "image":
        path = store.root / "assets" / "batch" / "f0-front.png"
        stat = path.stat()
        Image.new("RGB", (100, 100), (60, 10, 50)).save(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    elif change == "calibration":
        path = store.root / "clips" / "batch.json"
        clip = json.loads(path.read_text())
        clip["cameras"][0]["camera_from_ego"][0][3] = .05
        atomic_json(path, clip)
    else:
        poses["f0"][0][3] += .1
        atomic_json(store.root / "scene_poses" / "batch.json", {
            "clip_fingerprint": _clip_fingerprint(store, store.clip("batch")), "world_from_lidar": poses})
    response = client.get(proposal_url)
    assert response.status_code == 409, response.text


@pytest.mark.parametrize("options", [
    {"camera_id": "missing"}, {"frame_ids": []}, {"frame_ids": ["missing"]},
    {"frame_ids": ["f0", "f0"]}, {"score_threshold": 2}, {"weight_variant": "bad"},
])
def test_batch_selection_validation_rejects_unusable_requests(latr_api, options):
    client, store, manager, calls, poses = latr_api
    response = client.post(ROUTE, json={"camera_id": "front", **options})
    assert response.status_code == 422, response.text
    assert calls == []


@pytest.mark.parametrize("published", [False, True])
def test_recovery_keeps_atomically_published_results_after_interrupted_batch(latr_api, published):
    client, store, manager, calls, poses = latr_api
    job = finished(client, submit(client, frame_ids=["f0", "f1"])["id"])
    job["status"] = "running"
    job["items"][1].update(status="running", vector_count=0)
    if not published:
        (manager.directory / job["id"] / "results" / "f1.json").unlink()
    atomic_json(manager.directory / f"{job['id']}.json", job)
    manager.recover()
    recovered = client.get(f"{ROUTE}/{job['id']}").json()
    assert recovered["status"] == "failed"
    assert recovered["succeeded"] == (2 if published else 1)
    assert recovered["items"][1]["status"] == ("succeeded" if published else "skipped")
    response = client.get(f"{ROUTE}/{job['id']}/proposal")
    assert response.status_code == 200, response.text
    assert response.json()["source_frame_ids"] == (["f0", "f1"] if published else ["f0"])


def test_retry_keeps_successful_frame_files_and_retries_failed_frames_in_same_job(latr_api, monkeypatch):
    client, store, manager, calls, poses = latr_api
    attempted = []

    def worker(clip, frame, *args, **kwargs):
        attempted.append(frame["id"])
        if frame["id"] == "f1" and attempted.count("f1") == 1:
            raise inference.InferenceError("Fixture: one transient failure")
        return prediction(frame)

    monkeypatch.setattr(inference, "infer_proposal", worker)
    before = originals(store)
    initial = finished(client, submit(client)["id"])
    first_result = manager.directory / initial["id"] / "results" / "f0.json"
    first_bytes = first_result.read_bytes()
    before_proposal = client.get(f"{ROUTE}/{initial['id']}/proposal").json()
    assert (initial["succeeded"], initial["failed"], initial["skipped"]) == (1, 1, 1)
    response = client.post(f"{ROUTE}/{initial['id']}/retry")
    assert response.status_code == 202, response.text
    assert response.json()["id"] == initial["id"]
    retried = finished(client, initial["id"])
    assert attempted == ["f0", "f1", "f1"]
    assert (retried["total"], retried["succeeded"], retried["failed"], retried["skipped"]) == (3, 2, 0, 1)
    assert first_result.read_bytes() == first_bytes
    proposal_response = client.get(f"{ROUTE}/{initial['id']}/proposal")
    assert proposal_response.status_code == 200, proposal_response.text
    proposal = proposal_response.json()
    assert proposal["source_frame_ids"] == ["f0", "f1"]
    assert proposal["vectors"][0] == before_proposal["vectors"][0]
    assert len({vector["id"] for vector in proposal["vectors"]}) == 2
    assert originals(store) == before


@pytest.mark.parametrize("changed", ["scene", "image"])
def test_retry_rejects_changed_context_without_replacing_previous_results(latr_api, monkeypatch, changed):
    client, store, manager, calls, poses = latr_api
    attempted = []

    def worker(clip, frame, *args, **kwargs):
        attempted.append(frame["id"])
        if frame["id"] == "f1":
            raise inference.InferenceError("Fixture failure")
        return prediction(frame)

    monkeypatch.setattr(inference, "infer_proposal", worker)
    job = finished(client, submit(client, frame_ids=["f0", "f1"])["id"])
    result_path = manager.directory / job["id"] / "results" / "f0.json"
    original_result = result_path.read_bytes()
    if changed == "scene":
        scene_path = "/api/clips/batch/scene/annotations"
        scene = client.get(scene_path).json()
        assert client.put(scene_path, json=scene).status_code == 200
    else:
        Image.new("RGB", (100, 100), (11, 80, 50)).save(store.root / "assets" / "batch" / "f1-front.png")
    response = client.post(f"{ROUTE}/{job['id']}/retry")
    assert response.status_code == 409, response.text
    assert attempted == ["f0", "f1"]
    assert result_path.read_bytes() == original_result


def test_fifty_frame_batch_then_manual_fusion_forms_one_complete_lane_without_early_annotation_writes(latr_api, monkeypatch):
    client, store, manager, calls, poses = latr_api
    source = store.clip("batch")
    frames = []
    for index in range(50):
        pose = np.eye(4)
        pose[:3, 3] = [500000 + index * .5, 3500000, 1.7]
        frames.append(source.frames[1].model_copy(update={"id": f"frame-{index:03}", "timestamp": index * .1,
                                                        "ego_to_world": pose.tolist()}))
    clip = source.model_copy(update={"id": "fifty", "frames": frames})
    store.add_clip(clip, {frame.id: Annotations(point_labels={"0": "road"}) for frame in frames})
    scene_path = "/api/clips/fifty/scene/annotations"
    scene = client.get(scene_path).json()
    scene["vectors"] = [{"id": "existing-map-area", "label": "crosswalk", "closed": True,
                         "points": [[500000, 3500003, 0], [500005, 3500003, 0], [500005, 3500005, 0]]}]
    saved = client.put(scene_path, json=scene).json()
    before = originals(store)
    attempted = []

    def worker(clip_data, frame, annotations, model_id, prompt, **options):
        attempted.append(frame["id"])
        assert model_id == "latr"
        assert options["options"]["camera_id"] == "front"
        assert options["options"]["weight_variant"] == "openlane"
        assert set(frame["images"]) == {"front"}
        result = prediction(frame, identifier="same-worker-id")
        result["proposal"]["vectors"][0]["points"] = [[.01, 0, -1.7], [5.01, 0, -1.7]]
        return result

    monkeypatch.setattr(inference, "infer_proposal", worker)
    monkeypatch.setattr(Store, "materialize_frame", lambda *args: pytest.fail("Image batch must not load a pointcloud"))
    route = "/api/clips/fifty/latr-batches"
    response = client.post(route, json={"camera_id": "front", "weight_variant": "openlane"})
    assert response.status_code == 202, response.text
    job = finished(client, response.json()["id"], route)
    assert (job["total"], job["succeeded"], job["failed"], job["vector_count"]) == (50, 50, 0, 50)
    assert attempted == [frame.id for frame in frames]
    proposed = client.get(f"{route}/{job['id']}/proposal").json()
    assert proposed["scene_revision"] == saved["revision"]
    assert len(proposed["vectors"]) == len({vector["id"] for vector in proposed["vectors"]}) == 50
    assert originals(store) == before
    fusion = client.post("/api/clips/fifty/lanes/merge-preview", json={
        "scene_revision": saved["revision"], "vectors": saved["vectors"] + proposed["vectors"]})
    assert fusion.status_code == 200, fusion.text
    merged = fusion.json()
    assert merged["report"]["merged_count"] == 49
    lanes = [vector for vector in merged["vectors"] if vector["label"] == "lane"]
    assert len(lanes) == 1
    assert lanes[0]["points"][0] == pytest.approx([500000.01, 3500000, 0], abs=1e-8)
    assert lanes[0]["points"][-1] == pytest.approx([500029.51, 3500000, 0], abs=1e-8)
    assert merged["vectors"][0] == saved["vectors"][0]
    assert originals(store) == before
    published = client.put(scene_path, json={**saved, "vectors": merged["vectors"]})
    assert published.status_code == 200, published.text
    after = originals(store)
    assert {name for name in before if before[name] != after[name]} == {"scenes/fifty.json"}
