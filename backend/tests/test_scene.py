from __future__ import annotations

from copy import deepcopy
import json
from math import pi
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from omnilabel.main import create_app
from omnilabel.models import Annotations, Clip
from omnilabel.scene import (
    evaluate_scene, frame_annotations, get_scene, get_scene_map, initialize_scene,
    reconstruct_scene, registered_poses, restore_scene, sample_track,
    save_reconstruction, save_scene, scene_export, source_fingerprint,
    source_poses, validate_scene_import,
)
from omnilabel.scene_models import ReconstructionRequest, SceneAnnotations
from omnilabel.storage import ConflictError, Store


def clip_data(points=None):
    identity = np.eye(4).tolist()
    frames = []
    for i, timestamp in enumerate([0, 1, 4]):
        pose = np.eye(4)
        pose[0, 3] = i * 2
        frames.append({"id": f"f{i}", "timestamp": timestamp, "ego_to_world": pose.tolist(),
                       "lidar_to_ego": identity, "points": points if points is not None else [[10-i*2, 0, -1, .5], [11-i*2, 1, -1, .6]],
                       "images": {}})
    return {"id": "scene-test", "name": "Scene test", "fps": 10, "frames": frames,
            "cameras": [{"id": "front", "name": "Front", "width": 100, "height": 100,
                         "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]],
                         "camera_from_ego": identity}]}


def box(center, track="vehicle-1", yaw=0):
    return {"id": "box-" + track, "track_id": track, "label": "car", "center": center,
            "size": [4, 2, 2], "yaw": yaw}


def scene_state():
    return SceneAnnotations.model_validate({"revision": 0, "vectors": [
        {"id": "lane", "label": "lane", "closed": False, "points": [[0, 2, -1], [10, 2, -1]]}],
        "tracks": [{"id": "track-1", "label": "car", "mode": "dynamic", "keyframes": [
            {"frame_id": "f0", "center": [0, 0, 0], "size": [2, 2, 2], "yaw": 170*pi/180},
            {"frame_id": "f2", "center": [8, 0, 0], "size": [4, 2, 2], "yaw": -170*pi/180}]}]})


def test_legacy_migration_creates_one_world_vector_and_temporal_tracks_without_mutation():
    clip = Clip.model_validate(clip_data())
    legacy = {}
    for i, frame in enumerate(clip.frames):
        legacy[frame.id] = Annotations.model_validate({"boxes": [box([10-i*2, 0, 0])], "vectors": [
            {"id": "lane", "label": "lane", "closed": False, "points": [[-i*2, 2, -1], [10-i*2, 2, -1]]}]}).model_dump()
    before = deepcopy(legacy)
    scene = initialize_scene(clip, legacy, source_poses(clip))
    assert legacy == before
    assert len(scene.vectors) == 1
    assert len(scene.tracks) == 1
    assert scene.tracks[0].mode == "static"
    assert len(scene.tracks[0].keyframes) == 3
    assert all(np.allclose(k.center, [10, 0, 0]) for k in scene.tracks[0].keyframes)


def test_dynamic_interpolation_uses_real_time_and_shortest_yaw_not_frame_index():
    clip = Clip.model_validate(clip_data())
    track = scene_state().tracks[0].model_dump()
    middle = sample_track(track, clip, "f1")
    assert middle["center"] == pytest.approx([2, 0, 0])  # t=1 of four seconds
    assert middle["size"] == pytest.approx([2.5, 2, 2])
    assert middle["yaw"] == pytest.approx(175*pi/180)
    track["mode"] = "static"
    assert sample_track(track, clip, "f2")["center"] == pytest.approx([0, 0, 0])


def test_track_endpoint_hold_and_registered_world_to_lidar_transform_preserve_image_labels():
    clip = Clip.model_validate(clip_data())
    scene = scene_state()
    scene.tracks[0].keyframes = [scene.tracks[0].keyframes[1]]
    poses = source_poses(clip)
    poses["f1"] = np.array([[0, -1, 0, 5], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    legacy = Annotations.model_validate({"revision": 7, "point_labels": {"0": "road"}, "image_polygons": [
        {"id": "polygon", "camera_id": "front", "label": "road", "points": [[10,10],[20,10],[20,20]]}]})
    result = frame_annotations(scene, clip, clip.frames[1], poses, legacy)
    assert result.boxes[0].center == pytest.approx([0, -3, 0])
    assert result.boxes[0].yaw == pytest.approx(100*pi/180)
    assert result.vectors[0].points[0] == pytest.approx([2, 5, -1])
    assert result.revision == 7 and result.point_labels == {"0": "road"}
    assert result.image_polygons[0].id == "polygon"


def test_scene_revision_persistence_rejects_unknown_frame_and_nonfinite(tmp_path):
    clip = Clip.model_validate(clip_data())
    store = Store(tmp_path)
    store.add_clip(clip)
    original = get_scene(store, clip)
    saved = save_scene(store, clip, scene_state())
    assert original.revision == 0 and saved.revision == 1
    assert get_scene(Store(tmp_path), clip) == saved
    with pytest.raises(ConflictError):
        save_scene(store, clip, scene_state())
    invalid = scene_state().model_dump()
    invalid["tracks"][0]["keyframes"][0]["frame_id"] = "missing"
    with pytest.raises(ValueError, match="not present"):
        SceneAnnotations.model_validate(invalid).validate_for_clip(clip)
    invalid["tracks"][0]["keyframes"][0]["center"] = [float("nan"), 0, 0]
    with pytest.raises(ValidationError):
        SceneAnnotations.model_validate(invalid)


def test_pose_reconstruction_fuses_all_frames_and_keeps_source_poses_immutable():
    clip = Clip.model_validate(clip_data())
    before = clip.model_dump()
    result = reconstruct_scene(clip, {}, ReconstructionRequest(voxel_size=.25))
    assert clip.model_dump() == before
    assert result["coordinate_frame"] == "world" and result["frame_count"] == 3
    assert result["point_count"] == 2
    assert result["trajectory"][2]["position"] == [4, 0, 0]
    assert result["world_from_lidar"]["f2"][0][3] == 4
    assert result["bounds"] == {"min": [10, 0, -1], "max": [11, 1, -1]}


def test_icp_recovers_missing_translation_and_reports_actual_alignment_quality():
    rng = np.random.default_rng(62)
    world = rng.uniform([-6, -4, -2], [7, 5, 3], (1500, 3))
    data = clip_data()
    data["frames"] = data["frames"][:2]
    for frame in data["frames"]:
        frame["ego_to_world"] = np.eye(4).tolist()
    data["frames"][0]["points"] = world.tolist()
    data["frames"][1]["points"] = (world - [.35, -.18, .08]).tolist()
    clip = Clip.model_validate(data)
    result = reconstruct_scene(clip, {}, ReconstructionRequest(voxel_size=.2, registration="icp"))
    np.testing.assert_allclose(np.array(result["world_from_lidar"]["f1"])[:3, 3], [.35, -.18, .08], atol=.01)
    report = result["registration"]["frames"][0]
    assert report["fitness"] > .98 and report["rmse"] < .02
    assert result["registration"]["loop_closure"] is False
    assert clip.frames[1].ego_to_world[0][3] == 0


def test_icp_degenerate_or_nonoverlapping_scans_fail_instead_of_claiming_success():
    data = clip_data([[i, 0, 0] for i in range(50)])
    with pytest.raises(ValueError, match="non-collinear"):
        reconstruct_scene(Clip.model_validate(data), {}, ReconstructionRequest(registration="icp"))
    rng = np.random.default_rng(1)
    cloud = rng.normal(size=(300, 3))
    data = clip_data()
    data["frames"] = data["frames"][:2]
    data["frames"][0]["points"] = cloud.tolist()
    data["frames"][1]["points"] = (cloud + 100).tolist()
    with pytest.raises(ValueError, match="overlap"):
        reconstruct_scene(Clip.model_validate(data), {}, ReconstructionRequest(registration="icp"))


def test_static_tracks_are_not_removed_merely_because_vehicle_semantic_class():
    clip = Clip.model_validate(clip_data())
    annotations = {frame.id: {"boxes": [box([10-i*2, 0, -1])], "point_labels": {"0": "vehicle"},
                               "_static_track_ids": ["vehicle-1"]} for i, frame in enumerate(clip.frames)}
    result = reconstruct_scene(clip, annotations, ReconstructionRequest(voxel_size=.25))
    assert result["point_count"] == 2
    assert "vehicle" in result["point_labels"].values()
    assert all(not label.startswith("scene-static:") for label in result["point_labels"].values())
    result_all = reconstruct_scene(clip, annotations, ReconstructionRequest(voxel_size=.25, exclude_dynamic=False))
    assert result_all["point_count"] == 2 and "vehicle" in result_all["point_labels"].values()


def test_map_cache_invalidates_after_labels_change_but_registered_poses_survive(tmp_path):
    clip = Clip.model_validate(clip_data())
    store = Store(tmp_path)
    store.add_clip(clip)
    scene = get_scene(store, clip)
    old_poses = registered_poses(store, clip)
    result = reconstruct_scene(clip, {}, ReconstructionRequest())
    save_reconstruction(store, clip, result, source_fingerprint(store, clip), scene, old_poses)
    loaded = get_scene_map(store, clip)
    assert loaded["point_count"] == loaded["rendered_point_count"] == 2
    stored = json.loads((tmp_path / "scene_maps" / f"{clip.id}.json").read_text())
    assert "points" not in stored and stored["pointcloud_file"].endswith(".npy")
    preview = get_scene_map(store, clip, max_points=1)
    assert preview["point_count"] == 2 and preview["rendered_point_count"] == len(preview["points"]) == 1
    store.save_annotations(clip.id, "f0", Annotations(point_labels={"0": "road"}))
    with pytest.raises(KeyError, match="changed"):
        get_scene_map(store, clip)
    assert registered_poses(store, clip)["f2"][0, 3] == 4


def test_reconstruction_reads_lazy_frame_arrays(tmp_path):
    data = clip_data()
    point_root = tmp_path / "pointclouds" / "scene-test"
    point_root.mkdir(parents=True)
    for frame in data["frames"]:
        points = np.asarray(frame.pop("points"), dtype=float)
        path = point_root / f"{frame['id']}.npy"
        np.save(path, points, allow_pickle=False)
        frame["pointcloud_file"] = f"scene-test/{path.name}"
        frame["point_count"] = len(points)
    clip = Clip.model_validate(data)
    store = Store(tmp_path)
    frames = {frame.id: frame for frame in clip.frames}
    result = reconstruct_scene(
        clip,
        {},
        ReconstructionRequest(),
        point_loader=lambda value: store.frame_point_array(frames[value["id"]]),
    )
    assert result["input_point_count"] == 6
    assert result["point_count"] == 2


def test_old_fusion_cache_requires_refresh_without_losing_registration_or_labels(tmp_path):
    clip = Clip.model_validate(clip_data())
    store = Store(tmp_path)
    store.add_clip(clip)
    scene = save_scene(store, clip, scene_state())
    result = reconstruct_scene(clip, {}, ReconstructionRequest())
    assert result["preserve_detail"] and result["effective_voxel_size"] == .05
    save_reconstruction(store, clip, result, source_fingerprint(store, clip), scene, source_poses(clip))
    exported = scene_export(store, clip)
    assert exported["scene_map"]["point_reduction"] == "brightest-measured-point"
    path = tmp_path / "scene_maps" / f"{clip.id}.json"
    old = json.loads(path.read_text())
    old.pop("fusion_version")
    path.write_text(json.dumps(old))
    with pytest.raises(KeyError, match="profile changed"):
        get_scene_map(store, clip)
    assert get_scene(store, clip) == scene
    assert registered_poses(store, clip)["f2"][0, 3] == 4


def test_reconstruction_rejects_concurrent_annotation_changes(tmp_path):
    clip = Clip.model_validate(clip_data())
    store = Store(tmp_path)
    store.add_clip(clip)
    scene = get_scene(store, clip)
    fingerprint = source_fingerprint(store, clip)
    save_scene(store, clip, scene)
    result = reconstruct_scene(clip, {}, ReconstructionRequest())
    with pytest.raises(ConflictError, match="during reconstruction"):
        save_reconstruction(store, clip, result, fingerprint, scene, source_poses(clip))


def test_scene_export_restore_preserves_world_registration_and_legacy_labels(tmp_path):
    clip = Clip.model_validate(clip_data())
    source = Store(tmp_path / "source")
    source.add_clip(clip)
    scene = save_scene(source, clip, scene_state())
    poses = source_poses(clip)
    poses["f1"][0, 3] = 3.25
    registration = {"world_from_lidar": {fid: matrix.tolist() for fid, matrix in poses.items()}}
    restore_scene(source, clip, scene.model_dump(), scene_registration=registration)
    exported = scene_export(source, clip)
    target = Store(tmp_path / "target")
    target.add_clip(clip)
    restore_scene(target, clip, exported["scene_annotations"], scene_registration=exported["scene_registration"])
    assert get_scene(target, clip) == scene
    assert registered_poses(target, clip)["f1"][0, 3] == 3.25
    assert evaluate_scene(target, clip, "f1")["boxes"][0]["center"][0] == pytest.approx(-1.25)
    assert target.annotations(clip.id, "f1").boxes == []
    invalid = deepcopy(exported["scene_registration"])
    del invalid["world_from_lidar"]["f0"]
    with pytest.raises(ValueError, match="exactly"):
        validate_scene_import(clip, scene.model_dump(), scene_registration=invalid)


def test_scene_http_revision_evaluation_reconstruction_and_stale_cache(tmp_path):
    data = clip_data()
    with TestClient(create_app(tmp_path / "http", seed_demo=False)) as client:
        response = client.post("/api/clips/import", files={"file": ("clip.json", json.dumps(data))})
        assert response.status_code == 201, response.text
        scene_path = "/api/clips/scene-test/scene/annotations"
        initial = client.get(scene_path)
        assert initial.status_code == 200 and initial.json()["coordinate_frame"] == "world"
        saved = client.put(scene_path, json=scene_state().model_dump())
        assert saved.status_code == 200 and saved.json()["revision"] == 1
        assert client.put(scene_path, json=scene_state().model_dump()).status_code == 409
        evaluated = client.get("/api/clips/scene-test/scene/frame/f1")
        assert evaluated.status_code == 200
        assert evaluated.json()["boxes"][0]["center"] == pytest.approx([0,0,0])
        submitted = client.post("/api/clips/scene-test/scene/reconstruct", json={"voxel_size": .25, "registration": "poses", "exclude_dynamic": False})
        assert submitted.status_code == 202
        for _ in range(100):
            job = client.get(f"/api/jobs/{submitted.json()['id']}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(.01)
        assert job["status"] == "completed", job
        cached = client.get("/api/clips/scene-test/scene/map")
        assert cached.status_code == 200 and cached.json()["frame_count"] == 3
        assert client.put(scene_path, json=saved.json()).status_code == 200
        assert client.get("/api/clips/scene-test/scene/map").status_code == 404


def test_reconstruction_route_retains_previously_registered_pose_coordinates(tmp_path):
    app = create_app(tmp_path / "retained", seed_demo=False)
    with TestClient(app) as client:
        assert client.post("/api/clips/import", files={"file": ("clip.json", json.dumps(clip_data()))}).status_code == 201
        clip = app.state.store.clip("scene-test")
        poses = source_poses(clip)
        poses["f1"][0, 3] = 3.25
        restore_scene(app.state.store, clip, SceneAnnotations().model_dump(),
                      scene_registration={"world_from_lidar": {fid: matrix.tolist() for fid, matrix in poses.items()}})
        submitted = client.post("/api/clips/scene-test/scene/reconstruct", json={"registration": "poses"})
        for _ in range(100):
            job = client.get(f"/api/jobs/{submitted.json()['id']}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(.01)
        assert job["status"] == "completed", job
        assert job["result"]["world_from_lidar"]["f1"][0][3] == 3.25
        assert registered_poses(app.state.store, clip)["f1"][0, 3] == 3.25


def test_multiscale_submap_registration_preserves_full_resolution_across_partial_scans():
    rng = np.random.default_rng(77)
    world = rng.uniform([-8, -5, -2], [8, 5, 3], (2500, 3))
    parts = [world[world[:, 0] < 3], world[world[:, 0] > -3], world]
    translations = [np.array([0, 0, 0]), np.array([.35, -.15, .04]), np.array([.7, -.3, .08])]
    data = clip_data()
    for index, frame in enumerate(data["frames"]):
        frame["ego_to_world"] = np.eye(4).tolist()
        frame["points"] = (parts[index] - translations[index]).tolist()
    result = reconstruct_scene(Clip.model_validate(data), {}, ReconstructionRequest(registration="icp", fusion_mode="full"))
    np.testing.assert_allclose(np.array(result["world_from_lidar"]["f2"])[:3, 3], translations[-1], atol=.01)
    assert result["point_count"] == result["input_point_count"] == sum(len(part) for part in parts)
    report = result["registration"]["frames"][-1]
    assert report["reference_frame_ids"] == ["f0", "f1"]
    assert len(report["stages"]) == 3 and report["fitness"] > .98 and report["rmse"] < .03


def test_full_map_page_binary_export_and_registered_pose_fallback_endpoint(tmp_path):
    app = create_app(tmp_path, seed_demo=False)
    with TestClient(app) as client:
        assert client.post("/api/clips/import", files={"file": ("clip.json", json.dumps(clip_data()))}).status_code == 201
        clip = app.state.store.clip("scene-test")
        scene = get_scene(app.state.store, clip)
        result = reconstruct_scene(clip, {}, ReconstructionRequest(fusion_mode="full"))
        save_reconstruction(app.state.store, clip, result, source_fingerprint(app.state.store, clip), scene, source_poses(clip))
        full = client.get("/api/clips/scene-test/scene/map").json()
        assert full["rendered_point_count"] == full["point_count"] == 6
        assert not full["preview_truncated"]
        first = client.get("/api/clips/scene-test/scene/map/points?offset=0&limit=4").json()
        second = client.get("/api/clips/scene-test/scene/map/points?offset=4&limit=4").json()
        assert first["points"] + second["points"] == full["points"]
        assert first["next_offset"] == 4 and second["next_offset"] is None
        assert not first["sampling"]
        import io
        exported = client.get("/api/clips/scene-test/scene/map.npy")
        np.testing.assert_array_equal(np.load(io.BytesIO(exported.content)), full["points"])
        # Label edits stale the map but do not invalidate the coordinate system.
        save_scene(app.state.store, clip, scene)
        assert client.get("/api/clips/scene-test/scene/map").status_code == 404
        transform = client.get("/api/clips/scene-test/scene/registration").json()["world_from_lidar"]
        assert transform["f2"][0][3] == 4
