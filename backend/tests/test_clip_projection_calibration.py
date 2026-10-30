"""A camera's clip correction follows every image pose without rewriting data."""
from copy import deepcopy
import json
import os

import numpy as np
import pytest

from omnilabel.geometry import project_points
from omnilabel.calibration import calibration_document
from omnilabel.models import Clip
from omnilabel.projection_calibration import (
    Correspondence, ProjectionUpdate, _base_matrix, document_for, save_projection_calibration,
)
from omnilabel.storage import ConflictError, Store, atomic_json


LANDMARKS = np.array([
    [6, -3, 0], [6, 3, 0], [10, -4, 1], [10, 4, 1],
    [15, -5, 0], [15, 5, 2], [8, -2, 3], [8, 2, -.5],
    [20, 0, 1], [12, 0, 3], [18, 4, .5], [9, -4, 2],
], dtype=float)
POSES = [(0., 0., 0.), (1.4, .2, .08), (2.4, .5, .13)]


def adjustment(x=.12, y=-.06, z=.05, yaw=.02):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1.]])


@pytest.fixture
def projection_store(tmp_path):
    store = Store(tmp_path)
    mount = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
    frames = []
    for index, (x, y, yaw) in enumerate(POSES):
        c, s = np.cos(yaw), np.sin(yaw)
        ego = np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, 1.5], [0, 0, 0, 1.]])
        to_ego = np.linalg.inv(ego)
        # Importers may already provide a different reviewed correction for
        # each image. A new camera correction must retain those differences.
        source_correction = adjustment(.01 * index, .02 * index, .03 * index, 0)
        base = source_correction @ mount @ to_ego
        frame_id = f"f{index}"
        image = f"/api/assets/clip-projection/{frame_id}.jpg"
        asset = tmp_path / "assets" / "clip-projection" / f"{frame_id}.jpg"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(f"source-image-{index}".encode())
        frames.append({"id": frame_id, "timestamp": index * .1,
                       "ego_to_world": ego.tolist(), "lidar_to_ego": to_ego.tolist(),
                       "pointcloud_file": "clip-projection/map.npy", "point_count": len(LANDMARKS),
                       "images": {"front": image},
                       "annotation_camera_from_lidar": {"front": base.tolist()}})
    clip = Clip.model_validate({"id": "clip-projection", "name": "Moving supplied world map", "fps": 10,
        "source": "4d-clip-import", "frames": frames,
        "cameras": [{"id": "front", "name": "Front", "width": 640, "height": 360,
                     "intrinsics": [[300, 0, 320], [0, 300, 180], [0, 0, 1]],
                     "camera_from_ego": mount.tolist()}]})
    store.add_clip(clip, {})
    cloud = tmp_path / "pointclouds" / "clip-projection" / "map.npy"
    cloud.parent.mkdir(parents=True)
    np.save(cloud, np.column_stack((LANDMARKS, np.linspace(.2, .8, len(LANDMARKS)))))
    atomic_json(tmp_path / "scenes" / "clip-projection.json",
                {"revision": 2, "coordinate_frame": "world", "tracks": [],
                 "vectors": [{"id": "world-lane", "label": "lane", "closed": False,
                              "points": [[10., -1., 0.], [22., -1., 0.]]}]})
    return store


def pairs_for(matrix):
    optical = np.column_stack((LANDMARKS, np.ones(len(LANDMARKS)))) @ np.asarray(matrix).T
    pixels = optical[:, :2] / optical[:, 2:3] * 300 + [320, 180]
    return [Correspondence(id=f"p{index}", point=point.tolist(), pixel=pixel.tolist())
            for index, (point, pixel) in enumerate(zip(LANDMARKS, pixels))]


def base_for(store, frame_id):
    source = store.source_clip("clip-projection")
    frame = next(frame for frame in source.frames if frame.id == frame_id)
    return np.asarray(_base_matrix(frame, source.cameras[0]))


def publish(store, frame_id="f0", *, delta=None, scope="clip", revision=None):
    matrix = (adjustment() if delta is None else delta) @ base_for(store, frame_id)
    current = document_for(store, "clip-projection", frame_id, "front")
    return save_projection_calibration(store, "clip-projection", frame_id, "front",
        ProjectionUpdate(revision=revision or current["revision"], matrix=matrix.tolist(),
                         correspondences=pairs_for(matrix), scope=scope))


def reset(store, frame_id, scope):
    current = document_for(store, "clip-projection", frame_id, "front")
    return save_projection_calibration(store, "clip-projection", frame_id, "front",
        ProjectionUpdate(revision=current["revision"], matrix=None, scope=scope))


def source_snapshot(store):
    roots = ("clips", "annotations", "assets", "pointclouds", "scenes")
    return {str(path.relative_to(store.root)): path.read_bytes()
            for root in roots for path in (store.root / root).rglob("*") if path.is_file()}


def test_clip_correction_projects_world_lane_through_each_moving_camera(projection_store):
    store = projection_store
    unchanged = source_snapshot(store)
    delta = adjustment()
    saved = publish(store, delta=delta)
    assert saved["scope"] == "clip" and saved["clip_scope_active"]
    assert saved["clip_anchor_frame_id"] == "f0"
    assert not saved["inherited"]
    clip = store.clip("clip-projection")
    pixels = []
    for index, frame in enumerate(clip.frames):
        doc = document_for(store, clip.id, frame.id, "front")
        assert doc["scope"] == "clip" and doc["clip_scope_active"]
        assert doc["inherited"] == (index != 0)
        assert doc["saved_matrix"] is None
        expected_matrix = delta @ base_for(store, frame.id)
        np.testing.assert_allclose(doc["effective_matrix"], expected_matrix, atol=1e-12)
        np.testing.assert_allclose(frame.projection_camera_from_lidar["front"], expected_matrix, atol=1e-12)
        # Evaluate actual source/world coordinates through the shared point
        # projector. Expected pixels below use the independent vehicle pose.
        raw = frame.model_dump()
        raw["points"] = [[10., -1., 0., .5], [22., -1., 0., .7]]
        projected = project_points(raw, clip.cameras[0].model_dump())["points"]
        assert len(projected) == 2
        x, y, yaw = POSES[index]
        c, s = np.cos(yaw), np.sin(yaw)
        for point, result in zip(raw["points"], projected):
            dx, dy = point[0] - x, point[1] - y
            camera = np.array([s * dx - c * dy + .01 * index,
                               1.5 - point[2] + .02 * index,
                               c * dx + s * dy + .03 * index, 1.])
            corrected = delta @ camera
            expected = corrected[:2] / corrected[2] * 300 + [320, 180]
            np.testing.assert_allclose([result["u"], result["v"]], expected, atol=1e-9)
        pixels.append([[point["u"], point["v"]] for point in projected])
    assert not np.allclose(pixels[0], pixels[1])
    reloaded = Store(store.root)
    np.testing.assert_allclose(document_for(reloaded, clip.id, "f2", "front")["effective_matrix"],
                               delta @ base_for(store, "f2"), atol=1e-12)
    assert source_snapshot(store) == unchanged


def test_reviewed_local_targets_survive_and_newer_anchor_local_edit_can_override(projection_store):
    store = projection_store
    old_anchor = publish(store, "f0", scope="frame", delta=adjustment(.05, 0, 0, 0))
    reviewed = publish(store, "f1", scope="frame", delta=adjustment(-.1, .03, 0, 0))
    clip_anchor = publish(store, "f0", delta=adjustment(.2, -.02, .03, .01))
    assert clip_anchor["scope"] == "clip"
    assert not np.allclose(clip_anchor["effective_matrix"], old_anchor["effective_matrix"])
    target = document_for(store, "clip-projection", "f1", "front")
    assert target["scope"] == "frame"
    np.testing.assert_allclose(target["effective_matrix"], reviewed["effective_matrix"], atol=1e-12)
    assert document_for(store, "clip-projection", "f2", "front")["scope"] == "clip"
    later = publish(store, "f0", scope="frame", delta=adjustment(.07, -.08, 0, 0))
    assert later["scope"] == "frame"
    assert not np.allclose(later["effective_matrix"], clip_anchor["effective_matrix"])


def test_reset_applies_only_to_selected_scope(projection_store):
    store = projection_store
    publish(store)
    reviewed = publish(store, "f1", scope="frame", delta=adjustment(-.1, .03, 0, 0))
    fallback = reset(store, "f1", "frame")
    assert fallback["scope"] == "clip" and fallback["inherited"]
    assert not np.allclose(fallback["effective_matrix"], reviewed["effective_matrix"])
    reviewed = publish(store, "f1", scope="frame", delta=adjustment(-.1, .03, 0, 0))
    cleared = reset(store, "f0", "clip")
    assert not cleared["clip_scope_active"]
    assert document_for(store, "clip-projection", "f2", "front")["scope"] == "source"
    np.testing.assert_allclose(document_for(store, "clip-projection", "f2", "front")["effective_matrix"],
                               base_for(store, "f2"), atol=1e-12)
    np.testing.assert_allclose(document_for(store, "clip-projection", "f1", "front")["effective_matrix"],
                               reviewed["effective_matrix"], atol=1e-12)


@pytest.mark.parametrize("change", ["anchor_image", "pointcloud", "anchor_base", "target_pose", "camera_mount"])
def test_source_change_stales_clip_correction(projection_store, change):
    store = projection_store
    publish(store)
    if change in {"anchor_image", "pointcloud"}:
        path = (store.root / "assets" / "clip-projection" / "f0.jpg" if change == "anchor_image"
                else store.root / "pointclouds" / "clip-projection" / "map.npy")
        before = path.stat()
        if change == "anchor_image":
            path.write_bytes(b"replaced-source-image")
        else:
            points = np.load(path)
            points[0, 0] += .5
            np.save(path, points)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    else:
        path = store.root / "clips" / "clip-projection.json"
        raw = json.loads(path.read_text())
        if change == "anchor_base":
            raw["frames"][0]["annotation_camera_from_lidar"]["front"][0][3] += .02
        elif change == "target_pose":
            raw["frames"][2]["ego_to_world"][0][3] += .02
        else:
            raw["cameras"][0]["camera_from_ego"][0][3] += .02
        atomic_json(path, raw)
    doc = document_for(store, "clip-projection", "f2", "front")
    assert doc["clip_stale"] and not doc["clip_scope_active"]
    assert doc["scope"] == "source"
    np.testing.assert_allclose(doc["effective_matrix"], base_for(store, "f2"), atol=1e-12)
    assert not store.clip("clip-projection").frames[2].projection_camera_from_lidar


def test_clip_publication_conflicts_with_pending_other_frame_revision(projection_store):
    store = projection_store
    previous = document_for(store, "clip-projection", "f2", "front")
    publish(store)
    with pytest.raises(ConflictError):
        publish(store, "f2", scope="frame", revision=previous["revision"])


def test_saved_frame_matrix_can_be_published_for_clip_without_resolving(projection_store):
    store = projection_store
    saved = publish(store, "f0", scope="frame")
    reused = save_projection_calibration(store, "clip-projection", "f0", "front",
        ProjectionUpdate(revision=saved["revision"], matrix=saved["effective_matrix"],
                         correspondences=saved["correspondences"], method=saved["method"], scope="clip"))
    assert reused["scope"] == "clip" and reused["clip_scope_active"]
    np.testing.assert_allclose(document_for(store, "clip-projection", "f2", "front")["effective_matrix"],
                               adjustment() @ base_for(store, "f2"), atol=1e-12)


def test_physical_calibration_preview_receives_source_matrices_and_overlay_scopes(projection_store):
    store = projection_store
    source = store.source_clip("clip-projection")
    initial = calibration_document(store, source)
    assert initial["frame_projection_scopes"] == {
        "f0": {"front": "source"}, "f1": {"front": "source"}, "f2": {"front": "source"}}
    publish(store)
    reviewed = publish(store, "f1", scope="frame", delta=adjustment(-.1, .03, 0, 0))
    document = calibration_document(store, store.clip(source.id))
    assert document["frame_projection_scopes"] == {
        "f0": {"front": "clip"}, "f1": {"front": "frame"}, "f2": {"front": "clip"}}
    # Preview baselines are source parameters even when the caller holds an
    # effective clip containing both kinds of saved registration overlays.
    assert document["frame_projection_overrides"] == initial["frame_projection_overrides"]
    assert document["frame_projection_overrides"]["f1"]["projection_camera_from_lidar"] == {}
    assert not np.allclose(document["frame_projection_overrides"]["f1"]["annotation_camera_from_lidar"]["front"],
                           reviewed["effective_matrix"])
