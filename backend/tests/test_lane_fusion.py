"""Lane identity requires sustained geometry agreement, not just proximity."""
from __future__ import annotations

from copy import deepcopy

from fastapi.testclient import TestClient
import numpy as np
import pytest
from pydantic import ValidationError

from omnilabel.lane_fusion import LaneMergeRequest, merge_lanes
from omnilabel.main import create_app
from omnilabel.models import Clip


def lane(identifier, points, label="lane", closed=False):
    return {"id": identifier, "label": label, "points": points, "closed": closed}


def fuse(*vectors, **options):
    return merge_lanes(LaneMergeRequest(vectors=list(vectors), **options))


def test_reverse_partial_observation_extends_reviewed_lane_and_reports_lineage():
    reviewed = lane("reviewed", [[0, 0, -1], [7, .1, -.9], [15, 0, -.8]])
    predicted = lane("latr-f2", [[25, .08, -.65], [10, .08, -.85]])
    original = deepcopy([reviewed, predicted])
    result = fuse(reviewed, predicted)
    assert result.report.merged_count == 1
    assert result.report.groups[0].model_dump() == {"kept_id": "reviewed", "source_ids": ["reviewed", "latr-f2"]}
    assert np.asarray(result.vectors[0].points[:3]) == pytest.approx(np.asarray(reviewed["points"]))
    assert result.vectors[0].points[-1] == pytest.approx([25, .08, -.65])
    assert [reviewed, predicted] == original


def test_contained_duplicate_is_removed_and_larger_observation_can_extend_both_ends():
    reviewed = lane("reviewed", [[0, 0, 0], [10, 0, 0]])
    result = fuse(reviewed, lane("contained", [[2, .05, 0], [8, .05, 0]]),
                  lane("longer", [[-5, 0, 0], [15, 0, 0]]))
    assert len(result.vectors) == 1
    assert result.vectors[0].points == [(-5., 0., 0.), (0., 0., 0.), (10., 0., 0.), (15., 0., 0.)]
    assert result.report.merged_count == 2


def test_curved_lane_fusion_uses_local_correspondence_and_preserves_slopes_at_utm_origin():
    offset = np.asarray([520123.125, 3456789.5, 102.])
    # A long arc cannot be matched by sorting world X/Y or averaging endpoints.
    angle = np.linspace(0, 1.8, 30)
    points = np.column_stack([20 * np.sin(angle), 20 * (1 - np.cos(angle)), angle]) + offset
    result = fuse(lane("arc", points[:22].tolist()), lane("new", points[12:][::-1].tolist()))
    assert len(result.vectors) == 1
    combined = np.asarray(result.vectors[0].points)
    assert combined[0] == pytest.approx(points[0], abs=1e-8)
    assert combined[-1] == pytest.approx(points[-1], abs=1e-8)
    assert combined[:22] == pytest.approx(points[:22], abs=1e-8)


@pytest.mark.parametrize("other", [
    lane("parallel", [[0, .6, 0], [20, .6, 0]]),
    lane("crossing", [[10, -10, 0], [10, 10, 0]]),
    lane("overpass", [[0, 0, 2], [20, 0, 2]]),
    lane("branch", [[0, 0, 0], [5, 0, 0], [20, 2, 0]]),
    lane("solid", [[0, 0, 0], [20, 0, 0]], label="lane_solid"),
    lane("polygon", [[0, 0, 0], [20, 0, 0], [10, .1, 0]], closed=True),
    lane("next-dash", [[20.1, 0, 0], [30, 0, 0]]),
    lane("tiny-overlap", [[19.5, 0, 0], [30, 0, 0]]),
])
def test_distinct_lanes_junctions_classes_and_unobserved_gaps_remain_separate(other):
    result = fuse(lane("main", [[0, 0, 0], [20, 0, 0]]), other)
    assert result.report.merged_count == 0
    assert len(result.vectors) == 2


def test_overlapping_chain_becomes_one_complete_lane_without_bridging_missing_segments():
    result = fuse(lane("a", [[0, 0, 0], [10, 0, 0]]),
                  lane("b", [[18, .02, 0], [8, .02, 0]]),
                  lane("c", [[16, .02, 0], [25, .02, 0]]),
                  lane("gap", [[27, .02, 0], [35, .02, 0]]))
    assert len(result.vectors) == 2
    assert result.vectors[0].points[0][0] == 0
    assert result.vectors[0].points[-1][0] == 25
    assert result.report.groups[0].source_ids == ["a", "b", "c"]


def test_hundred_frame_predictions_collapse_into_two_long_separate_lanes():
    observations = []
    for frame in range(100):
        for side, y in enumerate([0., 3.5]):
            points = [[frame * 2, y + (frame % 3 - 1) * .02, 0],
                      [frame * 2 + 20, y + (frame % 3 - 1) * .02, 0]]
            observations.append(lane(f"f{frame}-{side}", points if frame % 2 else points[::-1]))
    result = fuse(*observations)
    assert result.report.merged_count == 198
    assert len(result.vectors) == 2
    assert all(len(group.source_ids) == 100 for group in result.report.groups)
    for vector in result.vectors:
        points = np.asarray(vector.points)
        assert points[:, 0].min() == 0
        assert points[:, 0].max() == 218
        assert np.max(np.abs(np.diff(points[:, 1]))) < .1


def test_latr_candidate_filter_does_not_fuse_unrelated_existing_manual_duplicates():
    vectors = [lane("manual-a", [[0, 0, 0], [10, 0, 0]]),
               lane("manual-b", [[0, .01, 0], [10, .01, 0]]),
               lane("new-a", [[20, 0, 0], [30, 0, 0]]),
               lane("old-far", [[28, .01, 0], [35, .01, 0]])]
    result = fuse(*vectors, candidate_ids=["new-a"])
    assert len(result.vectors) == 3
    assert result.report.groups[0].source_ids == ["new-a", "old-far"]
    assert [v.id for v in result.vectors][:2] == ["manual-a", "manual-b"]
    assert fuse(*vectors, candidate_ids=[]).report.merged_count == 0


def test_degenerate_repeated_vertices_do_not_generate_nan_or_destroy_input():
    result = fuse(lane("repeat", [[0, 0, 0], [0, 0, 0], [10, 0, 0]]),
                  lane("duplicate", [[0, 0, 0], [10, 0, 0]]),
                  lane("zero", [[3, 3, 0], [3, 3, 0]]))
    assert len(result.vectors) == 2
    assert np.isfinite(np.asarray(result.vectors[0].points)).all()
    assert result.vectors[1].id == "zero"


def test_request_rejects_duplicate_identifiers_and_unknown_candidate():
    vector = lane("same", [[0, 0, 0], [5, 0, 0]])
    with pytest.raises(ValidationError, match="IDs must be unique"):
        LaneMergeRequest(vectors=[vector, vector])
    with pytest.raises(ValidationError, match="candidate_ids"):
        LaneMergeRequest(vectors=[vector], candidate_ids=["missing"])


def test_preview_endpoint_never_saves_annotations_and_rejects_stale_scene_revision(tmp_path):
    identity = np.eye(4).tolist()
    clip = Clip.model_validate({"id": "test", "name": "test", "fps": 10,
                               "frames": [{"id": "f0", "timestamp": 0, "ego_to_world": identity,
                                           "lidar_to_ego": identity, "points": [], "images": {}}],
                               "cameras": [{"id": "front", "name": "front", "width": 100, "height": 100,
                                            "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]],
                                            "camera_from_ego": identity}]})
    app = create_app(tmp_path / "data", seed_demo=False)
    app.state.store.add_clip(clip)
    with TestClient(app) as client:
        initial_request = {"scene_revision": 0, "vectors": []}
        assert client.post("/api/clips/test/lanes/merge-preview", json=initial_request).status_code == 200
        assert not (app.state.store.root / "scenes").exists()
        scene_path = "/api/clips/test/scene/annotations"
        saved = client.get(scene_path).json()
        saved["vectors"] = [lane("saved", [[0, 0, 0], [10, 0, 0]])]
        saved = client.put(scene_path, json=saved).json()
        source_bytes = {path: path.read_bytes() for path in app.state.store.root.rglob("*.json")}
        request = {"scene_revision": saved["revision"], "vectors": saved["vectors"] + [
            lane("incoming", [[5, .05, 0], [15, .05, 0]])]}
        response = client.post("/api/clips/test/lanes/merge-preview", json=request)
        assert response.status_code == 200, response.text
        assert response.json()["report"]["merged_count"] == 1
        assert client.get(scene_path).json() == saved
        assert {path: path.read_bytes() for path in source_bytes} == source_bytes
        request["scene_revision"] -= 1
        assert client.post("/api/clips/test/lanes/merge-preview", json=request).status_code == 409
        request.pop("scene_revision")
        assert client.post("/api/clips/test/lanes/merge-preview", json=request).status_code == 200
        assert client.post("/api/clips/absent/lanes/merge-preview", json=request).status_code == 404
        request["max_distance"] = 50
        assert client.post("/api/clips/test/lanes/merge-preview", json=request).status_code == 422
