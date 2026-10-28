import numpy as np
import pytest

import omnilabel.geometry as geometry

from omnilabel.geometry import (
    build_occupancy, fuse_map, project_points, propagate_annotations, segment_ground,
)


def frame(points, frame_id="f0", translation=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, 3] = translation
    return {"id": frame_id, "points": points, "ego_to_world": pose.tolist(),
            "lidar_to_ego": np.eye(4).tolist(), "images": {}, "timestamp": 0}


def annotations(**kwargs):
    return {"boxes": [], "vectors": [], "image_polygons": [], "point_labels": {}, "revision": 0, **kwargs}


def camera(**kwargs):
    return {"id": "front", "width": 100, "height": 100,
            "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]],
            "camera_from_ego": np.eye(4).tolist(), **kwargs}


def test_projection_positive_depth_pixel_bounds_stride_and_original_indices():
    cloud = frame([[0, 0, 2, 1], [0, 0, -2, 1], [2, 0, 2, 1], [100, 0, 1, 1]])
    result = project_points(cloud, camera(), annotations(point_labels={"2": "car"}))
    assert [(p["index"], p["u"], p["v"]) for p in result["points"]] == [(0, 50, 50), (2, 60, 50)]
    assert result["points"][1]["label"] == "car"
    assert [p["index"] for p in project_points(cloud, camera(), stride=2)["points"]] == [0, 2]


def test_projection_composes_lidar_extrinsic_and_radial_tangential_distortion():
    cloud = frame([[0, 1, 2]])
    cloud["lidar_to_ego"][0][3] = 1
    result = project_points(cloud, camera(distortion=[0.1, 0, 0.02, 0.03, 0]))
    # x=y=.5; radial=1.05; dx=.01+.03; dy=.02+.015.
    assert result["points"][0]["u"] == pytest.approx(55.65)
    assert result["points"][0]["v"] == pytest.approx(55.60)


def test_projection_uses_same_imported_frame_correction_as_annotations():
    cloud = frame([[0, 0, 2]])
    imported = np.eye(4); imported[0, 3] = 1
    cloud["annotation_camera_from_lidar"] = {"front": imported.tolist()}
    assert project_points(cloud, camera())["points"][0]["u"] == pytest.approx(55)
    published = np.eye(4); published[0, 3] = 2
    cloud["projection_camera_from_lidar"] = {"front": published.tolist()}
    assert project_points(cloud, camera())["points"][0]["u"] == pytest.approx(60)


def test_projection_empty_and_bad_transform():
    assert project_points(frame([]), camera())["points"] == []
    invalid = camera()
    invalid["camera_from_ego"][0][0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        project_points(frame([[1, 2, 3]]), invalid)


def test_world_map_uses_ego_and_lidar_transform_voxel_centroids_and_dynamic_exclusion():
    a = frame([[1, 0, 0, 0.2], [10, 0, 0, 0.8]], "a")
    a["lidar_to_ego"][0][3] = 1
    b = frame([[0.1, 0, 0, 0.6]], "b", (2, 0, 0))
    box = {"center": [10, 0, 0], "size": [2, 2, 2], "yaw": 0, "label": "car"}
    result = fuse_map({"frames": [a, b]}, {"a": annotations(boxes=[box], point_labels={"0": "road"})}, 0.5)
    assert result["excluded_dynamic_points"] == 1
    assert result["frame_count"] == 2
    assert result["coordinate_frame"] == "world"
    np.testing.assert_allclose(result["points"], [[2.05, 0, 0, 0.4]])
    assert result["point_labels"] == {"0": "road"}


def test_map_rejects_coordinates_that_would_overflow_voxel_indices():
    with pytest.raises(ValueError, match="safe voxel-index range"):
        fuse_map({"frames": [frame([[1e20, 0, 0], [-1e20, 0, 0]])]}, {}, 0.5)


def test_detail_fusion_keeps_measured_paint_reflectance_and_position_within_fine_voxel():
    points = [[.011, .011, .021, .2], [.031, .021, .022, .95], [.131, .021, .022, .25]]
    clip = {"frames": [frame(points)]}
    labels = {"f0": annotations(point_labels={"0": "road", "1": "lane_marking", "2": "road"})}
    detailed = fuse_map(clip, labels, .5, preserve_detail=True, detail_voxel_size=.05)
    np.testing.assert_allclose(detailed["points"], [points[1], points[2]])
    assert detailed["point_labels"] == {"0": "lane_marking", "1": "road"}
    assert detailed["effective_voxel_size"] == .05 and detailed["voxel_size"] == .5
    assert detailed["intensity_available"] and detailed["intensity_has_contrast"]
    assert detailed["intensity_range"] == [.2, .95]
    assert detailed["point_reduction"] == "brightest-measured-point"
    # Legacy callers still receive centroids and mean intensity.
    averaged = fuse_map(clip, labels, .5)
    np.testing.assert_allclose(averaged["points"], [np.mean(points, axis=0)])


def test_detail_fusion_xyz_only_never_invents_intensity_or_extra_points():
    source = [[.01, 0, 0], [.02, 0, 0], [.1, 0, 0]]
    result = fuse_map({"frames": [frame(source)]}, {}, .5, preserve_detail=True)
    assert not result["intensity_available"] and not result["intensity_has_contrast"]
    assert result["intensity_range"] == [0, 0]
    assert all(point[:3] in source and point[3] == 0 for point in result["points"])


def test_map_fusion_loads_external_frames_without_a_point_limit(monkeypatch):
    source = np.asarray([[0, 0, 0, .1], [.1, 0, 0, .2], [1, 0, 0, .3]], dtype=float)
    monkeypatch.setattr(geometry, "MAX_POINTS", 2)
    lazy_frame = frame([])
    lazy_frame["point_count"] = len(source)
    result = fuse_map(
        {"frames": [lazy_frame]},
        {},
        .05,
        point_loader=lambda _: source,
    )
    assert result["input_point_count"] == len(source)
    assert result["point_count"] == len(source)


def test_occupancy_negative_coordinates_free_rays_and_occupied_precedence():
    cloud = frame([[2.1, 0.1, 0.1], [1.1, 0.1, 0.1], [-2.1, 0.1, 0.1]])
    result = build_occupancy(cloud, annotations(point_labels={"0": "car"}), 1)
    cells = {tuple(cell["index"]): cell for cell in result["voxels"]}
    assert cells[(2, 0, 0)]["label"] == "car"
    assert cells[(1, 0, 0)]["state"] == "occupied"
    assert cells[(0, 0, 0)]["state"] == "free"
    assert cells[(-1, 0, 0)]["state"] == "free"
    assert cells[(-3, 0, 0)]["state"] == "occupied"
    assert (2, 2, 2) not in cells  # unobserved is unknown, never free
    assert not result["truncated"]


def test_occupancy_diagonal_ray_does_not_mark_touching_cells_free():
    result = build_occupancy(frame([[2.1, 2.1, 0.1]]), annotations(), 1)
    free = {tuple(cell["index"]) for cell in result["voxels"] if cell["state"] == "free"}
    assert (1, 1, 0) in free
    assert (1, 0, 0) not in free


def test_occupancy_semantic_box_fallback_and_out_of_range():
    box = {"center": [1, 0, 0], "size": [2, 2, 2], "yaw": 0, "label": "vehicle"}
    result = build_occupancy(frame([[1, 0, 0], [1000, 0, 0]]), annotations(boxes=[box]), 0.5)
    assert result["out_of_range_points"] == 1
    assert next(v for v in result["voxels"] if v["state"] == "occupied")["label"] == "vehicle"
    with pytest.raises(ValueError, match="voxel_size"):
        build_occupancy(frame([]), annotations(), 0)


def test_occupancy_budget_is_explicit_and_does_not_remove_occupied_endpoints(monkeypatch):
    monkeypatch.setattr(geometry, "MAX_RAY_VISITS", 1)
    result = build_occupancy(frame([[10, 0, 0], [20, 0, 0]]), annotations(), 1)
    assert result["truncated"]
    assert result["traced_rays"] == 0
    assert result["occupied_count"] == 2
    assert result["total_rays"] == 2


def test_ground_segmentation_preserves_manual_edits_and_separates_raised_points():
    ground = [[x, y, -1.5, 1] for x in range(-3, 4) for y in range(-3, 4)]
    cloud = frame(ground + [[0, 0, 0, 1]])
    source = annotations(point_labels={"0": "road"})
    result = segment_ground(cloud, source)
    assert result["point_labels"]["0"] == "road"
    assert result["point_labels"]["1"] == "ground"
    assert result["point_labels"][str(len(ground))] == "obstacle"
    assert source["point_labels"] == {"0": "road"}


def test_propagation_compensates_ego_motion_and_preserves_target_annotations():
    box = {"id": "b0", "track_id": "track-1", "label": "car", "center": [10, 0, 1],
           "size": [4, 2, 2], "yaw": 0}
    vector = {"id": "v0", "label": "lane", "points": [[0, 2, 0], [10, 2, 0]], "closed": False}
    source = annotations(boxes=[box], vectors=[vector], point_labels={"0": "car"})
    target = annotations(image_polygons=[{"id": "p0"}], point_labels={"5": "road"})
    result = propagate_annotations(frame([], "a"), frame([], "b", (2, 0, 0)), source, target)
    assert result["boxes"][0]["center"] == pytest.approx([8, 0, 1])
    assert result["boxes"][0]["track_id"] == "track-1"
    assert result["vectors"][0]["points"][0] == pytest.approx([-2, 2, 0])
    assert result["point_labels"] == {"5": "road"}
    assert result["image_polygons"] == [{"id": "p0"}]
    again = propagate_annotations(frame([], "a"), frame([], "b", (2, 0, 0)), source, result)
    assert len(again["boxes"]) == 1
    assert len(again["vectors"]) == 1
    assert target["boxes"] == []


def test_propagation_rotates_heading_and_centers():
    target = frame([], "b")
    target["ego_to_world"] = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    box = {"id": "b0", "track_id": "t", "label": "car", "center": [10, 0, 0], "size": [4, 2, 2], "yaw": 0}
    result = propagate_annotations(frame([], "a"), target, annotations(boxes=[box]))
    assert result["boxes"][0]["center"] == pytest.approx([0, -10, 0])
    assert result["boxes"][0]["yaw"] == pytest.approx(-np.pi / 2)


def test_propagation_vector_id_collision_with_other_annotation_type_is_safe_and_idempotent():
    source = annotations(vectors=[{"id": "shared", "label": "lane", "points": [[0, 0, 0], [1, 0, 0]], "closed": False}])
    target = annotations(image_polygons=[{"id": "shared"}])
    result = propagate_annotations(frame([], "a"), frame([], "b"), source, target)
    assert result["vectors"][0]["id"] != "shared"
    again = propagate_annotations(frame([], "a"), frame([], "b"), source, result)
    assert len(again["vectors"]) == 1


def test_full_fusion_retains_every_measured_point_without_voxel_or_count_truncation():
    source = [[.01, .01, .01, .2], [.011, .01, .01, .9], [.01, .01, .01, .2]]
    result = fuse_map({"frames": [frame(source)]}, {"f0": annotations(point_labels={"1": "lane"})},
                      1, fusion_mode="full")
    assert result["points"] == source
    assert result["point_count"] == result["input_point_count"] == 3
    assert result["point_labels"] == {"1": "lane"}
    assert result["effective_voxel_size"] == 0 and result["point_reduction"] == "all-measured-points"
    assert not result["input_truncated"] and not result["output_truncated"]


def test_surface_fusion_keeps_a_measured_surface_sample_instead_of_bright_outlier():
    source = [[.001, .001, .001, .2], [.003, .001, .002, .3], [.048, .049, .049, 9]]
    value = fuse_map({"frames": [frame(source)]}, {}, .5, fusion_mode="surface")
    assert value["points"] == [source[1]]
    assert value["point_reduction"] == "nearest-measured-surface-point"
    bright = fuse_map({"frames": [frame(source)]}, {}, .5, fusion_mode="detail")
    assert bright["points"] == [source[2]]


def test_chunk_boundaries_preserve_full_point_indices_and_centroid_weights():
    source = np.tile([.01, .01, .01, .2], (100001, 1))
    source[-1, 3] = .8
    value = {"frames": [frame([])]}
    labels = {"f0": annotations(point_labels={"100000": "lane"})}
    full = fuse_map(value, labels, fusion_mode="full", point_loader=lambda _: source, return_array=True)
    assert len(full["points"]) == len(source) and full["point_labels"] == {"100000": "lane"}
    np.testing.assert_array_equal(full["points"], source)
    average = fuse_map(value, {}, point_loader=lambda _: source)
    np.testing.assert_allclose(average["points"], [source.mean(axis=0)], atol=1e-10)


def test_explicit_outlier_filter_reports_removed_samples_and_is_off_by_default():
    rng = np.random.default_rng(81)
    road = np.column_stack((rng.normal(0, .5, (500, 2)), np.zeros(500), np.full(500, .3)))
    noisy = np.vstack((road, [[30, 40, 50, 1], [-30, -40, 60, 1]]))
    original = fuse_map({"frames": [frame(noisy)]}, {}, fusion_mode="full")
    filtered = fuse_map({"frames": [frame(noisy)]}, {}, fusion_mode="full", outlier_filter=True)
    assert original["point_count"] == len(noisy) and original["excluded_outlier_points"] == 0
    assert filtered["excluded_outlier_points"] == 2 and filtered["point_count"] == len(road)
    np.testing.assert_array_equal(filtered["points"], road)
