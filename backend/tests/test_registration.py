"""Measured rigid-alignment checks, with noise and unmatched observations."""
import numpy as np
import pytest

from omnilabel.registration import register_multiscale, spatial_sample


def test_multiscale_registration_recovers_rigid_pose_with_noise_and_outliers_without_mutation():
    rng = np.random.default_rng(274)
    target = rng.uniform([-7, -5, -2], [8, 6, 4], (3000, 3))
    angle = .09
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    translation = np.array([.55, -.32, .14])
    source = (target - translation) @ rotation
    source += rng.normal(0, .002, source.shape)
    source = np.concatenate((source, rng.uniform([18, 18, 10], [30, 30, 20], (600, 3))))
    saved = source.copy()
    initial = np.eye(4)
    transform, report = register_multiscale(source, target, initial)
    np.testing.assert_allclose(transform[:3, :3], rotation, atol=.001)
    np.testing.assert_allclose(transform[:3, 3], translation, atol=.003)
    np.testing.assert_array_equal(source, saved)
    np.testing.assert_array_equal(initial, np.eye(4))
    assert report["fitness"] > .8 and report["rmse"] < .012
    assert len(report["stages"]) == 3
    assert report["correspondence_distance"] < report["stages"][0]["correspondence_distance"]


def test_registration_spatial_sample_does_not_depend_on_packet_order_or_take_a_prefix():
    xx, yy, zz = np.meshgrid(np.arange(20), np.arange(20), np.arange(4))
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(float)
    a = spatial_sample(points, .5, max_points=120)
    b = spatial_sample(points[np.random.default_rng(3).permutation(len(points))], .5, max_points=120)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 120
    assert np.ptp(a[:, 0]) > 17 and np.ptp(a[:, 1]) > 17 and np.ptp(a[:, 2]) == 3


def test_registration_rejects_nonoverlap_and_degenerate_geometry():
    cloud = np.random.default_rng(4).normal(size=(200, 3))
    with pytest.raises(ValueError, match="overlap"):
        register_multiscale(cloud + 100, cloud, np.eye(4))
    line = np.column_stack((np.arange(100), np.zeros(100), np.zeros(100)))
    with pytest.raises(ValueError, match="non-collinear"):
        register_multiscale(line, line, np.eye(4))
