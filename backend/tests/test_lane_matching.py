"""Real SciPy solves for partial observations, independent pixels and weak geometry."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omnilabel.lane_calibration import OptimizerConfig
from omnilabel.lane_matching import optimize_matched, prepare_matched, preview_matched
from omnilabel.lane_optimizer import OptimizationCancelled
from omnilabel.projection_calibration import project_camera_points


def matching_problem(perturbed=True, straight=False):
    truth = np.array([[0., -1, 0, .1], [0, 0, -1, 1.5], [1, 0, 0, .2], [0, 0, 0, 1]])
    initial = truth.copy()
    if perturbed:
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_euler('xyz', [.4, -.5, .2], degrees=True).as_matrix()
        initial = delta @ truth
    camera = {'id': 'front', 'name': 'Front', 'width': 960, 'height': 600,
              'intrinsics': [[600, 2, 480], [0, 605, 240], [0, 0, 1]],
              'distortion': [.005, -.0002, .0001, -.0001, 0]}
    origin = np.array([500000., 3500000., 10.])
    pose = np.eye(4); pose[:3, 3] = origin
    x = np.linspace(8, 70, 311)
    observations = []
    for index, side in enumerate((-3., 2.7)):
        y = np.full_like(x, side) if straight else side + .002*(x-25)**2
        z = np.zeros_like(x) if straight else .01*x + .08*np.sin(x/13)
        local = np.column_stack((x, y, z))
        selected = local[(x >= 18) & (x <= 42)]
        pixels, _ = project_camera_points(selected, truth, np.array(camera['intrinsics']), np.array(camera['distortion']))
        observations.append({'id': f'p{index}', 'frame_id': 'f0', 'lane_id': f'lane{index}',
            'world_points': (local+origin).tolist(), 'world_from_lidar': pose.tolist(),
            'world_from_primary': pose.tolist(), 'world_range': [0, 1],
            'image_points': pixels.tolist(), 'image_size': [960, 600], 'weight': 1., 'split': 'train'})
    return {'clip_id': 'matching', 'camera': camera, 'initial_matrix': initial.tolist(),
            'mode': 'frame', 'observations': observations,
            'config': OptimizerConfig(adam_iterations=90, outer_iterations=5, dof_mode='rotation').model_dump(),
            'conventions': {'points': 'world', 'update': 'Exp(camera_xi) @ T0'}}, truth


def test_correct_pose_with_short_image_observation_does_not_invent_translation():
    problem, truth = matching_problem(False)
    unchanged = deepcopy(problem)
    preview = preview_matched(problem)
    assert all(.2 < row['matched_fraction'] < .6 for row in preview['matching']['pairs'])
    assert preview['statistics']['splits']['train']['mean_px'] < 1e-5
    result = optimize_matched(problem)
    np.testing.assert_allclose(result['optimized_matrix'], truth, atol=1e-8)
    assert np.linalg.norm(result['increment']) < 1e-8
    assert problem == unchanged


def test_partial_lane_matching_recovers_camera_rotation_and_retains_support():
    problem, truth = matching_problem()
    result = optimize_matched(problem)
    assert result['after']['splits']['train']['mean_px'] < .1
    assert result['after']['splits']['train']['mean_px'] < result['before']['splits']['train']['mean_px']*.05
    estimated = np.asarray(result['optimized_matrix'])
    assert Rotation.from_matrix(estimated[:3, :3] @ truth[:3, :3].T).magnitude() < np.deg2rad(.05)
    assert all(row['visibility_retained'] == 1 for row in result['after']['pairs'])
    assert all(row['matched_samples'] >= .8*before['matched_samples']
               for row, before in zip(result['matching']['pairs'], result['matching']['initial_pairs']))
    # Pure optical rotation preserves the physical camera center.
    np.testing.assert_allclose(-estimated[:3, :3].T @ estimated[:3, 3],
                               -np.asarray(problem['initial_matrix'])[:3, :3].T @ np.asarray(problem['initial_matrix'])[:3, 3], atol=1e-9)


def test_reverse_observation_and_utm_coordinates_remain_consistent():
    problem, _ = matching_problem()
    reverse = deepcopy(problem)
    for observation in reverse['observations']:
        observation['image_points'].reverse()
    a, b = optimize_matched(problem), optimize_matched(reverse)
    np.testing.assert_allclose(a['optimized_matrix'], b['optimized_matrix'], atol=1e-6)
    assert all(row['direction'] == 'reverse' for row in b['matching']['pairs'])


def test_diverse_geometry_recovers_all_six_dof_without_endpoint_anchors():
    problem, truth = matching_problem(False)
    origin = np.array([500000., 3500000., 10.])
    for index in range(2):
        values = np.linspace(-5, 6, 81) if index == 0 else np.linspace(0, 4, 81)
        local = np.column_stack((np.full_like(values, 24 if index == 0 else 32),
            values if index == 0 else np.full_like(values, 4), np.zeros_like(values) if index == 0 else values))
        pixels, _ = project_camera_points(local, truth, np.asarray(problem['camera']['intrinsics']),
                                          np.asarray(problem['camera']['distortion']))
        problem['observations'].append({**problem['observations'][0], 'id': f'cross{index}', 'lane_id': f'cross{index}',
            'world_points': (local+origin).tolist(), 'image_points': pixels.tolist()})
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler('xyz', [.35, -.4, .2], degrees=True).as_matrix()
    delta[:3, 3] = [.06, -.04, .08]
    problem['initial_matrix'] = (delta @ truth).tolist(); problem['config']['dof_mode'] = 'se3'
    result = optimize_matched(problem)
    estimated = np.asarray(result['optimized_matrix'])
    assert result['observability']['rank'] == 6
    assert result['after']['splits']['train']['mean_px'] < .01
    assert np.linalg.norm(estimated[:3, 3]-truth[:3, 3]) < .01
    assert Rotation.from_matrix(estimated[:3, :3] @ truth[:3, :3].T).magnitude() < np.deg2rad(.01)


def test_parallel_lines_do_not_manufacture_six_dof_from_unobserved_endpoints():
    problem, _ = matching_problem(False, straight=True)
    problem['camera']['distortion'] = []
    # Regenerate exact undistorted observations.
    for observation in problem['observations']:
        local = np.asarray(observation['world_points'])-np.array([500000., 3500000., 10.])
        observation['image_points'] = project_camera_points(local[50:171], np.asarray(problem['initial_matrix']),
            np.asarray(problem['camera']['intrinsics']), np.asarray([]))[0].tolist()
    problem['config']['dof_mode'] = 'auto'
    preview = preview_matched(problem)
    assert preview['observability']['rank'] < 6
    assert preview['observability']['effective_dof_mode'] == 'rotation'
    result = optimize_matched(problem)
    assert np.linalg.norm(result['increment'][:3]) == 0
    assert any('六自由度' in reason for reason in result['rejection_reasons'])


def test_validation_observations_never_change_fitted_matrix():
    problem, _ = matching_problem()
    validation = deepcopy(problem['observations'][0]); validation.update(id='holdout', frame_id='f1', split='validation')
    problem['observations'].append(validation)
    alternate = deepcopy(problem)
    alternate['observations'][-1]['image_points'] = (np.asarray(validation['image_points'])+[4, -3]).tolist()
    a, b = optimize_matched(problem), optimize_matched(alternate)
    np.testing.assert_allclose(a['optimized_matrix'], b['optimized_matrix'], atol=1e-10)
    assert a['after']['splits']['validation']['mean_px'] != pytest.approx(b['after']['splits']['validation']['mean_px'])


def test_unmatched_pairs_cancel_and_timeout_do_not_produce_success():
    problem, _ = matching_problem()
    bad = deepcopy(problem)
    bad['observations'][0]['image_points'] = (np.asarray(bad['observations'][0]['image_points'])+[250, 0]).tolist()
    with pytest.raises(ValueError, match='连续重合'):
        prepare_matched(bad)
    with pytest.raises(OptimizationCancelled):
        optimize_matched(problem, cancelled=lambda: True)
    problem['config']['timeout_seconds'] = 0
    with pytest.raises(ValueError, match='超时'):
        optimize_matched(problem)
