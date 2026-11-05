import numpy as np
import pytest

from omnilabel.lane_overlap import match_overlap
from omnilabel.projection_calibration import project_camera_points


def example(reverse_world=False, reverse_image=False):
    camera = {'width': 960, 'height': 600, 'intrinsics': [[600,0,480],[0,600,240],[0,0,1]], 'distortion': []}
    matrix = np.eye(4); matrix[:3,:3] = [[0,-1,0],[0,0,-1],[1,0,0]]
    origin = np.array([500000.,3500000.,140.])
    source = np.array([[1,2,-1.6],[301,2,-1.6]])
    image_world = np.column_stack([np.linspace(20,100,80), np.full(80,2), np.full(80,-1.6)])
    pixels,_ = project_camera_points(image_world, matrix, np.array(camera['intrinsics']), np.array([]))
    pose = np.eye(4); pose[:3,3] = origin
    observation = {'id': 'partial', 'world_points': (source[::-1] if reverse_world else source)+origin,
        'world_from_lidar': pose.tolist(), 'image_points': pixels[::-1] if reverse_image else pixels, 'world_range': [0,1]}
    return observation, camera, matrix


@pytest.mark.parametrize('reverse_world', [False,True])
@pytest.mark.parametrize('reverse_image', [False,True])
def test_300m_world_line_matches_only_observed_80m_with_perspective_and_utm(reverse_world, reverse_image):
    observation, camera, matrix = example(reverse_world, reverse_image)
    original = np.asarray(observation['world_points']).copy()
    result = match_overlap(observation, camera, matrix)
    assert result['total_length_m'] == pytest.approx(300)
    assert result['length_m'] == pytest.approx(80, abs=.01)
    assert result['start_m'] == pytest.approx(201 if reverse_world else 19, abs=.01)
    assert result['end_m'] == pytest.approx(281 if reverse_world else 99, abs=.01)
    assert result['p90_distance_px'] < 1e-6
    np.testing.assert_array_equal(observation['world_points'], original)


def test_unrelated_image_line_and_ambiguous_folded_map_are_rejected():
    observation, camera, matrix = example()
    observation['image_points'] = observation['image_points']+[100,0]
    with pytest.raises(ValueError, match='相距过大'): match_overlap(observation, camera, matrix, 2)
    observation, camera, matrix = example()
    source = observation['world_points']
    observation['world_points'] = np.concatenate([source, source[::-1][1:]])
    with pytest.raises(ValueError, match='歧义|往返'): match_overlap(observation, camera, matrix)


def test_tiny_or_invisible_image_constraint_cannot_create_a_fake_overlap():
    observation, camera, matrix = example()
    observation['image_points'] = [[400,300],[401,301]]
    with pytest.raises(ValueError, match='过短'): match_overlap(observation, camera, matrix)
    observation, camera, matrix = example()
    matrix[2,3] = -1000
    with pytest.raises(ValueError, match='可见'): match_overlap(observation, camera, matrix)


def test_matching_removes_false_error_from_the_unobserved_220m_tail():
    from omnilabel.lane_optimizer import prepare, statistics
    observation, camera, matrix = example()
    observation.update(frame_id='f0', lane_id='road', split='train', weight=1)
    problem = {'camera': camera, 'initial_matrix': matrix, 'config': {'samples_per_lane': 48}, 'observations': [observation]}
    before = statistics(problem, prepare(problem), matrix)['splits']['train']['mean_px']
    assert before > 1
    match = match_overlap(observation, camera, matrix)
    observation = {**observation, 'world_range': match['world_range']}
    matched = {**problem, 'observations': [observation]}
    after = statistics(matched, prepare(matched), matrix)['splits']['train']['mean_px']
    assert after < 1e-6


def test_curved_partial_lane_uses_the_whole_polyline_with_brown_distortion():
    observation, camera, matrix = example()
    camera['distortion'] = [.02,-.003,.0002,-.0001,0]
    def curve(x): return np.column_stack([x, 2+.0001*x*x, -1.6+.0002*x])
    source = curve(np.linspace(1,301,1001))
    observed = curve(np.linspace(20,100,201))
    observation['world_points'] = source+np.asarray(observation['world_from_lidar'])[:3,3]
    observation['image_points'],_ = project_camera_points(observed, matrix, np.asarray(camera['intrinsics']), np.asarray(camera['distortion']))
    result = match_overlap(observation, camera, matrix)
    assert result['length_m'] == pytest.approx(np.linalg.norm(np.diff(observed,axis=0),axis=1).sum(), abs=.2)
    assert 0 < result['world_range'][0] < result['world_range'][1] < .5
    assert result['p90_distance_px'] < .05
