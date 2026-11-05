"""Known-transform multi-frame recovery, independent holdout, and derivatives."""
from copy import deepcopy

import numpy as np
import pytest
from importlib.util import find_spec
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
_missing_optional = any(find_spec(name) is None for name in ('torch', 'yaml', 'matplotlib'))
pytestmark = pytest.mark.skipif(_missing_optional, reason='Install optional lane calibration dependencies to run legacy numerical tests.')
if not _missing_optional:
    import torch

from omnilabel.lane_calibration import OptimizerConfig
from omnilabel.lane_optimizer import (closest_segments, optimize, prepare, project, resample,
                                     se3_exp, statistics, write_outputs)


def known_problem():
    optical = np.eye(4); optical[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    optical[:3, 3] = [.15, -1.5, .2]
    perturb = np.eye(4); perturb[:3, :3] = Rotation.from_euler('xyz', [.3, -.5, .4], degrees=True).as_matrix()
    perturb[:3, 3] = [.08, -.06, .1]
    problem = {"clip_id": "known", "camera": {"id": "front", "width": 960, "height": 600,
        "intrinsics": [[600, 2, 480], [0, 605, 240], [0, 0, 1]], "distortion": [.015, -.002, .0002, -.0001, 0]},
        "initial_matrix": (optical @ perturb).tolist(), "mode": "multi",
        "config": OptimizerConfig(adam_iterations=180, lbfgs_iterations=40).model_dump(), "observations": [],
        "conventions": {"pose": "world_from_lidar", "extrinsic": "camera_from_lidar"}}
    origin = np.array([500000., 3500000., 10.])
    x = np.linspace(12, 65, 45)
    for frame_id in range(4):
        pose = np.eye(4); pose[:3, :3] = Rotation.from_euler('z', frame_id * .025).as_matrix()
        pose[:3, 3] = origin + [frame_id * 2, frame_id * .1, 0]
        for index, side in enumerate([-3., 2.7]):
            points = np.column_stack([x, side + .002 * (x - 25)**2, .015*x + .2*np.sin(x/15)]) + origin
            observation = {"id": f"p-{frame_id}-{index}", "frame_id": f"f{frame_id}", "lane_id": f"lane{index}",
                "world_points": points.tolist(), "world_from_lidar": pose.tolist(), "image_points": [[0, 0], [1, 1]],
                "world_range": [0, 1], "split": "validation" if frame_id == 3 else "train", "weight": 1}
            local = (points - pose[:3, 3]) @ pose[:3, :3]
            uv, _ = project(problem, {"local": local}, optical)
            observation["image_points"] = uv.tolist()
            problem["observations"].append(observation)
    return problem, optical


def test_segment_distance_is_not_equal_index_matching():
    points = np.array([[2, 3], [-2, 1], [12, 4]])
    line = np.array([[0, 0], [10, 0]])
    assert np.allclose(np.linalg.norm(closest_segments(points, line), axis=1), [3, np.sqrt(5), np.sqrt(20)])
    assert np.allclose(resample([[0, 0], [1, 0], [10, 0]], 3), [[0, 0], [5, 0], [10, 0]])


def test_se3_exponential_has_correct_zero_gradient_and_translation_coupling():
    xi = torch.tensor([.1, .2, .3, .01, -.02, .03], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(se3_exp, (xi,), eps=1e-6, atol=1e-5)
    zero = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(se3_exp, (zero,), eps=1e-6, atol=1e-5)
    matrix = se3_exp(xi).detach().numpy()
    assert np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3))
    assert not np.allclose(matrix[:3, 3], xi.detach().numpy()[:3])


def test_multiframe_recovery_and_holdout_are_both_improved(tmp_path):
    from PIL import Image
    problem, truth = known_problem()
    source = tmp_path / 'source.png'; Image.new('RGB', (960, 600), 'black').save(source)
    for pair in problem['observations']: pair['image_path'] = str(source)
    result = optimize(problem)
    print('recovery:',result['error_reduction'],result['observability'],result['rejection_reasons'])
    for split in ('train', 'validation'):
        assert result['after']['splits'][split]['mean_px'] < result['before']['splits'][split]['mean_px'] * .2
        assert result['after']['splits'][split]['mean_px'] < .5
    assert result['usable'], result['rejection_reasons']
    assert any(row['phase']=='lbfgs_evaluation' for row in result['history'])
    estimated = np.asarray(result['optimized_matrix'])
    assert Rotation.from_matrix(estimated[:3, :3] @ truth[:3, :3].T).magnitude() < np.deg2rad(.2)
    assert np.linalg.norm(estimated[:3, 3] - truth[:3, 3]) < .1
    output = tmp_path / 'result'; write_outputs(problem, result, output)
    assert np.allclose(np.load(output/'optimized_extrinsic.npy'), estimated)
    assert np.allclose(np.load(output/'initial_extrinsic.npy'), problem['initial_matrix'])
    for name in ['optimized_extrinsic.txt','optimization_log.csv','optimization_config.yaml','error_statistics.json','loss_curve.png']:
        assert (output/name).stat().st_size > 0
    assert len(list((output/'before_after').glob('*.png'))) == 8


def test_fixed_visibility_support_cannot_hide_errors():
    problem, _ = known_problem(); groups = prepare(problem)
    before = statistics(problem, groups, np.asarray(problem['initial_matrix']))
    broken = np.eye(4); broken[2, 3] = -1000
    after = statistics(problem, groups, broken)
    assert after['splits']['train']['mean_px'] > before['splits']['train']['mean_px']
    assert all(row['visibility_retained'] == 0 for row in after['pairs'])


def test_validation_frames_never_enter_training_objective():
    problem, _ = known_problem(); problem['config'].update(adam_iterations=12, lbfgs_iterations=0)
    alternate = deepcopy(problem)
    for observation in alternate['observations']:
        if observation['split'] == 'validation':
            observation['image_points'] = (np.asarray(observation['image_points']) + [40, -30]).tolist()
    a, b = optimize(problem), optimize(alternate)
    assert np.allclose(a['optimized_matrix'], b['optimized_matrix'], atol=1e-12)
    assert a['after']['splits']['validation']['mean_px'] != b['after']['splits']['validation']['mean_px']
