"""Real matching jobs, fixed image evidence and per-frame projection persistence."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from omnilabel.main import create_app
from omnilabel.storage import atomic_json
from omnilabel.models import Clip
from omnilabel.lane_calibration import projection_matrices
from test_lane_matching import matching_problem


@pytest.fixture
def matching_api(tmp_path):
    problem, truth = matching_problem()
    pose = np.asarray(problem['observations'][0]['world_from_lidar'])
    camera = {**problem['camera'], 'camera_from_ego': problem['initial_matrix']}
    frames = []
    for index in range(2):
        current = pose.copy(); current[0, 3] += index*2
        primary = np.linalg.inv(current)
        frames.append({'id': f'f{index}', 'timestamp': index*.1, 'ego_to_world': current.tolist(),
            'lidar_to_ego': primary.tolist(), 'sensor_lidar_to_ego': np.eye(4).tolist(),
            'points': [problem['observations'][0]['world_points'][75]+[.8]],
            'annotation_camera_from_lidar': {'front': (np.asarray(problem['initial_matrix']) @ primary).tolist()},
            'images': {'front': f'/api/assets/matching/f{index}.png'}})
    app = create_app(tmp_path, seed_demo=False); store = app.state.store
    store.add_clip(Clip.model_validate({'id': 'matching', 'name': 'World-map matching', 'fps': 10,
        'source': '4d-clip-import', 'cameras': [camera], 'frames': frames}))
    vectors = [{'id': observation['lane_id'], 'label': 'lane', 'closed': False,
                'points': observation['world_points']} for observation in problem['observations']]
    atomic_json(store.root/'scenes/matching.json', {'revision': 2, 'coordinate_frame': 'world', 'tracks': [], 'vectors': vectors})
    for frame in frames:
        path = store.root/'assets/matching'/f'{frame["id"]}.png'; path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (960, 600), '#203040').save(path)
    endpoint = '/api/clips/matching/lane-calibration/front'
    pairs = [{key: observation[key] for key in ('id', 'frame_id', 'lane_id', 'image_points', 'image_size', 'world_range', 'split', 'weight')}
             for observation in problem['observations']]
    with TestClient(app) as client:
        response = client.put(endpoint+'/pairs', json={'revision': 0, 'pairs': pairs})
        assert response.status_code == 200, response.text
        request = {'pairing_revision': 1, 'mode': 'frame', 'frame_ids': ['f0'], 'conventions_confirmed': True,
                   'time_alignment_confirmed': True, 'config': problem['config']}
        yield client, store, endpoint, request, truth


def finish(client, endpoint, request):
    response = client.post(endpoint+'/jobs', json=request)
    assert response.status_code == 202, response.text
    job_id = response.json()['id']
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        job = client.get(endpoint+'/jobs/'+job_id).json()
        if job['status'] not in ('queued', 'running'):
            assert job['status'] == 'completed', job.get('error')
            return job
        time.sleep(.02)
    pytest.fail('Matching job did not finish')


def test_matching_preview_is_non_mutating_and_reports_real_overlap(matching_api):
    client, store, endpoint, request, _ = matching_api
    files = [store.root/'clips/matching.json', store.root/'scenes/matching.json',
             store.root/'lane_calibration/matching/front/pairs.json']
    before = {path: path.read_bytes() for path in files}
    result = client.post(endpoint+'/match-preview', json=request)
    assert result.status_code == 200, result.text
    preview = result.json()
    assert preview['matching']['pair_count'] == 2
    assert all(.2 < pair['matched_fraction'] < .6 for pair in preview['matching']['pairs'])
    assert set(preview['projection_matrices']) == {'f0'}
    for path, content in before.items():
        assert path.read_bytes() == content
    assert not list((store.root/'projection_calibrations').glob('**/*.json'))


def test_real_matching_job_preview_apply_and_reload_are_frame_specific(matching_api):
    client, store, endpoint, request, truth = matching_api
    source = store.root/'clips/matching.json'; scene = store.root/'scenes/matching.json'
    source_before, scene_before = source.read_bytes(), scene.read_bytes()
    frames_before = client.get('/api/clips/matching').json()['frames']
    job = finish(client, endpoint, request)
    result = job['result']
    assert result['after']['splits']['train']['mean_px'] < .1
    assert result['after']['splits']['train']['mean_px'] < result['before']['splits']['train']['mean_px']*.05
    assert job['can_apply']
    # Candidate computation does not publish a calibration or change source documents.
    assert client.get('/api/clips/matching').json()['frames'] == frames_before
    pose = np.asarray(store.source_clip('matching').frames[0].ego_to_world)
    expected = np.asarray(result['optimized_matrix']) @ np.linalg.inv(pose)
    np.testing.assert_allclose(result['projection_matrices']['f0'], expected, atol=1e-9)
    applied = client.post(endpoint+'/jobs/'+job['id']+'/apply', json={'confirm_frame_update': True})
    assert applied.status_code == 200, applied.text
    assert applied.json()['applied']
    current = client.get('/api/clips/matching').json()['frames']
    np.testing.assert_allclose(current[0]['projection_camera_from_lidar']['front'], expected, atol=1e-9)
    assert current[1] == frames_before[1]
    assert source.read_bytes() == source_before and scene.read_bytes() == scene_before
    exported = store.root/'lane_calibration/matching/front/jobs'/job['id']
    for name in ('optimization_log.csv', 'loss_curve.png', 'optimization_config.yaml', 'error_statistics.json'):
        assert (exported/name).stat().st_size > 0
    assert Image.open(exported/'loss_curve.png').size[0] > 100


def test_source_edit_rejects_old_candidate_and_bad_match_does_not_publish(matching_api):
    client, store, endpoint, request, _ = matching_api
    job = finish(client, endpoint, request)
    scene = json.loads((store.root/'scenes/matching.json').read_text())
    scene['revision'] += 1; scene['vectors'][0]['points'][0][0] += .01
    atomic_json(store.root/'scenes/matching.json', scene)
    response = client.post(endpoint+'/jobs/'+job['id']+'/apply', json={'confirm_frame_update': True})
    assert response.status_code == 409, response.text
    assert not list((store.root/'projection_calibrations').glob('**/*.json'))
    pairing = client.get(endpoint).json()['pairing']
    pairing['pairs'][0]['image_points'] = (np.asarray(pairing['pairs'][0]['image_points'])+[250, 0]).tolist()
    saved = client.put(endpoint+'/pairs', json=pairing)
    assert saved.status_code == 200
    request['pairing_revision'] = saved.json()['revision']
    bad = client.post(endpoint+'/match-preview', json=request)
    assert bad.status_code == 422 and '连续重合' in bad.text


def test_preview_matrix_uses_declared_sensor_pose_and_original_primary_coordinates():
    source = np.eye(4); source[:3, 3] = [12, -4, 1]
    primary = np.eye(4); primary[:3, 3] = [500000, 3500000, 10]
    optical = np.array([[0, -1, 0, .1], [0, 0, -1, 1.5], [1, 0, 0, .2], [0, 0, 0, 1.]])
    problem = {'observations': [{'frame_id': 'f0', 'world_from_lidar': source.tolist(), 'world_from_primary': primary.tolist()}]}
    resolved = projection_matrices(problem, optical)['f0']
    point = np.array([1., 2, 3, 1])
    np.testing.assert_allclose(np.asarray(resolved) @ point,
                               optical @ np.linalg.inv(source) @ primary @ point)


def test_two_selected_cloud_lines_keep_independent_image_pixels_and_use_same_solver(matching_api):
    client, store, endpoint, request, truth = matching_api
    problem, _ = matching_problem()
    frame = store.source_clip('matching').frames[0]
    pose = np.asarray(frame.ego_to_world)
    initial = np.asarray(problem['initial_matrix'])
    scene_path = store.root/'scenes/matching.json'
    scene = json.loads(scene_path.read_text())
    for index, observation in enumerate(problem['observations']):
        world = np.asarray(observation['world_points'])[50:171]
        local = world-pose[:3, 3]
        # Represents a line recovered from independently measured image pixels
        # using the recorded reference calibration; not an arbitrary 3D line.
        transform = np.linalg.inv(initial) @ truth
        reference_local = local @ transform[:3, :3].T+transform[:3, 3]
        scene['vectors'].append({'id': f'image{index}', 'label': 'image-lane', 'closed': False,
            'points': (reference_local+pose[:3, 3]).tolist()})
    atomic_json(scene_path, scene)
    response = client.put(endpoint+'/pairs', json={'revision': 1, 'pairs': []})
    assert response.status_code == 200
    revision = response.json()['revision']
    for index in range(2):
        response = client.post(endpoint+'/pairs/from-lanes', json={'revision': revision, 'id': f'selected{index}',
            'frame_id': 'f0', 'lane_id': f'lane{index}', 'image_lane_id': f'image{index}', 'split': 'train'})
        assert response.status_code == 200, response.text
        revision = response.json()['revision']
    pairing_before = response.json()
    assert all(pair['image_reference']['frame_id'] == 'f0' for pair in pairing_before['pairs'])
    request['pairing_revision'] = revision
    job = finish(client, endpoint, request)
    assert job['result']['after']['splits']['train']['mean_px'] < .1
    assert client.get(endpoint).json()['pairing'] == pairing_before


def test_matching_snapshot_replays_without_optional_torch_runtime(matching_api):
    client, store, endpoint, request, _ = matching_api
    job = finish(client, endpoint, request)
    directory = store.root/'lane_calibration/matching/front/jobs'/job['id']
    replay = store.root/'replay'
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2]/'scripts/optimize-lanes.py'),
        '--snapshot', str(directory/'input_snapshot.json'), '--output', str(replay)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(np.load(replay/'optimized_extrinsic.npy'), np.load(directory/'optimized_extrinsic.npy'), atol=1e-9)
