from copy import deepcopy
import io
import json
import time
import zipfile

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from omnilabel.main import create_app
from omnilabel.models import Clip
from omnilabel.storage import atomic_json
from test_lane_optimizer import known_problem


@pytest.fixture
def lane_api(tmp_path):
    problem, _ = known_problem(); camera = problem['camera']
    frames = []
    for index in range(4):
        observation = next(o for o in problem['observations'] if o['frame_id'] == f'f{index}')
        frames.append({'id':f'f{index}','timestamp':index*.1,'ego_to_world':observation['world_from_lidar'],
            'lidar_to_ego':np.eye(4).tolist(),'points':[[10,0,0,.5]],'images':{'front':f'/api/assets/known/f{index}.png'}})
    clip = Clip.model_validate({'id':'known','name':'Known synthetic lane constraints','fps':10,
        'cameras':[{**camera,'name':'Front','camera_from_ego':problem['initial_matrix']}],'frames':frames})
    app = create_app(tmp_path, seed_demo=False); store = app.state.store; store.add_clip(clip)
    vectors = [{'id':f'lane{i}','label':'lane','closed':False,
        'points':next(o['world_points'] for o in problem['observations'] if o['lane_id']==f'lane{i}')} for i in range(2)]
    atomic_json(tmp_path/'scenes/known.json', {'revision':0,'coordinate_frame':'world','tracks':[],'vectors':vectors})
    for frame in frames:
        path = tmp_path/'assets/known'/f'{frame["id"]}.png';path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(camera['width'],camera['height']),'#202020').save(path)
    pairs = [{key:o[key] for key in ['id','frame_id','lane_id','image_points','world_range','split','weight']} for o in problem['observations']]
    for pair in pairs:pair['image_size']=[camera['width'],camera['height']]
    route = '/api/clips/known/lane-calibration/front'
    with TestClient(app) as client:
        saved=client.put(route+'/pairs',json={'revision':0,'pairs':pairs});assert saved.status_code==200,saved.text
        yield client,store,route,{'pairing_revision':1,'conventions_confirmed':True,'time_alignment_confirmed':True,
            'config':{'adam_iterations':80,'lbfgs_iterations':25,'timeout_seconds':90}}


def test_ordered_pairing_revision_and_frame_level_holdout(lane_api):
    client,store,route,request=lane_api
    state=client.get(route).json();assert state['runtime']['available']
    document=state['pairing'];assert [p['id'] for p in document['pairs']]==[f'p-{frame}-{lane}' for frame in range(4) for lane in range(2)]
    assert client.put(route+'/pairs',json={'revision':0,'pairs':document['pairs']}).status_code==409
    broken=deepcopy(document);broken['pairs'][1]['split']='validation'
    assert client.put(route+'/pairs',json=broken).status_code==422
    broken=deepcopy(document);broken['pairs'][0]['lane_id']='missing'
    assert client.put(route+'/pairs',json=broken).status_code==422
    broken=deepcopy(document);broken['pairs'][0]['image_points'][0]=[-1,2]
    assert client.put(route+'/pairs',json=broken).status_code==422
    assert client.get(route).json()['pairing']['revision']==1


def test_preview_requires_confirmation_and_handles_inverse_interpolated_pose(lane_api):
    client,store,route,request=lane_api
    original=(store.root/'clips/known.json').read_bytes()
    assert client.post(route+'/preview',json={**request,'conventions_confirmed':False}).status_code==422
    normal=client.post(route+'/preview',json=request);assert normal.status_code==200,normal.text
    clip=store.source_clip('known')
    external={**request,'pose_mode':'provided','pose_direction':'lidar_from_world',
        'initial_matrix':np.linalg.inv(np.asarray(clip.cameras[0].camera_from_ego)).tolist(),'initial_direction':'lidar_from_camera',
        'max_pose_gap_seconds':.5,'pose_samples':[{'timestamp':f.timestamp,'matrix':np.linalg.inv(np.asarray(f.ego_to_world)).tolist()} for f in [clip.frames[0],clip.frames[-1]]],
        'image_timestamps':{f.id:f.timestamp for f in clip.frames}}
    response=client.post(route+'/preview',json=external);assert response.status_code==200,response.text
    assert response.json()['statistics']['splits']['train']['mean_px']==pytest.approx(normal.json()['statistics']['splits']['train']['mean_px'],abs=1e-6)
    external['image_timestamps']['f1']=10
    assert client.post(route+'/preview',json=external).status_code==422
    assert (store.root/'clips/known.json').read_bytes()==original


def test_fused_map_never_silently_uses_identity_map_registration_as_sensor_pose(lane_api):
    client,store,route,request=lane_api
    clip=store.source_clip('known').model_dump();clip['source']='4d-clip-import'
    for frame in clip['frames']:frame['lidar_to_ego']=np.linalg.inv(np.asarray(frame['ego_to_world'])).tolist()
    atomic_json(store.root/'clips/known.json',clip)
    response=client.post(route+'/preview',json=request)
    assert response.status_code==422 and '真实 LiDAR pose' in response.text
    accepted=client.post(route+'/preview',json={**request,'pose_mode':'ego'})
    assert accepted.status_code==200,accepted.text


def test_secondary_calibration_sensor_uses_its_own_mount_and_registered_pose(lane_api):
    client,store,route,request=lane_api
    original=client.post(route+'/preview',json=request).json()
    raw=store.source_clip('known').model_dump()
    mount=np.eye(4);mount[:3,3]=[.4,-.2,.1]
    for frame in raw['frames']:
        frame.update(calibration_pointcloud_file=f'known/calib-{frame["id"]}.npy',calibration_point_count=1,
            calibration_lidar_to_ego=mount.tolist(),calibration_timestamp=frame['timestamp'])
        path=store.root/'pointclouds'/frame['calibration_pointcloud_file'];path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,np.array([[1,2,3,.5]],dtype=np.float32))
    atomic_json(store.root/'clips/known.json',raw)
    response=client.post(route+'/preview',json=request);assert response.status_code==200,response.text
    assert response.json()['statistics']['splits']['train']['mean_px']==pytest.approx(original['statistics']['splits']['train']['mean_px'],abs=1e-6)


def test_fused_map_with_physical_mount_needs_no_raw_scan(lane_api):
    client,store,route,request=lane_api
    original=client.post(route+'/preview',json=request).json()
    raw=store.source_clip('known').model_dump();raw['source']='4d-clip-import'
    from scipy.spatial.transform import Rotation
    mount=np.eye(4);mount[:3,:3]=Rotation.from_euler('z',1.0764,degrees=True).as_matrix()
    for frame in raw['frames']:
        frame['lidar_to_ego']=np.linalg.inv(np.asarray(frame['ego_to_world'])).tolist()
        frame['sensor_lidar_to_ego']=mount.tolist()
    atomic_json(store.root/'clips/known.json',raw)
    response=client.post(route+'/preview',json=request);assert response.status_code==200,response.text
    assert client.get(route).json()['coordinate_info']['sensor_mount_available']
    np.testing.assert_allclose(response.json()['initial_matrix'],np.asarray(raw['cameras'][0]['camera_from_ego'])@mount)
    assert response.json()['statistics']['splits']['train']['mean_px']==pytest.approx(original['statistics']['splits']['train']['mean_px'],abs=1e-6)


def test_single_frame_debug_filters_saved_multiframe_pairs_without_leaking_validation(lane_api):
    client,store,route,request=lane_api
    preview=client.post(route+'/preview',json={**request,'mode':'single','frame_ids':['f3']})
    assert preview.status_code==200,preview.text
    splits=preview.json()['statistics']['splits']
    assert splits['train']['frames']==1 and splits['validation'] is None
    assert client.get(route).json()['pairing']['pairs'][-1]['split']=='validation'


def test_real_job_outputs_validation_provenance_conflict_and_explicit_application(lane_api):
    client,store,route,request=lane_api
    # The real example contains imported local-world projection corrections.
    # A shared solution must supersede them, rather than compound both fixes.
    raw=store.source_clip('known').model_dump()
    for frame in raw['frames']:
        correction=np.asarray(raw['cameras'][0]['camera_from_ego']).copy();correction[0,3]+=.2
        frame['annotation_camera_from_lidar']={'front':correction.tolist()}
        frame['projection_camera_from_lidar']={'front':correction.tolist()}
    atomic_json(store.root/'clips/known.json',raw)
    source=(store.root/'clips/known.json').read_bytes();scene_path=store.root/'scenes/known.json';scene=scene_path.read_bytes()
    response=client.post(route+'/jobs',json=request);assert response.status_code==202,response.text
    job_id=response.json()['id'];deadline=time.monotonic()+100
    while time.monotonic()<deadline:
        job=client.get(route+'/jobs/'+job_id).json()
        if job['status'] in {'completed','failed','cancelled'}:break
        time.sleep(.05)
    assert job['status']=='completed',job
    assert job['result']['usable'],job['result']['rejection_reasons']
    assert job['can_apply'] and not job['applied']
    assert (store.root/'clips/known.json').read_bytes()==source
    assert scene_path.read_bytes()==scene
    download=client.get(route+'/jobs/'+job_id+'/download');assert download.status_code==200
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert {'optimized_extrinsic.npy','optimized_extrinsic.txt','initial_extrinsic.npy','optimization_log.csv',
            'optimization_config.yaml','error_statistics.json','loss_curve.png'} <= set(archive.namelist())
        assert len([n for n in archive.namelist() if n.startswith('before_after/')])==8
        stats=json.loads(archive.read('error_statistics.json'))
        assert stats['after']['splits']['validation']['mean_px']<stats['before']['splits']['validation']['mean_px']
    apply=route+'/jobs/'+job_id+'/apply'
    assert client.post(apply,json={}).status_code==422
    changed=json.loads(scene);changed['revision']+=1;changed['vectors'][0]['points'][0][0]+=.2;atomic_json(scene_path,changed)
    assert client.post(apply,json={'confirm_shared_update':True}).status_code==409
    assert (store.root/'clips/known.json').read_bytes()==source
    scene_path.write_bytes(scene)
    applied=client.post(apply,json={'confirm_shared_update':True});assert applied.status_code==200,applied.text
    assert applied.json()['applied']
    assert scene_path.read_bytes()==scene
    assert (store.root/'calibration_history/known/original.json').is_file()
    np.testing.assert_allclose(store.source_clip('known').cameras[0].camera_from_ego,job['result']['optimized_matrix'],atol=1e-10)
    for frame in store.source_clip('known').frames:
        assert 'front' not in frame.annotation_camera_from_lidar and 'front' not in frame.projection_camera_from_lidar
    backup=json.loads((store.root/'calibration_history/known/original.json').read_text())
    assert backup['frame_projection_overrides']['f0']['annotation_camera_from_lidar']['front']==raw['frames'][0]['annotation_camera_from_lidar']['front']
    assert client.post(apply,json={'confirm_shared_update':True}).status_code==200
