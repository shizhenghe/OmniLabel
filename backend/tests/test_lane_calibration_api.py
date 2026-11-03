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


def _image_derived_lane(store, frame_id, lane_id, *, world_from_primary=None, matrix=None):
    """Represent independently measured image rays as saved world polylines."""
    from omnilabel.scene import registered_poses
    problem, truth = known_problem()
    clip=store.source_clip('known'); frame=next(f for f in clip.frames if f.id==frame_id)
    pose=registered_poses(store,clip)[frame_id] if world_from_primary is None else world_from_primary
    reference=np.asarray(clip.cameras[0].camera_from_ego) if matrix is None else matrix
    scene=json.loads((store.root/'scenes/known.json').read_text())
    source=next(v for v in scene['vectors'] if v['id']==lane_id)
    local=(np.asarray(source['points'])-pose[:3,3])@pose[:3,:3]
    measured_camera=local@truth[:3,:3].T+truth[:3,3]
    proxy=(measured_camera-reference[:3,3])@reference[:3,:3]
    world=proxy@pose[:3,:3].T+pose[:3,3]
    name=f'image-{frame_id}-{lane_id}'
    scene['vectors'].append({'id':name,'label':'image-lane','closed':False,'points':world.tolist()})
    atomic_json(store.root/'scenes/known.json',scene)
    return name


def test_selected_image_lane_uses_registered_pose_saved_override_and_frozen_pixels(lane_api):
    from omnilabel.scene import _clip_fingerprint, registered_poses
    from omnilabel.lane_optimizer import resample
    from omnilabel.projection_calibration import project_camera_points
    client,store,route,request=lane_api
    assert client.put(route+'/pairs',json={'revision':1,'pairs':[]}).status_code==200
    raw=store.source_clip('known').model_dump(); reference=np.asarray(raw['cameras'][0]['camera_from_ego'])
    reference[0,3]+=.08
    raw['frames'][1]['projection_camera_from_lidar']={'front':reference.tolist()}
    atomic_json(store.root/'clips/known.json',raw)
    clip=store.source_clip('known');poses=registered_poses(store,clip);poses['f1'][:3,3]+=[.15,-.2,.05]
    atomic_json(store.root/'scene_poses/known.json',{'clip_fingerprint':_clip_fingerprint(store,clip),
        'world_from_lidar':{key:value.tolist() for key,value in poses.items()}})
    image_id=_image_derived_lane(store,'f1','lane0',world_from_primary=poses['f1'],matrix=reference)
    response=client.post(route+'/pairs/from-lanes',json={'revision':2,'id':'selected','frame_id':'f1',
        'lane_id':'lane0','image_lane_id':image_id,'split':'validation'})
    assert response.status_code==200,response.text
    document=response.json();pair=document['pairs'][0]
    assert pair['lane_id']=='lane0' and pair['image_lane_id']==image_id
    np.testing.assert_allclose(pair['image_reference']['world_from_primary'],poses['f1'])
    np.testing.assert_allclose(pair['image_reference']['camera_from_primary'],reference)
    source=next(v for v in json.loads((store.root/'scenes/known.json').read_text())['vectors'] if v['id']==image_id)
    world=resample(source['points'],512)
    local=(world-poses['f1'][:3,3])@poses['f1'][:3,:3]
    expected,_=project_camera_points(local,reference,np.asarray(clip.cameras[0].intrinsics),np.asarray(clip.cameras[0].distortion))
    np.testing.assert_allclose(pair['image_points'],expected,atol=1e-7)
    settings={**request,'pairing_revision':3}
    before=client.post(route+'/preview',json=settings);assert before.status_code==200,before.text
    raw['cameras'][0]['camera_from_ego'][0][3]+=.2
    atomic_json(store.root/'clips/known.json',raw)
    after=client.post(route+'/preview',json=settings);assert after.status_code==200,after.text
    assert after.json()['statistics']['splits']['validation']['mean_px']!=before.json()['statistics']['splits']['validation']['mean_px']
    assert client.get(route).json()['pairing']['pairs'][0]['image_points']==pair['image_points']
    changed=deepcopy(document);changed['pairs'][0]['image_points'][0][0]+=.1
    assert client.put(route+'/pairs',json=changed).status_code==422
    # Reordering/changing the split preserves the capture instead of reprojecting.
    document['pairs'][0]['split']='train'
    updated=client.put(route+'/pairs',json=document);assert updated.status_code==200,updated.text
    assert updated.json()['pairs'][0]['image_points']==pair['image_points']
    scene_path=store.root/'scenes/known.json';scene=json.loads(scene_path.read_text())
    next(v for v in scene['vectors'] if v['id']==image_id)['points'][0][0]+=.1
    atomic_json(scene_path,scene)
    stale=client.post(route+'/preview',json={**settings,'pairing_revision':4})
    assert stale.status_code==409 and '已修改' in stale.text


def test_selected_lane_pairs_reject_self_missing_hidden_and_reused_sources(lane_api):
    client,store,route,request=lane_api
    assert client.put(route+'/pairs',json={'revision':1,'pairs':[]}).status_code==200
    image_id=_image_derived_lane(store,'f0','lane0')
    selection={'revision':2,'id':'selected','frame_id':'f0','lane_id':'lane0','image_lane_id':image_id}
    for extra in ({'image_lane_id':'lane0'},{'image_lane_id':'missing'},{'frame_id':'missing'},
                  {'image_world_range':[.8,.2]},{'revision':0}):
        response=client.post(route+'/pairs/from-lanes',json={**selection,**extra})
        assert response.status_code in {409,422},response.text
    scene_path=store.root/'scenes/known.json';scene=json.loads(scene_path.read_text())
    scene['vectors'].append({'id':'behind','label':'image-lane','closed':False,
        'points':[[499980,3500000,10],[499970,3500001,10]]})
    atomic_json(scene_path,scene)
    invisible=client.post(route+'/pairs/from-lanes',json={**selection,'image_lane_id':'behind'})
    assert invisible.status_code==422 and '可见' in invisible.text
    clip=store.source_clip('known');matrix=np.asarray(clip.cameras[0].camera_from_ego);pose=np.asarray(clip.frames[0].ego_to_world)
    camera_points=np.array([[0,0,20],[100,0,20],[0,1,20]])
    local=(camera_points-matrix[:3,3])@matrix[:3,:3]
    scene['vectors'].append({'id':'disconnected','label':'image-lane','closed':False,
        'points':(local@pose[:3,:3].T+pose[:3,3]).tolist()})
    atomic_json(scene_path,scene)
    disconnected=client.post(route+'/pairs/from-lanes',json={**selection,'image_lane_id':'disconnected'})
    assert disconnected.status_code==422 and '不连续' in disconnected.text
    captured=client.post(route+'/pairs/from-lanes',json=selection);assert captured.status_code==200,captured.text
    duplicate=client.post(route+'/pairs/from-lanes',json={**selection,'revision':3,'id':'duplicate','lane_id':'lane1'})
    assert duplicate.status_code==422 and '重复' in duplicate.text
    assert client.get(route).json()['pairing']==captured.json()


def test_multiframe_selected_lanes_recover_extrinsics_with_independent_holdout(lane_api):
    client,store,route,request=lane_api
    assert client.put(route+'/pairs',json={'revision':1,'pairs':[]}).status_code==200
    revision=2
    for index in range(4):
        for lane_index in range(2):
            fid,lid=f'f{index}',f'lane{lane_index}'
            image_id=_image_derived_lane(store,fid,lid)
            response=client.post(route+'/pairs/from-lanes',json={'revision':revision,'id':f'pair-{fid}-{lid}',
                'frame_id':fid,'lane_id':lid,'image_lane_id':image_id,'split':'validation' if index==3 else 'train'})
            assert response.status_code==200,response.text
            revision=response.json()['revision']
    original=client.get(route).json()['pairing']
    response=client.post(route+'/jobs',json={**request,'pairing_revision':revision})
    assert response.status_code==202,response.text
    deadline=time.monotonic()+100;job_id=response.json()['id']
    while time.monotonic()<deadline:
        job=client.get(route+'/jobs/'+job_id).json()
        if job['status'] in {'completed','failed','cancelled'}:break
        time.sleep(.05)
    assert job['status']=='completed',job
    assert job['result']['usable'],job['result']['rejection_reasons']
    for split in ('train','validation'):
        assert job['result']['after']['splits'][split]['mean_px']<.5
        assert job['result']['error_reduction'][split]>.9
    assert client.get(route).json()['pairing']==original


def test_one_world_lane_accepts_multiple_image_lanes_but_image_owner_is_unique(lane_api):
    client, store, route, _ = lane_api
    client.put(route+'/pairs', json={'revision': 1, 'pairs': []}).raise_for_status()
    first = _image_derived_lane(store, 'f0', 'lane0')
    second = _image_derived_lane(store, 'f0', 'lane1')
    selection = {'revision': 2, 'id': 'one', 'frame_id': 'f0', 'lane_id': 'lane0', 'image_lane_id': first}
    client.post(route+'/pairs/from-lanes', json=selection).raise_for_status()
    response = client.post(route+'/pairs/from-lanes', json={**selection, 'revision': 3, 'id': 'two', 'image_lane_id': second})
    assert response.status_code == 200, response.text
    document = response.json()
    assert [p['lane_id'] for p in document['pairs']] == ['lane0', 'lane0']
    assert [p['image_lane_id'] for p in document['pairs']] == [first, second]
    # Both entry points enforce ownership, including attempts in another frame.
    for frame in ('f0', 'f1'):
        rejected = client.post(route+'/pairs/from-lanes', json={**selection, 'revision': 4, 'id': 'other',
            'lane_id': 'lane1', 'frame_id': frame})
        assert rejected.status_code == 422, rejected.text
    broken = deepcopy(document)
    broken['pairs'][1]['image_lane_id'] = first
    assert client.put(route+'/pairs', json=broken).status_code == 422
    assert client.get(route).json()['pairing'] == document


def test_discard_invalidates_all_candidates_preserves_saved_data_and_allows_reselection(lane_api):
    client, store, route, _ = lane_api
    # Include a previously applied job and another candidate, as well as a
    # different camera's state: reset must be scoped and must not undo applying.
    directory = store.root/'lane_calibration/known/front'
    saved = {path: path.read_bytes() for path in store.root.rglob('*.json')}
    for index, applied in enumerate((True, False)):
        atomic_json(directory/f'jobs/old{index}/job.json', {'id': f'old{index}', 'status': 'completed',
            'applied': applied, 'can_apply': True, 'result': {'optimized_matrix': np.eye(4).tolist()}})
    other = store.root/'lane_calibration/known/other/pairs.json'
    atomic_json(other, {'revision': 9, 'pairs': []})
    original = client.get(route).json()['pairing']
    assert client.post(route+'/discard', json={'revision': 0, 'pairs': []}).status_code == 409
    assert client.post(route+'/discard', json=original).status_code == 422
    manager = client.app.state.lane_calibration
    manager.active['busy'] = True
    try:
        assert client.post(route+'/discard', json={'revision': 1, 'pairs': []}).status_code == 409
        assert client.get(route).json()['pairing'] == original
    finally:
        manager.active.pop('busy')
    response = client.post(route+'/discard', json={'revision': 1, 'pairs': []})
    assert response.status_code == 200, response.text
    assert response.json()['pairing'] == {'revision': 2, 'pairs': []}
    assert response.json()['latest_job'] is None
    assert client.get(route).json()['latest_job'] is None
    for index in range(2):
        job = client.get(route+f'/jobs/old{index}').json()
        assert job['discarded'] and job['result'] is None and not job['can_apply']
        assert job['applied'] == (index == 0)
        assert client.post(route+f'/jobs/old{index}/apply', json={'confirm_shared_update': True}).status_code == 409
        assert client.get(route+f'/jobs/old{index}/download').status_code == 409
    for path, content in saved.items():
        if path != directory/'pairs.json': assert path.read_bytes() == content
    assert json.loads(other.read_text()) == {'revision': 9, 'pairs': []}
    image = _image_derived_lane(store, 'f0', 'lane0')
    reselected = client.post(route+'/pairs/from-lanes', json={'revision': 2, 'id': 'new', 'frame_id': 'f0',
        'lane_id': 'lane0', 'image_lane_id': image})
    assert reselected.status_code == 200, reselected.text
