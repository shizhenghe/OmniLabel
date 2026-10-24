import {expect, test} from '@playwright/test';
import {applyCalibration, calibrationWorldPoses, correctExtrinsic, inverseExtrinsic, multiplyExtrinsics,
  type CalibrationDocument} from '../../src/geometry/calibration';
import {projectLidarPoint} from '../../src/components/CameraView';
import type {Clip} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const base: CalibrationDocument = {schema_version: '1.0', revision: 'original', clip_id: 'fixture',
  cameras: {front: identity}, lidar_to_ego: {f0: identity}, world_from_lidar: {f0: [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}};

test('zero correction preserves the optical transform and a known yaw uses destination-frame axes', () => {
  const optical = [[0, -1, 0, .6], [0, 0, -1, -.1], [1, 0, 0, 1.7], [0, 0, 0, 1]];
  expect(correctExtrinsic(optical, {translation: [0, 0, 0], rotation: [0, 0, 0]})).toEqual(optical);
  const changed = correctExtrinsic(identity, {translation: [.2, 0, 0], rotation: [0, 0, 90]});
  expect(changed[0][0]).toBeCloseTo(0);
  expect(changed[1][0]).toBeCloseTo(1);
  expect(changed[0][1]).toBeCloseTo(-1);
  expect(changed[0][3]).toBeCloseTo(.2);
});

test('LiDAR calibration preview preserves registered ego while keeping the world scene fixed', () => {
  const draft = {...base, lidar_to_ego: {f0: correctExtrinsic(identity, {translation: [.3, -.2, 0], rotation: [0, 0, 8]})}};
  const world = calibrationWorldPoses(base, draft).f0;
  const ego = multiplyExtrinsics(world, inverseExtrinsic(draft.lidar_to_ego.f0));
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) expect(ego[i][j]).toBeCloseTo(base.world_from_lidar.f0[i][j], 10);
});

test('camera preview changes calibrated pixels immediately without mutating or reindexing source points', () => {
  const points = [[2, 0, 5, .6]];
  const clip: Clip = {id: 'fixture', name: 'Fixture', fps: 10, description: '',
    cameras: [{id: 'front', name: 'Front', width: 200, height: 100, intrinsics: [[100, 0, 100], [0, 100, 50], [0, 0, 1]], camera_from_ego: identity}],
    frames: [{id: 'f0', timestamp: 0, ego_to_world: identity, lidar_to_ego: identity, points, point_count: 1, images: {}}]};
  const draft = {...base, cameras: {front: correctExtrinsic(identity, {translation: [.5, 0, 0], rotation: [0, 0, 0]})}};
  const preview = applyCalibration(clip, draft);
  expect(preview.frames[0].points).toBe(points);
  expect(preview.cameras[0].intrinsics).toBe(clip.cameras[0].intrinsics);
  expect(projectLidarPoint(points[0], clip.cameras[0], clip.frames[0])![0]).toBeCloseTo(140);
  expect(projectLidarPoint(points[0], preview.cameras[0], preview.frames[0])![0]).toBeCloseTo(150);
  expect(clip.cameras[0].camera_from_ego).toBe(identity);
});
