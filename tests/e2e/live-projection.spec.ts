import {expect, test} from '@playwright/test';
import {projectFramePoints} from '../../src/geometry/liveProjection';
import type {Camera, Frame} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const camera: Camera = {id: 'front', name: 'Front', width: 1280, height: 720,
  camera_from_ego: identity, intrinsics: [[600, 0, 640], [0, 600, 360], [0, 0, 1]]};
const frame: Frame = {id: 'f0', timestamp: 0, points: [[0, 0, 5], [0, 0, -1], [1, 0, 5]], point_count: 3,
  ego_to_world: identity, lidar_to_ego: identity, images: {}};

test('live camera and LiDAR mount changes immediately move raw projected pixels', () => {
  const before = projectFramePoints(frame, [camera]).front;
  expect(before.map(p => p.index)).toEqual([0, 2]);
  const changedCamera = {...camera, camera_from_ego: [[1, 0, 0, .5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]};
  expect(projectFramePoints(frame, [changedCamera]).front[0].u).toBeCloseTo(before[0].u + 60);
  const changedFrame = {...frame, lidar_to_ego: [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]};
  expect(projectFramePoints(changedFrame, [changedCamera]).front[0].u).toBeCloseTo(before[0].u + 180);
});

test('live projection does not stride or renumber fully loaded source points', () => {
  const points = Array.from({length: 751}, (_, index) => [index / 751, 0, 5, .4]);
  const result = projectFramePoints({...frame, points, point_count: points.length}, [camera]).front;
  expect(result).toHaveLength(points.length);
  expect(result.map(point => point.index)).toEqual(points.map((_, index) => index));
});
