import { expect, test } from '@playwright/test';
import { projectLidarPoint } from '../../src/components/CameraView';
import type { Camera, Frame } from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const camera: Camera = {
  id: 'front', name: 'FRONT', width: 1280, height: 720,
  intrinsics: [[600, 0, 640], [0, 600, 360], [0, 0, 1]],
  camera_from_ego: identity,
};
const frame: Frame = {
  id: 'frame', timestamp: 0, points: [], images: {},
  ego_to_world: identity, lidar_to_ego: identity,
};

test('camera projection composes LiDAR→ego→optical extrinsics in order', () => {
  // LiDAR [4,1,0] → ego [5,1,2]. Optical convention is x right,
  // y down, z forward, so this becomes [-1,0,5] with the 2m camera offset.
  const mountedFrame: Frame = {
    ...frame,
    lidar_to_ego: [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]],
    // World pose is unrelated to same-frame camera projection.
    ego_to_world: [[0, -1, 0, 100], [1, 0, 0, -20], [0, 0, 1, 7], [0, 0, 0, 1]],
  };
  const mountedCamera: Camera = {
    ...camera,
    camera_from_ego: [[0, -1, 0, 0], [0, 0, -1, 2], [1, 0, 0, 0], [0, 0, 0, 1]],
  };
  const projected = projectLidarPoint([4, 1, 0], mountedCamera, mountedFrame);
  expect(projected).not.toBeNull();
  expect(projected![0]).toBeCloseTo(520, 10);
  expect(projected![1]).toBeCloseTo(360, 10);
  expect(projected![2]).toBeCloseTo(5, 10);
});

test('Brown-Conrady radial and tangential terms match calibrated pixel coordinates', () => {
  const calibrated: Camera = {
    ...camera,
    intrinsics: [[600, 12, 640], [0, 500, 360], [0, 0, 1]],
    distortion: [0.1, -0.02, 0.01, -0.005, 0.003],
  };
  // Normalized x=.5, y=.25, r²=.3125. Independent expected terms.
  const r2 = 0.3125;
  const radial = 1 + 0.1 * r2 - 0.02 * r2 ** 2 + 0.003 * r2 ** 3;
  const distortedX = 0.5 * radial + 2 * 0.01 * 0.5 * 0.25 - 0.005 * (r2 + 2 * 0.5 ** 2);
  const distortedY = 0.25 * radial + 0.01 * (r2 + 2 * 0.25 ** 2) + 2 * -0.005 * 0.5 * 0.25;
  const projected = projectLidarPoint([2, 1, 4], calibrated, frame);
  expect(projected).not.toBeNull();
  expect(projected![0]).toBeCloseTo(600 * distortedX + 12 * distortedY + 640, 10);
  expect(projected![1]).toBeCloseTo(500 * distortedY + 360, 10);
  expect(projected![2]).toBe(4);
});

test('pinhole projection rejects points behind the camera and at the near plane', () => {
  expect(projectLidarPoint([0, 0, -5], camera, frame)).toBeNull();
  expect(projectLidarPoint([1, 1, 0], camera, frame)).toBeNull();
  expect(projectLidarPoint([0, 0, 0.05], camera, frame)).toBeNull();
  expect(projectLidarPoint([0, 0, 10], camera, frame)).toEqual([640, 360, 10]);
});
