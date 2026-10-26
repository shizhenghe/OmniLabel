import { expect, test } from '@playwright/test';
import { clipPixelSegment, projectAnnotationResults, projectLidarSegment, projectVector, projectionSignature } from '../../src/geometry/cameraProjection';
import type { Annotations, Camera, Frame, Vector } from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const camera: Camera = { id: 'front', name: 'Front', width: 640, height: 360,
  intrinsics: [[100, 0, 320], [0, 100, 180], [0, 0, 1]], camera_from_ego: identity };
const frame: Frame = { id: 'f0', timestamp: 0, points: [], images: {}, ego_to_world: identity, lidar_to_ego: identity };
const annotations: Annotations = { boxes: [], vectors: [], image_polygons: [], point_labels: {}, revision: 0 };

test('lane labels project calibrated vertices into pixels without raw cloud data', () => {
  const lane: Vector = { id: 'lane', label: 'lane', points: [[-1, 0, 5], [1, 0, 5]], closed: false };
  expect(projectVector(lane, camera, frame)).toEqual({ segments: [[[300, 180], [340, 180]]], polygon: [] });
  const translated: Frame = { ...frame, lidar_to_ego: [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] };
  expect(projectVector(lane, camera, translated).segments).toEqual([[[320, 180], [360, 180]]]);
});

test('per-frame annotation correction overrides the static point-cloud calibration', () => {
  const lane: Vector = { id: 'lane', label: 'lane', points: [[-1, 0, 5], [1, 0, 5]], closed: false };
  const corrected: Frame = {
    ...frame,
    lidar_to_ego: [[1, 0, 0, 9], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    annotation_camera_from_lidar: {
      front: [[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    },
  };
  expect(projectVector(lane, camera, corrected).segments).toEqual([[[340, 180], [380, 180]]]);
});

test('near-plane crossing lines are clipped and behind-camera results stay absent', () => {
  const visible = projectLidarSegment([-1, 0, -1], [0, 0, 2], camera, frame);
  expect(visible).toHaveLength(1);
  expect(visible[0][0][0]).toBeCloseTo(0, 8);
  expect(visible[0][1]).toEqual([320, 180]);
  expect(visible.flat(2).every(Number.isFinite)).toBeTruthy();
  expect(projectLidarSegment([0, 0, -3], [1, 1, -1], camera, frame)).toEqual([]);
  expect(clipPixelSegment([-200, 180], [800, 180], 640, 360)).toEqual([[0, 180], [640, 180]]);
  expect(clipPixelSegment([-10, -20], [800, -20], 640, 360)).toBeNull();
});

test('closed map regions retain a bounded projected fill with viewport clipping', () => {
  const area: Vector = { id: 'map', label: 'crosswalk', points: [[-20, -20, 10], [20, -20, 10], [20, 20, 10], [-20, 20, 10]], closed: true };
  const projected = projectVector(area, camera, frame);
  expect(projected.polygon).toHaveLength(4);
  expect(projected.polygon.every(([u, v]) => u >= 0 && u <= 640 && v >= 0 && v <= 360)).toBeTruthy();
  expect(projected.polygon).toEqual(expect.arrayContaining([[120, 0], [520, 0], [520, 360], [120, 360]]));
  expect(projected.segments).toHaveLength(2);
});

test('Brown distortion curves are sampled rather than flattened to one pixel chord', () => {
  const distorted: Camera = { ...camera, distortion: [.3, 0, 0, 0, 0] };
  const segments = projectLidarSegment([-2, 1, 5], [2, 1, 5], distorted, frame);
  expect(segments).toHaveLength(24);
  expect(segments[0][0][0]).toBeCloseTo(277.6, 8);
  expect(segments[0][0][1]).toBeCloseTo(201.2, 8);
  const ys = segments.flat().map(point => point[1]);
  expect(Math.min(...ys)).toBeCloseTo(200.24, 8);
  expect(Math.max(...ys)).toBeCloseTo(201.2, 8);
});

test('annotation results include only visible boxes and active-camera masks, and change after box edits', () => {
  const labels: Annotations = { ...annotations, boxes: [
    { id: 'front-car', track_id: 'car-1', label: 'vehicle', center: [0, 0, 8], size: [2, 2, 2], yaw: 0 },
    { id: 'rear-car', track_id: 'car-2', label: 'vehicle', center: [0, 0, -8], size: [2, 2, 2], yaw: 0 },
  ], image_polygons: [
    { id: 'front-road', camera_id: 'front', label: 'road', points: [[0, 0], [100, 0], [100, 100]] },
    { id: 'rear-road', camera_id: 'rear', label: 'road', points: [[0, 0], [100, 0], [100, 100]] },
  ] };
  const result = projectAnnotationResults(labels, camera, frame);
  expect(result.boxes.map(box => box.id)).toEqual(['front-car']);
  expect(result.boxes[0].segments).toHaveLength(12);
  expect(result.imagePolygons.map(polygon => polygon.id)).toEqual(['front-road']);
  const edited: Annotations = { ...labels, boxes: labels.boxes.map(box => box.id === 'front-car' ? { ...box, center: [2, 0, 8] } : box) };
  expect(projectionSignature(projectAnnotationResults(edited, camera, frame))).not.toBe(projectionSignature(result));
});
