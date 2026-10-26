import {expect, test} from '@playwright/test';
import {abortableDelay, frameDurationMs} from '../../src/playback';
import {reviewWorldAnnotations} from '../../src/reviewAnnotations';
import {EMPTY_ANNOTATIONS, IDENTITY} from '../../src/scene';

test('replay uses capture timestamps and requested speed without an arbitrary frame-rate floor', () => {
  const frames = [{timestamp: 120}, {timestamp: 120.04}, {timestamp: 120.24}];
  expect(frameDurationMs(frames, 0, 10, 2)).toBeCloseTo(20);
  expect(frameDurationMs(frames, 1, 10, .5)).toBeCloseTo(400);
  expect(frameDurationMs(frames, 2, 10, 1)).toBeCloseTo(200);
  expect(frameDurationMs([{timestamp: 0}], 0, 5, 1)).toBe(200);
});

test('pause cancels a pending presentation timer immediately', async () => {
  const controller = new AbortController();
  const timer = abortableDelay(60000, controller.signal);
  controller.abort();
  await expect(timer).rejects.toMatchObject({name: 'AbortError'});
});

test('replay includes frame-only labels in world space and avoids duplicating migrated scene labels', () => {
  const source = {...EMPTY_ANNOTATIONS, boxes: [
    {id: 'legacy-box', track_id: 'existing-track', label: 'car', center: [2, 0, 1] as [number, number, number], size: [4, 2, 2] as [number, number, number], yaw: 0},
    {id: 'new-frame-box', track_id: 'new-frame-track', label: 'car', center: [5, 0, 1] as [number, number, number], size: [4, 2, 2] as [number, number, number], yaw: 0},
  ]};
  const world = {...EMPTY_ANNOTATIONS, boxes: [{...source.boxes[0], id: 'existing-track', center: [12, 0, 1] as [number, number, number]}]};
  const frame = {id: 'f0', timestamp: 0, ego_to_world: [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    lidar_to_ego: IDENTITY, points: [], point_count: 0, images: {}};
  const result = reviewWorldAnnotations(world, source, frame, {world_from_lidar: {}});
  expect(result.boxes.map(box => box.id)).toEqual(['existing-track', 'new-frame-box']);
  expect(result.boxes[1].center).toEqual([15, 0, 1]);
  expect(source.boxes[1].center).toEqual([5, 0, 1]);
});
