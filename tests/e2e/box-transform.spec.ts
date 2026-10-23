import { expect, test } from '@playwright/test';
import { resizeBoxFace, normalizeYaw } from '../../src/components/BoxTransformGizmo';
import type { Box } from '../../src/types';

const box: Box = {id: 'b0', track_id: 'track-0', label: 'vehicle', center: [5, 2, 1], size: [4, 2, 2], yaw: Math.PI / 2};

test('yaw-aligned face resize fixes the opposite face instead of shifting both faces', () => {
  const result = resizeBoxFace(box, 0, 1, 2);
  expect(result.size).toEqual([6, 2, 2]);
  expect(result.center![0]).toBeCloseTo(5);
  expect(result.center![1]).toBeCloseTo(3);
  expect(result.center![1] - result.size![0] / 2).toBeCloseTo(box.center[1] - box.size[0] / 2);
});

test('negative face resize has the same local-axis geometry and positive minimum dimension', () => {
  const expanded = resizeBoxFace(box, 0, -1, 2);
  expect(expanded.center![1]).toBeCloseTo(1);
  expect(expanded.center![1] + expanded.size![0] / 2).toBeCloseTo(box.center[1] + box.size[0] / 2);
  const shrunk = resizeBoxFace(box, 2, -1, -100);
  expect(shrunk.size![2]).toBeCloseTo(0.05);
  expect(shrunk.center![2] + shrunk.size![2] / 2).toBeCloseTo(box.center[2] + box.size[2] / 2);
});

test('yaw normalization keeps shortest-equivalent rotation at the wrap boundary', () => {
  expect(normalizeYaw(2 * Math.PI + 0.2)).toBeCloseTo(0.2);
  expect(normalizeYaw(-2 * Math.PI - 0.2)).toBeCloseTo(-0.2);
});
