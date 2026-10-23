import { expect, test } from '@playwright/test';
import { groundDrawingPoint, intensityBrightness, intensityRange, visiblePointIndices } from '../../src/geometry/pointDisplay';

test('ground display keeps original point indices instead of relabeling the sliced cloud', () => {
  const points = [[0, 0, 8, 50], [0, 0, .1, 20], [1, 0, -.2, 240], [0, 0, 1, 80]];
  const sourceLabels = { '1': 'road', '2': 'lane_marking' };
  const visible = visiblePointIndices(points, true, 0, .35);
  expect(visible).toEqual([1, 2]);
  expect(visible.map(index => sourceLabels[String(index) as keyof typeof sourceLabels])).toEqual(['road', 'lane_marking']);
  expect(visiblePointIndices(points)).toEqual([0, 1, 2, 3]);
});

test('reflectance contrast supports unnormalised sensors and sparse bright road paint', () => {
  const points = Array.from({ length: 1000 }, (_, i) => [i, 0, 0, 35]);
  points.push([1001, 0, 0, 245]);
  const range = intensityRange(points, visiblePointIndices(points));
  expect(range).toEqual([35, 245]);
  expect(intensityBrightness(35, range)).toBeCloseTo(.18);
  expect(intensityBrightness(245, range)).toBeCloseTo(1);
});

test('constant or missing reflectance never invents markings from height', () => {
  const constant = [[0, 0, 0, .4], [1, 1, 5, .4]];
  expect(intensityRange(constant, [0, 1])).toBeNull();
  expect(intensityRange([[0, 0, 0], [1, 1, 0]], [0, 1])).toBeNull();
  expect(intensityBrightness(.4, null)).toBe(intensityBrightness(undefined, null));
});

test('ground vector placement ignores roofs and follows nearby measured road height', () => {
  const points = [[0, 0, 5, .6], [.15, .1, .08, .4], [.4, .1, -.03, .4]];
  expect(groundDrawingPoint(points, [0, 0, 0], .35)).toEqual([0, 0, .08]);
  expect(groundDrawingPoint(points, [8, 8, 0], .35)).toEqual([8, 8, 0]);
  expect(groundDrawingPoint([[0, 0, 5, .6]], [0, 0, 0], .35)).toEqual([0, 0, 0]);
});
