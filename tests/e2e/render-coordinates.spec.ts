import {expect, test} from '@playwright/test';
import {renderOriginForPoints, toRenderPoint} from '../../src/geometry/renderCoordinates';

test('UTM centimetre detail survives Float32 GPU geometry without changing world coordinates', () => {
  const points = [[500000, 3500000, 140, .3], [500000.01, 3500000.01, 140.005, .9]];
  const original = structuredClone(points);
  const origin = renderOriginForPoints(points);
  const local = new Float32Array(points.flatMap(point => toRenderPoint(point, origin)));
  expect(local[3] - local[0]).toBeCloseTo(.01, 7);
  expect(local[4] - local[1]).toBeCloseTo(.01, 7);
  expect(local[5] - local[2]).toBeCloseTo(.005, 7);
  expect(new Float32Array([points[0][1], points[1][1]])[1]).toBe(new Float32Array([points[0][1]])[0]);
  expect(points).toEqual(original);
});
