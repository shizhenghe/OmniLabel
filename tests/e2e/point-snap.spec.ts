import {expect, test} from '@playwright/test';
import {pointCloudSnapper} from '../../src/geometry/pointSnap';

test('snap finds the closest measured XYZ, preserves UTM precision and avoids roofs and empty space', () => {
  const x = 500000, y = 3500000;
  const points = [[x+.125, y+.25, .12, 1], [x+.4, y+.3, .1, 1],
    [x+.1, y+.2, 3, 1], [NaN, y, 0, 1]];
  for (const snap of [pointCloudSnapper(points), pointCloudSnapper([], new Float64Array(points.flat()))]) {
    expect(snap([x+.1, y+.2, 0])).toEqual(points[0].slice(0, 3));
    expect(snap([x+.1, y+.2, 0], .01)).toEqual([x+.1, y+.2, 0]);
    expect(snap([x+10, y+10, 0])).toEqual([x+10, y+10, 0]);
    expect(snap([x+.1, y+.2, 1])).toEqual([x+.1, y+.2, 1]);
  }
  expect(pointCloudSnapper([[-.61, -.61, 0], [-.59, -.59, .1]])([-.6, -.6, .08]))
    .toEqual([-.59, -.59, .1]);
});
