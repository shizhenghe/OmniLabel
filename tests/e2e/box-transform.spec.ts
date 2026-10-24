import { expect, test } from '@playwright/test';
import { resizeBoxFace, normalizeYaw, placeViewportGrips } from '../../src/components/BoxTransformGizmo';
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

test('width resize at an arbitrary yaw preserves the opposite local side and the other dimensions', () => {
  const original: Box = {...box, yaw: .63};
  const result = resizeBoxFace(original, 1, -1, .8);
  expect(result.size).toEqual([4, 2.8, 2]);
  const localWidthAxis = [-Math.sin(original.yaw), Math.cos(original.yaw), 0];
  for (let i = 0; i < 3; i++) {
    const oldOpposite = original.center[i] + localWidthAxis[i] * original.size[1] / 2;
    const newOpposite = result.center![i] + localWidthAxis[i] * result.size![1] / 2;
    expect(newOpposite).toBeCloseTo(oldOpposite, 10);
  }
});

test('projected face grips stay reachable at viewport edges without changing their world faces', () => {
  const grips = [
    {id: 'translate-xy', action: 'translate' as const, x: 4, y: 4, visible: true, world: [4, 7, 1]},
    {id: 'scale-x', action: 'scale' as const, x: -30, y: -20, visible: true, world: [6, 7, 1]},
    {id: 'scale-y', action: 'scale' as const, x: 640, y: 300, visible: true, world: [4, 8, 1]},
    {id: 'behind', action: 'scale' as const, x: -80, y: -60, visible: false, world: [4, 7, -1]},
  ];
  const placed = placeViewportGrips(grips, 600, 300, {x: 4, y: 4});
  for (const grip of placed.filter(grip => grip.visible)) {
    const halfWidth = grip.action === 'scale' ? 17 : 12.5;
    const halfHeight = grip.action === 'scale' ? 13 : 12.5;
    expect(grip.x - halfWidth).toBeGreaterThanOrEqual(4);
    expect(grip.x + halfWidth).toBeLessThanOrEqual(596);
    expect(grip.y - halfHeight).toBeGreaterThanOrEqual(4);
    expect(grip.y + halfHeight).toBeLessThanOrEqual(296);
  }
  expect(placed.map(grip => grip.world)).toEqual(grips.map(grip => grip.world));
  expect(grips[1].x).toBe(-30);
  expect(placed[3]).toEqual(grips[3]);
});

test('grip placement keeps overlapping controls apart and remains finite in tiny viewports', () => {
  const grips = [0, 1, 2].map(index => ({id: `scale-${index}`, action: 'scale' as const,
    x: 0, y: 0, visible: true}));
  const placed = placeViewportGrips(grips, 300, 200, {x: 0, y: 0});
  for (let i = 0; i < placed.length; i++) for (let j = 0; j < i; j++) {
    expect(Math.abs(placed[i].x - placed[j].x) >= 39 || Math.abs(placed[i].y - placed[j].y) >= 31).toBe(true);
  }
  for (const grip of placeViewportGrips(grips, 0, 8, {x: NaN, y: Infinity})) {
    expect(Number.isFinite(grip.x) && Number.isFinite(grip.y)).toBe(true);
  }
});
