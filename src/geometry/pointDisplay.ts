import type { Vec3 } from '../types';

export type PointColorMode = 'height' | 'intensity' | 'semantic';

/** Keep source indices: hiding roofs must never reindex semantic annotations. */
export function visiblePointIndices(points: number[][], groundOnly = false, groundZ = -1.7, groundBand = .35): number[] {
  const indices: number[] = [];
  const band = Math.max(0, groundBand);
  points.forEach((point, index) => {
    if (point.length < 3 || !Number.isFinite(point[0]) || !Number.isFinite(point[1]) || !Number.isFinite(point[2])) return;
    if (!groundOnly || Math.abs(point[2] - groundZ) <= band) indices.push(index);
  });
  return indices;
}

/** Robust contrast window; a constant/missing channel has no reflectance detail. */
export function intensityRange(points: number[][], indices: number[]): [number, number] | null {
  const samples: number[] = [];
  const step = Math.max(1, Math.floor(indices.length / 8192));
  let minimum = Infinity;
  let maximum = -Infinity;
  indices.forEach((index, offset) => {
    const value = points[index][3];
    if (!Number.isFinite(value)) return;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    if (offset % step === 0) samples.push(value);
  });
  if (!Number.isFinite(minimum) || maximum - minimum <= Math.max(1e-9, Math.abs(maximum) * 1e-7)) return null;
  if (!samples.length) return [minimum, maximum];
  samples.sort((a, b) => a - b);
  const low = samples[Math.floor((samples.length - 1) * .02)];
  const high = samples[Math.floor((samples.length - 1) * .995)];
  // Very sparse bright paint can occupy less than one percentile of a road scan.
  return high > low ? [low, high] : [minimum, maximum];
}

export function intensityBrightness(value: number | undefined, range: [number, number] | null): number {
  if (!range || !Number.isFinite(value)) return .48;
  const normalized = Math.max(0, Math.min(1, (value! - range[0]) / (range[1] - range[0])));
  return .18 + normalized * .82;
}

/** Place ground vectors at cursor XY; nearby measured road points supply only Z. */
export function groundDrawingPoint(points: number[][], planePoint: Vec3, groundBand = .35, radius = .6): Vec3 {
  let nearestDistance = radius * radius;
  let height = planePoint[2];
  points.forEach(point => {
    if (!Number.isFinite(point[0]) || !Number.isFinite(point[1]) || !Number.isFinite(point[2]) || Math.abs(point[2] - planePoint[2]) > groundBand) return;
    const distance = (point[0] - planePoint[0]) ** 2 + (point[1] - planePoint[1]) ** 2;
    if (distance < nearestDistance) { nearestDistance = distance; height = point[2]; }
  });
  return [planePoint[0], planePoint[1], height];
}
