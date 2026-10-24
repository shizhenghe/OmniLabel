import type {Vec3} from '../types';

/** Subtract large world coordinates before casting any geometry to GPU Float32. */
export function renderOriginForPoints(points: number[][]): Vec3 {
  const first = points.find(point => point.slice(0, 3).every(Number.isFinite));
  return first && first.slice(0, 3).some(value => Math.abs(value) >= 10000) ? first.slice(0, 3) as Vec3 : [0, 0, 0];
}

export function toRenderPoint(point: number[], origin: Vec3): Vec3 {
  return [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
}
