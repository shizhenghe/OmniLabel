import type {PointBuffer, Vec3} from '../types';

/** Index source coordinates, never Float32 render coordinates or display samples. */
export function pointCloudSnapper(points: number[][], binary?: PointBuffer) {
  const cellSize = .6;
  const cells = new Map<string, number[]>();
  const count = binary ? Math.floor(binary.length / 4) : points.length;
  const at = (i: number): Vec3 => binary
    ? [binary[i * 4], binary[i * 4 + 1], binary[i * 4 + 2]]
    : [points[i][0], points[i][1], points[i][2]];
  for (let i = 0; i < count; i++) {
    const p = at(i);
    if (!p.every(Number.isFinite)) continue;
    const key = `${Math.floor(p[0] / cellSize)},${Math.floor(p[1] / cellSize)}`;
    const bucket = cells.get(key);
    if (bucket) bucket.push(i); else cells.set(key, [i]);
  }
  return (point: Vec3, radius = .6, heightBand = .35): Vec3 => {
    let nearest = radius * radius;
    let result = point;
    for (let x = Math.floor((point[0] - radius) / cellSize); x <= Math.floor((point[0] + radius) / cellSize); x++) {
      for (let y = Math.floor((point[1] - radius) / cellSize); y <= Math.floor((point[1] + radius) / cellSize); y++) {
        for (const index of cells.get(`${x},${y}`) ?? []) {
          const candidate = at(index);
          if (Math.abs(candidate[2] - point[2]) > heightBand) continue;
          const distance = candidate.reduce((sum, v, axis) => sum + (v - point[axis]) ** 2, 0);
          if (distance <= nearest) {nearest = distance; result = candidate;}
        }
      }
    }
    return result;
  };
}
