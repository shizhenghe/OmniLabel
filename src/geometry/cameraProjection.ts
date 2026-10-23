import type { Annotations, Box, Camera, Frame, ImagePolygon, Vec3, Vector } from '../types';

export type Pixel = [number, number];
export type PixelDepth = [number, number, number];
export type PixelSegment = [Pixel, Pixel];
export const CAMERA_NEAR = 0.05;
const CLIP_NEAR = CAMERA_NEAR + 1e-6;
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];

export function transformPoint(matrix: number[][], point: number[]): Vec3 {
  return [0, 1, 2].map(row => matrix[row][0] * point[0] + matrix[row][1] * point[1] + matrix[row][2] * point[2] + matrix[row][3]) as Vec3;
}

export function lidarToCamera(point: number[], camera: Camera, frame: Frame): Vec3 {
  return transformPoint(camera.camera_from_ego, transformPoint(frame.lidar_to_ego, point));
}

/** Optical +Z is forward. Inputs have already been transformed into camera space. */
export function projectCameraPoint(point: number[], camera: Camera): PixelDepth | null {
  if (point[2] <= CAMERA_NEAR) return null;
  const x = point[0] / point[2], y = point[1] / point[2];
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = camera.distortion ?? [];
  const r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3;
  const xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  const k = camera.intrinsics;
  const u = k[0][0] * xd + k[0][1] * yd + k[0][2];
  const v = k[1][0] * xd + k[1][1] * yd + k[1][2];
  return Number.isFinite(u) && Number.isFinite(v) ? [u, v, point[2]] : null;
}

/** Frame annotations are LiDAR coordinates, including scene labels converted for this frame. */
export function projectLidarPoint(point: number[], camera: Camera, frame: Frame): PixelDepth | null {
  return projectCameraPoint(lidarToCamera(point, camera, frame), camera);
}

/** Liang–Barsky clipping keeps canvas commands within calibrated image bounds. */
export function clipPixelSegment(a: number[], b: number[], width: number, height: number): PixelSegment | null {
  if (![...a, ...b, width, height].every(Number.isFinite)) return null;
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const p = [-dx, dx, -dy, dy];
  const q = [a[0], width - a[0], a[1], height - a[1]];
  let start = 0, end = 1;
  for (let i = 0; i < 4; i++) {
    if (Math.abs(p[i]) < 1e-12) { if (q[i] < 0) return null; continue; }
    const t = q[i] / p[i];
    if (p[i] < 0) start = Math.max(start, t); else end = Math.min(end, t);
    if (start > end) return null;
  }
  return [[a[0] + start * dx, a[1] + start * dy], [a[0] + end * dx, a[1] + end * dy]];
}

function clipCameraSegment(start: Vec3, end: Vec3): [Vec3, Vec3] | null {
  if (start[2] < CLIP_NEAR && end[2] < CLIP_NEAR) return null;
  let a = start, b = end;
  if (a[2] < CLIP_NEAR || b[2] < CLIP_NEAR) {
    const t = (CLIP_NEAR - a[2]) / (b[2] - a[2]);
    const intersection = a.map((value, axis) => value + t * (b[axis] - value)) as Vec3;
    intersection[2] = CLIP_NEAR;
    if (a[2] < CLIP_NEAR) a = intersection; else b = intersection;
  }
  return [a, b];
}

function projectCameraSegment(a: Vec3, b: Vec3, camera: Camera): PixelSegment[] {
  const clipped = clipCameraSegment(a, b);
  if (!clipped) return [];
  const [start, end] = clipped;
  // Distortion bends an otherwise straight projected edge. Sample the edge in
  // 3D rather than drawing a single straight pixel chord across the image.
  const pieces = camera.distortion?.some(value => value !== 0) ? 24 : 1;
  const segments: PixelSegment[] = [];
  let previous = projectCameraPoint(start, camera);
  for (let i = 1; i <= pieces; i++) {
    const point = start.map((value, axis) => value + (end[axis] - value) * i / pieces);
    const pixel = projectCameraPoint(point, camera);
    if (previous && pixel) {
      const visible = clipPixelSegment(previous, pixel, camera.width, camera.height);
      if (visible) segments.push(visible);
    }
    previous = pixel;
  }
  return segments;
}

export function projectLidarSegment(start: number[], end: number[], camera: Camera, frame: Frame): PixelSegment[] {
  return projectCameraSegment(lidarToCamera(start, camera, frame), lidarToCamera(end, camera, frame), camera);
}

function clipPixelPolygon(points: Pixel[], width: number, height: number): Pixel[] {
  let polygon = points;
  for (const [axis, boundary, sign] of [[0, 0, 1], [0, width, -1], [1, 0, 1], [1, height, -1]]) {
    const clipped: Pixel[] = [];
    if (!polygon.length) break;
    let previous = polygon[polygon.length - 1];
    for (const point of polygon) {
      const a = (previous[axis] - boundary) * sign >= 0;
      const b = (point[axis] - boundary) * sign >= 0;
      if (a !== b) {
        const t = (boundary - previous[axis]) / (point[axis] - previous[axis]);
        clipped.push([previous[0] + (point[0] - previous[0]) * t, previous[1] + (point[1] - previous[1]) * t]);
      }
      if (b) clipped.push(point);
      previous = point;
    }
    polygon = clipped;
  }
  return polygon;
}

function projectCameraPolygon(points: Vec3[], camera: Camera): Pixel[] {
  const clipped: Vec3[] = [];
  let previous = points[points.length - 1];
  for (const point of points) {
    const a = previous[2] >= CLIP_NEAR, b = point[2] >= CLIP_NEAR;
    if (a !== b) {
      const t = (CLIP_NEAR - previous[2]) / (point[2] - previous[2]);
      const intersection = previous.map((value, axis) => value + t * (point[axis] - value)) as Vec3;
      intersection[2] = CLIP_NEAR;
      clipped.push(intersection);
    }
    if (b) clipped.push(point);
    previous = point;
  }
  if (clipped.length < 3) return [];
  const pixels: Pixel[] = [];
  const pieces = camera.distortion?.some(value => value !== 0) ? 24 : 1;
  for (let i = 0; i < clipped.length; i++) {
    const a = clipped[i], b = clipped[(i + 1) % clipped.length];
    for (let j = 0; j < pieces; j++) {
      const point = a.map((value, axis) => value + (b[axis] - value) * j / pieces);
      const pixel = projectCameraPoint(point, camera);
      if (!pixel) return [];
      pixels.push([pixel[0], pixel[1]]);
    }
  }
  return clipPixelPolygon(pixels, camera.width, camera.height);
}

export function boxCorners(box: Box): Vec3[] {
  const [length, width, height] = box.size.map(size => size / 2);
  return [[-length, -width, -height], [length, -width, -height], [length, width, -height], [-length, width, -height], [-length, -width, height], [length, -width, height], [length, width, height], [-length, width, height]].map(([x, y, z]) => [
    box.center[0] + x * Math.cos(box.yaw) - y * Math.sin(box.yaw),
    box.center[1] + x * Math.sin(box.yaw) + y * Math.cos(box.yaw),
    box.center[2] + z,
  ] as Vec3);
}

export function projectVector(vector: Vector, camera: Camera, frame: Frame): { segments: PixelSegment[]; polygon: Pixel[] } {
  const cameraPoints = vector.points.map(point => lidarToCamera(point, camera, frame));
  const segments: PixelSegment[] = [];
  const edgeCount = vector.closed ? cameraPoints.length : Math.max(0, cameraPoints.length - 1);
  for (let i = 0; i < edgeCount; i++) segments.push(...projectCameraSegment(cameraPoints[i], cameraPoints[(i + 1) % cameraPoints.length], camera));
  return { segments, polygon: vector.closed && cameraPoints.length >= 3 ? projectCameraPolygon(cameraPoints, camera) : [] };
}

export interface ProjectedResults {
  boxes: { id: string; label: string; trackId: string; segments: PixelSegment[]; anchor: Pixel | null }[];
  vectors: { id: string; label: string; segments: PixelSegment[]; polygon: Pixel[] }[];
  imagePolygons: ImagePolygon[];
}

export function projectAnnotationResults(annotations: Annotations, camera: Camera, frame: Frame): ProjectedResults {
  const boxes = annotations.boxes.map(box => {
    const points = boxCorners(box).map(point => lidarToCamera(point, camera, frame));
    const segments = EDGES.flatMap(([from, to]) => projectCameraSegment(points[from], points[to], camera));
    const top = projectLidarPoint([box.center[0], box.center[1], box.center[2] + box.size[2] / 2], camera, frame);
    const anchor = top && top[0] >= 0 && top[0] <= camera.width && top[1] >= 0 && top[1] <= camera.height
      ? [top[0], top[1]] as Pixel : segments[0]?.[0] ?? null;
    return { id: box.id, label: box.label, trackId: box.track_id, segments, anchor };
  }).filter(box => box.segments.length > 0);
  const vectors = annotations.vectors.map(vector => ({ id: vector.id, label: vector.label, ...projectVector(vector, camera, frame) }))
    .filter(vector => vector.segments.length > 0 || vector.polygon.length >= 3);
  return { boxes, vectors, imagePolygons: annotations.image_polygons.filter(polygon => polygon.camera_id === camera.id) };
}

/** Observable render geometry digest for interaction checks, not model provenance. */
export function projectionSignature(results: ProjectedResults): string {
  const geometry = JSON.stringify(results, (_, value) => typeof value === 'number' ? Math.round(value * 1000) / 1000 : value);
  let hash = 2166136261;
  for (let i = 0; i < geometry.length; i++) hash = Math.imul(hash ^ geometry.charCodeAt(i), 16777619);
  return (hash >>> 0).toString(16);
}
