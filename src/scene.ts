import type {Annotations, Box, Clip, Frame, SceneAnnotations, SceneKeyframe, SceneMap, SceneTrack, Vec3} from './types';

export const IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
export const EMPTY_ANNOTATIONS: Annotations = {boxes: [], vectors: [], image_polygons: [], point_labels: {}, revision: 0};

export function multiply(a: number[][], b: number[][]) {
  return a.map(row => b[0].map((_, column) => row.reduce((sum, value, k) => sum + value * b[k][column], 0)));
}

export function worldFromLidar(frame: Frame, map?: Pick<SceneMap, 'world_from_lidar'> | null) {
  return map?.world_from_lidar?.[frame.id] ?? multiply(frame.ego_to_world, frame.lidar_to_ego);
}

export function inverseRigid(m: number[][]) {
  return [[m[0][0], m[1][0], m[2][0], -(m[0][0] * m[0][3] + m[1][0] * m[1][3] + m[2][0] * m[2][3])],
    [m[0][1], m[1][1], m[2][1], -(m[0][1] * m[0][3] + m[1][1] * m[1][3] + m[2][1] * m[2][3])],
    [m[0][2], m[1][2], m[2][2], -(m[0][2] * m[0][3] + m[1][2] * m[1][3] + m[2][2] * m[2][3])], [0, 0, 0, 1]];
}

export function transformPoint(point: number[], matrix: number[][]): Vec3 {
  return [0, 1, 2].map(axis => matrix[axis][0] * point[0] + matrix[axis][1] * point[1] + matrix[axis][2] * point[2] + matrix[axis][3]) as Vec3;
}

export function transformBox(box: Box, matrix: number[][]): Box {
  const x = Math.cos(box.yaw), y = Math.sin(box.yaw);
  return {...box, center: transformPoint(box.center, matrix),
    yaw: Math.atan2(matrix[1][0] * x + matrix[1][1] * y, matrix[0][0] * x + matrix[0][1] * y)};
}

export function evaluateTrack(track: SceneTrack, frame: Frame, clip: Clip,
  timestamps = new Map(clip.frames.map(f => [f.id, f.timestamp]))): Box | null {
  const keys = track.keyframes.filter(k => timestamps.has(k.frame_id)).slice().sort((a, b) => timestamps.get(a.frame_id)! - timestamps.get(b.frame_id)!);
  if (!keys.length) return null;
  let state = keys[0];
  if (track.mode === 'dynamic') {
    if (frame.timestamp >= timestamps.get(keys[keys.length - 1].frame_id)!) state = keys[keys.length - 1];
    else if (frame.timestamp > timestamps.get(keys[0].frame_id)!) {
      const right = keys.findIndex(k => timestamps.get(k.frame_id)! >= frame.timestamp);
      const a = keys[right - 1], b = keys[right];
      const t = (frame.timestamp - timestamps.get(a.frame_id)!) / (timestamps.get(b.frame_id)! - timestamps.get(a.frame_id)!);
      const blend = (x: Vec3, y: Vec3) => x.map((value, axis) => value + t * (y[axis] - value)) as Vec3;
      const delta = Math.atan2(Math.sin(b.yaw - a.yaw), Math.cos(b.yaw - a.yaw));
      state = {frame_id: frame.id, center: blend(a.center, b.center), size: blend(a.size, b.size), yaw: a.yaw + t * delta};
    }
  }
  return {id: track.id, track_id: track.id, label: track.label, center: [...state.center], size: [...state.size], yaw: state.yaw};
}

export function evaluateScene(scene: SceneAnnotations | null, frame: Frame | null, clip: Clip | null): Annotations {
  if (!scene || !frame || !clip) return EMPTY_ANNOTATIONS;
  const timestamps = new Map(clip.frames.map(f => [f.id, f.timestamp]));
  return {boxes: scene.tracks.map(t => evaluateTrack(t, frame, clip, timestamps)).filter((b): b is Box => b !== null),
    vectors: scene.vectors, image_polygons: [], point_labels: {}, revision: scene.revision};
}

export function sceneInLidar(scene: Annotations, frame: Frame, map?: Pick<SceneMap, 'world_from_lidar'> | null): Annotations {
  const inverse = inverseRigid(worldFromLidar(frame, map));
  return {...scene, boxes: scene.boxes.map(b => transformBox(b, inverse)),
    vectors: scene.vectors.map(v => ({...v, points: v.points.map(p => transformPoint(p, inverse))}))};
}

export function upsertKeyframe(track: SceneTrack, box: Box, frameId: string): SceneTrack {
  const key: SceneKeyframe = {frame_id: frameId, center: [...box.center], size: [...box.size], yaw: box.yaw};
  // A static track has one world-space state. Editing it changes the whole scene.
  return {...track, id: box.track_id, label: box.label,
    keyframes: track.mode === 'static' ? [key] : [...track.keyframes.filter(k => k.frame_id !== frameId), key]};
}
