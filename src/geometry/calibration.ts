import type {Clip, Frame} from '../types';

export type Matrix4 = number[][];
export interface ExtrinsicCorrection {translation: [number, number, number]; rotation: [number, number, number]}
export interface CalibrationDocument {
  schema_version: string;
  clip_id: string;
  revision: string;
  cameras: Record<string, Matrix4>;
  lidar_to_ego: Record<string, Matrix4>;
  world_from_lidar: Record<string, Matrix4>;
  /** Source transforms, excluding per-frame/clip registration overlays. */
  frame_projection_overrides?: Record<string, Pick<Frame, 'annotation_camera_from_lidar' | 'projection_camera_from_lidar'>>;
  frame_projection_scopes?: Record<string, Record<string, 'frame' | 'clip' | 'source'>>;
  lidar_changed?: boolean;
  map_rebuild_required?: boolean;
}

export const ZERO_CORRECTION: ExtrinsicCorrection = {translation: [0, 0, 0], rotation: [0, 0, 0]};

/** Rz(yaw) Ry(pitch) Rx(roll) correction in the transform's destination frame.
 * Translation is added in that same frame; the stored baseline is not decomposed
 * to Euler angles, avoiding gimbal-lock drift when a control has zero correction.
 */
export function correctExtrinsic(base: Matrix4, correction: ExtrinsicCorrection): Matrix4 {
  const [roll, pitch, yaw] = correction.rotation.map(value => value * Math.PI / 180);
  const [cr, sr, cp, sp, cy, sy] = [Math.cos(roll), Math.sin(roll), Math.cos(pitch), Math.sin(pitch), Math.cos(yaw), Math.sin(yaw)];
  const rotation = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]];
  return [...rotation.map((row, i) => [...row.map((_, j) => row.reduce((sum, value, k) => sum + value * base[k][j], 0)),
    base[i][3] + correction.translation[i]]), [0, 0, 0, 1]];
}

export function applyCalibration(clip: Clip, calibration: Pick<CalibrationDocument, 'cameras' | 'lidar_to_ego'>,
  baseline?: CalibrationDocument): Clip {
  const cameras = new Map(clip.cameras.map(camera => [camera.id, camera]));
  const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
  const anyLidarChanged = baseline && Object.entries(baseline.lidar_to_ego).some(([id, matrix]) =>
    !same(calibration.lidar_to_ego[id] ?? matrix, matrix));
  return {...clip, cameras: clip.cameras.map(camera => ({...camera, camera_from_ego: calibration.cameras[camera.id] ?? camera.camera_from_ego})),
    frames: clip.frames.map(frame => {
      const lidar = calibration.lidar_to_ego[frame.id] ?? frame.lidar_to_ego;
      const source = baseline?.frame_projection_overrides?.[frame.id];
      const oldLidar = source ? baseline!.lidar_to_ego[frame.id] : frame.lidar_to_ego;
      const transport = (overrides: Record<string, Matrix4> | undefined) => overrides && Object.fromEntries(
        Object.entries(overrides).map(([id, matrix]) => {
          const camera = cameras.get(id);
          if (!camera) return [id, matrix];
          const oldCamera = source ? baseline!.cameras[id] : camera.camera_from_ego;
          const nextCamera = calibration.cameras[id] ?? camera.camera_from_ego;
          if (same([nextCamera, lidar], [oldCamera, oldLidar])) return [id, matrix];
          // Imported maps have direct per-frame transforms. Keep their existing
          // correction in ego coordinates while changing either sensor mount;
          // otherwise these overrides hide the edited extrinsics completely.
          return [id, multiplyExtrinsics(multiplyExtrinsics(multiplyExtrinsics(
            multiplyExtrinsics(nextCamera, inverseExtrinsic(oldCamera)), matrix),
            inverseExtrinsic(oldLidar)), lidar)];
        }));
      let projections = transport(source ? source.projection_camera_from_lidar : frame.projection_camera_from_lidar);
      // Match the server's overlay invalidation: a local refinement survives
      // edits to other frames/cameras; a clip refinement also depends on all
      // frame mounts. Start each preview from the original effective clip.
      for (const [id, scope] of Object.entries(baseline?.frame_projection_scopes?.[frame.id] ?? {})) {
        const cameraUnchanged = same(calibration.cameras[id], baseline!.cameras[id]);
        const keep = cameraUnchanged && (scope === 'frame' ? same(lidar, oldLidar) : scope === 'clip' && !anyLidarChanged);
        if (keep && frame.projection_camera_from_lidar?.[id]) {
          (projections ??= {})[id] = frame.projection_camera_from_lidar[id];
        }
      }
      return {...frame, lidar_to_ego: lidar,
        annotation_camera_from_lidar: transport(source ? source.annotation_camera_from_lidar : frame.annotation_camera_from_lidar),
        projection_camera_from_lidar: projections};
    })};
}

export function multiplyExtrinsics(a: Matrix4, b: Matrix4): Matrix4 {
  return a.map(row => b[0].map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));
}

export function inverseExtrinsic(matrix: Matrix4): Matrix4 {
  const rotation = [0, 1, 2].map(i => [0, 1, 2].map(j => matrix[j][i]));
  return [...rotation.map(row => [...row, -row.reduce((sum, value, j) => sum + value * matrix[j][3], 0)]), [0, 0, 0, 1]];
}

/** Keep the estimated ego pose when previewing a modified LiDAR mount. */
export function calibrationWorldPoses(base: CalibrationDocument, draft: CalibrationDocument): Record<string, Matrix4> {
  return Object.fromEntries(Object.entries(base.world_from_lidar).map(([id, pose]) => [id,
    multiplyExtrinsics(multiplyExtrinsics(pose, inverseExtrinsic(base.lidar_to_ego[id])), draft.lidar_to_ego[id])]));
}
