import type {Camera, Frame, ProjectionPoint} from '../types';
import {multiply} from '../scene';
import {cameraFromLidar, projectCameraPoint, transformPoint} from './cameraProjection';

/** Project every loaded source point with the current (including draft) extrinsics.
 * The matrix chain is composed once per camera. Indices always refer to source
 * points; points behind the lens or outside the image are not drawn.
 */
export function projectFramePoints(frame: Frame, cameras: Camera[]): Record<string, ProjectionPoint[]> {
  return Object.fromEntries(cameras.map(camera => {
    const matrix = cameraFromLidar(camera, frame);
    const result: ProjectionPoint[] = [];
    for (let index = 0; index < frame.points.length; index++) {
      const pixel = projectCameraPoint(transformPoint(matrix, frame.points[index]), camera);
      if (pixel && pixel[0] >= 0 && pixel[0] < camera.width && pixel[1] >= 0 && pixel[1] < camera.height) {
        result.push({index, u: pixel[0], v: pixel[1], depth: pixel[2]});
      }
    }
    return [camera.id, result];
  }));
}

/** A bounded image overlay for a full binary world map; never modifies the map.
 * Sampling is explicit here because expanding millions of projected points into
 * browser objects would block calibration. Source indices remain unchanged.
 */
export function projectMapPreview(frame: Frame, camera: Camera, points: Float32Array,
  frameFromWorld: number[][], maxSamples = 60000): ProjectionPoint[] {
  const frameMatrix = cameraFromLidar(camera, frame);
  const matrix = multiply(frameMatrix, frameFromWorld);
  const count = points.length / 4, step = Math.max(1, Math.ceil(count / maxSamples));
  const result: ProjectionPoint[] = [];
  for (let index = 0; index < count; index += step) {
    const offset = index * 4;
    const pixel = projectCameraPoint(transformPoint(matrix, [points[offset], points[offset + 1], points[offset + 2]]), camera);
    if (pixel && pixel[0] >= 0 && pixel[0] < camera.width && pixel[1] >= 0 && pixel[1] < camera.height) {
      result.push({index, u: pixel[0], v: pixel[1], depth: pixel[2]});
    }
  }
  return result;
}
