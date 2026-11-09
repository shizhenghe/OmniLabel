import type {Camera, Frame, PointBuffer, ProjectionPoint} from '../types';
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

/** Reference projection for a binary world map, complete by default.
 * The interactive view uses a worker-composed raster to avoid millions of UI objects.
 * An optional explicit preview budget is supported for external callers.
 */
export function projectMapPreview(frame: Frame, camera: Camera, points: PointBuffer,
  frameFromWorld: number[][], maxSamples = Infinity): ProjectionPoint[] {
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
