import type {Camera, Frame, ProjectionPoint} from '../types';
import {multiply} from '../scene';
import {projectCameraPoint, transformPoint} from './cameraProjection';

/** Project every loaded source point with the current (including draft) extrinsics.
 * The matrix chain is composed once per camera. Indices always refer to source
 * points; points behind the lens or outside the image are not drawn.
 */
export function projectFramePoints(frame: Frame, cameras: Camera[]): Record<string, ProjectionPoint[]> {
  return Object.fromEntries(cameras.map(camera => {
    const matrix = multiply(camera.camera_from_ego, frame.lidar_to_ego);
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
