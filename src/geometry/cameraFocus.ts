import type {Camera, Frame} from '../types';
import {cameraFromLidar} from './cameraProjection';
import {IDENTITY, inverseRigid, multiply, worldFromLidar} from '../scene';

export interface CameraFocus {
  request: number; cameraId: string; frameId: string;
  worldFromCamera: number[][]; intrinsics: number[][]; width: number; height: number;
}

export function cameraFocus(camera: Camera, frame: Frame, sceneMode: boolean,
  map: Parameters<typeof worldFromLidar>[1], request: number): CameraFocus {
  return {request, cameraId: camera.id, frameId: frame.id,
    worldFromCamera: multiply(sceneMode ? worldFromLidar(frame, map) : IDENTITY,
      inverseRigid(cameraFromLidar(camera, frame))),
    intrinsics: camera.intrinsics, width: camera.width, height: camera.height};
}

/** Fit the calibrated pinhole image into the viewport, including principal point and skew. */
export function focusProjection(focus: CameraFocus, width: number, height: number, near: number, far: number) {
  const scale = Math.min(width / focus.width, height / focus.height);
  const k = focus.intrinsics;
  const cx = (width-focus.width*scale)/2+k[0][2]*scale;
  const cy = (height-focus.height*scale)/2+k[1][2]*scale;
  return [[2*k[0][0]*scale/width, -2*k[0][1]*scale/width, 1-2*cx/width, 0],
    [-2*k[1][0]*scale/height, 2*k[1][1]*scale/height, 2*cy/height-1, 0],
    [0,0,-(far+near)/(far-near),-2*far*near/(far-near)], [0,0,-1,0]];
}
