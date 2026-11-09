import type {Camera} from '../types';
import {projectCameraPoint, transformPoint} from './cameraProjection';
let coordinates = new Float64Array(), stride = 1, version = 0;
self.onmessage = event => {
  const message = event.data;
  if (message.type === 'source') {
    coordinates = message.coordinates; stride = message.stride; version = message.version;
    return;
  }
  const output: {id: string; key: string; buffer: Float64Array}[] = [];
  if (message.version === version) for (const item of message.cameras as {camera: Camera; matrix: number[][]; key: string}[]) {
    const values: number[] = [];
    for (let offset = 0; offset < coordinates.length; offset += 3) {
      const pixel = projectCameraPoint(transformPoint(item.matrix, [coordinates[offset], coordinates[offset+1], coordinates[offset+2]]), item.camera);
      if (pixel && pixel[0] >= 0 && pixel[0] < item.camera.width && pixel[1] >= 0 && pixel[1] < item.camera.height)
        values.push(offset/3*stride, pixel[0], pixel[1], pixel[2]);
    }
    output.push({id: item.camera.id, key: item.key, buffer: new Float64Array(values)});
  }
  self.postMessage({id: message.id, version, output}, {transfer: output.map(item => item.buffer.buffer)});
};
