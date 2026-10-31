import {expect, test} from '@playwright/test';
import {projectionSignature, type Pixel} from '../../src/geometry/cameraProjection';
import type {Camera, Vec3} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const mount = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]];
const poses = [
  {x: 100, y: 200, z: 1.7, yaw: 10 * Math.PI / 180},
  {x: 102, y: 201, z: 1.7, yaw: 17 * Math.PI / 180},
];
const lane: Vec3[] = [[120, 197, .04], [133, 197, .05]];

function pixel(point: Vec3, index: number, camera: Camera, rightCorrection = 0): Pixel {
  const pose = poses[index], c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
  const dx = point[0] - pose.x, dy = point[1] - pose.y;
  const depth = c * dx + s * dy;
  return [camera.intrinsics[0][0] * (s * dx - c * dy + rightCorrection) / depth + camera.intrinsics[0][2],
    camera.intrinsics[1][1] * (pose.z - point[2]) / depth + camera.intrinsics[1][2]];
}

function npy(points: number[][]): Buffer {
  let header = `{'descr': '<f8', 'fortran_order': False, 'shape': (${points.length}, 4), }`;
  while ((10 + header.length + 1) % 64) header += ' ';
  header += '\n';
  const buffer = Buffer.alloc(10 + header.length + points.length * 32);
  buffer.set([0x93, ...Buffer.from('NUMPY'), 1, 0]); buffer.writeUInt16LE(header.length, 8);
  buffer.write(header, 10, 'ascii');
  points.flat().forEach((value, index) => buffer.writeDoubleLE(value, 10 + header.length + index * 8));
  return buffer;
}

test('saved frame correction can be promoted to clip and follows each frame pose in images', async ({page, request}) => {
  const id = 'zz-projection-clip-scope';
  const demo = await (await request.get('/api/clips/demo')).json();
  const camera: Camera = {...demo.cameras.find((item: Camera) => item.id === 'front'),
    camera_from_ego: mount, distortion: []};
  const other: Camera = {...camera, id: 'rear_right', name: 'Other camera'};
  const points = lane.map((point, index) => [...point, .5 + index * .1]);
  const frames = poses.map((pose, index) => {
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
    const external = [[s, -c, 0, -s * pose.x + c * pose.y], [0, 0, -1, pose.z],
      [c, s, 0, -c * pose.x - s * pose.y], [0, 0, 0, 1]];
    return {id: `frame-${String(index + 1).padStart(3, '0')}`, timestamp: index * .1,
      ego_to_world: [[c, -s, 0, pose.x], [s, c, 0, pose.y], [0, 0, 1, pose.z], [0, 0, 0, 1]],
      lidar_to_ego: [[c, s, 0, -c * pose.x - s * pose.y], [-s, c, 0, s * pose.x - c * pose.y],
        [0, 0, 1, -pose.z], [0, 0, 0, 1]],
      points, point_count: points.length,
      annotation_camera_from_lidar: {front: external, rear_right: external},
      images: {front: demo.frames[index].images.front, rear_right: demo.frames[index].images.rear_right},
    };
  });
  const scene = {revision: 0, coordinate_frame: 'world', tracks: [],
    vectors: [{id: 'world-lane', label: 'lane', points: lane, closed: false}]};
  const registration = Object.fromEntries(frames.map(frame => [frame.id, identity]));
  const imported = await request.post('/api/clips/import', {multipart: {file: {
    name: `${id}.json`, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({
      clip: {id, name: id, source: '4d-clip-import', fps: 10, cameras: [camera, other], frames},
      scene_annotations: scene, scene_registration: {world_from_lidar: registration},
    })),
  }}});
  expect(imported.ok(), await imported.text()).toBe(true);
  const base = `/api/clips/${id}/frames/${frames[0].id}/projection-calibration/front`;
  const before = await (await request.get(base)).json();
  const correction = .25;
  const matrix = before.base_matrix.map((row: number[]) => [...row]); matrix[0][3] += correction;
  const correspondences: {id: string; point: Vec3; pixel: Pixel}[] = [
    [108, 195, 0], [110, 205, 0], [118, 194, 0], [125, 206, 0],
    [115, 198, 2], [121, 203, 3], [130, 197, .5], [140, 202, 1],
  ].map((point, index) => ({id: `anchor-${index + 1}`, point: point as Vec3,
    pixel: pixel(point as Vec3, 0, camera, correction)}));
  const savedFrame = await request.put(base, {data: {revision: before.revision, scope: 'frame',
    matrix, correspondences, method: 'correspondences'}});
  expect(savedFrame.ok(), await savedFrame.text()).toBe(true);
  const nextBase = `/api/clips/${id}/frames/${frames[1].id}/projection-calibration/front`;
  expect((await (await request.get(nextBase)).json()).scope).toBe('source');
  // Only the prebuilt binary world map transport is a browser fixture. The
  // calibration documents, validation, inheritance and saving use real APIs.
  await page.route(`**/api/clips/${id}/scene/map?**`, route => route.fulfill({json: {
    coordinate_frame: 'world', frame_count: frames.length, point_count: points.length,
    trajectory: frames.map((frame, index) => ({frame_id: frame.id, timestamp: frame.timestamp,
      position: [poses[index].x, poses[index].y, poses[index].z]})),
    bounds: {min: [119, 196, 0], max: [134, 198, .1]}, point_labels: {},
    method: 'supplied reconstructed local-world map', voxel_size: .1, fusion_mode: 'full',
    world_from_lidar: registration,
  }}));
  await page.route(`**/api/clips/${id}/scene/map.npy`, route => route.fulfill({
    contentType: 'application/octet-stream', body: npy(points),
  }));
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('选择clip')).toHaveValue(id);
  await expect(page.getByLabel('逐帧投影校准', {exact: true})).toBeEnabled();
  const signature = (index: number, delta = 0) => projectionSignature({boxes: [], vectors: [{
    id: 'world-lane', label: 'lane', segments: [[pixel(lane[0], index, camera, delta), pixel(lane[1], index, camera, delta)]], polygon: [],
  }], imagePolygons: []});
  const front = page.getByTestId('camera-front');
  const rear = page.getByTestId('camera-rear_right');
  await expect(front).toHaveAttribute('data-projection-signature', signature(0, correction));
  await expect(rear).toHaveAttribute('data-projection-signature', signature(0));
  await page.getByLabel('逐帧投影校准', {exact: true}).click();
  const editor = page.getByTestId('projection-calibration-editor');
  await expect(editor.getByLabel('投影修正应用范围')).toHaveValue('frame');
  await expect(editor.getByLabel('将当前修正应用整个clip')).toBeEnabled();
  const promoting = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith(base));
  await editor.getByLabel('将当前修正应用整个clip').click();
  expect((await promoting).ok()).toBe(true);
  await expect(editor.getByText('已应用于整个 clip 的当前相机，切帧时按对应帧位姿投影；已有独立逐帧修正保留。')).toBeVisible();
  await editor.getByLabel('关闭投影校正', {exact: true}).click();
  const inherited = await (await request.get(nextBase)).json();
  expect(inherited.scope).toBe('clip'); expect(inherited.inherited).toBe(true);
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(front).toHaveAttribute('data-active-frame-id', frames[1].id);
  await expect(front).toHaveAttribute('data-projection-signature', signature(1, correction));
  await expect(rear).toHaveAttribute('data-projection-signature', signature(1));
  expect((await (await request.get(`/api/clips/${id}/scene/annotations`)).json()).vectors).toEqual(scene.vectors);
  // Clearing only the clip scope returns this inherited frame to its source
  // projection while retaining the explicitly saved reference-frame result.
  await page.getByLabel('逐帧投影校准', {exact: true}).click();
  await editor.getByLabel('投影修正应用范围').selectOption('clip');
  await expect(editor.getByLabel('清除整个clip投影修正')).toBeEnabled();
  await editor.getByLabel('清除整个clip投影修正').click();
  await expect(editor.getByText('已清除当前相机的整个 clip 修正；独立逐帧修正保留。')).toBeVisible();
  await expect(front).toHaveAttribute('data-projection-signature', signature(1));
  await editor.getByLabel('关闭投影校正', {exact: true}).click();
  await page.getByLabel('上一帧', {exact: true}).click();
  await expect(front).toHaveAttribute('data-projection-signature', signature(0, correction));
});
