import {expect, test} from '@playwright/test';
import {projectAnnotationResults, projectionSignature, type Pixel} from '../../src/geometry/cameraProjection';
import {applyCalibration, calibrationWorldPoses, type CalibrationDocument} from '../../src/geometry/calibration';
import {sceneInLidar} from '../../src/scene';
import type {Annotations, Camera, Clip, Frame, SceneAnnotations, Vec3} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const mount = [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]];
const poses = [
  {x: 100, y: 200, z: 1.7, yaw: 10 * Math.PI / 180},
  {x: 102, y: 201, z: 1.7, yaw: 17 * Math.PI / 180},
];
const initialPoints: Vec3[] = [[120, 197, .04], [133, 197, .05]];

function suppliedMapFrames(): Frame[] {
  return poses.map((pose, index) => {
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
    // Supplied reconstructed points are already in world coordinates. The
    // imported primary-point-to-ego transform is the inverse vehicle pose.
    return {id: `frame-${String(index + 1).padStart(3, '0')}`, timestamp: index * .1,
      ego_to_world: [[c, -s, 0, pose.x], [s, c, 0, pose.y], [0, 0, 1, pose.z], [0, 0, 0, 1]],
      lidar_to_ego: [[c, s, 0, -c * pose.x - s * pose.y], [-s, c, 0, s * pose.x - c * pose.y],
        [0, 0, 1, -pose.z], [0, 0, 0, 1]],
      annotation_camera_from_lidar: {front: [
        [s, -c, 0, -s * pose.x + c * pose.y], [0, 0, -1, pose.z],
        [c, s, 0, -c * pose.x - s * pose.y], [0, 0, 0, 1],
      ]},
      points: initialPoints.map((point, i) => [...point, .4 + i * .1]), point_count: initialPoints.length, images: {},
    };
  });
}

/** Independent pinhole calculation from a physical vehicle pose. Deliberately
 * does not use inverseRigid, sceneInLidar or cameraFromLidar for expectations. */
function expectedPixel(point: Vec3, index: number, camera: Camera): Pixel {
  const pose = poses[index], c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
  const dx = point[0] - pose.x, dy = point[1] - pose.y;
  const forward = c * dx + s * dy;
  const right = s * dx - c * dy;
  const down = pose.z - point[2];
  return [camera.intrinsics[0][0] * right / forward + camera.intrinsics[0][2],
    camera.intrinsics[1][1] * down / forward + camera.intrinsics[1][2]];
}

function annotations(points = initialPoints): Annotations {
  return {revision: 0, boxes: [], vectors: [{id: 'world-lane', label: 'lane', closed: false, points}],
    image_polygons: [], point_labels: {}};
}

for (const override of ['annotation_camera_from_lidar', 'projection_camera_from_lidar'] as const) {
  test(`world lane uses each moving frame's ${override} without moving its world vertices`, () => {
    const camera: Camera = {id: 'front', name: 'Front', width: 640, height: 360,
      intrinsics: [[295, 0, 320], [0, 295, 180], [0, 0, 1]], camera_from_ego: mount};
    const world = annotations();
    const unchanged = JSON.stringify(world);
    const frames = suppliedMapFrames();
    const map = {world_from_lidar: Object.fromEntries(frames.map(frame => [frame.id, identity]))};
    for (const [index, source] of frames.entries()) {
      const frame = override === 'projection_camera_from_lidar'
        ? {...source, projection_camera_from_lidar: source.annotation_camera_from_lidar,
          // An older annotation-only correction must not override an effective
          // physical projection calibration supplied for this particular frame.
          annotation_camera_from_lidar: {front: identity}}
        : source;
      const result = projectAnnotationResults(sceneInLidar(world, frame, map), camera, frame);
      expect(result.vectors).toHaveLength(1);
      expect(result.vectors[0].segments).toHaveLength(1);
      for (const [vertex, pixel] of result.vectors[0].segments[0].entries()) {
        const expected = expectedPixel(initialPoints[vertex], index, camera);
        expect(pixel[0]).toBeCloseTo(expected[0], 8);
        expect(pixel[1]).toBeCloseTo(expected[1], 8);
      }
    }
    expect(JSON.stringify(world)).toBe(unchanged);
  });

  test(`live calibration transports ${override} across frames and preserves its existing correction`, () => {
    const camera: Camera = {id: 'front', name: 'Front', width: 640, height: 360,
      intrinsics: [[295, 0, 320], [0, 295, 180], [0, 0, 1]], camera_from_ego: mount};
    const frames = suppliedMapFrames().map((source, index) => {
      // A previously reviewed correction in ego coordinates must survive a
      // new physical camera/LiDAR mount. Use a different one in each frame.
      const correction = source.annotation_camera_from_lidar!.front.map(row => [...row]);
      correction[0][3] += .08 * (index + 1);
      correction[1][3] -= .03 * (index + 1);
      correction[2][3] += .15 * (index + 1);
      return {...source, [override]: {front: correction}};
    });
    const clip: Clip = {id: 'calibrated-4d', name: 'Calibrated 4D', description: '', fps: 10,
      source: '4d-clip-import', cameras: [camera], frames};
    const original = JSON.stringify(clip);
    const base: CalibrationDocument = {schema_version: '1.0', clip_id: clip.id, revision: 'base',
      cameras: {front: camera.camera_from_ego},
      lidar_to_ego: Object.fromEntries(frames.map(frame => [frame.id, frame.lidar_to_ego])),
      world_from_lidar: Object.fromEntries(frames.map(frame => [frame.id, identity]))};
    const shift: Vec3 = [.4, -.3, .02];
    const nextMount = mount.map(row => [...row]);
    nextMount[0][3] = .2; nextMount[1][3] = -.1; nextMount[2][3] = .05;
    const draft: CalibrationDocument = {...base, cameras: {front: nextMount},
      lidar_to_ego: Object.fromEntries(frames.map((frame, index) => {
        const c = Math.cos(poses[index].yaw), s = Math.sin(poses[index].yaw);
        const matrix = frame.lidar_to_ego.map(row => [...row]);
        matrix[0][3] += c * shift[0] + s * shift[1];
        matrix[1][3] += -s * shift[0] + c * shift[1];
        matrix[2][3] += shift[2];
        return [frame.id, matrix];
      }))};
    const currentPoses = calibrationWorldPoses(base, draft);
    const changed = applyCalibration(clip, draft);
    for (const [index, frame] of changed.frames.entries()) {
      for (const axis of [0, 1, 2]) expect(currentPoses[frame.id][axis][3]).toBeCloseTo(shift[axis], 10);
      const result = projectAnnotationResults(sceneInLidar(annotations(), frame,
        {world_from_lidar: currentPoses}), changed.cameras[0], frame);
      expect(result.vectors).toHaveLength(1);
      const pose = poses[index], c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
      for (const [vertex, pixel] of result.vectors[0].segments[0].entries()) {
        const point = initialPoints[vertex], dx = point[0] - pose.x, dy = point[1] - pose.y;
        const right = s * dx - c * dy + .08 * (index + 1) + .2;
        const down = pose.z - point[2] - .03 * (index + 1) - .1;
        const forward = c * dx + s * dy + .15 * (index + 1) + .05;
        expect(pixel[0]).toBeCloseTo(295 * right / forward + 320, 8);
        expect(pixel[1]).toBeCloseTo(295 * down / forward + 180, 8);
      }
    }
    expect(JSON.stringify(clip)).toBe(original);
    // Reopening/reapplying the saved extrinsics must not apply the same delta
    // again and gradually drift already corrected per-frame projection data.
    expect(applyCalibration(changed, draft)).toEqual(changed);
  });
}

function pointNpy(points: number[][]): Buffer {
  let header = `{'descr': '<f8', 'fortran_order': False, 'shape': (${points.length}, 4), }`;
  while ((10 + header.length + 1) % 64) header += ' ';
  header += '\n';
  const buffer = Buffer.alloc(10 + header.length + points.length * 32);
  buffer.set([0x93, ...Buffer.from('NUMPY'), 1, 0]);
  buffer.writeUInt16LE(header.length, 8);
  buffer.write(header, 10, 'ascii');
  points.flat().forEach((value, index) => buffer.writeDoubleLE(value, 10 + header.length + index * 8));
  return buffer;
}

test('global calibration preview uses source transforms instead of baking an obsolete local correction', () => {
  const camera: Camera = {id: 'front', name: 'Front', width: 640, height: 360,
    intrinsics: [[300, 0, 320], [0, 300, 180], [0, 0, 1]], camera_from_ego: identity};
  const sourceMatrix = identity.map(row => [...row]);
  sourceMatrix[0][3] = .1;
  const localMatrix = identity.map(row => [...row]);
  localMatrix[0][3] = .8;
  const frame: Frame = {id: 'f0', timestamp: 0, ego_to_world: identity, lidar_to_ego: identity,
    points: [[0, 0, 5, 1]], point_count: 1, images: {}, projection_camera_from_lidar: {front: localMatrix}};
  const clip: Clip = {id: 'preview', name: 'Preview', description: '', fps: 10, cameras: [camera], frames: [frame]};
  const baseline: CalibrationDocument = {schema_version: '1.0', clip_id: clip.id, revision: 'base',
    cameras: {front: identity}, lidar_to_ego: {f0: identity}, world_from_lidar: {f0: identity},
    frame_projection_overrides: {f0: {annotation_camera_from_lidar: {}, projection_camera_from_lidar: {front: sourceMatrix}}}};
  const nextCamera = identity.map(row => [...row]);
  nextCamera[0][3] = .2;
  const draft = {...baseline, cameras: {front: nextCamera}};
  const preview = applyCalibration(clip, draft, baseline);
  const result = projectAnnotationResults(annotations([[0, 0, 5], [1, 0, 5]]), preview.cameras[0], preview.frames[0]);
  expect(result.vectors[0].segments[0][0][0]).toBeCloseTo(338, 8);
  expect(clip.frames[0].projection_camera_from_lidar!.front[0][3]).toBe(.8);
});

test('editing one frame mount preserves other frame refinements and invalidates dependent clip refinements', () => {
  const shifted = (x: number) => identity.map((row, index) => index === 0 ? [1, 0, 0, x] : [...row]);
  const camera: Camera = {id: 'front', name: 'Front', width: 640, height: 360,
    intrinsics: [[300, 0, 320], [0, 300, 180], [0, 0, 1]], camera_from_ego: identity};
  const frames: Frame[] = ['f0', 'f1'].map((id, index) => ({id, timestamp: index * .1,
    ego_to_world: identity, lidar_to_ego: identity, points: [], point_count: 0, images: {},
    projection_camera_from_lidar: {front: shifted(.8), rear: shifted(.5)}}));
  const clip: Clip = {id: 'mixed', name: 'Mixed', description: '', fps: 10,
    cameras: [camera, {...camera, id: 'rear'}], frames};
  const base: CalibrationDocument = {schema_version: '1.0', clip_id: clip.id, revision: 'base',
    cameras: {front: identity, rear: identity}, lidar_to_ego: {f0: identity, f1: identity},
    world_from_lidar: {f0: identity, f1: identity},
    frame_projection_overrides: Object.fromEntries(frames.map(frame => [frame.id, {
      annotation_camera_from_lidar: {}, projection_camera_from_lidar: {front: shifted(.1), rear: shifted(.2)},
    }])), frame_projection_scopes: {f0: {front: 'frame', rear: 'clip'}, f1: {front: 'frame', rear: 'clip'}}};
  const preview = applyCalibration(clip, {...base, lidar_to_ego: {f0: shifted(.3), f1: identity}}, base);
  expect(preview.frames[0].projection_camera_from_lidar!.front[0][3]).toBeCloseTo(.4);
  expect(preview.frames[1].projection_camera_from_lidar!.front[0][3]).toBe(.8);
  expect(preview.frames[1].projection_camera_from_lidar!.rear[0][3]).toBe(.2);
  expect(clip.frames[0].projection_camera_from_lidar!.front[0][3]).toBe(.8);
});

for (const staleMap of [false, true]) {
test(`editing and saving a world lane uses current frame registration with ${staleMap ? 'stale' : 'matching'} supplied-map metadata`, async ({page, request}) => {
  const id = `zz-cross-frame-supplied-map-${staleMap ? 'stale' : 'matching'}`;
  const demo = await (await request.get('/api/clips/demo')).json();
  const sourceCamera = demo.cameras.find((item: Camera) => item.id === 'front') as Camera;
  const camera: Camera = {...sourceCamera, camera_from_ego: mount, distortion: []};
  const frames = suppliedMapFrames().map((frame, index) => ({...frame, images: {front: demo.frames[index].images.front}}));
  const world = annotations();
  const scene: SceneAnnotations = {revision: 0, coordinate_frame: 'world', tracks: [], vectors: world.vectors};
  const worldFromLidar = Object.fromEntries(frames.map(frame => [frame.id, identity]));
  // A later calibration/registration can update this independently of the
  // immutable supplied map. Its cached transforms must not mask current ones.
  const shift: Vec3 = staleMap ? [.4, -.3, 0] : [0, 0, 0];
  const currentWorldFromLidar = {...worldFromLidar, [frames[1].id]: [
    [1, 0, 0, shift[0]], [0, 1, 0, shift[1]], [0, 0, 1, shift[2]], [0, 0, 0, 1],
  ]};
  const imported = await request.post('/api/clips/import', {multipart: {file: {
    name: `${id}.json`, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({
      clip: {id, name: id, fps: 10, source: '4d-clip-import', cameras: [camera], frames},
      scene_annotations: scene, scene_registration: {world_from_lidar: currentWorldFromLidar},
    })),
  }}});
  expect(imported.ok(), await imported.text()).toBe(true);
  // The supplied map is a browser fixture, not a reconstruction of independent
  // scans. All label reads/writes and frame transitions below use the real API.
  await page.route(`**/api/clips/${id}/scene/map?**`, route => route.fulfill({json: {
    coordinate_frame: 'world', frame_count: frames.length, point_count: frames[0].points.length,
    trajectory: frames.map((frame, index) => ({frame_id: frame.id, timestamp: frame.timestamp,
      position: [poses[index].x, poses[index].y, poses[index].z]})),
    bounds: {min: [119, 196, 0], max: [134, 198, .1]}, point_labels: {},
    method: 'supplied reconstructed local-world map', voxel_size: .1, fusion_mode: 'full',
    world_from_lidar: worldFromLidar,
  }}));
  await page.route(`**/api/clips/${id}/scene/map.npy`, route => route.fulfill({
    contentType: 'application/octet-stream', body: pointNpy(frames[0].points),
  }));
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('选择clip')).toHaveValue(id);
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-coordinate-frame', 'world');
  await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
  await page.locator('.object-row').filter({hasText: 'lane'}).click();
  const changedPoints: Vec3[] = [[120, 197.5, .04], [133, 197, .05]];
  await page.getByLabel('顶点1Y', {exact: true}).fill('197.5');
  const front = page.getByTestId('camera-front');
  const expectedSignature = (index: number) => projectionSignature({boxes: [], vectors: [{
    id: 'world-lane', label: 'lane',
    segments: [[expectedPixel(changedPoints[0].map((value, axis) => value - (index === 1 ? shift[axis] : 0)) as Vec3, index, camera),
      expectedPixel(changedPoints[1].map((value, axis) => value - (index === 1 ? shift[axis] : 0)) as Vec3, index, camera)]], polygon: [],
  }], imagePolygons: []});
  await expect(front).toHaveAttribute('data-projection-signature', expectedSignature(0));
  const saving = page.waitForResponse(response => response.request().method() === 'PUT'
    && response.url().endsWith(`/clips/${id}/scene/annotations`));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await saving).ok()).toBe(true);
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(front).toHaveAttribute('data-active-frame-id', frames[1].id);
  await expect(front).toHaveAttribute('data-projection-signature', expectedSignature(1));
  await expect(page.getByLabel('顶点1Y', {exact: true})).toHaveValue('197.5');
  const saved = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
  expect(saved.vectors[0].points).toEqual(changedPoints);
  await page.getByLabel('上一帧', {exact: true}).click();
  await expect(front).toHaveAttribute('data-projection-signature', expectedSignature(0));
});
}
