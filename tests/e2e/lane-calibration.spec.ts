import {expect, test} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {projectLidarPoint} from '../../src/geometry/cameraProjection';
import type {Camera, Frame, Vec3} from '../../src/types';

test('ordered 3D selection and raw-image polyline drive a real preview and debug optimization without changing source extrinsics', async ({page, request}) => {
  test.setTimeout(120_000);
  const id = 'lane-calibration-ui';
  const demo = await (await request.get('/api/clips/demo')).json();
  const camera: Camera = demo.cameras.find((c: Camera) => c.id === 'front');
  const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const lane: Vec3[] = [[10, -2, -1.6], [22, -2.5, -1.5], [40, -1, -1.3]];
  const frames = [0, 1].map(index => ({id: `f${index}`, timestamp: index * .1, ego_to_world: identity,
    lidar_to_ego: identity, points: Array.from({length: 400}, (_, n) => [8 + n % 20, -5 + Math.floor(n/20)*.5, -1.6, .5]),
    images: {front: demo.frames[index].images.front}}));
  const imported = await request.post('/api/clips/import', {multipart: {file: {name: `${id}.json`, mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({clip: {id, name: id, fps: 10, cameras: [camera], frames},
      scene_annotations: {revision: 0, coordinate_frame: 'world', tracks: [],
        vectors: [{id: 'world-lane', label: 'lane', points: lane, closed: false}]}}))}}});
  expect(imported.ok(), await imported.text()).toBe(true);
  const calibrationBefore = await (await request.get(`/api/clips/${id}/calibration`)).json();
  const sceneBefore = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('车道线外参优化', {exact: true}).click();
  const panel = page.getByTestId('lane-calibration-panel');
  await expect(panel).toBeVisible();
  await expect(panel.getByLabel('使用选中三维车道线')).toBeDisabled();
  await panel.getByLabel('优化三维车道线').selectOption('world-lane');
  await expect(panel.getByLabel('使用选中三维车道线')).toBeEnabled();
  await panel.getByLabel('使用选中三维车道线').click();
  await panel.getByLabel('绘制优化图像车道线').click();
  const image = page.getByTestId('camera-front');
  await expect(image).toHaveAttribute('data-image-state', 'ready');
  const bounds = await image.boundingBox(); expect(bounds).not.toBeNull();
  const scale = Math.min(bounds!.width/camera.width, bounds!.height/camera.height);
  for (const point of lane) {
    const pixel = projectLidarPoint(point, camera, {...frames[0], point_count: 400} as Frame)!;
    await page.mouse.click(bounds!.x + (bounds!.width-camera.width*scale)/2 + (pixel[0]+6)*scale,
      bounds!.y + (bounds!.height-camera.height*scale)/2 + (pixel[1]-4)*scale);
  }
  await expect(image).toHaveAttribute('data-lane-calibration-draft-count', '3');
  await panel.getByLabel('添加车道线配对').click();
  await expect(panel).toContainText('#1 f0 / world-lane');
  await expect(image).toHaveAttribute('data-lane-calibration-draft-count', '0');
  const endpoint = `/api/clips/${id}/lane-calibration/front`;
  const state = await (await request.get(endpoint)).json();
  expect(state.pairing.pairs).toHaveLength(1);
  expect(state.pairing.pairs[0].image_points).toHaveLength(3);
  expect(state.pairing.pairs[0].image_size).toEqual([camera.width, camera.height]);
  expect(state.pairing.pairs[0].image_points[0][0]).toBeGreaterThan(bounds!.width);
  await panel.getByLabel('确认车道线坐标约定').check();
  await panel.getByLabel('确认图像Pose同步').check();
  await panel.getByLabel('预览车道线投影误差').click();
  await expect(image).toHaveAttribute('data-lane-calibration-overlay-count', '2');
  await expect(panel).toContainText('像素误差（前 → 后）');
  await panel.getByLabel('车道线优化模式').selectOption('single');
  await panel.getByLabel('车道线优化迭代次数').fill('8');
  await panel.getByText('外参、时间戳与约束高级配置', {exact: true}).click();
  await panel.getByLabel('车道线优化高级配置').fill('{"config":{"lbfgs_iterations":0,"samples_per_lane":24}}');
  const start = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith(endpoint+'/jobs'));
  await panel.getByLabel('运行车道线外参优化').click();
  const response = await start; expect(response.status()).toBe(202);
  const jobId = (await response.json()).id;
  await expect(panel.getByText('任务：completed', {exact: true})).toBeVisible({timeout: 40_000});
  await expect(panel.getByLabel('应用车道线优化外参')).toBeDisabled();
  await expect(panel).toContainText('单帧结果仅供调试');
  await expect(image).toHaveAttribute('data-lane-calibration-overlay-count', '3');
  const downloading = page.waitForEvent('download');
  await panel.getByLabel('下载车道线优化结果').click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe(`${id}-front-lane-calibration.zip`);
  const bytes = await readFile((await download.path())!); expect(bytes.subarray(0,2).toString()).toBe('PK');
  expect((await (await request.get(endpoint+'/jobs/'+jobId)).json()).status).toBe('completed');
  expect((await (await request.get(`/api/clips/${id}/calibration`)).json()).revision).toBe(calibrationBefore.revision);
  expect(await (await request.get(`/api/clips/${id}/scene/annotations`)).json()).toEqual(sceneBefore);
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(panel).toContainText('当前帧 f1');
  await expect(image).toHaveAttribute('data-lane-calibration-overlay-count', '0');
  await page.reload();
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('车道线外参优化', {exact: true}).click();
  await expect(panel).toContainText('#1 f0 / world-lane');
  await expect(panel.getByText('任务：completed', {exact: true})).toBeVisible();
});
