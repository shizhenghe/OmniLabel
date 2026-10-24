import {expect, test} from '@playwright/test';

test('UTM point rendering and mouse resizing retain public world coordinates', async ({page, request}) => {
  const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const world = [[1, 0, 0, 500000], [0, 1, 0, 3500000], [0, 0, 1, 1.7], [0, 0, 0, 1]];
  const points = Array.from({length: 441}, (_, index) => [5 + (index % 21) * .1, -1 + Math.floor(index / 21) * .1, -1.7, .4]);
  const clip = {id: 'zz-utm-precision', name: 'UTM precision fixture', fps: 10,
    cameras: [{id: 'front', name: 'Front', width: 640, height: 360,
      intrinsics: [[295, 0, 320], [0, 295, 180], [0, 0, 1]],
      camera_from_ego: [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]}],
    frames: [{id: 'f0', timestamp: 0, ego_to_world: world, lidar_to_ego: identity, points, images: {}}]};
  const imported = await request.post('/api/clips/import', {multipart: {file: {name: 'utm.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(clip))}}});
  expect(imported.ok()).toBe(true);
  await page.goto('/');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('选择clip').selectOption(clip.id);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  const viewport = page.getByTestId('point-cloud-view');
  await expect(viewport).toHaveAttribute('data-render-origin', /^50000/);
  await page.getByLabel('新建3D框', {exact: true}).click();
  const canvas = page.getByLabel('三维点云标注视图');
  const bounds = await canvas.boundingBox();
  await canvas.click({position: {x: bounds!.width / 2, y: bounds!.height / 2}});
  await expect(page.getByTestId('box-transform-gizmo')).toBeVisible();
  const originalX = Number(await page.getByLabel('位置X', {exact: true}).inputValue());
  expect(originalX).toBeGreaterThan(499900);
  expect(originalX).toBeLessThan(500100);
  await page.getByLabel('聚焦选中目标', {exact: true}).click();
  const face = page.getByTestId('box-handle-scale-x');
  await face.scrollIntoViewIfNeeded();
  const box = await face.boundingBox();
  const center = await page.getByTestId('box-handle-translate-xy').boundingBox();
  const startX = box!.x + box!.width / 2, startY = box!.y + box!.height / 2;
  const dx = startX - center!.x - center!.width / 2, dy = startY - center!.y - center!.height / 2;
  const scale = 20 / Math.hypot(dx, dy);
  const length = Number(await page.getByLabel('尺寸长', {exact: true}).inputValue());
  await page.mouse.move(startX, startY); await page.mouse.down();
  await page.mouse.move(startX + dx * scale, startY + dy * scale, {steps: 8}); await page.mouse.up();
  await expect.poll(async () => Number(await page.getByLabel('尺寸长', {exact: true}).inputValue())).toBeGreaterThan(length + .05);
  const saving = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith('/scene/annotations'));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await saving).ok()).toBe(true);
  const scene = await (await request.get(`/api/clips/${clip.id}/scene/annotations`)).json();
  expect(scene.tracks[0].keyframes[0].center[0]).toBeGreaterThan(499900);
  expect(scene.tracks[0].keyframes[0].center[1]).toBeGreaterThan(3499900);
});
