import { expect, test, type Page } from '@playwright/test';

async function saveAnnotations(page: Page) {
  // The save button is also disabled while the request is in flight. Wait for
  // the actual response before reading persisted state from another client.
  const response = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith('/annotations'));
  await page.getByLabel('保存标注', { exact: true }).click();
  expect((await response).ok()).toBeTruthy();
  await expect(page.getByLabel('保存标注', { exact: true })).toBeDisabled();
}

test('seven calibrated views, cuboid edit and persistence across reload', async ({ page, request }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('单帧点云', {exact: true}).click();
  await expect(page.locator('.camera-grid .camera-card')).toHaveCount(7);
  await expect(page.getByLabel('三维点云标注视图')).toBeVisible();
  await expect(page.getByText('无法初始化 WebGL', { exact: false })).toHaveCount(0);
  await page.locator('.object-row').first().click();
  await page.getByLabel('位置X', { exact: true }).fill('18.75');
  await expect(page.getByLabel('保存标注', { exact: true })).toBeEnabled();
  const saved = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith('/annotations'));
  await page.getByLabel('保存标注', { exact: true }).click();
  expect((await saved).ok()).toBeTruthy();
  await page.reload();
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('单帧点云', {exact: true}).click();
  await page.locator('.object-row').first().click();
  await expect(page.getByLabel('位置X', { exact: true })).toHaveValue('18.75');
  await page.getByLabel('下一帧', { exact: true }).click();
  await expect(page.locator('.frame-position strong')).toHaveText('0002');
  const clip = await (await request.get('/api/clips/demo')).json();
  const projected = await request.post('/api/clips/demo/project', {
    data: { frame_id: clip.frames[1].id, camera_id: clip.cameras[0].id, stride: 2 },
  });
  expect(projected.ok()).toBeTruthy();
  const projection = await projected.json();
  expect(projection.points.length).toBeGreaterThan(100);
  expect(projection.points.every((p: { u: number; v: number; depth: number }) => p.depth > 0 && p.u >= 0 && p.u < 640 && p.v >= 0 && p.v < 360)).toBeTruthy();
  await page.screenshot({ path: 'test-results/workspace.png', fullPage: true });
  expect(errors).toEqual([]);
});

test('draw a lane and image polygon and export a semantic mask', async ({ page, request }) => {
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('单帧点云', {exact: true}).click();
  const clip = await (await request.get('/api/clips/demo')).json();
  const path = `/api/clips/demo/frames/${clip.frames[0].id}/annotations`;
  const before = await (await request.get(path)).json();
  await page.locator('.task-button').filter({ hasText: '车道线标注' }).click();
  await page.getByLabel('绘制矢量', { exact: true }).click();
  const scene = page.getByLabel('三维点云标注视图');
  await scene.click({ position: { x: 250, y: 160 } });
  await scene.click({ position: { x: 380, y: 220 } });
  await page.getByRole('button', { name: '完成绘制', exact: true }).click();
  await saveAnnotations(page);
  await page.locator('.task-button').filter({ hasText: '图像语义分割' }).click();
  const camera = page.getByTestId(`camera-${clip.cameras[0].id}`);
  const bounds = await camera.boundingBox();
  expect(bounds).not.toBeNull();
  const { width, height } = bounds!;
  await camera.click({ position: { x: width * .3, y: height * .4 } });
  await camera.click({ position: { x: width * .7, y: height * .4 } });
  await camera.click({ position: { x: width * .5, y: height * .7 } });
  await page.getByRole('button', { name: '完成多边形', exact: true }).click();
  await saveAnnotations(page);
  const after = await (await request.get(path)).json();
  expect(after.vectors.length).toBe(before.vectors.length + 1);
  expect(after.image_polygons.length).toBe(before.image_polygons.length + 1);
  const base = `/api/clips/demo/frames/${clip.frames[0].id}/masks/${clip.cameras[0].id}`;
  const mask = await request.get(`${base}.png`);
  expect(mask.ok()).toBeTruthy();
  expect((await mask.body()).subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
  const metadata = await (await request.get(`${base}.json`)).json();
  expect(metadata.width).toBe(640);
  expect(metadata.height).toBe(360);
});

test('map and occupancy jobs produce downloadable geometry', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('单帧点云', {exact: true}).click();
  await page.locator('.task-button').filter({ hasText: '4D 点云地图' }).click();
  await page.getByLabel('重建整个场景', {exact:true}).click();
  await expect(page.locator('.reconstruction-panel .job-status.completed')).toBeVisible();
  const mapDownload = page.waitForEvent('download');
  await page.getByLabel('导出场景地图', {exact:true}).click();
  expect((await mapDownload).suggestedFilename()).toContain('map');
  await page.locator('.task-button').filter({ hasText: 'OCC 真值转换' }).click();
  await page.getByRole('button', { name: '转换当前帧观测 OCC', exact: true }).click();
  await expect(page.locator('.job-status.completed').filter({hasText: 'occupancy'}).locator('small')).toHaveText('occupancy');
  const occDownload = page.waitForEvent('download');
  await page.getByLabel('导出计算结果', { exact: true }).click();
  expect((await occDownload).suggestedFilename()).toContain('occupancy');
});

test('point brush edits original point labels and cancelling a frame switch preserves edits', async ({ page, request }) => {
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('单帧点云', {exact: true}).click();
  const clip = await (await request.get('/api/clips/demo')).json();
  const path = `/api/clips/demo/frames/${clip.frames[0].id}/annotations`;
  const before = await (await request.get(path)).json();
  await page.locator('.task-button').filter({ hasText: '点云语义分割' }).click();
  await page.getByRole('button', { name: '俯视 BEV', exact: true }).click();
  await page.locator('.class-option').filter({ hasText: '建筑' }).click();
  await page.getByLabel('点云语义笔刷', { exact: true }).click();
  const canvas = page.getByLabel('三维点云标注视图');
  const bounds = await canvas.boundingBox();
  await canvas.click({ position: { x: bounds!.width / 2, y: bounds!.height / 2 } });
  await expect(page.getByLabel('保存标注', { exact: true })).toBeEnabled();
  page.once('dialog', dialog => dialog.dismiss());
  await page.getByLabel('下一帧', { exact: true }).click();
  await expect(page.locator('.frame-position strong')).toHaveText('0001');
  await expect(page.getByLabel('保存标注', { exact: true })).toBeEnabled();
  await saveAnnotations(page);
  const after = await (await request.get(path)).json();
  expect(Object.keys(after.point_labels).some(i => after.point_labels[i] === 'building' && before.point_labels[i] !== 'building')).toBeTruthy();
});

test('model response fixture remains unsaved until explicit human application and save', async ({ page, request }) => {
  const clip = await (await request.get('/api/clips/demo')).json();
  const path = `/api/clips/demo/frames/${clip.frames[0].id}/annotations`;
  const before = await (await request.get(path)).json();
  const proposal = structuredClone(before);
  proposal.boxes[0].center[0] = 36.6;
  await page.route('**/api/models', route => route.fulfill({ json: { models: [{
    id: 'sam2', name: 'Test worker fixture', tasks: ['prelabel-proposals'], available: true,
    description: 'Browser-only response fixture, no actual model inference.', url: 'https://github.com/shizhenghe/OmniLabel',
  }] } }));
  await page.route('**/api/clips/demo/jobs', async route => {
    if (route.request().postDataJSON().kind !== 'prelabel') return route.continue();
    return route.fulfill({ status: 202, json: {
      id: 'test-proposal', kind: 'prelabel', status: 'completed',
      result: { proposal, model_id: 'sam2', model_version: 'test-fixture', review_required: true,
        source: 'external-worker', clip_id: 'demo', frame_id: clip.frames[0].id },
    } });
  });
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('单帧点云', {exact: true}).click();
  await page.locator('.model-link').click();
  await page.getByLabel('预标注模型', {exact:true}).last().selectOption('sam2');
  await page.getByLabel('生成候选标注', { exact: true }).last().click();
  await expect(page.getByRole('button', { name: '应用候选标注', exact: true }).last()).toBeVisible();
  expect(await (await request.get(path)).json()).toEqual(before);
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: '应用候选标注', exact: true }).last().click();
  await expect(page.getByLabel('保存标注', { exact: true })).toBeEnabled();
  expect(await (await request.get(path)).json()).toEqual(before);
  await saveAnnotations(page);
  expect((await (await request.get(path)).json()).boxes[0].center[0]).toBe(36.6);
});

test('import a user clip through the actual file chooser', async ({ page }) => {
  const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const clip = { id: 'zz-user-clip', name: '用户导入测试', fps: 10,
    cameras: [{ id: 'front', name: 'Front', width: 100, height: 100,
      intrinsics: [[10, 0, 50], [0, 10, 50], [0, 0, 1]], camera_from_ego: identity }],
    frames: [{ id: 'f0', timestamp: 0, ego_to_world: identity, lidar_to_ego: identity,
      points: [[0, 0, 5, .5], [1, 1, 6, .8]], images: {} }],
  };
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('单帧点云', {exact: true}).click();
  await page.locator('input[type=file]').setInputFiles({ name: 'clip.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(clip)) });
  await expect(page.getByLabel('选择clip')).toHaveValue('zz-user-clip');
  await expect(page.locator('.camera-grid .camera-card')).toHaveCount(1);
  const portable = page.waitForEvent('download');
  await page.getByLabel('导出clip', { exact: true }).click();
  expect((await portable).suggestedFilename()).toBe('zz-user-clip-dataset.zip');
});
