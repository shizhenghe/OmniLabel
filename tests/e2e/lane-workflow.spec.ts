import {expect, test} from '@playwright/test';

test('discoverable lane drawing saves world vertices, edits coordinates and projects across frames', async ({page, request}) => {
  await page.goto('/');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
  await expect(page.getByTestId('lane-guide')).toContainText('左键依次加点');
  await page.getByLabel('加载道路标线示例', {exact: true}).click();
  await expect(page.getByLabel('选择clip')).toHaveValue('demo-ground-detail');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  const viewport = page.getByTestId('point-cloud-view');
  await expect(viewport).toHaveAttribute('data-coordinate-frame', 'world');
  await expect(viewport).toHaveAttribute('data-color-mode', 'intensity');
  await expect(page.getByLabel('仅看地面', {exact: true})).toBeChecked();
  const front = page.getByTestId('camera-front');
  const priorProjectionCount = Number(await front.getAttribute('data-projected-vector-count'));
  const path = '/api/clips/demo-ground-detail/scene/annotations';
  const before = await (await request.get(path)).json();
  await page.getByLabel('开始绘制车道线', {exact: true}).click();
  await expect(page.getByRole('button', {name: '完成绘制', exact: true})).toBeDisabled();
  const canvas = page.getByLabel('三维点云标注视图');
  const bounds = await canvas.boundingBox();
  for (const [x, y] of [[.45, .43], [.5, .48], [.55, .5]]) {
    await canvas.click({position: {x: bounds!.width * x, y: bounds!.height * y}});
  }
  await page.getByRole('button', {name: '使用说明', exact: true}).click();
  await page.keyboard.press('Backspace');
  await page.getByLabel('关闭对话框', {exact: true}).click();
  await expect(page.locator('.drawing-bar')).toContainText('3 个世界顶点');
  await page.keyboard.press('Backspace');
  await expect(page.locator('.drawing-bar')).toContainText('2 个世界顶点');
  await page.keyboard.press('Enter');
  await expect(page.locator('.drawing-bar')).toHaveCount(0);
  for (let i = 1; i <= 2; i++) {
    expect(Math.abs(Number(await page.getByLabel(`顶点${i}Z`, {exact: true}).inputValue()))).toBeLessThan(.35);
  }
  // Place this manually drawn world line in the calibrated front-camera frustum.
  for (const [i, x] of [[1, 12], [2, 24]]) {
    await page.getByLabel(`顶点${i}X`, {exact: true}).fill(String(x));
    await page.getByLabel(`顶点${i}Y`, {exact: true}).fill('-3.6');
    await page.getByLabel(`顶点${i}Z`, {exact: true}).fill('0.02');
  }
  await expect.poll(async () => Number(await front.getAttribute('data-projected-vector-count'))).toBeGreaterThan(priorProjectionCount);
  const saved = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith('/scene/annotations'));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await saved).ok()).toBeTruthy();
  const after = await (await request.get(path)).json();
  expect(after.vectors).toHaveLength(before.vectors.length + 1);
  const lane = after.vectors.at(-1);
  expect(lane.points).toHaveLength(2);
  for (const [index, point] of lane.points.entries()) {
    expect(point[0]).toBeCloseTo(index ? 24 : 12, 8);
    expect(point[1]).toBeCloseTo(-3.6, 8);
    expect(point[2]).toBeCloseTo(.02, 8);
  }
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(page.locator('.frame-position strong')).toHaveText('0002');
  await expect(page.getByLabel('顶点1X', {exact: true})).toHaveValue('12');
  const projectedAtNextTime = await front.getAttribute('data-projection-signature');
  await page.reload();
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  expect((await (await request.get(path)).json()).vectors.find((v: {id: string}) => v.id === lane.id)).toEqual(lane);
  expect(projectedAtNextTime).toBeTruthy();
});

test('ground view filters roofs and real XYZI contrast survives detail reconstruction', async ({page, request}) => {
  await page.goto('/');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('选择clip').selectOption('demo-ground-detail');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
  const viewport = page.getByTestId('point-cloud-view');
  await expect.poll(async () => Number(await viewport.getAttribute('data-visible-point-count'))).toBeGreaterThan(1000);
  const groundCount = Number(await viewport.getAttribute('data-visible-point-count'));
  const fullCount = Number(await viewport.getAttribute('data-cloud-point-count'));
  expect(groundCount).toBeLessThan(fullCount);
  await page.getByLabel('仅看地面', {exact: true}).uncheck();
  await expect(viewport).toHaveAttribute('data-visible-point-count', String(fullCount));
  await page.getByLabel('仅看地面', {exact: true}).check();
  await page.getByLabel('点云着色', {exact: true}).selectOption('semantic');
  await expect(viewport).toHaveAttribute('data-color-mode', 'semantic');
  await page.getByLabel('点云着色', {exact: true}).selectOption('intensity');
  const map = await (await request.get('/api/clips/demo-ground-detail/scene/map')).json();
  expect(map.preserve_detail).toBe(true);
  expect(map.effective_voxel_size).toBe(.05);
  expect(map.point_reduction).toBe('brightest-measured-point');
  const road = map.points.filter((p: number[]) => Math.abs(p[2]) < .08 && Math.abs(p[1]) < 7.9);
  expect(road.some((p: number[]) => p[3] > .8)).toBe(true);
  expect(road.some((p: number[]) => p[3] < .5)).toBe(true);
  // A known synthetic forward-arrow head is sampled in both input and map.
  expect(road.some((p: number[]) => p[0] > 13 && p[0] < 14 && Math.abs(p[1] - 1.8) < .5 && p[3] > .8)).toBe(true);
  await page.screenshot({path: 'test-results/lane-intensity-workspace.png', fullPage: true});
});
