import {expect, test, type Page} from '@playwright/test';

async function dragOutward(page: Page, id: string) {
  const face = page.getByTestId(id);
  await expect(face).toBeVisible();
  await face.scrollIntoViewIfNeeded();
  const a = await face.boundingBox();
  const b = await page.getByTestId('box-handle-translate-xy').boundingBox();
  if (!a || !b) throw new Error('Selected box must expose visible manipulation controls');
  const x = a.x + a.width / 2, y = a.y + a.height / 2;
  const dx = x - b.x - b.width / 2, dy = y - b.y - b.height / 2;
  const norm = Math.max(Math.hypot(dx, dy), 1);
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 20 * dx / norm, y + 20 * dy / norm, {steps: 8});
  await page.mouse.up();
}

test('HTTP intranet editing creates and resizes boxes and saves lane drafts while reconstruction is unavailable', async ({page, request}) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  // HTTP LAN IP origins do not expose crypto.randomUUID, unlike localhost.
  await page.addInitScript(() => Object.defineProperty(window.crypto, 'randomUUID', {value: undefined, configurable: true}));
  await page.route('**/api/clips/demo/scene/map', route => route.fulfill({status: 404, json: {detail: 'No reconstructed map'}}));
  await page.route('**/api/clips/demo/scene/reconstruct', route => route.fulfill({status: 202, json: {id: 'pending-map', kind: 'reconstruct', status: 'queued'}}));
  await page.route('**/api/jobs/pending-map', route => route.fulfill({json: {id: 'pending-map', kind: 'reconstruct', status: 'running'}}));
  const path = '/api/clips/demo/scene/annotations';
  const before = await (await request.get(path)).json();
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('新建3D框', {exact: true})).toBeEnabled();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-coordinate-frame', 'world');
  expect(await page.evaluate(() => typeof window.crypto.randomUUID)).toBe('undefined');
  await expect(page.locator('.scene-summary')).toContainText('后台重建中');
  await page.getByRole('button', {name: '俯视 BEV', exact: true}).click();
  await page.getByLabel('新建3D框', {exact: true}).click();
  const canvas = page.getByLabel('三维点云标注视图');
  const bounds = await canvas.boundingBox();
  if (!bounds) throw new Error('Missing point cloud canvas');
  await canvas.click({position: {x: bounds.width * .52, y: bounds.height * .52}});
  const id = await page.getByLabel('轨迹ID', {exact: true}).inputValue();
  expect(id).toMatch(/^track-[a-f0-9]{32}$/);
  await page.getByRole('button', {name: '透视 3D', exact: true}).click();
  await page.getByLabel('聚焦选中目标', {exact: true}).click();
  await expect(page.getByTestId('box-transform-gizmo')).toHaveAttribute('data-mode', 'scale');
  const dimensions = async () => Promise.all(['长', '宽', '高'].map(async axis => Number(await page.getByLabel(`尺寸${axis}`, {exact: true}).inputValue())));
  const initialSize = await dimensions();
  for (const [index, handle] of ['scale-x', 'scale-y-negative', 'scale-z'].entries()) {
    await dragOutward(page, `box-handle-${handle}`);
    await expect.poll(async () => (await dimensions())[index]).toBeGreaterThan(initialSize[index] + .01);
  }
  const resized = await dimensions();
  const boxSaved = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith('/scene/annotations'));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await boxSaved).ok()).toBe(true);
  const saved = await (await request.get(path)).json();
  const box = saved.tracks.find((track: {id: string}) => track.id === id);
  expect(saved.tracks).toHaveLength(before.tracks.length + 1);
  for (let index = 0; index < 3; index++) expect(box.keyframes[0].size[index]).toBeCloseTo(resized[index], 2);
  await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
  await expect(page.getByTestId('lane-guide')).toHaveCount(0);
  await expect(page.getByText('沿路面标线逐点绘制三维车道线', {exact: true})).toHaveCount(0);
  await page.getByLabel('开始绘制车道线', {exact: true}).click();
  const laneBounds = await canvas.boundingBox();
  if (!laneBounds) throw new Error('Missing lane canvas');
  for (const [x, y] of [[.4, .4], [.6, .6]]) await canvas.click({position: {x: laneBounds.width * x, y: laneBounds.height * y}});
  await expect(page.locator('.drawing-bar')).toContainText('2 个世界顶点');
  const laneSaved = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith('/scene/annotations'));
  // Save must finish and persist the current line without requiring Enter first.
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await laneSaved).ok()).toBe(true);
  await expect(page.locator('.drawing-bar')).toHaveCount(0);
  const final = await (await request.get(path)).json();
  expect(final.vectors).toHaveLength(before.vectors.length + 1);
  const line = final.vectors.at(-1);
  expect(line.points).toHaveLength(2);
  await page.reload();
  await expect(page.getByLabel('新建3D框', {exact: true})).toBeEnabled();
  const reloaded = await (await request.get(path)).json();
  expect(reloaded.vectors.find((vector: {id: string}) => vector.id === line.id)).toEqual(line);
  expect(reloaded.tracks.find((track: {id: string}) => track.id === id)).toEqual(box);
  expect(errors).toEqual([]);
});
