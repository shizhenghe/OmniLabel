import { expect, test, type Page } from '@playwright/test';

async function openScene(page: Page) {
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-coordinate-frame', 'world');
  await expect.poll(async () => Number(await page.getByTestId('point-cloud-view').getAttribute('data-cloud-point-count'))).toBeGreaterThan(1000);
  await expect(page.locator('.scene-summary')).toContainText('12 帧已融合');
}

async function drag(page: Page, id: string, dx: number, dy: number) {
  const handle = page.getByTestId(id);
  await expect(handle).toBeVisible();
  await handle.scrollIntoViewIfNeeded();
  const box = await handle.boundingBox();
  if (!box) throw new Error('Missing draggable box handle');
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + dx, y + dy, {steps: 8});
  await page.mouse.up();
}

test('whole-scene map stays fixed over time and camera result projections are left and scrollable', async ({page}) => {
  await openScene(page);
  const viewport = page.getByTestId('point-cloud-view');
  const count = await viewport.getAttribute('data-cloud-point-count');
  const rail = page.getByLabel('左侧相机标注投影');
  const cameras = page.getByTestId('camera-scroll');
  const a = await rail.boundingBox(), b = await viewport.boundingBox();
  expect(a!.x + a!.width).toBeLessThanOrEqual(b!.x + 2);
  const scrolling = await cameras.evaluate(el => ({scroll: el.scrollHeight, height: el.clientHeight, overflow: getComputedStyle(el).overflowY}));
  expect(scrolling.scroll).toBeGreaterThan(scrolling.height);
  expect(scrolling.overflow).toMatch(/auto|scroll/);
  await expect(page.getByLabel('切换点云投影')).not.toHaveClass(/\bon\b/);
  const front = page.getByTestId('camera-front');
  await expect(front).toHaveAttribute('data-image-state', 'ready');
  await expect.poll(async () => Number(await front.getAttribute('data-projected-box-count'))).toBeGreaterThan(0);
  await expect.poll(async () => Number(await front.getAttribute('data-projected-vector-count'))).toBeGreaterThan(0);
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(page.locator('.frame-position strong')).toHaveText('0002');
  await expect(viewport).toHaveAttribute('data-cloud-point-count', count!);
  await expect(viewport).toHaveAttribute('data-active-frame-id', 'frame-001');
  await page.screenshot({path: 'test-results/scene-workspace.png', fullPage: true});
});

test('actual pointer drags translate resize rotate and persist a temporal world box', async ({page, request}) => {
  await openScene(page);
  await page.locator('.object-row').first().click();
  const trackId = await page.getByLabel('轨迹ID', {exact: true}).inputValue();
  const originalX = Number(await page.getByLabel('位置X', {exact: true}).inputValue());
  const originalY = Number(await page.getByLabel('位置Y', {exact: true}).inputValue());
  const front = page.getByTestId('camera-front');
  const signature = await front.getAttribute('data-projection-signature');
  await drag(page, 'box-handle-translate-xy', 36, -22);
  await expect.poll(async () => Math.hypot(Number(await page.getByLabel('位置X', {exact: true}).inputValue()) - originalX, Number(await page.getByLabel('位置Y', {exact: true}).inputValue()) - originalY)).toBeGreaterThan(.1);
  await expect.poll(() => front.getAttribute('data-projection-signature')).not.toBe(signature);
  await page.getByLabel('缩放标注框', {exact: true}).click();
  const originalLength = Number(await page.getByLabel('尺寸长', {exact: true}).inputValue());
  await drag(page, 'box-handle-scale-x', 30, -24);
  await expect.poll(async () => Math.abs(Number(await page.getByLabel('尺寸长', {exact: true}).inputValue()) - originalLength)).toBeGreaterThan(.05);
  await page.getByLabel('旋转标注框', {exact: true}).click();
  const originalYaw = Number(await page.getByLabel('航向角', {exact: true}).inputValue());
  await drag(page, 'box-handle-rotate-z', 22, 26);
  await expect.poll(async () => Math.abs(Number(await page.getByLabel('航向角', {exact: true}).inputValue()) - originalYaw)).toBeGreaterThan(.1);
  const editedX = await page.getByLabel('位置X', {exact: true}).inputValue();
  const editedLength = await page.getByLabel('尺寸长', {exact: true}).inputValue();
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(page.locator('.frame-position strong')).toHaveText('0002');
  await page.getByLabel('上一帧', {exact: true}).click();
  await expect(page.getByLabel('位置X', {exact: true})).toHaveValue(editedX);
  await expect(page.getByLabel('尺寸长', {exact: true})).toHaveValue(editedLength);
  const saved = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith('/scene/annotations'));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await saved).ok()).toBeTruthy();
  const scene = await (await request.get('/api/clips/demo/scene/annotations')).json();
  const track = scene.tracks.find((t: {id: string}) => t.id === trackId);
  const key = track.keyframes.find((k: {frame_id: string}) => k.frame_id === 'frame-000');
  expect(key.size[0]).toBeCloseTo(Number(editedLength), 2);
  expect(key.center[0]).toBeCloseTo(Number(editedX), 2);
  await page.reload();
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.locator('.object-row').first().click();
  await expect(page.getByLabel('尺寸长', {exact: true})).toHaveValue(editedLength);
});

test('SAM2 point and box pointer prompts use original image pixels', async ({page}) => {
  await openScene(page);
  await page.getByLabel('SAM2正提示点', {exact: true}).click();
  const camera = page.getByTestId('camera-front');
  const bounds = await camera.boundingBox();
  await camera.click({position: {x: bounds!.width * .4, y: bounds!.height * .45}});
  await expect(camera).toHaveAttribute('data-model-point-count', '1');
  await page.getByLabel('SAM2负提示点', {exact: true}).click();
  await camera.click({position: {x: bounds!.width * .6, y: bounds!.height * .6}});
  await expect(camera).toHaveAttribute('data-model-point-count', '2');
  await page.getByLabel('SAM2提示框', {exact: true}).click();
  const x = bounds!.x, y = bounds!.y;
  await page.mouse.move(x + bounds!.width * .25, y + bounds!.height * .3);
  await page.mouse.down();
  await page.mouse.move(x + bounds!.width * .75, y + bounds!.height * .75, {steps: 5});
  await page.mouse.up();
  await expect(camera).toHaveAttribute('data-model-box', /,/);
  const box = (await camera.getAttribute('data-model-box'))!.split(',').map(Number);
  expect(box[2]).toBeGreaterThan(box[0]);
  expect(box[3]).toBeGreaterThan(box[1]);
  expect(box[2]).toBeLessThanOrEqual(640);
  expect(box[3]).toBeLessThanOrEqual(360);
});
