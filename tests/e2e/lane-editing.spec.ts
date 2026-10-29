import {expect, test, type APIRequestContext, type Page} from '@playwright/test';
import type {Vec3, Vector} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const localLane: Vector = {id: 'editable-lane', label: 'lane', closed: false,
  points: [[10, -1, -1.7], [15, -1, -1.58], [20, -1, -1.45]]};

async function importLaneFixture(request: APIRequestContext, id: string, offset: Vec3,
  localVectors: Vector[] = [localLane]) {
  const world = identity.map(row => [...row]);
  offset.forEach((value, index) => {world[index][3] = value;});
  const points = Array.from({length: 1089}, (_, index) => [7 + (index % 33) * .55,
    -6 + Math.floor(index / 33) * .375, -1.7, .4]);
  const clip = {id, name: id, fps: 10,
    cameras: [{id: 'front', name: 'Front', width: 640, height: 360,
      intrinsics: [[295, 0, 320], [0, 295, 180], [0, 0, 1]],
      camera_from_ego: [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]}],
    frames: [{id: 'f0', timestamp: 0, ego_to_world: world, lidar_to_ego: identity, points, images: {}}]};
  const imported = await request.post('/api/clips/import', {multipart: {file: {
    name: `${id}.json`, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(clip)),
  }}});
  expect(imported.ok()).toBe(true);
  const framePath = `/api/clips/${id}/frames/f0/annotations`;
  const frame = await (await request.get(framePath)).json();
  expect((await request.put(framePath, {data: {...frame, vectors: localVectors}})).ok()).toBe(true);
  const scenePath = `/api/clips/${id}/scene/annotations`;
  const scene = await (await request.get(scenePath)).json();
  const worldVectors = localVectors.map(vector => ({...vector,
    points: vector.points.map(point => point.map((value, axis) => value + offset[axis]))}));
  expect((await request.put(scenePath, {data: {...scene, vectors: worldVectors}})).ok()).toBe(true);
  return {framePath, scenePath};
}

async function openLaneFixture(page: Page, id: string, space: 'world' | 'lidar') {
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('选择clip')).toHaveValue(id);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
  if (space === 'lidar') await page.getByLabel('单帧点云', {exact: true}).click();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-coordinate-frame', space);
}

async function dragVertex(page: Page, index: number, dx: number, dy: number,
  duringDrag?: () => Promise<void>) {
  const handle = page.getByTestId(`vector-vertex-${index}`);
  await expect(handle).toBeVisible();
  const bounds = await handle.boundingBox();
  if (!bounds) throw new Error('Missing draggable lane vertex');
  const x = bounds.x + bounds.width / 2, y = bounds.y + bounds.height / 2;
  await page.mouse.move(x, y); await page.mouse.down();
  await expect(page.getByTestId('vector-vertex-gizmo')).toHaveAttribute('data-dragging', 'true');
  await page.mouse.move(x + dx, y + dy, {steps: 8});
  if (duringDrag) await duringDrag();
  await page.mouse.up();
  await expect(page.getByTestId('vector-vertex-gizmo')).toHaveAttribute('data-dragging', 'false');
}

for (const space of ['world', 'lidar'] as const) {
  test(`completed lane vertices drag in ${space} coordinates with live projection and saved geometry`, async ({page, request}) => {
    const id = `zz-lane-edit-${space}`;
    const offset: Vec3 = space === 'world' ? [500000, 3500000, 1.7] : [25, -8, 1.7];
    const {framePath, scenePath} = await importLaneFixture(request, id, offset);
    await openLaneFixture(page, id, space);
    if (space === 'world') await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-render-origin', /^5000/);
    await page.locator('.object-row').first().click();
    await expect(page.getByTestId('vector-vertex-gizmo')).toHaveAttribute('data-vector-id', localLane.id);
    const read = async (vertex: number) => Promise.all(['X', 'Y', 'Z'].map(async axis =>
      Number(await page.getByLabel(`顶点${vertex}${axis}`, {exact: true}).inputValue())));
    const before = await read(2), unchangedStart = await read(1), unchangedEnd = await read(3);
    const front = page.getByTestId('camera-front');
    await expect(front).toHaveAttribute('data-projected-vector-count', '1');
    const signature = await front.getAttribute('data-projection-signature');
    await dragVertex(page, 1, 32, -18, async () => {
      await expect.poll(() => front.getAttribute('data-projection-signature')).not.toBe(signature);
      await expect.poll(async () => Math.hypot(...(await read(2)).slice(0, 2).map((value, axis) => value - before[axis]))).toBeGreaterThan(.05);
    });
    const after = await read(2);
    expect(after[2]).toBe(before[2]);
    expect(await read(1)).toEqual(unchangedStart);
    expect(await read(3)).toEqual(unchangedEnd);
    await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '1');
    await expect(page.locator('.drawing-bar')).toHaveCount(0);
    const path = space === 'world' ? scenePath : framePath;
    const saved = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith(path));
    await page.getByLabel('保存标注', {exact: true}).click();
    expect((await saved).ok()).toBe(true);
    const stored = await (await request.get(path)).json();
    expect(stored.vectors).toHaveLength(1);
    const lane = stored.vectors[0];
    expect(lane.id).toBe(localLane.id);
    after.forEach((value, axis) => expect(lane.points[1][axis]).toBeCloseTo(value, 3));
    if (space === 'world') {
      expect(lane.points[1][0]).toBeGreaterThan(500000);
      expect(lane.points[1][1]).toBeGreaterThan(3499900);
    }
    await page.reload();
    await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
    await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
    if (space === 'lidar') await page.getByLabel('单帧点云', {exact: true}).click();
    await page.locator('.object-row').first().click();
    expect(await read(2)).toEqual(after);
    if (space === 'world') {
      // Leaving a UTM world map must restore the actual source cloud, its
      // coordinate origin and a usable camera rather than retain map indices.
      await page.getByLabel('单帧点云', {exact: true}).click();
      await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-render-origin', '0,0,0');
      await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-cloud-point-count', '1089');
      await page.locator('.object-row').first().click();
      await expect(page.getByTestId('vector-vertex-1')).toBeVisible();
      expect(await read(2)).toEqual(localLane.points[1]);
    }
  });
}

test('duplicate lane fusion previews without saving, extends a lane and preserves neighbouring features', async ({page, request}) => {
  const id = 'zz-lane-fusion';
  const vectors: Vector[] = [
    {id: 'lane-a', label: 'lane', closed: false, points: [[8, 0, -1.7], [13, 0, -1.7], [18, 0, -1.7]]},
    {id: 'lane-b', label: 'lane', closed: false, points: [[26, .06, -1.67], [20, .06, -1.67], [14, .06, -1.67]]},
    {id: 'parallel', label: 'lane', closed: false, points: [[8, 3, -1.7], [18, 3, -1.7], [26, 3, -1.7]]},
    {id: 'crossing', label: 'lane', closed: false, points: [[15, -4, -1.7], [15, 4, -1.7]]},
    {id: 'map-area', label: 'crosswalk', closed: true, points: [[10, -4, -1.7], [12, -4, -1.7], [12, -2, -1.7], [10, -2, -1.7]]},
  ];
  const {scenePath} = await importLaneFixture(request, id, [0, 0, 1.7], vectors);
  await openLaneFixture(page, id, 'world');
  const original = await (await request.get(scenePath)).json();
  await page.getByLabel('融合重复车道线', {exact: true}).click();
  await page.getByRole('button', {name: '预览融合', exact: true}).click();
  await expect(page.getByRole('button', {name: '应用融合', exact: true})).toBeEnabled();
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  await page.getByRole('button', {name: '应用融合', exact: true}).click();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '4');
  // Applying a preview creates an editable draft; the user's save remains the commit point.
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  const saved = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith(scenePath));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await saved).ok()).toBe(true);
  const stored = await (await request.get(scenePath)).json();
  expect(stored.vectors).toHaveLength(4);
  const merged = stored.vectors.find((vector: Vector) => vector.id === 'lane-a');
  expect(merged).toBeTruthy();
  expect(Math.min(...merged.points.map((point: number[]) => point[0]))).toBeCloseTo(8, 1);
  expect(Math.max(...merged.points.map((point: number[]) => point[0]))).toBeCloseTo(26, 1);
  for (const id of ['parallel', 'crossing', 'map-area']) {
    expect(stored.vectors.find((vector: Vector) => vector.id === id)).toEqual(original.vectors.find((vector: Vector) => vector.id === id));
  }
  await page.reload();
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '4');
});
