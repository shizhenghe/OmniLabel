import {expect, test, type APIRequestContext, type Page} from '@playwright/test';
import type {Vec3, Vector} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const localLane: Vector = {id: 'editable-lane', label: 'lane', closed: false,
  points: [[10, -1, -1.7], [15, -1, -1.58], [20, -1, -1.45]]};

async function importLaneFixture(request: APIRequestContext, id: string, offset: Vec3,
  localVectors: Vector[] = [localLane], displayName = id) {
  const world = identity.map(row => [...row]);
  offset.forEach((value, index) => {world[index][3] = value;});
  const points = Array.from({length: 1089}, (_, index) => [7 + (index % 33) * .55,
    -6 + Math.floor(index / 33) * .375, -1.7, .4]);
  const clip = {id, name: displayName, fps: 10,
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

for (const space of ['world', 'lidar'] as const) {
  test(`lane vertex snapping uses measured XYZ in ${space}, can be disabled, and persists`, async ({page, request}) => {
    const id = `zz-lane-snap-${space}`;
    const offset: Vec3 = space === 'world' ? [500000, 3500000, 1.7] : [0, 0, 0];
    const paths = await importLaneFixture(request, id, offset);
    await openLaneFixture(page, id, space);
    await page.locator('.object-row').first().click();
    await expect(page.getByLabel('车道线吸附点云')).not.toBeChecked();
    await page.getByLabel('车道线吸附点云').check();
    await page.getByLabel('车道线吸附距离').fill('0.6');
    await dragVertex(page, 1, 25, -12);
    const vertex = page.getByTestId('vector-vertex-1');
    const actual = await Promise.all(['x', 'y', 'z'].map(async axis => Number(await vertex.getAttribute(`data-world-${axis}`))));
    const shift = space === 'world' ? offset : [0, 0, 0];
    expect(actual[2]).toBeCloseTo(-1.7 + shift[2], 6);
    expect((actual[0] - shift[0] - 7) / .55).toBeCloseTo(Math.round((actual[0] - shift[0] - 7) / .55), 6);
    expect((actual[1] - shift[1] + 6) / .375).toBeCloseTo(Math.round((actual[1] - shift[1] + 6) / .375), 6);
    await page.getByLabel('车道线吸附点云').uncheck();
    await dragVertex(page, 1, 13, -7);
    const free = await Promise.all(['x', 'y', 'z'].map(async axis => Number(await vertex.getAttribute(`data-world-${axis}`))));
    expect(Math.hypot(free[0]-actual[0], free[1]-actual[1])).toBeGreaterThan(.05);
    expect(free[2]).toBe(actual[2]);
    // A vertex far above the road is left at its own height even with snapping.
    await page.getByLabel('顶点2Z', {exact: true}).fill(String(actual[2]+2));
    await page.getByLabel('车道线吸附点云').check();
    await dragVertex(page, 1, 10, -5);
    await expect(page.getByLabel('顶点2Z', {exact: true})).toHaveValue(String(Number((actual[2]+2).toFixed(3))));
    await page.getByLabel('顶点2Z', {exact: true}).fill(String(actual[2]));
    await dragVertex(page, 1, -10, 5);
    const snapped = await Promise.all(['x', 'y', 'z'].map(async axis => Number(await vertex.getAttribute(`data-world-${axis}`))));
    const path = space === 'world' ? paths.scenePath : paths.framePath;
    const saved = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith(path));
    await page.getByLabel('保存标注', {exact: true}).click();
    expect((await saved).ok()).toBe(true);
    expect((await (await request.get(path)).json()).vectors[0].points[1]).toEqual(snapped);
    await page.getByLabel('开始绘制车道线', {exact: true}).click();
    const canvas = page.locator('canvas[aria-label="三维点云标注视图"]');
    await canvas.click({position: {x: (await canvas.boundingBox())!.width * .48, y: (await canvas.boundingBox())!.height * .48}});
    await canvas.click({position: {x: (await canvas.boundingBox())!.width * .52, y: (await canvas.boundingBox())!.height * .52}});
    await page.getByRole('button', {name: '完成绘制', exact: true}).click();
    const savedDrawing = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith(path));
    await page.getByLabel('保存标注', {exact: true}).click();
    expect((await savedDrawing).ok()).toBe(true);
    const drawing = (await (await request.get(path)).json()).vectors.find((v: Vector) => v.id !== localLane.id);
    expect(drawing.points).toHaveLength(2);
    for (const p of drawing.points) {
      expect(p[2]).toBeCloseTo(-1.7 + shift[2], 6);
      expect((p[0] - shift[0] - 7) / .55).toBeCloseTo(Math.round((p[0] - shift[0] - 7) / .55), 6);
      expect((p[1] - shift[1] + 6) / .375).toBeCloseTo(Math.round((p[1] - shift[1] + 6) / .375), 6);
    }
  });
}

for (const space of ['world', 'lidar'] as const) {
  test(`invalid ${space} lane saves identify the field, preserve drafts, and can be corrected`, async ({page, request}) => {
    const id = `zz-save-errors-${space}`;
    const paths = await importLaneFixture(request, id, [25, -8, 1.7]);
    await openLaneFixture(page, id, space);
    await page.locator('.object-row').first().click();
    await page.getByLabel('要素类别', {exact: true}).fill('');
    const path = space === 'world' ? paths.scenePath : paths.framePath;
    const failed = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith(path));
    await page.getByLabel('保存标注', {exact: true}).click();
    expect((await failed).status()).toBe(422);
    await expect(page.getByRole('alert')).toContainText(`422 · PUT ${path}`);
    await expect(page.getByRole('alert')).toContainText('vectors.0.label');
    await expect(page.getByLabel('要素类别', {exact: true})).toHaveValue('');
    expect((await (await request.get(path)).json()).vectors[0].label).toBe('lane');
    await expect(page.getByRole('alert')).toContainText('尚未保存的修改仍保留');
    await page.getByLabel('要素类别', {exact: true}).fill('lane-corrected');
    const saved = page.waitForResponse(r => r.request().method() === 'PUT' && r.url().endsWith(path));
    await page.getByLabel('保存标注', {exact: true}).click();
    expect((await saved).ok()).toBe(true);
    expect((await (await request.get(path)).json()).vectors[0].label).toBe('lane-corrected');
  });
}

for (const space of ['world', 'lidar'] as const) {
  test(`BEV pan and zoom keep equal XY lane vertices aligned across heights in ${space}`, async ({page, request}) => {
    const id = `zz-bev-pan-${space}`;
    const elevated: Vector = {id: 'vertical-check', label: 'lane', closed: false,
      points: [[15, -1, -1.7], [15, -1, 3.3], [20, -1, -1.7]]};
    const paths = await importLaneFixture(request, id, [500000, 3500000, 140], [elevated]);
    await openLaneFixture(page, id, space);
    const cloud = page.getByTestId('point-cloud-view');
    await expect(cloud).toHaveAttribute('data-projection-mode', 'orthographic');
    await page.locator('.object-row').first().click();
    const center = async (index: number) => {
      const box = await page.getByTestId(`vector-vertex-${index}`).boundingBox();
      if (!box) throw new Error('Missing vertex handle');
      return [box.x + box.width/2, box.y + box.height/2];
    };
    const separation = async () => {
      const [a, b] = await Promise.all([center(0), center(1)]);
      return Math.hypot(a[0]-b[0], a[1]-b[1]);
    };
    await expect.poll(separation).toBeLessThan(.05);
    const original = await center(0);
    const path = space === 'world' ? paths.scenePath : paths.framePath;
    const saved = await (await request.get(path)).json();
    const canvas = cloud.locator('canvas');
    const image = await canvas.screenshot();
    const bounds = (await canvas.boundingBox())!;
    await page.mouse.move(bounds.x+bounds.width*.25, bounds.y+bounds.height*.25);
    await page.mouse.down({button: 'right'});
    await page.mouse.move(bounds.x+bounds.width*.25+65, bounds.y+bounds.height*.25+35, {steps: 10});
    await page.mouse.up({button: 'right'});
    await expect.poll(async () => {const p = await center(0); return Math.hypot(p[0]-original[0], p[1]-original[1]);}).toBeGreaterThan(30);
    await expect.poll(separation).toBeLessThan(.05);
    expect(await canvas.screenshot()).not.toEqual(image);
    await page.mouse.wheel(0, -100);
    await expect.poll(separation).toBeLessThan(.05);
    await page.setViewportSize({width: 1500, height: 1000});
    await expect.poll(separation).toBeLessThan(.05);
    await page.getByRole('button', {name: '透视 3D', exact: true}).click();
    await expect(cloud).toHaveAttribute('data-projection-mode', 'perspective');
    await expect.poll(separation).toBeGreaterThan(1);
    await page.getByRole('button', {name: '俯视 BEV', exact: true}).click();
    await expect.poll(separation).toBeLessThan(.05);
    await expect(page.getByLabel('保存标注', {exact: true})).toBeDisabled();
    expect(await (await request.get(path)).json()).toEqual(saved);
  });
}


test('BEV keeps equal XY aligned across heights while panning and zooming', async ({page, request}) => {
  const id = 'zz-bev-parallax';
  const lower: Vector = {...localLane, id: 'lower', points: [[10, -1, -1.7], [20, -1, -1.7]]};
  const upper: Vector = {...lower, id: 'upper', points: lower.points.map(([x, y, z]) => [x, y, z + 5])};
  await importLaneFixture(request, id, [500000, 3500000, 1.7], [lower, upper]);
  await openLaneFixture(page, id, 'world');
  const view = page.getByTestId('point-cloud-view');
  await expect(view).toHaveAttribute('data-projection', 'orthographic');
  const compare = async () => {
    const centers: number[][] = [];
    for (const index of [0, 1]) {
      await page.locator('.object-row').nth(index).click();
      await expect(page.getByTestId('vector-vertex-gizmo')).toHaveAttribute('data-vector-id', index ? 'upper' : 'lower');
      await page.waitForTimeout(200);
      const box = await page.getByTestId('vector-vertex-0').boundingBox();
      expect(box).not.toBeNull();
      centers.push([box!.x, box!.y]);
    }
    expect(Math.abs(centers[0][0] - centers[1][0])).toBeLessThan(1);
    expect(Math.abs(centers[0][1] - centers[1][1])).toBeLessThan(1);
  };
  await compare();
  const box = (await view.boundingBox())!;
  const x = box.x + box.width / 2, y = box.y + box.height * .7;
  for (const dx of [-160, 320]) {
    await page.mouse.move(x, y); await page.mouse.down({button: 'right'});
    await page.mouse.move(x + dx, y, {steps: 15}); await page.mouse.up({button: 'right'});
    let previousX = Infinity;
    await expect.poll(async () => {
      const handle = await page.getByTestId('vector-vertex-0').boundingBox();
      const movement = Math.abs(handle!.x - previousX); previousX = handle!.x;
      return movement;
    }, {intervals: [300], timeout: 15000}).toBeLessThan(.05);
    await compare();
  }
  await page.mouse.wheel(0, -100); await page.waitForTimeout(1500);
  await compare();
  await page.getByRole('button', {name: '透视 3D', exact: true}).click();
  await expect(view).toHaveAttribute('data-projection', 'perspective');
  await page.getByRole('button', {name: '俯视 BEV', exact: true}).click();
  await compare();
  await page.getByLabel('贴合点云路面', {exact: true}).click();
  await expect(page.getByLabel('顶点1Z', {exact: true})).toHaveValue('0');
  await expect(page.getByLabel('顶点2Z', {exact: true})).toHaveValue('0');
  await expect(page.getByLabel('顶点1X', {exact: true})).toHaveValue('500010');
  await page.getByLabel('保存标注', {exact: true}).click();
  const saved = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
  expect(saved.vectors.find((v: Vector) => v.id === 'upper').points).toEqual([[500010, 3499999, 0], [500020, 3499999, 0]]);
});

test('clip names show original prefix and deletion persists across reload', async ({page, request}) => {
  const id = 'zz-delete-clip';
  await importLaneFixture(request, id, [0, 0, 0], [localLane], '原始名称 · manager1 草稿标注');
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await expect(page.locator(`#clip-select option[value="${id}"]`)).toHaveText('原始名称');
  page.on('dialog', dialog => dialog.accept());
  await page.getByLabel('删除clip', {exact: true}).click();
  await expect(page.locator(`#clip-select option[value="${id}"]`)).toHaveCount(0);
  await page.reload();
  await expect(page.getByLabel('选择clip')).toBeVisible();
  await expect(page.locator(`#clip-select option[value="${id}"]`)).toHaveCount(0);
  expect((await request.get(`/api/clips/${id}`)).status()).toBe(404);
});

test('ground fitting keeps XY and unmatched elevations, excluding roofs for Float64 maps', async () => {
  const {groundVectorHeights} = await import('../../src/geometry/pointDisplay');
  const vertices: Vec3[] = [[500000.1, 3500000.1, 3], [500010, 3500010, 4]];
  const rows = [[500000.1, 3500000.1, 10, 1], [500000.12, 3500000.12, -.12, .7]];
  for (const cloud of [rows, new Float64Array(rows.flat())]) {
    expect(groundVectorHeights(vertices, cloud, 0, .35)).toEqual({
      points: [[500000.1, 3500000.1, -.12], [500010, 3500010, 4]], matched: 1,
    });
  }
  expect(vertices[0][2]).toBe(3);
});
