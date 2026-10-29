import {expect, test, type APIRequestContext, type Page} from '@playwright/test';
import type {SceneAnnotations, Vector} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

async function prepareLatrFixture(request: APIRequestContext, suffix: string) {
  const id = `zz-latr-fusion-${suffix}`;
  const demo = await (await request.get('/api/clips/demo')).json();
  const frames = ['f0', 'f1'].map((frameId, index) => ({id: frameId, timestamp: index * .1,
    ego_to_world: [[1, 0, 0, 100 + index * 5], [0, 1, 0, 200], [0, 0, 1, 1.7], [0, 0, 0, 1]],
    lidar_to_ego: identity,
    points: Array.from({length: 441}, (_, point) => [6 + point % 21, -5 + Math.floor(point / 21) * .5, -1.7, .5]),
    images: {front: demo.frames[index].images.front},
  }));
  const annotations = Object.fromEntries(frames.map(frame => [frame.id, {revision: 0,
    boxes: [{id: `source-box-${frame.id}`, track_id: `source-track-${frame.id}`, label: 'vehicle',
      center: [12, 2, -.9], size: [4.2, 1.8, 1.6], yaw: .1}],
    vectors: [], point_labels: {'0': 'road', '1': 'lane_marking'},
    image_polygons: [{id: `road-${frame.id}`, camera_id: 'front', label: 'road', points: [[30, 210], [240, 210], [180, 330]]}],
  }]));
  const scene: SceneAnnotations = {revision: 0, coordinate_frame: 'world',
    vectors: [
      {id: 'lane-0', label: 'lane', closed: false, points: [[110, 199, .02], [115, 199, .02], [120, 199, .02]]},
      // Unrelated existing duplicates must not be cleaned up as a side effect of applying this frame's proposal.
      {id: 'unrelated-manual-a', label: 'lane', closed: false, points: [[150, 204, .02], [160, 204, .02], [165, 204, .02]]},
      {id: 'unrelated-manual-b', label: 'lane', closed: false, points: [[155, 204.04, .02], [165, 204.04, .02], [170, 204.04, .02]]},
    ],
    tracks: [{id: 'manual-world-box', label: 'vehicle', mode: 'static', keyframes: [
      {frame_id: 'f0', center: [112, 202, .8], size: [4.2, 1.8, 1.6], yaw: .1},
    ]}],
  };
  const imported = await request.post('/api/clips/import', {multipart: {file: {
    name: `${id}.json`, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({
      schema_version: '1.1', clip: {id, name: id, fps: 10,
        cameras: demo.cameras.filter((camera: {id: string}) => camera.id === 'front'), frames},
      annotations, scene_annotations: scene,
    })),
  }}});
  expect(imported.ok()).toBe(true);
  return {id, scenePath: `/api/clips/${id}/scene/annotations`, framePath: `/api/clips/${id}/frames/f1/annotations`};
}

async function openWithBrowserWorker(page: Page, request: APIRequestContext, id: string, includeIndependent = false) {
  // Only model discovery and the worker response are browser fixtures; fusion and persistence use the real backend.
  await page.route('**/api/models', route => route.fulfill({json: {models: [{
    id: 'latr', name: 'LATR browser fixture', tasks: ['lane'], available: true, configured: true,
    description: 'Deterministic browser-only proposal fixture', url: '', supports: {lane_vectors: true},
  }]}}));
  const submitted: Record<string, unknown>[] = [];
  await page.route(`**/api/clips/${id}/jobs`, async route => {
    const body = route.request().postDataJSON();
    submitted.push(body);
    const [scene, frame] = await Promise.all([
      request.get(`/api/clips/${id}/scene/annotations`).then(response => response.json()),
      request.get(`/api/clips/${id}/frames/${body.frame_id}/annotations`).then(response => response.json()),
    ]);
    const start = submitted.length === 1 ? 10 : 20;
    await route.fulfill({json: {id: `browser-latr-${submitted.length}`, kind: 'prelabel', status: 'completed', result: {
      model_id: 'latr', scene_revision: scene.revision, weight_variant: 'once',
      proposal: {revision: frame.revision, boxes: [], image_polygons: [], point_labels: {}, vectors: [
        // Repeated worker IDs must not replace the existing human-edited world line before fusion.
        {id: 'lane-0', label: 'lane', closed: false,
          points: [[start, -.94, -1.65], [start + 7, -.94, -1.65], [start + 15, -.94, -1.65]]},
        ...(includeIndependent ? [{id: 'independent-prediction', label: 'lane', closed: false,
          points: [[10, -4, -1.7], [20, -4, -1.7]]}] : []),
      ]},
    }}});
  });
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('选择clip')).toHaveValue(id);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-active-frame-id', 'f1');
  await page.getByLabel('预标注模型', {exact: true}).selectOption('latr');
  await expect(page.getByLabel('应用LATR时融合重复车道线', {exact: true})).toBeChecked();
  await expect(page.getByLabel('生成候选标注', {exact: true})).toBeEnabled();
  page.on('dialog', dialog => dialog.accept());
  return submitted;
}

async function generateProposal(page: Page) {
  await page.getByLabel('生成候选标注', {exact: true}).click();
  await expect(page.locator('.inline-proposal')).toContainText('LATR');
  await expect(page.getByRole('button', {name: '应用候选标注', exact: true})).toBeEnabled();
}

async function saveScene(page: Page, path: string) {
  const response = page.waitForResponse(value => value.request().method() === 'PUT' && value.url().endsWith(path));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await response).ok()).toBe(true);
  await expect(page.getByLabel('保存标注', {exact: true})).toBeDisabled();
}

function expectPreservedManualLabels(scene: SceneAnnotations, original: SceneAnnotations) {
  expect(scene.tracks).toEqual(original.tracks);
  for (const id of ['unrelated-manual-a', 'unrelated-manual-b']) {
    expect(scene.vectors.find(vector => vector.id === id)).toEqual(original.vectors.find(vector => vector.id === id));
  }
}

test('LATR world fusion retains failed proposals for retry and extends repeated predictions without changing other tasks', async ({page, request}) => {
  const {id, scenePath, framePath} = await prepareLatrFixture(request, 'retry');
  const submitted = await openWithBrowserWorker(page, request, id);
  const original: SceneAnnotations = await (await request.get(scenePath)).json();
  const originalFrame = await (await request.get(framePath)).json();
  let mergeAttempts = 0;
  const mergePayloads: {vectors: Vector[]; candidate_ids: string[]}[] = [];
  await page.route(`**/api/clips/${id}/lanes/merge-preview`, async route => {
    mergePayloads.push(route.request().postDataJSON());
    mergeAttempts++;
    if (mergeAttempts === 1) await route.fulfill({status: 500, json: {detail: 'Browser fixture: retryable fusion failure'}});
    else await route.continue();
  });
  await generateProposal(page);
  expect(submitted[0]).toMatchObject({frame_id: 'f1', model_id: 'latr', scene_mode: true, options: {camera_id: 'front'}});
  await page.getByRole('button', {name: '应用候选标注', exact: true}).click();
  await expect(page.getByRole('alert')).toContainText('候选未应用');
  await expect(page.locator('.inline-proposal')).toBeVisible();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '3');
  await expect(page.getByLabel('保存标注', {exact: true})).toBeDisabled();
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  expect(await (await request.get(framePath)).json()).toEqual(originalFrame);
  await page.getByRole('button', {name: '应用候选标注', exact: true}).click();
  await expect(page.locator('.inline-proposal')).toHaveCount(0);
  await expect(page.getByLabel('保存标注', {exact: true})).toBeEnabled();
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '3');
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  const fusion = mergePayloads.at(-1)!;
  expect(fusion.candidate_ids).toHaveLength(1);
  expect(fusion.candidate_ids[0]).not.toBe('lane-0');
  const incoming = fusion.vectors.find(vector => vector.id === fusion.candidate_ids[0])!;
  expect(incoming.points[0][0]).toBeCloseTo(115, 8);
  expect(incoming.points[0][1]).toBeCloseTo(199.06, 8);
  expect(incoming.points[0][2]).toBeCloseTo(.05, 8);
  await saveScene(page, scenePath);
  const first: SceneAnnotations = await (await request.get(scenePath)).json();
  expect(first.vectors).toHaveLength(3);
  expectPreservedManualLabels(first, original);
  const firstLane = first.vectors.find(vector => vector.id === 'lane-0')!;
  expect(firstLane.points).toEqual(expect.arrayContaining(original.vectors[0].points));
  expect(Math.max(...firstLane.points.map(point => point[0]))).toBeCloseTo(130, 1);
  expect(await (await request.get(framePath)).json()).toEqual(originalFrame);

  // A second response repeats the worker ID but adds a further segment in this frame's LiDAR coordinates.
  await generateProposal(page);
  await page.getByRole('button', {name: '应用候选标注', exact: true}).click();
  await expect(page.locator('.inline-proposal')).toHaveCount(0);
  await saveScene(page, scenePath);
  const second: SceneAnnotations = await (await request.get(scenePath)).json();
  expect(second.vectors).toHaveLength(3);
  expectPreservedManualLabels(second, original);
  const finalLane = second.vectors.find(vector => vector.id === 'lane-0')!;
  expect(Math.min(...finalLane.points.map(point => point[0]))).toBeCloseTo(110, 1);
  expect(Math.max(...finalLane.points.map(point => point[0]))).toBeCloseTo(140, 1);
  expect(await (await request.get(framePath)).json()).toEqual(originalFrame);
});

test('disabling LATR fusion appends a distinct editable candidate without overwriting colliding IDs', async ({page, request}) => {
  const {id, scenePath, framePath} = await prepareLatrFixture(request, 'disabled');
  await openWithBrowserWorker(page, request, id);
  const original: SceneAnnotations = await (await request.get(scenePath)).json();
  const originalFrame = await (await request.get(framePath)).json();
  let mergeRequests = 0;
  page.on('request', value => {if (value.url().endsWith('/lanes/merge-preview')) mergeRequests++;});
  await page.getByLabel('应用LATR时融合重复车道线', {exact: true}).uncheck();
  await generateProposal(page);
  await page.getByRole('button', {name: '应用候选标注', exact: true}).click();
  await expect(page.locator('.inline-proposal')).toHaveCount(0);
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '4');
  expect(mergeRequests).toBe(0);
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  await saveScene(page, scenePath);
  const stored: SceneAnnotations = await (await request.get(scenePath)).json();
  expect(stored.vectors).toHaveLength(4);
  expect(stored.vectors.find(vector => vector.id === 'lane-0')).toEqual(original.vectors[0]);
  expectPreservedManualLabels(stored, original);
  const appended = stored.vectors.find(vector => !original.vectors.some(value => value.id === vector.id))!;
  expect(appended.id).not.toBe('lane-0');
  expect(appended.points[0][0]).toBeCloseTo(115, 8);
  expect(appended.points.at(-1)![0]).toBeCloseTo(130, 8);
  expect(await (await request.get(framePath)).json()).toEqual(originalFrame);
});

test('undoing automatic LATR fusion retains every new prediction as a separate editable draft', async ({page, request}) => {
  const {id, scenePath, framePath} = await prepareLatrFixture(request, 'undo');
  await openWithBrowserWorker(page, request, id, true);
  const original: SceneAnnotations = await (await request.get(scenePath)).json();
  const originalFrame = await (await request.get(framePath)).json();
  await generateProposal(page);
  await page.getByRole('button', {name: '应用候选标注', exact: true}).click();
  await expect(page.locator('.inline-proposal')).toHaveCount(0);
  const cloud = page.getByTestId('point-cloud-view');
  await expect(cloud).toHaveAttribute('data-annotation-vector-count', '4');
  await expect(page.getByLabel('撤销本次融合', {exact: true})).toBeEnabled();
  expect(await (await request.get(scenePath)).json()).toEqual(original);

  await page.getByLabel('撤销本次融合', {exact: true}).click();
  await expect(cloud).toHaveAttribute('data-annotation-vector-count', '5');
  await expect(page.getByLabel('撤销本次融合', {exact: true})).toHaveCount(0);
  await expect(page.getByLabel('保存标注', {exact: true})).toBeEnabled();
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  expect(await (await request.get(framePath)).json()).toEqual(originalFrame);

  // Review ends with an explicit save: undo only separates fused lines and must not discard new lanes.
  await saveScene(page, scenePath);
  const stored: SceneAnnotations = await (await request.get(scenePath)).json();
  expect(stored.vectors).toHaveLength(5);
  expectPreservedManualLabels(stored, original);
  expect(stored.vectors.find(vector => vector.id === 'lane-0')).toEqual(original.vectors[0]);
  const independent = stored.vectors.find(vector => vector.id === 'independent-prediction')!;
  expect(independent.points).toEqual([[115, 196, 0], [125, 196, 0]]);
  const overlap = stored.vectors.find(vector => vector.id !== independent.id
    && !original.vectors.some(value => value.id === vector.id))!;
  expect(overlap.id).not.toBe('lane-0');
  expect(overlap.points[0][0]).toBeCloseTo(115, 8);
  expect(overlap.points.at(-1)![0]).toBeCloseTo(130, 8);
  expect(await (await request.get(framePath)).json()).toEqual(originalFrame);
});
