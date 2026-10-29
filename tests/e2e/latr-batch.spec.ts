import {expect, test, type APIRequestContext, type Page} from '@playwright/test';
import type {SceneAnnotations, Vector} from '../../src/types';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
interface BatchItem {
  frame_id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'skipped';
  vector_count: number;
  error?: string;
  retryable?: boolean;
}
interface BatchParameters {
  camera_id: string;
  frame_ids?: string[];
  label: string;
  score_threshold: number;
  weight_variant: string;
}
interface BatchJob {
  id: string;
  clip_id: string;
  camera_id: string;
  status: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed';
  total: number;
  processed: number;
  succeeded: number;
  failed: number;
  skipped: number;
  vector_count: number;
  scene_revision: number;
  cancel_requested: boolean;
  created_at: string;
  updated_at: string;
  parameters: BatchParameters;
  items: BatchItem[];
}

async function fixture(request: APIRequestContext, suffix: string, count = 50) {
  const id = `zz-latr-batch-${suffix}`;
  const demo = await (await request.get('/api/clips/demo')).json();
  const frames = Array.from({length: count}, (_, index) => ({id: `f${index}`, timestamp: index * .1,
    ego_to_world: [[1, 0, 0, 100 + index * .5], [0, 1, 0, 200], [0, 0, 1, 1.7], [0, 0, 0, 1]],
    lidar_to_ego: identity, points: [[8, -2, -1.7, .2], [12, 0, -1.7, .4], [18, 2, -1.7, .6]],
    images: {front: demo.frames[0].images.front, rear_right: demo.frames[0].images.rear_right},
  }));
  const annotations = Object.fromEntries(frames.map(frame => [frame.id, {revision: 0, boxes: [], vectors: [],
    image_polygons: [{id: `road-${frame.id}`, label: 'road', camera_id: 'front', points: [[40, 210], [240, 210], [180, 330]]}],
    point_labels: {'0': 'road'},
  }]));
  const scene: SceneAnnotations = {revision: 0, coordinate_frame: 'world', vectors: [], tracks: [
    {id: 'reviewed-world-object', label: 'vehicle', mode: 'static', keyframes: [
      {frame_id: 'f0', center: [113, 202, .8], size: [4.2, 1.8, 1.6], yaw: .2},
    ]},
  ]};
  const response = await request.post('/api/clips/import', {multipart: {file: {
    name: `${id}.json`, mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({
      schema_version: '1.1', clip: {id, name: id, fps: 10,
        cameras: demo.cameras.filter((camera: {id: string}) => ['front', 'rear_right'].includes(camera.id)), frames},
      annotations, scene_annotations: scene,
    })),
  }}});
  expect(response.ok()).toBe(true);
  return {id, frames, scenePath: `/api/clips/${id}/scene/annotations`,
    framePath: `/api/clips/${id}/frames/f0/annotations`};
}

/** Only GPU/batch responses are browser fixtures. The scene, manual fusion and
 * annotation publication below use the real API and immutable input fixture. */
async function mockBatchWorker(page: Page, request: APIRequestContext, id: string, frameIds: string[], completeOnPoll = true) {
  await page.route('**/api/models', route => route.fulfill({json: {models: [{
    id: 'latr', name: 'LATR batch browser fixture', tasks: ['lane'], available: true, configured: true,
    description: 'Deterministic browser-only worker response', url: '', supports: {lane_vectors: true},
  }]}}));
  const state: {job: BatchJob | null; submitted: BatchParameters[]; completeOnPoll: boolean; retries: number} = {
    job: null, submitted: [], completeOnPoll, retries: 0,
  };
  const vectors = (items: BatchItem[]): Vector[] => items.filter(item => item.status === 'succeeded').map(item => {
    const index = frameIds.indexOf(item.frame_id);
    return {id: `batch-fixture-${id}-${index}`, label: 'lane', closed: false,
      points: [[110 + index * .5, 199, 0], [115 + index * .5, 199, 0]]};
  });
  const recalculate = (job: BatchJob) => {
    job.processed = job.items.filter(item => !['pending', 'running'].includes(item.status)).length;
    job.succeeded = job.items.filter(item => item.status === 'succeeded').length;
    job.failed = job.items.filter(item => item.status === 'failed').length;
    job.skipped = job.items.filter(item => item.status === 'skipped').length;
    job.vector_count = job.succeeded;
    return job;
  };
  await page.route(`**/api/clips/${id}/latr-batches**`, async route => {
    const path = new URL(route.request().url()).pathname;
    const base = `/api/clips/${id}/latr-batches`;
    if (path === base && route.request().method() === 'POST') {
      const parameters = route.request().postDataJSON() as BatchParameters;
      state.submitted.push(parameters);
      const saved = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
      const selected = parameters.frame_ids ?? frameIds;
      state.job = {id: 'fixture-batch', clip_id: id, camera_id: parameters.camera_id, status: 'queued',
        total: selected.length, processed: 0, succeeded: 0, failed: 0, skipped: 0, vector_count: 0,
        scene_revision: saved.revision, cancel_requested: false, created_at: '2026-10-29T00:00:00Z',
        updated_at: '2026-10-29T00:00:00Z', parameters,
        items: selected.map(frame_id => ({frame_id, status: 'pending', vector_count: 0}))};
      return route.fulfill({status: 202, json: state.job});
    }
    if (path === base) return route.fulfill({json: {latest_job: state.job, jobs: state.job ? [state.job] : []}});
    if (!state.job) return route.fulfill({status: 404, json: {detail: 'No fixture job'}});
    if (path.endsWith('/cancel')) {
      state.job.status = 'cancelled'; state.job.cancel_requested = true;
      state.job.items = state.job.items.map((item, index) => index === 0
        ? {...item, status: 'succeeded', vector_count: 1, retryable: false}
        : {...item, status: index === 1 ? 'failed' : 'skipped', vector_count: 0, retryable: true,
          error: index === 1 ? 'Fixture transient inference failure' : '用户取消，尚未执行此帧。'});
      return route.fulfill({json: recalculate(state.job)});
    }
    if (path.endsWith('/retry')) {
      state.retries++; state.completeOnPoll = true;
      state.job.status = 'queued'; state.job.cancel_requested = false;
      state.job.items = state.job.items.map(item => item.status === 'succeeded' ? item
        : {...item, status: 'pending', vector_count: 0, retryable: false, error: undefined});
      return route.fulfill({status: 202, json: recalculate(state.job)});
    }
    if (path.endsWith('/proposal')) return route.fulfill({json: {
      job_id: state.job.id, clip_id: id, camera_id: state.job.camera_id, coordinate_frame: 'world',
      scene_revision: state.job.scene_revision, vectors: vectors(state.job.items),
      source_frame_ids: state.job.items.filter(item => item.status === 'succeeded').map(item => item.frame_id),
      warnings: state.job.failed || state.job.skipped ? ['仅包含成功帧的候选。'] : [],
    }});
    if (['queued', 'running'].includes(state.job.status)) {
      state.job.status = state.completeOnPoll ? 'completed' : 'running';
      if (state.completeOnPoll) state.job.items = state.job.items.map(item => ({...item,
        status: 'succeeded', vector_count: 1, retryable: false, error: undefined}));
      else state.job.items[0].status = 'running';
    }
    return route.fulfill({json: recalculate(state.job)});
  });
  return state;
}

async function openPanel(page: Page, id: string, camera = 'front') {
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('选择clip')).toHaveValue(id);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled({timeout: 60_000});
  await page.getByLabel('预标注模型', {exact: true}).selectOption('latr');
  if (camera !== 'front') await page.getByTestId(`camera-${camera}`).click();
  await page.getByLabel('LATR批量预标注', {exact: true}).click();
  await expect(page.getByRole('dialog', {name: 'LATR批量预标注', exact: true})).toBeVisible();
}

async function applyBatch(page: Page) {
  const panel = page.getByRole('dialog', {name: 'LATR批量预标注', exact: true});
  await expect(panel.getByLabel('应用批量候选', {exact: true})).toBeEnabled();
  await panel.getByLabel('应用批量候选', {exact: true}).click();
  await expect(panel).toHaveCount(0);
}

test('fifty selected-camera predictions remain distinct until manual fusion and explicit save', async ({page, request}) => {
  const {id, frames, scenePath, framePath} = await fixture(request, 'fifty');
  const worker = await mockBatchWorker(page, request, id, frames.map(frame => frame.id));
  await openPanel(page, id);
  const original = await (await request.get(scenePath)).json();
  const originalFrame = await (await request.get(framePath)).json();
  const panel = page.getByRole('dialog', {name: 'LATR批量预标注', exact: true});
  await expect(panel.getByTestId('latr-batch-selection')).toContainText('50');
  await panel.getByLabel('启动LATR批量预标注', {exact: true}).click();
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-status', 'completed');
  expect(worker.submitted).toHaveLength(1);
  expect(worker.submitted[0]).toMatchObject({camera_id: 'front', label: 'lane', weight_variant: 'once'});
  expect(worker.submitted[0].frame_ids).toBeUndefined();
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  let fusionRequests = 0;
  page.on('request', value => {if (value.url().endsWith('/lanes/merge-preview')) fusionRequests++;});
  await applyBatch(page);
  const cloud = page.getByTestId('point-cloud-view');
  await expect(cloud).toHaveAttribute('data-coordinate-frame', 'world');
  await expect(cloud).toHaveAttribute('data-annotation-vector-count', '50');
  expect(fusionRequests).toBe(0);
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  if (!await page.getByRole('button', {name: '预览融合', exact: true}).count()) {
    await page.getByLabel('融合重复车道线', {exact: true}).click();
  }
  await page.getByRole('button', {name: '预览融合', exact: true}).click();
  await expect(page.getByTestId('lane-merge-result')).toContainText('50 → 1');
  await page.getByRole('button', {name: '应用融合', exact: true}).click();
  await expect(cloud).toHaveAttribute('data-annotation-vector-count', '1');
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  const saving = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith(scenePath));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await saving).ok()).toBe(true);
  const saved = await (await request.get(scenePath)).json();
  expect(saved.tracks).toEqual(original.tracks);
  expect(saved.vectors).toHaveLength(1);
  const points = saved.vectors[0].points as number[][];
  expect(points[0]).toEqual([110, 199, 0]);
  expect(points.at(-1)).toEqual([139.5, 199, 0]);
  expect(await (await request.get(framePath)).json()).toEqual(originalFrame);
});

test('camera-specific frame range survives close and reopen, and retry retains completed candidates', async ({page, request}) => {
  const {id, frames, scenePath} = await fixture(request, 'range', 10);
  const worker = await mockBatchWorker(page, request, id, frames.map(frame => frame.id), false);
  await openPanel(page, id, 'rear_right');
  let panel = page.getByRole('dialog', {name: 'LATR批量预标注', exact: true});
  await panel.getByLabel('批量预标注范围', {exact: true}).selectOption('range');
  await panel.getByLabel('批量起始帧', {exact: true}).fill('3');
  await panel.getByLabel('批量图像数量', {exact: true}).fill('4');
  await panel.getByLabel('批量抽帧间隔', {exact: true}).fill('2');
  await panel.getByLabel('启动LATR批量预标注', {exact: true}).click();
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-status', 'running');
  expect(worker.submitted[0]).toMatchObject({camera_id: 'rear_right', frame_ids: ['f2', 'f4', 'f6', 'f8']});
  const original = await (await request.get(scenePath)).json();
  await panel.getByLabel('关闭LATR批量预标注', {exact: true}).click();
  await expect(panel).toHaveCount(0);
  await page.getByLabel('LATR批量预标注', {exact: true}).click();
  panel = page.getByRole('dialog', {name: 'LATR批量预标注', exact: true});
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-camera-id', 'rear_right');
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-status', 'running');
  await panel.getByLabel('取消批量预标注', {exact: true}).click();
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-status', 'cancelled');
  await expect(panel.locator('.latr-batch-counts')).toContainText('成功 1');
  await expect(panel.locator('.latr-batch-counts')).toContainText('失败 1');
  await expect(panel.locator('.latr-batch-counts')).toContainText('跳过 2');
  await panel.getByLabel('重试失败帧', {exact: true}).click();
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-status', 'completed');
  expect(worker.retries).toBe(1);
  expect(worker.submitted).toHaveLength(1);
  await expect(panel.locator('.latr-batch-counts')).toContainText('成功 4');
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  await applyBatch(page);
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '4');
  expect(await (await request.get(scenePath)).json()).toEqual(original);
});

test('cancelled batch can apply only successful frames without implying full-clip completion', async ({page, request}) => {
  const {id, frames, scenePath} = await fixture(request, 'partial', 3);
  await mockBatchWorker(page, request, id, frames.map(frame => frame.id), false);
  await openPanel(page, id);
  const original = await (await request.get(scenePath)).json();
  const panel = page.getByRole('dialog', {name: 'LATR批量预标注', exact: true});
  await panel.getByLabel('启动LATR批量预标注', {exact: true}).click();
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-status', 'running');
  await panel.getByLabel('取消批量预标注', {exact: true}).click();
  await expect(panel.getByTestId('latr-batch-job')).toHaveAttribute('data-status', 'cancelled');
  await expect(panel.locator('.latr-batch-counts')).toContainText('成功 1');
  await expect(panel.locator('.latr-batch-counts')).toContainText('失败 1');
  await applyBatch(page);
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count', '1');
  expect(await (await request.get(scenePath)).json()).toEqual(original);
  const saving = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith(scenePath));
  await page.getByLabel('保存标注', {exact: true}).click();
  expect((await saving).ok()).toBe(true);
  const saved = await (await request.get(scenePath)).json();
  expect(saved.vectors).toHaveLength(1);
  expect(saved.vectors[0].points).toEqual([[110, 199, 0], [115, 199, 0]]);
  expect(saved.tracks).toEqual(original.tracks);
});
