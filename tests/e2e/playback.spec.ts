import {expect, test, type APIRequestContext, type Page} from '@playwright/test';

const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const frameIds = ['review-0', 'review-1', 'review-2'];

async function prepareClip(request: APIRequestContext, suffix: string) {
  const demo = await (await request.get('/api/clips/demo')).json();
  const id = `zz-review-${suffix}`;
  const frames = frameIds.map((frameId, index) => ({id: frameId, timestamp: index * 1.2,
    ego_to_world: [[1, 0, 0, index * .2], [0, 1, 0, 0], [0, 0, 1, 1.7], [0, 0, 0, 1]],
    lidar_to_ego: identity,
    points: Array.from({length: index + 4}, (_, point) => [3 + point * .25, 1, -1.7, .3 + point * .05]),
    images: demo.frames[index].images,
  }));
  const annotations = Object.fromEntries(frameIds.map((frameId, index) => [frameId, {
    revision: 0, boxes: [], vectors: [], point_labels: {'0': 'road', '1': 'road'},
    image_polygons: [{id: `review-mask-${index}`, camera_id: 'front', label: 'road',
      points: [[50 + index * 15, 210], [180 + index * 15, 210], [150, 320]]}],
  }]));
  const scene = {revision: 0, coordinate_frame: 'world',
    vectors: [{id: 'review-lane', label: 'lane', closed: false, points: [[6, -2, .02], [20, -2, .02]]}],
    tracks: [{id: 'review-car', label: 'vehicle', mode: 'dynamic', keyframes: [
      {frame_id: frameIds[0], center: [12, 0, .8], size: [4.2, 1.8, 1.6], yaw: 0},
      {frame_id: frameIds[2], center: [16, 0, .8], size: [4.2, 1.8, 1.6], yaw: .1},
    ]}],
  };
  const result = await request.post('/api/clips/import', {multipart: {file: {
    name: `${id}.json`, mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({schema_version: '1.1', clip: {
      id, name: `回放验证 ${suffix}`, description: 'Three real imported frame/image/annotation records', fps: 10,
      cameras: demo.cameras, frames,
    }, annotations, scene_annotations: scene})),
  }}});
  expect(result.ok()).toBe(true);
  return id;
}

async function openClip(page: Page, id: string) {
  await page.goto('/');
  await expect(page.getByLabel('选择clip')).toHaveValue('demo');
  await expect(page.getByLabel('选择clip')).toBeEnabled();
  await page.getByLabel('选择clip').selectOption(id);
  await expect(page.getByLabel('选择clip')).toHaveValue(id);
  await expect(page.getByLabel('回放标注clip', {exact: true})).toBeEnabled();
}

async function observeFrames(page: Page) {
  await page.evaluate(() => {
    const state = window as unknown as {reviewSamples: string[]; reviewMismatches: string[]};
    state.reviewSamples = []; state.reviewMismatches = [];
    const record = () => {
      const viewport = document.querySelector('[data-testid="point-cloud-view"]') as HTMLElement | null;
      const id = viewport?.dataset.activeFrameId;
      if (!id) return;
      if (state.reviewSamples.at(-1) !== id) state.reviewSamples.push(id);
      const cameras = [...document.querySelectorAll('[data-testid^="camera-"]')].filter(el => el.hasAttribute('data-active-frame-id'));
      for (const camera of cameras) {
        const element = camera as HTMLElement;
        if (element.dataset.activeFrameId !== id || element.dataset.renderedImageFrameId !== id) state.reviewMismatches.push(id);
      }
    };
    const observer = new MutationObserver(record);
    observer.observe(document.body, {subtree: true, attributes: true, attributeFilter: ['data-active-frame-id', 'data-rendered-image-frame-id']});
    record();
  });
}

test('clip replay advances synchronized images and cloud annotations, pauses and ends at the final timestamp', async ({page, request}) => {
  const id = await prepareClip(request, 'complete');
  await openClip(page, id);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('回放标注clip', {exact: true}).click();
  await expect(page.getByLabel('退出回放', {exact: true})).toBeVisible();
  await page.getByLabel('暂停', {exact: true}).click();
  await page.getByLabel('跳转到第1帧', {exact: true}).click();
  await expect(page.getByTestId('camera-front')).toHaveAttribute('data-rendered-image-frame-id', frameIds[0]);
  await observeFrames(page);
  await page.getByLabel('回放点云视图', {exact: true}).selectOption('frame');
  await page.getByLabel('回放速度', {exact: true}).selectOption('0.5');
  await expect(page.getByLabel('回放速度', {exact: true})).toHaveValue('0.5');
  const cloud = page.getByTestId('point-cloud-view');
  await expect(cloud).toHaveAttribute('data-annotation-box-count', '1');
  await expect(cloud).toHaveAttribute('data-annotation-vector-count', '1');
  await expect(cloud).toHaveAttribute('data-point-label-count', '2');
  await expect(page.getByTestId('box-transform-gizmo')).toHaveCount(0);
  const front = page.getByTestId('camera-front');
  await expect(front).toHaveAttribute('data-projected-box-count', '1');
  await expect(front).toHaveAttribute('data-projected-vector-count', '1');
  await expect(front).toHaveAttribute('data-projected-polygon-count', '1');
  const signature = await front.getAttribute('data-projection-signature');
  await page.getByLabel('回放速度', {exact: true}).selectOption('2');
  const startedAt = Date.now();
  await page.getByLabel('播放', {exact: true}).click();
  await expect(cloud).toHaveAttribute('data-active-frame-id', frameIds[2]);
  // Capture timestamps span 2.4 seconds, despite clip.fps=10. At 2x this must
  // take roughly 1.2 seconds, rather than advancing every arbitrary 100ms tick.
  expect(Date.now() - startedAt).toBeGreaterThanOrEqual(900);
  await expect(page.locator('.timestamp')).toContainText('2.400');
  await expect(cloud).toHaveAttribute('data-cloud-point-count', '6');
  await expect(front).toHaveAttribute('data-active-frame-id', frameIds[2]);
  for (const camera of ['front', 'front_left', 'left', 'rear_left', 'rear_right', 'right', 'front_right']) {
    await expect(page.getByTestId(`camera-${camera}`)).toHaveAttribute('data-rendered-image-frame-id', frameIds[2]);
  }
  await expect.poll(() => front.getAttribute('data-projection-signature')).not.toBe(signature);
  await expect(page.getByLabel('播放', {exact: true})).toBeVisible();
  const result = await page.evaluate(() => {
    const state = window as unknown as {reviewSamples: string[]; reviewMismatches: string[]};
    return {reviewSamples: state.reviewSamples, reviewMismatches: state.reviewMismatches};
  });
  expect(result.reviewSamples).toEqual(expect.arrayContaining(frameIds));
  expect(result.reviewMismatches).toEqual([]);
  const map = await (await request.get(`/api/clips/${id}/scene/map`)).json();
  await page.getByLabel('回放点云视图', {exact: true}).selectOption('scene');
  await expect(cloud).toHaveAttribute('data-cloud-point-count', String(map.point_count));
  await expect(cloud).toHaveAttribute('data-active-frame-id', frameIds[2]);
  await expect(front).toHaveAttribute('data-rendered-image-frame-id', frameIds[2]);
  await page.getByLabel('退出回放', {exact: true}).click();
  await expect(page.getByLabel('退出回放', {exact: true})).toHaveCount(0);
});

test('loop replay returns to the first frame and remains paused when requested', async ({page, request}) => {
  const id = await prepareClip(request, 'loop');
  await openClip(page, id);
  await observeFrames(page);
  await page.getByLabel('回放标注clip', {exact: true}).click();
  await page.getByLabel('暂停', {exact: true}).click();
  await page.getByLabel('回放速度', {exact: true}).selectOption('2');
  await page.getByLabel('循环回放', {exact: true}).check();
  await page.getByLabel('播放', {exact: true}).click();
  await expect.poll(() => page.evaluate(() => {
    const ids = (window as unknown as {reviewSamples: string[]}).reviewSamples;
    const last = ids.indexOf('review-2');
    return last >= 0 && ids.slice(last + 1).includes('review-0');
  })).toBe(true);
  await page.getByLabel('暂停', {exact: true}).click();
  await expect(page.getByLabel('播放', {exact: true})).toBeVisible();
  const frozen = await page.getByTestId('point-cloud-view').getAttribute('data-active-frame-id');
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-active-frame-id', frozen!);
});

test('starting annotation replay saves pending edits and a delayed frame cannot advance after pause', async ({page, request}) => {
  const id = await prepareClip(request, 'buffer');
  let release: () => void = () => {};
  const gate = new Promise<void>(resolve => {release = resolve;});
  let released: () => void = () => {};
  const delivered = new Promise<void>(resolve => {released = resolve;});
  let requested = false;
  await page.route(`**/api/clips/${id}/frames/${frameIds[1]}/points*`, async route => {
    const response = await route.fetch();
    requested = true;
    await gate;
    try {await route.fulfill({response});} catch { /* A correctly cancelled request may already be aborted. */ }
    finally {released();}
  });
  await openClip(page, id);
  await page.locator('.object-row').filter({hasText: 'review-car'}).click();
  await page.getByLabel('位置X', {exact: true}).fill('13.25');
  const saved = page.waitForResponse(response => response.request().method() === 'PUT' && response.url().endsWith('/scene/annotations'));
  await page.getByLabel('回放标注clip', {exact: true}).click();
  expect((await saved).ok()).toBe(true);
  const document = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
  expect(document.tracks[0].keyframes.find((key: {frame_id: string}) => key.frame_id === frameIds[0]).center[0]).toBeCloseTo(13.25, 8);
  await expect.poll(() => requested).toBe(true);
  await expect(page.getByLabel('暂停', {exact: true})).toBeEnabled();
  await page.getByLabel('暂停', {exact: true}).click();
  const frame = await page.getByTestId('point-cloud-view').getAttribute('data-active-frame-id');
  release();
  await delivered;
  await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-active-frame-id', frame!);
  await expect(page.getByLabel('播放', {exact: true})).toBeVisible();
});

test('manual review seeking stays read-only and clears pixel and point labels absent from the selected frame', async ({page, request}) => {
  const id = await prepareClip(request, 'seek');
  const labelPath = `/api/clips/${id}/frames/${frameIds[2]}/annotations`;
  const labels = await (await request.get(labelPath)).json();
  const cleared = await request.put(labelPath, {data: {...labels, image_polygons: [], point_labels: {}}});
  expect(cleared.ok()).toBe(true);
  const scenePath = `/api/clips/${id}/scene/annotations`;
  const scene = await (await request.get(scenePath)).json();
  await openClip(page, id);
  await page.getByLabel('回放标注clip', {exact: true}).click();
  await page.getByLabel('暂停', {exact: true}).click();
  await page.getByLabel('回放点云视图', {exact: true}).selectOption('frame');
  await expect(page.getByTestId('camera-front')).toHaveAttribute('data-projected-polygon-count', '1');
  await page.getByLabel('跳转到第3帧', {exact: true}).click();
  const cloud = page.getByTestId('point-cloud-view');
  await expect(cloud).toHaveAttribute('data-active-frame-id', frameIds[2]);
  await expect(cloud).toHaveAttribute('data-point-label-count', '0');
  await expect(page.getByTestId('camera-front')).toHaveAttribute('data-projected-polygon-count', '0');
  await expect(page.getByLabel('播放', {exact: true})).toBeVisible();
  await page.locator('.object-row').filter({hasText: 'review-car'}).click();
  await expect(page.getByTestId('box-transform-gizmo')).toHaveCount(0);
  const canvas = page.getByLabel('三维点云标注视图');
  await canvas.click({position: {x: 140, y: 140}});
  expect(await (await request.get(scenePath)).json()).toEqual(scene);
  expect((await (await request.get(labelPath)).json()).image_polygons).toEqual([]);
});
