import {expect, test} from '@playwright/test';
import {readFile} from 'node:fs/promises';

test('batch calibration polls results, saves accepted pairs and exports saved parameters', async ({page}) => {
  let latest: Record<string, unknown> | null = null;
  let saved = false;
  let startBody: unknown;
  const completed = () => ({id: 'ui-batch', clip_id: 'demo', engine: 'direct_nid', status: 'completed',
    created_at: '', updated_at: '', total: 3, processed: 3, accepted: saved ? 0 : 1, rejected: 1, failed: 0, skipped: 1, saved: saved ? 1 : 0,
    items: [
      {frame_id: 'frame-000', camera_id: 'front', status: saved ? 'saved' : 'accepted', reason: ''},
      {frame_id: 'frame-001', camera_id: 'front', status: 'rejected', reason: '验证区域改善不足'},
      {frame_id: 'frame-002', camera_id: 'front', status: 'skipped', reason: '此帧缺少该相机图像'},
    ]});
  await page.route('**/api/clips/demo/automatic-calibration**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith('/export')) return route.fulfill({json: {clip_id: 'demo', schema_version: '1.0', frames: [{frame_id: 'frame-000', saved}]}});
    if (path.endsWith('/apply')) {saved = true; latest = completed(); return route.fulfill({json: latest});}
    if (path.endsWith('/jobs') && route.request().method() === 'POST') {
      startBody = route.request().postDataJSON();
      latest = {...completed(), status: 'running', processed: 0, accepted: 0, rejected: 0, skipped: 0,
        items: [{frame_id: 'frame-000', camera_id: 'front', status: 'running'}]};
      return route.fulfill({json: latest});
    }
    if (path.endsWith('/jobs/ui-batch')) {latest = completed(); return route.fulfill({json: latest});}
    return route.fulfill({json: {engines: [
      {id: 'koide', name: 'Koide', available: false, detail: '未安装 Koide 运行环境'},
      {id: 'direct_nid', name: 'CPU', available: true, detail: 'CPU NID 可用'},
    ], latest_job: latest, total_pairs: 3, saved_pairs: saved ? 1 : 0}});
  });
  await page.goto('/');
  await expect(page.getByLabel('全clip自动标定', {exact: true})).toBeEnabled({timeout: 60_000});
  await page.getByLabel('全clip自动标定', {exact: true}).click();
  const panel = page.getByRole('dialog', {name: '全 clip 自动标定', exact: true});
  await expect(panel).toBeVisible();
  await expect(panel.getByRole('radio', {name: /Koide/})).toBeDisabled();
  await expect(panel.getByRole('radio', {name: /内置 CPU NID/})).toBeChecked();
  await panel.getByLabel('自动标定处理范围', {exact: true}).selectOption('all');
  await panel.getByRole('button', {name: '启动自动标定（3）', exact: true}).click();
  await expect(panel.getByRole('heading', {name: '任务完成', exact: true})).toBeVisible();
  expect(startBody).toEqual({engine: 'direct_nid', overwrite_saved: false});
  await expect(panel.getByText('验证区域改善不足', {exact: true})).toBeVisible();
  await expect(panel.getByText('此帧缺少该相机图像', {exact: true})).toBeVisible();
  await panel.getByRole('button', {name: '保存通过结果（1）', exact: true}).click();
  await expect(panel.getByRole('button', {name: '保存通过结果（0）', exact: true})).toBeDisabled();
  await expect(panel.getByText('本任务已有 1 组参数保存到工作区。', {exact: false})).toBeVisible();
  const downloading = page.waitForEvent('download');
  await panel.getByRole('button', {name: '导出已保存参数 JSON', exact: true}).click();
  const file = await downloading;
  const payload = JSON.parse(await readFile((await file.path())!, 'utf8'));
  expect(payload.frames[0].saved).toBe(true);
  await panel.getByLabel('关闭自动标定', {exact: true}).click();
  await expect(panel).toHaveCount(0);
  await page.getByLabel('全clip自动标定', {exact: true}).click();
  await expect(page.getByText('已保存并用于投影。', {exact: true})).toBeVisible();
});

test('manual frame extrinsics save independently and survive reload', async ({page, request}) => {
  const clip = await (await request.get('/api/clips/demo')).json();
  const current = clip.frames[0].id, next = clip.frames[1].id;
  const path = (frame: string) => `/api/clips/demo/frames/${frame}/projection-calibration/front`;
  const otherBefore = await (await request.get(path(next))).json();
  await page.goto('/?clip=demo');
  await expect(page.getByLabel('逐帧投影校准', {exact: true})).toBeEnabled({timeout: 60_000});
  await page.getByLabel('逐帧投影校准', {exact: true}).click();
  const editor = page.getByTestId('projection-calibration-editor');
  await editor.getByLabel('投影校正相机').selectOption('front');
  await expect(editor.getByLabel('本帧平移X加')).toBeEnabled();
  await expect(editor.getByLabel('投影修正应用范围')).toHaveValue('frame');
  const before = await (await request.get(path(current))).json();
  const front = page.getByTestId('camera-front');
  const signature = await front.getAttribute('data-projection-signature');
  await editor.getByLabel('本帧平移X加').click();
  await editor.getByLabel('本帧旋转Z加').click();
  await expect(front).not.toHaveAttribute('data-projection-signature', signature!);
  await expect(editor.getByText('正在手动微调当前帧相机外参。请检查近远处投影，确认后保存；未进行自动质量验证。')).toBeVisible();
  await editor.getByLabel('保存本帧本相机投影修正').click();
  await expect(editor.getByText('已保存，仅应用于当前帧和当前相机。')).toBeVisible();
  const saved = await (await request.get(path(current))).json();
  expect(saved.method).toBe('manual');
  expect(saved.saved_matrix[0][3]).toBeCloseTo(before.effective_matrix[0][3] + .01, 8);
  expect(saved.saved_matrix).not.toEqual(before.effective_matrix);
  expect(saved.metrics.validation).toBe('manual_review');
  expect((await (await request.get(path(next))).json()).effective_matrix).toEqual(otherBefore.effective_matrix);
  await page.reload();
  expect((await (await request.get(path(current))).json()).saved_matrix).toEqual(saved.saved_matrix);
  // Restore this shared demo fixture for the following correspondence test.
  expect((await request.put(path(current), {data: {revision: saved.revision, matrix: null, method: 'reset'}})).ok()).toBe(true);
});

test('single-pair editor loads actual calibration and accepts measured 3D and image picks', async ({page}) => {
  const pageErrors: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/');
  await expect(page.getByLabel('逐帧投影校准', {exact: true})).toBeEnabled({timeout: 60_000});
  await page.getByLabel('逐帧投影校准', {exact: true}).click();
  const editor = page.getByTestId('projection-calibration-editor');
  await expect(editor).toBeVisible();
  await expect(editor.getByRole('button', {name: '＋ 添加对应点', exact: true})).toBeEnabled();
  await page.getByRole('button', {name: '俯视 BEV', exact: true}).click();
  await editor.getByRole('button', {name: '＋ 添加对应点', exact: true}).click();
  const cloud = page.getByLabel('三维点云标注视图', {exact: true});
  const viewport = page.getByTestId('point-cloud-view');
  await expect(viewport).toHaveAttribute('data-calibration-picking', 'true');
  const size = await cloud.boundingBox();
  expect(size).not.toBeNull();
  // A pick must hit a measured point; scanning the visible cloud also verifies
  // the editor did not accidentally disable the canvas' pointer events.
  for (const y of [.4, .5, .6, .3, .7]) {
    for (const x of [.5, .4, .6, .3, .7]) {
      if (await viewport.getAttribute('data-calibration-picking') !== 'true') break;
      await cloud.click({position: {x: size!.width * x, y: size!.height * y}});
    }
    if (await viewport.getAttribute('data-calibration-picking') !== 'true') break;
  }
  await expect(editor.locator('.projection-pick-hint')).toContainText('三维点已选中');
  const camera = page.getByTestId('camera-front');
  const cameraSize = await camera.boundingBox();
  expect(cameraSize).not.toBeNull();
  await camera.click({position: {x: cameraSize!.width * .5, y: cameraSize!.height * .5}});
  await expect(editor.locator('.projection-anchor')).toHaveCount(1);
  await expect(editor.getByLabel('根据对应点求解投影', {exact: true})).toBeDisabled();
  await expect(editor.getByLabel('根据对应点求解投影', {exact: true})).toContainText('还需 5 对');
  expect(pageErrors).toEqual([]);
});
