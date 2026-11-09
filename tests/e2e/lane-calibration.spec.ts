import {expect, test} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {projectLidarPoint} from '../../src/geometry/cameraProjection';
import * as THREE from 'three';
import type {Camera, Frame, Vec3} from '../../src/types';

test('two point-cloud lane selections drive a frozen image reference and real optimization without redrawing pixels', async ({page, request}) => {
  test.setTimeout(120_000);
  const id = 'lane-calibration-ui';
  const demo = await (await request.get('/api/clips/demo')).json();
  const camera: Camera = demo.cameras.find((c: Camera) => c.id === 'front');
  const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const lane: Vec3[] = [[10, -2, -1.6], [22, -2.5, -1.5], [40, -1, -1.3]];
  const imageLane: Vec3[] = lane.map(p => [p[0], p[1]-1.2, p[2]+.3]);
  const frames = [0, 1].map(index => ({id: `f${index}`, timestamp: index * .1, ego_to_world: identity,
    lidar_to_ego: identity, points: Array.from({length: 400}, (_, n) => [8 + n % 20, -5 + Math.floor(n/20)*.5, -1.6, .5]),
    images: {front: demo.frames[index].images.front}}));
  const imported = await request.post('/api/clips/import', {multipart: {file: {name: `${id}.json`, mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({clip: {id, name: id, fps: 10, cameras: [camera], frames},
      scene_annotations: {revision: 0, coordinate_frame: 'world', tracks: [],
        vectors: [{id: 'world-lane', label: 'lane', points: lane, closed: false},
          {id: 'image-projected-lane', label: 'image-lane', points: imageLane, closed: false},
          {id: 'image-projected-lane-2', label: 'image-lane', points: imageLane.map(p => [p[0], p[1]-.8, p[2]]), closed: false}]}}))}}});
  expect(imported.ok(), await imported.text()).toBe(true);
  const calibrationBefore = await (await request.get(`/api/clips/${id}/calibration`)).json();
  const sceneBefore = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.locator('.task-button').filter({hasText: '车道线标注'}).click();
  await page.getByLabel('车道线外参优化', {exact: true}).click();
  const panel = page.getByTestId('lane-calibration-panel');
  await expect(panel).toBeVisible();
  await expect(panel.getByLabel('车道线优化模式')).toHaveValue('frame');
  await expect(panel.getByLabel('图像上手动选线', {exact: true})).toHaveAttribute('aria-pressed', 'true');
  await panel.getByLabel('点云中选择投影车道线', {exact: true}).click();
  await expect(panel.getByLabel('使用选中三维车道线')).toBeDisabled();
  await expect(panel.getByLabel('绘制优化图像车道线')).toHaveCount(0);
  await expect(panel.getByLabel('优化图像投影车道线')).toBeDisabled();
  const cloud = page.getByTestId('point-cloud-view');
  await expect(cloud).toHaveAttribute('data-annotation-vector-count', '3');
  await expect(cloud).toHaveAttribute('data-vector-editable', 'false');
  await expect(cloud.locator('..')).not.toHaveClass(/scene-disabled/);
  const clickWorld = async (point: Vec3) => {
    const bounds = (await cloud.locator('canvas[aria-label="三维点云标注视图"]').boundingBox())!;
    const halfHeight = 28.8 * Math.tan(THREE.MathUtils.degToRad(47 / 2));
    const view = new THREE.OrthographicCamera(-halfHeight*bounds.width/bounds.height, halfHeight*bounds.width/bounds.height, halfHeight, -halfHeight, .05, 4000);
    const center = new THREE.Vector3(17.5, -.25, -1.6);
    view.up.set(1, 0, 0); view.position.copy(center).add(new THREE.Vector3(0, 0, 28.8));
    view.lookAt(center); view.updateMatrixWorld();
    const pixel = new THREE.Vector3(...point).project(view);
    await page.mouse.click(bounds.x + (pixel.x+1)*bounds.width/2, bounds.y + (1-pixel.y)*bounds.height/2);
  };
  await clickWorld([22, -2.5, -1.5]);
  await expect(panel.getByLabel('使用选中三维车道线')).toContainText('world-lane');
  await panel.getByLabel('使用选中三维车道线').click();
  await expect(panel.getByLabel('优化三维车道线')).toHaveValue('world-lane');
  await expect(panel.getByLabel('使用选中图像投影车道线')).toBeDisabled();
  await clickWorld([22, -3.7, -1.2]);
  await expect(panel.getByLabel('使用选中图像投影车道线')).toContainText('image-projected-lane');
  await panel.getByLabel('使用选中图像投影车道线').click();
  await expect(panel.getByLabel('优化图像投影车道线')).toHaveValue('image-projected-lane');
  await expect(panel).toContainText('待添加：world-lane ↔ image-projected-lane');
  const image = page.getByTestId('camera-front');
  await expect(image).toHaveAttribute('data-image-state', 'ready');
  await panel.getByLabel('添加车道线配对').click();
  await expect(panel).toContainText('#1 f0 / world-lane ↔ image-projected-lane');
  const endpoint = `/api/clips/${id}/lane-calibration/front`;
  const state = await (await request.get(endpoint)).json();
  expect(state.pairing.pairs).toHaveLength(1);
  expect(state.pairing.pairs[0].image_points.length).toBeGreaterThan(20);
  expect(state.pairing.pairs[0].image_lane_id).toBe('image-projected-lane');
  expect(state.pairing.pairs[0].image_reference.frame_id).toBe('f0');
  expect(state.pairing.pairs[0].image_reference.camera.id).toBe('front');
  expect(state.pairing.pairs[0].image_size).toEqual([camera.width, camera.height]);
  const expected = projectLidarPoint(imageLane[0], camera, {...frames[0], point_count: 400} as Frame)!;
  expect(state.pairing.pairs[0].image_points[0][0]).toBeCloseTo(expected[0], 5);
  expect(state.pairing.pairs[0].image_points[0][1]).toBeCloseTo(expected[1], 5);
  await panel.getByLabel('确认车道线坐标约定').check();
  await panel.getByLabel('确认图像Pose同步').check();
  await panel.getByLabel('预览车道线投影误差').click();
  await expect(image).toHaveAttribute('data-lane-calibration-overlay-count', '2');
  await expect(panel).toContainText('像素误差（前 → 后）');
  await panel.getByLabel('车道线优化模式').selectOption('single');
  await panel.getByLabel('车道线优化迭代次数').fill('8');
  await panel.getByText('外参、时间戳与约束高级配置', {exact: true}).click();
  await panel.getByLabel('车道线优化高级配置').fill('{"config":{"lbfgs_iterations":0,"samples_per_lane":24}}');
  const start = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith(endpoint+'/jobs'));
  await panel.getByLabel('运行车道线外参优化').click();
  const response = await start; expect(response.status()).toBe(202);
  const jobId = (await response.json()).id;
  await expect(panel.getByText('任务：completed', {exact: true})).toBeVisible({timeout: 40_000});
  await expect(panel.getByLabel('应用车道线优化外参')).toBeDisabled();
  await expect(panel).toContainText('单帧结果仅供调试');
  await expect(image).toHaveAttribute('data-lane-calibration-overlay-count', '3');
  const downloading = page.waitForEvent('download');
  await panel.getByLabel('下载车道线优化结果').click();
  const download = await downloading;
  expect(download.suggestedFilename()).toBe(`${id}-front-lane-calibration.zip`);
  const bytes = await readFile((await download.path())!); expect(bytes.subarray(0,2).toString()).toBe('PK');
  expect((await (await request.get(endpoint+'/jobs/'+jobId)).json()).status).toBe('completed');
  expect((await (await request.get(`/api/clips/${id}/calibration`)).json()).revision).toBe(calibrationBefore.revision);
  expect(await (await request.get(`/api/clips/${id}/scene/annotations`)).json()).toEqual(sceneBefore);
  // Retain the real line so another image segment can be paired to it.
  await expect(panel.getByLabel('优化三维车道线')).toHaveValue('world-lane');
  await expect(panel.getByLabel('优化图像投影车道线').locator('option[value="image-projected-lane"]')).toHaveJSProperty('disabled', true);
  await panel.getByLabel('优化图像投影车道线').selectOption('image-projected-lane-2');
  await panel.getByLabel('添加车道线配对').click();
  await expect(panel).toContainText('#2 f0 / world-lane ↔ image-projected-lane-2');
  expect((await (await request.get(endpoint)).json()).pairing.pairs).toHaveLength(2);
  page.once('dialog', dialog => void dialog.dismiss());
  await panel.getByLabel('丢弃当前优化并重新选择').click();
  await expect(panel).toContainText('已保存配对（2）');
  page.once('dialog', dialog => void dialog.accept());
  await panel.getByLabel('丢弃当前优化并重新选择').click();
  await expect(panel).toContainText('已保存配对（0）');
  await expect(panel.getByLabel('优化三维车道线')).toHaveValue('');
  await expect(panel.getByLabel('优化图像投影车道线')).toHaveValue('');
  await expect(panel.getByLabel('车道线优化任务')).toHaveCount(0);
  await expect(panel.getByText('像素误差（前 → 后）', {exact: true})).toHaveCount(0);
  await expect(image).toHaveAttribute('data-lane-calibration-overlay-count', '0');
  const cleared = await (await request.get(endpoint)).json();
  expect(cleared.pairing.pairs).toEqual([]);
  expect(cleared.latest_job).toBeNull();
  expect((await request.post(endpoint+'/jobs/'+jobId+'/apply', {data: {confirm_shared_update: true}})).status()).toBe(409);
  expect(await (await request.get(`/api/clips/${id}/calibration`)).json()).toEqual(calibrationBefore);
  expect(await (await request.get(`/api/clips/${id}/scene/annotations`)).json()).toEqual(sceneBefore);
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(panel).toContainText('当前帧 f1');
  await expect(panel.getByLabel('优化三维车道线')).toHaveValue('');
  await expect(panel.getByLabel('优化图像投影车道线')).toHaveValue('');
  await expect(image).toHaveAttribute('data-lane-calibration-overlay-count', '0');
  await page.reload();
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('车道线外参优化', {exact: true}).click();
  await expect(panel).toContainText('已保存配对（0）');
  await expect(panel.getByLabel('车道线优化任务')).toHaveCount(0);
  await panel.getByLabel('优化三维车道线').selectOption('world-lane');
  await panel.getByLabel('点云中选择投影车道线', {exact: true}).click();
  await panel.getByLabel('优化图像投影车道线').selectOption('image-projected-lane');
  await panel.getByLabel('添加车道线配对').click();
  await expect(panel).toContainText('#1 f0 / world-lane ↔ image-projected-lane');
  // Keep both selection paths in one saved pairing document.
  await panel.getByLabel('图像上手动选线', {exact: true}).click();
  await panel.getByLabel('绘制优化图像车道线').click();
  await expect(image).toHaveAttribute('data-calibration-picking', 'true');
  // Keep the picking/zoom assertions while using a physically corresponding
  // image line, so it also passes the new association direction gate.
  const pixels: [number, number][] = lane.map(point => projectLidarPoint(point, camera,
    {...frames[0], point_count: 400} as Frame)!.slice(0, 2) as [number, number]);
  const clickPixel = async ([u, v]: [number, number]) => {
    const target = page.getByTestId('camera-front').last();
    await expect(target).toHaveAttribute('data-image-state', 'ready');
    const box = (await target.locator('canvas').boundingBox())!;
    const size = await target.evaluate(element => ({width: element.clientWidth, height: element.clientHeight}));
    const zoom = Number(await target.getAttribute('data-image-zoom'));
    const [dx, dy] = (await target.getAttribute('data-image-offset'))!.split(',').map(Number);
    const scale = Math.min(size.width/camera.width, size.height/camera.height)*zoom;
    await target.evaluate(element => element.addEventListener('click', event => {
      const mouse = event as MouseEvent;
      element.setAttribute('data-test-last-click', JSON.stringify([mouse.clientX, mouse.clientY]));
    }, {once: true}));
    await page.mouse.click(box.x+((size.width-camera.width*scale)/2+dx+u*scale)*box.width/size.width,
      box.y+((size.height-camera.height*scale)/2+dy+v*scale)*box.height/size.height);
    // Chromium quantizes MouseEvent client coordinates to CSS pixels. Check
    // the delivered click, rather than CDP's requested fractional coordinate.
    const [cx, cy] = JSON.parse((await target.getAttribute('data-test-last-click'))!) as number[];
    return [((cx-box.x)*size.width/box.width-(size.width-camera.width*scale)/2-dx)/scale,
      ((cy-box.y)*size.height/box.height-(size.height-camera.height*scale)/2-dy)/scale];
  };
  const expectedPixels = [];
  for (const pixel of pixels) expectedPixels.push(await clickPixel(pixel));
  await expect(panel).toContainText('手绘图像线：3 点');
  await panel.getByLabel('撤回优化图像线一点').click();
  await expect(panel).toContainText('手绘图像线：2 点');
  expectedPixels[2] = await clickPixel(pixels[2]);
  await panel.getByLabel('添加车道线配对').click();
  await expect(panel).toContainText('#2 f0 / world-lane ↔ 图像手绘线');
  await expect(image).toHaveAttribute('data-calibration-picking', 'false');
  const mixed = await (await request.get(endpoint)).json();
  expect(mixed.pairing.pairs).toHaveLength(2);
  expect(mixed.pairing.pairs[0].image_lane_id).toBe('image-projected-lane');
  const manual = mixed.pairing.pairs[1];
  expect(manual.image_lane_id).toBeNull();
  expect(manual.image_reference).toBeNull();
  expect(manual.image_size).toEqual([camera.width, camera.height]);
  expectedPixels.forEach((point, i) => point.forEach((value, axis) => expect(manual.image_points[i][axis]).toBeCloseTo(value, 4)));
  expect(await (await request.get(`/api/clips/${id}/scene/annotations`)).json()).toEqual(sceneBefore);
  await panel.getByLabel('确认车道线坐标约定').check();
  await panel.getByLabel('确认图像Pose同步').check();
  await panel.getByLabel('预览车道线投影误差').click();
  await expect(panel).toContainText('像素误差（前 → 后）');
  await panel.getByLabel('车道线优化模式').selectOption('single');
  await panel.getByLabel('车道线优化迭代次数').fill('2');
  await panel.getByText('外参、时间戳与约束高级配置', {exact: true}).click();
  await panel.getByLabel('车道线优化高级配置').fill('{"config":{"lbfgs_iterations":0,"samples_per_lane":24}}');
  await panel.getByLabel('运行车道线外参优化').click();
  await expect(panel.getByText('任务：completed', {exact: true})).toBeVisible({timeout: 40_000});
  // Frame navigation clears only the unfinished image stroke.
  await panel.getByLabel('绘制优化图像车道线').click();
  await page.getByLabel(`放大 ${camera.name}`, {exact: true}).first().click();
  await clickPixel(pixels[0]);
  await page.getByLabel('收起放大图像', {exact: true}).click();
  await expect(panel).toContainText('手绘图像线：1 点');
  await page.getByLabel('下一帧', {exact: true}).click();
  await expect(panel).toContainText('当前帧 f1');
  await expect(panel).toContainText('手绘图像线：0 点');
  await expect(image).toHaveAttribute('data-calibration-picking', 'false');
  await page.reload();
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('车道线外参优化', {exact: true}).click();
  await expect(panel).toContainText('#1 f0 / world-lane ↔ image-projected-lane');
  await expect(panel).toContainText('#2 f0 / world-lane ↔ 图像手绘线');
  await panel.getByLabel('优化三维车道线').selectOption('world-lane');
  await panel.getByLabel('绘制优化图像车道线').click();
  await page.getByLabel(`放大 ${camera.name}`, {exact: true}).first().click();
  const expanded = page.getByTestId('camera-front').last();
  await expanded.getByLabel('放大图像', {exact: true}).click();
  await expanded.press('+');
  await expect(expanded).toHaveAttribute('data-image-zoom', '1.5625');
  await expect(panel).toContainText('手绘图像线：0 点');
  const area = (await expanded.locator('canvas').boundingBox())!;
  await page.mouse.move(area.x+area.width/2, area.y+area.height/2);
  await page.mouse.wheel(0, -120);
  await expect.poll(async () => Number(await expanded.getAttribute('data-image-zoom'))).toBeGreaterThan(1.5625);
  const oldOffset = await expanded.getAttribute('data-image-offset');
  await page.mouse.down({button: 'right'});
  await page.mouse.move(area.x+area.width/2+55, area.y+area.height/2-30, {steps: 8});
  await page.mouse.up({button: 'right'});
  await expect(expanded).not.toHaveAttribute('data-image-offset', oldOffset!);
  await expect(panel).toContainText('手绘图像线：0 点');
  const zoomPixels = [await clickPixel([300, 160]), await clickPixel([340, 180])];
  await expect(panel).toContainText('手绘图像线：2 点');
  await expanded.getByLabel('复位图像缩放').click();
  await expect(expanded).toHaveAttribute('data-image-zoom', '1');
  await expect(expanded).toHaveAttribute('data-image-offset', '0,0');
  await expect(panel).toContainText('手绘图像线：2 点');
  await page.getByLabel('收起放大图像', {exact: true}).click();
  await panel.getByLabel('添加车道线配对').click();
  await expect(panel).toContainText('#3 f0 / world-lane ↔ 图像手绘线');
  const zoomSaved = (await (await request.get(endpoint)).json()).pairing.pairs[2].image_points;
  zoomPixels.forEach((point, i) => point.forEach((value, axis) => expect(zoomSaved[i][axis]).toBeCloseTo(value, 4)));
});


test('frame-only lane calibration applies successive frames without overwriting the first', async ({page, request}) => {
  test.setTimeout(120_000);
  const id = 'lane-frame-scope-ui';
  const demo = await (await request.get('/api/clips/demo')).json();
  const camera = demo.cameras.find((c: Camera) => c.id === 'front');
  const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const imported = await request.post('/api/clips/import', {multipart: {file: {name: id+'.json', mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({clip: {id, name: id, fps: 10, cameras: [camera], frames: [0,1].map(i => ({
      id: 'f'+i, timestamp: i*.1, ego_to_world: identity, lidar_to_ego: identity,
      points: [[10,-1,-1.6,.5],[20,-1,-1.6,.5]], images: {front: demo.frames[i].images.front}}))},
      scene_annotations: {revision: 0, coordinate_frame: 'world', tracks: [], vectors: [
        {id: 'truth', label: 'lane', closed: false, points: [[10,-2,-1.6],[20,-2,-1.6],[30,-2,-1.6]]}]}}))}}});
  expect(imported.ok()).toBe(true);
  const endpoint = `/api/clips/${id}/lane-calibration/front`;
  const sceneBefore = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
  const calibrationBefore = await (await request.get(`/api/clips/${id}/calibration`)).json();
  const initialFrame1 = await (await request.get(`/api/clips/${id}/frames/f1/projection-calibration/front`)).json();
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('重建整个场景', {exact: true})).toBeEnabled();
  await page.getByLabel('车道线外参优化', {exact: true}).click();
  const panel = page.getByTestId('lane-calibration-panel');
  await expect(panel.getByLabel('车道线优化模式')).toHaveValue('frame');
  await expect(panel.getByLabel('绘制优化图像车道线')).toBeVisible();
  let firstSaved: unknown;
  for (const fid of ['f0','f1']) {
    await expect(page.getByTestId('camera-front')).toHaveAttribute('data-active-frame-id', fid);
    await panel.getByLabel('优化三维车道线').selectOption('truth');
    await panel.getByLabel('绘制优化图像车道线').click();
    const image = page.getByTestId('camera-front');
    await expect(image).toHaveAttribute('data-calibration-picking', 'true');
    const box = (await image.locator('canvas').boundingBox())!;
    const size = await image.evaluate(element => ({width: element.clientWidth, height: element.clientHeight}));
    const scale = Math.min(size.width/camera.width, size.height/camera.height);
    // Use corresponding image evidence; arbitrary screen clicks are correctly
    // rejected by the new distance/direction association gates.
    for (const point of [[10,-2,-1.6],[30,-2,-1.6]]) {
      const uv = projectLidarPoint(point, camera, {id: fid, timestamp: 0, ego_to_world: identity,
        lidar_to_ego: identity, points: [], point_count: 0, images: {}})!;
      await page.mouse.click(box.x+(size.width-camera.width*scale)/2+uv[0]*scale,
        box.y+(size.height-camera.height*scale)/2+uv[1]*scale);
    }
    await panel.getByLabel('添加车道线配对').click();
    await expect(panel).toContainText(`${fid} / truth ↔ 图像手绘线`);
    await panel.getByLabel('确认车道线坐标约定').check();
    await panel.getByLabel('确认图像Pose同步').check();
    await panel.getByLabel('车道线优化迭代次数').fill('2');
    const started = page.waitForResponse(r => r.request().method() === 'POST' && r.url().endsWith(endpoint+'/jobs'));
    await panel.getByRole('button', {name: '运行车道线外参优化', exact: true}).click();
    const response = await started; expect(response.status()).toBe(202);
    const job = await response.json();
    expect(job.request).toMatchObject({mode: 'frame', frame_ids: [fid]});
    await expect(panel.getByText('任务：completed', {exact: true})).toBeVisible({timeout: 40_000});
    await expect(panel.getByLabel('应用车道线优化外参')).toContainText(`应用到当前帧 ${fid}`);
    if (fid === 'f0') {
      await page.getByLabel('下一帧', {exact: true}).click();
      await expect(panel.getByLabel('应用车道线优化外参')).toBeDisabled();
      await page.getByLabel('上一帧', {exact: true}).click();
    }
    page.once('dialog', dialog => {expect(dialog.message()).toContain(`帧 ${fid}`); return dialog.accept();});
    await panel.getByLabel('应用车道线优化外参').click();
    await expect(panel.getByLabel('应用车道线优化外参')).toHaveText('已应用');
    const calibrationAfter = await (await request.get(`/api/clips/${id}/calibration`)).json();
    const {frame_projection_scopes: beforeScopes, ...beforeShared} = calibrationBefore;
    const {frame_projection_scopes: afterScopes, ...afterShared} = calibrationAfter;
    expect(afterShared).toEqual(beforeShared);
    expect(afterScopes[fid].front).toBe('frame');
    expect(await (await request.get(`/api/clips/${id}/scene/annotations`)).json()).toEqual(sceneBefore);
    const frame0 = await (await request.get(`/api/clips/${id}/frames/f0/projection-calibration/front`)).json();
    if (fid === 'f0') {
      firstSaved = frame0;
      expect(await (await request.get(`/api/clips/${id}/frames/f1/projection-calibration/front`)).json()).toEqual(initialFrame1);
      await page.getByLabel('下一帧', {exact: true}).click();
    } else expect(frame0).toEqual(firstSaved);
  }
});


test('old crop mutation stays suspended while matching preview and frame range restore coexist', async ({page, request}) => {
  const id = 'lane-overlap-suspended';
  const demo = await (await request.get('/api/clips/demo')).json();
  const camera = demo.cameras.find((c: Camera) => c.id === 'front');
  const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const imported = await request.post('/api/clips/import', {multipart:{file:{name:id+'.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({
    clip:{id,name:id,fps:10,cameras:[camera],frames:[0,1].map(i=>({id:'f'+i,timestamp:i*.1,ego_to_world:identity,lidar_to_ego:identity,
      points:Array.from({length:400},(_,j)=>[8+j%20,-5+Math.floor(j/20)*.5,-1.6,.5]),images:{front:demo.frames[i].images.front}}))},
    scene_annotations:{revision:0,coordinate_frame:'world',tracks:[],vectors:[{id:'truth',label:'lane',closed:false,points:[[10,-2,-1.6],[30,-2,-1.6]]}]}
  }))}}});
  expect(imported.ok()).toBe(true);
  const endpoint = `/api/clips/${id}/lane-calibration/front`;
  const pairs = [0,1].map(i=>({id:'pair'+i,frame_id:'f'+i,lane_id:'truth',image_points:[[300,250],[320,220]],image_size:[camera.width,camera.height],world_range:[.1,.4],split:'train',weight:1}));
  expect((await request.put(endpoint+'/pairs',{data:{revision:0,pairs}})).ok()).toBe(true);
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled();
  const scene = await (await request.get(`/api/clips/${id}/scene/annotations`)).json();
  const calibration = await (await request.get(`/api/clips/${id}/calibration`)).json();
  await page.getByLabel('车道线外参优化',{exact:true}).click();
  const panel = page.getByTestId('lane-calibration-panel');
  await expect(panel).toContainText('重合匹配在每次求解中选择实际可见对应段，不改写配对范围');
  await expect(panel.getByLabel('预览重合匹配', {exact: true})).toBeVisible();
  await expect(panel.getByLabel('自动匹配重合线段',{exact:true})).toHaveCount(0);
  page.once('dialog', dialog=>dialog.dismiss());
  await panel.getByLabel('恢复当前帧完整取样',{exact:true}).click();
  expect((await (await request.get(endpoint)).json()).pairing.pairs[0].world_range).toEqual([.1,.4]);
  page.once('dialog', dialog=>dialog.accept());
  await panel.getByLabel('恢复当前帧完整取样',{exact:true}).click();
  await expect(panel.getByLabel('恢复当前帧完整取样',{exact:true})).toBeDisabled();
  const saved = (await (await request.get(endpoint)).json()).pairing;
  expect(saved.pairs[0].world_range).toEqual([0,1]);
  expect(saved.pairs[1].world_range).toEqual([.1,.4]);
  expect(saved.pairs[0].image_points).toEqual(pairs[0].image_points);
  expect(await (await request.get(`/api/clips/${id}/scene/annotations`)).json()).toEqual(scene);
  expect(await (await request.get(`/api/clips/${id}/calibration`)).json()).toEqual(calibration);
});
