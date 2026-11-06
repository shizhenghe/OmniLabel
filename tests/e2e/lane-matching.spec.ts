import {expect, test, type APIRequestContext, type Page} from '@playwright/test';

type Matrix = number[][];
const identity: Matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const truth: Matrix = [[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]];
const origin = [500000, 3500000, 10];
const multiply = (a: Matrix, b: Matrix) => a.map(row => b[0].map((_, j) => row.reduce((s, v, k) => s+v*b[k][j], 0)));
const pixel = (p: number[], matrix: Matrix, k: Matrix) => {
  const q = matrix.slice(0,3).map(row => row[0]*p[0]+row[1]*p[1]+row[2]*p[2]+row[3]);
  return [(k[0][0]*q[0]+k[0][1]*q[1])/q[2]+k[0][2], k[1][1]*q[1]/q[2]+k[1][2]];
};

async function fixture(request: APIRequestContext, id: string) {
  const demo = await (await request.get('/api/clips/demo')).json();
  const camera = {...demo.cameras.find((c: {id: string}) => c.id === 'front'), distortion: []};
  const angle = .6*Math.PI/180, c=Math.cos(angle), s=Math.sin(angle);
  const delta = [[c,-s,0,.04],[s,c,0,-.03],[0,0,1,.05],[0,0,0,1]];
  const initial = multiply(delta, truth); camera.camera_from_ego = initial;
  const lane = (side: number) => Array.from({length:311}, (_, i) => {
    const x=8+i*.2; return [x, side+.002*(x-25)**2, -1.6+.01*x+.08*Math.sin(x/13)];
  });
  const lines = [lane(-3), lane(2.7), Array.from({length:81}, (_,i) => [24,-5+i*11/80,-1.6]),
    Array.from({length:81}, (_,i) => [32,4,-1.6+i*4/80])];
  const vectors = lines.map((points, i) => ({id:`lane${i}`, label:'lane', closed:false,
    points:points.map(p=>p.map((v,j)=>v+origin[j]))}));
  const points = lines.flat().map((p,i)=>[...p,.2+(i%80)/100]);
  const frames = [0,1].map(i=>({id:`f${i}`,timestamp:i*.1,
    ego_to_world: [[1,0,0,origin[0]+i*2],[0,1,0,origin[1]],[0,0,1,origin[2]],[0,0,0,1]],
    lidar_to_ego:identity, points:points.map(p=>[p[0]-i*2,...p.slice(1)]),
    images:{front:demo.frames[i].images.front}}));
  const response=await request.post('/api/clips/import',{multipart:{file:{name:`${id}.json`,mimeType:'application/json',
    buffer:Buffer.from(JSON.stringify({clip:{id,name:id,fps:10,cameras:[camera],frames},
      scene_annotations:{revision:0,coordinate_frame:'world',tracks:[],vectors}}))}}});
  expect(response.ok(),await response.text()).toBe(true);
  const endpoint=`/api/clips/${id}/lane-calibration/front`;
  const pairs=lines.map((points,i)=>({id:`pair${i}`,frame_id:'f0',lane_id:`lane${i}`,world_range:[0,1],
    image_points:(i<2?points.slice(50,171):points).map(p=>pixel(p,truth,camera.intrinsics)),
    image_size:[camera.width,camera.height],split:'train',weight:1}));
  const saved=await request.put(endpoint+'/pairs',{data:{revision:0,pairs}});
  expect(saved.ok(),await saved.text()).toBe(true);
  return {id,endpoint,camera,frames,points,pairing:(await saved.json()),scene:await (await request.get(`/api/clips/${id}/scene/annotations`)).json()};
}

async function open(page: Page, id: string) {
  await page.goto(`/?clip=${id}`);
  await expect(page.getByLabel('选择clip')).toHaveValue(id);
  await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled({timeout:60000});
  await page.getByLabel('车道线外参优化',{exact:true}).click();
  const panel=page.getByTestId('lane-calibration-panel');
  await expect(panel.getByLabel('车道线优化模式')).toHaveValue('frame');
  await expect(panel.getByLabel('外参匹配方式')).toHaveValue('overlap');
  await panel.getByLabel('确认车道线坐标约定').check();
  await panel.getByLabel('确认图像Pose同步').check();
  await panel.getByText('外参、时间戳与约束高级配置',{exact:true}).click();
  await panel.getByLabel('车道线优化高级配置').fill('{"config":{"min_match_length_px":10,"matching_samples":256,"outer_iterations":5}}');
  return panel;
}

async function run(page: Page, endpoint: string) {
  const response=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith(endpoint+'/jobs'));
  await page.getByLabel('运行车道线外参优化',{exact:true}).click();
  const accepted=await response;expect(accepted.status(),await accepted.text()).toBe(202);
  const id=(await accepted.json()).id;
  await expect(page.getByText('任务：completed',{exact:true})).toBeVisible({timeout:40000});
  return id;
}

function rawPixel(signature: string) { return signature.split(':')[2].split(',').map(Number); }

test('partial matching previews real cloud and annotations without publishing and close restores saved projection',async({page,request})=>{
  const f=await fixture(request,'matching-preview');
  const panel=await open(page,f.id);
  const camera=page.getByTestId('camera-front');
  await expect.poll(()=>camera.getAttribute('data-raw-projection-signature')).not.toBe('');
  const rawBefore=(await camera.getAttribute('data-raw-projection-signature'))!;
  const annotationBefore=await camera.getAttribute('data-projection-signature');
  await panel.getByLabel('预览重合匹配',{exact:true}).click();
  await expect(panel.getByLabel('重合段匹配报告')).toContainText('4 组');
  expect((await (await request.get(f.endpoint)).json()).pairing).toEqual(f.pairing);
  const jobId=await run(page,f.endpoint);
  const job=await (await request.get(f.endpoint+'/jobs/'+jobId)).json();
  expect(job.result.after.splits.train.mean_px).toBeLessThan(.05);
  await expect(panel.getByLabel('预览优化后点云')).toHaveAttribute('aria-pressed','true');
  await expect(page.getByTestId('lane-projection-preview')).toContainText('f0');
  await expect.poll(()=>camera.getAttribute('data-raw-projection-signature')).not.toBe(rawBefore);
  await expect.poll(()=>camera.getAttribute('data-projection-signature')).not.toBe(annotationBefore);
  const actual=rawPixel((await camera.getAttribute('data-raw-projection-signature'))!);
  const expected=pixel(f.points[0],truth,f.camera.intrinsics);
  expect(actual[0]).toBeCloseTo(expected[0],1);expect(actual[1]).toBeCloseTo(expected[1],1);
  const source=(await (await request.get(`/api/clips/${f.id}`)).json()).frames[0];
  expect(source.projection_camera_from_lidar.front).toBeUndefined();
  await panel.getByLabel('预览优化前点云').click();
  await expect(camera).toHaveAttribute('data-raw-projection-signature',rawBefore);
  await panel.getByLabel('预览优化后点云').click();
  await expect.poll(()=>camera.getAttribute('data-raw-projection-signature')).not.toBe(rawBefore);
  await panel.getByLabel('恢复已保存投影').click();
  await expect(camera).toHaveAttribute('data-raw-projection-signature',rawBefore);
  await panel.getByLabel('关闭车道线优化').click();
  await expect(page.getByTestId('lane-projection-preview')).toHaveCount(0);
  await expect(camera).toHaveAttribute('data-raw-projection-signature',rawBefore);
  expect(await (await request.get(`/api/clips/${f.id}/scene/annotations`)).json()).toEqual(f.scene);
});

test('matching result saves only the current frame and survives reload while next frame stays unchanged',async({page,request})=>{
  const f=await fixture(request,'matching-apply');
  const before=(await (await request.get(`/api/clips/${f.id}`)).json()).frames;
  const panel=await open(page,f.id);await run(page,f.endpoint);
  page.once('dialog',d=>void d.accept());
  await panel.getByLabel('应用车道线优化外参').click();
  await expect(panel.getByLabel('应用车道线优化外参')).toContainText('已应用');
  await expect(page.getByTestId('lane-projection-preview')).toHaveCount(0);
  const after=(await (await request.get(`/api/clips/${f.id}`)).json()).frames;
  expect(after[0].projection_camera_from_lidar.front).toBeTruthy();
  expect(after[1]).toEqual(before[1]);
  expect(await (await request.get(`/api/clips/${f.id}/scene/annotations`)).json()).toEqual(f.scene);
  await panel.getByLabel('关闭车道线优化').click();
  const savedSignature=(await page.getByTestId('camera-front').getAttribute('data-raw-projection-signature'))!;
  await page.reload();
  await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled({timeout:60000});
  // Raw cloud projection is a display toggle; the saved annotation projection is always loaded.
  await page.getByLabel('切换点云投影',{exact:true}).click();
  await expect(page.getByTestId('camera-front')).toHaveAttribute('data-raw-projection-signature',savedSignature);
  await page.getByLabel('下一帧',{exact:true}).click();
  await expect(page.getByTestId('camera-front')).toHaveAttribute('data-active-frame-id','f1');
  const expected=pixel(f.frames[1].points[0],f.camera.camera_from_ego,f.camera.intrinsics);
  const actual=rawPixel((await page.getByTestId('camera-front').getAttribute('data-raw-projection-signature'))!);
  expect(actual[0]).toBeCloseTo(expected[0],3);expect(actual[1]).toBeCloseTo(expected[1],3);
});
