import {expect,test} from '@playwright/test';
import {projectFramePoints} from '../../src/geometry/liveProjection';
import {correctFrameCameraExtrinsic} from '../../src/geometry/calibration';

for(const width of [1600,1100])test(`expanded calibration image keeps side panels usable at ${width}px and worker projects the latest adjustment`,async({page,request})=>{
 await page.setViewportSize({width,height:1000});
 const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto('/');
 await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled({timeout:60000});
 await page.getByLabel('单帧点云',{exact:true}).click();
 await page.getByLabel('编辑标定外参',{exact:true}).click();
 const editor=page.getByTestId('calibration-editor');
 const input=editor.getByLabel('外参X',{exact:true});
 await expect(input).toBeEnabled();
 expect((await input.boundingBox())!.height).toBeGreaterThanOrEqual(40);
 expect((await editor.boundingBox())!.width).toBeGreaterThanOrEqual(330);
 await page.getByLabel('放大 前视主相机',{exact:true}).count().then(async count=>{
  if(count)await page.getByLabel('放大 前视主相机',{exact:true}).first().click();
  else await page.locator('.camera-card').first().locator('.camera-expand').click();
 });
 const expanded=page.getByTestId('expanded-camera-workspace');
 await expect(expanded).toBeVisible();
 const rail=await page.locator('.camera-rail').boundingBox(),right=await editor.boundingBox(),imageArea=await expanded.boundingBox();
 expect(imageArea!.x).toBeGreaterThanOrEqual(rail!.x+rail!.width-1);
 expect(imageArea!.x+imageArea!.width).toBeLessThanOrEqual(right!.x+1);
 expect(page.workers().some(worker=>worker.url().includes('projection.worker'))).toBeTruthy();
 const camera=expanded.getByTestId('camera-front');
 await expect.poll(async()=>Number(await camera.getAttribute('data-projected-point-count'))).toBeGreaterThan(0);
 const before=await camera.getAttribute('data-projection-signature');
 await camera.getByLabel('放大图像',{exact:true}).click();
 await expect(camera).toHaveAttribute('data-image-zoom','1.25');
 // A burst must settle on the final input, with native geometry and labels unchanged.
 for(const value of ['0.05','0.15','-0.1','0.2'])await input.fill(value);
 await expect(input).toHaveValue('0.2');
 await expect.poll(()=>camera.getAttribute('data-projection-signature')).not.toBe(before);
 const clip=await(await request.get('/api/clips/demo')).json();
 const baseline=await(await request.get('/api/clips/demo/frames/frame-000/projection-calibration/front')).json();
 const matrix=correctFrameCameraExtrinsic(baseline.effective_matrix,{translation:[.2,0,0],rotation:[0,0,0]});
 const cloud=await(await request.get('/api/clips/demo/frames/frame-000/points')).json();
 const frame={...clip.frames[0],points:cloud.points,projection_camera_from_lidar:{...clip.frames[0].projection_camera_from_lidar,front:matrix}};
 const expected=projectFramePoints(frame,[clip.cameras.find((c:{id:string})=>c.id==='front')]).front;
 await expect(camera).toHaveAttribute('data-projected-point-count',String(expected.length));
 await expect(camera).toHaveAttribute('data-raw-projection-signature',`${expected.length}:${expected[0].index}:${expected[0].u.toFixed(4)},${expected[0].v.toFixed(4)}`);
 await editor.getByLabel('外参X增加',{exact:true}).click();await expect(input).toHaveValue('0.205');
 await editor.getByLabel('外参X减小',{exact:true}).click();await expect(input).toHaveValue('0.2');
 const beforeSave=page.waitForResponse(r=>r.request().method()==='PUT'&&r.url().endsWith('/frames/frame-000/projection-calibration/front'));
 await editor.getByLabel('保存外参',{exact:true}).click();expect((await beforeSave).ok()).toBeTruthy();
 await expect(expanded).toBeVisible();
 const saved=await(await request.get('/api/clips/demo/frames/frame-000/projection-calibration/front')).json();
 expect(saved.saved_matrix).toEqual(matrix);
 await editor.getByLabel('外参目标',{exact:true}).selectOption('rear_right');
 await expect(expanded.getByTestId('camera-rear_right')).toBeVisible();
 await expect(editor.getByLabel('外参目标',{exact:true})).toHaveValue('rear_right');
 await page.getByLabel('收起放大图像',{exact:true}).click();
 await expect(page.getByTestId('point-cloud-view')).toBeVisible();
 expect(errors).toEqual([]);
});

test('3D calibration point picking returns from expanded image without blocking the editor',async({page})=>{
 await page.goto('/');
 await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled({timeout:60000});
 await page.getByLabel('逐帧投影校准',{exact:true}).click();
 const editor=page.getByTestId('projection-calibration-editor');
 await expect(editor.getByLabel('本帧平移X加')).toBeEnabled();
 await page.locator('.camera-card').first().locator('.camera-expand').click();
 await expect(page.getByTestId('expanded-camera-workspace')).toBeVisible();
 await editor.getByRole('button',{name:'＋ 添加对应点',exact:true}).click();
 await expect(page.getByTestId('expanded-camera-workspace')).toHaveCount(0);
 await expect(page.getByTestId('point-cloud-view')).toBeVisible();
 await expect(editor.getByText('在点云中点击三维位置',{exact:true})).toBeVisible();
});

test('background projection retains Float64 world precision and source indices',async({page})=>{
 await page.goto('/');
 await expect.poll(()=>page.workers().length).toBeGreaterThan(0);
 const url=page.workers().find(worker=>worker.url().includes('projection.worker'))!.url();
 const result=await page.evaluate(async url=>{
  const worker=new Worker(url,{type:'module'});
  try{return await new Promise<number[]>((resolve,reject)=>{
   worker.onmessage=event=>resolve(Array.from(event.data.output[0].buffer));
   worker.onerror=()=>reject(new Error('Projection worker failed'));
   const coordinates=new Float64Array([1000000.001,0,10,1000000.101,0,10,999999.999,0,-2]);
   const matrix=[[1,0,0,-1000000],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
   worker.postMessage({type:'source',coordinates,stride:3,version:1},[coordinates.buffer]);
   worker.postMessage({type:'project',id:1,version:1,cameras:[{key:'precision',matrix,camera:{id:'front',width:640,height:480,intrinsics:[[1000,0,320],[0,1000,240],[0,0,1]],distortion:[]}}]});
  });}finally{worker.terminate();}
 },url);
 expect(result).toHaveLength(8);
 expect(result[0]).toBe(0);expect(result[4]).toBe(3);
 expect(result[1]).toBeCloseTo(320.1,5);expect(result[5]).toBeCloseTo(330.1,5);
 expect(result[2]).toBe(240);expect(result[3]).toBe(10);
});
