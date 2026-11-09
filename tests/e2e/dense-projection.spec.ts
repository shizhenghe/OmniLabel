import {expect,test} from '@playwright/test';
import {projectFramePoints} from '../../src/geometry/liveProjection';

function npy(count:number):Buffer {
 let header=`{'descr': '<f8', 'fortran_order': False, 'shape': (${count}, 4), }`;
 while((10+header.length+1)%64)header+=' ';header+='\n';
 const bytes=Buffer.alloc(10+header.length+count*32);bytes.set([0x93,...Buffer.from('NUMPY'),1,0]);
 bytes.writeUInt16LE(header.length,8);bytes.write(header,10,'ascii');
 for(let i=0;i<count;i++){
  const o=10+header.length+i*32;
  bytes.writeDoubleLE((i%500)/250-1,o);bytes.writeDoubleLE(Math.floor(i/500)/150-1,o+8);
  bytes.writeDoubleLE(5,o+16);bytes.writeDoubleLE(.4,o+24);
 }
 return bytes;
}
test('supplied map projects every visible point beyond 60k without cropping the 3D map',async({page,request})=>{
 const id='dense-projection',count=120001;
 const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 const demo=await(await request.get('/api/clips/demo')).json();
 const clip={id,name:id,source:'4d-clip-import',fps:10,cameras:[{id:'front',name:'Front',width:640,height:360,intrinsics:[[100,0,320],[0,100,180],[0,0,1]],camera_from_ego:identity,distortion:[0,0,0,0,0]}],
  frames:[{id:'f0',timestamp:0,points:[],point_count:0,ego_to_world:identity,lidar_to_ego:identity,images:{front:demo.frames[0].images.front}}]};
 const imported=await request.post('/api/clips/import',{multipart:{file:{name:'dense.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({clip}))}}});
 expect(imported.ok(),await imported.text()).toBeTruthy();
 await page.route(`**/api/clips/${id}/scene/map?**`,route=>route.fulfill({json:{coordinate_frame:'world',frame_count:1,point_count:count,trajectory:[],bounds:{min:[-1,-1,5],max:[1,1,5]},point_labels:{},method:'supplied reconstructed local-world map',source_map:'supplied',voxel_size:.05,fusion_mode:'full',world_from_lidar:{f0:identity}}}));
 await page.route(`**/api/clips/${id}/scene/map.npy`,route=>route.fulfill({contentType:'application/octet-stream',body:npy(count)}));
 const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
 await page.goto(`/?clip=${id}`);
 await expect(page.getByLabel('选择clip')).toHaveValue(id);
 await page.getByLabel('切换点云投影',{exact:true}).click();
 const camera=page.getByTestId('camera-front');
 await expect(camera).toHaveAttribute('data-projection-mode','full-map-raster');
 await expect(camera).toHaveAttribute('data-projected-point-count',String(count));
 await expect(camera).toHaveAttribute('data-projection-source-count',String(count));
 await expect.poll(async()=>Number(await camera.getAttribute('data-projection-pixel-count'))).toBeGreaterThan(1000);
 await expect(page.getByLabel('完整地图投影统计')).toContainText('120,001');
 await page.locator('.camera-card').first().locator('.camera-expand').click();
 const expanded=page.getByTestId('expanded-camera-workspace').getByTestId('camera-front');
 await expect(expanded).toHaveAttribute('data-projected-point-count',String(count));
 await expanded.getByLabel('放大图像',{exact:true}).click();
 await page.getByLabel('收起放大图像').click();
 await page.getByLabel('切换点云投影',{exact:true}).click();
 await page.getByLabel('切换点云投影',{exact:true}).click();
 await expect(camera).toHaveAttribute('data-projected-point-count',String(count));
 expect(errors).toEqual([]);
});

test('full-map raster keeps nearest-point semantics, exact source counts and transparent background',async({page})=>{
 await page.goto('/');await expect.poll(()=>page.workers().length).toBeGreaterThan(0);
 const url=page.workers().find(w=>w.url().includes('projection.worker'))!.url();
 const result=await page.evaluate(async url=>{
  const worker=new Worker(url,{type:'module'});
  try{return await new Promise<any>((resolve,reject)=>{
   worker.onerror=()=>reject(new Error('worker failed'));
   worker.onmessage=e=>{
    const item=e.data.output[0],raster=item.raster;
    const canvas=new OffscreenCanvas(64,64),ctx=canvas.getContext('2d')!;ctx.drawImage(raster.bitmap,0,0);
    resolve({pointCount:raster.pointCount,pixelCount:raster.pixelCount,sourceCount:raster.sourceCount,first:Array.from(item.buffer),near:Array.from(ctx.getImageData(32,32,1,1).data),background:Array.from(ctx.getImageData(0,0,1,1).data)});
    raster.bitmap.close();
   };
   const coords=new Float64Array([0,0,2,0,0,0,8,0,-.02,0,2,0,0,0,-1,0]);
   worker.postMessage({type:'source',coordinates:coords,components:4,stride:1,version:1},[coords.buffer]);
   worker.postMessage({type:'project',id:1,version:1,cameras:[{key:'occlusion',raster:true,labels:{'0':'vehicle','1':'road'},camera:{id:'front',width:64,height:64,intrinsics:[[100,0,32],[0,100,32],[0,0,1]],distortion:[]},matrix:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]});
  });}finally{worker.terminate();}
 },url);
 expect(result).toMatchObject({pointCount:3,pixelCount:2,sourceCount:4,first:[0,32,32,2],near:[55,221,191,255],background:[0,0,0,0]});
});


test('full-map raster projection agrees with calibrated Brown distortion and Float64 coordinates',async({page})=>{
 const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 const matrix=[[1,0,0,-1000000],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 const camera={id:'front',name:'Front',width:640,height:480,intrinsics:[[600,1,320],[.5,610,240],[0,0,1]],camera_from_ego:matrix,distortion:[.01,-.001,.002,-.003,.0005]};
 const points=[[1000000.123,1,10,.4],[999999.9,-1,8,.5],[1000000,0,-1,.6]];
 const expected=projectFramePoints({id:'f0',timestamp:0,ego_to_world:identity,lidar_to_ego:identity,images:{},points,point_count:points.length},[camera]).front;
 await page.goto('/');await expect.poll(()=>page.workers().length).toBeGreaterThan(0);
 const url=page.workers().find(w=>w.url().includes('projection.worker'))!.url();
 const result=await page.evaluate(async ({url,camera,points,matrix})=>{
  const worker=new Worker(url,{type:'module'});
  try{return await new Promise<{first:number[];count:number}>((resolve,reject)=>{
   worker.onerror=()=>reject(new Error('worker failed'));
   worker.onmessage=e=>{const item=e.data.output[0];resolve({first:Array.from(item.buffer),count:item.raster.pointCount});item.raster.bitmap.close();};
   const coordinates=new Float64Array(points.flat());
   worker.postMessage({type:'source',coordinates,components:4,stride:1,version:1},[coordinates.buffer]);
   worker.postMessage({type:'project',id:1,version:1,cameras:[{camera,matrix,raster:true,key:'brown'}]});
  });}finally{worker.terminate();}
 },{url,camera,points,matrix});
 expect(result.count).toBe(expected.length);
 expect(result.first[0]).toBe(expected[0].index);
 expect(result.first[1]).toBeCloseTo(expected[0].u,9);
 expect(result.first[2]).toBeCloseTo(expected[0].v,9);
 expect(result.first[3]).toBe(expected[0].depth);
});
