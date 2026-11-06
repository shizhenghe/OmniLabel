import {expect,test} from '@playwright/test';
import {cameraFocus,focusProjection} from '../../src/geometry/cameraFocus';
import {IDENTITY} from '../../src/scene';
import type {Camera,Frame} from '../../src/types';

test('camera focus uses effective projection and registered world pose',()=>{
 const camera:Camera={id:'rear',name:'Rear',width:640,height:480,intrinsics:[[400,2,300],[0,410,220],[0,0,1]],camera_from_ego:IDENTITY};
 const frame:Frame={id:'f0',timestamp:0,ego_to_world:IDENTITY,lidar_to_ego:IDENTITY,points:[],point_count:0,images:{},projection_camera_from_lidar:{rear:[[1,0,0,2],[0,1,0,3],[0,0,1,4],[0,0,0,1]]}};
 const map={world_from_lidar:{f0:[[1,0,0,500000],[0,1,0,3500000],[0,0,1,10],[0,0,0,1]]}};
 const focus=cameraFocus(camera,frame,true,map,1);
 expect(focus.worldFromCamera.slice(0,3).map(row=>row[3])).toEqual([499998,3499997,6]);
 const matrix=focusProjection(focus,1000,600,.05,4000);
 const p=[1,-2,-10,1]; const q=matrix.map(row=>row.reduce((sum,v,i)=>sum+v*p[i],0));
 const x=(q[0]/q[3]+1)*500,y=(1-q[1]/q[3])*300;
 expect(x).toBeCloseTo(100+(400*.1+2*.2+300)*1.25,8);
 expect(y).toBeCloseTo((410*.2+220)*1.25,8);
});

test('clicking images repeatedly focuses that camera and keeps annotations unchanged',async({page,request})=>{
 await page.goto('/');
 await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled({timeout:60000});
 const before=await(await request.get('/api/clips/demo/scene/annotations')).json();
 const cloud=page.getByTestId('point-cloud-view');
 await page.getByTestId('camera-front').click();
 await expect(cloud).toHaveAttribute('data-focused-camera','front');
 const first=Number(await cloud.getAttribute('data-focus-request'));
 await page.getByTestId('camera-front').click();
 await expect.poll(async()=>Number(await cloud.getAttribute('data-focus-request'))).toBeGreaterThan(first);
 await page.getByTestId('camera-rear_right').click();
 await expect(cloud).toHaveAttribute('data-focused-camera','rear_right');
 await page.getByLabel('下一帧',{exact:true}).click();
 await expect(page.getByTestId('camera-front')).toHaveAttribute('data-active-frame-id','frame-001');
 await page.getByTestId('camera-front').click();
 await expect(cloud).toHaveAttribute('data-focused-camera','front');
 expect(await(await request.get('/api/clips/demo/scene/annotations')).json()).toEqual(before);
});
