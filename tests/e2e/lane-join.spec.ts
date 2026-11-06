import {expect,test} from '@playwright/test';
import {connectLanes,orderLaneParts} from '../../src/geometry/laneJoin';
import type {Vector} from '../../src/types';
const lanes:Vector[]=[
 {id:'join-a',label:'lane',closed:false,points:[[10,-2,0],[20,-2,.1]]},
 {id:'join-b',label:'lane',closed:false,points:[[30,-2,.2],[21,-2,.11]]},
 {id:'join-c',label:'lane',closed:false,points:[[31,-2,.21],[40,-2,.3]]},
 {id:'untouched',label:'lane',closed:false,points:[[10,4,0],[20,4,0]]},
];
test('endpoint ordering preserves measured vertices, elevations and unrelated lanes',()=>{
 const original=JSON.stringify(lanes);
 const parts=orderLaneParts(lanes,[{id:'join-c',reverse:false},{id:'join-a',reverse:false},{id:'join-b',reverse:false}]);
 const result=connectLanes(lanes,parts,3);
 expect(result.vectors).toHaveLength(2);
 const joined=result.vectors.find(v=>v.id!=='untouched')!;
 expect(joined.points).toHaveLength(6);
 expect(new Set(joined.points.map(p=>p.join(',')))).toEqual(new Set(lanes.slice(0,3).flatMap(v=>v.points.map(p=>p.join(',')))));
 expect(result.vectors.find(v=>v.id==='untouched')).toBe(lanes[3]);
 expect(result.gaps).toHaveLength(2);
 for(const gap of result.gaps)expect(gap).toBeCloseTo(Math.hypot(1,.01),10);
 expect(JSON.stringify(lanes)).toBe(original);
 expect(()=>connectLanes(lanes,[{id:'join-a',reverse:false},{id:'join-c',reverse:false}],3)).toThrow('超过连接最大间距');
 expect(()=>connectLanes([{...lanes[0],closed:true},lanes[1]],[{id:'join-a',reverse:false},{id:'join-b',reverse:true}],3)).toThrow('开放车道线');
});
test('selected fragments preview, connect, undo and persist as one scene lane',async({page,request})=>{
 const route='/api/clips/demo/scene/annotations';
 const before=await(await request.get(route)).json();
 const saved=await request.put(route,{data:{...before,vectors:[...before.vectors,...lanes]}});
 expect(saved.ok()).toBe(true);
 await page.goto('/');
 await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled({timeout:60000});
 await page.locator('.task-button').filter({hasText:'车道线标注'}).click();
 const front=page.getByTestId('camera-front');
 const projectionBefore=await front.getAttribute('data-projection-signature');
 await page.getByLabel('打开车道线连接').click();
 await expect(page.getByRole('region',{name:'连接选定车道线'})).toBeVisible();
 for(const id of ['join-c','join-a','join-b'])await page.getByLabel(`连接选择 ${id}`,{exact:true}).check();
 await page.getByLabel('自动排列连接段').click();
 await page.getByLabel('预览车道线连接').click();
 await expect.poll(()=>front.getAttribute('data-projection-signature')).not.toBe(projectionBefore);
 expect((await(await request.get(route)).json()).vectors).toHaveLength(before.vectors.length+4);
 await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count',String(before.vectors.length+2));
 await page.getByLabel('应用车道线连接').click();
 await page.getByLabel('撤销本次融合').click();
 await expect(page.getByTestId('point-cloud-view')).toHaveAttribute('data-annotation-vector-count',String(before.vectors.length+4));
 await page.getByLabel('打开车道线连接').click();
 for(const id of ['join-a','join-b','join-c'])await page.getByLabel(`连接选择 ${id}`,{exact:true}).check();
 await page.getByLabel('自动排列连接段').click();await page.getByLabel('预览车道线连接').click();await page.getByLabel('应用车道线连接').click();
 await page.getByLabel('保存标注',{exact:true}).click();
 await expect.poll(async()=>(await(await request.get(route)).json()).vectors.length).toBe(before.vectors.length+2);
 const after=await(await request.get(route)).json();
 expect(after.vectors.find((v:Vector)=>v.id==='untouched')).toEqual(lanes[3]);
 expect(after.vectors.find((v:Vector)=>v.id==='join-a').points).toEqual([[10,-2,0],[20,-2,.1],[21,-2,.11],[30,-2,.2],[31,-2,.21],[40,-2,.3]]);
 await page.reload();await expect(page.getByLabel('重建整个场景',{exact:true})).toBeEnabled({timeout:60000});
 expect((await(await request.get(route)).json()).vectors).toEqual(after.vectors);
});
