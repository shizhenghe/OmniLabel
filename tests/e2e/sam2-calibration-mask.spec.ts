import {expect,test} from '@playwright/test';

test('SAM2 regions are reviewed and saved directly as a PJLab mask without changing annotations',async({page,request})=>{
 const framePath='/api/clips/demo/frames/frame-000/annotations';
 const before=await(await request.get(framePath)).json();
 const proposals=[
  {id:'stripe',camera_id:'front',label:'lane',points:[[180,220],[460,220],[460,230],[180,230]]},
  {id:'road',camera_id:'front',label:'lane',points:[[40,280],[600,280],[600,350],[40,350]]},
  {id:'old',camera_id:'front',label:'lane',points:[[10,10],[40,10],[40,40]]},
 ];
 const submissions:any[]=[];let mask:any;
 await page.route('**/api/models',route=>route.fulfill({json:{models:[{id:'sam2',available:true,name:'SAM2 fixture'}]}}));
 await page.route('**/api/clips/demo/automatic-calibration**',async route=>{
  const path=new URL(route.request().url()).pathname;
  if(path.endsWith('/from-sam2')){
   mask=route.request().postDataJSON();
   return route.fulfill({json:{status:'saved',frame_id:'frame-000',camera_id:'front',selected_count:1}});
  }
  return route.fulfill({json:{engines:[{id:'pjlab',name:'PJLab 线特征标定',available:true,detail:'Fixture'}],latest_job:null,total_pairs:84,saved_pairs:0}});
 });
 await page.route('**/api/clips/demo/jobs',async route=>{
  const body=route.request().postDataJSON();if(body.kind!=='prelabel')return route.continue();
  submissions.push(body);
  return route.fulfill({status:202,json:{id:'sam2-line-job',status:'completed',result:{proposal:{...before,image_polygons:proposals},sam2_candidate_ids:['stripe','road']}}});
 });
 await page.goto('/');
 await expect(page.getByLabel('全clip自动标定',{exact:true})).toBeEnabled({timeout:60000});
 await page.getByLabel('全clip自动标定',{exact:true}).click();
 const panel=page.getByRole('dialog',{name:'全 clip 自动标定',exact:true});
 await panel.getByLabel('SAM2生成当前帧线特征候选',{exact:true}).click();
 await expect(panel.getByLabel('线特征候选 1',{exact:true})).toBeVisible();
 expect(submissions[0]).toMatchObject({frame_id:'frame-000',model_id:'sam2',scene_mode:false,options:{camera_id:'front',points:[],point_labels:[]}});
 await expect(panel.getByLabel('保存SAM2线特征掩码',{exact:true})).toBeDisabled();
 await expect(panel.getByLabel('线特征候选 3',{exact:true})).toHaveCount(0);
 await panel.getByLabel('线特征候选 1',{exact:true}).check();
 await expect(panel.getByTestId('sam2-line-mask-preview').locator('polygon')).toHaveCount(1);
 await expect(panel.getByTestId('sam2-line-mask-preview').locator('polygon')).toHaveAttribute('points','180,220 460,220 460,230 180,230');
 await panel.getByLabel('保存SAM2线特征掩码',{exact:true}).click();
 await expect(panel.getByText('已保存 frame-000 / front 的原图尺寸线特征掩码',{exact:false})).toBeVisible();
 expect(mask).toEqual({job_id:'sam2-line-job',polygon_ids:['stripe']});
 expect(await(await request.get(framePath)).json()).toEqual(before);
 // Narrow markings can be isolated with original-pixel positive/negative prompts.
 await panel.getByLabel('线特征SAM2生成方式').selectOption('points');
 const image=panel.getByTestId('sam2-mask-image');
 const viewBox=await image.getAttribute('viewBox');
 const [, ,width,height]=viewBox!.split(' ').map(Number);
 const bounds=(await image.boundingBox())!;
 await image.click({position:{x:bounds.width*.25,y:bounds.height*.5}});
 await panel.getByRole('button',{name:'负点',exact:true}).click();
 await image.click({position:{x:bounds.width*.75,y:bounds.height*.5}});
 await panel.getByLabel('SAM2生成当前帧线特征候选',{exact:true}).click();
 await expect.poll(()=>submissions.length).toBe(2);
 expect(submissions[1].options.point_labels).toEqual([1,0]);
 // Browser pointer coordinates round to CSS pixels before conversion to original pixels.
 expect(Math.abs(submissions[1].options.points[0][0]-width*.25)).toBeLessThan(1.5);
 expect(Math.abs(submissions[1].options.points[0][1]-height*.5)).toBeLessThan(1.5);
 expect(Math.abs(submissions[1].options.points[1][0]-width*.75)).toBeLessThan(1.5);
});
