import {expect,test} from '@playwright/test';

for(const accepted of [false,true]) test(`${accepted?'accepted':'rejected'} batch result loads its actual matrix for read-only before/after projection`,async({page,request})=>{
 const path='/api/clips/demo/frames/frame-000/projection-calibration/front';
 const baseline=await(await request.get(path)).json();
 const labels=await(await request.get('/api/clips/demo/frames/frame-000/annotations')).json();
 const candidate=structuredClone(baseline.effective_matrix);candidate[0][3]+=.9;
 const reason='修正超过 0.75 米或 3 度，请先手动校准初值并检查同步情况。';
 const job={id:'candidate-review',clip_id:'demo',engine:'pjlab',status:'completed',total:1,processed:1,accepted:accepted?1:0,rejected:accepted?0:1,failed:0,skipped:0,saved:0,items:[{
  frame_id:'frame-000',camera_id:'front',status:accepted?'accepted':'rejected',matrix:candidate,reason:accepted?'':reason,
  metrics:{input_source:'synchronized_sensor_scan',line_overlap_before:.8321,line_overlap_after:.8841,translation_delta_m:.9,rotation_delta_degrees:0},
 }]};
 await page.route('**/api/clips/demo/automatic-calibration**',async route=>{
  const url=new URL(route.request().url());
  if(url.pathname.includes('/preview/'))return route.fulfill({json:{...baseline,job_id:job.id,candidate_status:accepted?'accepted':'rejected',read_only:true,candidate_matrix:candidate,comparison_matrix:baseline.effective_matrix,comparison_source:'solver_initial',usable:accepted,metrics:job.items[0].metrics,rejection_reasons:accepted?[]:[reason],method:'pjlab'}});
  return route.fulfill({json:{engines:[{id:'pjlab',name:'PJLab',available:true,detail:'Fixture'}],latest_job:job,total_pairs:84,saved_pairs:0}});
 });
 await page.goto('/');
 await expect(page.getByLabel('全clip自动标定',{exact:true})).toBeEnabled({timeout:60000});
 const front=page.getByTestId('camera-front');
 const before=await front.getAttribute('data-projection-signature');
 await page.getByLabel('全clip自动标定',{exact:true}).click();
 await page.getByRole('button',{name:'查看本次结果 / 手动校准',exact:true}).click();
 const editor=page.getByTestId('projection-calibration-editor');
 await expect(editor.getByTestId('automatic-candidate-review')).toHaveAttribute('data-candidate-status',accepted?'accepted':'rejected');
 await expect(page.getByTestId('automatic-projection-preview')).toBeVisible();
 await expect.poll(()=>front.getAttribute('data-projection-signature')).not.toBe(before);
 const after=await front.getAttribute('data-projection-signature');
 const rawAfter=await front.getAttribute('data-raw-projection-signature');
 await editor.getByRole('button',{name:'修正前',exact:true}).click();
 await expect(front).toHaveAttribute('data-projection-signature',before!);
 await expect.poll(()=>front.getAttribute('data-raw-projection-signature')).not.toBe(rawAfter);
 await editor.getByRole('button',{name:'候选修正',exact:true}).click();
 await expect(front).toHaveAttribute('data-projection-signature',after!);
 await expect(editor.getByLabel('本帧平移X加')).toBeDisabled();
 await expect(editor.getByLabel('保存本帧本相机投影修正')).toHaveCount(0);
 expect(await(await request.get(path)).json()).toEqual(baseline);
 expect(await(await request.get('/api/clips/demo/frames/frame-000/annotations')).json()).toEqual(labels);
 if(accepted){
  await editor.getByRole('button',{name:'返回自动标定任务',exact:true}).click();
  await expect(page.getByRole('dialog',{name:'全 clip 自动标定',exact:true})).toBeVisible();
  await expect(front).toHaveAttribute('data-projection-signature',before!);
 }else{
  await editor.getByLabel('退出候选预览并手动校准').click();
  await expect(editor.getByLabel('本帧平移X加')).toBeEnabled();
  await expect(front).toHaveAttribute('data-projection-signature',before!);
  await editor.getByLabel('本帧平移X加').click();
  await expect(editor.getByLabel('保存本帧本相机投影修正')).toBeEnabled();
  // Manual adjustment starts at the saved initial value, never the rejected .9m result.
  const manualSignature=await front.getAttribute('data-projection-signature');
  expect(manualSignature).not.toBe(after);
  page.once('dialog',dialog=>dialog.accept());
  await editor.getByLabel('关闭投影校正').click();
  await expect(front).toHaveAttribute('data-projection-signature',before!);
 }
 expect(await(await request.get(path)).json()).toEqual(baseline);
});
