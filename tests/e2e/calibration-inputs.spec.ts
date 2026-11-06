import {expect,test} from '@playwright/test';

test('PJLab input check reads actual data, explains missing mask and refreshes after upload',async({page,request})=>{
 const clip=await(await request.get('/api/clips/demo')).json();
 const camera=clip.cameras.find((c:{id:string})=>c.id==='front');
 const path='/api/clips/demo/frames/frame-000/annotations';
 const before=await(await request.get(path)).json();
 await page.route('**/api/clips/demo/automatic-calibration',route=>route.fulfill({json:{
  engines:[{id:'pjlab',name:'PJLab',available:true,detail:'Browser engine-selection fixture; input diagnostics use the actual server.'}],
  latest_job:null,total_pairs:84,saved_pairs:0,
 }}));
 await page.goto('/');
 await expect(page.getByLabel('全clip自动标定',{exact:true})).toBeEnabled({timeout:60000});
 await page.getByLabel('全clip自动标定',{exact:true}).click();
 const panel=page.getByRole('dialog',{name:'全 clip 自动标定',exact:true});
 await panel.getByLabel('检查当前帧标定输入',{exact:true}).click();
 await expect(panel.getByTestId('calibration-input-image')).toHaveAttribute('data-status','ready');
 await expect(panel.getByTestId('calibration-input-intrinsics')).toHaveAttribute('data-status','ready');
 await expect(panel.getByTestId('calibration-input-initial')).toHaveAttribute('data-status','ready');
 await expect(panel.getByTestId('calibration-input-mask')).toHaveAttribute('data-status','missing');
 await expect(panel.getByTestId('calibration-input-intensity')).toHaveAttribute('data-status','ready');
 await expect(panel.getByTestId('calibration-input-synchronization')).toHaveAttribute('data-status','warning');
 await expect(panel.getByTestId('calibration-input-synchronization')).toContainText('使用提示（非缺失项）');
 await expect(panel.getByTestId('calibration-input-mask')).toContainText('SAM2');
 const data=await page.evaluate(({width,height})=>{
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d')!;ctx.fillStyle='black';ctx.fillRect(0,0,width,height);
  ctx.fillStyle='white';ctx.fillRect(width*.2,height*.7,width*.6,6);
  return canvas.toDataURL('image/png').split(',')[1];
 },{width:camera.width,height:camera.height});
 await panel.getByLabel('上传当前帧PJLab线特征掩码').setInputFiles({name:'mask.png',mimeType:'image/png',buffer:Buffer.from(data,'base64')});
 await expect(panel.getByText('已保存 frame-000 / front 的 PJLab 线特征掩码。',{exact:false})).toBeVisible();
 await expect(panel.getByTestId('calibration-input-report')).toHaveCount(0);
 await panel.getByLabel('检查当前帧标定输入',{exact:true}).click();
 await expect(panel.getByTestId('calibration-input-mask')).toHaveAttribute('data-status','ready');
 await expect(panel.getByTestId('calibration-input-synchronization')).toContainText('不能证明物理同步');
 await panel.getByText('第一次使用：按这个顺序操作',{exact:true}).click();
 await expect(panel.getByText('当前帧掩码不能代替其他图像的掩码。',{exact:false})).toBeVisible();
 expect(await(await request.get(path)).json()).toEqual(before);
});
