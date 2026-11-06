# 震动数据的逐帧相机 / LiDAR 标定

每个 `(clip_id, frame_id, camera_id)` 保存独立的 `camera_from_lidar` 刚体矩阵。
没有独立结果的帧沿用导入外参；自动候选不会立即替换已有参数。逐帧结果优先于
已有的整 clip 修正，同时用于原始点云、世界车道线、框等图像投影。原始扫描、
三维标注和场景配准不会被改写。导出的全 clip 参数包含每帧每相机的有效矩阵。

## 手动调整

原来的「外参校准」入口同样默认按当前帧、当前相机保存：选定帧和相机后，使用 XYZ、
Roll/Pitch/Yaw 滑块或数字输入，检查投影，点击「保存外参」。面板显示当前帧 ID、相机和
作用范围；切到下一帧会读取它自己的有效外参并将修正增量归零，刷新后保留各帧的独立结果。
「导出外参」默认导出当前帧相机的矩阵与草稿状态。修改只在保存后持久化。

「应用整个 clip」默认不勾选。显式勾选后才进入共享安装外参编辑，会影响全部帧；
切换传感器或重新打开编辑器恢复本帧默认范围。切帧、换相机和换范围时可取消放弃未保存
修改；保存过程中禁止切帧，避免结果落到错误的上下文。

需要对应点求解或六轴加减按钮时，可继续使用「逐帧校准」：

1. 选择帧和相机，打开「逐帧投影校准」。默认范围为「仅当前帧 · 当前相机」，
   包括已导入融合地图的场景，不再默认共享整段修正。
2. 使用「当前帧外参手动微调」的平移 X/Y/Z、旋转 X/Y/Z 加减按钮。
   平移单位米，旋转单位度；坐标为相机光学系：X 右、Y 下、Z 前。
   刚体增量在光学坐标系左乘整个投影矩阵；纯旋转保持相机在源坐标系中的位置不变，
   避免远离原点的融合地图发生异常平移，不分解原始 Euler 角。
3. 在「修正前 / 候选修正」间切换，检查近远处静态结构，点击「保存本帧本相机」。
   切换帧前会检查未保存修改。手动结果标记 `manual_review`，不冒充自动质量验收；
   可清除本帧修正恢复继承值。手动结果不能直接推广到整个 clip。

## PJLab CRLF 原生引擎

接入的是 [SensorsCalibration/lidar2camera/auto_calib](https://github.com/PJLab-ADG/SensorsCalibration/tree/master/lidar2camera/auto_calib)
的 CRLF 线特征优化器，固定源码版本 `7fe8d46caac040cbcd89cc65e79e5519df01bcd4`。
不是该项目另一个需要 SAM 掩码的 `auto_calib_v2.0 / CalibAnything`，也不把内置 CPU
NID 优化器当作 PJLab 算法。上游实现、模型和数据的许可分别适用；本仓库不复制其源码或权重。

平台提供原生 CLI 适配器和容器构建脚本。需要可用的 Docker daemon、Git、约 4 GB
空闲磁盘，以及 GitHub、Docker Hub、Ubuntu 软件源访问权限。构建使用两路并行。

```bash
cd /workspace/OmniLabel
bash scripts/setup-pjlab.sh
export OMNILABEL_PJLAB_BINARY="$PWD/scripts/pjlab-worker.sh"
bash scripts/serve.sh
```

也可将 `OMNILABEL_PJLAB_BINARY` 指向自行编译的原生 `run_lidar2camera`。
容器仅将上游 CMake 中静态 JsonCpp 链接名替换为 Ubuntu 支持的共享库链接，优化算法不变。

容器默认名 `omnilabel-pjlab:crlf`，可用 `OMNILABEL_PJLAB_DOCKER_IMAGE` 修改。
`OMNILABEL_PJLAB_TIMEOUT` 为单次命令超时（默认 300 秒，允许 1–3600 秒）。
容器只挂载当前组合的临时输入目录，以当前用户运行，不需要 GPU 或网络。

### 线特征掩码是必要输入

上游程序接收 `mask_path pcd_path intrinsic_json extrinsic_json`，不从原图直接提取语义特征。
需要真实 XYZI 反射强度和近似正确的初始外参；它主要针对带道路标线、杆体的前向道路场景，
上游内置了前向 LiDAR 范围和道路平面假设，不能保证侧向/后向相机或非道路场景的效果。
平台保留原始 LiDAR 坐标及反射率；仅将明确处于 `[0,1]` 的归一化强度映射到 `[0,255]`，
以匹配上游的 80/255 标线阈值。同步单帧扫描优先于融合地图，结果再转换到主点云坐标。

可选以下任一方式提供每张图像的掩码：

- SAM2 辅助生成：在「全 clip 自动标定」选择 PJLab，使用下方「SAM2 生成线特征掩码」。
  平台复用模型面板中已配置的 SAM2 服务，只发送当前帧、当前相机图像。选择「整图自动候选」生成区域，
  或选择「正负提示点分割」，在标线上添加正点、在应排除区域添加负点后生成。
  勾选候选中的道路标线/杆体区域，检查原图叠加和黑白预览，点击「保存选定区域为线特征掩码」。
  不需要下载或重新上传 PNG，服务器按原始像素坐标生成同尺寸黑底白色区域，PJLab 随后直接使用。
  SAM2 不按文本辨认车道线，因此不默认全选候选，不应选整片路面、车辆和天空。
  多边形近似可能丢失孔洞或细边界，需复核预览；细标线漏检时可改用提示点。
  此操作只保存标定掩码，不修改语义标注、车道线、三维目标或外参。
  每一帧、每一相机分别生成，不能把当前帧掩码复制用于整段；缺少掩码的批量项仍会报告失败。
  候选记录绑定原图版本、帧和相机，图像变化、未完成/错误模型候选、跨相机选择会被拒绝；
  自动标定运行或保存期间不能替换掩码。旧版本 SAM2 候选需要重新生成才能用于该入口。
- 手动上传：在「全 clip 自动标定」选择 PJLab，上传当前帧相机对应的 PNG。
  掩码必须与未去畸变的原图同尺寸、同像素坐标，道路标线/杆体为白色，背景为黑色。
  原图变化后必须重新上传。每个组合各自保存，批量任务对缺少掩码的组合单独报告失败。
- 自动生成：部署上游 `auto_calib/tool` 的语义分割依赖和模型，或经验证的同语义掩码生成程序，
  设置 `OMNILABEL_PJLAB_MASK_COMMAND` 为 JSON 参数数组，包含 `{image}`、`{mask}` 占位符。
  例如 `['/absolute/python', '/absolute/merge_mask.py', '{image}', '{mask}']` 的 JSON 双引号形式。
  程序需从任意工作目录运行，写出同尺寸 PNG；如上游脚本依赖其自身工作目录，请使用包装脚本
  先进入该工具目录再执行。平台逐帧传入原图并验证输出，已有上传掩码优先使用。

掩码生成模型和权重不随平台安装；未配置自动提取时仍可上传掩码运行。不会用原始彩色照片、
简单图像边缘或虚构分割结果替代上游要求的语义线特征。

在「处理范围」选择当前组合、全部帧的当前相机或全部组合，启动后逐项生成候选。
默认跳过已有保存外参，只有勾选重新计算并明确保存后才替换它们。失败、超时、质量不合格、
源数据变化的组合不会覆盖有效标定。通过项可逐项打开复核、手动微调后保存。

原生结果从上游最终 `refine extrinsic` 输出前的旋转矩阵及其 translation 读取，验证有限性、
正交性和右手系；不猜测 Euler 单位，不把初值或历史日志作为成功结果。每次执行有独立目录。
平台读取原生 before/after loss（线特征掩码匹配比例），要求匹配改善至少 0.01、
最终匹配比例至少 0.1，且修正不超过 0.75 米 / 3 度。只有来自服务端已结束任务（包含中断后保留的通过项）的
确切候选才可保存；保存时核对矩阵、源数据指纹，并重新检查评分和修正幅度，不接受客户端
自报评分。PJLab 使用完整输入扫描训练，这项评分是拟合后的检查，不是独立验证集或
像素真值精度证明。真实震动数据仍需逐帧视觉复核和质量验收。

### API 和验证

`PUT /api/clips/{clip}/automatic-calibration/masks/{frame}/{camera}` 的请求体为 PNG（最多 16 MiB）。
`POST /api/clips/{clip}/automatic-calibration/jobs` 接收 `engine: "pjlab"`、`frame_ids`、`camera_ids`。
保存继续使用任务 `/apply`；手动微调使用逐帧投影接口 `method: "manual", scope: "frame"`。
保存保留 revision 冲突检查、备份和源数据指纹检查。

```bash
PYTHONPATH=backend .venv/bin/python -m pytest
npm run build
npm run test:e2e -- tests/e2e/automatic-calibration.spec.ts tests/e2e/projection-clip-scope.spec.ts
```

原生 CLI 契约测试用受控进程验证输入、输出、超时和错误处理，不代表真实算法精度。
部署时还应运行上游自带样例，再用目标传感器真实数据验收。

### SAM2 掩码接口

生成候选复用 `POST /api/clips/{clip_id}/jobs`，参数为 `kind=prelabel`、`model_id=sam2`、
`frame_id` 与 `options.camera_id`；SAM2 的结果增加 `source_image_token`、`source_camera_id` 和
`sam2_candidate_ids`。只展示本次新增的当前相机区域，已有人工标注不作为新掩码候选。

`POST /api/clips/{clip_id}/automatic-calibration/masks/{frame_id}/{camera_id}/from-sam2` 接收
`{"job_id": "...", "polygon_ids": ["..."]}`。服务器读取已经验证的推理结果，
重新检查原图版本，栅格化选中区域并调用原有掩码校验/保存流程；不会接收客户端任意几何作为模型结果。
