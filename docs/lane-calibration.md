# 多帧车道线外参优化

此功能在已有世界坐标三维真值线与已有图像投影线之间建立明确配对，优化同一路相机的一组共享
`T_CL`。它用于固定安装偏差的联合标定；若真实相对外参随震动逐帧改变，请使用逐帧校准。
第二条线应独立来自图像标注；不可复制第一条真值线按当前外参生成的投影作为图像真值。

## 安装与使用

```bash
cd /workspace/OmniLabel
bash scripts/setup.sh
bash scripts/setup-lane-calibration.sh
bash scripts/serve.sh
```

优化是可选 CPU 功能，需要 PyTorch、PyYAML、Matplotlib。基础平台未安装它们时仍可启动、
管理配对和计算初值投影，优化按钮会显示未安装。无需 GPU、模型权重或云账号。

1. 将点云真值线和从图像投到点云的车道线都保存为世界坐标开放矢量线，选择对应帧及相机，点击顶部「车道线优化」。
2. 在点云中点选真值线后点击「使用选中三维线」，或在面板 ID 列表中选择并核对点云高亮。
3. 在点云中点选由当前帧、当前相机图像投影得到的另一条线，点击「使用选中图像投影线」，
   或使用第二个 ID 列表选择。无需重新在图像上描线；两条线不能使用同一个 ID。
   不要求两条线具有相同点数，也不将同序号顶点强制视为对应点。
4. 按「添加配对」保存；配对编号即人工选择顺序，可上移或删除。长地图车道线可用
   「三维线取样范围」和「图像投影线取样范围」分别选择可见对应段。添加后在原图核对绿色固定图像线。
5. 切换帧，重复配对。正式模式至少选择两个训练帧和一个独立验证帧，尽量覆盖左右、远近、
   转弯和不同方向的线。同一帧的全部配对只能属于同一集合。修改一项用途会更新该帧全部配对。
6. 阅读坐标信息，确认变换方向、原图坐标、畸变状态及图像/pose 同步，然后先点「预览当前投影与误差」。
7. 运行优化。绿色为保存时固定的图像线，橙色为真值线初值投影，青色为优化后投影；切帧查看对应比较。
8. 检查训练/验证 Mean、RMSE、Median、P95、误差下降率及每条线的误差，再下载结果。
   正式候选通过检查后才允许明确确认「应用到整段当前相机」；不会自动替换原始安装外参。

单帧调试只使用当前帧的配对，不改变保存的训练/验证分组；其结果仅供检查和导出，不能应用。
后台任务在关闭面板后继续运行，重新打开会恢复显示。可取消任务；服务重启会将未完成任务标为失败，
不会把部分结果当成完整标定。切帧会清空尚未添加的两次选择，提交配对时暂停切帧。
校准期间点云允许点选、旋转和缩放，几何编辑保持锁定，避免选线时改动三维标注。

## 已确认的数据格式与坐标

- 三维线：复用 `scene.vectors`，其 `points` 是世界坐标，要求开放折线、至少两个不同顶点。
- 图像投影线同样选自 `scene.vectors`。现有矢量只保存三维坐标，不含来源相机、来源帧或原始二维像素，
  因此操作者的当前帧/相机选择是明确的来源声明；不可自动推断某条线来自哪个图像。
- 配对保存到独立的 `lane_calibration/<clip>/<camera>/pairs.json`，包括 `lane_id`、`image_lane_id`、
  各自的取样范围、固定原图像素 `image_points` 和参考投影记录 `image_reference`。
  旧版已经保存的手绘二维配对仍可读取、重排、删除并参与优化；新界面使用两次点云选线。
- 内参与畸变：复用 `Camera.intrinsics`、`Camera.distortion`，Brown–Conrady 顺序
  `[k1,k2,p1,p2,k3]`。实际图像尺寸必须等于相机声明；不同分辨率、裁剪、去畸变图像必须先提供相匹配的 K 和 D，
  程序不会猜测缩放或裁剪参数。
- `ego_to_world` 为 ego→world；`lidar_to_ego` 为主点云坐标→ego；相机安装矩阵为
  `camera_from_ego`，ego→camera。优化的输出始终是实际标定传感器的 LiDAR→camera。

**全局地图的主点云坐标不一定是真实 LiDAR 坐标。** 现有 4D 导入器可将世界地图作为主点云，
把 `lidar_to_ego` 写成 `inverse(ego_to_world)`，并把地图注册矩阵写成单位阵。
因此不能直接把这个单位阵当作车辆 pose，也不能在已有世界→相机矩阵中重复叠加 pose。

自动模式先获取现有真实/ICP 注册的 `world_from_primary`，再按已知物理安装关系计算：

```text
T_WL(t) = world_from_primary(t) @ inverse(primary_to_ego(t)) @ sensor_to_ego
P_L(t)  = inverse(T_WL(t)) @ P_W
P_C(t)  = T_CL @ P_L(t)
q(t)    = BrownProject(K, D, P_C(t))
```

真实安装关系优先使用请求显式提供的 `sensor_lidar_to_ego`，其次使用已导入的同步扫描
`calibration_lidar_to_ego`，或无需原始扫描的帧字段 `sensor_lidar_to_ego`（4D 导入器从源 `lidar_calib` 保留）；
普通独立扫描可复用 `lidar_to_ego`。同步扫描与主点云即便来自不同传感器，
也通过上式进行转换。全局地图缺少真实安装关系时明确报错；只能由操作者补充安装矩阵、提供外部 Pose，
或明确选择「LiDAR 与 ego 重合」模式，不会自动假设两者相同。

已有逐帧投影修正不被混入共享外参初值；默认初值是 `camera_from_ego @ sensor_to_ego`。
如需使用别的初值，显式传入 `initial_matrix`，并通过 `initial_direction` 声明其方向。

### 将选中的图像投影线固定为图像约束

第二条线必须独立来自图像标注。不能将第一条真值线的投影复制一份当作测量值。
保存配对时，后端按当前帧实际注册的 `world_from_primary` 将第二条世界线转回主点云坐标，
再使用当前已保存的有效投影矩阵（包含逐帧、clip 或导入修正）以及 K/D 回投到原图。
最多采样 512 个点，剔除相机后方和画面外点；不连续可见区间会要求缩小取样范围，避免虚构跨区域线段。

该回投结果在保存时冻结，记录参考帧、相机 K/D、`world_from_primary`、`camera_from_primary` 和源线几何指纹。
优化只改变第一条真值线的投影，不重新计算第二条线。改变优化初值、标定或重排配对都不会移动固定像素。
若第二条线被修改或删除，旧配对必须重新选择，程序拒绝静默重建目标。

由于原始像素未保留，只有使用当初生成该投影线的帧、相机及投影参数时，才能恢复对应图像位置。
若当初使用了其他标定或帧，须先恢复对应参考；不能声称用任意当前外参回投便能找回原始二维真值。

### 时间同步与外部 Pose

现有 clip 按帧保存 `timestamp` 和 pose，没有普遍适用的独立相机时间戳字段。默认使用既有帧关联，
要求操作者确认同步，不声称已经自动验证采集同步。存在偏移时应提供真实图像时间戳和 pose 样本，
使用 SLERP 旋转插值与线性平移插值，不允许外推或跨过大的 pose 间隔。

高级 JSON 请求示例（数值需替换为实际数据，不要照抄单位矩阵）：

```json
{
  "initial_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "initial_direction": "camera_from_lidar",
  "pose_direction": "world_from_lidar",
  "pose_samples": [
    {"timestamp": 0.0, "matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
    {"timestamp": 0.1, "matrix": [[1,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
  ],
  "image_timestamps": {"frame-000": 0.02},
  "max_pose_gap_seconds": 0.2,
  "config": {"samples_per_lane": 48, "huber_delta_px": 4, "chamfer_weight": 0.5, "prior_weight": 0.01}
}
```

界面需同时选择外部 Pose 模式。支持 `lidar_from_world` 和 `lidar_from_camera` 输入，先统一求逆再使用。
外部同步 Pose 尚未写入原平台，因此该模式结果只能导出，不直接应用到仍使用旧 pose 的平台。

## 优化与质量边界

- 采用 float64，先减世界原点再旋转，保留大世界坐标精度。
- 以六维 SE(3) 增量更新 `T_CL = T0 @ Exp(xi^)`，包含平移–旋转耦合，不独立优化旋转矩阵九个元素。
- 按三维弧长和二维弧长独立采样，计算点到折线线段的精确最短距离；使用 Huber 鲁棒损失、
  可配置的对称 Chamfer 距离及初值先验。先按帧内车道线权重归一化，再对帧平均。
- CPU Adam 后可用 LBFGS 精调；默认平移增量范数小于 0.5 米、旋转小于 3 度，可在有限范围内配置。
- 过滤相机后方与画面外点。固定初值可见支持集，丢失点加入强惩罚；统计中丢失点计为图像对角线误差，
  防止通过移出画面取得虚假改善。投影分段断开无效点，不跨过不可见区连接虚构线段。
- 独立验证帧完全不参加训练、先验或最优迭代选择。报告常规点级 Mean/RMSE/Median/P95，同时报告帧平衡误差。
- 允许应用要求训练和验证的常规 Mean 与帧平衡 Mean 均改善至少 0.5%，可见支持保留至少 98%，
  数据残差雅可比局部秩为 6 且条件数合理。单直线、退化几何和已准确初值可能被拒绝应用。
- 可观测性与验证误差不是绝对标定真值证明。同步错误、滚动快门、运动畸变、SLAM 误差或真实逐帧安装变化，
  可能无法由一组共享刚体外参解释；仍需检查未参与优化的特征和实际采集数据。

## API

基础路径为 `/api/clips/{clip_id}/lane-calibration/{camera_id}`。

| 方法及路径 | 功能 |
| --- | --- |
| GET 基础路径 | 已保存三维线、配对、坐标提示、依赖状态、最近任务 |
| POST `/pairs/from-lanes` | 两次点云选线，捕获并固定当前参考图像位置，追加有序配对 |
| PUT `/pairs` | 原子保存有序配对，检查 revision、ID、原图坐标及帧级分组 |
| POST `/preview` | 初值投影与误差，无需先凑齐正式训练/验证集 |
| POST `/jobs` | 创建持久后台任务，返回 202 与任务 ID |
| GET `/jobs/{id}` | 状态、进度、候选矩阵、前后投影、统计和拒绝原因 |
| POST `/jobs/{id}/cancel` | 请求取消 |
| POST `/jobs/{id}/apply` | 明确确认后应用合格共享外参，检查源数据指纹 |
| GET `/jobs/{id}/download` | 下载结果 ZIP |

两次点云选线请求示例，发送到 `POST /pairs/from-lanes`：

```json
{
  "revision": 0, "id": "pair-1", "frame_id": "frame-000",
  "lane_id": "world-lane-left", "image_lane_id": "image-projected-left",
  "world_range": [0,1], "image_world_range": [0,1], "split": "train", "weight": 1
}
```

响应为递增 revision 后的完整配对列表，服务端生成固定像素与参考记录。
`PUT /pairs` 用于重排、删除、分组及保留旧版二维配对，不会重新回投目标。
同一帧的同一真值线或同一图像投影线均只能配对一次。

优化请求包含 `pairing_revision`、`mode: "multi"`、`conventions_confirmed: true`、
`time_alignment_confirmed: true`、`pose_mode: "auto"` 和可选 `config`。
`frame_ids` 可显式选择已配对帧；单帧模式必须选择一个帧，临时作为调试训练数据，
不会改写保存的验证分组。正式模式至少 2 个训练帧 + 1 个验证帧。
最多 128 组配对，每组二维线最多 512 顶点；采样数 10–128，迭代与超时有明确上限，不会静默裁掉配对。

优化不改写标注或安装外参。应用时检查世界线、配对、相机、pose 和图像指纹，拒绝过期候选，
复用原有安装外参备份与 revision 检查。确认应用共享候选时，当前相机的导入逐帧修正会被备份并替换，
已保存的本地逐帧校准因源矩阵变化而失效，避免在优化结果上重复叠加旧修正；其他相机不受影响。

## 输出与独立运行

结果位于 `data/lane_calibration/<clip>/<camera>/jobs/<id>/`，包括：

- `optimized_extrinsic.npy`、`optimized_extrinsic.txt`：优化后的 LiDAR→camera 4×4 矩阵。
- `initial_extrinsic.npy/txt`：统一为 LiDAR→camera 后的初值；`initial_input_extrinsic.npy` 保留原方向输入。
- `optimization_log.csv`：逐次 Adam 迭代、LBFGS 线搜索评估的损失、六维增量及最终选中的参数，旋转增量单位为弧度。
- `before_after/<frame>-before.png` 与 `-after.png`：原图、人工线、投影线和帧误差。
- `loss_curve.png`：训练目标曲线。
- `error_statistics.json`：按配对、训练/验证分组的前后统计、下降率、可见性、可观测性和拒绝原因。
- `optimization_config.yaml`：实际参数、方向和时间配置。
- `input_snapshot.json`：准确的输入几何、实际使用的 pose 和图像路径，便于复现。

数值核心与 FastAPI 解耦。可在同一数据环境独立重放导出的快照：

```bash
.venv/bin/python scripts/optimize-lanes.py \
  --snapshot data/lane_calibration/CLIP/CAMERA/jobs/JOB/input_snapshot.json \
  --output /tmp/lane-calibration-replay
```

也可直接读取平台已有标注和配对：

```bash
.venv/bin/python scripts/optimize-lanes.py --data-dir data --clip CLIP --camera CAMERA \
  --settings /path/to/optimization-request.yaml --output /tmp/lane-calibration-new
```

输出目录必须不存在，避免覆盖以前的结果。快照中的图像路径须在执行机器可用；跨机器使用需明确更新路径。
该脚本只读原数据，不调用平台应用接口。

验证命令：

```bash
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests/test_lane_optimizer.py backend/tests/test_lane_calibration_api.py
npm run build
npm run test:e2e -- tests/e2e/lane-calibration.spec.ts
```

已用带独立验证帧、平移/转弯、大世界坐标和已知外参真值的合成数据进行数值与 API 验证。

## 仓库真实 example 验证

2026-11-02 检查了用户在 `18fd154` 提交的
`example/4D_clip_DATA_01_20260808160513v12`，并导入为 `4d-clip-data-01-version2`：

- 50 帧、7 路 1920×1536 图像，13,477,403 个地图点全部保留，11 个世界地图标注中有 10 条开放折线。
- 源 `ego_pose_local` 为 ego→local-world；`lidar_calib` 为 LiDAR→ego，全部帧一致。
  验证 `camera_external(t) @ ego_to_world(t)` 固定，因此源 `camera_external` 已经是 local-world→camera，
  不能再把它当成 LiDAR→camera 重复叠加 pose。
- 对全部 350 个相机帧检查了物理雷达 pose、内参、畸变和实际图像尺寸。
  `T_CL @ inverse(T_WL(t))` 与源世界→相机矩阵的最大元素差为 `2.15e-10`，
  186,943 个可见投影样本的最大像素差为 `3.16e-7` 像素。
  首帧、中间帧、末帧各 7 路生成投影图，供目视检查。
- `label/map.json` 提供三维线；147 组 `projection_corrections` 只是旧投影修正，
  不是独立二维线。样例未提供独立图像车道线标注，也没有独立相机曝光时间戳。
  因此上述结果只验证坐标与投影实现一致性，不能宣称真实标定误差下降或时间同步已实测验证。
  正式验收需通过新面板手动配对至少 2 个训练帧和 1 个验证帧后运行优化。
- 在完整地图上检查了三维 ID 选择、原图临时取点、切帧放弃未保存线，确认源外参和世界标注未改变。
  点云视图在静止时复用画面，仅在视角或内容变化时重绘，避免软件 GPU 持续重复绘制 1,347 万点。
  初次完整地图加载和视角切换仍需等待；此调整不减少点数。

复现命令（不会覆盖现有同 ID 数据或现有报告目录）：

```bash
git lfs install --local
git lfs pull --include='example/**'
.venv/bin/python scripts/import_4d_clip.py example/4D_clip_DATA_01_20260808160513v12 \
  --clip-id 4d-clip-data-01-version2
.venv/bin/python scripts/validate-lane-example.py --output /tmp/real-lane-example-report
```

已导入该 ID 时跳过导入命令；旧版导入缺少物理安装矩阵时，应以新 ID 导入，
或明确在高级配置中提供源 `lidar_calib` 的矩阵，不能猜用单位阵。
