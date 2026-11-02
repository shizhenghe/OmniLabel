# 实时外参编辑与导出

打开场景后点击「外参校准」。默认选择当前相机，按当前帧独立编辑 `LiDAR → camera`。
面板显示帧 ID、相机和作用范围，不再默认修改整个 clip 的相机外参。
通过平移和 Roll / Pitch / Yaw 修正滑块、数字微调，实时查看相机上的标注和原始点投影。
平移单位为米，角度单位为度；当前 4×4 矩阵同时显示在面板中。

修正是相对当前选中变换的增量，不把原始旋转反解成欧拉角。
本帧相机模式以光学坐标系中的刚体增量左乘整个 `camera_from_lidar`，即 `ΔT @ T`；
纯旋转同时变换矩阵的平移列，保持相机在源点云坐标系的位置不变，支持大世界坐标融合地图。
共享安装外参与 LiDAR 模式保留旋转修正 `Rz(yaw) @ Ry(pitch) @ Rx(roll)` 左乘原旋转、
平移增量加到目标坐标系原平移向量的定义。
因此相机变换的修正轴属于光学相机坐标，LiDAR 变换的修正轴属于 ego 坐标。
零修正保持原矩阵，不会因为欧拉角奇异点而改变已有标定。

相机与 LiDAR 默认仅修改当前帧。相机的独立结果与逐帧投影校准、自动标定共用持久记录，
同时用于原始点云和三维标注投影，不改写共享相机安装外参、原始扫描或世界坐标标注。
切帧后读取目标帧的有效矩阵，增量归零；保存、关闭并重新打开后仍保持各帧的不同结果。

只有显式勾选「应用整个 clip」才修改共享的相机 `ego → camera` 安装外参，
或把 LiDAR 的同一增量应用到全部帧。此时所有帧都会受影响，已有逐帧修正可能因安装外参变化失效，需重新检查。
修改安装外参后，当前相机投影立即预览；全场景融合地图需要重新重建。
世界坐标轨迹和车道线保留原坐标，重建后应复核它们是否仍与点云对齐。
已有 ICP 估计的 ego 位姿会保留，新的注册变换为：

```text
world_from_lidar_new = world_from_lidar_old @ inverse(lidar_to_ego_old) @ lidar_to_ego_new
```

仅调整相机外参时，原来的点云地图和 ICP 注册继续有效。

点击「保存外参」默认只写入当前帧与相机的独立记录；并发修改会被 revision 检查拒绝，不会覆盖其他人的新标定。
保存期间暂停切帧；切帧、换相机或改变范围时，未保存修改可选择保留当前上下文或放弃。
点击「导出外参」下载 JSON；未保存的导出会在文件名及内容中标为草稿。
相机本帧模式导出包含 `frame_id`、`camera_id`、有效矩阵及未保存候选。
共享安装外参或 LiDAR 模式导出全部相机的 `camera_from_ego`、逐帧 `lidar_to_ego`，以及对应注册变换和坐标约定。
「还原当前」「还原全部」恢复到面板最近读取或保存的状态。

独立相机外参保存在 `data/projection_calibrations/<clip-id>/<frame-id>/<camera-id>.json`，
后续修改的旧版本保存在 `projection_calibration_history/`。共享安装外参首次保存时在
`data/calibration_history/<clip-id>/original.json` 留存原始外参，
每次保存前另存旧 revision。原始 NPY 点云文件、图像、单帧标注和场景标注不会被改写。

相机本帧模式使用 `GET/PUT /api/clips/<id>/frames/<frame-id>/projection-calibration/<camera-id>`，
保存时明确指定 `scope: "frame", method: "manual"`。
共享安装外参与 LiDAR 接口为 `GET /api/clips/<id>/calibration`、同址 `PUT`，以及
`GET /api/clips/<id>/calibration/export`。保存请求只含 revision 和变换矩阵，
不上传或重新编码原始图像、点云。
