# 实时外参编辑与导出

打开场景后点击「标定外参」。选择某一路相机的 `ego → camera`，或 `LiDAR → ego`。
通过平移和 Roll / Pitch / Yaw 修正滑块、数字微调，实时查看相机上的标注和原始点投影。
平移单位为米，角度单位为度；当前 4×4 矩阵同时显示在面板中。

修正是相对当前选中变换的增量，不把原始旋转反解成欧拉角。
旋转修正在变换的目标坐标系中采用 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`，
再左乘原有旋转；平移增量加到目标坐标系中的原平移向量。
因此相机变换的修正轴属于光学相机坐标，LiDAR 变换的修正轴属于 ego 坐标。
零修正保持原矩阵，不会因为欧拉角奇异点而改变已有标定。

LiDAR 可选择只修改当前帧，或把同样的增量应用到整个 clip。
修改安装外参后，当前相机投影立即预览；全场景融合地图需要重新重建。
世界坐标轨迹和车道线保留原坐标，重建后应复核它们是否仍与点云对齐。
已有 ICP 估计的 ego 位姿会保留，新的注册变换为：

```text
world_from_lidar_new = world_from_lidar_old @ inverse(lidar_to_ego_old) @ lidar_to_ego_new
```

仅调整相机外参时，原来的点云地图和 ICP 注册继续有效。

点击「保存外参」把修改写入 clip；并发修改会被 revision 检查拒绝，不会覆盖其他人的新标定。
点击「导出外参」下载 JSON；未保存的导出会在文件名及内容中标为草稿。
导出包括全部相机的 `camera_from_ego`、逐帧 `lidar_to_ego`，以及对应注册变换和坐标约定。
「还原当前」「还原全部」恢复到面板最近读取或保存的状态。

服务端首次保存时在 `data/calibration_history/<clip-id>/original.json` 留存原始外参，
每次保存前另存旧 revision。原始 NPY 点云文件、图像、单帧标注和场景标注不会被改写。

接口为 `GET /api/clips/<id>/calibration`、同址 `PUT`，以及
`GET /api/clips/<id>/calibration/export`。保存请求只含 revision 和变换矩阵，
不上传或重新编码原始图像、点云。
