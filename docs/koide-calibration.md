# Koide 图像 / LiDAR 自动精准配准

平台通过 `backend/omnilabel/koide_runtime.py` 调用真正的
[Koide direct_visual_lidar_calibration](https://github.com/koide3/direct_visual_lidar_calibration)，
使用当前 `camera_from_lidar` 作为初值，运行无标靶 NID 外参优化。
任务逐组处理同步图像和点云；同一个矩阵用于点云和三维标注的图像投影。
运行环境缺失、超时、进程异常、没有输出结果时，任务会报告失败，保留输入和日志，
不会把原外参或其他优化器的结果冒充 Koide 标定结果。

本功能优化 LiDAR 到相机的旋转和平移；相机内参、五个畸变系数沿用已有数据。
它不自动估计相机内参、曝光时刻、时间偏移、滚动快门或逐点运动补偿。
原图和点云需要已经同步，`intensity` 必须来自真实 LiDAR 反射强度。
全零强度、纯色图像、图像尺寸与内参记录不符会被拒绝。
精度依赖纹理、有效重叠、遮挡、动态物体和初值，运行成功仍需查看叠加投影和质量检查。

融合后的 SLAM / 里程计全局地图适合场景显示和世界坐标标注，但不是首选标定输入：它混合了多个时刻的观测、遮挡、动态目标与累计位姿误差。帧中存在
`calibration_pointcloud_file` 时，平台自动使用与图像精确同步的原始单帧扫描；先通过
`ego_to_world`、`calibration_lidar_to_ego` 建立单帧与主地图坐标的刚体关系，求解后再把候选转换成主地图坐标下的 `camera_from_lidar`。因此保存结果仍能直接投影融合地图和世界坐标车道线，单帧点云不会替换显示地图。

## 在平台中操作

1. 选择片段、当前帧和当前相机，点击顶部「自动标定」。选择已就绪的 Koide 或内置 CPU NID 引擎。
2. 处理范围默认是“当前帧 + 当前相机”。先验证这一组；确认有效后，可选择“全部帧 + 当前相机”或“全部帧 + 全部相机”。后台逐项报告通过、未通过、缺失数据和运行失败；关闭面板后任务继续。
3. 点击「保存通过结果」，将通过质量检查的候选保存并立即用于点云和标注投影。默认保留已经保存的参数；重新计算选项只生成候选，保存时才替换。
4. 对未通过的项目，点击「查看该帧 / 手动校准」，检查叠加图或选择至少六组三维点与图像对应点求解。
5. 点击「导出已保存参数 JSON」，下载所有相机的 K、畸变、原始外参和每帧有效 `camera_from_lidar`，并记录哪些参数经过校准。

平台在初始投影的可见范围内选择点云，按照图像空间划分训练和验证区域。
CPU 与 Koide 都只能使用训练点优化；固定的验证点用于检查 NID 改善、覆盖率和强度信息。
当前接受范围是相对初值平移小于 0.75 米、旋转小于 3 度；达到边界会拒绝。
因此这里接入的是已有外参基础上的精准修正，严重错误的初值需要先使用对应点或外参编辑。
验证 NID 是跨模态纹理分数，不是像素误差；对应点求解另外报告真实像素重投影误差。
没有图像、点云、有效强度或足够共同结构的组合不会被当作已校准。

保存的参数与源标定、图像和点云的文件身份关联。求解后输入变化会阻止保存；
新保存记录的输入变化后会停用旧修正，避免误用于其它数据。文件身份使用路径、inode、尺寸和修改时间；
直接替换源文件时应正常更新修改时间，不应保留原时间戳来绕过变更检测。
服务重启后会恢复已保存参数，保留已计算候选，并将被中断任务标记为失败。
取消任务在当前组合结束后生效，已完成候选仍可保存。Koide 默认单组最长运行 900 秒；可在后端启动环境用
`OMNILABEL_KOIDE_TIMEOUT_SECONDS` 调整。超时值同时约束 Docker worker，避免遗留容器。

任务状态在 `data/automatic_calibration_jobs/`，Koide 输入和日志在
`data/automatic_calibration_artifacts/<job>/<frame>/<camera>/`。
设置 `OMNILABEL_DATA_DIR` 时，以上路径相对于该数据根目录。

API 前缀为 `/api/clips/{clip_id}/automatic-calibration`：`GET` 查询引擎与最近任务，
`POST /jobs` 启动任务（`engine` 为 `koide` 或 `direct_nid`，`overwrite_saved` 默认为 `false`；
可用 `frame_ids`、`camera_ids` 限定范围），
`GET /jobs/{job_id}` 查询进度，`POST /jobs/{job_id}/cancel` 取消，
`POST /jobs/{job_id}/apply` 保存通过结果，`GET /export` 导出全部有效参数。

## 部署 Docker worker

在运行平台后端的 Linux 主机上准备 Docker，并让后端进程用户能够访问 Docker daemon。
仓库提供的 Dockerfile 基于
[Koide 官方 Humble 镜像](https://koide3.github.io/direct_visual_lidar_calibration/docker/)，
添加 Xvfb 和 Mesa 软件 OpenGL，以便无显示器运行。
上游的 `--background` 隐藏窗口，但仍创建 Iridescence / OpenGL 上下文，因此保留 Xvfb。

执行一次部署：

```bash
bash scripts/setup-koide.sh
export OMNILABEL_KOIDE_DOCKER_IMAGE=omnilabel/koide:humble
export OMNILABEL_KOIDE_TIMEOUT_SECONDS=900
```

将上面的环境变量加入实际启动后端的终端、systemd 或容器配置，并重启后端。
`setup-koide.sh` 会下载基础镜像和软件包，构建本地镜像，再检查 Docker 和镜像是否可用。
标定任务使用 `--pull=never`，不会在任务中临时下载镜像。
如果生产部署需要版本固定，可使用已审计镜像 digest 作为 Dockerfile 的 `BASE_IMAGE`：

```bash
docker build -f docker/koide/Dockerfile -t omnilabel/koide:humble \
  --build-arg BASE_IMAGE=koide3/direct_visual_lidar_calibration@sha256:YOUR_VERIFIED_DIGEST docker/koide
```

检查 worker：

```bash
python3 scripts/koide_worker.py --docker-image omnilabel/koide:humble --check
```

worker 为每个任务创建独立容器，只挂载该组标定数据目录，禁用容器网络。
任务超时或中断会终止子进程，并尝试清理对应容器。
后端如果本身在容器内运行，挂载目录必须在 Docker daemon 主机上有相同绝对路径，
或改用下述已配置的外部命令桥接；Docker bind mount 由 daemon 主机解析。

## 已有本机 ROS 环境

在启动后端前加载 ROS 和已编译的 Koide 工作空间。默认自动发现 `ros2` 或 `rosrun`。
也可以显式指定完整命令，配置值必须是 JSON 参数数组，使用 `shell=False` 执行。
`{data_dir}` 会替换成当前任务的预处理数据目录；没有该占位符时目录参数追加到末尾。
平台始终保证命令包含 `--background --auto_quit`。

```bash
export OMNILABEL_KOIDE_COMMAND='["xvfb-run","-a","ros2","run","direct_visual_lidar_calibration","calibrate","{data_dir}"]'
```

ROS1 示例：

```bash
export OMNILABEL_KOIDE_COMMAND='["xvfb-run","-a","rosrun","direct_visual_lidar_calibration","calibrate","{data_dir}"]'
```

配置优先级为 `OMNILABEL_KOIDE_COMMAND`、`OMNILABEL_KOIDE_DOCKER_IMAGE`、自动发现 ROS。
自定义命令只能由服务端管理员设置，不接受浏览器提交的可执行命令。
状态中的 `verified=false` 表示自定义命令只检查了启动程序，最终以实际运行和结果校验为准。

## 输入、坐标约定和保存

适配器直接生成上游所需的预处理目录，无需重新包装 rosbag：

```text
calib.input.json         原始任务输入的只读留档
calib.json               Koide 输入及输出
frame_000000.png         图像等尺寸灰度直方图均衡副本
frame_000000.ply         binary_little_endian float x/y/z/intensity
koide.log                stdout / stderr
```

上游 [`VisualLiDARData`](https://github.com/koide3/direct_visual_lidar_calibration/blob/main/src/vlcal/common/visual_lidar_data.cpp)
按 `meta.bag_names` 读取同名 PNG 和 PLY。
适配器沿用 [`preprocess.cpp`](https://github.com/koide3/direct_visual_lidar_calibration/blob/main/src/vlcal/preprocess/preprocess.cpp)
的强度排序均衡规则 `floor(256 * rank / N) / 256`，
剔除非有限返回和距 LiDAR 小于 1 米的点。
PNG 和 PLY 均写入任务目录，源图像、NPY 点云和标注不被重写。

对于 4D 融合地图导入，可在首次导入时附加同步扫描，或给现有片段安装扫描：

```bash
python scripts/import_4d_clip.py tests/4D_clip_DATA_01_20260808160513v12 \
  --data-root data \
  --calibration-pointcloud-dir tests/sq_data/pointclouds_10hz \
  --install-calibration-scans
```

安装过程按源 `lidar_timestamp_us` 查找完全同时间戳的 PCD；缺帧、错帧或 PCD 字段不符合 float32 XYZI 时会直接失败，不使用模糊近邻替代。

平台投影矩阵为 `p_camera = camera_from_lidar @ p_lidar`。
Koide 的 `results.init_T_lidar_camera` 和输出 `results.T_lidar_camera`
均为 `[tx,ty,tz,qx,qy,qz,qw]`，方向是 `camera → lidar`；
适配器在输入和输出处分别求逆，避免投影方向错误。
该方向及输出写入逻辑见
[`calibrate.cpp`](https://github.com/koide3/direct_visual_lidar_calibration/blob/main/src/calibrate.cpp)。

每个任务的 `artifacts` 保留目录、原始输入、Koide 输出和日志路径。
平台持久化通过质量检查并确认应用的 `camera_from_lidar`，投影标定记录位于
`data/projection_calibrations/<clip>/<frame>/<camera>.json`；更新前的记录保留在
`data/projection_calibration_history/`。导出参数用于复现投影；`calib.json` 可用于上游结果检查。

独立重跑某组已导出的任务：

```bash
python3 scripts/koide_worker.py \
  --docker-image omnilabel/koide:humble \
  --data-dir /absolute/path/to/task/directory
```

## 多帧共享外参和验证范围

Python 适配器接受 `frames=[KoideFrame(points, image_path), ...]`，将多组数据联合优化为同一个外参。
仅适用于相机与 LiDAR 安装关系固定、输入处于一致传感器坐标约定的场景。
单组接口为 `run_koide_calibration(points=..., image_path=..., camera=..., initial_camera_from_lidar=..., work_dir=...)`。

契约测试使用明确标记的假子进程，覆盖真实子进程调用、参数传递、PLY/PNG 协议、坐标方向、
退出失败、无结果和超时终止。测试通过表示适配协议正确，不代表已在当前机器运行过 Koide 或证明标定精度。
实际部署后需要用该机器上的真实 Koide 和有已知对应关系的数据检查投影。

更多上游命令和结果约定见
[Program details](https://koide3.github.io/direct_visual_lidar_calibration/programs/)。
