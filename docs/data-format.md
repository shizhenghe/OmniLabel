# Clip 数据接入格式

平台以 clip 为单位加载同步传感器数据。内置样例是 **7 路相机、12 帧的合成数据**；真实 clip 请提供原始图像、LiDAR 点云、标定和每帧 ego 位姿。当前交互式格式允许 1–32 路相机，正常生产数据应为配置的完整七路环视；缺失图像不应伪造或用重复图片替代。

## 坐标和时间约定

- 长度单位为米，角度为弧度，时间戳为秒。帧按时间严格递增，`fps` 用于回放。七路同步图像放在同一 frame 的 `images` 中；目前不做硬件时间同步或逐点 deskew。
- LiDAR/ego 坐标均为右手系：x 向前、y 向左、z 向上。每帧点云、3D 框和地图矢量使用该帧 **LiDAR 坐标**。
- 相机使用光学坐标：x 向右、y 向下、z 向前。`camera_from_ego` 把 ego 点变换为相机光学点，不是相机位姿矩阵；若已有 camera→ego 位姿，应先取逆。
- 所有 4×4 矩阵按行存储，用列向量左乘：`p_world = ego_to_world @ lidar_to_ego @ p_lidar`。
- 相机投影使用 `camera_from_ego @ lidar_to_ego @ p_lidar`，归一化后施加 Brown–Conrady 畸变 `[k1,k2,p1,p2,k3]`，再乘 3×3 内参 `K`。只有相机前方且在图像范围内的点才返回。投影不等于深度遮挡推理。
- 若源数据提供逐帧、逐相机的标注投影微调，可在 frame 中设置可选的
  `annotation_camera_from_lidar: {camera_id: 4×4矩阵}`。该矩阵直接把当前标注坐标变到相机光学坐标，
  优先于上述静态链路，但只用于框和地图矢量的图像叠加；原始点云投影仍使用未经微调的标定。
- 矩阵必须是有限数值，旋转正交且行列式 +1，末行为 `[0,0,0,1]`。图片实际像素大小必须与 `width`、`height` 一致。当前不支持鱼眼模型；鱼眼环视应先以一致的新内参完成去畸变并导入对应图像。
- 3D 框 `center` 是几何中心，`size=[长,宽,高]` 均为正数，`yaw` 绕 LiDAR +z；每帧同一 `track_id` 只能出现一次。

## 文件组织

推荐导入 ZIP，根目录必须有 `manifest.json`，图像和点云为 ZIP 内相对路径：

```text
drive-001.zip
├── manifest.json
├── images/
│   ├── frame-000-front.jpg
│   ├── frame-000-front_left.jpg
│   └── ...其余五路及其他帧
└── lidar/
    ├── frame-000.bin
    └── ...
```

支持的点云形式：

- JSON 内联 `points: [[x,y,z,intensity], ...]`；XYZ 三列也可，强度补 0。
- 帧中使用 `points_path: "lidar/frame-000.bin"`：小端 float32、连续 XYZI 四列、无文件头。
- `.npy`：格式版本 1/2 的真实数值 Nx3 或 Nx4 数组，禁止 pickle/object 数组。
- `.pcd`：仅支持 `DATA ascii`，必须有 x/y/z 字段，可选 intensity；不支持多 count 字段或压缩/二进制 PCD。

同一帧使用 `points` 或 `points_path` 之一。点的顺序定义语义标注的索引，导入后不要在保留原标注的情况下重新排序。
二进制点云导入后保存在独立的 NPY 文件中；清单使用服务器内部的 `pointcloud_file` 和
完整 `point_count`，不会将全部扫描膨胀为清单 JSON，也不会按点数裁剪。
逐帧读取使用内存映射；坐标、反射强度和原始点顺序保留。

## 最小可用 manifest 示例

下面是用于验证坐标和上传的 **单相机小夹具**，需一张 1280×720 的 `images/front.jpg`。真实七路 clip 应将 `cameras` 扩展为各自真实内外参的七项，并为每帧填齐七个 camera ID。可从平台导出内置合成 clip 查看完整的七路、多帧结构。

```json
{
  "schema_version": "1.0",
  "clip": {
    "id": "drive-001",
    "name": "真实采集片段 001",
    "description": "请填入采集地点、标定版本、时间同步方式",
    "fps": 10,
    "source": "import",
    "cameras": [{
      "id": "front",
      "name": "前视",
      "width": 1280,
      "height": 720,
      "intrinsics": [[700,0,640],[0,700,360],[0,0,1]],
      "camera_from_ego": [[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],
      "distortion": [0,0,0,0,0]
    }],
    "frames": [{
      "id": "frame-000",
      "timestamp": 0.0,
      "ego_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "lidar_to_ego": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "points": [[10,0,0,0.8],[10,1,0,0.7],[12,-1,-1,0.4]],
      "images": {"front": "images/front.jpg"}
    }]
  },
  "annotations": {
    "frame-000": {
      "revision": 0,
      "boxes": [{
        "id": "box-001", "track_id": "vehicle-001", "label": "vehicle",
        "center": [10,0,0], "size": [4.5,1.9,1.6], "yaw": 0
      }],
      "vectors": [{
        "id": "lane-001", "label": "lane", "closed": false,
        "points": [[0,3.5,-1.7],[10,3.5,-1.7],[20,3.5,-1.7]]
      }],
      "image_polygons": [{
        "id": "mask-001", "camera_id": "front", "label": "vehicle",
        "points": [[580,310],[700,310],[700,420],[580,420]]
      }],
      "point_labels": {"0":"vehicle","1":"vehicle","2":"road"}
    }
  }
}
```

示例数字仅用于格式演示，不能用于真实传感器标定。`annotations` 可省略。也支持直接以 clip 字段为顶层、另加 `annotations`，但推荐使用上述带 `schema_version` 的封装。

ID 使用字母/数字开头，后续仅字母、数字、`_`、`-`、`.`，长度不超过 96。camera/frame/标注 ID 在其作用域内唯一。语义 label 为非空字符串；建议在整套数据中固定类别表。`point_labels` 的键是当前帧点索引的十进制字符串。图像多边形顶点位于**原图像素坐标**，不随界面缩放改变。

## 导入、保存与导出

- 浏览器“导入”上传 `.zip` 或 `.json`；后端 `POST /api/clips/import` 使用 multipart 字段 `file`。
- ZIP 不接受绝对路径、`..` 路径、符号链接、加密项和重复项。图片支持 JPEG、PNG、WebP；不主动下载清单中的外部 URL。
- JSON 单文件可内联点云，并使用 base64 data-image 图像，或引用**同一服务器已存在**的 `/api/assets/...` 资源。普通本地相对图片路径必须随 ZIP 上传。
- 已有 clip ID 不会被静默覆盖。复制导入时应先修改 `clip.id`。标注保存携带 revision，旧版本写入会返回 409 冲突，需重新加载并协调修改。
- JSON 导出 `GET /api/clips/{id}/export` 返回 `{schema_version:"1.1", clip, annotations, scene_annotations, scene_registration}`，包含已保存的源帧标注、世界坐标场景及配准变换，不包含未保存编辑。JSON 保留服务器 `/api/assets/...` 引用，可在原服务器重新导入。
- **可迁移 ZIP**：`GET /api/clips/{id}/export.zip` 同时打包 `manifest.json`、点云及相对路径的图像资源，可以导入另一台 OmniLabel 实例。导入已存在的 clip ID 前应更改 manifest 中的 ID。
- 输出 3D 框、矢量线/多边形、图像多边形和逐点 label。语义分割可单独导出 PNG，见下节。尚未宣称兼容 COCO、nuScenes、OpenLABEL 或特定 OCC 训练集；这些格式应增加显式转换器和测试。

源点云没有单帧或整段点数上限。上传与 ZIP 的字节/条目配额默认均为 0（不限制），
部署方可通过 `OMNILABEL_MAX_UPLOAD_BYTES`、`OMNILABEL_MAX_EXPANDED_BYTES`、
`OMNILABEL_MAX_ARCHIVE_MEMBERS` 显式设置；这些配额拒绝超额输入，不会静默丢点。
上传文件通过磁盘暂存，ZIP 中的点文件作为独立 NPY 保存，JSON 与 ZIP 导出均保留完整点云。
可迁移 ZIP 会包含实际点文件，不再依赖原服务器的内部文件路径。

`GET /api/clips/{id}` 返回帧元数据及完整 `point_count`，帧 `points` 为空，避免一次传输整段扫描。
`GET /api/clips/{id}/frames/{frame_id}/points` 默认返回该帧所有点；可选 `offset`、`count`
进行连续分块读取，响应包含 `point_count`、`offset`、`returned_point_count`。
分块只是传输方式，逐点标注索引始终使用原始点编号，不按块重新编号。
SQ 打包脚本默认不按点数或文件字节拆分场景，只有显式帧数/字节分组参数会改变分包。

当前清单仍最多包含 2000 帧。每张图片不超过 24 MiB、4000 万像素；相机宽高各不超过 16384。
无点数配额不代表无限内存：重建输出、浏览器完整显示和模型推理仍受机器内存、显存及磁盘容量约束。

## 地图和 OCC 输出解释

4D 工作区将整个 clip 作为统一世界场景。默认用输入位姿重建，也可执行真实的逐帧 ICP 配准。地图、车道线和矢量标注持续保留，动态目标按世界坐标关键帧插值；时间轴只切换活动时刻和相机。重建后以 `scene_registration.world_from_lidar` 作为实际坐标变换，导入导出会保留它。旧版逐帧标注仍可在单帧模式使用；两套标注不会静默覆盖。细节和 ICP 质量限制见 [4d-scene.md](4d-scene.md)。

OCC 是当前帧 LiDAR 坐标下、指定 voxel size 的稀疏观测转换。`occupied` 表示有扫描终点证据，`free` 表示射线经过，未返回的空间为 `unknown`；占据证据优先于自由证据。语义来自已保存点标注。没有稠密补全、相机可见性 mask 或 benchmark 专用裁剪范围，不能直接作为所有 OCC 模型的完整真值格式。

OpenOccupancy 模型预测单独标记为学习型候选结果，包含其体素网格原点、各轴分辨率与置信度，经人工应用后查看。不会将上述 CPU 射线结果标成大模型预测。模型部署和 7 路相机适配见 [model-worker.md](model-worker.md)。

## 语义分割 mask PNG

对已保存的图像多边形，可调用：

- `GET /api/clips/{id}/frames/{frame_id}/masks/{camera_id}.zip`（推荐，界面默认）：同一快照的 `mask.png` 与 `labels.json`，避免并发编辑造成掩码与类别映射不一致。
- `GET /api/clips/{id}/frames/{frame_id}/masks/{camera_id}.png`：原图尺寸的 16 位单通道 PNG，每个像素值是类别 ID，0 为未标注。多边形按保存顺序栅格化，后绘制者覆盖前者；没有多边形时输出全 0。
- 同路径 `.json`：类别 ID→名称的 `labels`、宽高、标注 revision 等元数据。下载 PNG 时应同时保存元数据。

内置类别 ID 固定为 road=1、sidewalk=2、vehicle=3、pedestrian=4、vegetation=5、building=6、lane=7、sky=8、terrain=9、pole=10、traffic-sign=11、cyclist=12；自定义类别在这些 ID 之后按 clip 内类别名称排序分配。新增自定义类别后映射可能变化，因此训练数据必须绑定导出时的类别表和 revision。此转换不运行视觉模型，不推断多边形以外区域的语义。
