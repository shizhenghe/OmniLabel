# SAM2、MMDetection3D、OpenOccupancy 模型接入

平台提供三个实际推理适配器，位于 `model_workers/`。它们分别调用官方模型 API，
不使用几何结果冒充模型预测。当前云环境未安装 GPU 运行时和权重；远程 GPU 服务地址
尚未提供，因此真实权重推理还需在目标 GPU 机器上验收。协议和转换测试使用明确的测试数据。

## 连接已有服务

在「模型与预标注」面板分别填写三个完整推理地址，点击「保存并检查连接」。
配置持久化在数据目录的 `model-connections.json`，不提交到 Git。

- `sam2`：SAM2 / SAM2.1 图像提示分割。
- `mmdetection3d`：LiDAR 目标检测，或配置 `MMDET3D_TASK=segmentation` 的逐点分割。
- `openoccupancy`：OpenOccupancy / CONet 的学习型语义占用预测。

端点是完整 POST 地址，例如 `https://gpu.example/sam2/predict`。状态接口位于同级
`/models`，例如 `https://gpu.example/sam2/models`。直接部署下面的 Worker 时路径为
`/predict` 和 `/models`。地址要求 HTTPS；本机调试允许 loopback HTTP。
如果已有服务采用其他协议，需要将其封装成下面的请求/响应协议，不能只填任意模型网页地址。

也可通过服务端环境变量配置 `OMNILABEL_SAM2_URL`、`OMNILABEL_MMDET3D_URL`、
`OMNILABEL_OPENOCCUPANCY_URL`。界面保存的单模型设置优先；旧的
`OMNILABEL_INFERENCE_URL` 继续作为统一网关的兼容回退。

鉴权可分别注入 `OMNILABEL_SAM2_TOKEN`、`OMNILABEL_MMDET3D_TOKEN`、
`OMNILABEL_OPENOCCUPANCY_TOKEN`，或共享 `OMNILABEL_INFERENCE_TOKEN`。
它们通过 Bearer header 发送，不能放在 URL、提示词或 Git 文件中。
TLS 校验保持开启，不跟随重定向。受限网络还需要在环境设置允许目标 GPU 域名。

## 在远程 GPU 机器运行本仓库 Worker

三个模型使用**独立 Python 环境和进程**，尤其不能把旧版 OpenOccupancy 与现代
MMDetection3D / SAM2 的依赖安装到同一个环境。主平台环境不安装 CUDA 模型依赖。

1. 在目标机器准备与 CUDA 驱动匹配的模型环境及可信 checkpoint。
2. 激活对应环境，在本仓库运行 `bash model_workers/setup.sh <模型ID>`。
   此脚本复用已有 PyTorch，不替换 CUDA；OpenOccupancy 需要先按官方安装说明准备旧环境。
3. 配置下面的模型变量，再启动 `model_workers/serve.sh`。

### SAM2

[官方安装说明](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)要求
Python 3.10+、PyTorch 2.5.1+ 和匹配的 torchvision。安装脚本固定官方源码提交
`2b90b9f5ceec907a1c18123530e92e794ad901a4`，可用 `SAM2_SOURCE_REF` 指定经核验版本。

```bash
export OMNILABEL_WORKER_MODEL=sam2
export SAM2_CHECKPOINT=/path/to/trusted/sam2.1_hiera_tiny.pt
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_t.yaml
export OMNILABEL_MODEL_DEVICE=cuda:0
export MODEL_WORKER_PORT=9101
bash model_workers/serve.sh
```

SAM2 调用 `build_sam2`、`SAM2ImagePredictor`；没有点/框提示时使用自动 mask generator。
在左侧相机添加正点、负点或拖动提示框。`label` 是人工赋予的类别；SAM2 不根据文本类别做检测。
输出的原图 mask 转换为多边形，转换会报告孔洞和简化误差，需要检查复杂边界。

### MMDetection3D

使用现代 [MMDetection3D 1.4.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0)
与匹配的 MMEngine、MMCV 2.1.0、MMDetection 3.3.0。CUDA / PyTorch 的具体组合应按
OpenMMLab 官方兼容表准备；脚本不会忽略缺失的 CUDA 扩展。

```bash
export OMNILABEL_WORKER_MODEL=mmdetection3d
export MMDET3D_CONFIG=/path/to/model_config.py
export MMDET3D_CHECKPOINT=/path/to/trusted/checkpoint.pth
export MMDET3D_TASK=detection
export MODEL_WORKER_PORT=9102
bash model_workers/serve.sh
```

适配器调用 `init_model`、`inference_detector` / `inference_segmentor`。
检测框使用 `gravity_center` 转换为几何中心，避免底部中心错位；只接收 LiDAR 坐标框。
模型的数据域、输入特征数、类别表必须与上传点云匹配。分割输出必须保持原始点数量与顺序。

### OpenOccupancy

使用[官方安装流程](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/install.md)：
其原版环境是 PyTorch 1.10.1 / CUDA 11.3、MMCV-full 1.4.0、MMDetection 2.14.0、
MMSegmentation 0.14.1、MMDetection3D v0.17.1，并需安装 occupancy pooling。
这与现代 MMDetection3D 环境不同，必须独立部署。已核对官方源码提交
`eafd14f54b3044f7fe2734c660caab9e4b06b340` 的 `OccNet.forward_dummy` 和相机预处理接口。

```bash
export OMNILABEL_WORKER_MODEL=openoccupancy
export OPENOCCUPANCY_REPO=/path/to/OpenOccupancy
export OPENOCCUPANCY_CONFIG=/path/to/occupancy_config.py
export OPENOCCUPANCY_CHECKPOINT=/path/to/trusted/checkpoint.pth
export MODEL_WORKER_PORT=9103
bash model_workers/serve.sh
```

相机/融合配置还需要 `OPENOCCUPANCY_CAMERA_ORDER`，内容为配置相机顺序对应的平台
camera ID JSON 数组。**官方 nuScenes checkpoint 通常为 6 路相机，不能直接当作适配
7 路环视的模型**；必须提供匹配 7 路标定和训练数据域的配置、权重，或使用 LiDAR-only 配置。
适配器严格校验相机数量和顺序，不会悄悄丢掉第七路图像。相机输入要求去畸变图像及新内参。

实际调用 `build_model` / `load_checkpoint` 和 `OccNet.forward_dummy`，不伪造评估 GT。
粗/细体素 logits 会转换为带原点、各轴分辨率、类别、置信度的稀疏占用结果。
单扫描的 time-lag 为真实的 0；没有伪造额外 sweeps。低置信度和未返回空间仍为未知。

## Worker 服务与验证

启动时尝试加载已配置模型。`GET /models` 仅在模型真正加载后返回 `available:true`；
缺包、缺权重、缺配置或加载失败分别报告。可选 `MODEL_WORKER_TOKEN` 为 Worker 设置鉴权，
应与主平台对应的 token 绑定匹配。Worker 默认只监听本机，可用反向代理提供远程 HTTPS。

请求 JSON：

```json
{
  "schema_version":"1.0", "model_id":"sam2", "prompt":"road",
  "clip":{"id":"clip-001","cameras":[]},
  "frame":{"id":"frame-000","timestamp":0,"ego_to_world":[],"lidar_to_ego":[],"points":[],"images":{}},
  "annotations":{"boxes":[],"vectors":[],"image_polygons":[],"point_labels":{},"revision":0},
  "options":{"camera_id":"front","label":"road","points":[[320,200]],"point_labels":[1],"score_threshold":0.3}
}
```

示例的矩阵、相机和点数组需要替换为真实数据，结构见 [contract.md](contract.md)。
图像由主平台读取合法本地资源后嵌入 `data:image/...;base64,...`，远程服务能收到真实像素。
请求上限 64 MiB，单图 24 MiB；响应解压后上限 32 MiB。点/框提示为原图像素坐标。

响应包含 `annotations`、`model_version`，可附 `confidence`、`warnings`、`conversion`；
OpenOccupancy 另附 `occupancy`（LiDAR 坐标、`voxel_size`、`voxel_dimensions`、`voxels`）。
主平台校验类别、坐标、索引、图像范围后返回 proposal，不自动覆盖标注。
场景模式下同时检查场景与源帧 revision，人工应用后再保存；轨迹候选转换回世界坐标关键帧。

```bash
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests/test_model_workers.py backend/tests/test_model_connections.py backend/tests/test_proposal_workflow.py
```

这些测试验证真实 API 调用适配代码的协议、转换与错误处理，不代表远程 checkpoint
已经跑过。收到远程服务地址后，仍需以真实图像、点云、标定进行实际推理和质量验收。
