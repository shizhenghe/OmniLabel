# 接入真实模型推理 Worker

OmniLabel 的 HTTP 适配器把一帧图像、点云、标定和现有标注发送给部署者配置的
推理端点，并把经过校验的结果作为**待人工审核的 proposal**返回。它不会自动覆盖
标注，也没有内置模型权重、GPU 推理或伪造的模型输出。

可以在独立 GPU 服务中部署 SAM 2 / Grounding DINO 做图像提示分割，或部署适合
自己 LiDAR 数据集的 MMDetection3D 模型。不同模型需要各自的数据适配、类别映射、
权重许可与版本管理。SAM 2 本身不是 3D 检测或全场景真值生成模型。

在启动后端的进程环境中设置：

```bash
export OMNILABEL_INFERENCE_URL='https://your-gpu-service.example/v1/predict'
# 如需鉴权，在部署平台的安全变量设置中注入 OMNILABEL_INFERENCE_TOKEN。
```

URL 是完整 POST 端点，不会自动拼接 `/predict`。远程服务要求 HTTPS；本机开发
可以使用 `http://127.0.0.1:9000/predict`、`http://[::1]:9000/predict` 或 localhost。
URL 中不能包含用户名、密码或片段。可选 token 通过 `Authorization: Bearer ...`
发送；不要把 token 写入仓库、提示词、URL 或日志。TLS 证书校验始终开启，不跟随
重定向。网络受限的云环境还需要允许访问你配置的 GPU 服务域名。

`worker_configured()` 仅表示 URL 配置有效，不表示网络可达、GPU 就绪或模型已加载。
请求失败时返回经过脱敏的错误，原有标注保持不变。

## 请求协议

Worker 接收 `Content-Type: application/json`：

```json
{
  "schema_version": "1.0",
  "model_id": "sam2",
  "prompt": "road, vehicle, pedestrian",
  "clip": {
    "id": "clip-001",
    "cameras": ["与 docs/contract.md 中 Camera 相同的对象"]
  },
  "frame": {
    "id": "frame-0001",
    "timestamp": 0.0,
    "ego_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "lidar_to_ego": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "points": [[10.0,0.0,0.5,0.8]],
    "images": {"front": "data:image/png;base64,..."}
  },
  "annotations": {
    "boxes": [], "vectors": [], "image_polygons": [],
    "point_labels": {}, "revision": 3
  }
}
```

示例中的 `cameras` 占位字符串应替换成实际 Camera 对象，包括像素尺寸、内参、
`camera_from_ego` 和 Brown–Conrady 畸变系数。参考 [数据坐标约定](contract.md)。
后端把已导入的本地图像嵌入为 base64 data URI，因此远程 Worker 不需要读取
OmniLabel 的本地磁盘或访问相对 `/api/assets/...` URL。适配器不会下载任意图像 URL。
图像应在导入时通过图片格式与像素尺寸校验。请求总大小上限 64 MiB，单图上限
24 MiB；过大的数据应缩放图像并同步缩放内参，或分批推理。

## 响应协议

Worker 完成实际推理和坐标转换后返回 HTTP 200 JSON：

```json
{
  "model_version": "model-name/checkpoint-sha256:...",
  "annotations": {
    "boxes": [],
    "vectors": [],
    "image_polygons": [
      {
        "id": "mask-001", "camera_id": "front", "label": "road",
        "points": [[10,20],[80,20],[80,70],[10,70]]
      }
    ],
    "point_labels": {"0": "vehicle"}
  }
}
```

这里的多边形仅展示协议格式，不是模型执行示例。Worker 需要把像素 mask 转换成
原图像素坐标多边形；复杂 mask 的孔洞与多个连通域需按平台的多边形数据能力处理。
3D box / vector 必须转换到当前帧 LiDAR 坐标系，点类别的 key 是原始点数组索引，
不得使用网络采样后的索引。返回完整的候选标注集合；要保留已有内容，Worker 应合并
请求中的现有标注。模型响应不应包含自定义未知字段。

适配器验证尺寸、有限数值、点索引范围、类别非空、相机引用、图像范围、对象和轨迹
ID 唯一性。Worker 的 revision 被当前应用 revision 替代；返回值为：

```json
{
  "proposal": {"boxes":[],"vectors":[],"image_polygons":[],"point_labels":{},"revision":3},
  "model_id": "sam2",
  "model_version": "model-name/checkpoint-sha256:...",
  "review_required": true,
  "source": "external-worker"
}
```

响应解压后上限 32 MiB；连接超时 10 秒，读取超时 120 秒。非 200 响应、重定向、
超时、超限及格式错误均失败，不会保存任何标注。人工确认 proposal 后仍通过现有
标注保存接口和 revision 冲突检查持久化。

## 开发验证

```bash
.venv/bin/python -m pytest backend/tests/test_inference.py -q
```

测试使用 HTTPX MockTransport 检查协议、图像传输、鉴权、proposal 校验、错误脱敏、
重定向拒绝及容量限制。它们验证适配器，不代表已经运行 SAM 2 等模型。真实部署还需
使用真实 Worker、模型权重和代表性 clip 验证标签质量与推理性能。
