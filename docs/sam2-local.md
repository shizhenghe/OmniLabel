# 在自己的 GPU 服务器部署 SAM2

SAM2 服务运行在 GPU 服务器，OmniLabel 后端通过 HTTP 请求发送原图和提示点/框，
收到候选多边形后由你审核、应用、保存。浏览器不直接连接 GPU，也不用把权重上传到平台。
目前平台调用 SAM2 的**单图分割**接口；不会仅凭类别名称检测目标，也不自动执行跨帧视频传播。

## 1. 在本地 GPU 服务器安装

以下命令适用于 Linux NVIDIA 服务器，使用独立环境，不修改平台的 `.venv`。
需要 Git、curl、Python 3.10–3.12 的 venv 支持以及能正常运行 `nvidia-smi` 的驱动。
脚本采用官方兼容组合 **PyTorch 2.5.1 / torchvision 0.20.1 / CUDA 12.1 wheel**，
先用 SAM2.1 tiny 验证单图提示分割。显存需求随输入和自动掩码数量变化。
你提供的 10 张 RTX 2080、每卡约 12 GB，以及 `nvidia-smi` 显示 CUDA 12.2，
适合先按下面的 **单卡 tiny + FP16** 方案验证。这里的 CUDA 12.2 表示驱动支持的
CUDA 版本上限，不代表必须安装 CUDA 12.2 的 torch；CUDA 12.1 wheel 可在这个驱动上运行。
RTX 2080 属于 Turing（计算能力 7.5），不使用原生 BF16 或 Ampere Flash Attention。
官方 SAM2 会针对老架构选择可用的注意力实现，可能出现禁用 Flash Attention 的提示。
RTX 50 系列等需要更新 CUDA/PyTorch 的硬件，应按 PyTorch 官方兼容说明准备环境，
再走 [现有环境安装方式](model-worker.md#sam2)，不要强行使用这个固定版本组合。

在新目录中执行；已有仓库则更新到 `work` 分支后进入仓库：

```bash
git clone --branch work https://github.com/shizhenghe/OmniLabel.git
cd OmniLabel
nvidia-smi
bash model_workers/sam2-local.sh install
```

可指定系统 Python，例如：

```bash
SAM2_PYTHON=python3.10 bash model_workers/sam2-local.sh install
```

安装脚本会创建 `.model-envs/sam2`，安装固定的官方 SAM2 源码版本，下载
`model_weights/sam2/sam2.1_hiera_tiny.pt`，并检查 CUDA、权重和配置。
权重和环境已在 `.gitignore` 中排除。为当前适配器跳过可选的 CUDA 后处理扩展，
因此不必单独安装 nvcc；NVIDIA 驱动仍需与 CUDA 12.1 运行时兼容。

权重地址来自[官方 checkpoint 下载脚本](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/checkpoints/download_ckpts.sh)：

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

若服务器网络无法下载，可在能访问官方来源的机器下载后复制到上述位置，再重新运行安装。
脚本会保留已有的非空 checkpoint，不会删除或覆盖它。不同尺寸权重需要匹配配置，
不要把 large 权重改名当作 tiny 使用。
已有权重可先设置绝对路径 `SAM2_CHECKPOINT` 和匹配的 `SAM2_CONFIG` 再安装，
脚本会使用该文件并跳过默认 tiny 下载；自定义权重必须事先存在。

## 2. 启动与检查

```bash
bash model_workers/sam2-local.sh check
CUDA_VISIBLE_DEVICES=0 SAM2_PRECISION=auto bash model_workers/sam2-local.sh serve
```

保持这个进程运行；另开终端在 **GPU 服务器上**检查：

```bash
curl --fail http://127.0.0.1:9101/models
```

必须看到 `"id":"sam2"`、`"status":"ready"`、`"available":true`。
`/health` 的 HTTP 200 只说明服务存活，不能代替模型就绪检查。
服务启动后会实际加载模型；缺少权重、依赖、配置或 GPU 时不会返回伪造结果。
默认只监听 `127.0.0.1:9101`，一次运行一个 Worker 进程。

改用第 2 张可见显卡或其他端口时：

```bash
CUDA_VISIBLE_DEVICES=1 MODEL_WORKER_PORT=9102 bash model_workers/sam2-local.sh serve
```

脚本未指定时默认 `CUDA_VISIBLE_DEVICES=0`，先仅使用物理第 1 张卡。
上例暴露物理第 2 张卡，Worker 内仍是 `cuda:0`，GPU 编号遵循可见编号。
常驻运行可交给你已有的 systemd、
容器管理或进程管理服务，启动命令仍为以上 `serve` 命令。

`SAM2_PRECISION=auto` 会在 RTX 2080 上选择 **FP16 自动混合精度**，
在具有原生 BF16 支持的 Ampere 或更新显卡上选择 BF16；CPU 使用 FP32。
也可显式设置 `float16` 或 `float32`，不在 2080 上强制 `bfloat16`。
`/models` 就绪信息会记录实际选择的精度，SAM2 候选响应也记录 `inference_precision`。
若自动掩码生成显存不足，可使用 `SAM2_POINTS_PER_BATCH=4` 或 `8` 后重启；
默认值为 16，这个参数控制自动掩码提示点的每批数量，不改变点/框提示的类别含义。
先以真实图片验证质量和峰值显存，再决定是否换更大模型。

当前一个 Worker 一次只使用一张卡，**10 张卡的显存不会自动合并成一个显存池**。
单卡验证通过后可按不同 `CUDA_VISIBLE_DEVICES`、端口启动多个独立 Worker，
再由服务网关调度；平台目前每个模型保存一个推理端点，不会自动分配 10 张卡。

## 3. 让平台后端访问 GPU

**平台和 Worker 在同一主机/网络命名空间**：在平台「模型预标注 → SAM2 服务地址」填：

```text
http://127.0.0.1:9101/predict
```

**平台在云端，GPU 在自己的机房或家中**：云端的 `127.0.0.1` 是云端机器，
不是你的 GPU 服务器。浏览器能打开本地地址也不能证明云端后端能访问。
可选以下连接方式：

- 已有可访问的 HTTPS 网关：在 GPU 上用反向代理转发至 `127.0.0.1:9101`，
  平台填写完整地址，如 `https://gpu.example.com/predict`。代理必须同时转发同级的
  `/models`，允许较大的图像请求并使用有效证书。平台不接受 `http://192.168...`。
- 云端后端主机提供 SSH：从 **GPU 服务器**建立反向隧道，保持该进程运行：

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:49101:127.0.0.1:9101 cloud_user@cloud_host
```

将 `cloud_user@cloud_host` 替换为运行平台后端的 SSH 主机账户与地址，
不是平台网页 URL。然后在 **云端后端主机上**验证：

```bash
curl --fail http://127.0.0.1:49101/models
```

验证成功后，平台填写 `http://127.0.0.1:49101/predict`。
隧道在 SSH 中加密，远端端口只绑定 loopback。云端 SSH 服务需要允许端口转发。
如果云环境没有 SSH 入口，则使用前面的 HTTPS 网关方案；这个命令不能凭空给云环境增加 SSH。
容器中的 loopback 属于容器自身，需要把隧道与平台后端放在可互通的网络命名空间中。

点击「保存并检查连接」，状态应变为「推理就绪」。也可在平台后端进程中设置
`OMNILABEL_SAM2_URL` 后重启；若界面已保存单模型配置，界面配置优先。

如果 Worker 设置了 `MODEL_WORKER_TOKEN`，平台后端还要配置相同的
`OMNILABEL_SAM2_TOKEN` 并重启。不要把 token 填入 URL 或提交到 Git。
使用鉴权时，以上 curl 检查也要携带对应的 Bearer header。

## 4. 在平台运行一次真实分割

1. 导入含真实图片的 clip，打开「图像语义分割」。
2. 选择 SAM2，选择左侧相机，再在目标内部加正提示点；背景误选区域加负点，
   或使用「SAM2 提示框」。提示坐标自动转换为原始图片像素。
3. 填写候选类别，点击「运行模型预标注」。SAM2 只生成掩码，类别由你指定。
4. 检查候选边界，点击「应用候选标注」，最后「保存标注」。

首次验证优先使用一个正点或提示框。无提示会运行自动掩码生成，耗时及显存通常更高。
当前后端推理读取超时为 120 秒；超时应先检查 GPU 负载及输入，避免反复提交相同大任务。

## 验证范围

仓库测试验证 Worker 协议、官方 API 适配与结果转换。
本次云环境没有你的 GPU、服务器地址或权重运行环境，未在你的机器执行安装或真实推理；
以你的 `/models` 就绪状态及真实图像分割结果作为部署验收。
