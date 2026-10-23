# 开源参考与自动真值路线

以下链接的官方 README 和 LICENSE 已在 2026-10-22 通过 HTTPS 读取核对。本项目参考其任务组织和交互思路，未复制这些项目的实现、模型权重或数据集。许可证信息是集成前的初步核对，模型、数据集、第三方组件仍需分别核对所选版本的条款。

| 官方项目 | 可参考的能力 | 核对到的仓库许可证 | OmniLabel 的使用方式 |
| --- | --- | --- | --- |
| [CVAT](https://github.com/cvat-ai/cvat) | 任务与数据集管理、图像/视频标注、模型辅助标注 | MIT 核心；部分组件有其他条款 | 参考任务切换、人工编辑与标注持久化流程 |
| [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS) | 点云 3D 框、多视图、跟踪 ID、半自动编辑 | GPLv3 | 参考多传感器联动及 3D 框工作流；不复制 GPL 实现 |
| [LATTE](https://github.com/bernwang/latte) | LiDAR 与图像融合、单击辅助标注、跟踪 | Apache-2.0 | 参考传感器融合和减少重复操作的方法 |
| [SAM 2](https://github.com/facebookresearch/sam2) | 图像/视频提示分割、跨帧 mask 传播 | Apache-2.0；第三方组件另行核对 | 已实现官方 predictor / automatic generator 适配器，部署远程权重后使用正负点和框提示 |
| [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) | 文本 grounding、检测后分割、视频对象跟踪 | Apache-2.0 仓库；上游模型/API另行核对 | 建议自然语言类别提示生成图像候选，不直接当作真值 |
| [SAM 3](https://github.com/facebookresearch/sam3) | 开放词汇的图像/视频检测、分割、跟踪 | 自定义 **SAM License**，并非 Apache-2.0 | 可选候选生成模型；README 已包含 2026-03-27 的 SAM 3.1 更新；先评估条款和算力再部署 |
| [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) | LiDAR/多模态 3D 检测、3D 语义分割 | Apache-2.0 | 已实现 init_model / inference_detector / segmentor 适配，需匹配模型配置和权重 |
| [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy) | 环视语义 OCC、nuScenes 稠密占据标注、AAP 过程 | Apache-2.0 代码；数据条款单独核对 | 已实现 OccNet.forward_dummy 实际预测适配；不宣称复现 AAP 稠密真值流水线 |

## 当前实现的边界

- 提供人工 3D 框、同 ID 跨帧处理、车道/地图矢量、图像多边形分割、逐点语义、标定投影、按位姿融合和稀疏 OCC 转换的基础闭环。
- CPU 几何任务与合成演示可以直接运行；演示的图像、点云和语义来自同一程序化场景，全部明确标为合成数据。
- SAM2、MMDetection3D、OpenOccupancy 的实际推理适配器已提供，模型面板显示远程依赖/权重加载状态；当前地址尚未提供，不能声称实际 GPU 推理已经通过。部署和协议见 [模型服务接入](model-worker.md)。
- 地图支持已知位姿融合及实际局部 ICP 配准，整个 clip 共用世界坐标场景。没有回环优化、逐点去畸变或全局 SLAM 精度保证。静态矢量持续保留，动态框按世界坐标时间关键帧插值。
- 时间关键帧插值不等于学习型运动跟踪。模型产生的候选仍需人工修正；图像多边形可确定性栅格化为带类别表的 16 位 PNG mask，候选经人工应用和保存后才进入导出结果。
- 稀疏 OCC 的观测终点是 occupied，观测射线覆盖的体素是 free，未观测空间保持 unknown。它不是由稀疏扫描凭空补全出的完整稠密真值；是否满足训练集标准必须另行质检。

## 推荐的真实数据预标注流程

1. 导入原始 clip、时间戳、七路相机内参/畸变参数、LiDAR→ego 外参、ego→world 位姿；检查相机图像与点云同步。对运动场景补充逐点时间与去畸变流程。
2. 通过独立推理服务运行 3D 检测和 3D 分割；记录模型版本、输入版本、置信度和候选来源。检测器需要适配本车点云通道、坐标系和类别体系。
3. 对图像运行 SAM 2 / Grounded SAM 2 / 经条款审核的 SAM 3，产出带原图尺寸、类别、置信度及帧索引的 mask 候选。mask 经人工确认后才进入真值版本。
4. 利用标定及深度遮挡检验把图像语义转移到可见 LiDAR 点，结合直接 3D 语义与多帧证据投票；标记冲突和低置信度区域供人工复核。单靠相机内参投影不解决遮挡。
5. 使用 ego 位姿融合静态几何，并用逐帧人工确认的动态框/实例轨迹剔除动态点；单独维护动态实例的时间轨迹。车道/地图矢量需固定地图坐标原点、拓扑关系与版本。
6. 对静态地图和动态实例分别进行 OCC 体素化、可见性 ray casting、语义投票与质检。采用模型补全时必须保留 observed/inferred 标记，不能把模型猜测当作观测真值。
7. 在平台中完成逐帧、跨相机、BEV 和时间一致性复核；导出前校验轨迹 ID、几何合法性、类别覆盖率、遮挡/截断属性及 OCC 的 unknown 约定。

用于多人生产环境时，还应增加用户权限与审计、数据库迁移、对象存储、分布式任务队列、数据集版本/锁、任务分配与复核发布；当前工作站原型不包含这些部署能力。
