**模型模块技术文档（limo/src/models）**

- **目的**: 说明模型体系结构、训练/验证接口、度量指标、以及模型组件如何协同将图像和目标（goal）映射为 SE(2) 路径预测。

**结构总览**
- 目录文件: [limo/src/models/limo_model.py](limo/src/models/limo_model.py), [limo/src/models/components/limo_net.py](limo/src/models/components/limo_net.py), [limo/src/models/components/limo_net_side_cams.py](limo/src/models/components/limo_net_side_cams.py)

**高层架构**
- `LimoModel` (LightningModule): 负责训练循环、指标收集、可视化上报（wandb）、以及 optimizer/scheduler 的配置。
- `LimoNet` (backbone + transformer decoder): 将图像 token（来自 DINOv2 风格 backbone）与 `goal` embedding、时间位置 embedding 结合，通过 Transformer decoder 生成序列化的 SE(2) 坐标输出（形状 `[B, path_length, 3]`）。

**`LimoModel` 关键点**
- **初始化**:
  - 接收 `net`、`loss`、`optimizer`、`scheduler` 等外部对象。
  - 保存并统计指标：`train_loss/val_loss/test_loss`（MeanMetric）、`train_mae/val_mae/test_mae`（MeanAbsoluteError）、`val_loss_best`（MinMetric）。
  - `camera_info` 可选路径，用于可视化时叠加相机投影信息。

- **前向与训练步骤**:
  - `forward(batch)` 简单地委托到 `self.net(batch)`。
  - `model_step(batch)` 用于训练/验证/test：调用 `self(batch)` 得到 `preds`，然后 `loss = self.loss(preds, path)`。

- **指标与日志**:
  - 路径误差度量: `_compute_path_metrics(preds, targets)` 返回 `ade`, `fde`, `mean_yaw_deg`, `final_yaw_deg`。
  - 在 `training_step` 中同时记录 epoch 级别（`train/loss`, `train/mae`, `train/ade` 等）和 step 级别的指标。
  - wandb 可视化: `log_images_wandb(...)` 在 validation（或训练的间隔）上传结合图像与路径的合成可视化（使用 `create_combined_visualization`）。

- **优化器与 scheduler 的配置**:
  - `configure_optimizers()` 从 `self.hparams.optimizer` / `self.hparams.scheduler` 构建 optimizer/scheduler。
  - 注意点: 为了避免 Lightning 在 configure 时因为 datamodule 未 setup 导致 `estimated_stepping_batches` 报错，代码在必要时调用 `datamodule.setup(stage='fit')` 作为补救。

**`LimoNet`（图像编码 + 解码器）**
- 位置: [limo/src/models/components/limo_net.py](limo/src/models/components/limo_net.py)
- 主要实现要点:
  - **Backbone**: 使用 DINOv2-style backbone（通过 `torch.hub.load` 从本地 `cache/torch/hub/facebookresearch_dinov2_main` 加载），并在 `pretrained=True` 时手动从 `cache/hub/checkpoints/dinov2_vits14_pretrain.pth` 加载权重。
  - **参数冻结**: backbone 的参数默认 `requires_grad=False`，但 `LayerNorm` 层被解冻以便 fine-tuning 少量参数。
  - **Patch 位置嵌入**: `row_embed` 和 `col_embed` 将 patch 网格坐标映射到同 embed 维度并加到 patch tokens 上。
  - **Goal & Time Embedding**: `goal_proj` 将 goal (3-dim) 投影到 `embed_dim`；`time_embed` 为每个时间步创建可学习向量作为 decoder queries 的基础。
  - **Transformer Decoder**: 多层 `nn.TransformerDecoder` 将 `queries`（time + goal）与 `memory`（patch tokens）交互，最后通过 `out_proj` 映射到 `se2_dim=3`。
  - **输出形状**: `(B, path_length, 3)`，每个 waypoint 为 `[x, y, yaw]`。

**`LimoNet`（多视图扩展）**
- 位置: [limo/src/models/components/limo_net_side_cams.py](limo/src/models/components/limo_net_side_cams.py)
- 额外点:
  - 支持三视图（front,left,right），对每个视角做单独编码并通过 `view_embed` 注入视角信息，最后在 token 级别拼接三视角 tokens 作为 memory。
  - 其余 decoder、goal 投影、时间嵌入逻辑与单视图类一致。

**训练/推理流程（端到端）**
- 1) DataModule 提供 batch 包含 `image_front`（可选 side cams）、`goal`、`path`。
- 2) `LimoModel.training_step` 调用 `model_step`，通过 `loss(preds, path)` 计算损失并更新指标。
- 3) `configure_optimizers` 返回 optimizer/scheduler 以供 Trainer 使用；训练过程中，`log_images_wandb` 负责周期性将预测与 GT 上传到 wandb，便于可视化诊断。

**设计与工程要点**
- **可重复性**: `LimoNet.setup()` 尝试在本地 hub 缓存中加载 backbone，避免在线下载，这使离线训练/受限网络环境下能稳定运行。
- **可扩展性**: 模块化拆分（backbone、pos embedding、goal/time embedding、decoder）使得替换 backbone 或扩展 decoder 层数简单。
- **数值稳定性**: 在角度误差计算上使用 `_wrap_angle` 保证角差在 [-pi, pi] 区间，防止角度跳变导致异常指标。

**示例用法**
- 构造模型并进行一次前向：

```python
from limo.src.models.components.limo_net import LimoNet
from limo.src.models.limo_model import LimoModel
import torch

net = LimoNet()
loss_fn = torch.nn.MSELoss()
# optimizer/scheduler 由外部配置（示例略）

model = LimoModel(net=net, loss=loss_fn, optimizer=lambda params: torch.optim.Adam(params, lr=1e-4), scheduler=None, compile=False)

batch = {
    'image_front': torch.randn(2,3,308,476),
    'goal': torch.randn(2,3),
    'path': torch.randn(2,50,3),
}
out = model.forward(batch)
```

---
该文档基于当前源码实现。如果你希望我：
- 把关键函数/类精确到行号并在文档中添加链接；或
- 生成一个更详尽的 README 示例（包含训练命令与示例配置），
我可以继续补充。
