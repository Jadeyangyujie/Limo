**数据模块技术文档（limo/src/dataset）**

- **目的**: 阐明数据加载、预处理、和数据模块（Lightning DataModule）在训练/推理流程中的作用，以及如何从 Hugging Face 或本地 zarr 构建数据集。

**结构总览**
- 目录文件: [limo/src/dataset/limo_datset.py](limo/src/dataset/limo_datset.py), [limo/src/dataset/limo_datamodule.py](limo/src/dataset/limo_datamodule.py), [limo/src/dataset/local_datamodule.py](limo/src/dataset/local_datamodule.py)

**主要组件与职责**
- **`parse_missions_csv(missions_csv: Path) -> dict[str,str]`**: 从 missions CSV 读取 Timestamp→Split 映射（train/val/test）。
- **数据下载/移动**:
  - `pull_missions_from_hf(missions, topics, dataset_folder)`:
    - 从 Hugging Face `leggedrobotics/grand_tour_dataset` 拉取指定 missions 与 topics（使用 `huggingface_hub.snapshot_download` 的 `allow_patterns` 选择性下载）。
    - 返回下载缓存路径，然后调用 `move_dataset` 将需要的文件从缓存复制/解压到目标 `dataset_folder`。
  - `move_dataset(cache, dataset_folder, allow_patterns)`:
    - 将下载的文件（包括 `.tar`）解包到目标目录，支持通过 glob 模式过滤感兴趣的 topic 文件。
    - 包含健壮性处理：解压错误会被记录（不抛出中断整个流程）。

- **`MissionDataset` (继承 `torch.utils.data.Dataset`)**:
  - 构造参数: `dataset_type` (`tel`/`geo`)、`dataset_folder`、`mission_name`、`transform`、`with_side_cams`。
  - 行为: 打开 mission 下的 zarr group（`teleop_paths` 或 `geometric_paths`），`__len__` 返回 `z['path']` 的长度。
  - `__getitem__(idx)`:
    - 从 `images/{topic}/{image_id:06d}.jpeg` 载入前向图像（及侧视图可选），应用 `transform`。
    - 从 zarr 读取 `goal` 和 `path`，返回字典 `{ 'image_front', 'goal', 'path', ... }`。
  - 错误处理: 当图片文件缺失时会记录并抛出 `FileNotFoundError`。

- **Dataset 构造函数 (`get_dataset`, `get_mission_dataset`, `get_mission_dataset` 的组合逻辑)**:
  - `get_dataset(...)`:
    - 解析 `missions_csv`，构造 torchvision `transform`（resize+toTensor），决定需要的 `topics` 列表（是否包含 `teleop_paths` / `geometric_paths` / 侧视图）。
    - 自动调用 `pull_missions_from_hf` 将缺失的 mission 数据下载到 `dataset_folder`（对远程训练十分方便）。
    - 将各 mission 转换为 `MissionDataset`，并按 CSV 中的 split 合并为 `ConcatDataset`。
    - 返回一个包含 `train/val/test` 的字典（若不存在某 split，则对应为 None）。

- **Lightning DataModule**:
  - `LimoDataModule` ([limo/src/dataset/limo_datamodule.py](limo/src/dataset/limo_datamodule.py)):
    - 参数化接口（dataset_folder、missions_csv、dataset_type、batch_size、num_workers、pin_memory、shuffle_*、with_side_cams、image_size）。
    - `setup()` 在第一次调用时使用 `get_dataset()` 加载并填充 `data_train/data_val/data_test`。
    - `train_dataloader()/val_dataloader()/test_dataloader()` 返回标准 `DataLoader`。
  - `LocalLimoDataModule` ([limo/src/dataset/local_datamodule.py](limo/src/dataset/local_datamodule.py)):
    - 与 `LimoDataModule` 接口一致，但跳过 Hugging Face 下载，假设 `dataset_folder` 已经包含本地构建好的 `grandtour/` 数据集。
    - 在缺失 zarr 组时抛出带有建议命令的 `FileNotFoundError`，提示如何用 `dataset_builder` 补全数据。

**行为细节与注意事项**
- **可复现性**: `pull_missions_from_hf` 使用 `snapshot_download` 的 `allow_patterns` 选择文件，`get_dataset` 不显式改变随机种子。若希望完全可复现，请在上层脚本中设置 `np.random.seed` / `torch.manual_seed`。
- **性能**: `DataLoader` 的 `num_workers`、`pin_memory` 通过 DataModule 参数传入。图片 resize 与 ToTensor 在 CPU 端执行于 `transform`，可根据硬件调整 `num_workers`。
- **错误与健壮性**: `move_dataset` 对 tar 解包包含异常捕获；`MissionDataset.load_image` 则在图片缺失时抛出，训练脚本应保证数据完整或使用 `LocalLimoDataModule` 的前置构建步骤。
- **扩展性**: 若需要额外预处理（例如归一化、色彩扰动、数据增强），在 `get_dataset` 中的 `transform` 可直接替换或传入外部 `transform`。

**示例用法**
- 简单训练数据模块示例：

```python
from limo.src.dataset.limo_datamodule import LimoDataModule

dm = LimoDataModule(
    dataset_folder="/path/to/dataset",
    missions_csv="/path/to/missions.csv",
    dataset_type="geo",
    batch_size=16,
    num_workers=8,
    pin_memory=True,
)
dm.setup()
loader = dm.train_dataloader()
batch = next(iter(loader))
```

---
生成此文档基于当前实现（参见上方文件链接）。如需将函数说明扩展为行号引用或添加示例异常处理策略，我可以继续补充。
