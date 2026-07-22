# 当前 Reachability 五分类生成方法

当前实现位于 `dataset_builder/reachability/teacher_a.py`，输出适合作为后续 RGB 网络监督的五分类标签和一个独立 ignore 掩码：

```text
semantic_label: uint8[H, W]，取值 0--4
ignore_mask:    bool[H, W]
```

当前阶段 `ignore_mask` 默认为全 False；以后可根据相机内外参和 FOV 单独赋值，而不增加第六个互斥类别。

## 五类语义

```text
UNKNOWN                      = 0
LOCALLY_BLOCKED              = 1
CLEARANCE_BLOCKED            = 2
TRAVERSABLE_BUT_DISCONNECTED = 3
REACHABLE                    = 4
```

- `UNKNOWN`：中心栅格没有可靠 elevation/risk；unknown 不再向邻近中心膨胀。
- `LOCALLY_BLOCKED`：中心栅格已知，且 risk 达到 fatal threshold。
- `CLEARANCE_BLOCKED`：中心可通行，但圆形 footprint 覆盖 fatal 栅格。
- `TRAVERSABLE_BUT_DISCONNECTED`：配置空间局部可行，但不能从机器人根节点到达。
- `REACHABLE`：配置空间可行，并且与机器人根节点连通。

## Ego unknown 特殊处理

只有固定 `ego_mask` 内、由机器人本体遮挡产生的 unknown 才被信任：

```text
trusted_ego_unknown = ego_mask & ~known_trav
effective_known = known_trav | trusted_ego_unknown
```

默认可视化从 `build.yaml` 的 MPPI 车体矩形生成固定 ego mask：

```text
[[-0.55, -0.26], [0.55, 0.26]] metres
```

该放行发生在 obstacle clearance 和 Dijkstra 之前。代码不会在搜索完成后把 ego mask 强制写成 `REACHABLE`。

已知 fatal 栅格即使位于 ego mask 内也不会被覆盖。ego mask 外的墙后、视野边缘和未观测中心仍然是 `UNKNOWN`，但 unknown 不再作为 footprint 障碍影响相邻已知中心。

## 配置空间

圆形 footprint 半径按地图分辨率向上取整。例如分辨率为 0.04 m 时，请求半径 0.26 m 对应实际栅格半径 0.28 m。该 footprint 现在只用于膨胀 fatal 障碍。

不再腐蚀地图边界，也不再生成 outside-domain 环。整张 elevation map 直接参与标签生成：中心没有有效 reachability risk 的像素为 `UNKNOWN`，边缘有效像素不会因为人工 `border_cells` 再变成 unknown。

```text
locally_blocked = known_reachability & (reachability_risk >= fatal_threshold)

local_traversable = (
    (known_reachability & (reachability_risk < fatal_threshold)) |
    trusted_ego_unknown
)

blocked_overlap = dilate(locally_blocked, circular_footprint)
clearance_blocked = local_traversable & blocked_overlap
configuration_free = local_traversable & ~blocked_overlap
```

`reachability_risk` 的 CNN 感受野半径是 3 格。实现会仅在 CNN 推理期间，以最近的有效 elevation 填充 unknown 上下文，并在地图外添加 3 格复制边界；推理完成后，严格按原始 elevation 的中心 NaN 恢复 risk NaN。这样既不会让输入 NaN 经卷积传播而形成隐式 unknown 膨胀，也不会让 CNN 的 zero padding 在数组边缘制造虚假断崖。

reachability 分支不应用 MPPI 的 `fatal_cells_buffer` 和 `border_cells`；fatal 障碍只在 `build_teacher_a` 中按所请求的圆形半径膨胀一次。MPPI 自己使用的原始 `score/risk/trav_cost` 保持不变。

只有 root 自身属于 `configuration_free` 才执行 strict 8-neighbour Dijkstra。对角移动要求两个相邻正交栅格也为 free，禁止 diagonal corner cutting。

```text
reachable = configuration_free & isfinite(geodesic)
disconnected = configuration_free & ~reachable
```

## 最终标签优先级

```python
semantic_label = np.full(shape, UNKNOWN, dtype=np.uint8)
semantic_label[locally_blocked] = LOCALLY_BLOCKED
semantic_label[clearance_blocked] = CLEARANCE_BLOCKED
semantic_label[disconnected] = TRAVERSABLE_BUT_DISCONNECTED
semantic_label[reachable] = REACHABLE

ignore_mask = np.zeros(shape, dtype=bool)
```

## 坐标和 image_id 关联

数组轴 0 是机器人前方 x，数组轴 1 是机器人左方 y。显示时使用 `origin="lower"`，并反转页面横轴，因此：

- 上：机器人前方；
- 下：机器人后方；
- 左：机器人左侧；
- 右：机器人右侧。

可视化不会把 `image_id` 当作 elevation 行号，而是严格查找：

```python
rows = np.flatnonzero(elevation_group["image_id"] == image_id)
elevation = elevation_group["elevation"][rows[0]]
```

三路 HDR 则使用共享文件 ID：`images/<camera>/{image_id:06d}.jpeg`，不使用 `sequence_id` 作为文件名。

## 可视化

最终入口为 `dataset_builder/src/plot_topology_batch.py`：

```bash
conda run -n limo python -m dataset_builder.src.plot_topology_batch \
  --mission /path/to/LIMO_DATASET/<mission> \
  --source both \
  --count 10 \
  --radius 0.26 \
  --output-dir /path/to/output
```

每张图包含严格共享 image_id 的 hdr_left/front/right、elevation、geometric/teleop paths、五分类标签、机器人根节点和固定 ego mask 边界。每个 mission 同时写出 manifest，包含五类像素数、ignore 数量和 trusted ego unknown 数量。

## 正式 Zarr 输出

生成入口为 `dataset_builder/src/build_reachability_5labels.py`。它固定读取：

```text
<mission_timestamp>/data/elevation_map/
```

并固定写入时间戳目录直属的：

```text
<mission_timestamp>/reachability_5labels/
```

输出数组仅包括：

```text
state_label [N,H,W] uint8
ignore_mask [N,H,W] bool
risk        [N,H,W] float32
geodesic_m  [N,H,W] float32
image_id    [N]，dtype 和数值原样复制
timestamp   [N]，仅当输入存在，dtype 和数值原样复制
```

`geodesic_m` 只在 `state_label == 4` 的位置有限，其余位置为 NaN；root 无法启动 Dijkstra 时整帧均为 NaN。`geodesic_valid` 和 `root_valid` 不会被保存。每一行严格对应输入 elevation axis 0 的同一行，不选择、重排或跳过帧。

默认拒绝覆盖已有输出：

```bash
conda run -n limo python -m dataset_builder.src.build_reachability_5labels \
  --mission-dir /path/to/LIMO_DATASET/<mission_timestamp> \
  --device cuda
```

确认重新生成时显式增加：

```text
--overwrite
```

生成参数和类别定义保存在 Zarr group attrs 中，不创建额外数组字段。

一次生成数据集根目录下的全部 mission：

```bash
conda run -n limo python -m dataset_builder.src.build_reachability_5labels \
  --dataset-dir /path/to/LIMO_DATASET \
  --device cuda
```

批量模式会显示 mission 总进度和当前 mission 的帧进度；单 mission 模式显示逐帧进度。批量执行前会统一检查已有输出，未指定 `--overwrite` 时不会生成一部分后才中止。
