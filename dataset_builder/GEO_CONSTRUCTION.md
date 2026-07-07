**Geo 构造实现说明**

- **目的**: 说明代码中如何从栅格高度图（elevation）构造地理（geodesic）距离场（GDF），及其在 MPPI 规划中的使用。

**整体流程（高层）**
- 输入: 机器人中心化栅格高度图（elevation），配置信息（阈值、缓冲区等），起点与目标位姿。
- 步骤: 高程 -> 可行性/ traversability 估计 -> 二值障碍掩码 -> 膨胀/缓冲 -> 计算广义地质距离场（GDF） -> 将 GDF 转换为米并返回给规划器。
- 输出: 与地图分辨率同单位的 GDF（不可达处为 NaN 或指定值），供代价函数使用。

**关键文件与函数对应**
- 算法入口与调用关系: [dataset_builder/mppi_planner/mppi_planner.py](dataset_builder/mppi_planner/mppi_planner.py)
  - `MPPIPlanner.plan(...)` 调用 `MPPIObjective.set_observation(...)`。
  - `MPPIObjective.set_observation(...)` 在内部会触发 `MPPIObjective._compute_gdf()`，最终得到 GDF 并存于 `self._gdf`。

- 可行性（traversability）估计: [dataset_builder/mppi_planner/traversability_filter.py](dataset_builder/mppi_planner/traversability_filter.py)
  - `get_filter_torch()` 加载预训练权重（文件: weights.npz）并返回 `TraversabilityFilter` 实例。
  - `TraversabilityFilter.forward(elevation)` 接受高度图（Tensor），经过多尺度卷积后输出 [0,1] 风险/可行性估计（代码返回 `torch.exp(-out)`）。

- 二值障碍掩码构建与缓冲: [dataset_builder/mppi_planner/mppi_planner.py](dataset_builder/mppi_planner/mppi_planner.py) 中的 `MPPIObjective._compute_gdf()`
  - 首先计算 `self._trav`（在 `set_map` 时通过 `_compute_traversability()` 产生）。
  - 将 `trav` 与 `cfg.fatal_value` 比较，得到 `gdf_mask = (self._trav >= cfg.fatal_value).float()`（致命不可行的格点被视作障碍）。
  - 将 `NaN` 的位置（地图边界或未知）设为 0，避免被视作障碍。
  - 使用 `max_pool`（基于 `torch.nn.functional.max_pool2d` 的简单膨胀）与配置 `cfg.gdf_obstacle_buffer` 对障碍做缓冲：
    - `gdf_mask = (max_pool(gdf_mask, cfg.gdf_obstacle_buffer) > 0).float().unsqueeze(0)`。
  - 目标索引通过 `world_to_map_idx(...)` 转换为栅格坐标，并用 `clip_on_ray(...)` 将超出地图的目标投影到图边界射线上，保证目标在地图内用于 GDF 计算。

- GDF 计算封装: [dataset_builder/mppi_planner/fast_geodis_wrapper.py](dataset_builder/mppi_planner/fast_geodis_wrapper.py)
  - 函数: `fast_gdf_wrapper(image, goal_row, goal_col, obstacle_gdf_value=float('nan'), iterations=2)`。
  - 输入: `image` 形状应是 `(1, 1, H, W)` 且值为 0 表示自由、>0 表示障碍（函数内部会按 H*W 缩放）。
  - 关键实现步骤:
    - 将 `image` 乘以 `mult = H * W` 用作“不可达”阈值哨兵（后面所有 >mult 的 GDF 值视为不可达）。
    - 构建 `mask`，全 1 并把目标索引位置设为 0（这是 FastGeodis 的接口约定：0 表示种子点/目标）。
    - 将地图边界行列置 0（ `image[..., 0, :] = 0` 等），避免边界被视作障碍阻断路径。
    - 调用 `FastGeodis.generalised_geodesic2d(image, mask, v, lamb, iterations)`，其中 `v = 1e10`，`lamb = 0.5`（混合度），并将返回的结果乘以 2。
    - 若以混合度（lamb=0.5）计算得到的 GDF 在地图中心点处大于 `mult`（表示对 blended metric 中心不可达），则降级为纯 geodesic（`lamb=0.0`）重新计算，以便穿越狭窄通道。
    - 将 GDF 中大于 `mult` 的值设为 `obstacle_gdf_value`（默认 NaN），作为不可达标记并返回。

  - 说明: `FastGeodis.generalised_geodesic2d` 提供了一种将像素强度（这里为障碍）与欧几里得距离混合的度量（广义地质距离）。通过调整 `lamb` 可以在靠近障碍的代价与距离之间做 trade-off。

**GDF 的单位与后处理**
- `fast_gdf_wrapper` 返回的 GDF 是以“栅格单位”为基准（函数内部未乘分辨率），而 `MPPIObjective._compute_gdf()` 在返回时做了 `* self._gm.resolution`，将 GDF 转换为米（或与 `GridMap2D.resolution` 相同的物理单位）。

**GDF 在代价函数中的使用**
- 文件: [dataset_builder/mppi_planner/mppi_planner.py](dataset_builder/mppi_planner/mppi_planner.py)
  - `MPPIObjective.get_position_distance_to_goal(states_xy)`:
    - 计算每个状态的欧氏 L2 距离 `l2` 到目标。
    - 在栅格上查表取出对应位置的 GDF（缺失值用 GDF 有穷值的平均值填充）。
    - 返回 `torch.max(l2, gdf)` —— 即对每个状态采用欧氏距离与 GDF 的较大者作为“到目标的距离”度量（这样可以在障碍或不可达区域强制增大距离感知）。
  - 在 `MPPIObjective.states_cost(...)` 中，最终的位置代价基于该距离度量计算（`pos_cost`），并与控制代价、朝向代价、以及 traversability 代价值相加形成最终代价。

**与地图、坐标转换相关的辅助函数**
- `world_to_map_idx(xy, gm)` (mppi_planner.py): 将世界坐标（x,y）转换为地图索引 i,j，使用 `gm.origin_xy` 与 `gm.resolution`。
- `valid_mask(ij, gm)`: 检查索引是否落在地图边界内。
- `clip_on_ray(bounds_hw, goal_ij)`: 若目标索引超出地图边界，则沿从地图中心到目标的射线将其裁剪到边界上（保持方向，但使其在地图内部）。这用于确保 GDF 的种子点在地图内。

**参数与配置要点**
- 代码使用到的配置项主要在 `cfg`（mppi 段），关键字段及含义包括：
  - `fatal_value`: 超过该 traversability 值被视为障碍/致命，不可通过（用于构建 GDF 掩码）。
  - `gdf_obstacle_buffer`: 对障碍做膨胀的半径（以栅格计），通过 `max_pool` 实现缓冲。
  - `gdf` 计算中的 `iterations` 参数在 `fast_gdf_wrapper` 可调（默认 2），影响迭代细化程度。

**实现细节与设计考量**
- 使用预训练的 traversability CNN 将高程转为风险评分，避免直接基于坡度/断层硬编码阈值。
- 将“致命 traversability”视为障碍并进行膨胀，从而在 GDF 中反映机器人应该避免的区域。
- `FastGeodis.generalised_geodesic2d` 采用混合 metric（通过 `lamb`），使路径在远离障碍时更像欧氏距离，但在接近障碍时能更加保守；通过 fallback 到纯 geodesic（lamb=0）确保在某些狭窄处仍能找到可达通道。
- 将不可达点标记为 `NaN`（或由 `obstacle_gdf_value` 指定）并在后续代价计算中用均值或其他策略填充，避免完全丢弃路径评估的鲁棒性。

**参考源码位置**
- 主要实现: [dataset_builder/mppi_planner/fast_geodis_wrapper.py](dataset_builder/mppi_planner/fast_geodis_wrapper.py)
- 掩码与 GDF 构建: [dataset_builder/mppi_planner/mppi_planner.py](dataset_builder/mppi_planner/mppi_planner.py#L1-L200)
- traversability CNN 与权重: [dataset_builder/mppi_planner/traversability_filter.py](dataset_builder/mppi_planner/traversability_filter.py)
- 坐标变换与工具函数: [dataset_builder/helpers/transform_helpers.py](dataset_builder/helpers/transform_helpers.py)

**可能的扩展或注意事项**
- 若想改变不可达的表示方式，可在 `_compute_gdf()` 中将 `obstacle_gdf_value` 设为一个大数而非 NaN，并在 `get_position_distance_to_goal` 中调整填充策略。
- `lamb`、`v`、和 `iterations` 是影响 GDF 形态的超参数；可在 `fast_geodis_wrapper` 外暴露为配置项以便调整。

---
该文档旨在成为工程级参考；如需我将文中引用增加更精确的行号链接或补充配置项对应的默认值（来自配置文件），我可以继续补充。

**目标采样与可行路径生成（详细）**
- **采样位置**: 每个栅格帧在构建 D_geo 时会基于配置随机生成若干目标：
  - 采样函数: [dataset_builder/src/build_paths.py](dataset_builder/src/build_paths.py) 中的 `_sample_geo_goal(rng, cfg)`，它使用多元正态分布，均值为 `[cfg.goal_x_mean, cfg.goal_y_mean, cfg.goal_yaw_mean]`，协方差为对角矩阵 `diag([cfg.goal_x_std**2, cfg.goal_y_std**2, cfg.goal_yaw_std**2])`。
  - 每帧采样次数由 `cfg.paths_per_image` 控制，采样器由 `np.random.Generator` (`rng`) 创建并通过 `cfg.seed` 固定以保持可复现性。
  - 在采样前，代码会检查高程图有效性：跳过 NaN 太多的帧（`np.isnan(elev_np).mean() > cfg.max_nan_frac`）或 NaN 太靠近中心的帧（`_check_min_nan_dist`），以避免目标不可达或地图数据不足。

- **路径生成入口**: 对于每个采样目标，调用 `planner.plan(gm, start, goal)` 生成一条路径。
  - `planner` 是 `MPPIPlanner`（参见 [dataset_builder/mppi_planner/mppi_planner.py](dataset_builder/mppi_planner/mppi_planner.py)），`plan(...)` 的返回为 `(H, 3)` 的 SE(2) 路径点序列（单位同 `GridMap2D` 的分辨率，最终为米）。

- **MPPI 优化器工作流程**: 路径不是简单跟踪可视连通性，而是通过 MPPI 随机优化得到的最优控制序列并积分得到路径：
  - 初始: `MPPIOptimizer._allocate_from_cfg()` 根据 cfg 分配 `mean`（初始化为 0 的动作序列，形状 `(H, 3)`），以及动作上下界 `lower_bound`/`upper_bound`、噪声方差 `var`。
  - 迭代采样和评估（`MPPIOptimizer.optimize()`）:
    - 每次迭代从截断正态分布生成噪声样本（`truncated_normal_`），构建动作族（population）。
    - 对动作加上时序平滑与过去动作影响（`beta` 参数），并裁剪到上下界。
    - 可选地附加全零动作作为候选（`provide_zero_action`）。
    - 调用 `MPPIObjective.evaluate(population)` 以并行评估每条动作序列的轨迹价值：
      - `evaluate` 调用 `zero_small_actions`（将微小速度零化）并 `rollout` 动作序列得到状态轨迹。
      - `rollout` 在 `MPPIObjective` 中实现：对每步将控制 `(vx, vy, wz)` 积分为位姿变化，使用 pre-step yaw 旋转线速度，累积得到位置与朝向（详见 `rollout` 实现）。
      - `states_cost` 计算每条轨迹的逐步代价：位置代价（基于 `get_position_distance_to_goal` 使用 `max(l2, gdf)`）、控制代价（线性、侧向、旋转分量加权）、朝向代价、以及 traversability 相关代价（来自 `_compute_traversability()` 返回的 `self._trav`）。
    - 优化器基于回报值计算权重 `weights = exp(gamma * (values - values.max()))`，更新 `self.mean` 为加权平均的动作序列；同时记录本轮最佳样本用于回退（`best_traj`）。
  - 结束: 经过若干迭代（`cfg.num_iterations`），取 `best_u` 或 `mean_u`，通过 `MPPIObjective.rollout(...)` 得到最终 `states`（路径）。

- **为何生成的是“可行”路径**:
  - 代价函数在设计上对不可行或危险区域惩罚严厉：`traversability_cost` 与 `gdf` 的结合确保在靠近或位于障碍/致命区域的动作会收到较高代价；MPPI 在采样时倾向于选择代价低的轨迹。
  - GDF 的使用确保路径考虑到地图上的连通性（若某方向被障碍包围，GDF 会很大或不可达，从而抬高代价）。
  - 若混合度导致 GDF 在中心不可达，代码会降级到纯 geodesic（`lamb=0`），以提高通过狭窄通道的能力，减少误判不可达的情况。

- **重要配置项（影响路径特性）**:
  - `cfg.mppi.horizon` (`H`): 规划步数，影响路径长度和控制序列维度。
  - `cfg.mppi.population_size`, `cfg.mppi.sigma`, `cfg.mppi.gamma`, `cfg.mppi.beta`, `cfg.mppi.num_iterations`: MPPI 的核心超参数，控制探索强度、收敛行为与更新速率。
  - `cfg.map_resolution`, `cfg.map_size`: 地图物理尺寸与分辨率，会影响 GDF 与位置代价的物理尺度。
  - `cfg.paths_per_image`: 每帧采样目标数量（数据集规模的主要决定因子）。

将这些细节已追加进文档中。如需我把文中每个引用精确到行号，或把优化器关键超参数列出默认值，我可以继续补充并把配置文件中的默认值一并抓入文档。
