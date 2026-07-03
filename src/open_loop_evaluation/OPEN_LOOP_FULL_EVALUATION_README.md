# LiMO Open-Loop Full Evaluation README

这份文档用于正式跑完整开环评测，并生成可以直接和 LiMO 论文 TABLE II 对比的结果。

当前评测不再使用运行时 eligible set。所有可达性过滤都在 `prepare` 阶段写成固定 zarr group，`evaluate` 阶段只读取这些固定 group 做评测和汇总。

## 1. 数据集定义

本代码使用以下 zarr group：

| 逻辑数据集 | 使用的 zarr group | 说明 |
|---|---|---|
| `D_TEL` | `teleop_paths_evalution` | 从原始 `teleop_paths` 中筛出的 MPPI 可达目标；path 是遥控轨迹 |
| `D_GEO` | `geometric_paths_evalution` | 从原始 `geometric_paths` 中筛出的可达几何轨迹 |
| `D_AUG` | `geometric_paths_evalution + teleop_paths_evalution` | D_GEO eval + D_TEL eval |

另有一个和 `teleop_paths_evalution` 一一对应的 group：

```text
teleop_paths_planner
```

它保存同一批 goal/image/time 对应的 MPPI 规划路径。`Geometric Planner` 遇到 `teleop_paths_evalution` 样本时都会直接读取这个 group，不重新规划；因此在 `D_TEL` 上会全量读取它，在 `D_AUG` 上遇到其中的 D_TEL 部分也会读取它。

## 2. 入口和容器

host repo：

```text
/home/enping/VisualNav/project/robotic-planning_fm/less-is-more
```

container repo：

```text
/root/ros2_ws/src/robotic-planning_fm/less-is-more
```

入口：

```text
algorithms/evalution/src/run_open_loop_evaluation.py
```

查看参数：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py --help
```

### 自定义数据集：直接评测，不做 GD 过滤

若自定义数据集已经有固定的参考路径，使用 `evaluate` 子命令即可。它不会运行
`prepare`，不会调用 MPPI，也不会执行任何 `GD <= 1m` 的样本过滤。

当前自定义数据只需将 zarr group 命名为：

```text
<data_dir>/<mission_name>/data/teleop_paths_evalution
<data_dir>/<mission_name>/data/geometric_paths_evalution
```

也就是说，可将原始 `teleop_paths_once_time` 和
`geometric_paths_smac_lattice_narrow` 直接重命名或建立同名软链接。两者均需至少包含
`path`、`goal`、`image_id`、`goal_time` 数组。

运行时还需要：

```text
<data_dir>/<mission_name>/data/elevation_map
<data_dir>/<mission_name>/images/hdr_front
<data_dir>/<mission_name>/images/hdr_left
<data_dir>/<mission_name>/images/hdr_right
<data_dir>/<mission_name>/images_undistorted_308x476/hdr_front
<data_dir>/<mission_name>/images_undistorted_308x476/hdr_left
<data_dir>/<mission_name>/images_undistorted_308x476/hdr_right
<data_dir>/<mission_name>/data/hdr_front
<data_dir>/<mission_name>/data/hdr_left
<data_dir>/<mission_name>/data/hdr_right
<data_dir>/<mission_name>/data/dlio_map_odometry
```

最后五项是 `LiMO Side Cams Sync + Front History Depth` 所需的时间戳和 DLIO
历史位姿数据。`D_TEL` 使用全部 `teleop_paths_evalution` 样本；`D_AUG` 使用
`geometric_paths_evalution + teleop_paths_evalution`，不做任何基于 GD 的样本筛选。
历史模型仍会按训练时的相机同步和 DLIO pose 可用性约束排除无效输入，并在结果中单独记录；
这不是对 path 做的质量筛选。

假设自定义数据已挂载到容器内的 `<container_data_dir>`，一次评测六个方法并同时输出
`D_TEL`、`D_AUG` 的命令为：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py evaluate \
  --non-interactive \
  --dataset-root <container_data_dir> \
  --output-root algorithms/evalution/results/open_loop_evaluation \
  --run-id custom-no-gd-filter \
  --datasets D_TEL,D_AUG \
  --methods limo_D_tel,limo_D_aug,limo_side_cams,limo_side_cams_undistorted,limo_side_cams_sync_history_depth,straight_line \
  --success-distance-m 1.0 \
  --goal-source raw_goal \
  --collision-mode footprint_any \
  --batch-size 4
```

这里必须使用 `evaluate`，不能使用 `prepare` 或 `all`。`--batch-size 4` 是为了给
`LiMO Side Cams Sync + Front History Depth` 留出显存；如果容器显存不足，改成 `1`。

若自定义 mission 不在默认 `missions_split.csv` 中，评测本身仍可运行；但
`dataset_statistics.csv` 的 train/test 列会显示为 0。需要正确统计 split 时，额外传入一个
CSV（列至少为 `Timestamp`、`Split`），例如：

```text
Timestamp,Split
<mission_name>,test
```

并在上述命令中加入：

```text
--missions-csv <container_custom_missions_csv>
```

### 配置参考

#### 运行命令参数

| 参数 | 默认值 | 作用 | 自定义数据集注意事项 |
|---|---|---|---|
| `--dataset-root` | `algorithms/evalution/dataset/grandtour` | mission 目录的根路径 | 必须是容器内路径，目录下直接是 `<mission_name>`。 |
| `--missions-csv` | `algorithms/less-is-more/limo/configs/dataset/missions_split.csv` | mission 的 split 定义 | 支持任意 CSV 路径。未显式传入时使用该默认文件；自定义 mission 不在其中时仍可评测，但统计表没有正确的 train/test 分组。 |
| `--output-root` | `algorithms/evalution/results/open_loop_evaluation` | 结果目录 | 每个 `run-id` 会生成独立子目录。 |
| `--datasets` | 交互模式选择；非交互默认 `D_TEL,D_GEO` | 选择 `D_TEL`、`D_GEO`、`D_AUG` | 自定义实验通常显式写 `D_TEL,D_AUG`。 |
| `--methods` | 交互模式选择；非交互有默认值 | 选择方法 ID | 建议始终显式写出，确保实验可复现。 |
| `--device` | `cuda` | PyTorch / metric 的设备 | 无 GPU 时写 `cpu`，但历史模型会明显变慢。 |
| `--batch-size` | `64` | LiMO 的推理 batch | 含 `limo_side_cams_sync_history_depth` 时建议从 `4` 开始；显存不足则改为 `1`。 |
| `--debug-visualize` | 关闭 | 为被判 collision 的样本保存本地 PNG 诊断图 | 默认关闭，不影响正式 Table II 结果。 |
| `--debug-visualize-max-samples` | `20` | 每个 `dataset × method` 保存的 collision PNG 上限 | 设为 `0` 表示保存全部 collision 样本；全量可能产生数千张 PNG。 |
| `--max-samples` | 不限制 | 仅评测时截断样本数 | 只用于 smoke test，不应用于正式结果。 |
| `--collision-modes` | 空 | 逗号分隔的多个 collision 规则，例如 `footprint_any,footprint_fraction` | 为空时使用单个 `--collision-mode`；非空时同一批预测 path 会被多套 collision 规则分别评测，不重复跑 LiMO 推理。 |

`missions.csv` 最少需要这两列；可有其他列：

```csv
Timestamp,Split
<mission_name>,test
```

#### 每个方法的图像目录和网络输入

`image_subdir` 是 [methods.py](../src/open_loop_evaluation/methods.py) 中每个 `MethodSpec` 的独立参数，默认值只影响该方法，不会影响其他方法。需要让某一个模型读取去畸变图像时，直接在该模型的 `MethodSpec(...)` 内增加或修改：

```python
image_subdir="images_undistorted_308x476"
```

| 方法 | 默认 `image_subdir` | 当前图像输入 | 说明 |
|---|---|---|---|
| `limo_D_tel` | `images` | `hdr_front` | 可单独改为去畸变目录，模型会读取该目录下的前视图。 |
| `limo_D_aug` | `images` | `hdr_front` | 同上。 |
| `limo_side_cams` | `images` | `hdr_front`、`hdr_left`、`hdr_right` | 默认按相同 image id 读取三相机。 |
| `limo_side_cams_undistorted` | `images_undistorted_308x476` | 三相机 | 对应去畸变三相机 checkpoint。 |
| `limo_side_cams_sync_history_depth` | `images_undistorted_308x476` | 同步三相机 + 前视历史 | 对应同步历史 checkpoint；同步和历史参数见下一表。 |
| `straight_line` | 不适用 | 无图像输入 | 只使用 zarr 中的 `goal`。 |

所有 LiMO 图像在读取后都会 resize 到 `308 x 476`。因此，`image_subdir` 可因方法而不同，但目录内的 `hdr_front/<image_id>.jpeg` 必须存在；三相机方法还要求 `hdr_left`、`hdr_right`。

#### 同步历史模型的固定训练契约

以下参数位于 `limo_side_cams_sync_history_depth` 的 `MethodSpec` 中，当前没有 CLI 覆盖项，目的是避免无意中偏离该 checkpoint 的训练设定：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `side_cam_use_nearest` | `True` | left/right 用 front timestamp 查找最近图像，而非直接复用 front image id。 |
| `side_cam_time_tolerance_s` | `0.02` | 最近侧相机帧的最大允许时间误差。 |
| `front_history_size` | `4` | 使用四张前视历史图。 |
| `front_history_stride` | `2` | 历史序列为 `[id-6, id-4, id-2, id]`。 |
| `front_history_pose_yaw_offset_rad` | `-pi/2` | GrandTour DLIO yaw 到训练坐标系的修正。 |
| `front_history_pose_tolerance_s` | `0.1` | front timestamp 到最近 DLIO pose 的最大允许误差。 |

自定义数据若不能满足上述 timestamp 和 DLIO 对齐条件，历史模型会写出 `excluded_samples_*.csv`；不要把它与 GD/path 过滤混为一谈。

#### Metric 与地图参数

| 参数/位置 | 当前值 | 何时需要调整 |
|---|---:|---|
| `--success-distance-m` | `1.0` m | 定义最终 reach/success 的 GD 阈值；这是评测指标，不是样本预筛选。 |
| `--goal-source` | `raw_goal` | 需和参考实验保持一致；改为 `reference_endpoint` 会改变 success 与 SPL。 |
| `--collision-mode` | `footprint_any` | 需和目标实验保持一致；可选 `footprint_fraction`、`centerline`。 |
| `--collision-modes` | 空 | 可一次给出多个 collision 规则，例如 `footprint_any,footprint_fraction`；为空时沿用 `--collision-mode`。 |
| `--collision-unknown` | `False` | 自定义地图中 unknown 是否计为碰撞。 |
| `--collision-out-of-map` | `False` | 自定义地图中走出 map 是否计为碰撞。 |
| `--map-resolution-m` | `0.04` m | elevation map 分辨率；自定义数据不同必须在命令中显式传入。 |
| `--map-origin-xy X Y` | `-4.0 -4.0` m | elevation map 原点；自定义数据不同必须在命令中显式传入，坐标系必须与 path/goal 一致。 |
| `--elevation-rotate-k` | `0` | elevation map 在进入 collision、GDF 和 Geometric Planner 前逆时针旋转 `k × 90°`。仅支持 `0` 或 `2`；`0` 保持 Table II 默认行为，`2` 用于 base-frame 下高程图前后、左右都翻转的 180° 失配。 |
| `--footprint-front-m`、`--footprint-rear-m`、`--footprint-left-m`、`--footprint-right-m` | `build.yaml` 默认 footprint | 机器人相对 base frame 原点的前、后、左、右延伸量；四项必须同时传入。 |
| `STRAIGHT_LINE_MAX_LENGTH_M` | `25.0` m | 当前 straight-line 会在该长度截断；若要复现论文定义，应改回 `5.0` m。 |

自定义数据的地图坐标系不同而不传入正确的 `--map-resolution-m`、`--map-origin-xy`，碰撞、GD 和 SPL 都没有解释价值。这两个参数会保存到 `run_args.yaml`，并显示在终端的 `Using Metric` 表中。

#### MPPI traversability / GDF buffer 参数

评测中的 collision 和 GD 都复用 `dataset_builder/configs/build.yaml` 里的 `mppi`
配置。下面三个参数容易和“障碍物膨胀层”混在一起，但它们影响的对象并不完全一样：

| 参数 | 默认值 | 实际作用 | 对 collision 的影响 |
|---|---:|---|---|
| `mppi.fatal_cells_buffer` | `0` cell | 在 elevation map 经过 traversability 网络后，对原始 traversability 分数做 `max_pool`。`n=1` 表示用 `3×3` kernel，等价于把高风险/fatal 分数向周围扩一圈 cell；`n=2` 是 `5×5`，依此类推。 | 会直接改变 `_trav`，因此会直接改变 `footprint_any` / `footprint_fraction` / `centerline` 的 collision 判定。当前默认 `0` 表示不额外扩张 traversability fatal 区域。 |
| `mppi.gdf_obstacle_buffer` | `1` cell | 在 `_trav` 已经离散成 fatal/non-fatal 后，只对 GDF 使用的障碍 mask 再做一次 `max_pool`。默认 `1` 表示 GDF 最短路距离场会把 fatal 周围一圈 cell 也当作障碍。 | 不直接改变 collision，因为 `_footprint_status()` 读的是 `_trav`；但会改变 `shortest_gd_m`、`final_gd_m`、reach/success 和 SPL。 |
| `mppi.border_cells` | `3` cells | 把 traversability map 四周 `b` 个 cell 强制设为 `NaN` unknown，避免路径或距离场依赖地图边界处不可靠区域。 | 默认情况下 unknown 不算 collision；只有显式传入 `--collision-unknown` 才会把这些边界 unknown contact 合并为 collision。但它仍会影响 unknown debug 标记，也可能影响 GDF。 |

换算到米时，cell 数需要乘以 `--map-resolution-m`。例如 ROS2 real 常用
`--map-resolution-m 0.05` 时：

- `fatal_cells_buffer=1` 约等于向周围扩 `0.05m` 的栅格半径；
- `gdf_obstacle_buffer=1` 约等于只在 GDF 中增加 `0.05m` 障碍 buffer；
- `border_cells=3` 约等于地图四周各 `0.15m` 标为 unknown。

因此，如果 debug 图里红色 fatal footprint cell 看起来“离实体障碍还有距离”，优先区分两件事：

1. 当前默认 `fatal_cells_buffer=0`，collision 图上的红色 fatal 区域通常不是这个参数额外膨胀出来的，而是 traversability 网络/高程边缘本身输出为 fatal。
2. `footprint_any` 非常严格：只要 footprint 里 1 个离散 cell 命中 fatal，就判 collision。若标题里 `max_fatal_fraction=0.004` 或 `0.008`，通常只代表 1～2 个 footprint cell 命中。自定义数据可用 `--collision-modes footprint_any,footprint_fraction --fatal-fraction-threshold 0.03` 做敏感性分析；正式对比时要明确记录所采用的 collision 口径。

一次评测多种 collision 口径时，路径只生成一次，随后同一条 path 会分别计算多套指标。输出会变成：

```text
summary.csv                                      # 每个 collision mode 一行 summary
per_sample_<dataset>_<method>_footprint_any.csv
per_sample_<dataset>_<method>_footprint_fraction_0p03.csv
debug_visualizations/<dataset>/<method>/footprint_any/*.png
debug_visualizations/<dataset>/<method>/footprint_fraction_0p03/*.png
```

如果只使用单个 `--collision-mode`，文件命名保持旧格式，例如
`per_sample_D_TEL_real_world.csv`。

若 debug 图显示 elevation map 相对 path/goal 恰好旋转了 180°，可额外传入：

```text
--elevation-rotate-k 2
```

它会在读取每帧 elevation map 后执行 `np.rot90(elevation, 2)`；路径、goal 和
footprint 均保持原来的 base-frame 定义。该选项默认是 `0`，所以不会改变 GrandTour
Table II 的原有口径。不要把 `2` 用于仅镜像或 90° 旋转的地图；这类情况需要重新确认
map origin/axis 定义，而不是套用本开关。

**重要：** `prepare` 阶段的 D_TEL 可达性过滤和 `teleop_paths_planner` 同样使用 elevation
map。因此第一次使用 `--elevation-rotate-k 2` 时，应使用相同的地图参数和
`--force-dtel-eval` 重建 D_TEL groups；旧 group 的 MPPI 可达性筛选不再与新的碰撞/GDF
度量一致。

例如 ROS2 real 机器人若相对 base frame 的尺寸为前 `0.5m`、后 `0.6m`、左/右各 `0.3m`，加入：

```text
--footprint-front-m 0.5
--footprint-rear-m 0.6
--footprint-left-m 0.3
--footprint-right-m 0.3
```

这会覆盖评测和 Geometric Planner 的 footprint 为 `x=[-0.6, 0.5]m`、`y=[-0.3, 0.3]m`。

#### Collision 本地诊断图

开启 `--debug-visualize` 后，评测不会改变 collision 规则，只会额外保存被当前规则判为
collision 的 PNG：

```text
<output-root>/<run-id>/debug_visualizations/<dataset>/<method>/*.png
```

若使用 `--collision-modes` 同时评测多种 collision 口径，则 debug 图会再按 mode 分目录保存：

```text
<output-root>/<run-id>/debug_visualizations/<dataset>/<method>/<collision_mode>/*.png
```

每张图同时显示 elevation、traversability、fatal/unknown cell、评测路径、goal、抽样
footprint，以及以下精确触发信息：

- 红色 `X` 和 `t=<index>`：发生 fatal footprint contact 的 path waypoint；
- 红色方块：该 waypoint 的 footprint 内真正命中的 fatal 栅格；
- 标题中的 `max_fatal_fraction`：任意 waypoint 上 fatal footprint cell 的最大比例。

默认的 `footprint_any` 规则只要任意一个 footprint cell fatal 即判 collision。因此
`max_fatal_fraction=0.004` 一类结果可能只代表一个离散 footprint 栅格命中；应先检查
debug 图，再决定是否需要对自定义 elevation map 校准阈值或使用
`footprint_fraction` 做敏感性分析。

例如，先审计 100 个 real-world collision：

```bash
python3 algorithms/evalution/src/run_open_loop_evaluation.py evaluate \
  --non-interactive \
  --dataset-root <container_data_dir> \
  --missions-csv <container_custom_missions_csv> \
  --output-root algorithms/evalution/results/open_loop_evaluation \
  --run-id ros2-real-real-world-debug \
  --datasets D_TEL \
  --methods real_world \
  --map-resolution-m 0.05 \
  --map-origin-xy -4.0 -4.0 \
  --elevation-rotate-k 2 \
  --footprint-front-m 0.5 \
  --footprint-rear-m 0.6 \
  --footprint-left-m 0.3 \
  --footprint-right-m 0.3 \
  --collision-modes footprint_any,footprint_fraction \
  --fatal-fraction-threshold 0.03 \
  --debug-visualize \
  --debug-visualize-max-samples 100
```

将 `--debug-visualize-max-samples` 设为 `0` 会保存全部 collision 样本；只建议在确认磁盘
空间和运行时间足够时使用。

## 3. Prepare：生成固定评测集

如果你已经生成过 `geometric_paths_evalution`，它会被复用。现在还需要为 D_TEL 生成：

```text
<mission>/data/teleop_paths_evalution
<mission>/data/teleop_paths_planner
```

运行：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py prepare \
  --non-interactive \
  --run-id open-loop-prepare-eval-groups
```

prepare 会：

- 遍历 test split 的原始 `teleop_paths`；
- 对每个 teleop goal 运行一次 MPPI / Geometric Planner；
- 若 planner path 的 final waypoint 到 raw goal 的 GD `<= 1.0m`，则保留该样本；
- 把遥控路径写入 `teleop_paths_evalution`；
- 把对应 MPPI 路径写入 `teleop_paths_planner`；
- 对 D_GEO 复用或生成 `geometric_paths_evalution`。

需要强制重建 D_TEL eval groups：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py prepare \
  --non-interactive \
  --force-dtel-eval \
  --run-id open-loop-prepare-dtel-rebuild
```

需要强制重建 D_GEO eval group：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py prepare \
  --non-interactive \
  --force-dgeo-eval \
  --run-id open-loop-prepare-dgeo-rebuild
```

prepare 输出：

```text
algorithms/evalution/results/open_loop_evaluation/<run-id>/dataset_statistics.csv
algorithms/evalution/results/open_loop_evaluation/<run-id>/dtel_reachable_filter.csv
algorithms/evalution/results/open_loop_evaluation/<run-id>/dgeo_reachable_filter.csv
```

## 4. 评测口径

论文 success 定义：

```text
A path is deemed successful if
(i) the GD of the final waypoint to the goal is <= 1.0 m
and (ii) the robot does not collide.
```

建议固定使用：

```text
--success-distance-m 1.0
--goal-source raw_goal
--collision-mode footprint_any
```

`--collision-unknown` 和 `--collision-out-of-map` 默认关闭。

## 5. 去畸变三相机与历史深度模型

新增两个只支持 `D_TEL` 与 `D_AUG` 的方法：

| 方法 ID | 权重 | 图像输入 |
|---|---|---|
| `limo_side_cams_undistorted` | `grandtour_limo_side_cams_D_aug_undistort.safetensors` | 当前 front/left/right，读取 `<mission>/images_undistorted_308x476` |
| `limo_side_cams_sync_history_depth` | `grandtour_limo_side_cams_sync_D_aug_front_history_depth.safetensors` | 当前同步 front/left/right，加 front history depth=4、stride=2 |

历史模型的输入严格对齐训练配置：

- 左右相机按当前 front timestamp 找最近帧，时间误差必须 `<= 0.02s`；
- 前视历史帧顺序为 `[id-6, id-4, id-2, id]`；
- 每帧按最近 DLIO pose 对齐，最大时间误差 `<= 0.1s`；
- DLIO yaw 使用 `-pi/2` 修正；`front_history_delta_pose` 表达在当前机器人坐标系；
- 若 history 起始帧不存在，训练和评测都会用零图像填充；
- 若当前帧找不到满足同步阈值的侧相机或 DLIO pose，该样本不会进入此历史模型的指标分母；若存在这类样本，排除原因保存在 `excluded_samples_<dataset>_limo_side_cams_sync_history_depth.csv`。

`metadata_undistorted_308x476` 保存的是去畸变图像的相机内参；上述 LiMO 网络的前向输入并不包含内参张量，因此评测不读取它。它用于确认 `images_undistorted_308x476` 的标定和尺寸（`308 x 476`）与训练一致。

分别跑两个新方法的 D_TEL：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py evaluate \
  --non-interactive \
  --run-id open-loop-DTEL-undistorted-side-cams \
  --datasets D_TEL \
  --methods limo_side_cams_undistorted,limo_side_cams_sync_history_depth \
  --success-distance-m 1.0 \
  --goal-source raw_goal \
  --collision-mode footprint_any \
  --batch-size 4
```

分别跑两个新方法的 D_AUG：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py evaluate \
  --non-interactive \
  --run-id open-loop-DAUG-undistorted-side-cams \
  --datasets D_AUG \
  --methods limo_side_cams_undistorted,limo_side_cams_sync_history_depth \
  --success-distance-m 1.0 \
  --goal-source raw_goal \
  --collision-mode footprint_any \
  --batch-size 4
```

历史模型的 `summary.csv` 会额外记录 `num_candidate_samples`、`num_excluded_samples`；终端中的 `#Input` 和 `#Excluded` 也是同一含义。普通去畸变三相机模型不会应用这个时间同步/pose 过滤。

## 6. Smoke Test

正式全量前先跑小样本，确认 eval groups 和模型都能读到：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py evaluate \
  --non-interactive \
  --run-id open-loop-smoke-fixed-groups \
  --datasets D_TEL \
  --methods limo_D_tel,limo_D_aug,limo_side_cams,straight_line,real_world,geometric_planner \
  --max-samples 4 \
  --success-distance-m 1.0 \
  --goal-source raw_goal \
  --collision-mode footprint_any \
  --batch-size 1
```

如果 smoke 通过，再跑全量。

## 7. 全量实验 A：Evaluated on D_TEL

方法：

- `Trained on D_TEL`
- `Trained on D_AUG`
- `LiMO Side Cams`
- `Straight-Line Paths`
- `Real-World Paths`
- `Geometric Planner`

命令：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py evaluate \
  --non-interactive \
  --run-id open-loop-full-DTEL-fixed-groups \
  --datasets D_TEL \
  --methods limo_D_tel,limo_D_aug,limo_side_cams,straight_line,real_world,geometric_planner \
  --success-distance-m 1.0 \
  --goal-source raw_goal \
  --collision-mode footprint_any \
  --batch-size 64
```

## 8. 全量实验 B：Evaluated on D_AUG

方法：

- `Trained on D_TEL`
- `Trained on D_AUG`
- `LiMO Side Cams`
- `Straight-Line Paths`
- `Geometric Planner`

`D_AUG` 不包含 `Real-World Paths`，因为 `Real-World Paths` 只对 teleop 路径有明确含义。

命令：

```bash
docker exec -w /root/ros2_ws/src/robotic-planning_fm/less-is-more visual_navigation \
  python3 algorithms/evalution/src/run_open_loop_evaluation.py evaluate \
  --non-interactive \
  --run-id open-loop-full-DAUG-fixed-groups \
  --datasets D_AUG \
  --methods limo_D_tel,limo_D_aug,limo_side_cams,straight_line,geometric_planner \
  --success-distance-m 1.0 \
  --goal-source raw_goal \
  --collision-mode footprint_any \
  --batch-size 64
```

## 9. 输出文件

每次 run 的输出目录：

```text
algorithms/evalution/results/open_loop_evaluation/<run-id>
```

关键文件：

| 文件 | 说明 |
|---|---|
| `run_args.yaml` | 本次运行参数快照 |
| `dataset_statistics.csv` | 本地数据集统计 |
| `summary.csv` | 每个 dataset/method 的最终汇总 |
| `paper_comparison.csv` | 本地结果和论文 TABLE II 数值的 CSV 对比 |
| `paper_comparison.md` | 本地结果和论文 TABLE II 数值的 Markdown 对比 |
| `per_sample_<dataset>_<method>.csv` | 每个样本的详细结果 |

## 10. 论文 TABLE II 目标值

`Evaluated on D_TEL`：

| Planner | Paper Col. % | Paper Succ. % | Paper SPL % |
|---|---:|---:|---:|
| `limo_D_tel` | 10.8 | 87.1 | 84.4 |
| `limo_D_aug` | 11.1 | 88.7 | 86.4 |
| `straight_line` | 12.5 | 87.5 | 87.5 |
| `real_world` | 11.0 | 89.0 | 87.5 |
| `geometric_planner` | 3.7 | 96.3 | 88.8 |

`Evaluated on D_AUG`：

| Planner | Paper Col. % | Paper Succ. % | Paper SPL % |
|---|---:|---:|---:|
| `limo_D_tel` | 14.1 | 51.4 | 49.7 |
| `straight_line` | 23.1 | 76.9 | 76.9 |
| `limo_D_aug` | 14.5 | 82.2 | 80.0 |
| `geometric_planner` | 1.0 | 99.0 | 95.4 |

## 11. 结果检查

查看 D_TEL 对比报告：

```bash
sed -n '1,120p' algorithms/evalution/results/open_loop_evaluation/open-loop-full-DTEL-fixed-groups/paper_comparison.md
```

查看 D_AUG 对比报告：

```bash
sed -n '1,120p' algorithms/evalution/results/open_loop_evaluation/open-loop-full-DAUG-fixed-groups/paper_comparison.md
```

查看 summary：

```bash
column -s, -t < algorithms/evalution/results/open_loop_evaluation/open-loop-full-DTEL-fixed-groups/summary.csv | less -S
column -s, -t < algorithms/evalution/results/open_loop_evaluation/open-loop-full-DAUG-fixed-groups/summary.csv | less -S
```

重点检查：

- `D_TEL / Geometric Planner` 的 `Reach %` 应接近或等于 100，因为它直接读取 `teleop_paths_planner`；
- `D_GEO` 中 `Geometric Planner` 直接读取 `geometric_paths_evalution`；
- `Real-World Paths Reach %` 不一定是 100，因为 D_TEL eval 是按“goal 被 MPPI 可达”过滤，不是按“teleop path 终点到 raw goal 的 GD <= 1m”过滤。

## 12. 权限整理

如果输出文件在 host 上不能编辑：

```bash
docker exec visual_navigation \
  chown -R 1000:1000 /root/ros2_ws/src/robotic-planning_fm/less-is-more/algorithms/evalution/results/open_loop_evaluation
```

如果 prepare 写出的 zarr group 权限不方便编辑：

```bash
docker exec visual_navigation \
  find /root/ros2_ws/src/robotic-planning_fm/less-is-more/algorithms/evalution/dataset/grandtour \
  \( -path '*/data/teleop_paths_evalution' -o -path '*/data/teleop_paths_planner' -o -path '*/data/geometric_paths_evalution' \) \
  -exec chown -R 1000:1000 {} +
```
