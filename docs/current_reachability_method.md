# 当前 Reachability 生成方法

本文档依据当前仓库代码整理，主要实现位于 dataset_builder/reachability/teacher_a.py、traversability.py 和 coordinates.py。

当前方法是一个以机器人中心为根、带圆形 footprint 约束的二维几何可达场。地图状态只有 (x,y)，没有 yaw 维度，也没有把转弯半径、速度或动力学约束放进搜索，因此不是完整 SE(2) reachability。

## 1. 输入与坐标

默认 elevation map 大小为 8 m x 8 m、分辨率 0.04 m，通常为 200 x 200 数组。

MapGeometry 规定：

- 数组轴 0 是机器人前方 x；
- 数组轴 1 是机器人左方 y；
- index (0,0) 的世界坐标锚点是 (-4,-4)；
- 映射使用 floor((xy-origin)/resolution)；
- 根位置是世界坐标 (0,0)。

因此 index (i,j) 的栅格锚点约为：

    x = -4 + i * 0.04
    y = -4 + j * 0.04

该 floor 量化与 MPPI 的 world_to_map_idx 一致。

## 2. Elevation 到 risk

Teacher-A 不直接把 elevation 数值作为障碍，而是先调用 TraversabilityFilter：

    score = TraversabilityFilter(elevation)
    risk = 1 - score

score 越大越安全，risk 越大越危险。原始 elevation 的 NaN 会恢复为 unknown；配置 border_cells=3 还会把最外侧 3 个栅格设为 NaN。因此：

    known_trav = isfinite(risk)

默认 fatal threshold 为 0.9：

    fatal = known_trav and risk >= 0.9

safe/risky 区间主要用于 MPPI soft cost；Teacher-A 的硬障碍使用 fatal threshold。

## 3. 圆形 footprint 离散化

Teacher-A 调用 circular_structure(radius_m, resolution)。请求半径向上量化：

    radius_cells = ceil(radius_m / resolution)
    effective_radius = radius_cells * resolution

默认网格下：

| 请求半径 | 栅格半径 | 有效半径 |
|---:|---:|---:|
| 0.26 m | 7 cells | 0.28 m |
| 0.28 m | 7 cells | 0.28 m |
| 0.61 m | 16 cells | 0.64 m |

圆结构满足 di^2 + dj^2 <= radius_cells^2。因此图中写 r=0.26 时，实际网格 footprint 是有效半径 0.28 m 的圆。

## 4. planning domain

默认整张地图是 domain support。为了保证完整 footprint 不越过地图边界，先做：

    planning_domain = binary_erosion(domain_support, footprint)
    outside_domain = not planning_domain

即中心在原始地图内，但 footprint 越界时也会标为 outside domain。

## 5. 中心局部状态

在 planning domain 内先检查中心：

    unknown_center    = planning_domain and not known_trav
    local_traversable = planning_domain and known_trav and risk < fatal_th
    local_blocked     = planning_domain and known_trav and risk >= fatal_th

这一步尚未检查 footprint 周围栅格。

## 6. Clearance 与 unknown footprint

先构造 fatal mask，并用圆结构做 binary dilation：

    raw_local_blocked = known_trav and risk >= fatal_th
    blocked_overlap = dilate(raw_local_blocked, footprint, border_value=0)
    unknown_overlap = dilate(not known_trav, footprint, border_value=1)

然后：

    clearance_blocked = local_traversable and blocked_overlap

    unknown_footprint_overlap = local_traversable and unknown_overlap

    unknown_footprint =
        local_traversable and not clearance_blocked and unknown_overlap

    configuration_free =
        local_traversable and not blocked_overlap and not unknown_overlap

clearance blocked 优先于 unknown footprint。unknown dilation 的 border value 为 1，因此靠近未知或边界的 footprint 是保守处理。

## 7. 根节点

默认 root_xy=(0,0)。根节点必须满足：

1. 在 grid 内；
2. 在 planning domain 内；
3. 中心 known；
4. 中心 local traversable；
5. footprint 不 clearance blocked；
6. footprint 不 unknown footprint。

只有 configuration_free[root_index] 为真才执行搜索。当前实现不会自动把无效 root 移到邻近 free cell；审计脚本中的 root anchoring 只是诊断比较，不改变正式 Teacher-A。

## 8. Strict Dijkstra

有效根节点上执行 strict_dijkstra(configuration_free, root_index, resolution)。

搜索使用 8 邻域：

- 水平/垂直代价为 1 * resolution；
- 对角代价为 sqrt(2) * resolution；
- 禁止 diagonal corner cutting：对角移动时水平和垂直邻格也必须 free。

输出：

- geodesic_m：根到每个栅格的距离；
- predecessor：路径回溯指针；
- reachable = configuration_free 且 geodesic_m 有限；
- traversable_but_disconnected = configuration_free 且不可从 root 到达。

因此 reachable 不只是局部 free，而是从机器人根真正连通到的区域。

## 9. 最终状态

状态枚举为：

    OUTSIDE_DOMAIN               = 0
    UNKNOWN_CENTER               = 1
    LOCALLY_BLOCKED              = 2
    CLEARANCE_BLOCKED            = 3
    UNKNOWN_FOOTPRINT            = 4
    TRAVERSABLE_BUT_DISCONNECTED = 5
    REACHABLE                    = 6

最终语义优先级为：

    outside_domain
     -> unknown_center
     -> locally_blocked
     -> clearance_blocked
     -> unknown_footprint
     -> traversable_but_disconnected
     -> reachable

## 10. 与矩形和 MPPI 的区别

正式 Teacher-A 使用圆形、无 yaw footprint，不能表达矩形前后悬伸、相同中心不同 yaw 的碰撞差异，也不包含动力学约束。

Rectangle Conflict Audit 的 yaw-aware rectangle 是诊断工具，不修改 Teacher-A。

MPPI 的 get_trav_cost 会：

1. 按 yaw 旋转 footprint sample；
2. floor quantize 到栅格；
3. 对有效 sample 读取 traversability cost；
4. 单独计数 unknown；
5. 对有效样本做 arithmetic mean；
6. unknown fraction 乘以 unknown cost；
7. 再进入 50-state trajectory objective。

所以 MPPI 不是 footprint risk 的 max，也不是任一 fatal sample 就立即令轨迹失败。Teacher-A hard blocked 与 MPPI soft cost 是不同判据。

## 11. 当前拓扑图脚本

当前脚本为：

    dataset_builder/src/plot_topology_batch.py

它对每个 image_id：

1. 读取 elevation；
2. 重新计算 traversability/risk；
3. 调用 build_teacher_a(radius=0.26)；
4. 显示 reachable、clearance blocked、local blocked、unknown、disconnected；
5. 叠加该 image_id 下所有 geo 或 tel paths；
6. 可选重新运行 MPPI，使用该 image_id 第一个 path 的 goal；
7. 显示 left/front/right 原图和 elevation/topology 图。

示例：

    conda run -n limo python -m dataset_builder.src.plot_topology_batch       --mission /home/robot-device/yangyujie/BEV_LIMO/LIMO_DATASET/2024-11-02-21-12-51       --source geo --count 10 --radius 0.26       --output-dir /home/robot-device/yangyujie/Try2/topology_geo_r026

--source tel 对应 teleop_paths；--skip-mppi 可关闭重新规划。

## 12. 输出与保守性

TeacherAResult 还保留 planning_domain、known_trav、local_traversable、local_blocked、clearance_blocked、unknown masks、configuration_free、reachable、disconnected、geodesic、predecessor 和 root status。

统计时不应只报告 reachable，还应报告 configuration-free、disconnected、unknown footprint、clearance blocked 和 locally blocked。

主要保守性来源是：

1. 半径向上取整；
2. NaN 和边界进入 unknown 传播；
3. unknown footprint 从 configuration free 中排除；
4. fatal threshold 硬切分；
5. strict Dijkstra 禁止 diagonal corner cutting；
6. root 必须自身 configuration free；
7. 二维圆形 footprint 不包含 yaw。

总结：当前方法是“risk 图 + 圆形 footprint 腐蚀/膨胀 + 根节点 strict Dijkstra”的二维中心拓扑场。它适合作为中心线拓扑候选和保守安全基准，但不能解释为带方向的完整整机 SE(2) 可达性。

