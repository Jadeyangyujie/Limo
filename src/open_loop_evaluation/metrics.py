from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from .common import DEFAULT_BUILD_CONFIG
from .data import MissionStore, Sample, path_length_m

from dataset_builder.mppi_planner.mppi_planner import (  # noqa: E402
    GridMap2D,          # 把 elevation map 包装成 2D 地图
    MPPIObjective,      # 根据地图、起点、目标生成可通行性图和 goal distance field
    clip_on_ray,        # 地图外点沿射线裁剪回地图内
    valid_mask,         # 判断地图格子是否在地图范围内
    world_to_map_idx,   # 世界坐标转地图格子坐标
)

"""
    给定一条预测路径 path
    它有没有碰撞？
    有没有到达目标？
    路径效率 SPL 是多少？
    最终离目标多远？"""


@dataclass(frozen=True)
class MetricOptions:
    """用户选择的评测规则"""
    success_distance_m: float = 1.0           # 最终离目标多近算到达，默认 1.0m
    goal_source: str = "raw_goal"             # 评测目标来源，默认用 zarr 的原始 goal; goal_source 有两种：raw_goal 使用 sample.goal / reference_endpoint 使用 sample.reference_path 最后一个点
    collision_mode: str = "footprint_any"     # 碰撞判定方式; collision_mode 有三种：footprint_any footprint 任意格子碰到 fatal 就算碰撞 / footprint_fraction fatal 格子比例超过阈值才算碰撞 / centerline 只看路径中心线是否碰到 fatal
    fatal_fraction_threshold: float = 0.03    # 如果用 footprint 比例判碰撞，阈值是多少
    collision_unknown: bool = False           # 未知区域是否算碰撞
    collision_out_of_map: bool = False        # 走出地图是否算碰撞


@dataclass(frozen=True)
class MetricResult:
    """评测结果的数据结构 每评估一条 path 就返回一个 MetricResult"""
    collision: bool                           # 是否碰撞
    reach: bool                               # 是否到达目标
    success: bool                             # 是否成功，通常是 reach and not collision
    spl: float                                # Success weighted by Path Length，成功且路径越短越高
    path_len_m: float                         # 实际路径长度
    shortest_gd_m: float                      # 从起点到目标的最短 GD 距离
    final_gd_m: float                         # 路径终点到目标的 GD 距离
    final_l2_m: float                         # 路径终点到 metric goal 的欧氏距离
    final_raw_goal_l2_m: float                # 路径终点到原始 goal 的欧氏距离
    final_reference_l2_m: float               # 路径终点到参考路径终点的欧氏距离
    unknown_contact: bool                     # 是否碰到未知地图区域
    out_of_map: bool                          # 是否出地图
    footprint_any_fatal: bool                 # 机器人 footprint 是否碰到任意 fatal 区域
    centerline_fatal: bool                    # 路径中心线是否碰到 fatal 区域
    max_fatal_footprint_fraction: float       # 某个时刻 footprint 中 fatal 格子的最大比例


class OpenLoopPathMetric:
    """Metric implementation matching the paper-level success definition."""
    """真正计算 Collision / Reach / Success / SPL 的类"""

    def __init__(
        self,
        device: str = "cuda",
        build_config: Path = DEFAULT_BUILD_CONFIG,
        map_resolution: float = 0.04,
        origin_xy: tuple[float, float] = (-4.0, -4.0),
        footprint_extents_m: tuple[float, float, float, float] | None = None,
        elevation_rotate_k: int = 0,
    ) -> None:
        if elevation_rotate_k not in (0, 2):
            raise ValueError(
                "elevation_rotate_k must be 0 or 2. A 90-degree rotation would "
                "also require a different map-frame origin/axis convention."
            )
        self.device = torch.device(device)
        # 读取 MPPI 配置，并创建 MPPIObjective。MPPIObjective 很关键，它内部会根据 elevation map 生成：
        # _trav: traversability / 可通行性图
        # _gdf: goal distance field / 到目标的距离场
        # _robot_footprint: 机器人 footprint 覆盖的局部格子
        # _gm: grid map
        # _goal: 当前目标
        build_cfg = OmegaConf.load(build_config)
        self.mppi_cfg = build_cfg.mppi
        self.footprint_extents_m = footprint_extents_m
        if footprint_extents_m is not None:
            front, rear, left, right = footprint_extents_m
            self.mppi_cfg.footprint = [[[-rear, -right], [front, left]]]
        self.objective = MPPIObjective(self.mppi_cfg, str(self.device))
        self.map_resolution = float(map_resolution)
        self.elevation_rotate_k = int(elevation_rotate_k)
        self.origin_xy = torch.tensor(origin_xy, dtype=torch.float32, device=self.device)
        # 起点固定是 [0, 0, 0]，表示机器人局部坐标系原点
        self.start = torch.zeros(3, dtype=torch.float32, device=self.device)
        # 这是缓存。因为同一个 image_id 可能会评估多种方法，如果每次都重新加载 elevation map 会浪费时间，所以缓存上一次的 gridmap
        self._last_key: tuple[str, int] | None = None
        self._last_gridmap: GridMap2D | None = None

    def metric_goal(self, sample: Sample, goal_source: str) -> np.ndarray:
        """这个函数决定“评测时的目标点用哪个” 很重要，因为有时候原始 goal 和参考路径终点不完全一致，换目标来源会影响 success / SPL"""
        if goal_source == "raw_goal":
            return np.asarray(sample.goal, dtype=np.float32)
        if goal_source == "reference_endpoint":
            return np.asarray(sample.reference_path[-1], dtype=np.float32)
        raise ValueError(f"Unknown goal_source: {goal_source}")

    def gridmap_for_sample(self, store: MissionStore, sample: Sample) -> GridMap2D:
        """给当前 sample 取对应的 elevation map 并包装成 GridMap2D"""
        key = (sample.mission, sample.image_id)
        if self._last_key == key and self._last_gridmap is not None:
            # 如果上次已经加载过同一个 mission + image_id，就直接复用
            return self._last_gridmap
        # The map rotation is applied before every collision/GDF computation.
        elevation = self.elevation_for_sample(store, sample)
        gridmap = GridMap2D(
            elevation=torch.from_numpy(elevation).to(self.device),
            resolution=self.map_resolution,
            origin_xy=self.origin_xy,
        )
        self._last_key = key
        self._last_gridmap = gridmap
        return gridmap

    def elevation_for_sample(self, store: MissionStore, sample: Sample) -> np.ndarray:
        """Return elevation in the same base-frame orientation as paths."""
        elevation = store.elevation_for_image(sample.image_id)
        if self.elevation_rotate_k:
            elevation = np.rot90(elevation, self.elevation_rotate_k).copy()
        return elevation

    def set_goal(self, gridmap: GridMap2D, goal: np.ndarray) -> None:
        """告诉 MPPIObjective 当前地图是谁？起点在哪里？目标在哪里？"""
        goal_tensor = torch.tensor(goal, dtype=torch.float32, device=self.device)
        # 调用之后，objective 内部就会准备好 _trav、_gdf 等后面评估要用的数据
        self.objective.set_observation(gridmap, self.start, goal_tensor)

    def final_gd_to_goal(self, store: MissionStore, sample: Sample, path: np.ndarray) -> float:
        """路径最后一个点到 raw goal 的 GD 距离 """
        # 它主要给 prepare.py 用，用来筛选 D_GEO 中 reachable 的路径。
        # 注意它这里固定用 sample.goal，也就是 raw goal，不看 MetricOptions.goal_source
        gridmap = self.gridmap_for_sample(store, sample)
        self.set_goal(gridmap, sample.goal)
        final_xy = torch.tensor(path[-1:, :2], dtype=torch.float32, device=self.device)
        return float(self._distance_to_goal(final_xy)[0].item())

    def evaluate(self, store: MissionStore, sample: Sample, path: np.ndarray, options: MetricOptions) -> MetricResult:
        """这是整个文件最重要的函数。输入一条路径，输出完整指标。"""
        # Step 1: 准备 path / gridmap / goal 评估目标可以由 options.goal_source 决定
        path = np.asarray(path, dtype=np.float32)
        gridmap = self.gridmap_for_sample(store, sample)
        goal = self.metric_goal(sample, options.goal_source)
        self.set_goal(gridmap, goal)

        # Step 2: 拿出路径终点和各种目标点
        path_tensor = torch.tensor(path, dtype=torch.float32, device=self.device)
        final_xy = path_tensor[-1:, :2]
        goal_xy = torch.tensor(goal[:2], dtype=torch.float32, device=self.device)
        raw_goal_xy = torch.tensor(sample.goal[:2], dtype=torch.float32, device=self.device)
        ref_xy = torch.tensor(sample.reference_path[-1, :2], dtype=torch.float32, device=self.device)

        # Step 3: 计算距离
        # 终点到当前 metric goal 的 goal distance field 距离
        final_gd = float(self._distance_to_goal(final_xy)[0].item())
        # 起点 [0,0] 到目标的 GDF 距离，可理解为地图上的近似最短可达距离
        shortest = float(
            self._distance_to_goal(torch.zeros(1, 2, dtype=torch.float32, device=self.device))[0].item()
        )
        # 实际路径长度
        path_len = path_length_m(path)
        # 终点到当前 metric goal 的直线距离
        final_l2 = float(torch.norm(final_xy[0] - goal_xy).item())
        # 终点到 raw goal 的直线距离
        final_raw_l2 = float(torch.norm(final_xy[0] - raw_goal_xy).item())
        # 终点到 reference endpoint 的直线距离
        final_ref_l2 = float(torch.norm(final_xy[0] - ref_xy).item())

        # Step 4: 计算碰撞和成功指标
        # _footprint_status() 会先收集 footprint 是否碰到 fatal / unknown / out-of-map
        footprint = self._footprint_status(path)
        # _collision_from_footprint() 再根据用户选择的规则决定最终是否算 collision
        collision = self._collision_from_footprint(footprint, options)
        # 判断到达：
        reach = math.isfinite(final_gd) and final_gd <= options.success_distance_m
        # 判断成功： 到达目标 + 没有碰撞 + shortest_gd 是有效值
        success = bool(reach and not collision and math.isfinite(shortest))
        # 算 SPL： 路径越接近最短路，SPL 越接近 1。路径绕远了，SPL 会变小
        denom = max(path_len, shortest)
        spl = float(shortest / denom) if success and denom > 1e-6 else 0.0

        return MetricResult(
            collision=collision,
            reach=reach,
            success=success,
            spl=spl,
            path_len_m=path_len,
            shortest_gd_m=shortest,
            final_gd_m=final_gd,
            final_l2_m=final_l2,
            final_raw_goal_l2_m=final_raw_l2,
            final_reference_l2_m=final_ref_l2,
            unknown_contact=bool(footprint["unknown_contact"]),
            out_of_map=bool(footprint["out_of_map"]),
            footprint_any_fatal=bool(footprint["footprint_any_fatal"]),
            centerline_fatal=bool(footprint["centerline_fatal"]),
            max_fatal_footprint_fraction=float(footprint["max_fatal_footprint_fraction"]),
        )

    def _collision_from_footprint(self, footprint: dict[str, Any], options: MetricOptions) -> bool:
        """把 footprint 状态转换成最终 collision"""
        if options.collision_mode == "footprint_any":
            # 只要 footprint 任意格子碰到 fatal，就算碰撞
            collision = bool(footprint["footprint_any_fatal"])
        elif options.collision_mode == "footprint_fraction":
            # 如果 fatal footprint 比例超过阈值，才算碰撞。比如阈值是 0.03，表示超过 3% 的 footprint 格子 fatal 才算
            collision = (
                float(footprint["max_fatal_footprint_fraction"])
                > options.fatal_fraction_threshold
            )
        elif options.collision_mode == "centerline":
            # 只看路径中心点是否 fatal，不看完整机器人外形
            collision = bool(footprint["centerline_fatal"])
        else:
            raise ValueError(f"Unknown collision mode: {options.collision_mode}")

        # 如果用户设置了未知区域或出地图也算碰撞，那就合并进去
        if options.collision_unknown:
            collision = collision or bool(footprint["unknown_contact"])
        if options.collision_out_of_map:
            collision = collision or bool(footprint["out_of_map"])
        return collision

    def _distance_to_goal(self, states_xy: torch.Tensor) -> torch.Tensor:
        """计算一批 states_xy 到当前 goal 的距离 它结合了两种距离"""
        # 普通欧氏距离
        goal_xy = self.objective._goal[:2].to(states_xy.device)
        l2 = torch.norm(states_xy - goal_xy, dim=-1)

        # goal distance field，表示地图上每个格子到目标的路径距离，考虑可通行性，比纯直线距离更接近“导航距离”
        gdf2d = self.objective._gdf.squeeze(0)
        # 处理 inf/nan: 如果 GDF 中有无穷或非法值，就用有限值均值填掉，避免后面索引出问题
        finite = torch.isfinite(gdf2d)
        fill = (
            gdf2d[finite].mean()
            if finite.any()
            else torch.tensor(0.0, device=gdf2d.device)
        )
        gdf2d = torch.where(finite, gdf2d, fill)

        # 把世界坐标转成地图坐标
        ij = world_to_map_idx(states_xy, self.objective._gm)
        flat_ij = ij.reshape(-1, 2).to(device=gdf2d.device, dtype=torch.long)
        valid = valid_mask(flat_ij, self.objective._gm)
        if (~valid).any():
            clipped = flat_ij.clone()
            for idx in torch.nonzero(~valid, as_tuple=False).flatten():
                # 如果点在地图外：就沿射线裁剪回地图边界
                # TODO: GEO path 的生成本身就基于地图，所以这种情况应该不会有；但是tel应该会有很多，导致gdf错误 > 1?
                clipped[idx] = clip_on_ray(gdf2d.shape, flat_ij[idx]).to(torch.long)
            flat_ij = clipped

        height, width = gdf2d.shape
        flat_ij[:, 0].clamp_(0, height - 1)
        flat_ij[:, 1].clamp_(0, width - 1)
        # 最后读取对应格子的 GDF
        flat_gdf = gdf2d.reshape(-1).index_select(
            0, flat_ij[:, 0] * width + flat_ij[:, 1]
        )
        # 这里取 max(l2, gdf)，意思是最终距离至少不能比直线距离还小。这样可以避免 GDF 插值/裁剪导致距离不合理地偏小
        return torch.max(l2, flat_gdf.reshape(states_xy.shape[:-1]))

    def _footprint_status(self, path: np.ndarray) -> dict[str, Any]:
        """判断整条路径和地图的碰撞关系，是最复杂的一段"""
        # path: [1, horizon, 3]
        states = torch.tensor(path, dtype=torch.float32, device=self.device)
        if states.ndim == 2:
            states = states.unsqueeze(0)
        # 拿机器人 footprint： cells_xy 表示机器人身体覆盖的一组局部坐标点
        cells_xy = self.objective._robot_footprint
        _, horizon, _ = states.shape
        num_cells = int(cells_xy.shape[0])

        xy, yaw = states[..., :2], states[..., 2]
        # 根据每个 waypoint 的 yaw 建旋转矩阵
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        rot = torch.stack(
            [
                torch.stack([cos_y, -sin_y], dim=-1),
                torch.stack([sin_y, cos_y], dim=-1),
            ],
            dim=-2,
        )
        fp = cells_xy.view(1, 1, num_cells, 2).expand(1, horizon, num_cells, 2)
        # 把机器人局部 footprint 旋转和平移到世界坐标 [1, horizon, num_cells, 2] ---> 每个时间点，机器人 footprint 的每个格子，在世界坐标下在哪里
        world_points = (
            torch.matmul(
                fp.reshape(horizon, num_cells, 2),
                rot.reshape(horizon, 2, 2).transpose(-1, -2),
            )
            + xy.reshape(horizon, 1, 2)
        ).reshape(1, horizon, num_cells, 2)

        # 然后转成地图格子
        ij = world_to_map_idx(world_points, self.objective._gm)
        valid = valid_mask(ij.reshape(-1, 2), self.objective._gm).reshape(
            1, horizon, num_cells
        )
        # 如果任意 footprint 点在地图外，out_of_map=True
        out_of_map = bool((~valid).any().item())

        # 然后初始化记录矩阵：
        fatal_grid = torch.zeros((1, horizon, num_cells), device=self.device)
        trav_grid = torch.zeros((1, horizon, num_cells), device=self.device)
        unknown_contact = False
        if valid.any():
            valid_ij = ij[valid]
            # 对地图内的 footprint 点，读取 traversability：
            trav_values = self.objective._trav[valid_ij[:, 0], valid_ij[:, 1]]
            # 如果 traversability 是 NaN，说明未知区域：
            unknown_contact = bool(torch.isnan(trav_values).any().item())
            # 把 NaN 转成 0，方便后续计算：
            finite_values = torch.nan_to_num(trav_values, nan=0.0)
            trav_grid[valid] = finite_values
            # 判断 fatal： 可通行性值超过 fatal_value 的格子被认为是危险/碰撞
            fatal_grid[valid] = (
                finite_values >= float(self.mppi_cfg.fatal_value)
            ).float()

        # 计算 footprint fatal 比例 ---> dim=2 是 footprint cells 这一维。它表示每个 waypoint 上，机器人身体有多少比例压在 fatal 区域上
        fatal_fraction = fatal_grid.float().mean(dim=2)
        # 接着单独算中心线是否 fatal： 这里只看路径点中心 [x, y]，不看 footprint 外形
        center_ij = world_to_map_idx(states[0, :, :2], self.objective._gm)
        center_valid = valid_mask(center_ij, self.objective._gm)
        centerline_fatal = False
        if center_valid.any():
            center_values = self.objective._trav[
                center_ij[center_valid, 0], center_ij[center_valid, 1]
            ]
            center_values = torch.nan_to_num(center_values, nan=0.0)
            # 如果中心线经过 fatal cell：centerline_fatal = True
            centerline_fatal = bool(
                (center_values >= float(self.mppi_cfg.fatal_value)).any().item()
            )

        return {
            "footprint_any_fatal": bool(fatal_grid.any().item()),
            "centerline_fatal": centerline_fatal,
            "unknown_contact": unknown_contact,
            "out_of_map": out_of_map,
            "max_fatal_footprint_fraction": float(fatal_fraction.max().item()),
            "max_trav": float(trav_grid.max().item()) if trav_grid.numel() else 0.0,
        }
