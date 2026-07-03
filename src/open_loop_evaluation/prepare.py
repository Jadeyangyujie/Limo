from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from .common import progress
from .data import (
    GEOMETRIC_EVAL_GROUP,
    GEOMETRIC_GROUP,
    TELEOP_EVAL_GROUP,
    TELEOP_GROUP,
    TELEOP_PLANNER_GROUP,
    MissionStore,
    collect_dataset_stats,
)
from .metrics import OpenLoopPathMetric
from .methods import GeometricPlannerPredictor

"""生成 D_TEL / D_GEO 可达评测子集"""
"""在正式 evaluate 之前，把原始的 geometric paths 过滤一遍，只保留“几何路径确实能到达目标”的样本，写成一个新的 zarr group, 后面评测时就用这个过滤后的 geo_eval 数据"""
# 目的： 让 D_GEO evaluation set 只包含几何规划器本身可以到达目标的样本，避免把本来就不可达的目标也拿来评测 LiMO 或 baseline


def build_teleop_evaluation_sets(
    stores: list[MissionStore],
    metric: OpenLoopPathMetric,
    success_distance_m: float = 1.0,
    force: bool = False,
) -> list[dict[str, object]]:
    """把原始 teleop_paths 过滤成 teleop_paths_evalution，并保存一一对应的 teleop_paths_planner。"""
    rows: list[dict[str, object]] = []
    planner = GeometricPlannerPredictor(metric, device=str(metric.device))
    for store in stores:
        if store.split != "test" or not store.has_group("tel"):
            continue

        source = store.groups["tel"]
        eval_path = store.mission_dir / "data" / TELEOP_EVAL_GROUP
        planner_path = store.mission_dir / "data" / TELEOP_PLANNER_GROUP
        if eval_path.exists() and planner_path.exists() and not force:
            existing = zarr.open_group(str(eval_path), mode="r")
            source_count = int(existing.attrs.get("source_count", source["path"].shape[0]))
            selected_count = int(existing["path"].shape[0])
            rows.append(
                {
                    "mission": store.mission,
                    "source_group": TELEOP_GROUP,
                    "eval_group": TELEOP_EVAL_GROUP,
                    "planner_group": TELEOP_PLANNER_GROUP,
                    "source_samples": source_count,
                    "reachable_samples": selected_count,
                    "reachable_percent": 100.0 * selected_count / source_count if source_count else 0.0,
                    "status": "reused",
                }
            )
            continue

        selected_indices: list[int] = []
        planner_paths: list[np.ndarray] = []
        final_gds: list[float] = []
        total = int(source["path"].shape[0])
        for index in progress(range(total), total=total, desc=f"filter tel {store.mission}"):
            sample = store.get_sample("D_TEL", "tel", index)
            planned_path = planner.predict(store, sample)
            final_gd = metric.final_gd_to_goal(store, sample, planned_path)
            if np.isfinite(final_gd) and final_gd <= success_distance_m:
                selected_indices.append(index)
                planner_paths.append(planned_path)
                final_gds.append(final_gd)

        write_teleop_evaluation_groups(
            source=source,
            eval_path=eval_path,
            planner_path=planner_path,
            indices=selected_indices,
            planner_paths=planner_paths,
            final_gds=final_gds,
            success_distance_m=success_distance_m,
        )
        rows.append(
            {
                "mission": store.mission,
                "source_group": TELEOP_GROUP,
                "eval_group": TELEOP_EVAL_GROUP,
                "planner_group": TELEOP_PLANNER_GROUP,
                "source_samples": total,
                "reachable_samples": len(selected_indices),
                "reachable_percent": 100.0 * len(selected_indices) / total if total else 0.0,
                "status": "written",
            }
        )
    return rows

def build_geometric_evaluation_sets(
    stores: list[MissionStore],           # 所有 mission 的数据对象。每个 store 代表一个 mission
    metric: OpenLoopPathMetric,                  # 指标计算器，用来判断路径最终点是否到达目标
    success_distance_m: float = 1.0,      # 成功阈值，默认 1 米。也就是说，如果路径最终点到目标的 GD 距离小于等于 1 米，就认为这条 geometric path 是 reachable
    force: bool = False,                  # 是否强制重建。如果 force=False 且输出 group 已经存在，就直接复用旧结果；如果 force=True，即使已有结果也重新筛选并覆盖写入
) -> list[dict[str, object]]:
    """遍历所有 mission 把 test split 里的原始 geo 样本过滤成 geo_eval 样本"""
    # 返回值是：每个 dict 是一行统计信息，比如某个 mission 原始有多少样本、筛出来多少、比例是多少、状态是 reused 还是 written。
    rows: list[dict[str, object]] = []
    for store in stores:
        if store.split != "test" or not store.has_group("geo"):
            continue

        # source 是原始 geometric paths zarr group
        source = store.groups["geo"]
        # <mission>/data/geometric_paths_evalution
        out_path = store.mission_dir / "data" / GEOMETRIC_EVAL_GROUP
        # 如果目标路径已经存在，并且没有要求强制重建，就不重新计算，直接读取已有结果。
        if out_path.exists() and not force:
            existing = zarr.open_group(str(out_path), mode="r")
            source_count = int(existing.attrs.get("source_count", source["path"].shape[0]))
            selected_count = int(existing["path"].shape[0])
            rows.append(
                {
                    "mission": store.mission,
                    "source_samples": source_count,
                    "reachable_samples": selected_count,
                    "reachable_percent": 100.0 * selected_count / source_count if source_count else 0.0,
                    "status": "reused",
                }
            )
            continue

        # 如果没有已有结果，就重新筛选
        selected_indices: list[int] = []       # 保存被选中的原始样本下标
        final_gds: list[float] = []            # 保存这些样本的最终 GD 距离
        total = int(source["path"].shape[0])   # 原始 geometric paths 的样本数量
        # 遍历当前 mission 里的每一条 geometric path，同时显示进度条
        for index in progress(range(total), total=total, desc=f"filter {store.mission}"):
            # 从原始 geo group 里取出第 index 个样本，包装成 Sample
            sample = store.get_sample("D_GEO", "geo", index)
            # sample.reference_path 的最后一个 waypoint 到 sample.goal 的 goal-distance-field 距离
            final_gd = metric.final_gd_to_goal(store, sample, sample.reference_path)
            # 只有当 final_gd 是有限数，并且小于等于阈值时，才保留
            if np.isfinite(final_gd) and final_gd <= success_distance_m:
                # 记录它的原始 index 和 final GD 值
                selected_indices.append(index)
                final_gds.append(final_gd)

        # 筛选完当前 mission 后，把这些被选中的样本写到新的 zarr group 里，路径是 <mission>/data/geometric_paths_evalution
        write_filtered_group(source, out_path, selected_indices, final_gds, success_distance_m)
        rows.append(
            {
                "mission": store.mission,
                "source_samples": total,
                "reachable_samples": len(selected_indices),
                "reachable_percent": 100.0 * len(selected_indices) / total if total else 0.0,
                "status": "written",
            }
        )
    return rows


def write_filtered_group(
    source: zarr.hierarchy.Group,
    out_path: Path,
    indices: list[int],
    final_gds: list[float],
    success_distance_m: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open_group(str(out_path), mode="w")
    idx = np.asarray(indices, dtype=np.int64)

    # 它从原始 group 里复制四个字段："path", "goal", "image_id", "goal_time"
    # 但不是全量复制，而是只复制 idx 对应的那些样本
    for name in ("path", "goal", "image_id", "goal_time"):
        data = np.asarray(source[name][idx])
        chunks = chunks_for_filtered_data(source[name].chunks, data.shape)
        out.create_dataset(name, data=data, chunks=chunks)

    # 这个字段保存“过滤后的每个样本来自原始 source 的哪个 index”
    out.create_dataset("source_index", data=idx, chunks=(min(1000, max(len(idx), 1)),))
    # 这个字段保存每条被保留路径的 final GD 值
    out.create_dataset(
        "final_goal_gd_m",
        data=np.asarray(final_gds, dtype=np.float32),
        chunks=(min(1000, max(len(final_gds), 1)),),
    )
    out.attrs["source_group"] = GEOMETRIC_GROUP
    out.attrs["source_count"] = int(source["path"].shape[0])
    out.attrs["reachable_count"] = int(len(idx))
    out.attrs["success_distance_m"] = float(success_distance_m)
    out.attrs["description"] = (
        "Filtered D_GEO evaluation set: source geometric paths whose final waypoint "
        "has GD <= success_distance_m to the sampled raw goal."
    )


def write_teleop_evaluation_groups(
    source: zarr.hierarchy.Group,
    eval_path: Path,
    planner_path: Path,
    indices: list[int],
    planner_paths: list[np.ndarray],
    final_gds: list[float],
    success_distance_m: float,
) -> None:
    idx = np.asarray(indices, dtype=np.int64)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_group = zarr.open_group(str(eval_path), mode="w")
    planner_group = zarr.open_group(str(planner_path), mode="w")

    for name in ("path", "goal", "image_id", "goal_time"):
        data = np.asarray(source[name][idx])
        chunks = chunks_for_filtered_data(source[name].chunks, data.shape)
        eval_group.create_dataset(name, data=data, chunks=chunks)
        if name == "path":
            if planner_paths:
                planner_data = np.asarray(planner_paths, dtype=np.float32)
            else:
                planner_data = np.empty((0,) + tuple(source[name].shape[1:]), dtype=np.float32)
            chunks = chunks_for_filtered_data(source[name].chunks, planner_data.shape)
            planner_group.create_dataset(name, data=planner_data, chunks=chunks)
        else:
            planner_group.create_dataset(name, data=data, chunks=chunks)

    source_index_chunks = (min(1000, max(len(idx), 1)),)
    final_gd_chunks = (min(1000, max(len(final_gds), 1)),)
    for group in (eval_group, planner_group):
        group.create_dataset("source_index", data=idx, chunks=source_index_chunks)
        group.create_dataset(
            "final_goal_gd_m",
            data=np.asarray(final_gds, dtype=np.float32),
            chunks=final_gd_chunks,
        )
        group.attrs["source_group"] = TELEOP_GROUP
        group.attrs["source_count"] = int(source["path"].shape[0])
        group.attrs["reachable_count"] = int(len(idx))
        group.attrs["success_distance_m"] = float(success_distance_m)

    eval_group.attrs["paired_planner_group"] = TELEOP_PLANNER_GROUP
    eval_group.attrs["description"] = (
        "Filtered D_TEL evaluation set: source teleop paths whose raw goals are "
        "reachable by MPPI planner with final GD <= success_distance_m."
    )
    planner_group.attrs["paired_eval_group"] = TELEOP_EVAL_GROUP
    planner_group.attrs["description"] = (
        "MPPI planner paths paired one-to-one with teleop_paths_evalution. "
        "Path, goal, image_id, and goal_time count/order match the eval group."
    )


def chunks_for_filtered_data(source_chunks: tuple[int, ...], data_shape: tuple[int, ...]) -> tuple[int, ...]:
    """复用 source chunk 的空间维度，同时让第一维适配过滤后的样本数。"""
    if not source_chunks:
        return source_chunks
    if not data_shape:
        return source_chunks
    first_dim = max(int(data_shape[0]), 1)
    chunks = list(source_chunks)
    chunks[0] = min(int(chunks[0]), first_dim)
    return tuple(chunks)


def stats_after_prepare(stores: list[MissionStore]) -> list[dict[str, object]]:
    """用于 prepare 之后重新统计数据集"""
    # 它重新创建一批 MissionStore。为什么要重新创建？因为 prepare 过程可能刚刚写入了新的 geo_eval group，旧的 store 对象可能还不知道磁盘上多了这个 group。
    # 重新创建的 MissionStore 可能只知道目录里的数据，但不知道原来从 missions_csv 里读出来的 split 和 short_name。所以这里把旧 store 的这些元信息复制回来。
    refreshed = [MissionStore(store.mission_dir) for store in stores]
    for refreshed_store, old_store in zip(refreshed, stores):
        refreshed_store.split = old_store.split
        refreshed_store.short_name = old_store.short_name
    return collect_dataset_stats(refreshed)
