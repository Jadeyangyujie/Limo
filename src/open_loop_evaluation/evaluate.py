from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import batched, progress
from .console import print_section, print_table
from .data import MissionStore, Sample, dataset_subsets, iter_samples, write_csv
from .methods import MethodRunner, MethodSpec
from .metrics import MetricOptions, OpenLoopPathMetric


@dataclass
class MethodEvaluation:
    """一个 dataset/method 组合的有效样本结果及其输入筛选记录。"""

    rows: list[dict[str, Any]]
    candidate_samples: int
    excluded_samples: list[dict[str, object]]
    options: MetricOptions


PER_SAMPLE_FIELDS = [
    "dataset",
    "method",
    "method_label",
    "collision_mode",
    "fatal_fraction_threshold",
    "subset",
    "mission",
    "sample_index",
    "source_index",
    "image_id",
    "collision",
    "reach",
    "success",
    "spl",
    "path_len_m",
    "shortest_gd_m",
    "final_gd_m",
    "final_l2_m",
    "final_raw_goal_l2_m",
    "final_reference_l2_m",
    "unknown_contact",
    "out_of_map",
    "footprint_any_fatal",
    "centerline_fatal",
    "max_fatal_footprint_fraction",
    "goal_time",
]

PAPER_TABLE_II = {
    ("D_TEL", "limo_D_tel"): {"collision_percent": 10.8, "success_percent": 87.1, "spl_percent": 84.4},
    ("D_TEL", "limo_D_aug"): {"collision_percent": 11.1, "success_percent": 88.7, "spl_percent": 86.4},
    ("D_TEL", "straight_line"): {"collision_percent": 12.5, "success_percent": 87.5, "spl_percent": 87.5},
    ("D_TEL", "real_world"): {"collision_percent": 11.0, "success_percent": 89.0, "spl_percent": 87.5},
    ("D_TEL", "geometric_planner"): {"collision_percent": 3.7, "success_percent": 96.3, "spl_percent": 88.8},
    ("D_AUG", "limo_D_tel"): {"collision_percent": 14.1, "success_percent": 51.4, "spl_percent": 49.7},
    ("D_AUG", "straight_line"): {"collision_percent": 23.1, "success_percent": 76.9, "spl_percent": 76.9},
    ("D_AUG", "limo_D_aug"): {"collision_percent": 14.5, "success_percent": 82.2, "spl_percent": 80.0},
    ("D_AUG", "geometric_planner"): {"collision_percent": 1.0, "success_percent": 99.0, "spl_percent": 95.4},
}

"""evaluate.py = 对选中的数据集和方法逐个跑评测，并输出 per-sample 结果和 summary。"""
"""是一个“实验循环调度器”：它不关心 LiMO 网络细节，也不关心 footprint 怎么算，它只负责 选数据、选方法、跑路径、算指标、写结果、汇总表格"""
"""evaluate()
  遍历 datasets
    遍历 methods
      evaluate_method()
        iter_samples()
        MethodRunner.predict_many()
        OpenLoopPathMetric.evaluate()
        row_for_sample()
        write_csv(per_sample_*.csv)
      summarize_rows()
      print_method_summary()
  write_csv(summary.csv)
  打印 Final Summary"""
"""
  dataset = D_TEL
  subsets = ["tel_eval"]
  method = limo_D_tel
    samples = iter_samples(... D_TEL tel_eval ...)
    batch size = 64
    runner.predict_many()  -> LiMO 输出 paths
    metric.evaluate()      -> 每条 path 算指标
    写 per_sample_D_TEL_limo_D_tel.csv
    summarize_rows()       -> 汇总 LiMO 结果
  method = straight_line
    samples = iter_samples(... D_TEL tel_eval ...)
    batch size = 1
    runner.predict_many()  -> 生成直线路径
    metric.evaluate()      -> 算指标
    写 per_sample_D_TEL_straight_line.csv
    summarize_rows()       -> 汇总直线结果
  写 summary.csv
  打印 Final Summary"""
"""data.py:
    给 evaluate.py 提供 samples
   methods.py:
    给 evaluate.py 提供 path prediction
   metrics.py:
    给 evaluate.py 提供 metric calculation
   evaluate.py:
    负责把 samples、methods、metrics 串起来跑完整实验"""

def evaluate(
    run_dir: Path,                    # 本次运行的输出目录
    stores: list[MissionStore],       # 所有 mission 的数据访问对象
    dataset_root: Path,               # 数据集根目录，LiMO 读图片需要
    datasets: list[str],              # 要评测哪些数据集，例如 ["D_TEL", "D_GEO"]
    methods: list[MethodSpec],        # 要评测哪些方法，例如 LiMO、straight_line
    metric: OpenLoopPathMetric,              # 指标计算器
    options: MetricOptions | list[MetricOptions],           # 评测规则，比如成功距离、碰撞模式
    device: str,                      # cuda 或 cpu
    torch_hub_dir: Path,              # LiMO 加载 DINOv2 用的本地 hub 目录
    batch_size: int,                  # batch 大小，主要给 LiMO 用
    max_samples: int | None = None,   # 最多评测多少样本，方便 smoke test
    debug_visualize: bool = False,
    debug_visualize_max_samples: int = 20,
) -> list[dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    store_by_mission = {store.mission: store for store in stores}
    all_summary: list[dict[str, Any]] = []
    option_list = options if isinstance(options, list) else [options]

    for dataset in datasets:
        subsets = dataset_subsets(dataset)
        total = sum(store.group_len(subset) for subset in subsets for store in stores)
        if max_samples is not None:
            total = min(total, max_samples)
        dataset_methods = [method for method in methods if dataset in method.allowed_datasets]
        skipped_methods = [method for method in methods if dataset not in method.allowed_datasets]
        for method in skipped_methods:
            print(f"Skip {method.label} on {dataset}: unsupported dataset.")

        samples = list(iter_samples(stores, dataset, subsets, max_samples=max_samples))
        for method in dataset_methods:
            method_evaluations = evaluate_method(
                run_dir=run_dir,
                samples=samples,
                store_by_mission=store_by_mission,
                dataset_root=dataset_root,
                dataset=dataset,
                method=method,
                metric=metric,
                options=option_list,
                device=device,
                torch_hub_dir=torch_hub_dir,
                batch_size=batch_size,
                debug_visualize=debug_visualize,
                debug_visualize_max_samples=debug_visualize_max_samples,
            )
            for method_evaluation in method_evaluations:
                summary = summarize_rows(method_evaluation.rows, dataset, method)
                summary["collision_mode"] = method_evaluation.options.collision_mode
                summary["fatal_fraction_threshold"] = method_evaluation.options.fatal_fraction_threshold
                summary["collision_label"] = option_key(method_evaluation.options)
                summary["num_candidate_samples"] = method_evaluation.candidate_samples
                summary["num_excluded_samples"] = len(method_evaluation.excluded_samples)
                all_summary.append(summary)
                print_method_summary(summary)

    write_csv(run_dir / "summary.csv", all_summary)
    write_paper_comparison(run_dir, all_summary)
    print_section("Final Summary")
    print_table(
        ["Dataset", "Method", "Collision", "#Samples", "#Input", "#Excluded", "Col %", "Succ %", "SPL %", "Reach %", "GD p90"],
        [
            [
                row["dataset"],
                row["method_label"],
                row.get("collision_label", row.get("collision_mode", "")),
                row["num_samples"],
                row["num_candidate_samples"],
                row["num_excluded_samples"],
                row["collision_percent"],
                row["success_percent"],
                row["spl_percent"],
                row["reach_percent"],
                row["final_gd_p90_m"],
            ]
            for row in all_summary
        ],
    )
    return all_summary


def write_paper_comparison(run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    """把本地 summary 和论文 TABLE II 数值放在一起，方便实验后直接检查差异。"""
    rows = []
    for row in summary_rows:
        paper = PAPER_TABLE_II.get((row["dataset"], row["method"]))
        compared = {
            "dataset": row["dataset"],
            "method": row["method"],
            "method_label": row["method_label"],
            "collision_mode": row.get("collision_mode", ""),
            "fatal_fraction_threshold": row.get("fatal_fraction_threshold", ""),
            "num_samples": row["num_samples"],
            "num_candidate_samples": row.get("num_candidate_samples", row["num_samples"]),
            "num_excluded_samples": row.get("num_excluded_samples", 0),
            "local_collision_percent": row["collision_percent"],
            "local_success_percent": row["success_percent"],
            "local_spl_percent": row["spl_percent"],
            "paper_collision_percent": "",
            "paper_success_percent": "",
            "paper_spl_percent": "",
            "delta_collision_percent": "",
            "delta_success_percent": "",
            "delta_spl_percent": "",
        }
        if paper is not None and row.get("collision_mode", "footprint_any") == "footprint_any":
            compared["paper_collision_percent"] = paper["collision_percent"]
            compared["paper_success_percent"] = paper["success_percent"]
            compared["paper_spl_percent"] = paper["spl_percent"]
            compared["delta_collision_percent"] = row["collision_percent"] - paper["collision_percent"]
            compared["delta_success_percent"] = row["success_percent"] - paper["success_percent"]
            compared["delta_spl_percent"] = row["spl_percent"] - paper["spl_percent"]
        rows.append(compared)

    write_csv(run_dir / "paper_comparison.csv", rows)
    write_paper_comparison_markdown(run_dir / "paper_comparison.md", rows)


def write_paper_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if value == "":
            return ""
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = [
        "# Open-Loop Evaluation vs. LiMO TABLE II",
        "",
        "Percentages are computed from the fixed evaluation zarr groups used by each dataset.",
        "",
        "| Dataset | Method | Collision | #Samples | #Input | #Excluded | Local Col. | Paper Col. | Delta Col. | Local Succ. | Paper Succ. | Delta Succ. | Local SPL | Paper SPL | Delta SPL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["dataset"]),
                    fmt(row["method_label"]),
                    fmt(row["collision_mode"]),
                    fmt(row["num_samples"]),
                    fmt(row["num_candidate_samples"]),
                    fmt(row["num_excluded_samples"]),
                    fmt(row["local_collision_percent"]),
                    fmt(row["paper_collision_percent"]),
                    fmt(row["delta_collision_percent"]),
                    fmt(row["local_success_percent"]),
                    fmt(row["paper_success_percent"]),
                    fmt(row["delta_success_percent"]),
                    fmt(row["local_spl_percent"]),
                    fmt(row["paper_spl_percent"]),
                    fmt(row["delta_spl_percent"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_method(
    run_dir: Path,
    samples: list[Sample],
    store_by_mission: dict[str, MissionStore],
    dataset_root: Path,
    dataset: str,
    method: MethodSpec,
    metric: OpenLoopPathMetric,
    options: list[MetricOptions],
    device: str,
    torch_hub_dir: Path,
    batch_size: int,
    debug_visualize: bool,
    debug_visualize_max_samples: int,
) -> list[MethodEvaluation]:
    """评测一个 dataset + method 负责一个组合"""
    use_option_suffix = len(options) > 1
    per_sample_paths = {
        option_key(option): run_dir / per_sample_filename(dataset, method.id, option, use_option_suffix)
        for option in options
    }
    runner = MethodRunner(
        spec=method,
        dataset_root=dataset_root,
        metric=metric,
        device=device,
        torch_hub_dir=torch_hub_dir,
        batch_size=batch_size,
    )
    selected_samples, excluded_samples = runner.select_samples(samples)
    if excluded_samples:
        write_csv(run_dir / f"excluded_samples_{dataset}_{method.id}.csv", excluded_samples)
    rows_by_option = evaluate_samples(
        per_sample_paths=per_sample_paths,
        dataset=dataset,
        samples=selected_samples,
        store_by_mission=store_by_mission,
        dataset_root=dataset_root,
        method=method,
        metric=metric,
        options=options,
        device=device,
        torch_hub_dir=torch_hub_dir,
        batch_size=batch_size,
        runner=runner,
        debug_visualize_dir=(run_dir / "debug_visualizations" / dataset / method.id)
        if debug_visualize
        else None,
        debug_visualize_max_samples=debug_visualize_max_samples,
        use_option_debug_subdir=use_option_suffix,
    )
    return [
        MethodEvaluation(
            rows=rows_by_option[option_key(option)],
            candidate_samples=len(samples),
            excluded_samples=excluded_samples,
            options=option,
        )
        for option in options
    ]


def evaluate_samples(
    per_sample_paths: dict[str, Path] | None,
    dataset: str,
    samples: list[Sample],
    store_by_mission: dict[str, MissionStore],
    dataset_root: Path,
    method: MethodSpec,
    metric: OpenLoopPathMetric,
    options: list[MetricOptions],
    device: str,
    torch_hub_dir: Path,
    batch_size: int,
    runner: MethodRunner | None = None,
    debug_visualize_dir: Path | None = None,
    debug_visualize_max_samples: int = 20,
    use_option_debug_subdir: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """评测指定 samples；per_sample_paths 为 None 时只返回 rows，不落盘。"""
    if runner is None:
        runner = MethodRunner(
            spec=method,
            dataset_root=dataset_root,
            metric=metric,
            device=device,
            torch_hub_dir=torch_hub_dir,
            batch_size=batch_size,
        )

    rows_by_option: dict[str, list[dict[str, Any]]] = {option_key(option): [] for option in options}
    debug_saved_by_option: dict[str, int] = {option_key(option): 0 for option in options}
    for batch in progress(
        batched(samples, batch_size if method.kind == "limo" else 1),
        total=(len(samples) + batch_size - 1) // batch_size if method.kind == "limo" else len(samples),
        desc=f"{dataset}:{method.id}",
    ):
        # 生成 path + 计算 metric
        paths = runner.predict_many(store_by_mission, batch)
        for sample, path in zip(batch, paths):
            store = store_by_mission[sample.mission]
            for option in options:
                key = option_key(option)
                result = metric.evaluate(store, sample, path, option)
                rows_by_option[key].append(row_for_sample(sample, method, result, option))
                if (
                    debug_visualize_dir is not None
                    and result.collision
                    and (
                        debug_visualize_max_samples == 0
                        or debug_saved_by_option[key] < debug_visualize_max_samples
                    )
                ):
                    from .visualize import save_collision_debug_visualization

                    option_debug_dir = (
                        debug_visualize_dir / key
                        if use_option_debug_subdir
                        else debug_visualize_dir
                    )
                    saved_path = save_collision_debug_visualization(
                        option_debug_dir,
                        dataset=dataset,
                        method_id=f"{method.id}__{key}" if use_option_debug_subdir else method.id,
                        metric=metric,
                        store=store,
                        sample=sample,
                        path=path,
                        result=result,
                    )
                    debug_saved_by_option[key] += 1
                    print(f"Saved collision debug visualization: {saved_path}")

    if per_sample_paths is not None:
        for key, rows in rows_by_option.items():
            write_csv(per_sample_paths[key], rows)
    return rows_by_option


def option_key(options: MetricOptions) -> str:
    if options.collision_mode != "footprint_fraction":
        return options.collision_mode
    threshold = f"{options.fatal_fraction_threshold:.6g}".replace(".", "p").replace("-", "m")
    return f"footprint_fraction_{threshold}"


def per_sample_filename(dataset: str, method_id: str, options: MetricOptions, use_option_suffix: bool) -> str:
    if not use_option_suffix:
        return f"per_sample_{dataset}_{method_id}.csv"
    return f"per_sample_{dataset}_{method_id}_{option_key(options)}.csv"


def row_for_sample(sample: Sample, method: MethodSpec, result: Any, options: MetricOptions) -> dict[str, Any]:
    """把结果变成 CSV 行"""
    values = asdict(result)
    return {
        "dataset": sample.dataset,
        "method": method.id,
        "method_label": method.label,
        "collision_mode": options.collision_mode,
        "fatal_fraction_threshold": options.fatal_fraction_threshold,
        "subset": sample.subset,
        "mission": sample.mission,
        "sample_index": sample.index,
        "source_index": "" if sample.source_index is None else sample.source_index,
        "image_id": sample.image_id,
        "collision": int(values["collision"]),
        "reach": int(values["reach"]),
        "success": int(values["success"]),
        "spl": values["spl"],
        "path_len_m": values["path_len_m"],
        "shortest_gd_m": values["shortest_gd_m"],
        "final_gd_m": values["final_gd_m"],
        "final_l2_m": values["final_l2_m"],
        "final_raw_goal_l2_m": values["final_raw_goal_l2_m"],
        "final_reference_l2_m": values["final_reference_l2_m"],
        "unknown_contact": int(values["unknown_contact"]),
        "out_of_map": int(values["out_of_map"]),
        "footprint_any_fatal": int(values["footprint_any_fatal"]),
        "centerline_fatal": int(values["centerline_fatal"]),
        "max_fatal_footprint_fraction": values["max_fatal_footprint_fraction"],
        "goal_time": sample.goal_time,
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    method: MethodSpec,
) -> dict[str, Any]:
    """输入一个 method 的所有 per-sample rows 输出一行 summary。"""
    if not rows:
        return {
            "dataset": dataset,
            "method": method.id,
            "method_label": method.label,
            "collision_mode": "",
            "fatal_fraction_threshold": 0.0,
            "collision_label": "",
            "num_samples": 0,
            "collision_percent": 0.0,
            "success_percent": 0.0,
            "spl_percent": 0.0,
            "reach_percent": 0.0,
            "final_gd_p90_m": 0.0,
            "path_len_mean_m": 0.0,
        }
    collision = np.asarray([row["collision"] for row in rows], dtype=np.float64)
    success = np.asarray([row["success"] for row in rows], dtype=np.float64)
    reach = np.asarray([row["reach"] for row in rows], dtype=np.float64)
    spl = np.asarray([row["spl"] for row in rows], dtype=np.float64)
    final_gd = np.asarray([row["final_gd_m"] for row in rows], dtype=np.float64)
    path_len = np.asarray([row["path_len_m"] for row in rows], dtype=np.float64)
    return {
        "dataset": dataset,
        "method": method.id,
        "method_label": method.label,
        "num_samples": len(rows),
        "collision_percent": float(collision.mean() * 100.0),      # 碰撞率，越低越好
        "success_percent": float(success.mean() * 100.0),          # 成功率，越高越好
        "spl_percent": float(spl.mean() * 100.0),                  # SPL 百分比，越高越好
        "reach_percent": float(reach.mean() * 100.0),              # 到达率，不考虑碰撞，只看是否接近目标
        "final_gd_p90_m": float(np.nanpercentile(final_gd, 90)),   # 最终 GD 距离的 90 分位数。越小越好
        "path_len_mean_m": float(np.nanmean(path_len)),            # 平均路径长度
    }


def print_method_summary(row: dict[str, Any]) -> None:
    collision_label = row.get("collision_label") or row.get("collision_mode", "")
    print_section(f"{row['dataset']} / {row['method_label']} / {collision_label}")
    print_table(
        ["Collision", "#Samples", "#Input", "#Excluded", "Collision %", "Success %", "SPL %", "Reach %", "Final GD p90", "Mean path"],
        [
            [
                collision_label,
                row["num_samples"],
                row.get("num_candidate_samples", row["num_samples"]),
                row.get("num_excluded_samples", 0),
                row["collision_percent"],
                row["success_percent"],
                row["spl_percent"],
                row["reach_percent"],
                row["final_gd_p90_m"],
                row["path_len_mean_m"],
            ]
        ],
    )
