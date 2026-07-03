from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_MISSIONS_CSV,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TORCH_HUB_DIR,
    timestamp_run_id,
)
from .console import (
    ask_float,
    ask_yes_no,
    choose_many,
    choose_one,
    print_section,
    print_table,
)
from .data import collect_dataset_stats, dataset_subsets, load_mission_stores, write_csv
from .evaluate import evaluate
from .methods import available_methods
from .metrics import MetricOptions, OpenLoopPathMetric
from .prepare import build_geometric_evaluation_sets, build_teleop_evaluation_sets

"""交互式、开环评测流程"""

def main() -> None:
    args = parse_args()
    if args.map_resolution_m <= 0.0:
        raise ValueError("--map-resolution-m must be positive.")
    if args.elevation_rotate_k not in (0, 2):
        raise ValueError("--elevation-rotate-k must be either 0 or 2.")
    if args.debug_visualize_max_samples < 0:
        raise ValueError("--debug-visualize-max-samples must be non-negative.")
    collision_modes = collision_modes_from_args(args)
    footprint_extents_m = footprint_extents_from_args(args)
    # 本地数据集地址
    dataset_root = Path(args.dataset_root)
    # 评测结果输出地址
    output_root = Path(args.output_root)
    # 默认 mission split 配置
    missions_csv = Path(args.missions_csv)
    # 默认 DINOv2 torch hub 缓存目录
    torch_hub_dir = Path(args.torch_hub_dir)
    method_registry = available_methods(
        bev_limo_a_weights=(
            Path(args.bev_limo_a_weights) if args.bev_limo_a_weights else None
        ),
        ptc_limo_weights=(
            Path(args.ptc_limo_weights) if args.ptc_limo_weights else None
        ),
        limo_d_tel_weights=(
            Path(args.limo_d_tel_weights) if args.limo_d_tel_weights else None
        ),
        limo_d_aug_weights=(
            Path(args.limo_d_aug_weights) if args.limo_d_aug_weights else None
        ),
        limo_side_cams_weights=(
            Path(args.limo_side_cams_weights) if args.limo_side_cams_weights else None
        ),
        limo_image_subdir=args.limo_image_subdir or None,
        ptc_limo_image_subdir=args.ptc_limo_image_subdir or None,
        limo_d_tel_image_subdir=args.limo_d_tel_image_subdir or None,
        limo_d_aug_image_subdir=args.limo_d_aug_image_subdir or None,
        limo_side_cams_image_subdir=args.limo_side_cams_image_subdir or None,
    )

    # 每次运行可以记录关键词参数，生成一个唯一的 run_id 作为输出目录，方便结果管理和对比
    run_id = args.run_id or timestamp_run_id()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_args(run_dir, args)

    # stores 可以理解成“所有 mission 的数据索引”。每个 store 代表一个 mission，里面能访问 teleop paths、geometric paths、elevation map 等
    stores = load_mission_stores(dataset_root, missions_csv)
    # 指标计算器，用来判断 GD 距离、碰撞、成功率等
    metric = OpenLoopPathMetric(
        device=args.device,
        map_resolution=args.map_resolution_m,
        origin_xy=tuple(args.map_origin_xy),
        footprint_extents_m=footprint_extents_m,
        elevation_rotate_k=args.elevation_rotate_k,
    )

    # 先打印并保存本地数据集统计信息
    print_dataset_statistics(stores, run_dir, "Local Dataset Statistics")

    # 如果选择了 prepare 或 all，先准备 D_TEL / D_GEO 的评测子集（如果需要的话），并打印过滤统计信息
    # 如果选择了 evaluate，只跑评测，不准备数据
    if args.command in {"prepare", "all"}:
        should_prepare_dtel = args.force_dtel_eval or args.non_interactive
        should_prepare_dgeo = args.force_dgeo_eval or args.non_interactive
        if not args.non_interactive:
            missing_dtel = any(
                store.split == "test"
                and store.has_group("tel")
                and (not store.has_group("tel_eval") or not store.has_group("tel_planner"))
                for store in stores
            )
            missing_dgeo = any(
                store.split == "test" and store.has_group("geo") and not store.has_group("geo_eval")
                for store in stores
            )
            dtel_question = (
                "Build/rebuild D_TEL reachable evaluation set "
                "(<mission>/data/teleop_paths_evalution and teleop_paths_planner)?"
            )
            dgeo_question = (
                "Build/rebuild D_GEO reachable evaluation set "
                "(<mission>/data/geometric_paths_evalution)?"
            )
            should_prepare_dtel = (
                args.force_dtel_eval
                or missing_dtel
                or ask_yes_no(dtel_question, default=missing_dtel)
            )
            should_prepare_dgeo = (
                args.force_dgeo_eval
                or missing_dgeo
                or ask_yes_no(dgeo_question, default=missing_dgeo)
            )
        if should_prepare_dtel:
            rows = build_teleop_evaluation_sets(
                stores=stores,
                metric=metric,
                success_distance_m=args.dtel_reachable_gd_m,
                force=args.force_dtel_eval,
            )
            write_csv(run_dir / "dtel_reachable_filter.csv", rows)
            print_section("D_TEL Reachable Goal Filter")
            print_table(
                ["Mission", "Source", "Reachable", "Reachable %", "Status"],
                [
                    [
                        row["mission"],
                        row["source_samples"],
                        row["reachable_samples"],
                        row["reachable_percent"],
                        row["status"],
                    ]
                    for row in rows
                ],
            )
            stores = load_mission_stores(dataset_root, missions_csv)
            print_dataset_statistics(stores, run_dir, "Statistics After D_TEL Filtering")
        if should_prepare_dgeo:
            # 筛选 D_GEO reachable samples。它会用 metric 判断某些 geometric samples 的目标是否可达
            rows = build_geometric_evaluation_sets(
                stores=stores,
                metric=metric,
                success_distance_m=args.geo_reachable_gd_m,
                force=args.force_dgeo_eval,
            )
            write_csv(run_dir / "dgeo_reachable_filter.csv", rows)
            print_section("D_GEO Reachable Goal Filter")
            print_table(
                ["Mission", "Source", "Reachable", "Reachable %", "Status"],
                [
                    [
                        row["mission"],
                        row["source_samples"],
                        row["reachable_samples"],
                        row["reachable_percent"],
                        row["status"],
                    ]
                    for row in rows
                ],
            )
            stores = load_mission_stores(dataset_root, missions_csv)
            print_dataset_statistics(stores, run_dir, "Statistics After D_GEO Filtering")

    if args.command == "prepare":
        print(f"\nPrepare-only run finished: {run_dir}")
        return

    # 先从命令行参数构造默认指标设置。如果不是非交互模式，就继续问用户：
    # success GD 阈值是多少
    # goal source 用 raw goal 还是 reference endpoint
    # collision mode 用哪种
    # unknown 是否算碰撞
    # out-of-map 是否算碰撞
    # 这些设置最后都会保存在 MetricOptions 里
    metric_options = metric_options_from_args(args, collision_modes)
    if not args.non_interactive:
        metric_options = [ask_metric_options(metric_options[0])]

    # 支持3个数据集
    # D_TEL: prepared teleop evaluation set
    # D_GEO: prepared geometric evaluation set
    # D_AUG: D_GEO eval + D_TEL eval
    datasets = parse_list(args.datasets)

    # 支持多种方式
    # ptc_limo,limo_D_tel,limo_D_aug,limo_side_cams,limo_side_cams_undistorted,
    # limo_side_cams_sync_history_depth,straight_line,real_world,geometric_planner
    methods = parse_list(args.methods)
    if args.non_interactive:
        datasets = datasets or ["D_TEL", "D_GEO"]
        methods = methods or ["limo_D_tel", "limo_D_aug", "straight_line"]
    if not args.non_interactive:
        datasets = choose_many(
            "Choose datasets to evaluate",
            [
                ("D_TEL", "D_TEL prepared teleop evaluation set"),
                ("D_GEO", "D_GEO reachable evaluation set"),
                ("D_AUG", "D_GEO eval + D_TEL eval"),
            ],
            datasets or ["D_TEL", "D_GEO"],
        )
        methods = choose_many(
            "Choose methods to evaluate",
            [(key, spec.label) for key, spec in method_registry.items()],
            methods or ["limo_D_tel", "limo_D_aug", "limo_side_cams", "straight_line"],
        )

    # 检查 dataset 名称是否合法，并检查某些方法不能用于某些 dataset
    # 比如 D_GEO 不能评估 real_world，因为 real-world path 是 teleop 数据才有意义
    method_specs = [method_registry[method] for method in methods]
    validate_selection(datasets, method_specs)
    # 如果用户选择了 D_TEL / D_GEO / D_AUG，就必须已经准备好对应 eval group。
    # teleop_paths_planner 只在实际选择 Geometric Planner 时才是必需输入。
    ensure_eval_groups_ready_for_selection(datasets, method_specs, stores)
    # 把当前指标设置打印出来，例如：
    # Success GD <= 1.0 m
    # Goal source raw_goal
    # Collision mode footprint_any
    # Fatal fraction threshold 0.03
    # Collision unknown False
    # Collision out-of-map False
    print_metric_settings(
        metric_options,
        map_resolution_m=args.map_resolution_m,
        map_origin_xy=tuple(args.map_origin_xy),
        footprint_extents_m=footprint_extents_m,
        elevation_rotate_k=args.elevation_rotate_k,
    )

    # 调用核心评测函数
    evaluate(
        run_dir=run_dir,
        stores=stores,
        dataset_root=dataset_root,
        datasets=datasets,
        methods=method_specs,
        metric=metric,
        options=metric_options,
        device=args.device,
        torch_hub_dir=torch_hub_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        debug_visualize=args.debug_visualize,
        debug_visualize_max_samples=args.debug_visualize_max_samples,
    )
    print(f"\nRun directory: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Readable interactive open-loop evaluation workflow."
    )
    parser.add_argument(
        "--config-yaml",
        default="",
        help="Optional YAML file with CLI argument defaults. CLI flags override YAML values.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["prepare", "evaluate", "all"],
        default="all",
        help="prepare: stats/filter only; evaluate: evaluate only; all: prepare then evaluate.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Mission dataset root. Defaults to the bundled GrandTour evaluation dataset.",
    )
    parser.add_argument(
        "--missions-csv",
        default=str(DEFAULT_MISSIONS_CSV),
        help=(
            "Path to a mission split CSV. Defaults to the LiMO missions_split.csv; "
            "pass a custom CSV for custom mission names or split statistics."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for run outputs such as summary.csv and per-sample CSV files.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-hub-dir", default=str(DEFAULT_TORCH_HUB_DIR))
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--force-dtel-eval", action="store_true")
    parser.add_argument("--force-dgeo-eval", action="store_true")
    parser.add_argument("--dtel-reachable-gd-m", type=float, default=1.0)
    parser.add_argument("--geo-reachable-gd-m", type=float, default=1.0)
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated datasets: D_TEL,D_GEO,D_AUG.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help=(
            "Comma-separated methods: bev_limo_a,ptc_limo,limo_D_tel,limo_D_aug,limo_side_cams,"
            "limo_side_cams_undistorted,limo_side_cams_sync_history_depth,"
            "straight_line,real_world,geometric_planner."
        ),
    )
    parser.add_argument(
        "--bev-limo-a-weights",
        default="",
        help=(
            "Path to BEV-LIMO-A SafeTensors weights. Defaults to the local "
            "2026-06-25_21-29-28 training run if present."
        ),
    )
    parser.add_argument(
        "--ptc-limo-weights",
        default="",
        help=(
            "Path to PTC-LIMO front-camera weights. Supports .safetensors "
            "exports and Lightning .ckpt checkpoints. Inference uses the "
            "original LiMO inputs only: image_front and goal."
        ),
    )
    parser.add_argument(
        "--limo-d-tel-weights",
        default="",
        help="Path to the original front LiMO D_TEL SafeTensors weights.",
    )
    parser.add_argument(
        "--limo-d-aug-weights",
        default="",
        help="Path to the original front LiMO D_AUG SafeTensors weights.",
    )
    parser.add_argument(
        "--limo-side-cams-weights",
        default="",
        help="Path to the LiMO side-cameras SafeTensors weights.",
    )
    parser.add_argument(
        "--limo-image-subdir",
        default="",
        help=(
            "Override the image subdirectory for LiMO methods "
            "(ptc_limo, limo_D_tel, limo_D_aug, limo_side_cams)."
        ),
    )
    parser.add_argument(
        "--ptc-limo-image-subdir",
        default="",
        help="Override image subdirectory for ptc_limo only. Defaults to images.",
    )
    parser.add_argument(
        "--limo-d-tel-image-subdir",
        default="",
        help="Override image subdirectory for limo_D_tel only.",
    )
    parser.add_argument(
        "--limo-d-aug-image-subdir",
        default="",
        help="Override image subdirectory for limo_D_aug only.",
    )
    parser.add_argument(
        "--limo-side-cams-image-subdir",
        default="",
        help="Override image subdirectory for limo_side_cams only.",
    )
    parser.add_argument(
        "--max-samples", # 只评估最多多少个样本，适合 smoke test
        type=int,
        default=None,
        help="Evaluation-only sample cap for smoke tests; prepare always writes complete eval sets.",
    )
    parser.add_argument("--success-distance-m", type=float, default=1.0)
    parser.add_argument(
        "--map-resolution-m",
        type=float,
        default=0.04,
        help="Elevation-map resolution in meters per cell. Must match the dataset map.",
    )
    parser.add_argument(
        "--map-origin-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(-4.0, -4.0),
        help="Elevation-map origin in meters. Must match the dataset map coordinate frame.",
    )
    parser.add_argument(
        "--elevation-rotate-k",
        type=int,
        choices=(0, 2),
        default=0,
        help=(
            "Rotate every elevation map counter-clockwise by k*90 degrees before "
            "collision/GDF/planning. 0 preserves Table II behavior; 2 corrects a "
            "180-degree map-frame mismatch."
        ),
    )
    parser.add_argument(
        "--footprint-front-m",
        type=float,
        default=None,
        help="Robot footprint extent forward from the base frame origin, in meters.",
    )
    parser.add_argument(
        "--footprint-rear-m",
        type=float,
        default=None,
        help="Robot footprint extent rearward from the base frame origin, in meters.",
    )
    parser.add_argument(
        "--footprint-left-m",
        type=float,
        default=None,
        help="Robot footprint extent left of the base frame origin, in meters.",
    )
    parser.add_argument(
        "--footprint-right-m",
        type=float,
        default=None,
        help="Robot footprint extent right of the base frame origin, in meters.",
    )
    parser.add_argument(
        "--goal-source",
        choices=["raw_goal", "reference_endpoint"],
        default="raw_goal",
    )
    parser.add_argument(
        "--collision-mode",
        choices=["footprint_any", "footprint_fraction", "centerline"],
        default="footprint_any",
        help=(
            "Single collision rule. Kept for backward compatibility; ignored when "
            "--collision-modes is provided."
        ),
    )
    parser.add_argument(
        "--collision-modes",
        default="",
        help=(
            "Comma-separated collision rules evaluated from the same predicted paths, "
            "for example footprint_any,footprint_fraction. If empty, --collision-mode "
            "is used."
        ),
    )
    parser.add_argument("--fatal-fraction-threshold", type=float, default=0.03)
    parser.add_argument("--collision-unknown", action="store_true") # store_true means 不写就是 False，写了就是 True
    parser.add_argument("--collision-out-of-map", action="store_true")
    parser.add_argument(
        "--debug-visualize",
        action="store_true",
        help=(
            "Save local PNG diagnostics for collision samples: elevation, fatal/unknown "
            "cells, evaluated path, and discrete footprint cells."
        ),
    )
    parser.add_argument(
        "--debug-visualize-max-samples",
        type=int,
        default=20,
        help=(
            "Maximum collision diagnostic PNGs to save for each dataset/method pair; "
            "set 0 to save every collision sample."
        ),
    )
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-yaml", default="")
    config_args, _ = config_parser.parse_known_args()
    if config_args.config_yaml:
        apply_yaml_defaults(parser, Path(config_args.config_yaml))
    return parser.parse_args()


def apply_yaml_defaults(parser: argparse.ArgumentParser, path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")

    actions = {action.dest: action for action in parser._actions}
    for key, value in config.items():
        dest = key.replace("-", "_")
        if dest not in actions:
            raise ValueError(f"Unknown YAML config key: {key}")
        actions[dest].default = value


def print_dataset_statistics(stores, run_dir: Path, title: str) -> None:
    """统计并打印数据集情况"""
    """["Dataset", "Split", "#Samples", "Length [m]", "Time [h]", "Avg. vel. [m/s]"]"""
    rows = collect_dataset_stats(stores)
    write_csv(run_dir / "dataset_statistics.csv", rows)
    print_section(title)
    print_table(
        ["Dataset", "Split", "#Samples", "Length [m]", "Time [h]", "Avg. vel. [m/s]"],
        [
            [
                row["dataset"],
                row["split"],
                row["#samples"],
                row["length_m"],
                row["time_h"],
                row["avg_vel_mps"],
            ]
            for row in rows
        ],
    )
    if all(row["#samples"] == 0 for row in rows if row["split"] == "train"):
        print("\nNote: local evaluation folder currently contains no train split missions.")


def ensure_eval_groups_ready_for_selection(
    datasets: list[str],
    methods: list,
    stores,
) -> None:
    """检查用户选择的数据集所需 eval zarr group 是否已经准备好。"""
    needs_dtel = bool(set(datasets) & {"D_TEL", "D_AUG"})
    needs_dgeo = bool(set(datasets) & {"D_GEO", "D_AUG"})
    needs_dtel_planner = needs_dtel and any(
        method.id == "geometric_planner" for method in methods
    )

    # 正式 GrandTour 评测按 missions CSV 的 test split 选择 mission。
    # 自定义数据集通常没有该 CSV 中的名字，此时回退到所有已经提供 eval group 的 mission。
    test_stores = [store for store in stores if store.split == "test"]
    candidate_stores = test_stores or [
        store
        for store in stores
        if store.has_group("tel_eval") or store.has_group("geo_eval")
    ]
    if not candidate_stores:
        raise FileNotFoundError(
            "No evaluation zarr groups found. Expected teleop_paths_evalution "
            "and/or geometric_paths_evalution under <dataset-root>/<mission>/data."
        )

    missing_dtel = []
    missing_dgeo = []
    missing_dtel_planner = []
    for store in candidate_stores:
        if needs_dtel and not store.has_group("tel_eval"):
            missing_dtel.append(store.mission)
        if needs_dtel_planner and not store.has_group("tel_planner"):
            missing_dtel_planner.append(store.mission)
        if needs_dgeo and not store.has_group("geo_eval"):
            missing_dgeo.append(store.mission)
    if missing_dtel or missing_dtel_planner or missing_dgeo:
        parts = []
        if missing_dtel:
            parts.append(
                "D_TEL evaluation set is missing teleop_paths_evalution for: "
                f"{', '.join(missing_dtel)}"
            )
        if missing_dtel_planner:
            parts.append(
                "Geometric Planner on D_TEL requires teleop_paths_planner for: "
                f"{', '.join(missing_dtel_planner)}"
            )
        if missing_dgeo:
            parts.append(
                "D_GEO evaluation set is missing geometric_paths_evalution for: "
                f"{', '.join(missing_dgeo)}"
            )
        raise FileNotFoundError("Missing required evaluation zarr group(s). " + " | ".join(parts))


def collision_modes_from_args(args: argparse.Namespace) -> list[str]:
    """解析单个或多个 collision mode，并保持用户给定顺序。"""
    raw_modes = parse_list(args.collision_modes) if args.collision_modes else [args.collision_mode]
    known_modes = {"footprint_any", "footprint_fraction", "centerline"}
    modes: list[str] = []
    for mode in raw_modes:
        if mode not in known_modes:
            raise ValueError(
                f"Unknown collision mode: {mode}. Choose from: "
                f"{', '.join(sorted(known_modes))}."
            )
        if mode not in modes:
            modes.append(mode)
    return modes


def metric_options_from_args(args: argparse.Namespace, collision_modes: list[str]) -> list[MetricOptions]:
    """把 argparse 的参数包装成 MetricOptions"""
    return [
        MetricOptions(
            success_distance_m=args.success_distance_m,
            goal_source=args.goal_source,
            collision_mode=collision_mode,
            fatal_fraction_threshold=args.fatal_fraction_threshold,
            collision_unknown=args.collision_unknown,
            collision_out_of_map=args.collision_out_of_map,
        )
        for collision_mode in collision_modes
    ]


def footprint_extents_from_args(
    args: argparse.Namespace,
) -> tuple[float, float, float, float] | None:
    """将具名的前后左右范围转换为 (front, rear, left, right)。"""
    values = (
        args.footprint_front_m,
        args.footprint_rear_m,
        args.footprint_left_m,
        args.footprint_right_m,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "Set all four footprint extents together: --footprint-front-m, "
            "--footprint-rear-m, --footprint-left-m, --footprint-right-m."
        )
    if any(value <= 0.0 for value in values):
        raise ValueError("All footprint extents must be positive.")
    return tuple(float(value) for value in values)


def ask_metric_options(defaults: MetricOptions) -> MetricOptions:
    """交互式询问指标设置"""
    print_section("Metric Settings")
    # 成功阈值，比如最终 GD 距离小于多少米算到达
    success_distance = ask_float("Success GD threshold in meters", defaults.success_distance_m)
    # 目标来源： 
    # raw_goal：使用 zarr 里原始 goal
    # reference_endpoint：使用参考路径的终点作为 metric goal
    goal_source = choose_one(
        "Metric goal source",
        [
            ("raw_goal", "Use zarr raw goal"),
            ("reference_endpoint", "Use stored reference path endpoint"),
        ],
        defaults.goal_source,
    )
    # 碰撞模式：
    # footprint_any：机器人 footprint 任何 fatal cell 都算碰撞
    # footprint_fraction：fatal footprint 比例超过阈值才算碰撞
    # centerline：只看路径中心线是否碰撞
    collision_mode = choose_one(
        "Collision mode",
        [
            ("footprint_any", "Any fatal footprint cell"),
            ("footprint_fraction", "Fatal footprint fraction threshold"),
            ("centerline", "Centerline fatal cell only"),
        ],
        defaults.collision_mode,
    )
    threshold = defaults.fatal_fraction_threshold
    if collision_mode == "footprint_fraction":
        threshold = ask_float("Fatal footprint fraction threshold", threshold)
    # 是否把未知区域、地图外区域算作碰撞
    collision_unknown = ask_yes_no("Count unknown traversability as collision", defaults.collision_unknown)
    collision_out_of_map = ask_yes_no("Count out-of-map footprint as collision", defaults.collision_out_of_map)
    return MetricOptions(
        success_distance_m=success_distance,
        goal_source=goal_source,
        collision_mode=collision_mode,
        fatal_fraction_threshold=threshold,
        collision_unknown=collision_unknown,
        collision_out_of_map=collision_out_of_map,
    )


def print_metric_settings(
    options: MetricOptions | list[MetricOptions],
    map_resolution_m: float,
    map_origin_xy: tuple[float, float],
    footprint_extents_m: tuple[float, float, float, float] | None,
    elevation_rotate_k: int,
) -> None:
    """最终使用的 metric 配置打印成表格，方便用户确认"""
    option_list = options if isinstance(options, list) else [options]
    first_option = option_list[0]
    collision_modes = ", ".join(option.collision_mode for option in option_list)
    print_section("Using Metric")
    print_table(
        ["Setting", "Value"],
        [
            ["Success GD", f"<= {first_option.success_distance_m} m"],
            ["Goal source", first_option.goal_source],
            ["Collision mode(s)", collision_modes],
            ["Fatal fraction threshold", first_option.fatal_fraction_threshold],
            ["Collision unknown", first_option.collision_unknown],
            ["Collision out-of-map", first_option.collision_out_of_map],
            ["Map resolution", f"{map_resolution_m} m/cell"],
            ["Map origin", map_origin_xy],
            ["Elevation rotation", f"{elevation_rotate_k * 90} degrees (k={elevation_rotate_k})"],
            [
                "Robot footprint",
                (
                    "build.yaml default"
                    if footprint_extents_m is None
                    else (
                        f"front={footprint_extents_m[0]} m, rear={footprint_extents_m[1]} m, "
                        f"left={footprint_extents_m[2]} m, right={footprint_extents_m[3]} m"
                    )
                ),
            ],
        ],
    )


def validate_selection(datasets: list[str], methods: list) -> None:
    """有效性检查"""
    # dataset 名称必须合法
    known_datasets = {"D_TEL", "D_GEO", "D_AUG"}
    unknown = [dataset for dataset in datasets if dataset not in known_datasets]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    for dataset in datasets:
        dataset_subsets(dataset)
    # 限制 real_world 不能用于 D_GEO
    if "D_GEO" in datasets:
        for method in methods:
            if method.id == "real_world":
                raise ValueError("D_GEO does not support Real-World Paths.")


def parse_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [token.strip() for token in value.split(",") if token.strip()]


def write_run_args(run_dir: Path, args: argparse.Namespace) -> None:
    with (run_dir / "run_args.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True, allow_unicode=False)
