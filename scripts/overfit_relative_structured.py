#!/usr/bin/env python
"""Overfit RelativeStructured LiMo on a fixed, tiny real-data subset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision import transforms

from limo.src.dataset.limo_datset import MissionDataset
from limo.src.models.components.se2_dynamics import path_to_body_motion, wrap_angle

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_NAMES = (
    "total",
    "xy",
    "yaw",
    "motion",
    "endpoint",
    "progress",
    "smooth",
    "ade",
    "fde",
)
PLANNER_LOWER_BOUNDS = (-0.2, -0.2, -0.8)
PLANNER_UPPER_BOUNDS = (1.0, 0.2, 0.8)
VY_THRESHOLDS = (0.2, 0.3, 0.4, 0.5)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/robot-device/yangyujie/EXP2/LIMO_DATASET"),
    )
    parser.add_argument("--mission", default="2024-11-02-21-12-51")
    parser.add_argument("--samples", type=int, default=16, choices=range(16, 33))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--min-steps", type=int, default=300)
    parser.add_argument("--motion-stability-window", type=int, default=5)
    parser.add_argument("--motion-stability-rel-range", type=float, default=0.03)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "logs" / "overfit_relative_structured",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_subset(args: argparse.Namespace) -> tuple[Dict[str, torch.Tensor], list[int]]:
    transform = transforms.Compose(
        [transforms.Resize((308, 476)), transforms.ToTensor()]
    )
    dataset = MissionDataset(
        "geo", args.dataset_root, args.mission, transform, with_side_cams=False
    )
    paths = np.asarray(dataset.z["path"])
    origin = np.zeros((len(paths), 1, 2), dtype=paths.dtype)
    points = np.concatenate([origin, paths[..., :2]], axis=1)
    total_lengths = np.linalg.norm(np.diff(points, axis=1), axis=-1).sum(axis=1)
    valid = np.flatnonzero(np.isfinite(paths).all(axis=(1, 2)) & (total_lengths >= 0.5))
    if len(valid) < args.samples:
        raise RuntimeError(
            f"Only {len(valid)} finite paths with length >= 0.5 m are available"
        )

    positions = np.linspace(0, len(valid) - 1, args.samples, dtype=np.int64)
    selected = valid[positions].tolist()
    samples = [dataset[index] for index in selected]
    batch = {
        key: torch.stack([sample[key] for sample in samples])
        for key in ("image_front", "goal", "path")
    }
    return batch, selected


@torch.no_grad()
def evaluate(net, loss_fn, batch: Dict[str, torch.Tensor], step: int):
    net.eval()
    pred_path, pred_motion = net.forward_with_motion(batch)
    components = loss_fn.compute_components(pred_path, batch["path"], pred_motion)
    position_error = torch.linalg.vector_norm(
        pred_path[..., :2] - batch["path"][..., :2], dim=-1
    )
    maxima = pred_motion.abs().amax(dim=(0, 1))
    extreme_threshold = loss_fn.motion_scales.to(pred_motion) * 5.0
    extreme_fraction = (pred_motion.abs() > extreme_threshold).float().mean()
    lower_bounds = pred_motion.new_tensor(PLANNER_LOWER_BOUNDS)
    upper_bounds = pred_motion.new_tensor(PLANNER_UPPER_BOUNDS)
    bound_violations = (pred_motion < lower_bounds) | (pred_motion > upper_bounds)
    target_motion = path_to_body_motion(batch["path"], loss_fn.dt)
    pred_abs_vy = pred_motion[..., 1].abs().flatten()
    target_abs_vy = target_motion[..., 1].abs().flatten()
    row = {name: float(value.detach().cpu()) for name, value in components.items()}
    row.update(
        {
            "step": step,
            "ade": float(position_error.mean().cpu()),
            "fde": float(position_error[:, -1].mean().cpu()),
            "vx_abs_max": float(maxima[0].cpu()),
            "vy_abs_max": float(maxima[1].cpu()),
            "wz_abs_max": float(maxima[2].cpu()),
            "extreme_control_fraction": float(extreme_fraction.cpu()),
            "planner_bound_violation_fraction": float(
                bound_violations.float().mean().cpu()
            ),
            "outputs_finite": bool(
                torch.isfinite(pred_path).all() and torch.isfinite(pred_motion).all()
            ),
        }
    )
    for prefix, values in (("pred_vy", pred_abs_vy), ("target_vy", target_abs_vy)):
        quantiles = torch.quantile(values, values.new_tensor([0.5, 0.95, 0.99]))
        row[f"{prefix}_abs_p50"] = float(quantiles[0].cpu())
        row[f"{prefix}_abs_p95"] = float(quantiles[1].cpu())
        row[f"{prefix}_abs_p99"] = float(quantiles[2].cpu())
        row[f"{prefix}_abs_max"] = float(values.max().cpu())
        for threshold in VY_THRESHOLDS:
            suffix = str(threshold).replace(".", "p")
            row[f"{prefix}_abs_gt_{suffix}"] = float(
                (values > threshold).float().mean().cpu()
            )
    return row, pred_path.detach().cpu(), pred_motion.detach().cpu()


def save_metrics(rows: list[dict], output_dir: Path) -> None:
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory_plot(
    target: torch.Tensor,
    initial: torch.Tensor,
    final: torch.Tensor,
    output_dir: Path,
) -> None:
    count = min(4, len(target))
    fig, axes = plt.subplots(count, 2, figsize=(10, 4 * count), squeeze=False)
    for row in range(count):
        for col, (title, prediction) in enumerate(
            (("Before training", initial), ("After training", final))
        ):
            ax = axes[row, col]
            ax.plot(target[row, :, 0], target[row, :, 1], "k-", label="teacher")
            ax.plot(
                prediction[row, :, 0],
                prediction[row, :, 1],
                "r--",
                label="prediction",
            )
            ax.scatter([0.0], [0.0], c="tab:green", marker="o", label="start")
            ax.set_title(f"sample {row}: {title}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=160)
    plt.close(fig)


def save_metric_plot(rows: list[dict], output_dir: Path) -> None:
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in ("xy", "yaw", "motion", "endpoint", "progress", "smooth"):
        axes[0].plot(steps, [row[name] for row in rows], marker="o", label=name)
    axes[0].set_title("Structured loss components")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for name in ("ade", "fde"):
        axes[1].plot(steps, [row[name] for row in rows], marker="o", label=name)
    axes[1].set_title("Trajectory position error")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("metres")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metric_curves.png", dpi=160)
    plt.close(fig)


def save_vy_trend_plot(rows: list[dict], output_dir: Path) -> None:
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for statistic in ("p50", "p95", "p99", "max"):
        axes[0].plot(
            steps,
            [row[f"pred_vy_abs_{statistic}"] for row in rows],
            marker="o",
            label=f"pred {statistic}",
        )
        axes[0].axhline(
            rows[0][f"target_vy_abs_{statistic}"],
            linestyle="--",
            alpha=0.5,
            label=f"target {statistic}",
        )
    axes[0].set_title("Absolute lateral velocity statistics")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("m/s")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2)

    for threshold in VY_THRESHOLDS:
        suffix = str(threshold).replace(".", "p")
        axes[1].plot(
            steps,
            [row[f"pred_vy_abs_gt_{suffix}"] for row in rows],
            marker="o",
            label=f"|pred vy| > {threshold}",
        )
        axes[1].axhline(
            rows[0][f"target_vy_abs_gt_{suffix}"],
            linestyle="--",
            alpha=0.5,
        )
    axes[1].set_title("Lateral velocity tail fractions")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "vy_trends.png", dpi=160)
    plt.close(fig)


def save_vy_diagnostic_plots(
    target_motion: torch.Tensor,
    pred_motion: torch.Tensor,
    output_dir: Path,
) -> None:
    target_vy = target_motion[..., 1].flatten().numpy()
    pred_vy = pred_motion[..., 1].flatten().numpy()
    residual = pred_vy - target_vy

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(target_vy, pred_vy, s=12, alpha=0.4)
    limit = max(float(np.abs(target_vy).max()), float(np.abs(pred_vy).max()))
    axes[0].plot([-limit, limit], [-limit, limit], "k--", label="pred = target")
    axes[0].axhline(0.2, color="tab:red", linestyle=":")
    axes[0].axhline(-0.2, color="tab:red", linestyle=":")
    axes[0].set_xlabel("target vy [m/s]")
    axes[0].set_ylabel("predicted vy [m/s]")
    axes[0].set_title("Predicted versus target lateral velocity")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].hist(residual, bins=50, alpha=0.8)
    axes[1].axvline(0.0, color="k", linestyle="--")
    axes[1].set_xlabel("predicted vy - target vy [m/s]")
    axes[1].set_ylabel("count")
    axes[1].set_title("Lateral velocity residuals")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "vy_scatter_and_residual.png", dpi=160)
    plt.close(fig)


def save_large_vy_sample_plot(
    target_path: torch.Tensor,
    pred_path: torch.Tensor,
    target_motion: torch.Tensor,
    pred_motion: torch.Tensor,
    selected: list[int],
    output_dir: Path,
) -> None:
    counts = (pred_motion[..., 1].abs() > 0.2).sum(dim=1)
    sample_slots = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
    if not sample_slots:
        return

    time = torch.arange(pred_motion.shape[1]).numpy() * 0.1 + 0.1
    fig, axes = plt.subplots(
        len(sample_slots), 4, figsize=(18, 4 * len(sample_slots)), squeeze=False
    )
    for row, slot in enumerate(sample_slots):
        axes[row, 0].plot(
            target_path[slot, :, 0], target_path[slot, :, 1], "k-", label="target"
        )
        axes[row, 0].plot(
            pred_path[slot, :, 0], pred_path[slot, :, 1], "r--", label="pred"
        )
        axes[row, 0].set_aspect("equal", adjustable="box")
        axes[row, 0].set_title(f"dataset index {selected[slot]}: path")
        axes[row, 0].legend()

        axes[row, 1].plot(time, target_motion[slot, :, 1], "k-", label="target vy")
        axes[row, 1].plot(time, pred_motion[slot, :, 1], "r--", label="pred vy")
        axes[row, 1].axhline(0.2, color="tab:red", linestyle=":")
        axes[row, 1].axhline(-0.2, color="tab:red", linestyle=":")
        axes[row, 1].set_title("lateral velocity")
        axes[row, 1].legend()

        axes[row, 2].plot(time, target_motion[slot, :, 2], "k-", label="target wz")
        axes[row, 2].plot(time, pred_motion[slot, :, 2], "r--", label="pred wz")
        axes[row, 2].set_title("yaw rate")
        axes[row, 2].legend()

        xy_error = torch.linalg.vector_norm(
            pred_path[slot, :, :2] - target_path[slot, :, :2], dim=-1
        )
        yaw_error = wrap_angle(
            pred_path[slot, :, 2] - target_path[slot, :, 2]
        ).abs()
        axes[row, 3].plot(time, xy_error, label="XY error [m]")
        axes[row, 3].plot(time, yaw_error, label="yaw error [rad]")
        axes[row, 3].set_title("tracking errors")
        axes[row, 3].legend()

        for ax in axes[row]:
            ax.grid(alpha=0.25)
            if ax is not axes[row, 0]:
                ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(output_dir / "large_vy_samples.png", dpi=160)
    plt.close(fig)


def analyze_large_lateral_events(
    target_path: torch.Tensor,
    pred_path: torch.Tensor,
    target_motion: torch.Tensor,
    pred_motion: torch.Tensor,
    selected: list[int],
    dt: float,
    output_dir: Path,
) -> dict:
    previous_pred = torch.cat(
        [torch.zeros_like(pred_path[:, :1]), pred_path[:, :-1]], dim=1
    )
    delta_to_target = target_path[..., :2] - previous_pred[..., :2]
    previous_yaw = previous_pred[..., 2]
    corrective_vy = (
        -torch.sin(previous_yaw) * delta_to_target[..., 0]
        + torch.cos(previous_yaw) * delta_to_target[..., 1]
    ) / dt

    endpoint_delta = target_path[:, -1:, :2] - previous_pred[..., :2]
    endpoint_vy = -torch.sin(previous_yaw) * endpoint_delta[..., 0] + torch.cos(
        previous_yaw
    ) * endpoint_delta[..., 1]
    residual_vy = pred_motion[..., 1] - target_motion[..., 1]
    xy_error = torch.linalg.vector_norm(
        pred_path[..., :2] - target_path[..., :2], dim=-1
    )
    yaw_error = wrap_angle(pred_path[..., 2] - target_path[..., 2]).abs()
    final_error = xy_error[:, -1]
    large = pred_motion[..., 1].abs() > 0.2

    events = []
    for sample_slot, time_step in large.nonzero(as_tuple=False).tolist():
        residual = float(residual_vy[sample_slot, time_step])
        correction = float(corrective_vy[sample_slot, time_step])
        endpoint_correction = float(endpoint_vy[sample_slot, time_step])
        events.append(
            {
                "sample_slot": sample_slot,
                "dataset_index": selected[sample_slot],
                "time_step": time_step,
                "time_s": (time_step + 1) * dt,
                "pred_vx": float(pred_motion[sample_slot, time_step, 0]),
                "pred_vy": float(pred_motion[sample_slot, time_step, 1]),
                "target_vy": float(target_motion[sample_slot, time_step, 1]),
                "vy_residual": residual,
                "pred_wz": float(pred_motion[sample_slot, time_step, 2]),
                "target_wz": float(target_motion[sample_slot, time_step, 2]),
                "xy_error_m": float(xy_error[sample_slot, time_step]),
                "yaw_error_rad": float(yaw_error[sample_slot, time_step]),
                "sample_final_error_m": float(final_error[sample_slot]),
                "corrective_vy_to_teacher": correction,
                "residual_matches_local_correction": residual * correction > 0.0,
                "endpoint_lateral_error_body": endpoint_correction,
                "residual_matches_endpoint_correction": residual * endpoint_correction
                > 0.0,
            }
        )

    fieldnames = list(events[0].keys()) if events else []
    with (output_dir / "large_vy_events.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

    counts_by_sample = {
        str(slot): int(large[slot].sum()) for slot in range(len(selected))
    }
    counts_by_time_bin = {
        f"{start}-{start + 9}": int(large[:, start : start + 10].sum())
        for start in range(0, pred_motion.shape[1], 10)
    }
    large_wz = pred_motion[..., 2].abs()[large]
    all_wz = pred_motion[..., 2].abs()
    analysis = {
        "event_count": len(events),
        "event_fraction": float(large.float().mean()),
        "counts_by_sample_slot": counts_by_sample,
        "dataset_indices_by_sample_slot": {
            str(slot): index for slot, index in enumerate(selected)
        },
        "counts_by_time_bin": counts_by_time_bin,
        "fraction_target_also_gt_0p2": float(
            (target_motion[..., 1].abs()[large] > 0.2).float().mean()
        )
        if large.any()
        else 0.0,
        "fraction_residual_matches_local_correction": float(
            ((residual_vy[large] * corrective_vy[large]) > 0.0).float().mean()
        )
        if large.any()
        else 0.0,
        "fraction_residual_matches_endpoint_correction": float(
            ((residual_vy[large] * endpoint_vy[large]) > 0.0).float().mean()
        )
        if large.any()
        else 0.0,
        "mean_abs_wz_large_vy": float(large_wz.mean()) if large.any() else 0.0,
        "mean_abs_wz_all": float(all_wz.mean()),
        "mean_yaw_error_large_vy": float(yaw_error[large].mean())
        if large.any()
        else 0.0,
        "mean_yaw_error_all": float(yaw_error.mean()),
        "mean_xy_error_large_vy": float(xy_error[large].mean())
        if large.any()
        else 0.0,
        "mean_xy_error_all": float(xy_error.mean()),
    }
    with (output_dir / "large_vy_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    return analysis


def summarize(
    rows: list[dict],
    selected: list[int],
    target_motion: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
    completed_steps: int,
    stopped_for_motion_stability: bool,
    large_vy_analysis: dict,
) -> dict:
    initial, final = rows[0], rows[-1]
    slopes = {}
    for name in METRIC_NAMES:
        slopes[name] = float(
            np.polyfit(
                [row["step"] for row in rows],
                [row[name] for row in rows],
                deg=1,
            )[0]
        )

    loss_decrease = {
        name: {
            "initial": initial[name],
            "final": final[name],
            "ratio": final[name] / max(initial[name], 1e-12),
            "slope": slopes[name],
            "decreased": final[name] < initial[name] and slopes[name] < 0.0,
        }
        for name in ("xy", "yaw", "motion", "endpoint", "progress", "smooth")
    }
    target_max = target_motion.abs().amax(dim=(0, 1))
    summary = {
        "settings": {
            "samples": args.samples,
            "requested_steps": args.steps,
            "completed_steps": completed_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": str(choose_device(args.device)),
            "selected_indices": selected,
            "stopped_for_motion_stability": stopped_for_motion_stability,
        },
        "losses": loss_decrease,
        "ade": {
            "initial": initial["ade"],
            "final": final["ade"],
            "ratio": final["ade"] / max(initial["ade"], 1e-12),
            "slope": slopes["ade"],
        },
        "fde": {
            "initial": initial["fde"],
            "final": final["fde"],
            "ratio": final["fde"] / max(initial["fde"], 1e-12),
            "slope": slopes["fde"],
        },
        "controls": {
            "target_abs_max": target_max.tolist(),
            "prediction_abs_max": [
                final["vx_abs_max"],
                final["vy_abs_max"],
                final["wz_abs_max"],
            ],
            "extreme_fraction": final["extreme_control_fraction"],
            "planner_bounds": {
                "lower": PLANNER_LOWER_BOUNDS,
                "upper": PLANNER_UPPER_BOUNDS,
            },
            "final_planner_bound_violation_fraction": final[
                "planner_bound_violation_fraction"
            ],
            "all_evaluations_finite": all(row["outputs_finite"] for row in rows),
            "vy_statistics": {
                prefix: {
                    statistic: final[f"{prefix}_vy_abs_{statistic}"]
                    for statistic in ("p50", "p95", "p99", "max")
                }
                for prefix in ("pred", "target")
            },
            "vy_tail_fractions": {
                prefix: {
                    str(threshold): final[
                        f"{prefix}_vy_abs_gt_{str(threshold).replace('.', 'p')}"
                    ]
                    for threshold in VY_THRESHOLDS
                }
                for prefix in ("pred", "target")
            },
        },
        "large_vy_analysis": large_vy_analysis,
        "checks": {
            "all_six_losses_decreased": all(
                result["decreased"] for result in loss_decrease.values()
            ),
            "ade_decreased_by_half": final["ade"] < initial["ade"] * 0.5,
            "fde_decreased_by_half": final["fde"] < initial["fde"] * 0.5,
            "finite_outputs": all(row["outputs_finite"] for row in rows),
            "no_exploding_controls": final["extreme_control_fraction"] == 0.0,
            "within_planner_control_bounds": final[
                "planner_bound_violation_fraction"
            ]
            == 0.0,
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0:
        raise ValueError("steps, batch-size, and eval-every must be positive")
    if args.batch_size > args.samples:
        raise ValueError("batch-size cannot exceed samples")
    if args.motion_stability_window < 2:
        raise ValueError("motion-stability-window must be at least 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch, selected = load_subset(args)
    batch = {key: value.to(device) for key, value in batch.items()}

    repo_root = Path(__file__).resolve().parents[1]
    model_cfg = OmegaConf.load(repo_root / "limo/configs/model/relative_structured.yaml")
    net = instantiate(model_cfg.net).to(device)
    loss_fn = instantiate(model_cfg.loss).to(device)
    trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)

    rows = []
    initial_row, initial_prediction, _ = evaluate(net, loss_fn, batch, step=0)
    rows.append(initial_row)
    print(json.dumps(initial_row, sort_keys=True))

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(args.samples, generator=generator)
    cursor = 0
    completed_steps = 0
    stopped_for_motion_stability = False
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > args.samples:
            order = torch.randperm(args.samples, generator=generator)
            cursor = 0
        indices = order[cursor : cursor + args.batch_size].to(device)
        cursor += args.batch_size
        mini_batch = {key: value[indices] for key, value in batch.items()}

        net.train()
        pred_path, pred_motion = net.forward_with_motion(mini_batch)
        components = loss_fn.compute_components(
            pred_path, mini_batch["path"], pred_motion
        )
        if not torch.isfinite(components["total"]):
            raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        components["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            row, _, _ = evaluate(net, loss_fn, batch, step=step)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
            if (
                not args.disable_early_stop
                and step >= args.min_steps
                and len(rows) >= args.motion_stability_window
            ):
                recent = np.array(
                    [row["motion"] for row in rows[-args.motion_stability_window :]]
                )
                relative_range = float(
                    (recent.max() - recent.min()) / max(recent.mean(), 1e-12)
                )
                if relative_range <= args.motion_stability_rel_range:
                    stopped_for_motion_stability = True
                    completed_steps = step
                    print(
                        "Motion loss stabilized: "
                        f"relative range={relative_range:.6f} over "
                        f"{args.motion_stability_window} evaluations"
                    )
                    break
        completed_steps = step

    final_row, final_prediction, final_motion = evaluate(
        net, loss_fn, batch, step=completed_steps
    )
    rows[-1] = final_row
    target_path_cpu = batch["path"].detach().cpu()
    target_motion = path_to_body_motion(target_path_cpu, loss_fn.dt)
    large_vy_analysis = analyze_large_lateral_events(
        target_path_cpu,
        final_prediction,
        target_motion,
        final_motion,
        selected,
        loss_fn.dt,
        args.output_dir,
    )

    save_metrics(rows, args.output_dir)
    save_metric_plot(rows, args.output_dir)
    save_vy_trend_plot(rows, args.output_dir)
    save_vy_diagnostic_plots(target_motion, final_motion, args.output_dir)
    save_large_vy_sample_plot(
        target_path_cpu,
        final_prediction,
        target_motion,
        final_motion,
        selected,
        args.output_dir,
    )
    save_trajectory_plot(
        target_path_cpu, initial_prediction, final_prediction, args.output_dir
    )
    summary = summarize(
        rows,
        selected,
        target_motion,
        args,
        args.output_dir,
        completed_steps,
        stopped_for_motion_stability,
        large_vy_analysis,
    )
    torch.save(
        {
            "selected_indices": selected,
            "target_path": target_path_cpu,
            "predicted_path": final_prediction,
            "predicted_motion": final_motion,
        },
        args.output_dir / "predictions.pt",
    )
    print(json.dumps(summary["checks"], sort_keys=True))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
