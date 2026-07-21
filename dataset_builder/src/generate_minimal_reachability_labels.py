from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import zarr
from numcodecs import Blosc
from omegaconf import OmegaConf

from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import (
    ReachabilityState,
    STATE_NAMES,
    TeacherAResult,
    build_teacher_a,
)
from dataset_builder.reachability.traversability import compute_limo_traversability


LABEL_UNKNOWN = np.uint8(0)
LABEL_BLOCKED = np.uint8(1)
LABEL_DISCONNECTED_FREE = np.uint8(2)
LABEL_REACHABLE = np.uint8(3)
IGNORE_INDEX = np.uint8(255)
VALID_LABELS = frozenset((0, 1, 2, 3, 255))
TARGET_X_RANGE_M = (0.0, 4.0)
TARGET_Y_RANGE_M = (-3.0, 3.0)
TARGET_RESOLUTION_M = 0.1
TARGET_SHAPE = (40, 60)
BODY_X_RANGE_M = (-0.55, 0.55)
BODY_Y_RANGE_M = (-0.26, 0.26)


def teacher_state_to_four_state(state: np.ndarray) -> np.ndarray:
    """Map the final Teacher-A state enum image to uint8 four-state labels."""
    source = np.asarray(state)
    labels = np.full(source.shape, IGNORE_INDEX, dtype=np.uint8)
    labels[source == int(ReachabilityState.UNKNOWN_CENTER)] = LABEL_UNKNOWN
    labels[source == int(ReachabilityState.UNKNOWN_FOOTPRINT)] = LABEL_UNKNOWN
    labels[source == int(ReachabilityState.LOCALLY_BLOCKED)] = LABEL_BLOCKED
    labels[source == int(ReachabilityState.CLEARANCE_BLOCKED)] = LABEL_BLOCKED
    labels[source == int(ReachabilityState.TRAVERSABLE_BUT_DISCONNECTED)] = LABEL_DISCONNECTED_FREE
    labels[source == int(ReachabilityState.REACHABLE)] = LABEL_REACHABLE
    return labels


def target_bev_centers() -> tuple[np.ndarray, np.ndarray]:
    """Return target-cell center coordinates with axes (x-forward, y-left)."""
    x = TARGET_X_RANGE_M[0] + (np.arange(TARGET_SHAPE[0]) + 0.5) * TARGET_RESOLUTION_M
    y = TARGET_Y_RANGE_M[0] + (np.arange(TARGET_SHAPE[1]) + 0.5) * TARGET_RESOLUTION_M
    return np.meshgrid(x, y, indexing="ij")


def sample_state_to_forward_bev(
    state: np.ndarray, geometry: MapGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest/index sample final Teacher-A states into [40,60] target cells."""
    source_state = np.asarray(state)
    if source_state.shape != (geometry.height, geometry.width):
        raise ValueError(f"state shape {source_state.shape} != geometry {(geometry.height, geometry.width)}")
    source = teacher_state_to_four_state(source_state)
    xx, yy = target_bev_centers()
    points = np.column_stack((xx.ravel(), yy.ravel()))
    indices = geometry.world_to_map_idx(points)
    valid = geometry.valid_indices(indices)
    labels = np.full(points.shape[0], IGNORE_INDEX, dtype=np.uint8)
    labels[valid] = source[indices[valid, 0], indices[valid, 1]].astype(np.uint8)
    labels = labels.reshape(TARGET_SHAPE)
    return labels, valid.reshape(TARGET_SHAPE)


def body_ignore_mask() -> np.ndarray:
    xx, yy = target_bev_centers()
    return (
        (xx >= max(TARGET_X_RANGE_M[0], BODY_X_RANGE_M[0]))
        & (xx < min(TARGET_X_RANGE_M[1], BODY_X_RANGE_M[1]))
        & (yy >= BODY_Y_RANGE_M[0])
        & (yy <= BODY_Y_RANGE_M[1])
    )


def make_four_state_label(result: TeacherAResult) -> np.ndarray:
    """Convert full Teacher-A state, then apply target sampling and body ignore."""
    label, _ = sample_state_to_forward_bev(result.state, result.geometry)
    label[body_ignore_mask()] = IGNORE_INDEX
    if result.root.configuration_valid is False:
        label.fill(IGNORE_INDEX)
    invalid = ~np.isin(label, list(VALID_LABELS))
    if invalid.any():
        raise AssertionError("four-state label contains an invalid value")
    return label.astype(np.uint8, copy=False)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _lookup_unique_image_ids(group) -> tuple[np.ndarray, dict[int, int], list[int]]:
    ids = np.asarray(group["image_id"], dtype=np.int64)
    first: dict[int, int] = {}
    duplicates: list[int] = []
    for row, image_id in enumerate(ids):
        key = int(image_id)
        if key in first:
            duplicates.append(key)
        else:
            first[key] = row
    return np.asarray(sorted(first), dtype=np.int64), first, sorted(set(duplicates))


def _percentiles(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"p10": None, "p50": None, "p90": None}
    x = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(x, p)) for p in (10, 50, 90)}


def summarize_labels(
    labels: np.ndarray,
    image_ids: np.ndarray,
    label_valid: np.ndarray,
    root_valid: np.ndarray,
    frame_rows: list[dict],
    duplicate_image_ids: list[int],
) -> dict:
    valid_labels = labels[label_valid]
    pixel_counts = {str(value): int(np.count_nonzero(valid_labels == value)) for value in (0, 1, 2, 3, 255)}
    total_valid_pixels = max(int(valid_labels.size), 1)
    pixel_ratios = {key: value / total_valid_pixels for key, value in pixel_counts.items()}
    disc = pixel_counts["2"]
    reachable = pixel_counts["3"]
    disc_free_ratio = disc / max(disc + reachable, 1)
    valid_rows = [row for row in frame_rows if row["label_valid"]]
    summary = {
        "total_unique_image_id": int(len(image_ids)),
        "valid_label_frames": int(np.count_nonzero(label_valid)),
        "root_invalid_frames": int(np.count_nonzero(~root_valid)),
        "root_invalid_ratio": float(np.count_nonzero(~root_valid) / max(len(root_valid), 1)),
        "frames_containing_unknown": int(sum(row["unknown_count"] > 0 for row in valid_rows)),
        "frames_containing_blocked": int(sum(row["blocked_count"] > 0 for row in valid_rows)),
        "frames_containing_disconnected_free": int(sum(row["disconnected_free_count"] > 0 for row in valid_rows)),
        "frames_containing_reachable": int(sum(row["reachable_count"] > 0 for row in valid_rows)),
        "frames_containing_all_four_trainable_classes": int(
            sum(all(row[f"{name}_count"] > 0 for name in ("unknown", "blocked", "disconnected_free", "reachable")) for row in valid_rows)
        ),
        "topology_frame_ratio": float(
            sum(row["disconnected_free_count"] > 0 for row in valid_rows) / max(len(valid_rows), 1)
        ),
        "pixel_counts_valid_frames": pixel_counts,
        "pixel_ratios_valid_frames": pixel_ratios,
        "disc_free_ratio": float(disc / max(disc + reachable, 1)),
        "disconnected_free_ratio_percentiles": _percentiles([row["disconnected_free_ratio"] for row in valid_rows]),
        "reachable_ratio_percentiles": _percentiles([row["reachable_ratio"] for row in valid_rows]),
        "ignore_ratio_percentiles": _percentiles([row["ignore_ratio"] for row in valid_rows]),
        "illegal_label_values": sorted(set(np.unique(labels).astype(int)) - set(VALID_LABELS)),
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_image_id_count": len(duplicate_image_ids),
        "all_ignore_valid_frames": int(sum(row["all_ignore_valid"] for row in valid_rows)),
        "root_invalid_all_label_invalid": bool(np.all(label_valid[root_valid == 0] == 0)),
        "body_mask_all_ignore": bool(all(row["body_mask_bad_count"] == 0 for row in frame_rows)),
        "target_shape": list(TARGET_SHAPE),
        "target_dtype": str(labels.dtype),
    }
    return summary


def _frame_stats(label: np.ndarray, image_id: int, label_valid: bool, root_valid: bool, reason: str) -> dict:
    body = body_ignore_mask()
    counts = {name: int(np.count_nonzero(label == value)) for name, value in (
        ("unknown", 0), ("blocked", 1), ("disconnected_free", 2), ("reachable", 3), ("ignore", 255)
    )}
    valid_pixel_count = sum(counts[name] for name in ("unknown", "blocked", "disconnected_free", "reachable"))
    return {
        "image_id": int(image_id),
        "label_valid": bool(label_valid),
        "root_valid": bool(root_valid),
        "root_invalid_reason": reason,
        **{f"{name}_count": value for name, value in counts.items()},
        "valid_pixel_count": valid_pixel_count,
        "unknown_ratio": counts["unknown"] / max(valid_pixel_count, 1),
        "blocked_ratio": counts["blocked"] / max(valid_pixel_count, 1),
        "disconnected_free_ratio": counts["disconnected_free"] / max(valid_pixel_count, 1),
        "reachable_ratio": counts["reachable"] / max(valid_pixel_count, 1),
        "ignore_ratio": counts["ignore"] / label.size,
        "all_ignore_valid": bool(label_valid and valid_pixel_count == 0),
        "body_mask_bad_count": int(np.count_nonzero(label[body] != IGNORE_INDEX)),
        "label_shape": str(list(label.shape)),
        "illegal_value_count": int(np.count_nonzero(~np.isin(label, list(VALID_LABELS)))),
    }


def _draw_visualization(
    output: Path,
    mission: str,
    image_id: int,
    result: Optional[TeacherAResult],
    label: np.ndarray,
    path_groups: dict,
    kind: str,
) -> None:
    state_cmap = ListedColormap(["#9e9e9e", "#e53935", "#f39c12", "#2e7d32", "#000000"])
    state_norm = {0: 0, 1: 1, 2: 2, 3: 3, 255: 4}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].set_title(f"{mission} image_id={image_id} RGB")
    image_dir = Path(mission) / "images" / "hdr_front"
    image_path = image_dir / f"{image_id:06d}.jpeg"
    if image_path.exists():
        axes[0].imshow(plt.imread(image_path))
    else:
        axes[0].text(0.5, 0.5, "front image not found", ha="center", va="center")
    axes[0].axis("off")

    if result is not None:
        display_state = result if isinstance(result, np.ndarray) else result.state
        axes[1].imshow(np.fliplr(display_state.astype(np.int16)), origin="lower", cmap="tab10", interpolation="nearest")
        axes[1].set_title("Teacher-A 7-state (source grid)")
    else:
        axes[1].text(0.5, 0.5, "root invalid: no searchable field", ha="center", va="center")
        axes[1].set_title("Teacher-A 7-state")
    axes[1].axis("off")

    display_label = np.vectorize(state_norm.get)(label)
    axes[2].imshow(np.fliplr(display_label), origin="lower", cmap=state_cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[2].set_title("Forward BEV four-state [40,60]")
    axes[2].set_xlabel("y-left columns")
    axes[2].set_ylabel("x-forward rows")
    axes[2].set_xticks([0, 20, 40, 59], labels=["-3", "-1", "1", "3"])
    axes[2].set_yticks([0, 20, 39], labels=["0", "2", "4"])
    for source, group in path_groups.items():
        ids = np.asarray(group["image_id"], dtype=np.int64)
        for row in np.flatnonzero(ids == image_id):
            path = np.asarray(group["path"][int(row)], dtype=np.float32)
            x = np.floor((path[:, 0] - TARGET_X_RANGE_M[0]) / TARGET_RESOLUTION_M).astype(int)
            y = np.floor((path[:, 1] - TARGET_Y_RANGE_M[0]) / TARGET_RESOLUTION_M).astype(int)
            valid = (x >= 0) & (x < TARGET_SHAPE[0]) & (y >= 0) & (y < TARGET_SHAPE[1])
            axes[2].plot((TARGET_SHAPE[1] - 1) - y[valid], x[valid], linewidth=1, label=source)
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend(fontsize=8)
    target = output / f"{kind}_image_{image_id:06d}.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)


def generate_mission(
    mission: Path,
    output_dir: Path,
    radius_m: float,
    max_images: Optional[int],
    visualize_count: int,
    seed: int,
    device: str,
    overwrite: bool,
    mppi_config: Path,
    map_size: float,
    map_resolution: float,
    output_format: str,
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite explicitly")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(mppi_config).mppi
    elevation_ids_group = zarr.open_group(str(mission / "data" / "elevation_map"), mode="r")
    first_elevation = np.asarray(elevation_ids_group["elevation"][0])
    source_resolution = float(map_resolution)
    source_map_size = float(map_size)
    geometry = MapGeometry(
        int(first_elevation.shape[0]), int(first_elevation.shape[1]),
        source_resolution, (-source_map_size, -source_map_size)
    )
    if first_elevation.ndim != 2 or first_elevation.shape[0] != first_elevation.shape[1]:
        raise ValueError(f"source elevation must be square 2-D, got {first_elevation.shape}")
    elevation_group = elevation_ids_group
    image_ids, first_rows, duplicates = _lookup_unique_image_ids(elevation_group)
    if max_images is not None:
        image_ids = image_ids[:max_images]
    filter_model = get_filter_torch(device)
    path_groups = {}
    for source in ("geometric_paths", "teleop_paths"):
        p = mission / "data" / source
        if p.exists():
            path_groups[source] = zarr.open_group(str(p), mode="r")

    labels = []
    label_valid = []
    root_valid = []
    reasons = []
    source_rows = []
    frame_rows = []
    results_for_visualization = {}
    visual_valid_ids: list[int] = []
    visual_topology_ids: list[int] = []
    effective_radius = None
    for image_id in image_ids:
        elevation_row = first_rows[int(image_id)]
        elevation = np.asarray(elevation_group["elevation"][elevation_row], dtype=np.float32)
        if elevation.shape != (geometry.height, geometry.width):
            raise ValueError(f"Expected source elevation {(geometry.height, geometry.width)}, got {elevation.shape}")
        trav = compute_limo_traversability(elevation, filter_model, cfg, device)
        result = build_teacher_a(
            geometry=geometry,
            known_trav=trav.known_trav,
            risk=trav.risk,
            fatal_threshold=float(cfg.fatal_th),
            inflation_radius_m=radius_m,
        )
        effective_radius = float(result.effective_radius_m)
        valid = bool(result.root.configuration_valid)
        if valid:
            label = make_four_state_label(result)
        else:
            label = np.full(TARGET_SHAPE, IGNORE_INDEX, dtype=np.uint8)
        labels.append(label)
        label_valid.append(valid)
        root_valid.append(valid)
        reasons.append(result.root.failure_reason or "")
        source_rows.append(elevation_row)
        frame_rows.append(_frame_stats(label, int(image_id), valid, valid, result.root.failure_reason or ""))
        if valid and len(visual_valid_ids) < visualize_count:
            visual_valid_ids.append(int(image_id))
            results_for_visualization[int(image_id)] = result.state.copy()
        if valid and result.traversable_but_disconnected.any() and len(visual_topology_ids) < visualize_count:
            visual_topology_ids.append(int(image_id))
            results_for_visualization[int(image_id)] = result.state.copy()
    labels_array = np.stack(labels).astype(np.uint8)
    label_valid_array = np.asarray(label_valid, dtype=bool)
    root_valid_array = np.asarray(root_valid, dtype=bool)
    stats = summarize_labels(labels_array, image_ids, label_valid_array, root_valid_array, frame_rows, duplicates)
    stats["mission"] = mission.name
    stats["requested_radius_m"] = float(radius_m)
    stats["effective_radius_m"] = effective_radius
    stats["source_geometry"] = {"shape": [geometry.height, geometry.width], "resolution_m": geometry.resolution, "origin_xy": list(geometry.origin_xy)}
    stats["config_path"] = str(mppi_config)
    metadata = {
        "teacher_name": "Teacher-A",
        "teacher_interface": "dataset_builder.reachability.teacher_a.build_teacher_a",
        "requested_radius_m": float(radius_m),
        "effective_radius_m": effective_radius,
        "fatal_threshold": float(cfg.fatal_th),
        "source_geometry": {"shape": [geometry.height, geometry.width], "resolution_m": geometry.resolution, "origin_xy": list(geometry.origin_xy)},
        "target": {"x_range_m": list(TARGET_X_RANGE_M), "y_range_m": list(TARGET_Y_RANGE_M), "resolution_m": TARGET_RESOLUTION_M, "shape": list(TARGET_SHAPE)},
        "class_mapping": {"unknown": 0, "blocked": 1, "disconnected_free": 2, "reachable": 3, "ignore": 255},
        "ignore_index": 255,
        "robot_body_ignore_m": {"x": list(BODY_X_RANGE_M), "y": list(BODY_Y_RANGE_M)},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mppi_config": str(mppi_config),
        "git_commit": _git_commit(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text("# Minimal Reachability Labels\\n\\n" + json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_format in ("npz", "both"):
        np.savez_compressed(
            output_dir / "labels.npz",
            image_id=image_ids.astype(np.int64),
            state=labels_array,
            label_valid=label_valid_array,
            root_valid=root_valid_array,
            source_elevation_index=np.asarray(source_rows, dtype=np.int64),
            root_invalid_reason=np.asarray(reasons, dtype="U64"),
        )
    if output_format in ("zarr", "both"):
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
        zg = zarr.open_group(str(output_dir), mode="a")
        zg.create_dataset("image_id", data=image_ids.astype(np.int64), chunks=(1024,), compressor=compressor, overwrite=True)
        zg.create_dataset("elevation_row", data=np.asarray(source_rows, dtype=np.int64), chunks=(1024,), compressor=compressor, overwrite=True)
        zg.create_dataset("state", data=labels_array, chunks=(32, TARGET_SHAPE[0], TARGET_SHAPE[1]), compressor=compressor, overwrite=True)
        zg.create_dataset("label_valid", data=label_valid_array.astype(bool), chunks=(1024,), compressor=compressor, overwrite=True)
        zg.create_dataset("root_valid", data=root_valid_array.astype(bool), chunks=(1024,), compressor=compressor, overwrite=True)
        zg.create_dataset("root_invalid_reason", data=np.asarray(reasons, dtype="U64"), chunks=(1024,), compressor=compressor, overwrite=True)
        zg.create_dataset("timestamp", data=np.asarray(elevation_group["timestamp"][:], dtype=np.float64)[np.asarray(source_rows, dtype=np.int64)], chunks=(1024,), compressor=compressor, overwrite=True)
        zg.attrs.update(metadata)
        zg.attrs["index_mode"] = "elevation_row"
        zg.attrs["output_name"] = "reachability_labels_minimal"
    _write_frame_csv(output_dir / "frame_stats.csv", frame_rows)
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value])
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    id_to_position = {int(image_id): i for i, image_id in enumerate(image_ids)}
    valid_positions = np.asarray([id_to_position[x] for x in visual_valid_ids], dtype=np.int64)
    random_positions = rng.choice(valid_positions, size=min(visualize_count, len(valid_positions)), replace=False) if len(valid_positions) else []
    topology_positions = [id_to_position[x] for x in visual_topology_ids]
    for pos in random_positions:
        _draw_visualization(visual_dir, str(mission), int(image_ids[pos]), results_for_visualization.get(int(image_ids[pos])), labels_array[pos], path_groups, "random")
    for pos in topology_positions:
        _draw_visualization(visual_dir, str(mission), int(image_ids[pos]), results_for_visualization.get(int(image_ids[pos])), labels_array[pos], path_groups, "topology")
    return output_dir


def _write_frame_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--radius", type=float, default=0.26)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--visualize-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mppi-config", type=Path, default=Path("dataset_builder/configs/build.yaml"))
    parser.add_argument("--map-size", type=float, default=4.0)
    parser.add_argument("--map-resolution", type=float, default=0.04)
    parser.add_argument("--format", choices=("zarr", "npz", "both"), default="zarr")
    args = parser.parse_args()
    output_dir = args.output_dir or (args.mission / "reachability_labels_minimal")
    generate_mission(
        args.mission, output_dir, args.radius, args.max_images,
        args.visualize_count, args.seed, args.device, args.overwrite, args.mppi_config,
        args.map_size, args.map_resolution, args.format
    )


if __name__ == "__main__":
    main()
