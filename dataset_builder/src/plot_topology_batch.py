"""Generate large Teacher-A circular topological reachability figures.

Each output figure contains the r=0.26 m Teacher-A state map and all geo/tel
paths belonging to the selected image_id.  Inputs are read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from PIL import Image

from dataset_builder.mppi_planner.mppi_planner import GridMap2D, MPPIPlanner
from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.path_calibration import circular_structure
from dataset_builder.reachability.teacher_a import (
    ReachabilityState,
    build_teacher_a,
)
from dataset_builder.reachability.traversability import compute_limo_traversability
from dataset_builder.src.visualize import load_cameras, _project_fisheye


def _lookup_elevation(group, image_id: int) -> np.ndarray:
    ids = np.asarray(group["image_id"], dtype=np.int64)
    rows = np.flatnonzero(ids == image_id)
    if len(rows) != 1:
        raise KeyError(f"elevation image_id={image_id}: expected one row, found {len(rows)}")
    return np.asarray(group["elevation"][int(rows[0])], dtype=np.float32)


def _path_ids(group) -> np.ndarray:
    return np.unique(np.asarray(group["image_id"], dtype=np.int64))


def _evenly_spaced(values: np.ndarray, count: int) -> list[int]:
    if len(values) <= count:
        return [int(v) for v in values]
    positions = np.linspace(0, len(values) - 1, count, dtype=int)
    return [int(values[p]) for p in positions]


def _draw_paths(ax, group, image_id: int, color: str) -> int:
    ids = np.asarray(group["image_id"], dtype=np.int64)
    rows = np.flatnonzero(ids == image_id)
    for row in rows:
        path = np.asarray(group["path"][int(row)], dtype=np.float32)
        goal = np.asarray(group["goal"][int(row)], dtype=np.float32)
        # Display convention: screen-left is robot-left, so y is negated.
        ax.plot(-path[:, 1], path[:, 0], color=color, alpha=0.75, linewidth=1.2)
        ax.plot(-goal[1], goal[0], marker="D", color=color, markersize=4)
    return int(len(rows))


def plot_one(
    mission: Path,
    source: str,
    image_id: int,
    output: Path,
    *,
    radius_m: float,
    map_size: float,
    map_resolution: float,
    mppi_cfg,
    filter_model,
    device: str,
    run_mppi: bool = True,
) -> None:
    geometry = MapGeometry(
        int(round(2 * map_size / map_resolution)),
        int(round(2 * map_size / map_resolution)),
        map_resolution,
        (-map_size, -map_size),
    )
    elevation_group = zarr.open_group(str(mission / "data" / "elevation_map"), mode="r")
    path_group = zarr.open_group(str(mission / "data" / f"{source}_paths"), mode="r")
    elevation = _lookup_elevation(elevation_group, image_id)
    elevation_rows = np.flatnonzero(
        np.asarray(elevation_group["image_id"], dtype=np.int64) == image_id
    )
    elevation_row = int(elevation_rows[0])
    elevation_timestamp = float(elevation_group["timestamp"][elevation_row])
    trav = compute_limo_traversability(elevation, filter_model, mppi_cfg, device)
    result = build_teacher_a(
        geometry=geometry,
        known_trav=trav.known_trav,
        risk=trav.risk,
        fatal_threshold=float(mppi_cfg.fatal_th),
        inflation_radius_m=radius_m,
    )

    path_ids = np.asarray(path_group["image_id"], dtype=np.int64)
    path_rows = np.flatnonzero(path_ids == image_id)
    if len(path_rows) == 0:
        raise KeyError(f"no {source} path for image_id={image_id}")
    reference_path = np.asarray(path_group["path"][int(path_rows[0])], dtype=np.float32)
    reference_goal = np.asarray(path_group["goal"][int(path_rows[0])], dtype=np.float32)
    mppi_path = None
    if run_mppi:
        planner = MPPIPlanner(mppi_cfg, device)
        gridmap = GridMap2D(
            elevation=torch.as_tensor(elevation, dtype=torch.float32, device=planner.objective.device),
            resolution=map_resolution,
            origin_xy=torch.tensor((-map_size, -map_size), dtype=torch.float32, device=planner.objective.device),
        )
        mppi_path = planner.plan(
            gridmap,
            torch.zeros(3, dtype=torch.float32, device=planner.objective.device),
            torch.as_tensor(reference_goal, dtype=torch.float32, device=planner.objective.device),
        ).detach().cpu().numpy()

    # Strict shared image_id camera lookup for the visual comparison.
    front_image = np.asarray(Image.open(mission / "images" / "hdr_front" / f"{image_id:06d}.jpeg").convert("RGB"))
    camera_images = []
    for camera_name in ("hdr_left", "hdr_front", "hdr_right"):
        image_path = mission / "images" / camera_name / f"{image_id:06d}.jpeg"
        camera_images.append(np.asarray(Image.open(image_path).convert("RGB")) if image_path.exists() else None)
    available = [im for im in camera_images if im is not None]
    common_height = min(im.shape[0] for im in available)
    resized = []
    for im in camera_images:
        if im is None:
            resized.append(np.full((common_height, common_height, 3), 35, dtype=np.uint8))
        else:
            width = int(round(im.shape[1] * common_height / im.shape[0]))
            resized.append(np.asarray(Image.fromarray(im).resize((width, common_height))))
    camera_strip = np.concatenate(resized, axis=1)

    state = np.asarray(result.state, dtype=np.int16)
    # Reachable is visually emphasized; other states remain distinct.
    display = np.full(state.shape, 0, dtype=np.int8)  # unknown/outside
    display[state == int(ReachabilityState.REACHABLE)] = 1
    display[state == int(ReachabilityState.CLEARANCE_BLOCKED)] = 2
    display[state == int(ReachabilityState.LOCALLY_BLOCKED)] = 3
    display[state == int(ReachabilityState.TRAVERSABLE_BUT_DISCONNECTED)] = 4
    cmap = ListedColormap(["#bdbdbd", "#35a853", "#f6c344", "#e53935", "#8e44ad"])

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.75, 1.25))
    camera_ax = fig.add_subplot(grid[0, :])
    camera_ax.imshow(camera_strip)
    camera_ax.set_title("hdr_left | hdr_front | hdr_right  (shared image_id)", fontsize=14)
    camera_ax.axis("off")
    axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    extent = (-map_size, map_size, -map_size, map_size)
    finite = np.isfinite(elevation)
    elev_masked = np.ma.masked_where(~finite, elevation)
    elev_cmap = plt.get_cmap("terrain").copy()
    elev_cmap.set_bad("#bdbdbd")
    # Display convention matches visualize_bev_map.py:
    # top=front, bottom=rear, left=robot-left, right=robot-right.
    axes[0].imshow(np.fliplr(elev_masked), origin="lower", extent=extent, cmap=elev_cmap, aspect="equal")
    axes[0].set_title(f"Elevation + {source} paths, image_id={image_id}", fontsize=14)
    axes[0].set_xlabel("horizontal display (left = robot-left)")
    axes[0].set_ylabel("x forward [m]")
    n_paths = _draw_paths(axes[0], path_group, image_id, "#1565c0" if source == "geometric" else "#8e24aa")
    if mppi_path is not None:
        axes[0].plot(-mppi_path[:, 1], mppi_path[:, 0], color="#ff00ff", linewidth=2.5, label="recomputed MPPI")
    axes[0].scatter([0], [0], c="black", marker="+", s=100, linewidths=2, label="robot")
    axes[0].set_xlim(-map_size, map_size)
    axes[0].set_ylim(-map_size, map_size)
    axes[0].legend(loc="upper right")

    axes[1].imshow(np.fliplr(display), origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=4, aspect="equal", interpolation="nearest")
    axes[1].set_title(f"Teacher-A topological reachability, r={radius_m:.2f} m", fontsize=14)
    axes[1].set_xlabel("horizontal display (left = robot-left)")
    axes[1].set_ylabel("x forward [m]")
    _draw_paths(axes[1], path_group, image_id, "white")
    if mppi_path is not None:
        axes[1].plot(-mppi_path[:, 1], mppi_path[:, 0], color="#ff00ff", linewidth=2.5, label="recomputed MPPI")
    axes[1].scatter([0], [0], c="black", marker="+", s=100, linewidths=2)
    axes[1].set_xlim(-map_size, map_size)
    axes[1].set_ylim(-map_size, map_size)
    axes[1].legend(
        handles=[
            plt.Line2D([], [], color="#35a853", linewidth=8, label="reachable"),
            plt.Line2D([], [], color="#f6c344", linewidth=8, label="clearance blocked"),
            plt.Line2D([], [], color="#e53935", linewidth=8, label="locally blocked"),
            plt.Line2D([], [], color="#8e44ad", linewidth=8, label="disconnected"),
            plt.Line2D([], [], color="#bdbdbd", linewidth=8, label="unknown/outside"),
            plt.Line2D([], [], color="#ff00ff", linewidth=3, label="recomputed MPPI"),
        ],
        loc="upper right",
        fontsize=9,
    )
    fig.suptitle(
        f"{mission.name} | {source} | image_id={image_id} | elevation_row={elevation_row} | "
        f"elev_ts={elevation_timestamp:.6f} | paths={n_paths} | "
        f"MPPI goal=({reference_goal[0]:.2f}, {reference_goal[1]:.2f})",
        fontsize=16,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--source", choices=("geo", "tel", "geometric", "teleop"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.26)
    parser.add_argument("--map-size", type=float, default=4.0)
    parser.add_argument("--map-resolution", type=float, default=0.04)
    parser.add_argument("--mppi-config", type=Path, default=Path("dataset_builder/configs/build.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-mppi", action="store_true")
    args = parser.parse_args()
    source = {"geo": "geometric", "tel": "teleop"}.get(args.source, args.source)
    cfg = OmegaConf.load(args.mppi_config).mppi
    group = zarr.open_group(str(args.mission / "data" / f"{source}_paths"), mode="r")
    image_ids = _evenly_spaced(_path_ids(group), args.count)
    filter_model = get_filter_torch(args.device)
    for image_id in image_ids:
        plot_one(
            args.mission,
            source,
            image_id,
            args.output_dir / f"{source}_image_{image_id:06d}_r{args.radius:.2f}.png",
            radius_m=args.radius,
            map_size=args.map_size,
            map_resolution=args.map_resolution,
            mppi_cfg=cfg,
            filter_model=filter_model,
            device=args.device,
            run_mppi=not args.skip_mppi,
        )
    print(f"generated {len(image_ids)} figures in {args.output_dir}")


if __name__ == "__main__":
    main()
