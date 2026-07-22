"""Generate large, seven-state Teacher-A reachability figures.

Each output figure joins hdr_left/front/right by their shared ``image_id`` and
shows the matching elevation map, all requested geo/tel paths, and the complete
Teacher-A state map. Inputs are read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import zarr
from omegaconf import OmegaConf
from PIL import Image

from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import ReachabilityState, build_teacher_a
from dataset_builder.reachability.traversability import compute_limo_traversability


PATH_COLORS = {"geometric": "#00d5ff", "teleop": "#ff4dcc"}
PATH_LABELS = {"geometric": "geometric path", "teleop": "teleop path"}
STATE_COLORS = (
    "#000000",  # outside domain
    "#a8a8a8",  # unknown center
    "#e53935",  # locally blocked
    "#7f0000",  # clearance blocked
    "#666666",  # unknown footprint
    "#f59e0b",  # traversable but disconnected
    "#2ca25f",  # reachable
)
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), STATE_CMAP.N)
STATE_DESCRIPTIONS = (
    "footprint crosses planning domain",
    "center cell is unknown",
    "center risk reaches fatal threshold",
    "safe center; footprint overlaps fatal",
    "known safe center; footprint overlaps unknown",
    "configuration-free but disconnected from root",
    "configuration-free and connected to root",
)


def _state_legend_handles() -> list:
    return [
        plt.Line2D(
            [],
            [],
            marker="s",
            color=color,
            markeredgecolor="white" if index == 0 else color,
            linestyle="",
            markersize=9,
            label=f"{index} {ReachabilityState(index).name} — {description}",
        )
        for index, (color, description) in enumerate(zip(STATE_COLORS, STATE_DESCRIPTIONS))
    ]


def _lookup_elevation(group, image_id: int) -> np.ndarray:
    ids = np.asarray(group["image_id"], dtype=np.int64)
    rows = np.flatnonzero(ids == image_id)
    if len(rows) != 1:
        raise KeyError(f"elevation image_id={image_id}: expected one row, found {len(rows)}")
    return np.asarray(group["elevation"][int(rows[0])], dtype=np.float32)


def _open_path_groups(mission: Path, sources: tuple[str, ...]) -> dict[str, object]:
    groups = {}
    for source in sources:
        path = mission / "data" / f"{source}_paths"
        if path.exists():
            groups[source] = zarr.open_group(str(path), mode="r")
    return groups


def _eligible_image_ids(
    mission: Path, elevation_group, path_groups: dict[str, object]
) -> np.ndarray:
    elevation_ids = set(map(int, np.asarray(elevation_group["image_id"], dtype=np.int64)))
    path_ids: set[int] = set()
    for group in path_groups.values():
        path_ids.update(map(int, np.asarray(group["image_id"], dtype=np.int64)))
    candidates = elevation_ids & path_ids
    cameras = ("hdr_left", "hdr_front", "hdr_right")
    valid = [
        image_id
        for image_id in sorted(candidates)
        if all(
            (mission / "images" / camera / f"{image_id:06d}.jpeg").exists()
            for camera in cameras
        )
    ]
    return np.asarray(valid, dtype=np.int64)


def _evenly_spaced(values: np.ndarray, count: int) -> list[int]:
    if len(values) <= count:
        return [int(v) for v in values]
    positions = np.linspace(0, len(values) - 1, count, dtype=int)
    return [int(values[p]) for p in positions]


def _draw_paths(ax, group, image_id: int, color: str, label: str) -> int:
    ids = np.asarray(group["image_id"], dtype=np.int64)
    rows = np.flatnonzero(ids == image_id)
    for local_index, row in enumerate(rows):
        path = np.asarray(group["path"][int(row)], dtype=np.float32)
        goal = np.asarray(group["goal"][int(row)], dtype=np.float32)
        line = ax.plot(
            path[:, 1],
            path[:, 0],
            color=color,
            alpha=0.9,
            linewidth=2.0,
            label=label if local_index == 0 else None,
            zorder=12,
        )[0]
        line.set_path_effects(
            [path_effects.Stroke(linewidth=3.5, foreground="black", alpha=0.7), path_effects.Normal()]
        )
        ax.plot(
            goal[1],
            goal[0],
            marker="D",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=6,
            zorder=13,
        )
    return int(len(rows))


def _format_map_axis(ax, geometry: MapGeometry) -> None:
    extent = geometry.imshow_extent_yx
    # Array rows are robot x and columns are robot y. Reverse the page x-axis
    # so positive robot-left appears on the left side of the figure.
    ax.set_xlim(extent[1], extent[0])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]  (robot left +)")
    ax.set_ylabel("x [m]  (robot forward +)")
    ax.axhline(0.0, color="white", alpha=0.25, linewidth=0.7)
    ax.axvline(0.0, color="white", alpha=0.25, linewidth=0.7)


def _camera_image(mission: Path, camera: str, image_id: int) -> np.ndarray:
    path = mission / "images" / camera / f"{image_id:06d}.jpeg"
    return np.asarray(Image.open(path).convert("RGB"))


def plot_one(
    mission: Path,
    sources: tuple[str, ...],
    image_id: int,
    output: Path,
    *,
    radius_m: float,
    map_size: float,
    map_resolution: float,
    mppi_cfg,
    filter_model,
    device: str,
) -> dict:
    geometry = MapGeometry(
        int(round(2 * map_size / map_resolution)),
        int(round(2 * map_size / map_resolution)),
        map_resolution,
        (-map_size, -map_size),
    )
    elevation_group = zarr.open_group(str(mission / "data" / "elevation_map"), mode="r")
    path_groups = _open_path_groups(mission, sources)
    elevation = _lookup_elevation(elevation_group, image_id)
    trav = compute_limo_traversability(elevation, filter_model, mppi_cfg, device)
    result = build_teacher_a(
        geometry=geometry,
        known_trav=trav.known_trav,
        risk=trav.risk,
        fatal_threshold=float(mppi_cfg.fatal_th),
        inflation_radius_m=radius_m,
    )

    matching_rows = {
        source: np.flatnonzero(np.asarray(group["image_id"], dtype=np.int64) == image_id)
        for source, group in path_groups.items()
    }
    if not any(len(rows) for rows in matching_rows.values()):
        raise KeyError(f"no requested path for image_id={image_id}")

    cameras = ("hdr_left", "hdr_front", "hdr_right")
    camera_images = [_camera_image(mission, camera, image_id) for camera in cameras]

    fig = plt.figure(figsize=(24, 15.5), constrained_layout=True)
    grid = fig.add_gridspec(
        3, 3, height_ratios=(0.78, 1.45, 0.24), width_ratios=(1, 1, 1)
    )
    for column, (camera, camera_image) in enumerate(zip(cameras, camera_images)):
        camera_ax = fig.add_subplot(grid[0, column])
        camera_ax.imshow(camera_image)
        camera_ax.set_title(f"{camera}  |  image_id={image_id}", fontsize=13)
        camera_ax.axis("off")

    elevation_ax = fig.add_subplot(grid[1, 0])
    state_ax = fig.add_subplot(grid[1, 1:])
    finite = np.isfinite(elevation)
    elev_masked = np.ma.masked_where(~finite, elevation)
    elev_cmap = plt.get_cmap("terrain").copy()
    elev_cmap.set_bad("#bdbdbd")
    elevation_image = elevation_ax.imshow(
        elev_masked,
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=elev_cmap,
        aspect="equal",
    )
    elevation_ax.set_title("Elevation + matched expert paths", fontsize=14)
    path_counts = {}
    for source, group in path_groups.items():
        path_counts[source] = _draw_paths(
            elevation_ax, group, image_id, PATH_COLORS[source], PATH_LABELS[source]
        )
    elevation_ax.scatter(
        [0], [0], c="#ffffff", edgecolors="black", marker="*", s=150, linewidths=1.2, label="robot"
    )
    _format_map_axis(elevation_ax, geometry)
    elevation_ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(elevation_image, ax=elevation_ax, fraction=0.046, pad=0.04, label="elevation [m]")

    state_ax.imshow(
        result.state,
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=STATE_CMAP,
        norm=STATE_NORM,
        aspect="equal",
        interpolation="nearest",
    )
    state_ax.set_title(
        f"Teacher-A seven-state reachability  |  requested r={radius_m:.2f} m  "
        f"(grid r={result.effective_radius_m:.2f} m)",
        fontsize=14,
    )
    for source, group in path_groups.items():
        _draw_paths(state_ax, group, image_id, PATH_COLORS[source], PATH_LABELS[source])
    state_ax.scatter(
        [0], [0], c="#ffffff", edgecolors="black", marker="*", s=150, linewidths=1.2, zorder=15
    )
    _format_map_axis(state_ax, geometry)
    state_handles = _state_legend_handles()
    state_handles.extend(
        plt.Line2D([], [], color=PATH_COLORS[source], linewidth=3, label=PATH_LABELS[source])
        for source in path_groups
        if path_counts.get(source, 0) > 0
    )
    state_handles.append(
        plt.Line2D([], [], marker="*", color="white", markeredgecolor="black", linestyle="", markersize=11, label="robot root")
    )
    legend_ax = fig.add_subplot(grid[2, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=state_handles,
        loc="center",
        fontsize=9.5,
        ncol=3,
        framealpha=0.96,
    )

    counts_text = " | ".join(f"{source} paths={path_counts.get(source, 0)}" for source in sources)
    root_text = "valid" if result.root.configuration_valid else f"invalid: {result.root.failure_reason}"
    fig.suptitle(
        f"mission={mission.name}  |  image_id={image_id}  |  {counts_text}  |  root={root_text}",
        fontsize=17,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "mission": mission.name,
        "image_id": image_id,
        "output": str(output),
        "path_counts": path_counts,
        "root_configuration_valid": result.root.configuration_valid,
        "root_failure_reason": result.root.failure_reason,
        "requested_radius_m": radius_m,
        "effective_radius_m": result.effective_radius_m,
        "state_counts": result.state_counts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument(
        "--source", choices=("geo", "tel", "geometric", "teleop", "both"), default="both"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.26)
    parser.add_argument("--map-size", type=float, default=4.0)
    parser.add_argument("--map-resolution", type=float, default=0.04)
    parser.add_argument("--mppi-config", type=Path, default=Path("dataset_builder/configs/build.yaml"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = {"geo": "geometric", "tel": "teleop"}.get(args.source, args.source)
    sources = ("geometric", "teleop") if source == "both" else (source,)
    cfg = OmegaConf.load(args.mppi_config).mppi
    elevation_group = zarr.open_group(str(args.mission / "data" / "elevation_map"), mode="r")
    path_groups = _open_path_groups(args.mission, sources)
    image_ids = _evenly_spaced(
        _eligible_image_ids(args.mission, elevation_group, path_groups), args.count
    )
    filter_model = get_filter_torch(args.device)
    records = []
    for image_id in image_ids:
        records.append(
            plot_one(
                args.mission,
                sources,
                image_id,
                args.output_dir / f"{args.mission.name}_image_{image_id:06d}_r{args.radius:.2f}.png",
                radius_m=args.radius,
                map_size=args.map_size,
                map_resolution=args.map_resolution,
                mppi_cfg=cfg,
                filter_model=filter_model,
                device=args.device,
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"{args.mission.name}_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"generated {len(image_ids)} figures in {args.output_dir}")


if __name__ == "__main__":
    main()
