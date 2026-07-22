"""Visualize stored reachability_5labels with elevation and three HDR cameras."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image
from tqdm import tqdm
import zarr

from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import ReachabilityState


CAMERAS = ("hdr_left", "hdr_front", "hdr_right")
REQUIRED_LABEL_ARRAYS = {
    "state_label",
    "ignore_mask",
    "risk",
    "geodesic_m",
    "image_id",
}
STATE_COLORS = (
    "#8c8c8c",  # unknown
    "#e53935",  # locally blocked
    "#7f0000",  # clearance blocked
    "#f59e0b",  # traversable but disconnected
    "#2ca25f",  # reachable
)
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), STATE_CMAP.N)


def _camera_path(mission: Path, camera: str, image_id: int) -> Path | None:
    candidates = (
        mission / "images" / camera / f"{image_id:06d}.jpeg",
        mission / "images" / camera / f"{image_id:06d}.jpg",
        mission / "images" / camera / f"{image_id}.jpeg",
        mission / "images" / camera / f"{image_id}.jpg",
    )
    return next((path for path in candidates if path.is_file()), None)


def _validate_groups(labels, elevation) -> tuple[int, int, int]:
    label_keys = set(labels.array_keys())
    missing = REQUIRED_LABEL_ARRAYS - label_keys
    if missing:
        raise KeyError(f"reachability_5labels missing arrays: {sorted(missing)}")
    if "elevation" not in elevation or "image_id" not in elevation:
        raise KeyError("elevation_map must contain elevation and image_id")

    state_shape = labels["state_label"].shape
    elevation_shape = elevation["elevation"].shape
    if len(state_shape) != 3:
        raise ValueError(f"state_label must have shape [N,H,W], got {state_shape}")
    if elevation_shape != state_shape:
        raise ValueError(
            f"elevation shape {elevation_shape} != state_label shape {state_shape}"
        )
    for key in ("ignore_mask", "risk", "geodesic_m"):
        if labels[key].shape != state_shape:
            raise ValueError(
                f"{key} shape {labels[key].shape} != state_label shape {state_shape}"
            )

    label_ids = np.asarray(labels["image_id"][:])
    elevation_ids = np.asarray(elevation["image_id"][:])
    if not np.array_equal(label_ids, elevation_ids):
        raise ValueError(
            "reachability_5labels/image_id is not exactly aligned with "
            "elevation_map/image_id"
        )
    if label_ids.shape != (state_shape[0],):
        raise ValueError(
            f"image_id shape {label_ids.shape} does not match N={state_shape[0]}"
        )

    return tuple(map(int, state_shape))


def _choose_rows(mission: Path, image_ids: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    eligible = np.asarray(
        [
            row
            for row, image_id in enumerate(image_ids)
            if all(
                _camera_path(mission, camera, int(image_id)) is not None
                for camera in CAMERAS
            )
        ],
        dtype=np.int64,
    )
    if len(eligible) < count:
        raise RuntimeError(
            f"requested {count} figures, but only {len(eligible)} rows have all "
            "three exact image_id camera files"
        )
    positions = np.linspace(0, len(eligible) - 1, count, dtype=np.int64)
    return eligible[positions]


def _format_bev_axis(ax, geometry: MapGeometry) -> None:
    left, right, bottom, top = geometry.imshow_extent_yx
    ax.set_xlim(right, left)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]  (robot left +)")
    ax.set_ylabel("x [m]  (robot forward +)")
    ax.axhline(0.0, color="white", alpha=0.3, linewidth=0.7)
    ax.axvline(0.0, color="white", alpha=0.3, linewidth=0.7)


def _mark_root(ax) -> None:
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=125,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=10,
    )


def _finite_masked(array: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~np.isfinite(array), array)


def _state_legend_handles() -> list:
    return [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=color,
            markersize=10,
            label=f"{int(state)}  {state.name}",
        )
        for state, color in zip(ReachabilityState, STATE_COLORS)
    ]


def _plot_one(
    *,
    mission: Path,
    labels,
    elevation_group,
    row: int,
    geometry: MapGeometry,
    output_path: Path,
) -> dict:
    image_id = int(labels["image_id"][row])
    state = np.asarray(labels["state_label"][row], dtype=np.uint8)
    ignore = np.asarray(labels["ignore_mask"][row], dtype=bool)
    risk = np.asarray(labels["risk"][row], dtype=np.float32)
    geodesic = np.asarray(labels["geodesic_m"][row], dtype=np.float32)
    elevation = np.asarray(elevation_group["elevation"][row], dtype=np.float32)
    values = np.unique(state)
    if np.any(values > int(ReachabilityState.REACHABLE)):
        raise ValueError(
            f"row={row}, image_id={image_id}: invalid state values {values.tolist()}"
        )
    timestamp = (
        float(labels["timestamp"][row]) if "timestamp" in labels else None
    )

    camera_paths = {
        camera: _camera_path(mission, camera, image_id) for camera in CAMERAS
    }
    if any(path is None for path in camera_paths.values()):
        raise FileNotFoundError(f"row={row}, image_id={image_id}: camera image missing")

    fig = plt.figure(figsize=(22, 17), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(0.72, 1.0, 1.0))

    for column, camera in enumerate(CAMERAS):
        ax = fig.add_subplot(grid[0, column])
        with Image.open(camera_paths[camera]) as image:
            camera_image = np.array(image.convert("RGB"))
        ax.imshow(camera_image)
        ax.set_title(f"{camera}  |  image_id={image_id}", fontsize=12)
        ax.axis("off")

    elevation_ax = fig.add_subplot(grid[1, 0])
    elevation_cmap = plt.get_cmap("terrain").copy()
    elevation_cmap.set_bad("#8c8c8c")
    elevation_image = elevation_ax.imshow(
        _finite_masked(elevation),
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=elevation_cmap,
        aspect="equal",
    )
    elevation_ax.set_title("Elevation map")
    _mark_root(elevation_ax)
    _format_bev_axis(elevation_ax, geometry)
    fig.colorbar(
        elevation_image,
        ax=elevation_ax,
        fraction=0.046,
        pad=0.04,
        label="elevation [m]",
    )

    state_ax = fig.add_subplot(grid[1, 1])
    state_ax.imshow(
        state,
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=STATE_CMAP,
        norm=STATE_NORM,
        interpolation="nearest",
        aspect="equal",
    )
    state_ax.set_title("Five-class reachability")
    _mark_root(state_ax)
    _format_bev_axis(state_ax, geometry)

    risk_ax = fig.add_subplot(grid[1, 2])
    risk_cmap = plt.get_cmap("magma").copy()
    risk_cmap.set_bad("#8c8c8c")
    risk_image = risk_ax.imshow(
        _finite_masked(risk),
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=risk_cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    risk_ax.set_title("Risk (NaN = unknown)")
    _mark_root(risk_ax)
    _format_bev_axis(risk_ax, geometry)
    fig.colorbar(risk_image, ax=risk_ax, fraction=0.046, pad=0.04, label="risk")

    geodesic_ax = fig.add_subplot(grid[2, 0])
    geodesic_cmap = plt.get_cmap("viridis").copy()
    geodesic_cmap.set_bad("#8c8c8c")
    geodesic_image = geodesic_ax.imshow(
        _finite_masked(geodesic),
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=geodesic_cmap,
        vmin=0.0,
        aspect="equal",
    )
    geodesic_ax.set_title("Geodesic distance (REACHABLE only)")
    _mark_root(geodesic_ax)
    _format_bev_axis(geodesic_ax, geometry)
    fig.colorbar(
        geodesic_image,
        ax=geodesic_ax,
        fraction=0.046,
        pad=0.04,
        label="geodesic [m]",
    )

    ignore_ax = fig.add_subplot(grid[2, 1])
    ignore_image = ignore_ax.imshow(
        ignore.astype(np.uint8),
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=ListedColormap(("#f5f5f5", "#5e3c99")),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    ignore_ax.set_title(f"Ignore mask  |  ignored={int(ignore.sum())}")
    _mark_root(ignore_ax)
    _format_bev_axis(ignore_ax, geometry)
    fig.colorbar(
        ignore_image,
        ax=ignore_ax,
        fraction=0.046,
        pad=0.04,
        ticks=(0, 1),
    )

    info_ax = fig.add_subplot(grid[2, 2])
    info_ax.axis("off")
    state_counts = {
        state_value.name.lower(): int(np.count_nonzero(state == int(state_value)))
        for state_value in ReachabilityState
    }
    root_i, root_j = geometry.root_index
    root_geodesic = float(geodesic[root_i, root_j])
    root_valid = np.isfinite(root_geodesic)
    info_lines = [
        f"mission: {mission.name}",
        f"row: {row}",
        f"image_id: {image_id}",
        f"timestamp: {timestamp if timestamp is not None else 'not stored'}",
        f"root geodesic finite: {root_valid}",
        f"finite risk: {int(np.isfinite(risk).sum())}",
        f"finite geodesic: {int(np.isfinite(geodesic).sum())}",
        "",
        "state counts:",
    ]
    info_lines.extend(f"  {name}: {count}" for name, count in state_counts.items())
    info_ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=info_ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )
    info_ax.legend(
        handles=_state_legend_handles(),
        loc="lower left",
        fontsize=10,
        framealpha=0.95,
    )

    title_timestamp = "" if timestamp is None else f"  |  timestamp={timestamp:.6f}"
    fig.suptitle(
        f"mission={mission.name}  |  row={row}  |  image_id={image_id}"
        f"{title_timestamp}",
        fontsize=16,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "row": int(row),
        "image_id": image_id,
        "timestamp": timestamp,
        "output": str(output_path),
        "camera_files": {
            camera: str(path) for camera, path in camera_paths.items()
        },
        "state_counts": state_counts,
        "ignore_count": int(ignore.sum()),
        "root_geodesic_finite": bool(root_valid),
    }


def _replace_output(temp_output: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {output}; pass --overwrite to replace it"
            )
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        output.rename(backup)
        try:
            temp_output.rename(output)
        except BaseException:
            backup.rename(output)
            raise
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    else:
        temp_output.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize stored reachability labels, aligned elevation, and exact "
            "image_id left/front/right cameras."
        )
    )
    parser.add_argument("--mission-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mission = args.mission_dir.expanduser().resolve()
    labels_path = mission / "reachability_5labels"
    elevation_path = mission / "data" / "elevation_map"
    if not labels_path.is_dir():
        raise FileNotFoundError(f"reachability_5labels not found: {labels_path}")
    if not elevation_path.is_dir():
        raise FileNotFoundError(f"elevation_map not found: {elevation_path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else mission / f"reachability_5labels_visualization_{args.count}"
    )
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output_dir}; pass --overwrite to replace it"
        )

    labels = zarr.open_group(str(labels_path), mode="r")
    elevation = zarr.open_group(str(elevation_path), mode="r")
    _, height, width = _validate_groups(labels, elevation)
    image_ids = np.asarray(labels["image_id"][:])
    selected_rows = _choose_rows(mission, image_ids, args.count)

    resolution = float(labels.attrs.get("map_resolution_m", 0.04))
    default_origin = (-0.5 * height * resolution, -0.5 * width * resolution)
    origin = labels.attrs.get("map_origin_xy_m", default_origin)
    geometry = MapGeometry(
        height=height,
        width=width,
        resolution=resolution,
        origin_xy=(float(origin[0]), float(origin[1])),
    )

    temp_output = output_dir.with_name(
        f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        temp_output.mkdir(parents=True)
        records = []
        for row in tqdm(selected_rows, desc=mission.name, unit="figure"):
            image_id = int(image_ids[row])
            filename = (
                f"{mission.name}_row_{int(row):06d}_image_{image_id:06d}.png"
            )
            records.append(
                _plot_one(
                    mission=mission,
                    labels=labels,
                    elevation_group=elevation,
                    row=int(row),
                    geometry=geometry,
                    output_path=temp_output / filename,
                )
            )
        (temp_output / "manifest.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _replace_output(temp_output, output_dir, args.overwrite)
    except BaseException:
        if temp_output.exists():
            shutil.rmtree(temp_output)
        raise

    print(
        json.dumps(
            {
                "mission": mission.name,
                "figure_count": len(selected_rows),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
