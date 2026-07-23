"""Preview the hdr_front geometric FOV on one elevation-map frame.

This script is read-only: it does not add ``front_ignore`` to any Zarr group.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"limo-matplotlib-{os.getuid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import yaml
import zarr

from dataset_builder.reachability.coordinates import MapGeometry


def _load_calibration(mission: Path, extrinsics_config: Path) -> tuple:
    intrinsics_path = mission / "metadata" / "hdr_front_caminfo.yaml"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"front camera calibration not found: {intrinsics_path}")
    if not extrinsics_config.is_file():
        raise FileNotFoundError(
            f"canonical T_camera_base config not found: {extrinsics_config}"
        )

    with intrinsics_path.open("r", encoding="utf-8") as file:
        camera_info = yaml.safe_load(file)["camera_info"]
    with extrinsics_config.open("r", encoding="utf-8") as file:
        canonical_info = yaml.safe_load(file)

    if camera_info.get("distortion_model") != "equidistant":
        raise ValueError(
            "expected hdr_front distortion_model=equidistant, got "
            f"{camera_info.get('distortion_model')}"
        )

    K = np.asarray(camera_info["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(camera_info["D"], dtype=np.float64).reshape(-1)
    if len(D) < 4:
        raise ValueError(f"equidistant model requires at least 4 D values, got {len(D)}")
    E = np.asarray(canonical_info["E"], dtype=np.float64).reshape(4, 4)
    height = int(camera_info["height"])
    width = int(camera_info["width"])

    # The canonical reachability frame uses x forward, y left, z up. Reject a
    # calibration that does not put a point in front of the robot in front of
    # the camera, because mission tf.yaml uses a different base-axis convention.
    test_point = np.array([1.0, 0.0, -0.57, 1.0])
    if float((E @ test_point)[2]) <= 0.1:
        raise ValueError(
            "T_camera_base is incompatible with canonical x-forward reachability axes"
        )
    return K, D[:4], E, height, width, intrinsics_path


def _project_equidistant(
    xyz_base: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(xyz_base, dtype=np.float64)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    xyz_camera = (E @ homogeneous.T).T[:, :3]
    x_camera, y_camera, z_camera = xyz_camera.T

    radius = np.hypot(x_camera, y_camera)
    theta = np.arctan2(radius, z_camera)
    theta2 = theta * theta
    distorted_radius = theta * (
        1.0
        + D[0] * theta2
        + D[1] * theta2**2
        + D[2] * theta2**3
        + D[3] * theta2**4
    )
    safe_radius = np.where(radius > 1e-12, radius, 1.0)
    u = K[0, 0] * distorted_radius * x_camera / safe_radius + K[0, 2]
    v = K[1, 1] * distorted_radius * y_camera / safe_radius + K[1, 2]
    visible = (
        (z_camera > 0.1)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < image_width)
        & (v >= 0.0)
        & (v < image_height)
    )
    return u, v, visible


def _resolve_row(group, row: int | None, image_id: int | None) -> int:
    frame_count = int(group["elevation"].shape[0])
    if row is not None:
        if not 0 <= row < frame_count:
            raise IndexError(f"row {row} outside [0, {frame_count})")
        return int(row)
    if image_id is not None:
        ids = np.asarray(group["image_id"][:], dtype=np.int64)
        matches = np.flatnonzero(ids == int(image_id))
        if len(matches) != 1:
            raise KeyError(
                f"image_id={image_id}: expected exactly one elevation row, got {len(matches)}"
            )
        return int(matches[0])
    return frame_count // 2


def _camera_path(mission: Path, image_id: int) -> Path:
    candidates = (
        mission / "images" / "hdr_front" / f"{image_id:06d}.jpeg",
        mission / "images" / "hdr_front" / f"{image_id:06d}.jpg",
        mission / "images" / "hdr_front" / f"{image_id}.jpeg",
        mission / "images" / "hdr_front" / f"{image_id}.jpg",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"hdr_front image not found for image_id={image_id}")
    return path


def _geometry(mission: Path, height: int, width: int) -> MapGeometry:
    labels_path = mission / "reachability_5labels"
    if labels_path.is_dir():
        labels = zarr.open_group(str(labels_path), mode="r")
        resolution = float(labels.attrs.get("map_resolution_m", 0.04))
        default_origin = (
            -0.5 * height * resolution,
            -0.5 * width * resolution,
        )
        origin = labels.attrs.get("map_origin_xy_m", default_origin)
    else:
        resolution = 0.04
        origin = (-0.5 * height * resolution, -0.5 * width * resolution)
    return MapGeometry(
        height=height,
        width=width,
        resolution=resolution,
        origin_xy=(float(origin[0]), float(origin[1])),
    )


def _format_bev_axis(ax, geometry: MapGeometry) -> None:
    left, right, bottom, top = geometry.imshow_extent_yx
    ax.set_xlim(right, left)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]  (robot left +)")
    ax.set_ylabel("x [m]  (robot forward +)")
    ax.axhline(0.0, color="white", alpha=0.4, linewidth=0.8)
    ax.axvline(0.0, color="white", alpha=0.4, linewidth=0.8)
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=140,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=10,
    )


def main() -> None:
    default_extrinsics = (
        Path(__file__).resolve().parents[2]
        / "limo"
        / "configs"
        / "model"
        / "camera_info.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Visualize hdr_front geometric FOV on one elevation frame."
    )
    parser.add_argument("--mission-dir", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--row", type=int, default=None)
    selector.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ground-z", type=float, default=-0.57)
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--extrinsics-config", type=Path, default=default_extrinsics)
    args = parser.parse_args()

    if args.point_stride <= 0:
        raise ValueError("point_stride must be positive")
    mission = args.mission_dir.expanduser().resolve()
    elevation_path = mission / "data" / "elevation_map"
    if not elevation_path.is_dir():
        raise FileNotFoundError(f"elevation_map not found: {elevation_path}")
    group = zarr.open_group(str(elevation_path), mode="r")
    if "elevation" not in group or "image_id" not in group:
        raise KeyError("elevation_map must contain elevation and image_id")

    row = _resolve_row(group, args.row, args.image_id)
    elevation = np.asarray(group["elevation"][row], dtype=np.float32)
    if elevation.ndim != 2:
        raise ValueError(f"expected 2-D elevation, got shape {elevation.shape}")
    image_id = int(group["image_id"][row])
    image_path = _camera_path(mission, image_id)
    with Image.open(image_path) as image:
        front_image = np.array(image.convert("RGB"))

    K, D, E, camera_height, camera_width, intrinsics_path = _load_calibration(
        mission,
        args.extrinsics_config.expanduser().resolve(),
    )
    if front_image.shape[:2] != (camera_height, camera_width):
        raise ValueError(
            f"front image shape {front_image.shape[:2]} does not match camera metadata "
            f"{(camera_height, camera_width)}"
        )

    height, width = elevation.shape
    geometry = _geometry(mission, height, width)
    x_coordinates = geometry.origin_xy[0] + np.arange(height) * geometry.resolution
    y_coordinates = geometry.origin_xy[1] + np.arange(width) * geometry.resolution
    xx, yy = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")
    zz = np.where(np.isfinite(elevation), elevation, float(args.ground_z))
    xyz_base = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    u, v, visible_flat = _project_equidistant(
        xyz_base,
        K,
        D,
        E,
        camera_height,
        camera_width,
    )
    visible = visible_flat.reshape(height, width)
    front_ignore = ~visible

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path.cwd()
        / f"{mission.name}_row_{row:06d}_image_{image_id:06d}_front_fov.png"
    )
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

    camera_ax = axes[0]
    camera_ax.imshow(front_image)
    sampled = np.zeros((height, width), dtype=bool)
    sampled[:: args.point_stride, :: args.point_stride] = True
    sampled_flat = sampled.reshape(-1) & visible_flat
    scatter = camera_ax.scatter(
        u[sampled_flat],
        v[sampled_flat],
        c=xyz_base[sampled_flat, 0],
        cmap="turbo",
        s=7,
        alpha=0.72,
        linewidths=0,
    )
    camera_ax.set_xlim(0, camera_width)
    camera_ax.set_ylim(camera_height, 0)
    camera_ax.set_title("hdr_front + projected visible BEV cells\ncolor = robot x forward [m]")
    camera_ax.axis("off")
    fig.colorbar(scatter, ax=camera_ax, fraction=0.046, pad=0.03, label="x forward [m]")

    elevation_ax = axes[1]
    elevation_cmap = plt.get_cmap("terrain").copy()
    elevation_cmap.set_bad("#8c8c8c")
    elevation_image = elevation_ax.imshow(
        np.ma.masked_where(~np.isfinite(elevation), elevation),
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=elevation_cmap,
        aspect="equal",
    )
    elevation_ax.contourf(
        visible.astype(np.uint8),
        levels=[0.5, 1.5],
        origin="lower",
        extent=geometry.imshow_extent_yx,
        colors=["#00ff66"],
        alpha=0.25,
    )
    elevation_ax.contour(
        visible.astype(np.uint8),
        levels=[0.5],
        origin="lower",
        extent=geometry.imshow_extent_yx,
        colors=["#00ff66"],
        linewidths=[1.8],
    )
    elevation_ax.set_title("Elevation + front-visible region (green)")
    _format_bev_axis(elevation_ax, geometry)
    fig.colorbar(
        elevation_image,
        ax=elevation_ax,
        fraction=0.046,
        pad=0.03,
        label="elevation [m]",
    )

    mask_ax = axes[2]
    mask_image = mask_ax.imshow(
        front_ignore.astype(np.uint8),
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=ListedColormap(("#2ca25f", "#8c8c8c")),
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    visible_count = int(visible.sum())
    mask_ax.set_title(
        "Proposed front_ignore\n"
        f"False/visible={visible_count} ({visible.mean():.1%}), "
        f"True/ignore={front_ignore.size - visible_count}"
    )
    _format_bev_axis(mask_ax, geometry)
    colorbar = fig.colorbar(
        mask_image,
        ax=mask_ax,
        fraction=0.046,
        pad=0.03,
        ticks=(0, 1),
    )
    colorbar.ax.set_yticklabels(("0 visible", "1 ignore"))

    timestamp = (
        float(group["timestamp"][row]) if "timestamp" in group else None
    )
    timestamp_text = "" if timestamp is None else f" | timestamp={timestamp:.6f}"
    fig.suptitle(
        f"mission={mission.name} | elevation_row={row} | image_id={image_id}"
        f"{timestamp_text}\n"
        f"intrinsics={intrinsics_path.name} | ground fallback z={args.ground_z:.2f} m | "
        "geometric FOV only (no occlusion)",
        fontsize=14,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {output}")
    print(
        f"visible cells: {visible_count}/{visible.size} ({visible.mean():.3%}); "
        f"front_ignore cells: {int(front_ignore.sum())}"
    )


if __name__ == "__main__":
    main()
