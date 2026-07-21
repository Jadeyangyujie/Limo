"""Plot left/front/right camera views and the matching elevation map.

Example:
    python -m dataset_builder.src.plot_three_views_elevation \
      --mission /path/to/LIMO_DATASET/2024-11-02-21-12-51 \
      --image-id 500 --output three_views_500.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Circle

from dataset_builder.src.mission_data_source import GrandTourZarrSource
from dataset_builder.src.visualize import _project_fisheye, load_cameras


def _show_camera(ax, image, title: str) -> None:
    if image is None:
        ax.text(0.5, 0.5, "image unavailable", ha="center", va="center")
        ax.set_facecolor("0.15")
    else:
        ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def plot_three_views_elevation(
    mission: str | Path,
    image_id: int,
    output: str | Path,
    *,
    map_size: float = 4.0,
    map_resolution: float = 0.04,
    dpi: int = 180,
    path_source: str | None = None,
    radius_m: float = 0.26,
) -> Path:
    source = GrandTourZarrSource(mission, map_size, map_resolution)
    mission = Path(mission)
    front_group = zarr.open_group(str(mission / "data" / "hdr_front"), mode="r")
    # hdr_front stores timestamps/sequence_id but not an image_id column;
    # in this dataset the camera image filename/index is the image_id.
    front_row = int(image_id)
    if front_row < 0 or front_row >= len(front_group["timestamp"]):
        raise KeyError(f"hdr_front image_id/index out of range: {image_id}")
    timestamp = float(front_group["timestamp"][front_row])

    def same_image_id_camera(camera: str):
        group = zarr.open_group(str(mission / "data" / camera), mode="r")
        timestamps = np.asarray(group["timestamp"], dtype=np.float64)
        image_path = mission / "images" / camera / f"{image_id:06d}.jpeg"
        if not image_path.exists():
            return None, image_id, np.nan
        from PIL import Image

        camera_timestamp = float(timestamps[image_id]) if image_id < len(timestamps) else np.nan
        return np.asarray(Image.open(image_path).convert("RGB")), image_id, camera_timestamp

    # Camera filenames are the shared image_id namespace in this dataset.
    # Do not nearest-timestamp match here: sequence_id values differ between
    # cameras, while images/<camera>/<image_id>.jpeg is the intended join key.
    left, left_row, left_timestamp = same_image_id_camera("hdr_left")
    right, right_row, right_timestamp = same_image_id_camera("hdr_right")
    front = source.get_image(front_row)

    elevation_group = zarr.open_group(str(mission / "data" / "elevation_map"), mode="r")
    elevation_ids = np.asarray(elevation_group["image_id"], dtype=np.int64)
    elevation_matches = np.flatnonzero(elevation_ids == image_id)
    if len(elevation_matches) != 1:
        raise KeyError(
            f"expected one elevation image_id={image_id}, found {len(elevation_matches)}"
        )
    elevation_row = int(elevation_matches[0])
    elevation = np.asarray(elevation_group["elevation"][elevation_row], dtype=np.float32)

    path_groups = {}
    for source_name in ("geometric", "teleop"):
        if path_source is not None and source_name != path_source:
            continue
        path_group_path = mission / "data" / f"{source_name}_paths"
        if not path_group_path.exists():
            continue
        group = zarr.open_group(str(path_group_path), mode="r")
        ids = np.asarray(group["image_id"], dtype=np.int64)
        rows = np.flatnonzero(ids == image_id)
        path_groups[source_name] = (group, rows)
    finite = np.isfinite(elevation)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bdbdbd")
    masked_elevation = np.ma.masked_where(~finite, elevation)

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1, 1.1))
    camera_axis = fig.add_subplot(grid[0, 0])
    camera_axis.axis("off")
    # Concatenate the three views horizontally after resizing to a common height.
    from PIL import Image

    camera_images = [left, front, right]
    available = [image for image in camera_images if image is not None]
    if not available:
        raise RuntimeError("no camera images found")
    common_height = min(image.shape[0] for image in available)
    resized = []
    for image in camera_images:
        if image is None:
            resized.append(np.full((common_height, common_height, 3), 35, dtype=np.uint8))
            continue
        width = int(round(image.shape[1] * common_height / image.shape[0]))
        resized.append(np.asarray(Image.fromarray(image).resize((width, common_height))))
    camera_strip = np.concatenate(resized, axis=1)
    camera_axis.imshow(camera_strip)
    camera_axis.set_title(
        f"left | front | right   shared image_id={image_id}  "
        f"rows={left_row}/{front_row}/{right_row}"
    )

    ax = fig.add_subplot(grid[1, 0])
    extent = (-map_size, map_size, -map_size, map_size)
    image = ax.imshow(
        masked_elevation,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="equal",
    )
    ax.scatter([0], [0], marker="+", c="red", s=100, linewidths=1.5, label="robot")

    # Project every elevation-map cell at the map's ground/reference height
    # into each camera.  A cell is in that camera's radiative footprint when
    # its fisheye projection is in front of the camera and inside the image.
    camera_shapes = {
        "front": front.shape[:2],
        "left": left.shape[:2] if left is not None else front.shape[:2],
        "right": right.shape[:2] if right is not None else front.shape[:2],
    }
    cameras = load_cameras(mission, img_w=front.shape[1], img_h=front.shape[0])
    axis_values = np.linspace(-map_size + map_resolution / 2, map_size - map_resolution / 2, elevation.shape[0])
    xx, yy = np.meshgrid(axis_values, axis_values, indexing="ij")
    map_points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -0.57)])
    camera_colors = {"front": "#ff7f0e", "left": "#1f77b4", "right": "#2ca02c"}
    for camera_name in ("front", "left", "right"):
        camera = cameras.get(camera_name)
        if camera is None:
            continue
        pixels = _project_fisheye(map_points, camera, camera_shapes[camera_name])
        visible = np.isfinite(pixels[:, 0]).reshape(elevation.shape)
        color = camera_colors[camera_name]
        # Use translucent filled footprints plus a boundary contour.
        ax.contourf(yy, xx, visible.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.16)
        ax.contour(yy, xx, visible.astype(float), levels=[0.5], colors=[color], linewidths=1.3)
        ax.plot([], [], color=color, linewidth=3, label=f"{camera_name} FOV")

    # Draw all geometric/teleop paths belonging to this image_id, with a
    # sparse r=0.26 m circular footprint along each path.
    path_colors = {"geometric": "#e41a1c", "teleop": "#984ea3"}
    for source_name, (group, rows) in path_groups.items():
        for local_index, row in enumerate(rows):
            path = np.asarray(group["path"][int(row)], dtype=np.float32)
            goal = np.asarray(group["goal"][int(row)], dtype=np.float32)
            color = path_colors[source_name]
            label = f"{source_name} path" if local_index == 0 else None
            ax.plot(path[:, 1], path[:, 0], color=color, linewidth=1.5, alpha=0.8, label=label)
            ax.plot(goal[1], goal[0], marker="D", color=color, markersize=4, alpha=0.9)
            # Keep roughly five to eight circles per path for readability.
            step = max(1, len(path) // 7)
            for px, py, _ in path[::step]:
                ax.add_patch(
                    Circle(
                        (float(py), float(px)),
                        radius_m,
                        fill=False,
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.45,
                    )
                )
    ax.set_title(f"elevation map + paths + circular footprint r={radius_m:.2f} m (robot frame)")
    ax.set_xlabel("y left [m]")
    ax.set_ylabel("x forward [m]")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="elevation")
    fig.suptitle(
        f"image_id={image_id}  elevation_row={elevation_row}  "
        f"timestamp={timestamp:.3f}s",
        fontsize=14,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-size", type=float, default=4.0)
    parser.add_argument("--map-resolution", type=float, default=0.04)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--path-source", choices=("geo", "tel", "geometric", "teleop"), default=None)
    parser.add_argument("--radius", type=float, default=0.26)
    args = parser.parse_args()
    source = args.path_source
    if source == "geo":
        source = "geometric"
    elif source == "tel":
        source = "teleop"
    result = plot_three_views_elevation(
        args.mission,
        args.image_id,
        args.output,
        map_size=args.map_size,
        map_resolution=args.map_resolution,
        dpi=args.dpi,
        path_source=source,
        radius_m=args.radius,
    )
    print(result)


if __name__ == "__main__":
    main()
