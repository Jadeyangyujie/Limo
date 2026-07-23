"""Visualize static camera-FOV ignore masks with aligned elevation and images."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"limo-matplotlib-{os.getuid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from tqdm import tqdm
import zarr


CAMERAS = ("hdr_left", "hdr_front", "hdr_right")
MASK_KEYS = ("front_ignore", "multiview_ignore")


@dataclass(frozen=True)
class MapGeometry:
    height: int
    width: int
    resolution: float
    origin_xy: tuple[float, float]

    @property
    def imshow_extent_yx(self) -> tuple[float, float, float, float]:
        ox, oy = self.origin_xy
        half = 0.5 * self.resolution
        return (
            oy - half,
            oy + (self.width - 1) * self.resolution + half,
            ox - half,
            ox + (self.height - 1) * self.resolution + half,
        )


def _camera_path(mission: Path, camera: str, image_id: int) -> Path | None:
    candidates = (
        mission / "images" / camera / f"{image_id:06d}.jpeg",
        mission / "images" / camera / f"{image_id:06d}.jpg",
        mission / "images" / camera / f"{image_id}.jpeg",
        mission / "images" / camera / f"{image_id}.jpg",
    )
    return next((path for path in candidates if path.is_file()), None)


def _validate_and_load(mission: Path):
    labels_path = mission / "reachability_5labels"
    elevation_path = mission / "data" / "elevation_map"
    if not labels_path.is_dir():
        raise FileNotFoundError(f"reachability_5labels not found: {labels_path}")
    if not elevation_path.is_dir():
        raise FileNotFoundError(f"elevation_map not found: {elevation_path}")

    labels = zarr.open_group(str(labels_path), mode="r")
    elevation = zarr.open_group(str(elevation_path), mode="r")
    label_keys = set(labels.array_keys())
    missing = {"state_label", "image_id", *MASK_KEYS} - label_keys
    if missing:
        raise KeyError(f"{mission.name}: missing label arrays {sorted(missing)}")
    if "elevation" not in elevation or "image_id" not in elevation:
        raise KeyError(f"{mission.name}: elevation_map requires elevation and image_id")

    state_shape = labels["state_label"].shape
    if len(state_shape) != 3:
        raise ValueError(f"state_label must have shape [N,H,W], got {state_shape}")
    n, height, width = map(int, state_shape)
    if elevation["elevation"].shape != state_shape:
        raise ValueError(
            f"elevation shape {elevation['elevation'].shape} != {state_shape}"
        )
    for key in MASK_KEYS:
        if labels[key].shape != (height, width):
            raise ValueError(
                f"{key} shape {labels[key].shape} != {(height, width)}"
            )
        if labels[key].dtype != np.dtype(bool):
            raise TypeError(f"{key} dtype {labels[key].dtype} != bool")

    image_ids = np.asarray(labels["image_id"][:])
    elevation_ids = np.asarray(elevation["image_id"][:])
    if image_ids.shape != (n,) or not np.array_equal(image_ids, elevation_ids):
        raise ValueError(
            "reachability_5labels/image_id is not exactly aligned with "
            "data/elevation_map/image_id"
        )

    resolution = float(labels.attrs.get("map_resolution_m", 0.04))
    default_origin = (-0.5 * height * resolution, -0.5 * width * resolution)
    origin = labels.attrs.get("map_origin_xy_m", default_origin)
    geometry = MapGeometry(
        height=height,
        width=width,
        resolution=resolution,
        origin_xy=(float(origin[0]), float(origin[1])),
    )
    masks = {
        key: np.asarray(labels[key][:], dtype=bool)
        for key in MASK_KEYS
    }
    return labels, elevation, image_ids, geometry, masks


def _choose_random_rows(
    mission: Path,
    image_ids: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
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
            f"{mission.name}: requested {count}, but only {len(eligible)} rows "
            "have all three exact image_id camera files"
        )
    return np.sort(rng.choice(eligible, size=count, replace=False))


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
        s=130,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=10,
    )


def _plot_elevation(ax, elevation: np.ndarray, geometry: MapGeometry):
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad("#8c8c8c")
    image = ax.imshow(
        np.ma.masked_where(~np.isfinite(elevation), elevation),
        origin="lower",
        extent=geometry.imshow_extent_yx,
        cmap=cmap,
        aspect="equal",
    )
    _format_bev_axis(ax, geometry)
    return image


def _overlay_mask(
    ax,
    mask: np.ndarray,
    geometry: MapGeometry,
    color: tuple[float, float, float],
    alpha: float = 0.58,
) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask, :3] = color
    rgba[mask, 3] = alpha
    ax.imshow(
        rgba,
        origin="lower",
        extent=geometry.imshow_extent_yx,
        interpolation="nearest",
        aspect="equal",
    )
    ax.contour(
        mask.astype(np.uint8),
        levels=[0.5],
        origin="lower",
        extent=geometry.imshow_extent_yx,
        colors=[color],
        linewidths=[1.3],
    )


def _plot_one(
    mission: Path,
    labels,
    elevation_group,
    row: int,
    image_id: int,
    geometry: MapGeometry,
    masks: dict[str, np.ndarray],
    output: Path,
) -> dict:
    elevation = np.asarray(elevation_group["elevation"][row], dtype=np.float32)
    camera_paths = {
        camera: _camera_path(mission, camera, image_id) for camera in CAMERAS
    }
    if any(path is None for path in camera_paths.values()):
        raise FileNotFoundError(
            f"{mission.name}: missing camera for row={row}, image_id={image_id}"
        )

    fig = plt.figure(figsize=(22, 13), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.72, 1.0))
    for column, camera in enumerate(CAMERAS):
        ax = fig.add_subplot(grid[0, column])
        with Image.open(camera_paths[camera]) as image:
            ax.imshow(np.asarray(image.convert("RGB")))
        ax.set_title(f"{camera}  |  image_id={image_id}", fontsize=12)
        ax.axis("off")

    elevation_ax = fig.add_subplot(grid[1, 0])
    elevation_image = _plot_elevation(elevation_ax, elevation, geometry)
    elevation_ax.set_title("Elevation map")
    fig.colorbar(
        elevation_image,
        ax=elevation_ax,
        fraction=0.046,
        pad=0.04,
        label="elevation [m]",
    )

    front_ax = fig.add_subplot(grid[1, 1])
    _plot_elevation(front_ax, elevation, geometry)
    _overlay_mask(front_ax, masks["front_ignore"], geometry, (0.90, 0.12, 0.12))
    front_count = int(masks["front_ignore"].sum())
    front_ax.set_title(
        "front_ignore overlay (red = ignored)\n"
        f"ignored={front_count}/{masks['front_ignore'].size} "
        f"({front_count / masks['front_ignore'].size:.1%})"
    )
    front_ax.legend(
        handles=[Patch(facecolor="#e61f1f", alpha=0.58, label="front_ignore=True")],
        loc="upper right",
    )

    multi_ax = fig.add_subplot(grid[1, 2])
    _plot_elevation(multi_ax, elevation, geometry)
    _overlay_mask(
        multi_ax,
        masks["multiview_ignore"],
        geometry,
        (0.49, 0.18, 0.72),
    )
    multi_count = int(masks["multiview_ignore"].sum())
    multi_ax.set_title(
        "multiview_ignore overlay (purple = ignored)\n"
        f"ignored={multi_count}/{masks['multiview_ignore'].size} "
        f"({multi_count / masks['multiview_ignore'].size:.1%})"
    )
    multi_ax.legend(
        handles=[
            Patch(
                facecolor="#7d2eb8",
                alpha=0.58,
                label="multiview_ignore=True",
            )
        ],
        loc="upper right",
    )

    timestamp = (
        float(labels["timestamp"][row]) if "timestamp" in labels else None
    )
    timestamp_text = "" if timestamp is None else f" | timestamp={timestamp:.6f}"
    fig.suptitle(
        f"mission={mission.name} | row={row} | image_id={image_id}"
        f"{timestamp_text}\n"
        "Static geometric FOV masks on reference ground z=-0.57 m; "
        "no terrain/dynamic occlusion",
        fontsize=15,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "row": int(row),
        "image_id": int(image_id),
        "timestamp": timestamp,
        "output": str(output),
        "front_ignore_count": front_count,
        "multiview_ignore_count": multi_count,
    }


def _replace_output(temp_output: Path, output: Path, overwrite: bool) -> None:
    if not output.exists():
        temp_output.rename(output)
        return
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly visualize static front/multiview ignore masks with "
            "aligned elevation and exact image_id camera images."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path)
    source.add_argument("--mission-dir", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.dataset_dir is not None:
        dataset = args.dataset_dir.expanduser().resolve()
        missions = sorted(
            path
            for path in dataset.iterdir()
            if path.is_dir() and (path / "reachability_5labels").is_dir()
        )
    else:
        missions = [
            path.expanduser().resolve() for path in args.mission_dir
        ]
    if not missions:
        raise FileNotFoundError("no mission containing reachability_5labels found")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {output_root}; pass --overwrite to replace it"
        )
    temp_root = output_root.with_name(
        f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    )
    rng = np.random.default_rng(args.seed)
    try:
        temp_root.mkdir(parents=True)
        all_records = []
        for mission in tqdm(missions, desc="missions", unit="mission"):
            labels, elevation, image_ids, geometry, masks = _validate_and_load(
                mission
            )
            selected_rows = _choose_random_rows(
                mission, image_ids, args.count, rng
            )
            mission_output = temp_root / mission.name
            mission_output.mkdir()
            records = []
            for row in tqdm(
                selected_rows,
                desc=mission.name,
                unit="figure",
                leave=False,
            ):
                image_id = int(image_ids[row])
                filename = (
                    f"{mission.name}_row_{int(row):06d}"
                    f"_image_{image_id:06d}_ignore_masks.png"
                )
                records.append(
                    _plot_one(
                        mission,
                        labels,
                        elevation,
                        int(row),
                        image_id,
                        geometry,
                        masks,
                        mission_output / filename,
                    )
                )
            (mission_output / "manifest.json").write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            all_records.append(
                {
                    "mission": mission.name,
                    "count": len(records),
                    "rows": [record["row"] for record in records],
                    "image_ids": [record["image_id"] for record in records],
                }
            )
        (temp_root / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "count_per_mission": args.count,
                    "missions": all_records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _replace_output(temp_root, output_root, args.overwrite)
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "mission_count": len(missions),
                "count_per_mission": args.count,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
