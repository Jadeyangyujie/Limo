#!/usr/bin/env python3
"""Visualize PTC-LIMO ptc_labels zarr groups.

This is the PTC counterpart of the older BEV label debug visualizer. It reads
<mission>/ptc_labels, shows synchronized camera images when available, and
plots the privileged risk/valid maps used by PTC trajectory-cost training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from generate_ptc_labels_from_bev import ZarrV2Array


DEFAULT_DATASET_ROOT = Path("/home/robot-device/yangyujie/BEV_LIMO/LIMO_DATASET")
DEFAULT_GROUP_NAME = "ptc_labels"
plt = None
mpimg = None


def setup_matplotlib() -> None:
    global plt, mpimg
    if plt is not None and mpimg is not None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as matplotlib_image
        import matplotlib.pyplot as matplotlib_pyplot
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "visualize_ptc_labels.py requires matplotlib to write PNG files. "
            "Run it inside the project environment that has matplotlib installed, "
            "or install matplotlib in the current Python environment."
        ) from exc
    plt = matplotlib_pyplot
    mpimg = matplotlib_image


def timestamp_to_float_seconds(ts: Any) -> float:
    ts = float(ts)
    abs_ts = abs(ts)
    if abs_ts > 1e17:
        return ts / 1e9
    if abs_ts > 1e14:
        return ts / 1e6
    if abs_ts > 1e11:
        return ts / 1e3
    return ts


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_optional_array(array_path: Path) -> ZarrV2Array | None:
    if not (array_path / ".zarray").exists():
        return None
    return ZarrV2Array(array_path)


def read_scalar(arr: ZarrV2Array, idx: int) -> Any:
    return np.asarray(arr[idx]).reshape(-1)[0]


def read_map(arr: ZarrV2Array, idx: int, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(arr[idx], dtype=dtype)[0]


def zarr_array_to_seconds(arr: ZarrV2Array) -> np.ndarray:
    x = np.asarray(arr[:], dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    med = np.nanmedian(np.abs(x))
    if med > 1e17:
        return x / 1e9
    if med > 1e14:
        return x / 1e6
    if med > 1e11:
        return x / 1e3
    return x


def nearest_index(sorted_ts: np.ndarray, target_ts: float) -> int | None:
    if len(sorted_ts) == 0:
        return None
    j = int(np.searchsorted(sorted_ts, target_ts))
    candidates: list[int] = []
    if 0 <= j < len(sorted_ts):
        candidates.append(j)
    if 0 <= j - 1 < len(sorted_ts):
        candidates.append(j - 1)
    if not candidates:
        return None
    return min(candidates, key=lambda k: abs(float(sorted_ts[k]) - float(target_ts)))


def find_image_file_by_index(img_dir: Path, image_index: int) -> Path | None:
    image_index = int(image_index)
    candidates = [
        img_dir / f"{image_index:06d}.jpeg",
        img_dir / f"{image_index:06d}.jpg",
        img_dir / f"{image_index:06d}.png",
        img_dir / f"{image_index}.jpeg",
        img_dir / f"{image_index}.jpg",
        img_dir / f"{image_index}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_topic_timestamp_by_row(
    mission_dir: Path,
    topic: str,
    row_idx: int,
) -> tuple[float, str]:
    ts_arr = read_optional_array(mission_dir / "data" / topic / "timestamp")
    if ts_arr is None:
        return np.nan, "missing"
    if not (0 <= int(row_idx) < ts_arr.shape[0]):
        return np.nan, "out_of_range"
    return (
        timestamp_to_float_seconds(read_scalar(ts_arr, int(row_idx))),
        f"data/{topic}/timestamp[{row_idx}]",
    )


def get_topic_sequence_id_by_row(
    mission_dir: Path,
    topic: str,
    row_idx: int,
) -> int | None:
    seq_arr = read_optional_array(mission_dir / "data" / topic / "sequence_id")
    if seq_arr is None or not (0 <= int(row_idx) < seq_arr.shape[0]):
        return None
    return int(read_scalar(seq_arr, int(row_idx)))


def find_nearest_image_by_timestamp(
    mission_dir: Path,
    view: str,
    target_ts: float,
    max_dt: float,
) -> dict[str, Any]:
    ts_arr = read_optional_array(mission_dir / "data" / view / "timestamp")
    seq_arr = read_optional_array(mission_dir / "data" / view / "sequence_id")
    if ts_arr is None:
        return {
            "img_path": None,
            "row_idx": None,
            "sequence_id": None,
            "timestamp": np.nan,
            "dt": np.nan,
            "status": "missing_timestamp",
        }

    ts_sec = zarr_array_to_seconds(ts_arr)
    j = nearest_index(ts_sec, target_ts)
    if j is None:
        return {
            "img_path": None,
            "row_idx": None,
            "sequence_id": None,
            "timestamp": np.nan,
            "dt": np.nan,
            "status": "empty_timestamp",
        }

    dt = float(ts_sec[j] - target_ts)
    if abs(dt) > max_dt:
        return {
            "img_path": None,
            "row_idx": int(j),
            "sequence_id": int(read_scalar(seq_arr, j)) if seq_arr is not None else None,
            "timestamp": float(ts_sec[j]),
            "dt": dt,
            "status": "dt_too_large",
        }

    img_path = find_image_file_by_index(mission_dir / "images" / view, int(j))
    return {
        "img_path": img_path,
        "row_idx": int(j),
        "sequence_id": int(read_scalar(seq_arr, j)) if seq_arr is not None else None,
        "timestamp": float(ts_sec[j]),
        "dt": dt,
        "status": "ok" if img_path is not None else "image_not_found",
    }


def get_camera_debug_info(
    mission_dir: Path,
    view: str,
    image_id: int,
    ptc_ts: float,
    use_timestamp_fallback: bool,
    max_dt: float,
) -> dict[str, Any]:
    image_id = int(image_id)
    img_dir = mission_dir / "images" / view

    direct_path = find_image_file_by_index(img_dir, image_id)
    direct_ts, direct_ts_src = get_topic_timestamp_by_row(mission_dir, view, image_id)
    direct_seq = get_topic_sequence_id_by_row(mission_dir, view, image_id)

    info: dict[str, Any] = {
        "view": view,
        "mode": "direct_image_id",
        "img_path": direct_path,
        "row_idx": image_id,
        "sequence_id": direct_seq,
        "timestamp": direct_ts,
        "timestamp_source": direct_ts_src,
        "dt_to_ptc": (
            float(direct_ts - ptc_ts)
            if np.isfinite(direct_ts) and np.isfinite(ptc_ts)
            else np.nan
        ),
        "fallback": None,
    }

    if direct_path is not None:
        return info

    if use_timestamp_fallback and np.isfinite(ptc_ts):
        fallback = find_nearest_image_by_timestamp(
            mission_dir=mission_dir,
            view=view,
            target_ts=ptc_ts,
            max_dt=max_dt,
        )
        info["fallback"] = fallback
        if fallback["img_path"] is not None:
            info.update(
                {
                    "mode": "timestamp_nearest_row_fallback",
                    "img_path": fallback["img_path"],
                    "row_idx": fallback["row_idx"],
                    "sequence_id": fallback["sequence_id"],
                    "timestamp": fallback["timestamp"],
                    "timestamp_source": (
                        f"nearest data/{view}/timestamp[{fallback['row_idx']}]"
                    ),
                    "dt_to_ptc": fallback["dt"],
                }
            )
    return info


def show_camera_image(ax: Any, cam_info: dict[str, Any], title_prefix: str) -> None:
    ax.axis("off")
    img_path = cam_info.get("img_path")
    row_idx = cam_info.get("row_idx")
    seq_id = cam_info.get("sequence_id")
    dt = cam_info.get("dt_to_ptc", np.nan)
    mode = cam_info.get("mode", "unknown")
    name = img_path.name if img_path is not None else "image not found"
    ax.set_title(
        f"{title_prefix}\nmode={mode}\nrow/image_id={row_idx}, "
        f"seq={seq_id}, dt={dt:+.6f}s\n{name}",
        fontsize=9,
    )
    if img_path is None:
        ax.text(
            0.5,
            0.5,
            f"Image Not Found\nrow/image_id={row_idx}\nseq={seq_id}",
            ha="center",
            va="center",
            color="red",
            fontsize=10,
        )
        return
    try:
        ax.imshow(mpimg.imread(str(img_path)))
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"Read Failed\n{img_path.name}\n{repr(exc)}",
            ha="center",
            va="center",
            color="red",
            fontsize=10,
        )


def orient_bev_for_display(arr: np.ndarray, flip_lr: bool = True) -> np.ndarray:
    arr_vis = arr
    if flip_lr:
        arr_vis = np.fliplr(arr_vis)
    return arr_vis


def plot_bev(
    ax: Any,
    arr: np.ndarray,
    title: str,
    origin: str = "lower",
    flip_lr: bool = True,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    im = ax.imshow(
        orient_bev_for_display(arr, flip_lr=flip_lr),
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.nan
    return float(np.mean(arr[mask]))


def parse_rows_arg(rows: str | None) -> list[int] | None:
    if rows is None or rows.strip() == "":
        return None
    return [int(x.strip()) for x in rows.split(",") if x.strip()]


def ptc_array_paths(ptc_dir: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in ptc_dir.iterdir()
        if path.is_dir() and (path / ".zarray").exists()
    }


def load_ptc_arrays(ptc_dir: Path) -> dict[str, ZarrV2Array]:
    paths = ptc_array_paths(ptc_dir)
    return {name: ZarrV2Array(path) for name, path in paths.items()}


def build_sampling_indices(
    arrays: dict[str, ZarrV2Array],
    num_samples: int,
    seed: int,
    only_used_by_paths: bool,
    min_path_count: int,
    explicit_rows: list[int] | None,
) -> np.ndarray:
    image_id = arrays["image_id"]
    n = image_id.shape[0]
    mask = np.ones(n, dtype=bool)

    if "frame_valid" in arrays:
        frame_valid = np.asarray(arrays["frame_valid"][:], dtype=np.uint8).reshape(-1)
        mask &= frame_valid > 0

    if only_used_by_paths and "path_sample_count" in arrays:
        path_count = np.asarray(arrays["path_sample_count"][:], dtype=np.int64).reshape(-1)
        mask &= path_count >= int(min_path_count)

    if explicit_rows:
        rows = np.asarray(explicit_rows, dtype=np.int64)
        rows = rows[(rows >= 0) & (rows < n)]
        rows = rows[mask[rows]]
        return rows.astype(np.int64)

    valid_indices = np.where(mask)[0]
    if valid_indices.size == 0:
        raise RuntimeError(
            "No rows selected. Try removing --only-used-by-paths or lowering --min-path-count."
        )
    rng = np.random.default_rng(seed)
    if valid_indices.size > num_samples:
        return np.asarray(rng.choice(valid_indices, size=num_samples, replace=False), dtype=np.int64)
    return valid_indices.astype(np.int64)


def resolve_ptc_dir(args: argparse.Namespace) -> Path:
    if args.ptc_dir is not None:
        return args.ptc_dir.expanduser()
    if args.dataset_root is None or args.mission is None:
        raise ValueError("Either --ptc-dir or both --dataset-root and --mission are required.")
    return args.dataset_root.expanduser() / args.mission / args.group_name


def default_save_dir(ptc_dir: Path) -> Path:
    mission = ptc_dir.parent.name
    return Path(__file__).resolve().parent / "ptc_labels_vis" / mission


def blank_map_like(reference: np.ndarray) -> np.ndarray:
    return np.full(reference.shape, np.nan, dtype=np.float32)


def plot_debug_samples_with_cams(
    ptc_dir: Path,
    save_dir: Path,
    num_samples: int,
    seed: int,
    risk_threshold: float,
    bev_origin: str,
    flip_lr_for_display: bool,
    use_timestamp_side_view_fallback: bool,
    max_side_view_dt: float,
    only_used_by_paths: bool,
    min_path_count: int,
    rows: str | None,
) -> list[Path]:
    setup_matplotlib()
    ptc_dir = ptc_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    mission_dir = ptc_dir.parent

    arrays = load_ptc_arrays(ptc_dir)
    required = ["image_id", "timestamp", "source_row", "risk_map", "valid_mask"]
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Missing required arrays in {ptc_dir}: {missing}")

    attrs = read_json(ptc_dir / ".zattrs")
    summary = read_json(ptc_dir / "summary.json")

    print("=" * 80)
    print("[PTC LABEL VIS]")
    print(f"[INFO] ptc_dir: {ptc_dir}")
    print(f"[INFO] mission_dir: {mission_dir}")
    print(f"[INFO] save_dir: {save_dir}")
    print(f"[INFO] keys: {sorted(arrays)}")
    print(f"[INFO] risk_source: {attrs.get('risk_source', 'unknown')}")
    print(f"[INFO] normalization: {attrs.get('normalization', 'unknown')}")
    print(f"[INFO] valid_source: {attrs.get('valid_source', 'unknown')}")
    print(f"[INFO] orientation: {attrs.get('orientation', 'unknown')}")
    print(f"[INFO] roi: x=[{attrs.get('roi_x_min')}, {attrs.get('roi_x_max')}], "
          f"y=[{attrs.get('roi_y_min')}, {attrs.get('roi_y_max')}]")
    print(f"[INFO] ready_for_training: {summary.get('ready_for_training', 'unknown')}")
    print(f"[INFO] bev_origin: {bev_origin}")
    print(f"[INFO] flip_lr_for_display: {flip_lr_for_display}")
    print("=" * 80)

    chosen = build_sampling_indices(
        arrays=arrays,
        num_samples=num_samples,
        seed=seed,
        only_used_by_paths=only_used_by_paths,
        min_path_count=min_path_count,
        explicit_rows=parse_rows_arg(rows),
    )

    saved: list[Path] = []
    for out_i in chosen:
        out_i = int(out_i)
        image_id = int(read_scalar(arrays["image_id"], out_i))
        source_row = int(read_scalar(arrays["source_row"], out_i))
        timestamp = timestamp_to_float_seconds(read_scalar(arrays["timestamp"], out_i))
        elevation_row = (
            int(read_scalar(arrays["elevation_row"], out_i))
            if "elevation_row" in arrays
            else source_row
        )
        frame_valid = (
            int(read_scalar(arrays["frame_valid"], out_i))
            if "frame_valid" in arrays
            else 1
        )
        path_count = (
            int(read_scalar(arrays["path_sample_count"], out_i))
            if "path_sample_count" in arrays
            else -1
        )

        risk_map = read_map(arrays["risk_map"], out_i, dtype=np.float32)
        valid_mask = read_map(arrays["valid_mask"], out_i, dtype=np.uint8)
        raw_risk = (
            read_map(arrays["raw_risk"], out_i, dtype=np.float32)
            if "raw_risk" in arrays
            else blank_map_like(risk_map)
        )
        trav_cost = (
            read_map(arrays["trav_cost"], out_i, dtype=np.float32)
            if "trav_cost" in arrays
            else blank_map_like(risk_map)
        )
        finite_mask = (
            read_map(arrays["finite_mask"], out_i, dtype=np.uint8)
            if "finite_mask" in arrays
            else np.isfinite(raw_risk).astype(np.uint8)
        )

        invalid_mask = (valid_mask == 0).astype(np.uint8)
        high_risk = ((risk_map >= risk_threshold) & (valid_mask > 0)).astype(np.uint8)
        trav_cost_vis = np.log10(np.where(np.isfinite(trav_cost), trav_cost + 1.0, np.nan))

        valid_ratio = float(np.mean(valid_mask > 0))
        finite_ratio = float(np.mean(finite_mask > 0))
        risk_mean = finite_mean(risk_map[valid_mask > 0])
        raw_mean = finite_mean(raw_risk)
        trav_mean = finite_mean(trav_cost)

        left_info = get_camera_debug_info(
            mission_dir=mission_dir,
            view="hdr_left",
            image_id=image_id,
            ptc_ts=timestamp,
            use_timestamp_fallback=use_timestamp_side_view_fallback,
            max_dt=max_side_view_dt,
        )
        front_info = get_camera_debug_info(
            mission_dir=mission_dir,
            view="hdr_front",
            image_id=image_id,
            ptc_ts=timestamp,
            use_timestamp_fallback=False,
            max_dt=max_side_view_dt,
        )
        right_info = get_camera_debug_info(
            mission_dir=mission_dir,
            view="hdr_right",
            image_id=image_id,
            ptc_ts=timestamp,
            use_timestamp_fallback=use_timestamp_side_view_fallback,
            max_dt=max_side_view_dt,
        )

        fig, axes = plt.subplots(3, 3, figsize=(19, 15))
        fig.suptitle(
            "PTC Label Debug | "
            f"out_i={out_i} | source_row={source_row} | elevation_row={elevation_row} | "
            f"image_id={image_id} | timestamp={timestamp:.6f} | "
            f"frame_valid={frame_valid} | path_count={path_count}\n"
            f"valid_ratio={valid_ratio:.3f}, finite_ratio={finite_ratio:.3f}, "
            f"risk_mean={risk_mean:.3f}, raw_mean={raw_mean:.3f}, "
            f"trav_cost_mean={trav_mean:.3f}",
            fontsize=13,
        )

        show_camera_image(axes[0, 0], left_info, "Left Camera")
        show_camera_image(axes[0, 1], front_info, "Front Camera")
        show_camera_image(axes[0, 2], right_info, "Right Camera")

        im = plot_bev(
            axes[1, 0],
            risk_map,
            "PTC: risk_map\nsource: ptc_labels/risk_map",
            origin=bev_origin,
            flip_lr=flip_lr_for_display,
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        im = plot_bev(
            axes[1, 1],
            valid_mask,
            "PTC: valid_mask\nfinite raw/trav/risk + optional known mask",
            origin=bev_origin,
            flip_lr=flip_lr_for_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

        im = plot_bev(
            axes[1, 2],
            raw_risk,
            "PTC optional: raw_risk\nlarger = riskier",
            origin=bev_origin,
            flip_lr=flip_lr_for_display,
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

        im = plot_bev(
            axes[2, 0],
            trav_cost_vis,
            "PTC optional: log10(trav_cost + 1)\nlarge fatal values may appear",
            origin=bev_origin,
            flip_lr=flip_lr_for_display,
            cmap="viridis",
        )
        plt.colorbar(im, ax=axes[2, 0], fraction=0.046)

        im = plot_bev(
            axes[2, 1],
            invalid_mask,
            "Derived: invalid area\nrisk_map should be 0 here",
            origin=bev_origin,
            flip_lr=flip_lr_for_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=axes[2, 1], fraction=0.046)

        im = plot_bev(
            axes[2, 2],
            high_risk,
            f"Derived: high risk\nrisk_map >= {risk_threshold}",
            origin=bev_origin,
            flip_lr=flip_lr_for_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=axes[2, 2], fraction=0.046)

        height, width = risk_map.shape
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        for r in range(1, 3):
            for c in range(3):
                axes[r, c].plot(
                    cx,
                    cy,
                    marker="+",
                    color="red",
                    markersize=15,
                    markeredgewidth=2,
                )

        axes[0, 1].text(
            0.01,
            0.01,
            (
                f"risk_source={attrs.get('risk_source', 'unknown')}\n"
                f"normalization={attrs.get('normalization', 'unknown')}\n"
                f"orientation={attrs.get('orientation', 'unknown')}\n"
                f"valid_source={attrs.get('valid_source', 'unknown')}"
            ),
            transform=axes[0, 1].transAxes,
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
            va="bottom",
            ha="left",
        )

        fig.tight_layout(rect=[0, 0.03, 1, 0.92])
        out_filename = (
            save_dir
            / f"debug_ptc_out_{out_i:06d}"
            f"_src_{source_row:06d}"
            f"_img_{image_id:06d}.png"
        )
        fig.savefig(out_filename, dpi=150)
        plt.close(fig)
        saved.append(out_filename)
        print(f"[SAVED] {out_filename}")

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PTC-LIMO ptc_labels maps with synchronized cameras."
    )
    parser.add_argument(
        "--ptc-dir",
        type=Path,
        default=None,
        help="Path to <mission>/ptc_labels. Overrides --dataset-root/--mission.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mission", type=str, default=None)
    parser.add_argument("--group-name", type=str, default=DEFAULT_GROUP_NAME)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save visualization png files.",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help="Optional explicit ptc_labels rows, e.g. '0,10,100'. These are rows, not image_id.",
    )
    parser.add_argument(
        "--only-used-by-paths",
        action="store_true",
        help="Only visualize rows with path_sample_count >= min_path_count.",
    )
    parser.add_argument("--min-path-count", type=int, default=1)
    parser.add_argument("--risk-threshold", type=float, default=0.9)
    parser.add_argument(
        "--bev-origin",
        type=str,
        default="lower",
        choices=["lower", "upper"],
    )
    parser.add_argument(
        "--no-flip-lr",
        action="store_true",
        help="Disable left-right flip for BEV display.",
    )
    parser.add_argument(
        "--no-timestamp-side-view-fallback",
        action="store_true",
        help="Disable timestamp nearest-row fallback for side cameras.",
    )
    parser.add_argument(
        "--max-side-view-dt",
        type=float,
        default=0.08,
        help="Max timestamp difference in seconds for side-view fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ptc_dir = resolve_ptc_dir(args)
    save_dir = args.save_dir.expanduser() if args.save_dir is not None else default_save_dir(ptc_dir)
    try:
        plot_debug_samples_with_cams(
            ptc_dir=ptc_dir,
            save_dir=save_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            risk_threshold=args.risk_threshold,
            bev_origin=args.bev_origin,
            flip_lr_for_display=not args.no_flip_lr,
            use_timestamp_side_view_fallback=not args.no_timestamp_side_view_fallback,
            max_side_view_dt=args.max_side_view_dt,
            only_used_by_paths=args.only_used_by_paths,
            min_path_count=args.min_path_count,
            rows=args.rows,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
