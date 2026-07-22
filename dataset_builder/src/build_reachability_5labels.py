"""Build five-class reachability supervision for every elevation-map frame.

Input is always ``<mission>/data/elevation_map`` and output is always
``<mission>/reachability_5labels``. Frames are processed strictly in elevation
axis-0 order; no frame-selection or skip path exists in this exporter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import (
    ReachabilityState,
    build_teacher_a,
    ego_mask_from_rectangles,
)
from dataset_builder.reachability.traversability import compute_reachability_risk


OUTPUT_GROUP_NAME = "reachability_5labels"
REQUIRED_OUTPUT_ARRAYS = {
    "state_label",
    "ignore_mask",
    "risk",
    "geodesic_m",
    "image_id",
}
LABEL_DEFINITIONS = {
    str(int(state)): state.name.lower() for state in ReachabilityState
}


def _validate_source(group) -> tuple[int, int, int, bool]:
    keys = set(group.array_keys())
    missing = {"elevation", "image_id"} - keys
    if missing:
        raise KeyError(f"elevation_map is missing required arrays: {sorted(missing)}")

    elevation = group["elevation"]
    if elevation.ndim != 3:
        raise ValueError(
            f"elevation must have shape [N,H,W], got {elevation.shape}"
        )
    n, height, width = map(int, elevation.shape)
    if group["image_id"].shape != (n,):
        raise ValueError(
            "image_id must align exactly with elevation axis 0: "
            f"expected {(n,)}, got {group['image_id'].shape}"
        )

    has_timestamp = "timestamp" in keys
    if has_timestamp and group["timestamp"].shape != (n,):
        raise ValueError(
            "timestamp must align exactly with elevation axis 0: "
            f"expected {(n,)}, got {group['timestamp'].shape}"
        )
    return n, height, width, has_timestamp


def _create_output_arrays(
    output_group,
    source_group,
    *,
    n: int,
    height: int,
    width: int,
    has_timestamp: bool,
) -> dict[str, object]:
    map_chunks = (1, height, width)
    arrays = {
        "state_label": output_group.create_dataset(
            "state_label",
            shape=(n, height, width),
            chunks=map_chunks,
            dtype=np.uint8,
            fill_value=int(ReachabilityState.UNKNOWN),
        ),
        "ignore_mask": output_group.create_dataset(
            "ignore_mask",
            shape=(n, height, width),
            chunks=map_chunks,
            dtype=bool,
            fill_value=False,
        ),
        "risk": output_group.create_dataset(
            "risk",
            shape=(n, height, width),
            chunks=map_chunks,
            dtype=np.float32,
            fill_value=np.nan,
        ),
        "geodesic_m": output_group.create_dataset(
            "geodesic_m",
            shape=(n, height, width),
            chunks=map_chunks,
            dtype=np.float32,
            fill_value=np.nan,
        ),
    }

    id_source = source_group["image_id"]
    vector_chunks = (max(1, min(n, int(id_source.chunks[0]))),)
    arrays["image_id"] = output_group.create_dataset(
        "image_id",
        data=np.asarray(id_source[:]),
        chunks=vector_chunks,
        dtype=id_source.dtype,
    )
    if has_timestamp:
        timestamp_source = source_group["timestamp"]
        timestamp_chunks = (
            max(1, min(n, int(timestamp_source.chunks[0]))),
        )
        arrays["timestamp"] = output_group.create_dataset(
            "timestamp",
            data=np.asarray(timestamp_source[:]),
            chunks=timestamp_chunks,
            dtype=timestamp_source.dtype,
        )
    return arrays


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
    except Exception:
        backup.rename(output)
        raise
    if backup.is_symlink() or backup.is_file():
        backup.unlink()
    elif backup.is_dir():
        shutil.rmtree(backup)


def _validate_frame_output(
    row: int,
    state_label: np.ndarray,
    risk: np.ndarray,
    geodesic_m: np.ndarray,
) -> None:
    if state_label.dtype != np.uint8:
        raise RuntimeError(f"row {row}: state_label dtype is {state_label.dtype}")
    if risk.dtype != np.float32:
        raise RuntimeError(f"row {row}: risk dtype is {risk.dtype}")
    if geodesic_m.dtype != np.float32:
        raise RuntimeError(f"row {row}: geodesic_m dtype is {geodesic_m.dtype}")
    if np.any(state_label > int(ReachabilityState.REACHABLE)):
        values = np.unique(state_label).tolist()
        raise RuntimeError(f"row {row}: invalid state_label values {values}")

    reachable = state_label == int(ReachabilityState.REACHABLE)
    finite_geodesic = np.isfinite(geodesic_m)
    if not np.array_equal(finite_geodesic, reachable):
        raise RuntimeError(
            f"row {row}: finite geodesic_m mask does not equal REACHABLE mask"
        )


def build_mission(
    mission_dir: Path,
    *,
    radius_m: float,
    map_resolution: float,
    mppi_config: Path,
    device: str,
    overwrite: bool,
    mppi_cfg=None,
    filter_model=None,
    progress_position: int = 0,
    leave_progress: bool = True,
) -> Path:
    mission = mission_dir.expanduser().resolve()
    source_path = mission / "data" / "elevation_map"
    output_path = mission / OUTPUT_GROUP_NAME
    if not source_path.is_dir():
        raise FileNotFoundError(f"input elevation_map does not exist: {source_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    if not np.isfinite(map_resolution) or map_resolution <= 0:
        raise ValueError("map_resolution must be finite and positive")
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("radius_m must be finite and non-negative")

    source = zarr.open_group(str(source_path), mode="r")
    n, height, width, has_timestamp = _validate_source(source)
    geometry = MapGeometry(
        height=height,
        width=width,
        resolution=float(map_resolution),
        origin_xy=(
            -0.5 * height * float(map_resolution),
            -0.5 * width * float(map_resolution),
        ),
    )

    cfg = OmegaConf.load(mppi_config).mppi if mppi_cfg is None else mppi_cfg
    ego_rectangles = OmegaConf.to_container(cfg.footprint, resolve=True)
    ego_mask = ego_mask_from_rectangles(geometry, ego_rectangles)
    model = get_filter_torch(device) if filter_model is None else filter_model

    temp_output = mission / f".{OUTPUT_GROUP_NAME}.tmp-{uuid.uuid4().hex}"
    try:
        output = zarr.open_group(str(temp_output), mode="w")
        arrays = _create_output_arrays(
            output,
            source,
            n=n,
            height=height,
            width=width,
            has_timestamp=has_timestamp,
        )
        effective_radius_m = None
        root_index = list(geometry.root_index)
        for row in tqdm(
            range(n),
            desc=mission.name,
            unit="frame",
            position=progress_position,
            leave=leave_progress,
        ):
            elevation = np.asarray(source["elevation"][row], dtype=np.float32)
            risk = compute_reachability_risk(
                elevation,
                model,
                device=device,
            )
            result = build_teacher_a(
                geometry=geometry,
                known_trav=np.isfinite(elevation) & np.isfinite(risk),
                risk=risk,
                fatal_threshold=float(cfg.fatal_th),
                inflation_radius_m=float(radius_m),
                ego_mask=ego_mask,
            )

            state_label = result.semantic_label.astype(np.uint8, copy=False)
            geodesic_m = result.geodesic_m.astype(np.float32, copy=True)
            geodesic_m[state_label != int(ReachabilityState.REACHABLE)] = np.nan
            _validate_frame_output(row, state_label, risk, geodesic_m)

            arrays["state_label"][row] = state_label
            arrays["risk"][row] = risk
            arrays["geodesic_m"][row] = geodesic_m
            # ignore_mask is intentionally left at its all-False fill value.
            effective_radius_m = result.effective_radius_m

        output.attrs.update(
            {
                "schema": "reachability_5labels_v1",
                "source_group": "data/elevation_map",
                "axis_0_policy": "strict_source_order_no_skips",
                "label_definitions": LABEL_DEFINITIONS,
                "ignore_policy": "all_false",
                "risk_definition": (
                    "1 - traversability_score; unknown context nearest-filled for "
                    "CNN only; original unknown centers restored to NaN"
                ),
                "geodesic_definition": (
                    "finite metres only where state_label == 4; NaN elsewhere"
                ),
                "map_axes": {"0": "robot_x_forward", "1": "robot_y_left"},
                "map_resolution_m": float(map_resolution),
                "map_origin_xy_m": list(geometry.origin_xy),
                "root_xy_m": [0.0, 0.0],
                "root_index": root_index,
                "fatal_threshold": float(cfg.fatal_th),
                "requested_inflation_radius_m": float(radius_m),
                "effective_inflation_radius_m": effective_radius_m,
                "inflation_policy": "fatal_obstacles_only",
                "unknown_policy": "center_only_except_fixed_ego_unknown_trusted_free",
                "ego_rectangles_xy_m": ego_rectangles,
                "diagonal_corner_cutting": False,
                "contains_timestamp": bool(has_timestamp),
                "frame_count": int(n),
                "map_shape": [height, width],
            }
        )

        expected_keys = set(REQUIRED_OUTPUT_ARRAYS)
        if has_timestamp:
            expected_keys.add("timestamp")
        actual_keys = set(output.array_keys())
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"unexpected output arrays: expected {sorted(expected_keys)}, "
                f"got {sorted(actual_keys)}"
            )

        _replace_output(temp_output, output_path, overwrite)
    except BaseException:
        if temp_output.exists():
            shutil.rmtree(temp_output)
        raise

    return output_path


def main() -> None:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "build.yaml"
    parser = argparse.ArgumentParser(
        description=(
            "Build reachability_5labels for one mission or every mission in a dataset."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--mission-dir",
        type=Path,
        help="One mission directory containing data/elevation_map.",
    )
    source.add_argument(
        "--dataset-dir",
        type=Path,
        help="Dataset root; process every direct child containing data/elevation_map.",
    )
    parser.add_argument("--radius", type=float, default=0.26)
    parser.add_argument("--map-resolution", type=float, default=0.04)
    parser.add_argument("--mppi-config", type=Path, default=default_config)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing <mission>/reachability_5labels only after a complete new build.",
    )
    args = parser.parse_args()

    if args.mission_dir is not None:
        missions = [args.mission_dir.expanduser().resolve()]
    else:
        dataset_dir = args.dataset_dir.expanduser().resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
        missions = sorted(
            path
            for path in dataset_dir.iterdir()
            if path.is_dir() and (path / "data" / "elevation_map").is_dir()
        )
        if not missions:
            raise FileNotFoundError(
                f"no missions containing data/elevation_map found in {dataset_dir}"
            )

    if not args.overwrite:
        existing = [
            mission / OUTPUT_GROUP_NAME
            for mission in missions
            if (mission / OUTPUT_GROUP_NAME).exists()
        ]
        if existing:
            formatted = "\n".join(f"  - {path}" for path in existing)
            raise FileExistsError(
                "existing outputs found; pass --overwrite to replace them:\n"
                f"{formatted}"
            )

    config_path = args.mppi_config.expanduser().resolve()
    shared_cfg = OmegaConf.load(config_path).mppi
    shared_filter_model = get_filter_torch(args.device)
    outputs = []
    mission_progress = tqdm(
        missions,
        desc="missions",
        unit="mission",
        disable=len(missions) == 1,
        position=0,
    )
    for mission in mission_progress:
        if len(missions) > 1:
            mission_progress.set_postfix_str(mission.name)
        output = build_mission(
            mission,
            radius_m=args.radius,
            map_resolution=args.map_resolution,
            mppi_config=config_path,
            device=args.device,
            overwrite=args.overwrite,
            mppi_cfg=shared_cfg,
            filter_model=shared_filter_model,
            progress_position=1 if len(missions) > 1 else 0,
            leave_progress=len(missions) == 1,
        )
        outputs.append(str(output))

    print(
        json.dumps(
            {"mission_count": len(outputs), "outputs": outputs},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
