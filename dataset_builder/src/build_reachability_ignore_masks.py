"""Add static front/multiview geometric-FOV masks to reachability_5labels.

The masks are two-dimensional because camera calibration, BEV geometry, and
the reference ground plane are static across all timestamps in a mission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm
import yaml
import zarr


CAMERAS = ("hdr_front", "hdr_left", "hdr_right")
MASK_KEYS = ("front_ignore", "multiview_ignore")
REQUIRED_LABEL_ARRAYS = {
    "state_label",
    "ignore_mask",
    "risk",
    "geodesic_m",
    "image_id",
}


@dataclass(frozen=True)
class MapGeometry:
    height: int
    width: int
    resolution: float
    origin_xy: tuple[float, float]


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required calibration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _pose_to_camera_extrinsic(transform: dict) -> np.ndarray:
    quaternion = transform["rotation"]
    translation = transform["translation"]
    x, y, z, w = np.asarray(
        [
            quaternion["x"],
            quaternion["y"],
            quaternion["z"],
            quaternion["w"],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("camera transform contains an invalid quaternion")
    x, y, z, w = (np.asarray([x, y, z, w]) / norm).tolist()
    rotation = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    position = np.array(
        [translation["x"], translation["y"], translation["z"]],
        dtype=np.float64,
    )
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation.T
    extrinsic[:3, 3] = -rotation.T @ position
    return extrinsic


def _load_intrinsics(mission: Path, camera: str) -> tuple:
    path = mission / "metadata" / f"{camera}_caminfo.yaml"
    camera_info = _load_yaml(path)["camera_info"]
    if camera_info.get("distortion_model") != "equidistant":
        raise ValueError(
            f"{camera}: expected equidistant distortion, got "
            f"{camera_info.get('distortion_model')}"
        )
    K = np.asarray(camera_info["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(camera_info["D"], dtype=np.float64).reshape(-1)
    if len(D) < 4:
        raise ValueError(f"{camera}: expected at least 4 distortion coefficients")
    return K, D[:4], int(camera_info["height"]), int(camera_info["width"]), path


def _load_calibrations(mission: Path, front_extrinsics_config: Path) -> dict:
    calibrations = {}
    front_K, front_D, front_H, front_W, front_intrinsics_path = _load_intrinsics(
        mission, "hdr_front"
    )
    front_config = _load_yaml(front_extrinsics_config)
    front_E = np.asarray(front_config["E"], dtype=np.float64).reshape(4, 4)
    calibrations["hdr_front"] = {
        "K": front_K,
        "D": front_D,
        "E": front_E,
        "height": front_H,
        "width": front_W,
        "intrinsics_path": front_intrinsics_path,
        "extrinsics_path": front_extrinsics_config,
    }

    expected_directions = {
        "hdr_left": np.array([0.0, 5.0, -0.57, 1.0]),
        "hdr_right": np.array([0.0, -5.0, -0.57, 1.0]),
    }
    for camera in ("hdr_left", "hdr_right"):
        K, D, height, width, intrinsics_path = _load_intrinsics(mission, camera)
        extrinsics_path = mission / "metadata" / f"{camera}.yaml"
        transform = _load_yaml(extrinsics_path)["transform"]
        E = _pose_to_camera_extrinsic(transform)

        # Some exported side-camera transforms use the opposite optical-axis
        # sign. This is the same correction already used by the dataset-builder
        # camera overlay, made explicit against canonical y-left directions.
        if float((E @ expected_directions[camera])[2]) <= 0.1:
            E = np.diag([-1.0, -1.0, -1.0, 1.0]) @ E
        calibrations[camera] = {
            "K": K,
            "D": D,
            "E": E,
            "height": height,
            "width": width,
            "intrinsics_path": intrinsics_path,
            "extrinsics_path": extrinsics_path,
        }

    direction_tests = {
        "hdr_front": np.array([5.0, 0.0, -0.57, 1.0]),
        "hdr_left": expected_directions["hdr_left"],
        "hdr_right": expected_directions["hdr_right"],
    }
    for camera, point in direction_tests.items():
        if float((calibrations[camera]["E"] @ point)[2]) <= 0.1:
            raise ValueError(
                f"{camera} extrinsic is incompatible with canonical x-forward/y-left axes"
            )
    return calibrations


def _project_visible(points: np.ndarray, calibration: dict) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    camera_points = (calibration["E"] @ homogeneous.T).T[:, :3]
    x_camera, y_camera, z_camera = camera_points.T
    radius = np.hypot(x_camera, y_camera)
    theta = np.arctan2(radius, z_camera)
    theta2 = theta * theta
    D = calibration["D"]
    distorted_radius = theta * (
        1.0
        + D[0] * theta2
        + D[1] * theta2**2
        + D[2] * theta2**3
        + D[3] * theta2**4
    )
    safe_radius = np.where(radius > 1e-12, radius, 1.0)
    K = calibration["K"]
    u = K[0, 0] * distorted_radius * x_camera / safe_radius + K[0, 2]
    v = K[1, 1] * distorted_radius * y_camera / safe_radius + K[1, 2]
    return (
        (z_camera > 0.1)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < calibration["width"])
        & (v >= 0.0)
        & (v < calibration["height"])
    )


def _geometry_from_group(group) -> MapGeometry:
    if "state_label" not in group:
        raise KeyError("reachability_5labels is missing state_label")
    state_shape = group["state_label"].shape
    if len(state_shape) != 3:
        raise ValueError(f"state_label must have shape [N,H,W], got {state_shape}")
    _, height, width = map(int, state_shape)
    resolution = float(group.attrs.get("map_resolution_m", 0.04))
    default_origin = (-0.5 * height * resolution, -0.5 * width * resolution)
    origin = group.attrs.get("map_origin_xy_m", default_origin)
    return MapGeometry(
        height=height,
        width=width,
        resolution=resolution,
        origin_xy=(float(origin[0]), float(origin[1])),
    )


def _validate_mission_contract(mission: Path, labels) -> MapGeometry:
    label_keys = set(labels.array_keys())
    missing = REQUIRED_LABEL_ARRAYS - label_keys
    if missing:
        raise KeyError(
            f"{mission.name}/reachability_5labels is missing arrays: "
            f"{sorted(missing)}"
        )

    state = labels["state_label"]
    if state.ndim != 3:
        raise ValueError(f"state_label must have shape [N,H,W], got {state.shape}")
    n, height, width = map(int, state.shape)
    expected_map_shape = (n, height, width)
    for key in ("ignore_mask", "risk", "geodesic_m"):
        if labels[key].shape != expected_map_shape:
            raise ValueError(
                f"{key} shape {labels[key].shape} != {expected_map_shape}"
            )
    if labels["image_id"].shape != (n,):
        raise ValueError(
            f"image_id shape {labels['image_id'].shape} != expected {(n,)}"
        )
    expected_dtypes = {
        "state_label": np.dtype(np.uint8),
        "ignore_mask": np.dtype(bool),
        "risk": np.dtype(np.float32),
        "geodesic_m": np.dtype(np.float32),
    }
    for key, expected_dtype in expected_dtypes.items():
        if labels[key].dtype != expected_dtype:
            raise TypeError(
                f"{key} dtype {labels[key].dtype} != expected {expected_dtype}"
            )

    elevation_path = mission / "data" / "elevation_map"
    if not elevation_path.is_dir():
        raise FileNotFoundError(f"elevation_map not found: {elevation_path}")
    elevation = zarr.open_group(str(elevation_path), mode="r")
    elevation_keys = set(elevation.array_keys())
    missing_elevation = {"elevation", "image_id"} - elevation_keys
    if missing_elevation:
        raise KeyError(
            f"{mission.name}/data/elevation_map is missing arrays: "
            f"{sorted(missing_elevation)}"
        )
    if elevation["elevation"].shape != expected_map_shape:
        raise ValueError(
            f"elevation shape {elevation['elevation'].shape} != "
            f"state_label shape {expected_map_shape}"
        )
    if not np.array_equal(
        np.asarray(labels["image_id"][:]),
        np.asarray(elevation["image_id"][:]),
    ):
        raise ValueError(
            "reachability_5labels/image_id is not exactly aligned with "
            "data/elevation_map/image_id"
        )

    label_has_timestamp = "timestamp" in label_keys
    elevation_has_timestamp = "timestamp" in elevation_keys
    if label_has_timestamp != elevation_has_timestamp:
        raise ValueError(
            "timestamp presence differs between reachability_5labels and "
            "data/elevation_map"
        )
    if label_has_timestamp:
        if labels["timestamp"].shape != (n,):
            raise ValueError(
                f"timestamp shape {labels['timestamp'].shape} != expected {(n,)}"
            )
        if not np.array_equal(
            np.asarray(labels["timestamp"][:]),
            np.asarray(elevation["timestamp"][:]),
            equal_nan=True,
        ):
            raise ValueError(
                "reachability_5labels/timestamp is not exactly aligned with "
                "data/elevation_map/timestamp"
            )
    return _geometry_from_group(labels)


def _cache_key(
    geometry: MapGeometry, calibrations: dict, ground_z: float
) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.asarray(
            [
                geometry.height,
                geometry.width,
                geometry.resolution,
                *geometry.origin_xy,
                ground_z,
            ],
            dtype=np.float64,
        ).tobytes()
    )
    for camera in CAMERAS:
        calibration = calibrations[camera]
        for key in ("K", "D", "E"):
            digest.update(np.asarray(calibration[key], dtype=np.float64).tobytes())
        digest.update(
            np.asarray(
                [calibration["height"], calibration["width"]], dtype=np.int64
            ).tobytes()
        )
    return digest.hexdigest()


def _compute_masks(
    geometry: MapGeometry,
    calibrations: dict,
    ground_z: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = geometry.origin_xy[0] + np.arange(geometry.height) * geometry.resolution
    y = geometry.origin_xy[1] + np.arange(geometry.width) * geometry.resolution
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.stack(
        [xx, yy, np.full_like(xx, float(ground_z))], axis=-1
    ).reshape(-1, 3)

    visibility = {
        camera: _project_visible(points, calibrations[camera]).reshape(
            geometry.height, geometry.width
        )
        for camera in CAMERAS
    }
    front_ignore = ~visibility["hdr_front"]
    multiview_visible = np.logical_or.reduce(
        [visibility[camera] for camera in CAMERAS]
    )
    multiview_ignore = ~multiview_visible
    counts = {
        f"{camera}_visible": int(visibility[camera].sum()) for camera in CAMERAS
    }
    counts["front_ignore"] = int(front_ignore.sum())
    counts["multiview_visible"] = int(multiview_visible.sum())
    counts["multiview_ignore"] = int(multiview_ignore.sum())
    return front_ignore, multiview_ignore, counts


def _write_masks(
    group,
    front_ignore: np.ndarray,
    multiview_ignore: np.ndarray,
    *,
    overwrite: bool,
) -> None:
    existing = [key for key in MASK_KEYS if key in group]
    if existing and not overwrite:
        raise FileExistsError(
            f"ignore mask arrays already exist: {existing}; pass --overwrite"
        )

    temporary = {
        key: f".{key}.tmp-{uuid.uuid4().hex}"
        for key in MASK_KEYS
    }
    try:
        group.create_dataset(
            temporary["front_ignore"],
            data=np.asarray(front_ignore, dtype=bool),
            chunks=front_ignore.shape,
            dtype=bool,
        )
        group.create_dataset(
            temporary["multiview_ignore"],
            data=np.asarray(multiview_ignore, dtype=bool),
            chunks=multiview_ignore.shape,
            dtype=bool,
        )
        for key in existing:
            del group[key]
        for key in MASK_KEYS:
            group.move(temporary[key], key)
    except BaseException:
        for temporary_key in temporary.values():
            if temporary_key in group:
                del group[temporary_key]
        raise


def _preflight(missions: list[Path], overwrite: bool) -> None:
    problems = []
    for mission in missions:
        group_path = mission / "reachability_5labels"
        if not group_path.is_dir():
            problems.append(f"missing reachability_5labels: {group_path}")
            continue
        if not overwrite:
            group = zarr.open_group(str(group_path), mode="r")
            existing = [key for key in MASK_KEYS if key in group]
            if existing:
                problems.append(f"{mission.name}: existing arrays {existing}")
    if problems:
        raise RuntimeError("preflight failed:\n  - " + "\n  - ".join(problems))


def build_masks_for_missions(
    missions: list[Path],
    *,
    ground_z: float,
    front_extrinsics_config: Path,
    overwrite: bool,
) -> list[dict]:
    _preflight(missions, overwrite)
    cache: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    records = []
    for mission in tqdm(missions, desc="missions", unit="mission"):
        group_path = mission / "reachability_5labels"
        group = zarr.open_group(str(group_path), mode="a")
        geometry = _validate_mission_contract(mission, group)
        calibrations = _load_calibrations(mission, front_extrinsics_config)
        key = _cache_key(geometry, calibrations, ground_z)
        cache_hit = key in cache
        if not cache_hit:
            cache[key] = _compute_masks(geometry, calibrations, ground_z)
        front_ignore, multiview_ignore, counts = cache[key]
        _write_masks(
            group,
            front_ignore,
            multiview_ignore,
            overwrite=overwrite,
        )
        group.attrs.update(
            {
                "camera_ignore_schema": "static_geometric_fov_v1",
                "camera_ignore_static_across_frames": True,
                "camera_ignore_shape": list(front_ignore.shape),
                "camera_ignore_ground_z_m": float(ground_z),
                "camera_ignore_projection": "equidistant_raw_image_center_point",
                "front_ignore_definition": (
                    "True where the reference-ground BEV cell center is outside "
                    "the hdr_front geometric image FOV"
                ),
                "multiview_ignore_definition": (
                    "True where the reference-ground BEV cell center is outside "
                    "the union of hdr_front/hdr_left/hdr_right geometric image FOVs"
                ),
                "camera_ignore_occlusion_modelled": False,
                "camera_ignore_cameras": list(CAMERAS),
                "front_extrinsics_source": str(front_extrinsics_config),
                "side_extrinsics_source": "<mission>/metadata/<camera>.yaml",
                "camera_intrinsics_source": (
                    "<mission>/metadata/<camera>_caminfo.yaml"
                ),
                "camera_ignore_calibration_hash": key,
            }
        )
        records.append(
            {
                "mission": mission.name,
                "group": str(group_path),
                "shape": list(front_ignore.shape),
                "cache_hit": cache_hit,
                **counts,
            }
        )
    return records


def main() -> None:
    default_front_extrinsics = (
        Path(__file__).resolve().parents[2]
        / "limo"
        / "configs"
        / "model"
        / "camera_info.yaml"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Add static front_ignore and multiview_ignore [H,W] arrays to "
            "reachability_5labels."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mission-dir", type=Path)
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--ground-z", type=float, default=-0.57)
    parser.add_argument(
        "--front-extrinsics-config",
        type=Path,
        default=default_front_extrinsics,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.ground_z):
        raise ValueError("ground_z must be finite")
    if args.mission_dir is not None:
        missions = [args.mission_dir.expanduser().resolve()]
    else:
        dataset = args.dataset_dir.expanduser().resolve()
        if not dataset.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {dataset}")
        missions = sorted(
            mission
            for mission in dataset.iterdir()
            if mission.is_dir() and (mission / "reachability_5labels").is_dir()
        )
        if not missions:
            raise FileNotFoundError(
                f"no missions containing reachability_5labels found in {dataset}"
            )

    records = build_masks_for_missions(
        missions,
        ground_z=float(args.ground_z),
        front_extrinsics_config=args.front_extrinsics_config.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
