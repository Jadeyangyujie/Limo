"""Small-sample expert-path consistency calibration for Teacher-A."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/limo_teacher_a_matplotlib")

import numpy as np
import zarr
from omegaconf import OmegaConf

from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.path_calibration import (
    alignment_checks,
    build_clearance_maps,
    densify_path,
    evaluate_circular_teacher,
    evaluate_yaw_aware_rectangle,
    evaluation_metrics,
    first_violation_record,
    rectangle_footprint_offsets,
    weighted_ratio,
)
from dataset_builder.reachability.path_calibration_visualization import (
    save_path_calibration_figure,
)
from dataset_builder.reachability.teacher_a import build_teacher_a
from dataset_builder.reachability.traversability import compute_limo_traversability
from dataset_builder.src.mission_data_source import GrandTourZarrSource

log = logging.getLogger(__name__)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_dense_diagnostics(
    path: Path,
    dense,
    evaluations: dict[str, object],
) -> None:
    """Persist dense states and independent reason masks outside formal Zarr."""
    arrays = {
        "dense_path_xytheta": dense.xytheta,
        "arc_length_m": dense.arc_length_m,
        "arc_weights_m": dense.arc_weights_m,
    }
    for model, evaluation in evaluations.items():
        for name in (
            "display_state",
            "reachable",
            "configuration_free",
            "locally_blocked",
            "clearance_blocked",
            "unknown_center",
            "unknown_footprint",
            "outside_domain",
            "disconnected",
            "invalid_coordinate",
            "overlaps_blocked",
            "overlaps_unknown",
            "crosses_domain",
            "forbidden",
        ):
            arrays[f"{model}_{name}"] = getattr(evaluation, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _evenly_select(rows: list[int], count: int) -> list[int]:
    if len(rows) <= count:
        return rows
    positions = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(position)] for position in positions]


def _select_path_rows(groups: dict[str, zarr.Group], count: int, paired_count: int):
    ids = {source: np.asarray(group["image_id"], dtype=np.int64) for source, group in groups.items()}
    by_image = {
        source: defaultdict(list) for source in groups
    }
    for source in groups:
        for row, image_id in enumerate(ids[source]):
            by_image[source][int(image_id)].append(row)
    common = sorted(set(by_image["geometric"]) & set(by_image["teleop"]))
    paired_ids = _evenly_select(common, min(paired_count, len(common)))
    selected = {source: [] for source in groups}
    for image_id in paired_ids:
        for source in groups:
            selected[source].append(by_image[source][image_id][0])
    for source in groups:
        candidates = list(range(len(ids[source])))
        for row in _evenly_select(candidates, count * 3):
            if len(selected[source]) >= count:
                break
            if row not in selected[source]:
                selected[source].append(row)
        selected[source] = sorted(selected[source][:count])
    return selected, paired_ids


def _elevation_lookup(elevation_group: zarr.Group):
    ids = np.asarray(elevation_group["image_id"], dtype=np.int64)
    lookup: dict[int, int] = {}
    duplicates = set()
    for row, image_id in enumerate(ids):
        key = int(image_id)
        if key in lookup:
            duplicates.add(key)
        else:
            lookup[key] = row
    for key in duplicates:
        lookup.pop(key, None)
    return lookup, sorted(duplicates)


def _first_reason(metrics: dict, prefix: str) -> str:
    ratios = {
        "unknown": metrics[f"{prefix}_unknown_ratio"],
        "locally_blocked": metrics[f"{prefix}_local_blocked_ratio"],
        "clearance_blocked": metrics[f"{prefix}_clearance_blocked_ratio"],
        "outside_domain": metrics[f"{prefix}_outside_ratio"],
        "disconnected": metrics[f"{prefix}_disconnected_ratio"],
        "invalid_coordinate": metrics[f"{prefix}_invalid_coordinate_ratio"],
    }
    active = [key for key, value in ratios.items() if value > 1e-12]
    if not active:
        return "none"
    if len(active) == 1:
        return active[0]
    return "mixed:" + "+".join(sorted(active, key=lambda key: ratios[key], reverse=True))


def _aggregate(path_rows: list[dict], model_prefixes: dict[str, str]) -> list[dict]:
    output = []
    groups = defaultdict(list)
    for row in path_rows:
        groups[(row["mission"], row["path_source"])].append(row)
    for (mission, source), rows in groups.items():
        for model, prefix in model_prefixes.items():
            forbidden = np.array([float(row[f"{prefix}_forbidden_ratio"]) for row in rows])
            first = np.array(
                [
                    float(row[f"{prefix}_first_violation_arc_length_m"])
                    for row in rows
                    if row[f"{prefix}_first_violation_arc_length_m"] not in (None, "")
                ]
            )
            reasons = [row[f"{prefix}_failure_reason"] for row in rows]
            output.append(
                {
                    "mission": mission,
                    "path_source": source,
                    "footprint_model": model,
                    "num_paths": len(rows),
                    "num_strict_rejected": int(
                        np.count_nonzero([not bool(row[f"{prefix}_strict_pass"]) for row in rows])
                    ),
                    "num_tolerance_1_rejected": int(
                        np.count_nonzero([not bool(row[f"{prefix}_tolerance_1_pass"]) for row in rows])
                    ),
                    "num_tolerance_5_rejected": int(
                        np.count_nonzero([not bool(row[f"{prefix}_tolerance_5_pass"]) for row in rows])
                    ),
                    "root_valid_rate": float(np.mean([bool(row[f"{prefix}_root_valid"]) for row in rows])),
                    "strict_pass_rate": float(np.mean([bool(row[f"{prefix}_strict_pass"]) for row in rows])),
                    "tolerance_1_pass_rate": float(
                        np.mean([bool(row[f"{prefix}_tolerance_1_pass"]) for row in rows])
                    ),
                    "tolerance_5_pass_rate": float(
                        np.mean([bool(row[f"{prefix}_tolerance_5_pass"]) for row in rows])
                    ),
                    "continuous_failure_rate": float(
                        np.mean([bool(row[f"{prefix}_continuous_failure"]) for row in rows])
                    ),
                    "forbidden_ratio_median": float(np.median(forbidden)),
                    "forbidden_ratio_p90": float(np.quantile(forbidden, 0.9)),
                    "first_violation_m_p10": float(np.quantile(first, 0.1)) if len(first) else None,
                    "first_violation_m_p50": float(np.quantile(first, 0.5)) if len(first) else None,
                    "first_violation_m_p90": float(np.quantile(first, 0.9)) if len(first) else None,
                    "unknown_only_failure_rate": reasons.count("unknown") / len(rows),
                    "locally_blocked_only_failure_rate": reasons.count("locally_blocked") / len(rows),
                    "clearance_only_failure_rate": reasons.count("clearance_blocked") / len(rows),
                    "outside_only_failure_rate": reasons.count("outside_domain") / len(rows),
                    "unknown_any_failure_rate": float(
                        np.mean([float(row[f"{prefix}_unknown_ratio"]) > 1e-12 for row in rows])
                    ),
                    "locally_blocked_any_failure_rate": float(
                        np.mean([float(row[f"{prefix}_local_blocked_ratio"]) > 1e-12 for row in rows])
                    ),
                    "clearance_any_failure_rate": float(
                        np.mean([float(row[f"{prefix}_clearance_blocked_ratio"]) > 1e-12 for row in rows])
                    ),
                    "outside_any_failure_rate": float(
                        np.mean([float(row[f"{prefix}_outside_ratio"]) > 1e-12 for row in rows])
                    ),
                }
            )
    return output


def _aggregate_by_root_and_reason(
    path_rows: list[dict], model_prefixes: dict[str, str]
) -> list[dict]:
    output = []
    groups = defaultdict(list)
    for row in path_rows:
        for model, prefix in model_prefixes.items():
            key = (
                row["mission"],
                row["path_source"],
                model,
                bool(row[f"{prefix}_root_valid"]),
                row[f"{prefix}_failure_reason"],
            )
            groups[key].append((row, prefix))
    for (mission, source, model, root_valid, reason), items in groups.items():
        forbidden = np.asarray(
            [float(row[f"{prefix}_forbidden_ratio"]) for row, prefix in items]
        )
        output.append(
            {
                "mission": mission,
                "path_source": source,
                "footprint_model": model,
                "root_valid": root_valid,
                "failure_reason": reason,
                "num_paths": len(items),
                "strict_pass_rate": float(
                    np.mean([bool(row[f"{prefix}_strict_pass"]) for row, prefix in items])
                ),
                "tolerance_1_pass_rate": float(
                    np.mean([bool(row[f"{prefix}_tolerance_1_pass"]) for row, prefix in items])
                ),
                "tolerance_5_pass_rate": float(
                    np.mean([bool(row[f"{prefix}_tolerance_5_pass"]) for row, prefix in items])
                ),
                "continuous_failure_rate": float(
                    np.mean([bool(row[f"{prefix}_continuous_failure"]) for row, prefix in items])
                ),
                "forbidden_ratio_median": float(np.median(forbidden)),
                "forbidden_ratio_p90": float(np.quantile(forbidden, 0.9)),
            }
        )
    return output


def _aggregate_radius_scan(radius_rows: list[dict]) -> list[dict]:
    output = []
    groups = defaultdict(list)
    for row in radius_rows:
        groups[
            (
                row["mission"],
                row["path_source"],
                row["requested_radius_m"],
                row["effective_radius_m"],
            )
        ].append(row)
    for (mission, source, requested, effective), rows in groups.items():
        forbidden = np.asarray([float(row["forbidden_arc_ratio"]) for row in rows])
        output.append(
            {
                "mission": mission,
                "path_source": source,
                "requested_radius_m": requested,
                "effective_radius_m": effective,
                "num_paths": len(rows),
                "strict_pass_rate": float(np.mean([bool(row["expert_strict_pass"]) for row in rows])),
                "tolerance_1_pass_rate": float(
                    np.mean([bool(row["expert_tolerance_1_pass"]) for row in rows])
                ),
                "tolerance_5_pass_rate": float(
                    np.mean([bool(row["expert_tolerance_5_pass"]) for row in rows])
                ),
                "forbidden_arc_ratio_median": float(np.median(forbidden)),
                "forbidden_arc_ratio_p90": float(np.quantile(forbidden, 0.9)),
                "rectangle_agreement_ratio_mean": float(
                    np.mean([float(row["rectangle_agreement_ratio"]) for row in rows])
                ),
                "rectangle_false_reject_ratio_mean": float(
                    np.mean([float(row["rectangle_false_reject_ratio"]) for row in rows])
                ),
                "rectangle_false_accept_ratio_mean": float(
                    np.mean([float(row["rectangle_false_accept_ratio"]) for row in rows])
                ),
                "root_valid_rate": float(np.mean([bool(row["root_valid"]) for row in rows])),
                "reachable_area_ratio_mean": float(
                    np.mean([float(row["reachable_area_ratio"]) for row in rows])
                ),
            }
        )
    return output


def _representatives(rows: list[dict]) -> dict:
    manifest = {}

    def choose(label, predicate):
        candidates = [row for row in rows if predicate(row)]
        manifest[label] = (
            {
                "mission": candidates[0]["mission"],
                "path_source": candidates[0]["path_source"],
                "path_row": candidates[0]["path_row"],
                "image_id": candidates[0]["image_id"],
                "figure": candidates[0]["figure"],
            }
            if candidates
            else None
        )

    choose(
        "all_models_strict_pass",
        lambda r: r["center_strict_pass"] and r["circle_0p26_strict_pass"]
        and r["circle_0p61_strict_pass"] and r["rectangle_strict_pass"],
    )
    choose(
        "circle_0p26_pass_circle_0p61_fail",
        lambda r: r["circle_0p26_tolerance_1_pass"] and not r["circle_0p61_tolerance_5_pass"],
    )
    choose(
        "circle_0p61_fail_rectangle_pass",
        lambda r: not r["circle_0p61_tolerance_5_pass"] and r["rectangle_tolerance_1_pass"],
    )
    choose("circle_0p26_long_failure", lambda r: r["circle_0p26_continuous_failure"])
    choose(
        "unknown_dominant",
        lambda r: r["rectangle_unknown_ratio"] > 0
        and r["rectangle_unknown_ratio"] >= r["rectangle_blocked_ratio"]
        and r["rectangle_unknown_ratio"] >= r["rectangle_outside_ratio"],
    )
    choose(
        "locally_blocked_dominant",
        lambda r: r["center_local_blocked_ratio"] > 0
        and r["center_local_blocked_ratio"] >= r["center_unknown_ratio"]
        and r["center_local_blocked_ratio"] >= r["center_outside_ratio"],
    )
    choose(
        "first_waypoint_failure",
        lambda r: r["rectangle_first_violation_arc_length_m"] is not None
        and r["rectangle_first_violation_arc_length_m"] <= 0.02,
    )
    choose(
        "endpoint_only_failure",
        lambda r: r["rectangle_first_violation_arc_length_m"] is not None
        and r["rectangle_first_violation_arc_length_m"] >= 0.8 * r["path_length_m"],
    )

    paired = defaultdict(dict)
    for row in rows:
        paired[(row["mission"], row["image_id"])][row["path_source"]] = row
    geo_pass_tele_fail = []
    tele_pass_geo_fail = []
    for pair in paired.values():
        if "geometric" not in pair or "teleop" not in pair:
            continue
        geo, tele = pair["geometric"], pair["teleop"]
        if geo["rectangle_tolerance_5_pass"] and not tele["rectangle_tolerance_5_pass"]:
            geo_pass_tele_fail.extend([geo, tele])
        if tele["rectangle_tolerance_5_pass"] and not geo["rectangle_tolerance_5_pass"]:
            tele_pass_geo_fail.extend([geo, tele])
    manifest["paired_geometric_pass_teleop_fail"] = [r["figure"] for r in geo_pass_tele_fail] or None
    manifest["paired_teleop_pass_geometric_fail"] = [r["figure"] for r in tele_pass_geo_fail] or None
    return manifest


def run(cfg) -> Path:
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    mppi_cfg = OmegaConf.load(Path(cfg.mppi_config)).mppi
    geometry = MapGeometry(
        int(round(2 * cfg.map_size / cfg.map_resolution)),
        int(round(2 * cfg.map_size / cfg.map_resolution)),
        float(cfg.map_resolution),
        (-float(cfg.map_size), -float(cfg.map_size)),
    )
    filter_model = get_filter_torch(str(cfg.device))
    rectangles = OmegaConf.to_container(mppi_cfg.footprint, resolve=True)
    offsets = rectangle_footprint_offsets(rectangles, geometry.resolution, str(cfg.device))
    rect = np.asarray(rectangles[0], dtype=float)
    rect_length = float(rect[1, 0] - rect[0, 0])
    rect_width = float(rect[1, 1] - rect[0, 1])
    primary = [float(value) for value in cfg.primary_radii_m]
    scan = sorted(set(primary + [float(value) for value in cfg.radius_scan_m]))
    path_rows: list[dict] = []
    radius_rows: list[dict] = []
    alignment_rows: list[dict] = []
    selection = {}

    for mission_value in cfg.missions:
        mission = str(mission_value)
        mission_dir = Path(cfg.dataset_root) / mission
        source_reader = GrandTourZarrSource(mission_dir, cfg.map_size, cfg.map_resolution)
        elevation_group = zarr.open_group(str(mission_dir / "data/elevation_map"), mode="r")
        elevation_lookup, duplicate_ids = _elevation_lookup(elevation_group)
        groups = {
            "geometric": zarr.open_group(str(mission_dir / "data/geometric_paths"), mode="r"),
            "teleop": zarr.open_group(str(mission_dir / "data/teleop_paths"), mode="r"),
        }
        selected, paired_ids = _select_path_rows(
            groups, int(cfg.paths_per_source_per_mission), int(cfg.paired_image_ids_per_mission)
        )
        selection[mission] = {"rows": selected, "paired_image_ids": paired_ids, "duplicates": duplicate_ids}
        cache = {}

        for path_source, selected_rows in selected.items():
            group = groups[path_source]
            frame_id = str(group.attrs.get("frame_id", ""))
            for path_row in selected_rows:
                image_id = int(group["image_id"][path_row])
                elevation_row = elevation_lookup.get(image_id)
                path = np.asarray(group["path"][path_row], dtype=np.float32)
                goal = np.asarray(group["goal"][path_row], dtype=np.float32)
                checks = alignment_checks(
                    path, frame_id=frame_id, elevation_row=elevation_row
                )
                timestamp = (
                    float(elevation_group["timestamp"][elevation_row])
                    if elevation_row is not None and "timestamp" in elevation_group
                    else None
                )
                image_timestamp = source_reader.get_timestamp(image_id)
                timestamp_delta_s = (
                    abs(timestamp - image_timestamp) if timestamp is not None else None
                )
                alignment_row = {
                    "mission": mission,
                    "path_source": path_source,
                    "path_row": path_row,
                    "image_id": image_id,
                    "elevation_row": elevation_row,
                    "timestamp": timestamp,
                    "image_timestamp": image_timestamp,
                    "timestamp_delta_s": timestamp_delta_s,
                    "timestamp_alignment_valid": timestamp_delta_s is not None
                    and timestamp_delta_s <= float(cfg.timestamp_tolerance_s),
                    "path_coordinate_frame": frame_id,
                    "theta_zero_axis": "+x_forward",
                    "theta_positive_direction": "CCW_toward_+y_left",
                    **checks,
                }
                alignment_row["path_alignment_valid"] = bool(
                    checks["alignment_valid"]
                    and alignment_row["timestamp_alignment_valid"]
                )
                alignment_rows.append(alignment_row)
                if elevation_row is None or not checks["path_shape_valid"] or not checks["path_finite"]:
                    continue

                if image_id not in cache:
                    elevation = np.asarray(elevation_group["elevation"][elevation_row], dtype=np.float32)
                    trav = compute_limo_traversability(
                        elevation, filter_model, mppi_cfg, str(cfg.device)
                    )
                    results = {
                        radius: build_teacher_a(
                            geometry=geometry,
                            known_trav=trav.known_trav,
                            risk=trav.risk,
                            fatal_threshold=float(mppi_cfg.fatal_th),
                            inflation_radius_m=radius,
                        )
                        for radius in scan
                    }
                    clearances = build_clearance_maps(
                        trav.known_trav, trav.risk, float(mppi_cfg.fatal_th), geometry.resolution
                    )
                    root_dense = densify_path(
                        np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]], dtype=np.float32),
                        float(cfg.dense_sampling_interval_m),
                    )
                    root_rectangle = evaluate_yaw_aware_rectangle(
                        root_dense,
                        geometry=geometry,
                        footprint_offsets_xy=offsets,
                        known_trav=trav.known_trav,
                        risk=trav.risk,
                        fatal_threshold=float(mppi_cfg.fatal_th),
                        domain_support=np.ones((geometry.height, geometry.width), bool),
                        obstacle_clearance=clearances[0],
                        unknown_clearance=clearances[1],
                    )
                    cache[image_id] = (elevation, trav, results, clearances, root_rectangle)
                elevation, trav, results, (obstacle_clearance, unknown_clearance), root_rectangle = cache[image_id]
                dense = densify_path(path, float(cfg.dense_sampling_interval_m))
                evaluations = {
                    radius: evaluate_circular_teacher(
                        dense, results[radius], obstacle_clearance, unknown_clearance
                    )
                    for radius in scan
                }
                rectangle_eval = evaluate_yaw_aware_rectangle(
                    dense,
                    geometry=geometry,
                    footprint_offsets_xy=offsets,
                    known_trav=trav.known_trav,
                    risk=trav.risk,
                    fatal_threshold=float(mppi_cfg.fatal_th),
                    domain_support=np.ones((geometry.height, geometry.width), bool),
                    obstacle_clearance=obstacle_clearance,
                    unknown_clearance=unknown_clearance,
                )
                eval_named = {
                    "center": evaluations[0.0],
                    "circle_0p26": evaluations[0.26],
                    "circle_0p61": evaluations[0.61],
                    "rectangle": rectangle_eval,
                }
                row = {
                    **alignment_row,
                    "path_length_m": dense.total_length_m,
                    "endpoint_distance_m": float(np.linalg.norm(path[-1, :2])),
                    "endpoint_goal_error_m": float(np.linalg.norm(path[-1, :2] - goal[:2])),
                    "num_dense_samples": len(dense.xytheta),
                    "sampling_interval_max_m": float(np.diff(dense.arc_length_m).max())
                    if len(dense.arc_length_m) > 1
                    else 0.0,
                    "root_valid": results[0.0].root.configuration_valid,
                    "min_center_clearance_m": evaluations[0.0].min_obstacle_clearance_m,
                    "min_rectangle_clearance_m": rectangle_eval.min_obstacle_clearance_m,
                    "rectangle_overlaps_blocked_ratio": weighted_ratio(
                        rectangle_eval.overlaps_blocked, dense
                    ),
                    "rectangle_overlaps_unknown_ratio": weighted_ratio(
                        rectangle_eval.overlaps_unknown, dense
                    ),
                    "rectangle_crosses_domain_ratio": weighted_ratio(
                        rectangle_eval.crosses_domain, dense
                    ),
                    "unknown_center_ratio": weighted_ratio(
                        rectangle_eval.unknown_center, dense
                    ),
                    "unknown_footprint_ratio": weighted_ratio(
                        rectangle_eval.unknown_footprint, dense
                    ),
                }
                for name, evaluation in eval_named.items():
                    row.update(evaluation_metrics(evaluation, dense, name))
                    row[f"{name}_root_valid"] = (
                        bool(root_rectangle.configuration_free[0])
                        if name == "rectangle"
                        else results[0.0].root.configuration_valid
                        if name == "center"
                        else results[0.26 if name == "circle_0p26" else 0.61].root.configuration_valid
                    )
                    row[f"{name}_failure_reason"] = _first_reason(row, name)
                first = first_violation_record(rectangle_eval, dense)
                row.update(
                    {
                        "first_violation_type": first["type"],
                        "first_violation_path_index": first["index"],
                        "first_violation_arc_length_m": first["arc_length_m"],
                        "first_violation_x": first["x"],
                        "first_violation_y": first["y"],
                        "first_violation_theta": first["theta"],
                        "max_consecutive_violation_length_m": row[
                            "rectangle_max_consecutive_violation_length_m"
                        ],
                        "endpoint_state": row["rectangle_endpoint_state"],
                        "whole_path_valid": row["rectangle_strict_pass"],
                        "whole_path_valid_tolerance_1": row["rectangle_tolerance_1_pass"],
                        "whole_path_valid_tolerance_5": row["rectangle_tolerance_5_pass"],
                    }
                )

                figure = (
                    output
                    / mission
                    / path_source
                    / f"path_{path_row:06d}_image_{image_id:06d}.png"
                )
                save_path_calibration_figure(
                    rgb=source_reader.get_image(image_id),
                    elevation=elevation,
                    trav=trav,
                    results={
                        "center": results[0.0],
                        "circle_0p26": results[0.26],
                        "circle_0p61": results[0.61],
                    },
                    dense=dense,
                    evaluations=eval_named,
                    rectangle_length_m=rect_length,
                    rectangle_width_m=rect_width,
                    title=f"{mission} {path_source} row={path_row} image_id={image_id}",
                    output_path=figure,
                )
                row["figure"] = str(figure)
                dense_file = figure.with_suffix(".npz")
                _save_dense_diagnostics(dense_file, dense, eval_named)
                row["dense_path_file"] = str(dense_file)
                path_rows.append(row)

                for radius in scan:
                    evaluation = evaluations[radius]
                    metric = evaluation_metrics(evaluation, dense, "scan")
                    rectangle_agreement = weighted_ratio(
                        evaluation.configuration_free == rectangle_eval.configuration_free, dense
                    )
                    radius_rows.append(
                        {
                            "mission": mission,
                            "path_source": path_source,
                            "path_row": path_row,
                            "image_id": image_id,
                            "requested_radius_m": radius,
                            "effective_radius_m": results[radius].effective_radius_m,
                            "root_valid": results[radius].root.configuration_valid,
                            "reachable_area_ratio": float(results[radius].reachable.mean()),
                            "expert_strict_pass": metric["scan_strict_pass"],
                            "expert_tolerance_1_pass": metric["scan_tolerance_1_pass"],
                            "expert_tolerance_5_pass": metric["scan_tolerance_5_pass"],
                            "forbidden_arc_ratio": metric["scan_forbidden_ratio"],
                            "rectangle_agreement_ratio": rectangle_agreement,
                            "rectangle_false_reject_ratio": weighted_ratio(
                                rectangle_eval.configuration_free
                                & ~evaluation.configuration_free,
                                dense,
                            ),
                            "rectangle_false_accept_ratio": weighted_ratio(
                                ~rectangle_eval.configuration_free
                                & evaluation.configuration_free,
                                dense,
                            ),
                        }
                    )

    model_prefixes = {
        "point_center": "center",
        "circle_0p26": "circle_0p26",
        "circle_0p61": "circle_0p61",
        "yaw_aware_rectangle": "rectangle",
    }
    aggregate = _aggregate(path_rows, model_prefixes)
    aggregate_by_root_reason = _aggregate_by_root_and_reason(path_rows, model_prefixes)
    radius_aggregate = _aggregate_radius_scan(radius_rows)
    representatives = _representatives(path_rows)
    _write_csv(output / "alignment_records.csv", alignment_rows)
    _write_csv(output / "path_metrics.csv", path_rows)
    _write_csv(output / "radius_scan.csv", radius_rows)
    _write_csv(output / "aggregate_summary.csv", aggregate)
    _write_csv(output / "aggregate_by_root_and_reason.csv", aggregate_by_root_reason)
    _write_csv(output / "radius_scan_summary.csv", radius_aggregate)
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (output / "representative_manifest.json").write_text(
        json.dumps(representatives, indent=2), encoding="utf-8"
    )
    summary = {
        "scope": "small-sample only; no formal Zarr writes",
        "num_paths": len(path_rows),
        "alignment_valid": sum(bool(row["path_alignment_valid"]) for row in alignment_rows),
        "num_alignment_records": len(alignment_rows),
        "coordinate_convention": "base frame: +x forward, +y left, positive theta CCW x->y",
        "dense_sampling_interval_m": float(cfg.dense_sampling_interval_m),
        "rectangle": {"length_m": rect_length, "width_m": rect_width, "num_offsets": len(offsets)},
        "aggregate": aggregate,
        "aggregate_by_root_and_reason": aggregate_by_root_reason,
        "radius_scan_summary": radius_aggregate,
        "representatives": representatives,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    log.info("Wrote %d path diagnostics to %s", len(path_rows), output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="dataset_builder/configs/teacher_a_path_calibration.yaml"
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = OmegaConf.load(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run(cfg)


if __name__ == "__main__":
    main()
