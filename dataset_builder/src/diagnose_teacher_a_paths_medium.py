"""Medium-scale, denominator-correct expert-path calibration for Teacher-A.

This is read-only with respect to all Zarr inputs and canonical Teacher-A.
Root anchoring is a diagnostic seed-only comparison; it never changes a
TeacherAResult or any configuration-free/unknown mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr
from omegaconf import OmegaConf

from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.path_calibration import (
    alignment_checks,
    build_clearance_maps,
    conditional_footprint_metrics,
    densify_path,
    diagnostic_root_anchor,
    evaluate_circular_teacher,
    evaluate_yaw_aware_rectangle,
    first_true_arc_length,
    prefix_before,
    rectangle_footprint_offsets,
    scoped_evaluation_metrics,
)
from dataset_builder.reachability.teacher_a import build_teacher_a
from dataset_builder.reachability.traversability import compute_limo_traversability

log = logging.getLogger(__name__)


MODEL_NAMES = {
    0.0: "point_center",
    0.28: "circle_0p28",
    0.61: "circle_0p61_upper_bound",
}


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


def _elevation_lookup(group: zarr.Group) -> tuple[dict[int, int], list[int]]:
    lookup: dict[int, int] = {}
    duplicates: set[int] = set()
    for row, image_id in enumerate(np.asarray(group["image_id"], dtype=np.int64)):
        key = int(image_id)
        if key in lookup:
            duplicates.add(key)
        else:
            lookup[key] = row
    for key in duplicates:
        lookup.pop(key, None)
    return lookup, sorted(duplicates)


def _temporal_image_sample(image_ids: np.ndarray, target: int) -> list[int]:
    unique = np.unique(np.asarray(image_ids, dtype=np.int64))
    if len(unique) <= target:
        return [int(value) for value in unique]
    positions = np.linspace(0, len(unique) - 1, target, dtype=np.int64)
    return [int(unique[position]) for position in positions]


def _path_lengths(paths: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(paths[:, :, :2], axis=1), axis=2).sum(axis=1)


def _geometric_stratified_rows(
    group: zarr.Group, target_images: int, paths_per_image: int
) -> tuple[list[int], list[dict]]:
    image_ids = np.asarray(group["image_id"], dtype=np.int64)
    paths = np.asarray(group["path"], dtype=np.float32)
    goals = np.asarray(group["goal"], dtype=np.float32)
    lengths = _path_lengths(paths)
    angles = np.arctan2(goals[:, 1], goals[:, 0])
    length_edges = np.quantile(lengths, [1.0 / 3.0, 2.0 / 3.0])
    length_bins = np.digitize(lengths, length_edges, right=False)
    direction_bins = np.digitize(angles, [-np.pi / 6.0, np.pi / 6.0], right=False)
    selected_images = _temporal_image_sample(image_ids, target_images)
    by_image: dict[int, list[int]] = defaultdict(list)
    for row, image_id in enumerate(image_ids):
        by_image[int(image_id)].append(row)

    selected_rows: list[int] = []
    records: list[dict] = []
    targets = [(length_bin, direction_bin) for length_bin in range(3) for direction_bin in range(3)]
    for image_position, image_id in enumerate(selected_images):
        candidates = by_image[image_id].copy()
        chosen: list[int] = []
        for path_position in range(min(paths_per_image, len(candidates))):
            target = targets[(image_position + 4 * path_position) % len(targets)]
            row = min(
                (candidate for candidate in candidates if candidate not in chosen),
                key=lambda candidate: (
                    abs(int(length_bins[candidate]) - target[0])
                    + abs(int(direction_bins[candidate]) - target[1]),
                    abs(float(lengths[candidate]) - float(np.median(lengths))),
                    candidate,
                ),
            )
            chosen.append(row)
            selected_rows.append(row)
            records.append(
                {
                    "image_id": image_id,
                    "path_row": row,
                    "path_length_m": float(lengths[row]),
                    "goal_angle_rad": float(angles[row]),
                    "length_stratum": int(length_bins[row]),
                    "direction_stratum": int(direction_bins[row]),
                    "target_length_stratum": target[0],
                    "target_direction_stratum": target[1],
                }
            )
    return sorted(selected_rows), records


def _teleop_image_rows(group: zarr.Group, target_images: int) -> tuple[list[int], list[dict]]:
    image_ids = np.asarray(group["image_id"], dtype=np.int64)
    selected_images = set(_temporal_image_sample(image_ids, target_images))
    rows = []
    records = []
    for row, image_id in enumerate(image_ids):
        if int(image_id) in selected_images:
            rows.append(row)
            records.append({"image_id": int(image_id), "path_row": row})
    return rows, records


def _sample_grid_mask(dense, geometry: MapGeometry, mask: np.ndarray) -> np.ndarray:
    indices = geometry.world_to_map_idx(dense.xytheta[:, :2])
    valid = geometry.valid_indices(indices)
    output = np.zeros(len(indices), dtype=bool)
    output[valid] = mask[indices[valid, 0], indices[valid, 1]]
    return output


def _anchored_evaluation(evaluation, dense, geometry, anchoring):
    reachable = _sample_grid_mask(dense, geometry, anchoring.reachable)
    disconnected = evaluation.configuration_free & ~reachable
    return replace(
        evaluation,
        reachable=reachable,
        disconnected=disconnected,
        forbidden=~reachable,
    )


def _rectangle_root_status(root_rectangle) -> dict:
    raw = bool(root_rectangle.configuration_free[0])
    blocked = bool(root_rectangle.overlaps_blocked[0])
    unknown = bool(root_rectangle.overlaps_unknown[0])
    outside = bool(root_rectangle.crosses_domain[0])
    anchor = bool(not raw and unknown and not blocked and not outside)
    if raw:
        reason = "raw_valid"
    elif anchor:
        reason = "seed_only:rectangle_unknown_self_occlusion"
    elif blocked:
        reason = "ineligible:rectangle_blocked"
    elif outside:
        reason = "ineligible:rectangle_outside"
    else:
        reason = "ineligible:rectangle_invalid"
    return {
        "raw_valid": raw,
        "anchored_valid": raw or anchor,
        "anchor_applied": anchor,
        "anchor_reason": reason,
        "root_overlaps_blocked": blocked,
        "root_overlaps_unknown": unknown,
        "root_crosses_domain": outside,
    }


def _scope_rows_for_evaluation(
    *,
    base: dict,
    model: str,
    evaluation,
    anchored_evaluation,
    root_status: dict,
    dense,
    tolerance: float,
) -> list[dict]:
    local_prefix = prefix_before(evaluation.crosses_domain, dense)
    observable_prefix = prefix_before(
        evaluation.crosses_domain | evaluation.overlaps_unknown, dense
    )
    scopes = {
        "full_path": np.ones(len(dense.xytheta), dtype=bool),
        "local_domain_prefix": local_prefix,
        "observable_domain_prefix": observable_prefix,
    }
    domain_metrics = scoped_evaluation_metrics(
        evaluation, dense, local_prefix, tolerance_ratio=tolerance
    )
    observable_metrics = scoped_evaluation_metrics(
        evaluation, dense, observable_prefix, tolerance_ratio=tolerance
    )
    whole_path_inside_domain = not bool(np.any(evaluation.crosses_domain))
    rows = []
    for scope_name, mask in scopes.items():
        metrics = scoped_evaluation_metrics(
            evaluation, dense, mask, tolerance_ratio=tolerance
        )
        anchored_metrics = scoped_evaluation_metrics(
            anchored_evaluation, dense, mask, tolerance_ratio=tolerance
        )
        rows.append(
            {
                **base,
                "footprint_model": model,
                "scope": scope_name,
                "root_raw_valid": root_status["raw_valid"],
                "root_anchored_valid": root_status["anchored_valid"],
                "anchor_applied": root_status["anchor_applied"],
                "anchor_reason": root_status["anchor_reason"],
                "domain_evaluable": whole_path_inside_domain,
                "observable_prefix_evaluable": observable_metrics["evaluable"],
                **metrics,
                "anchored_forbidden_ratio": anchored_metrics["forbidden_ratio"],
                "anchored_strict_pass": anchored_metrics["strict_pass"],
                "anchored_tolerant_pass": anchored_metrics["tolerant_pass"],
                "anchored_disconnected_ratio": anchored_metrics["disconnected_ratio"],
            }
        )
    return rows


def _aggregate_scope_rows(scope_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in scope_rows:
        groups[(row["mission"], row["path_source"], row["footprint_model"], row["scope"])].append(row)
        groups[("ALL_MISSIONS", row["path_source"], row["footprint_model"], row["scope"])].append(row)
    output = []
    for (mission, source, model, scope), rows in groups.items():
        aligned = [row for row in rows if row["alignment_valid"]]
        evaluable = [row for row in aligned if row["evaluable"]]
        strict = [row for row in evaluable if row["strict_pass"]]
        tolerant = [row for row in evaluable if row["tolerant_pass"]]
        anchored_strict = [row for row in evaluable if row["anchored_strict_pass"]]
        anchored_tolerant = [row for row in evaluable if row["anchored_tolerant_pass"]]
        output.append(
            {
                "mission": mission,
                "path_source": source,
                "footprint_model": model,
                "scope": scope,
                "total_paths": len(rows),
                "alignment_valid_paths": len(aligned),
                "root_raw_valid_paths": sum(bool(row["root_raw_valid"]) for row in aligned),
                "root_anchored_valid_paths": sum(bool(row["root_anchored_valid"]) for row in aligned),
                "domain_evaluable_paths": sum(bool(row["domain_evaluable"]) for row in aligned),
                "observable_prefix_evaluable_paths": sum(
                    bool(row["observable_prefix_evaluable"]) for row in aligned
                ),
                "scope_evaluable_paths": len(evaluable),
                "strict_pass_paths": len(strict),
                "tolerant_pass_paths": len(tolerant),
                "anchored_strict_pass_paths": len(anchored_strict),
                "anchored_tolerant_pass_paths": len(anchored_tolerant),
                "strict_pass_rate_of_evaluable": len(strict) / max(len(evaluable), 1),
                "tolerant_pass_rate_of_evaluable": len(tolerant) / max(len(evaluable), 1),
                "anchored_tolerant_pass_rate_of_evaluable": len(anchored_tolerant)
                / max(len(evaluable), 1),
                "collision_strict_pass_paths": sum(
                    bool(row["collision_strict_pass"]) for row in evaluable
                ),
                "collision_tolerant_pass_paths": sum(
                    bool(row["collision_tolerant_pass"]) for row in evaluable
                ),
                "locally_blocked_conflict_paths": sum(
                    float(row["locally_blocked_ratio"]) > 1e-12 for row in evaluable
                ),
                "clearance_conflict_paths": sum(
                    float(row["clearance_ratio"]) > 1e-12 for row in evaluable
                ),
                "unknown_paths_in_scope": sum(
                    float(row["unknown_ratio"]) > 1e-12 for row in evaluable
                ),
                "outside_paths_in_scope": sum(
                    float(row["outside_ratio"]) > 1e-12 for row in evaluable
                ),
                "disconnected_paths": sum(
                    float(row["disconnected_ratio"]) > 1e-12 for row in evaluable
                ),
            }
        )
    return output


def _build_image_rows(scope_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in scope_rows:
        groups[
            (
                row["mission"],
                row["path_source"],
                row["image_id"],
                row["footprint_model"],
                row["scope"],
            )
        ].append(row)
    image_rows = []
    for (mission, source, image_id, model, scope), rows in groups.items():
        evaluable = [row for row in rows if row["alignment_valid"] and row["evaluable"]]
        image_rows.append(
            {
                "mission": mission,
                "path_source": source,
                "image_id": image_id,
                "footprint_model": model,
                "scope": scope,
                "total_paths": len(rows),
                "alignment_valid_paths": sum(bool(row["alignment_valid"]) for row in rows),
                "root_raw_valid": bool(rows[0]["root_raw_valid"]),
                "root_anchored_valid": bool(rows[0]["root_anchored_valid"]),
                "domain_any_evaluable": any(
                    bool(row["domain_evaluable"]) for row in rows
                ),
                "domain_evaluable": all(
                    bool(row["domain_evaluable"]) for row in rows
                ),
                "observable_prefix_any_evaluable": any(
                    bool(row["observable_prefix_evaluable"]) for row in rows
                ),
                "observable_prefix_evaluable": all(
                    bool(row["observable_prefix_evaluable"]) for row in rows
                ),
                "scope_evaluable_paths": len(evaluable),
                "strict_pass_paths": sum(bool(row["strict_pass"]) for row in evaluable),
                "tolerant_pass_paths": sum(bool(row["tolerant_pass"]) for row in evaluable),
                "strict_pass_image": bool(evaluable)
                and all(bool(row["strict_pass"]) for row in evaluable),
                "tolerant_pass_image": bool(evaluable)
                and all(bool(row["tolerant_pass"]) for row in evaluable),
                "anchored_tolerant_pass_image": bool(evaluable)
                and all(bool(row["anchored_tolerant_pass"]) for row in evaluable),
            }
        )

    summary_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in image_rows:
        summary_groups[(row["mission"], row["path_source"], row["footprint_model"], row["scope"])].append(row)
        summary_groups[("ALL_MISSIONS", row["path_source"], row["footprint_model"], row["scope"])].append(row)
    summaries = []
    for (mission, source, model, scope), rows in summary_groups.items():
        summaries.append(
            {
                "mission": mission,
                "path_source": source,
                "footprint_model": model,
                "scope": scope,
                "total_images": len(rows),
                "total_paths": sum(int(row["total_paths"]) for row in rows),
                "alignment_valid_images": sum(
                    int(row["alignment_valid_paths"]) == int(row["total_paths"]) for row in rows
                ),
                "root_raw_valid_images": sum(bool(row["root_raw_valid"]) for row in rows),
                "root_anchored_valid_images": sum(
                    bool(row["root_anchored_valid"]) for row in rows
                ),
                "domain_evaluable_images": sum(bool(row["domain_evaluable"]) for row in rows),
                "domain_any_evaluable_images": sum(
                    bool(row["domain_any_evaluable"]) for row in rows
                ),
                "observable_prefix_evaluable_images": sum(
                    bool(row["observable_prefix_evaluable"]) for row in rows
                ),
                "observable_prefix_any_evaluable_images": sum(
                    bool(row["observable_prefix_any_evaluable"]) for row in rows
                ),
                "strict_pass_images": sum(bool(row["strict_pass_image"]) for row in rows),
                "tolerant_pass_images": sum(bool(row["tolerant_pass_image"]) for row in rows),
                "anchored_tolerant_pass_images": sum(
                    bool(row["anchored_tolerant_pass_image"]) for row in rows
                ),
            }
        )
    return image_rows, summaries


def _aggregate_radius_rows(radius_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in radius_rows:
        groups[(row["mission"], row["path_source"], row["requested_radius_m"], row["effective_radius_m"])].append(row)
        groups[("ALL_MISSIONS", row["path_source"], row["requested_radius_m"], row["effective_radius_m"])].append(row)
    output = []
    for (mission, source, requested, effective), rows in groups.items():
        unique_images = {}
        for row in rows:
            # image_id is mission-local.  Preserve the mission component when
            # producing the synthetic ALL_MISSIONS aggregate.
            unique_images.setdefault((row["mission"], row["image_id"]), row)
        support = sum(float(row["conditional_support_arc_length_m"]) for row in rows)
        rectangle_free = sum(
            float(row["conditional_rectangle_free_arc_length_m"]) for row in rows
        )
        rectangle_blocked = sum(
            float(row["conditional_rectangle_blocked_arc_length_m"]) for row in rows
        )
        agreement = sum(float(row["conditional_agreement_arc_length_m"]) for row in rows)
        false_reject = sum(float(row["circle_false_reject_arc_length_m"]) for row in rows)
        false_accept = sum(float(row["circle_false_accept_arc_length_m"]) for row in rows)
        conflict = sum(
            float(row["conditional_clearance_conflict_arc_length_m"]) for row in rows
        )
        obs = [row for row in rows if row["observable_prefix_evaluable"]]
        output.append(
            {
                "mission": mission,
                "path_source": source,
                "requested_radius_m": requested,
                "effective_radius_m": effective,
                "total_paths": len(rows),
                "total_images": len(unique_images),
                "alignment_valid_paths": sum(bool(row["alignment_valid"]) for row in rows),
                "root_raw_valid_paths": sum(bool(row["root_raw_valid"]) for row in rows),
                "root_anchored_valid_paths": sum(
                    bool(row["root_anchored_valid"]) for row in rows
                ),
                "root_raw_valid_images": sum(
                    bool(row["root_raw_valid"]) for row in unique_images.values()
                ),
                "root_anchored_valid_images": sum(
                    bool(row["root_anchored_valid"]) for row in unique_images.values()
                ),
                "observable_prefix_evaluable_paths": len(obs),
                "expert_observable_prefix_strict_pass_paths": sum(
                    bool(row["observable_prefix_strict_pass"]) for row in obs
                ),
                "expert_observable_prefix_tolerant_pass_paths": sum(
                    bool(row["observable_prefix_tolerant_pass"]) for row in obs
                ),
                "expert_observable_prefix_tolerant_pass_rate": sum(
                    bool(row["observable_prefix_tolerant_pass"]) for row in obs
                )
                / max(len(obs), 1),
                "conditional_support_arc_length_m": support,
                "conditional_rectangle_free_arc_length_m": rectangle_free,
                "conditional_rectangle_blocked_arc_length_m": rectangle_blocked,
                "conditional_agreement_arc_length_m": agreement,
                "circle_false_reject_arc_length_m": false_reject,
                "circle_false_accept_arc_length_m": false_accept,
                "conditional_clearance_conflict_arc_length_m": conflict,
                "conditional_circle_rectangle_agreement": agreement / support
                if support > 1e-12
                else None,
                "circle_false_reject_ratio_given_rectangle_free": false_reject
                / rectangle_free
                if rectangle_free > 1e-12
                else None,
                "circle_false_accept_ratio_given_rectangle_blocked": false_accept
                / rectangle_blocked
                if rectangle_blocked > 1e-12
                else None,
                "conditional_clearance_conflict_ratio": conflict / support
                if support > 1e-12
                else None,
                "root_raw_valid_rate_images": sum(
                    bool(row["root_raw_valid"]) for row in unique_images.values()
                )
                / max(len(unique_images), 1),
                "root_anchored_valid_rate_images": sum(
                    bool(row["root_anchored_valid"]) for row in unique_images.values()
                )
                / max(len(unique_images), 1),
                "reachable_area_ratio_raw_mean_images": float(
                    np.mean([float(row["reachable_area_ratio_raw"]) for row in unique_images.values()])
                ),
                "reachable_area_ratio_anchored_mean_images": float(
                    np.mean(
                        [float(row["reachable_area_ratio_anchored"]) for row in unique_images.values()]
                    )
                ),
                "anchor_false_crossing_images": sum(
                    bool(row["anchor_crosses_unknown_or_forbidden"])
                    for row in unique_images.values()
                ),
            }
        )
    return output


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
    radii = [float(value) for value in cfg.radius_scan_m]
    tolerance = float(cfg.tolerant_forbidden_ratio)
    path_rows: list[dict] = []
    scope_rows: list[dict] = []
    radius_rows: list[dict] = []
    root_rows: list[dict] = []
    selection: dict[str, dict] = {}

    for mission_value in cfg.missions:
        mission = str(mission_value)
        mission_dir = Path(cfg.dataset_root) / mission
        elevation_group = zarr.open_group(str(mission_dir / "data/elevation_map"), mode="r")
        camera_group = zarr.open_group(str(mission_dir / "data/hdr_front"), mode="r")
        elevation_lookup, duplicates = _elevation_lookup(elevation_group)
        groups = {
            "geometric": zarr.open_group(str(mission_dir / "data/geometric_paths"), mode="r"),
            "teleop": zarr.open_group(str(mission_dir / "data/teleop_paths"), mode="r"),
        }
        geo_rows, geo_selection = _geometric_stratified_rows(
            groups["geometric"],
            int(cfg.geometric_target_images_per_mission),
            int(cfg.geometric_paths_per_image),
        )
        tele_rows, tele_selection = _teleop_image_rows(
            groups["teleop"], int(cfg.teleop_target_images_per_mission)
        )
        selected = {"geometric": geo_rows, "teleop": tele_rows}
        selection[mission] = {
            "elevation_duplicate_image_ids": duplicates,
            "geometric": geo_selection,
            "teleop": tele_selection,
            "geometric_stratum_counts": {
                f"length_{key[0]}_direction_{key[1]}": value
                for key, value in Counter(
                    (row["length_stratum"], row["direction_stratum"])
                    for row in geo_selection
                ).items()
            },
        }
        cache = {}
        processed_images = 0

        for source, selected_rows in selected.items():
            group = groups[source]
            frame_id = str(group.attrs.get("frame_id", ""))
            for path_row in selected_rows:
                image_id = int(group["image_id"][path_row])
                elevation_row = elevation_lookup.get(image_id)
                path = np.asarray(group["path"][path_row], dtype=np.float32)
                goal = np.asarray(group["goal"][path_row], dtype=np.float32)
                checks = alignment_checks(path, frame_id=frame_id, elevation_row=elevation_row)
                timestamp = (
                    float(elevation_group["timestamp"][elevation_row])
                    if elevation_row is not None
                    else None
                )
                camera_timestamp = float(camera_group["timestamp"][image_id])
                timestamp_delta = (
                    abs(timestamp - camera_timestamp) if timestamp is not None else None
                )
                alignment_valid = bool(
                    checks["alignment_valid"]
                    and timestamp_delta is not None
                    and timestamp_delta <= float(cfg.timestamp_tolerance_s)
                )
                base = {
                    "mission": mission,
                    "path_source": source,
                    "path_row": path_row,
                    "image_id": image_id,
                    "elevation_row": elevation_row,
                    "timestamp": timestamp,
                    "timestamp_delta_s": timestamp_delta,
                    "alignment_valid": alignment_valid,
                }
                if not alignment_valid:
                    path_rows.append({**base, **checks})
                    continue

                if image_id not in cache:
                    elevation = np.asarray(
                        elevation_group["elevation"][elevation_row], dtype=np.float32
                    )
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
                        for radius in radii
                    }
                    obstacle_clearance, unknown_clearance = build_clearance_maps(
                        trav.known_trav,
                        trav.risk,
                        float(mppi_cfg.fatal_th),
                        geometry.resolution,
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
                        obstacle_clearance=obstacle_clearance,
                        unknown_clearance=unknown_clearance,
                    )
                    rectangle_root = _rectangle_root_status(root_rectangle)
                    anchoring = {
                        radius: diagnostic_root_anchor(
                            result,
                            rectangle_overlaps_blocked=bool(root_rectangle.overlaps_blocked[0]),
                            rectangle_overlaps_unknown=bool(root_rectangle.overlaps_unknown[0]),
                            rectangle_crosses_domain=bool(root_rectangle.crosses_domain[0]),
                        )
                        for radius, result in results.items()
                    }
                    cache[image_id] = (
                        trav,
                        results,
                        anchoring,
                        rectangle_root,
                        obstacle_clearance,
                        unknown_clearance,
                    )
                    for radius in radii:
                        anchor = anchoring[radius]
                        root_rows.append(
                            {
                                "mission": mission,
                                "image_id": image_id,
                                "footprint_model": f"circle_r{radius:.2f}",
                                "requested_radius_m": radius,
                                "effective_radius_m": results[radius].effective_radius_m,
                                "root_raw_valid": anchor.raw_valid,
                                "root_anchored_valid": anchor.anchored_valid,
                                "anchor_applied": anchor.anchor_applied,
                                "anchor_reason": anchor.anchor_reason,
                                "reachable_area_ratio_raw": float(results[radius].reachable.mean()),
                                "reachable_area_ratio_anchored": anchor.reachable_area_ratio,
                                "seeded_nonfree_count": anchor.seeded_nonfree_count,
                                "traversed_nonfree_count_excluding_seed": anchor.traversed_nonfree_count_excluding_seed,
                                "anchor_crosses_unknown_or_forbidden": anchor.crosses_unknown_or_forbidden,
                                **{
                                    key: rectangle_root[key]
                                    for key in (
                                        "root_overlaps_blocked",
                                        "root_overlaps_unknown",
                                        "root_crosses_domain",
                                    )
                                },
                            }
                        )
                    processed_images += 1
                    if processed_images % 25 == 0:
                        log.info("%s: processed %d unique images", mission, processed_images)

                (
                    trav,
                    results,
                    anchoring,
                    rectangle_root,
                    obstacle_clearance,
                    unknown_clearance,
                ) = cache[image_id]
                dense = densify_path(path, float(cfg.dense_sampling_interval_m))
                circular = {
                    radius: evaluate_circular_teacher(
                        dense, result, obstacle_clearance, unknown_clearance
                    )
                    for radius, result in results.items()
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
                path_row_output = {
                    **base,
                    **checks,
                    "path_length_m": dense.total_length_m,
                    "endpoint_distance_m": float(np.linalg.norm(path[-1, :2])),
                    "endpoint_goal_error_m": float(np.linalg.norm(path[-1, :2] - goal[:2])),
                    "num_dense_samples": len(dense.xytheta),
                    "first_unknown_s": first_true_arc_length(
                        rectangle_eval.overlaps_unknown, dense
                    ),
                    "first_outside_s": first_true_arc_length(
                        rectangle_eval.crosses_domain, dense
                    ),
                    "first_blocked_s": first_true_arc_length(
                        rectangle_eval.locally_blocked, dense
                    ),
                    "first_clearance_s": first_true_arc_length(
                        rectangle_eval.clearance_blocked, dense
                    ),
                    "rectangle_root_raw_valid": rectangle_root["raw_valid"],
                    "rectangle_root_anchored_valid": rectangle_root["anchored_valid"],
                    "rectangle_anchor_reason": rectangle_root["anchor_reason"],
                }
                path_rows.append(path_row_output)

                for radius, evaluation in circular.items():
                    result = results[radius]
                    anchor = anchoring[radius]
                    model = f"circle_r{result.effective_radius_m:.2f}"
                    root_status = {
                        "raw_valid": anchor.raw_valid,
                        "anchored_valid": anchor.anchored_valid,
                        "anchor_applied": anchor.anchor_applied,
                        "anchor_reason": anchor.anchor_reason,
                    }
                    anchored_eval = _anchored_evaluation(
                        evaluation, dense, geometry, anchor
                    )
                    scope_rows.extend(
                        _scope_rows_for_evaluation(
                            base=base,
                            model=model,
                            evaluation=evaluation,
                            anchored_evaluation=anchored_eval,
                            root_status=root_status,
                            dense=dense,
                            tolerance=tolerance,
                        )
                    )
                    conditional = conditional_footprint_metrics(
                        evaluation, rectangle_eval, dense
                    )
                    observable_mask = prefix_before(
                        evaluation.crosses_domain | evaluation.overlaps_unknown, dense
                    )
                    observable_metrics = scoped_evaluation_metrics(
                        evaluation,
                        dense,
                        observable_mask,
                        tolerance_ratio=tolerance,
                    )
                    radius_rows.append(
                        {
                            **base,
                            "requested_radius_m": radius,
                            "effective_radius_m": result.effective_radius_m,
                            "root_raw_valid": anchor.raw_valid,
                            "root_anchored_valid": anchor.anchored_valid,
                            "anchor_applied": anchor.anchor_applied,
                            "anchor_reason": anchor.anchor_reason,
                            "anchor_crosses_unknown_or_forbidden": anchor.crosses_unknown_or_forbidden,
                            "reachable_area_ratio_raw": float(result.reachable.mean()),
                            "reachable_area_ratio_anchored": anchor.reachable_area_ratio,
                            "observable_prefix_evaluable": observable_metrics["evaluable"],
                            "observable_prefix_strict_pass": observable_metrics["strict_pass"],
                            "observable_prefix_tolerant_pass": observable_metrics["tolerant_pass"],
                            "observable_prefix_collision_ratio": observable_metrics["collision_ratio"],
                            **conditional,
                        }
                    )

                rectangle_scope_rows = _scope_rows_for_evaluation(
                    base=base,
                    model="yaw_aware_rectangle",
                    evaluation=rectangle_eval,
                    anchored_evaluation=rectangle_eval,
                    root_status=rectangle_root,
                    dense=dense,
                    tolerance=tolerance,
                )
                scope_rows.extend(rectangle_scope_rows)

    aggregate_rows = _aggregate_scope_rows(scope_rows)
    image_rows, image_summary = _build_image_rows(scope_rows)
    radius_summary = _aggregate_radius_rows(radius_rows)
    _write_csv(output / "path_metrics.csv", path_rows)
    _write_csv(output / "path_scope_metrics.csv", scope_rows)
    _write_csv(output / "path_radius_metrics.csv", radius_rows)
    _write_csv(output / "root_diagnostics.csv", root_rows)
    _write_csv(output / "aggregate_denominators.csv", aggregate_rows)
    _write_csv(output / "image_metrics.csv", image_rows)
    _write_csv(output / "image_summary.csv", image_summary)
    _write_csv(output / "radius_summary.csv", radius_summary)
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2, default=_json_default), encoding="utf-8"
    )
    summary = {
        "scope": "medium-scale read-only calibration; no Teacher-A or Zarr mutation",
        "definitions": {
            "domain_evaluable_path": "the entire dense path stays inside the model planning domain",
            "local_domain_prefix": "continuous prefix strictly before first outside-domain sample",
            "observable_domain_prefix": "continuous prefix strictly before first unknown-footprint/center or outside-domain sample",
            "tolerant_pass": f"forbidden arc ratio <= {tolerance}",
            "conditional_footprint_support": "mutually inside-domain with known center and known footprint support",
            "conditional_clearance_conflict": "circle/rectangle footprint blocked/free XOR on conditional support",
            "root_anchor": "only root cell is added to Dijkstra seed mask; final reachable intersects canonical configuration_free",
        },
        "num_path_rows": len(path_rows),
        "num_scope_rows": len(scope_rows),
        "num_radius_rows": len(radius_rows),
        "num_root_rows": len(root_rows),
        "aggregate_denominators": aggregate_rows,
        "image_summary": image_summary,
        "radius_summary": radius_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    log.info("Wrote %d medium-scale path rows to %s", len(path_rows), output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="dataset_builder/configs/teacher_a_path_calibration_medium.yaml",
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
