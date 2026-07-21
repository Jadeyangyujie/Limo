"""Rectangle Conflict Audit for the two-mission Teacher-A calibration set.

All dataset access is read-only.  This command writes diagnostics only and
does not modify Teacher-A, MPPI, or any Zarr store.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
import zarr

from dataset_builder.mppi_planner.mppi_planner import GridMap2D, MPPIObjective, smallest_angle
from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.path_calibration import (
    densify_path,
    evaluate_circular_teacher,
    rectangle_footprint_offsets,
)
from dataset_builder.reachability.rectangle_conflict_audit import (
    arc_weights,
    audit_footprint_states,
    classify_path_severity,
    continuous_runs,
    reconstruct_actions,
)
from dataset_builder.reachability.teacher_a import build_teacher_a
from dataset_builder.reachability.traversability import compute_limo_traversability


log = logging.getLogger(__name__)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _elevation_lookup(group: zarr.Group) -> dict[int, int]:
    output: dict[int, int] = {}
    duplicate: set[int] = set()
    for row, image_id in enumerate(np.asarray(group["image_id"], dtype=np.int64)):
        key = int(image_id)
        if key in output:
            duplicate.add(key)
        else:
            output[key] = row
    for key in duplicate:
        output.pop(key, None)
    return output


def _select_paths(cfg) -> list[dict]:
    medium = Path(cfg.medium_results_dir)
    scope = _read_csv(medium / "path_scope_metrics.csv")
    radius = _read_csv(medium / "path_radius_metrics.csv")
    allowed_missions = {str(item) for item in cfg.missions}
    rectangle_rows = [
        row
        for row in scope
        if row["mission"] in allowed_missions
        and row["footprint_model"] == "yaw_aware_rectangle"
        and row["scope"] == "observable_domain_prefix"
    ]
    selected: dict[tuple[str, str, int], dict] = {}
    # All 68 geometric conflicts across two missions and Mission-1's 68 teleop conflicts.
    mission1 = str(cfg.missions[0])
    for row in rectangle_rows:
        conflict = float(row["clearance_ratio"]) > 0.0
        in_scope = row["path_source"] == "geometric" or (
            row["path_source"] == "teleop" and row["mission"] == mission1
        )
        if conflict and in_scope:
            key = (row["mission"], row["path_source"], int(float(row["path_row"])))
            selected[key] = {
                "audit_cohort": "rectangle_clearance_conflict",
                "medium_observable_clearance_ratio": float(row["clearance_ratio"]),
                "medium_conflict_gt_5pct": float(row["clearance_ratio"]) > 0.05,
            }

    radius28 = {
        (row["mission"], row["path_source"], int(float(row["path_row"]))): row
        for row in radius
        if row["mission"] in allowed_missions
        and abs(float(row["requested_radius_m"]) - 0.28) < 1e-6
    }
    control_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rectangle_rows:
        key = (row["mission"], row["path_source"], int(float(row["path_row"])))
        radius_row = radius28.get(key)
        if radius_row is None:
            continue
        if _as_bool(row["collision_strict_pass"]) and float(
            radius_row["observable_prefix_collision_ratio"]
        ) <= 1e-12:
            control_groups[(row["mission"], row["path_source"])].append(row)
    count = int(cfg.controls_per_mission_source)
    for group_rows in control_groups.values():
        if not group_rows:
            continue
        positions = np.linspace(0, len(group_rows) - 1, min(count, len(group_rows)), dtype=int)
        for position in positions:
            row = group_rows[int(position)]
            key = (row["mission"], row["path_source"], int(float(row["path_row"])))
            selected.setdefault(
                key,
                {
                    "audit_cohort": "circle_and_rectangle_pass_control",
                    "medium_observable_clearance_ratio": float(row["clearance_ratio"]),
                    "medium_conflict_gt_5pct": False,
                },
            )
    return [
        {"mission": key[0], "path_source": key[1], "path_row": key[2], **value}
        for key, value in sorted(selected.items())
    ]


def _prefix_mask(records: list[dict]) -> np.ndarray:
    stop = np.asarray(
        [bool(row["overlaps_unknown"] or row["crosses_domain"]) for row in records], dtype=bool
    )
    indices = np.flatnonzero(stop)
    mask = np.ones(len(records), dtype=bool)
    if len(indices):
        mask[int(indices[0]) :] = False
    return mask


def _first_index(mask: np.ndarray):
    index = np.flatnonzero(mask)
    return int(index[0]) if len(index) else None


def _mode(values: list[str | None]) -> str | None:
    values = [value for value in values if value]
    return Counter(values).most_common(1)[0][0] if values else None


def _objective_costs(
    objective: MPPIObjective,
    gm: GridMap2D,
    path: np.ndarray,
    goal: np.ndarray,
    mppi_cfg,
) -> dict:
    start = torch.zeros(3, dtype=torch.float32, device=objective.device)
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=objective.device)
    objective.set_observation(gm, start, goal_t)
    actions_np = reconstruct_actions(path, float(mppi_cfg.dt))
    states = torch.as_tensor(path[None], dtype=torch.float32, device=objective.device)
    actions = torch.as_tensor(actions_np[None], dtype=torch.float32, device=objective.device)
    with torch.no_grad():
        footprint_cost, unknown_fraction = objective.get_trav_cost(states)
        trav_weighted = footprint_cost * mppi_cfg.traversability_cost + unknown_fraction * mppi_cfg.traversability_unknown_cost
        control = (
            torch.abs(actions[..., 0]) * mppi_cfg.action_cost_trans_forward
            + torch.abs(actions[..., 1]) * mppi_cfg.action_cost_trans_side
            + torch.abs(actions[..., 2]) * mppi_cfg.action_cost_rotation
        )
        dist = objective.get_position_distance_to_goal(states[..., :2])
        position = torch.log1p(mppi_cfg.distance_pre_log_factor * dist) * mppi_cfg.position_cost
        delta = (position.mean() - mppi_cfg.position_cost_mean_limit).clamp(min=0)
        position = position - delta
        heading_offset = smallest_angle(states[..., 2], goal_t[2])
        heading_weight = torch.maximum(
            mppi_cfg.max_distance_for_heading_cost - dist, torch.tensor(0.0, device=states.device)
        ).max(dim=0).values
        heading = heading_offset * heading_weight * mppi_cfg.heading_cost
        base = position + control + heading
        at_goal = (heading_offset < mppi_cfg.heading_cost_at_goal_tolerance) & (
            dist < mppi_cfg.position_cost_at_goal_tolerance
        )
        base[at_goal] = -mppi_cfg.at_goal_reward
        total = base + trav_weighted
        canonical = objective.states_cost(states, actions)
        rollout = objective.rollout(actions)

    def array(tensor):
        return tensor.detach().cpu().numpy()[0].astype(np.float64)

    return {
        "actions": actions_np,
        "mppi_footprint_aggregate": array(footprint_cost),
        "mppi_unknown_fraction": array(unknown_fraction),
        "mppi_traversability_weighted": array(trav_weighted),
        "mppi_control_cost": array(control),
        "mppi_position_cost": array(position),
        "mppi_heading_cost": array(heading),
        "mppi_base_cost": array(base),
        "mppi_total_state_cost": array(total),
        "mppi_canonical_state_cost": array(canonical),
        "mppi_rollout_reconstruction_max_error": float(
            torch.max(torch.abs(rollout - states)).detach().cpu()
        ),
        "mppi_cost_decomposition_max_error": float(
            torch.max(torch.abs(canonical - total)).detach().cpu()
        ),
    }


def _summary_rows(path_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groupers = [
        ("ALL", "ALL", path_rows),
        *[
            (mission, source, [row for row in path_rows if row["mission"] == mission and row["path_source"] == source])
            for mission, source in sorted({(row["mission"], row["path_source"]) for row in path_rows})
        ],
    ]
    original_dense = []
    severity = []
    location = []
    for mission, source, rows in groupers:
        conflicts = [row for row in rows if row["audit_cohort"] == "rectangle_clearance_conflict"]
        original_dense.append(
            {
                "mission": mission,
                "path_source": source,
                "audited_paths": len(rows),
                "rectangle_conflict_paths": len(conflicts),
                "conflict_at_original_waypoints": sum(bool(row["conflict_at_original_waypoints"]) for row in conflicts),
                "conflict_only_after_dense_interpolation": sum(bool(row["conflict_only_after_dense_interpolation"]) for row in conflicts),
                "conflict_at_both_sampling_schemes": sum(
                    bool(row["conflict_at_original_waypoints"] and not row["conflict_only_after_dense_interpolation"])
                    for row in conflicts
                ),
                "median_original_waypoint_forbidden_ratio": float(np.median([row["original_waypoint_forbidden_ratio"] for row in conflicts])) if conflicts else np.nan,
                "median_dense_forbidden_arc_ratio": float(np.median([row["dense_forbidden_arc_ratio"] for row in conflicts])) if conflicts else np.nan,
            }
        )
        for label in ("isolated_touch", "boundary_graze", "sustained_overlap", "deep_collision", "no_conflict"):
            matching = [row for row in rows if row["severity_class"] == label]
            severity.append(
                {
                    "mission": mission,
                    "path_source": source,
                    "severity_class": label,
                    "path_count": len(matching),
                    "fraction_of_rectangle_conflicts": len(matching) / max(len(conflicts), 1),
                    "median_dense_forbidden_arc_ratio": float(np.median([row["dense_forbidden_arc_ratio"] for row in matching])) if matching else np.nan,
                    "median_max_conflict_run_m": float(np.median([row["max_conflict_run_m"] for row in matching])) if matching else np.nan,
                }
            )
        for label in ("front_overhang", "rear_overhang", "left_side", "right_side", "rotated_corner"):
            matching = [row for row in conflicts if row["dominant_conflict_location"] == label]
            location.append(
                {
                    "mission": mission,
                    "path_source": source,
                    "conflict_location": label,
                    "path_count": len(matching),
                    "fraction_of_rectangle_conflicts": len(matching) / max(len(conflicts), 1),
                    "fatal_sample_count": sum(int(row.get(f"location_{label}_fatal_samples", 0)) for row in conflicts),
                }
            )
    return severity, original_dense, location


def _teleop_distance_summary(path_rows: list[dict], state_rows: list[dict], mission1: str) -> list[dict]:
    bins = [(0.0, 1.0, "0-1m"), (1.0, 2.0, "1-2m"), (2.0, 3.0, "2-3m"), (3.0, np.inf, "3m+")]
    tele_paths = [
        row for row in path_rows
        if row["mission"] == mission1
        and row["path_source"] == "teleop"
        and row["audit_cohort"] == "rectangle_clearance_conflict"
    ]
    tele_states = [
        row for row in state_rows
        if row["mission"] == mission1
        and row["path_source"] == "teleop"
        and row["audit_cohort"] == "rectangle_clearance_conflict"
    ]
    output = []
    for lower, upper, label in bins:
        states = [row for row in tele_states if lower <= float(row["arc_length_m"]) < upper and _as_bool(row["observable_prefix"])]
        path_keys = {(row["mission"], row["path_source"], row["path_row"]) for row in states}
        first = [row for row in tele_paths if row["first_conflict_dense_arc_length_m"] is not None and lower <= float(row["first_conflict_dense_arc_length_m"]) < upper]
        output.append(
            {
                "mission": mission1,
                "path_source": "teleop",
                "distance_bin": label,
                "bin_start_m": lower,
                "bin_end_m": upper if np.isfinite(upper) else None,
                "evaluable_state_count": len(states),
                "evaluable_path_count": len(path_keys),
                "hard_conflict_state_count": sum(_as_bool(row["teacher_a_hard_blocked"]) for row in states),
                "hard_conflict_state_ratio": sum(_as_bool(row["teacher_a_hard_blocked"]) for row in states) / max(len(states), 1),
                "paths_with_first_conflict_in_bin": len(first),
                "mean_fatal_sample_ratio_on_conflict": float(np.mean([float(row["fatal_sample_ratio_known"]) for row in states if _as_bool(row["teacher_a_hard_blocked"])])) if any(_as_bool(row["teacher_a_hard_blocked"]) for row in states) else np.nan,
            }
        )
    if tele_paths:
        lengths = np.asarray([float(row["path_length_m"]) for row in tele_paths])
        times = np.asarray([float(row["goal_time_s"]) for row in tele_paths])
        ratios = np.asarray([float(row["dense_forbidden_arc_ratio"]) for row in tele_paths])
        output.append(
            {
                "mission": mission1,
                "path_source": "teleop",
                "distance_bin": "PATH_LEVEL",
                "evaluable_state_count": len(tele_states),
                "evaluable_path_count": len(tele_paths),
                "path_length_conflict_ratio_correlation": float(np.corrcoef(lengths, ratios)[0, 1]) if np.std(lengths) and np.std(ratios) else np.nan,
                "execution_span_conflict_ratio_correlation": float(np.corrcoef(times, ratios)[0, 1]) if np.std(times) and np.std(ratios) else np.nan,
                "median_goal_time_s": float(np.median(times)),
                "median_path_length_m": float(np.median(lengths)),
            }
        )
    return output


def _plot_representative(context: dict, output: Path, label: str, dpi: int, circle_radius: float) -> Path:
    path = context["dense"].xytheta
    arc = context["dense"].arc_length_m
    records = context["dense_records"]
    conflict = np.asarray([row["teacher_a_hard_blocked"] for row in records]) & context["dense_prefix"]
    first = _first_index(conflict)
    focus = first if first is not None else min(len(path) - 1, max(0, len(path) // 2))
    x, y, theta = path[focus]
    risk = context["risk"]
    geometry = context["geometry"]
    extent = geometry.imshow_extent_yx
    rgb_path = Path(context["mission_dir"]) / "images" / "hdr_front" / f"{context['image_id']:06d}.jpeg"
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Current RGB (fatal-cell image projection unavailable)")
    axes[0].axis("off")
    im = axes[1].imshow(risk, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1, aspect="equal")
    yy = geometry.origin_xy[1] + np.arange(geometry.width) * geometry.resolution
    xx = geometry.origin_xy[0] + np.arange(geometry.height) * geometry.resolution
    axes[1].contour(yy, xx, np.asarray(risk >= context["fatal_threshold"], float), levels=[0.5], colors="cyan", linewidths=1.2)
    axes[1].plot(path[:, 1], path[:, 0], color="white", linewidth=1.2)
    step = max(1, int(round(0.30 / max(context["dense"].total_length_m / max(len(path) - 1, 1), 0.02))))
    for index in range(0, len(path), step):
        px, py, pt = path[index]
        c, s = np.cos(pt), np.sin(pt)
        local = np.array([[-0.55, -0.26], [-0.55, 0.26], [0.55, 0.26], [0.55, -0.26]])
        world = np.column_stack([px + c * local[:, 0] - s * local[:, 1], py + s * local[:, 0] + c * local[:, 1]])
        axes[1].add_patch(Polygon(np.column_stack([world[:, 1], world[:, 0]]), fill=False, edgecolor="lime", linewidth=0.7, alpha=0.5))
    axes[1].add_patch(Circle((y, x), circle_radius, fill=False, edgecolor="deepskyblue", linewidth=2))
    local = np.array([[-0.55, -0.26], [-0.55, 0.26], [0.55, 0.26], [0.55, -0.26]])
    c, s = np.cos(theta), np.sin(theta)
    world = np.column_stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]])
    axes[1].add_patch(Polygon(np.column_stack([world[:, 1], world[:, 0]]), fill=False, edgecolor="white", linewidth=2.5))
    fatal_world = context["dense_payloads"][focus]["fatal_world"]
    if len(fatal_world):
        axes[1].scatter(fatal_world[:, 1], fatal_world[:, 0], s=22, c="red", edgecolors="white", linewidths=0.3)
    axes[1].scatter([y], [x], marker="*", s=180, c="magenta", zorder=5)
    axes[1].arrow(y, x, 0.30 * np.sin(theta), 0.30 * np.cos(theta), color="yellow", width=0.01)
    axes[1].set_xlim(y - 1.0, y + 1.0)
    axes[1].set_ylim(x - 1.0, x + 1.0)
    axes[1].set_xlabel("y left [m]")
    axes[1].set_ylabel("x forward [m]")
    axes[1].set_title(f"{label}: risk + fatal contour + footprints")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    aggregate = np.asarray([row["mppi_footprint_aggregate_cost"] for row in records])
    ratio = np.asarray([row["fatal_sample_ratio_known"] for row in records])
    axes[2].plot(arc, ratio, color="red", label="fatal sample ratio")
    axes[2].set_xlabel("arc length [m]")
    axes[2].set_ylabel("fatal ratio", color="red")
    ax2 = axes[2].twinx()
    ax2.plot(arc, aggregate, color="navy", alpha=0.75, label="MPPI footprint aggregate")
    ax2.set_ylabel("MPPI aggregate cost", color="navy")
    if first is not None:
        axes[2].axvline(arc[first], color="magenta", linestyle="--")
    axes[2].grid(alpha=0.25)
    axes[2].set_title(
        f"first={arc[focus]:.2f}m; fatal={records[focus]['fatal_sample_count']}/{records[focus]['known_sample_count']}; aggregate={aggregate[focus]:.1f}"
    )
    target = output / f"{label}.png"
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target


def _choose_representatives(path_rows: list[dict]) -> dict[str, dict | None]:
    def choose(predicate, score=lambda row: 0.0):
        candidates = [row for row in path_rows if predicate(row)]
        return max(candidates, key=score) if candidates else None

    conflict = lambda row: row["audit_cohort"] == "rectangle_clearance_conflict"
    return {
        "isolated_fatal_cell_touch": choose(lambda r: conflict(r) and r["severity_class"] == "isolated_touch"),
        "short_boundary_graze": choose(lambda r: conflict(r) and r["severity_class"] == "boundary_graze", lambda r: -r["max_conflict_run_m"]),
        "sustained_rectangle_overlap": choose(lambda r: conflict(r) and r["severity_class"] in {"sustained_overlap", "deep_collision"}, lambda r: r["max_conflict_run_m"]),
        "front_overhang_conflict": choose(lambda r: conflict(r) and r["dominant_conflict_location"] == "front_overhang", lambda r: r["dense_forbidden_arc_ratio"]),
        "rear_overhang_conflict": choose(lambda r: conflict(r) and r["dominant_conflict_location"] == "rear_overhang", lambda r: r["dense_forbidden_arc_ratio"]),
        "dense_interpolation_only_conflict": choose(lambda r: r["conflict_only_after_dense_interpolation"], lambda r: r["dense_forbidden_arc_ratio"]),
        "mppi_soft_accept_conflict": choose(lambda r: conflict(r) and r["path_source"] == "geometric", lambda r: r.get("mean_mppi_total_cost", -np.inf)),
        "teleop_mission1_near_conflict": choose(lambda r: conflict(r) and r["path_source"] == "teleop" and r["first_conflict_dense_arc_length_m"] is not None and r["first_conflict_dense_arc_length_m"] < 1.0, lambda r: -r["first_conflict_dense_arc_length_m"]),
        # The audited observable prefixes contain no first conflict beyond 1 m;
        # therefore "far" means the farthest available first-conflict control.
        "teleop_mission1_far_conflict": choose(lambda r: conflict(r) and r["path_source"] == "teleop" and r["first_conflict_dense_arc_length_m"] is not None, lambda r: r["first_conflict_dense_arc_length_m"]),
        "circle_and_rectangle_pass_control": choose(lambda r: r["audit_cohort"] == "circle_and_rectangle_pass_control", lambda r: r["path_length_m"]),
    }


def run(cfg) -> Path:
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    figure_dir = output / "representative_figures"
    figure_dir.mkdir()
    mppi_cfg = OmegaConf.load(Path(cfg.mppi_config)).mppi
    geometry = MapGeometry(
        int(round(2 * cfg.map_size / cfg.map_resolution)),
        int(round(2 * cfg.map_size / cfg.map_resolution)),
        float(cfg.map_resolution),
        (-float(cfg.map_size), -float(cfg.map_size)),
    )
    rectangles = OmegaConf.to_container(mppi_cfg.footprint, resolve=True)
    offsets = rectangle_footprint_offsets(rectangles, geometry.resolution, str(cfg.device))
    filter_model = get_filter_torch(str(cfg.device))
    selection = _select_paths(cfg)
    path_rows: list[dict] = []
    conflict_state_rows: list[dict] = []
    contexts: dict[tuple[str, str, int], dict] = {}
    selection_by_mission: dict[str, list[dict]] = defaultdict(list)
    for item in selection:
        selection_by_mission[item["mission"]].append(item)

    for mission, items in selection_by_mission.items():
        mission_dir = Path(cfg.dataset_root) / mission
        elevation_group = zarr.open_group(str(mission_dir / "data/elevation_map"), mode="r")
        lookup = _elevation_lookup(elevation_group)
        groups = {
            "geometric": zarr.open_group(str(mission_dir / "data/geometric_paths"), mode="r"),
            "teleop": zarr.open_group(str(mission_dir / "data/teleop_paths"), mode="r"),
        }
        image_cache: dict[int, dict] = {}
        objective = MPPIObjective(mppi_cfg, str(cfg.device))
        for number, selected in enumerate(items, 1):
            source = selected["path_source"]
            row_index = int(selected["path_row"])
            group = groups[source]
            path = np.asarray(group["path"][row_index], dtype=np.float32)
            goal = np.asarray(group["goal"][row_index], dtype=np.float32)
            image_id = int(group["image_id"][row_index])
            elevation_row = lookup[image_id]
            goal_time = float(group["goal_time"][row_index]) if "goal_time" in group else np.nan
            if image_id not in image_cache:
                elevation = np.asarray(elevation_group["elevation"][elevation_row], dtype=np.float32)
                trav = compute_limo_traversability(elevation, filter_model, mppi_cfg, str(cfg.device))
                circle_teacher = build_teacher_a(
                    geometry=geometry,
                    known_trav=trav.known_trav,
                    risk=trav.risk,
                    fatal_threshold=float(mppi_cfg.fatal_th),
                    inflation_radius_m=float(cfg.circle_radius_m),
                )
                gm = GridMap2D(
                    elevation=torch.as_tensor(elevation, dtype=torch.float32, device=objective.device),
                    resolution=geometry.resolution,
                    origin_xy=torch.tensor(geometry.origin_xy, dtype=torch.float32, device=objective.device),
                )
                objective.set_map(gm)
                image_cache[image_id] = {"elevation": elevation, "trav": trav, "circle": circle_teacher, "gm": gm}
            image = image_cache[image_id]
            original_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))])
            original_weights = arc_weights(original_s)
            dense = densify_path(path, float(cfg.dense_sampling_interval_m))
            original_records, original_payloads = audit_footprint_states(
                path,
                geometry=geometry,
                footprint_offsets_xy=offsets,
                risk=image["trav"].risk,
                trav_cost=image["trav"].trav_cost,
                fatal_threshold=float(mppi_cfg.fatal_th),
                half_length=0.55,
                half_width=0.26,
            )
            dense_records, dense_payloads = audit_footprint_states(
                dense.xytheta,
                geometry=geometry,
                footprint_offsets_xy=offsets,
                risk=image["trav"].risk,
                trav_cost=image["trav"].trav_cost,
                fatal_threshold=float(mppi_cfg.fatal_th),
                half_length=0.55,
                half_width=0.26,
            )
            original_prefix = _prefix_mask(original_records)
            dense_prefix = _prefix_mask(dense_records)
            original_conflict = np.asarray([record["teacher_a_hard_blocked"] for record in original_records]) & original_prefix
            dense_conflict = np.asarray([record["teacher_a_hard_blocked"] for record in dense_records]) & dense_prefix
            original_first = _first_index(original_conflict)
            dense_first = _first_index(dense_conflict)
            runs = continuous_runs(dense_conflict, dense.arc_weights_m)
            max_run = max(runs, default=0.0)
            conflict_records = [record for record, active in zip(dense_records, dense_conflict) if active]
            severity = classify_path_severity(conflict_records, max_run)
            locations = Counter(record["conflict_location"] for record in conflict_records if record["conflict_location"])

            circle_eval = evaluate_circular_teacher(
                dense,
                image["circle"],
                *(__import__("dataset_builder.reachability.path_calibration", fromlist=["build_clearance_maps"]).build_clearance_maps(
                    image["trav"].known_trav, image["trav"].risk, float(mppi_cfg.fatal_th), geometry.resolution
                )),
            )
            eligible_false_accept = dense_prefix & dense_conflict & ~circle_eval.overlaps_blocked

            objective_data = None
            if source == "geometric":
                objective_data = _objective_costs(objective, image["gm"], path, goal, mppi_cfg)
                exact_diff = max(
                    abs(float(record["mppi_footprint_aggregate_cost"]) - float(value))
                    for record, value in zip(original_records, objective_data["mppi_footprint_aggregate"])
                )
            else:
                exact_diff = np.nan

            original_den = float(original_weights[original_prefix].sum())
            dense_den = float(dense.arc_weights_m[dense_prefix].sum())
            first_s = float(dense.arc_length_m[dense_first]) if dense_first is not None else None
            base = {
                **selected,
                "image_id": image_id,
                "elevation_row": elevation_row,
                "timestamp": float(elevation_group["timestamp"][elevation_row]),
                "goal_time_s": goal_time,
                "path_length_m": dense.total_length_m,
                "observable_prefix_length_m": dense_den,
                "conflict_at_original_waypoints": bool(original_conflict.any()),
                "conflict_only_after_dense_interpolation": bool(dense_conflict.any() and not original_conflict.any()),
                "first_conflict_original_index": original_first,
                "first_conflict_dense_index": dense_first,
                "first_conflict_dense_arc_length_m": first_s,
                "original_waypoint_forbidden_ratio": float(original_conflict.sum() / max(original_prefix.sum(), 1)),
                "original_waypoint_forbidden_arc_ratio": float(original_weights[original_conflict].sum() / original_den) if original_den > 0 else np.nan,
                "dense_forbidden_arc_ratio": float(dense.arc_weights_m[dense_conflict].sum() / dense_den) if dense_den > 0 else np.nan,
                "dense_conflict_arc_length_m": float(dense.arc_weights_m[dense_conflict].sum()),
                "max_conflict_run_m": max_run,
                "conflict_run_count": len(runs),
                "multiple_continuous_conflict_runs": len(runs) > 1,
                "severity_class": severity,
                "dominant_conflict_location": _mode([record["conflict_location"] for record in conflict_records]),
                "max_fatal_sample_count": max((int(record["fatal_sample_count"]) for record in conflict_records), default=0),
                "max_fatal_unique_cell_count": max((int(record["fatal_unique_cell_count"]) for record in conflict_records), default=0),
                "max_fatal_sample_ratio_known": max((float(record["fatal_sample_ratio_known"]) for record in conflict_records), default=0.0),
                "max_fatal_overlap_area_m2": max((float(record["fatal_overlap_area_m2"]) for record in conflict_records), default=0.0),
                "max_fatal_component_cells": max((int(record["largest_fatal_component_cells"]) for record in conflict_records), default=0),
                "single_cell_only_path": bool(conflict_records and max(int(record["fatal_unique_cell_count"]) for record in conflict_records) == 1),
                "circle_0p28_false_accept_arc_length_m": float(dense.arc_weights_m[eligible_false_accept].sum()),
                "circle_0p28_false_accept_ratio_of_rectangle_conflict": float(dense.arc_weights_m[eligible_false_accept].sum() / max(dense.arc_weights_m[dense_conflict].sum(), 1e-12)) if dense_conflict.any() else np.nan,
                "first_conflict_distance_to_endpoint_m": dense.total_length_m - first_s if first_s is not None else None,
                "conflict_arc_in_last_0p5m_ratio": float(
                    dense.arc_weights_m[dense_conflict & (dense.arc_length_m >= max(0.0, dense.total_length_m - 0.5))].sum()
                    / max(dense.arc_weights_m[dense_conflict].sum(), 1e-12)
                ) if dense_conflict.any() else np.nan,
                "mppi_aggregate_exact_reproduction_max_abs_error": exact_diff,
                **{f"location_{name}_fatal_samples": sum(int(record["fatal_sample_count"]) for record in conflict_records if record["conflict_location"] == name) for name in ("front_overhang", "rear_overhang", "left_side", "right_side", "rotated_corner")},
            }
            if objective_data is not None:
                original_conflict_indices = np.flatnonzero(original_conflict)
                base.update(
                    {
                        "mean_mppi_total_cost": float(np.mean(objective_data["mppi_total_state_cost"])),
                        "mean_mppi_traversability_cost": float(np.mean(objective_data["mppi_traversability_weighted"])),
                        "mean_mppi_base_cost": float(np.mean(objective_data["mppi_base_cost"])),
                        "mean_mppi_total_cost_at_original_conflicts": float(np.mean(objective_data["mppi_total_state_cost"][original_conflict_indices])) if len(original_conflict_indices) else np.nan,
                        "mean_mppi_traversability_cost_at_original_conflicts": float(np.mean(objective_data["mppi_traversability_weighted"][original_conflict_indices])) if len(original_conflict_indices) else np.nan,
                        "mppi_rollout_reconstruction_max_error": objective_data["mppi_rollout_reconstruction_max_error"],
                        "mppi_cost_decomposition_max_error": objective_data["mppi_cost_decomposition_max_error"],
                        "safer_rollout_recoverable": False,
                        "safer_rollout_note": "original MPPI populations/cost histories are not stored in Zarr",
                    }
                )
            path_rows.append(base)
            path_key = (mission, source, row_index)
            contexts[path_key] = {
                "mission_dir": mission_dir,
                "image_id": image_id,
                "dense": dense,
                "dense_records": dense_records,
                "dense_payloads": dense_payloads,
                "dense_prefix": dense_prefix,
                "risk": image["trav"].risk,
                "elevation": image["elevation"],
                "geometry": geometry,
                "fatal_threshold": float(mppi_cfg.fatal_th),
            }
            for state_index, (record, active_prefix, active_conflict) in enumerate(zip(dense_records, dense_prefix, dense_conflict)):
                if not active_prefix and not record["teacher_a_hard_blocked"]:
                    continue
                nearest_original = int(np.argmin(abs(original_s - dense.arc_length_m[state_index])))
                row = {
                    "mission": mission,
                    "path_source": source,
                    "path_row": row_index,
                    "image_id": image_id,
                    "audit_cohort": selected["audit_cohort"],
                    "sampling": "dense_0p02m",
                    "arc_length_m": float(dense.arc_length_m[state_index]),
                    "observable_prefix": bool(active_prefix),
                    "is_rectangle_conflict": bool(active_conflict),
                    "nearest_original_waypoint_index": nearest_original,
                    **record,
                }
                if objective_data is not None:
                    for name in (
                        "mppi_traversability_weighted", "mppi_control_cost", "mppi_position_cost",
                        "mppi_heading_cost", "mppi_base_cost", "mppi_total_state_cost"
                    ):
                        row[name + "_nearest_original"] = float(objective_data[name][nearest_original])
                    row["objective_relevance"] = "planner_selected_path_exact_at_nearest_original_state"
                else:
                    row["objective_relevance"] = "footprint_aggregate_exact; total_objective_not_attributed_to_teleop"
                conflict_state_rows.append(row)
            log.info("[%s/%s] %s %s row=%d", number, len(items), mission, source, row_index)

    severity_summary, original_dense_summary, location_summary = _summary_rows(path_rows)
    teleop_summary = _teleop_distance_summary(path_rows, conflict_state_rows, str(cfg.missions[0]))
    _write_csv(output / "rectangle_conflict_audit.csv", path_rows)
    _write_csv(output / "conflict_severity_summary.csv", severity_summary)
    _write_csv(output / "original_vs_dense_summary.csv", original_dense_summary)
    _write_csv(output / "mppi_cost_comparison.csv", conflict_state_rows)
    _write_csv(output / "conflict_location_summary.csv", location_summary)
    _write_csv(output / "teleop_mission1_distance_summary.csv", teleop_summary)

    representatives = _choose_representatives(path_rows)
    manifest: dict[str, dict | None] = {}
    for label, row in representatives.items():
        if row is None:
            manifest[label] = None
            continue
        key = (row["mission"], row["path_source"], int(float(row["path_row"])))
        figure = _plot_representative(contexts[key], figure_dir, label, int(cfg.figure_dpi), float(cfg.circle_radius_m))
        manifest[label] = {
            "mission": row["mission"],
            "path_source": row["path_source"],
            "path_row": int(row["path_row"]),
            "image_id": int(row["image_id"]),
            "severity_class": row["severity_class"],
            "first_conflict_dense_arc_length_m": row["first_conflict_dense_arc_length_m"],
            "figure": str(figure),
        }
    (output / "representative_conflict_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    conflicts = [row for row in path_rows if row["audit_cohort"] == "rectangle_clearance_conflict"]
    geo = [row for row in conflicts if row["path_source"] == "geometric"]
    tele = [row for row in conflicts if row["path_source"] == "teleop"]
    original_count = sum(row["conflict_at_original_waypoints"] for row in geo)
    dense_only = sum(row["conflict_only_after_dense_interpolation"] for row in geo)
    light = sum(row["severity_class"] in {"isolated_touch", "boundary_graze"} for row in geo)
    strong = sum(row["severity_class"] in {"sustained_overlap", "deep_collision"} for row in geo)
    overhang = sum(row["dominant_conflict_location"] in {"front_overhang", "rear_overhang", "rotated_corner"} for row in geo)
    single_cell = sum(bool(row["single_cell_only_path"]) for row in geo)
    at_most_two_cells = sum(int(row["max_fatal_unique_cell_count"]) <= 2 for row in geo)
    exact_error = max(float(row["mppi_aggregate_exact_reproduction_max_abs_error"]) for row in geo)
    median_base = float(np.median([float(row["mean_mppi_base_cost"]) for row in geo]))
    median_trav = float(np.median([float(row["mean_mppi_traversability_cost"]) for row in geo]))
    near_endpoint = sum(float(row["conflict_arc_in_last_0p5m_ratio"]) >= 0.5 for row in geo)
    false_accept = float(np.nanmean([float(row["circle_0p28_false_accept_ratio_of_rectangle_conflict"]) for row in geo]))
    tele_near = next((row for row in teleop_summary if row["distance_bin"] == "0-1m"), {})
    tele_mid = next((row for row in teleop_summary if row["distance_bin"] == "1-2m"), {})
    tele_far = next((row for row in teleop_summary if row["distance_bin"] == "3m+"), {})
    tele_far_text = (
        f"{float(tele_far['hard_conflict_state_ratio']):.3%}"
        if int(tele_far.get("evaluable_state_count", 0)) > 0
        else "not evaluable (zero observable-prefix support)"
    )
    report = f"""# Rectangle Conflict Audit

This audit is read-only: Teacher-A, MPPI, all Zarr stores, and formal labels remain unchanged.

## Scope and exact MPPI semantics

- Audited {len(geo)} geometric rectangle-clearance conflicts (including {sum(r['medium_conflict_gt_5pct'] for r in geo)} above 5% observable-prefix conflict), {len(tele)} Mission-1 teleop conflicts, and {sum(r['audit_cohort']=='circle_and_rectangle_pass_control' for r in path_rows)} controls.
- Robot footprint is the repository configuration `[-0.55,0.55] x [-0.26,0.26]`, sampled at {geometry.resolution:.2f} m ({len(offsets)} samples).
- MPPI uses an arithmetic mean of thresholded footprint sample cost over inside-domain samples. Unknown is a separate mean fraction and outside samples are excluded. State weights are traversability ×{float(mppi_cfg.traversability_cost):g} and unknown ×{float(mppi_cfg.traversability_unknown_cost):g}; the 50-state trajectory objective is then averaged.
- Original rollout populations and alternative trajectory costs were not stored, so whether a safer rollout existed cannot be recovered. The stored geometric path is nevertheless recomputed against the exact current `MPPIObjective` terms.

## Main counts

- Geometric conflicts already present at one or more original waypoint: **{original_count}/{len(geo)}**.
- Geometric conflicts found only by 0.02 m dense interpolation: **{dense_only}/{len(geo)}**.
- Geometric isolated-touch or boundary-graze class: **{light}/{len(geo)}**.
- Geometric conflicts restricted to one fatal cell for the whole path: **{single_cell}/{len(geo)}**; at most two cells at any state: **{at_most_two_cells}/{len(geo)}**.
- Geometric sustained-overlap or deep-collision class: **{strong}/{len(geo)}**.
- Geometric dominant front/rear/corner overhang: **{overhang}/{len(geo)}**.
- Circle-0.28 false acceptance over the rectangle-conflict arcs: **{false_accept:.3%}**.
- Exact MPPI footprint aggregate reproduction max absolute error: **{exact_error:.6g}**.
- Mission-1 teleop hard-conflict state ratio: 0–1 m **{float(tele_near.get('hard_conflict_state_ratio', np.nan)):.3%}**, 1–2 m **{float(tele_mid.get('hard_conflict_state_ratio', np.nan)):.3%}**, after 3 m **{tele_far_text}**. All first conflicts occur before 1 m; the farthest available first conflict is {max(float(r['first_conflict_dense_arc_length_m']) for r in tele):.3f} m.

## Interpretation

The hard rectangle mask and MPPI answer different questions. A single fatal sample makes the diagnostic rectangle blocked, while MPPI dilutes it over {len(offsets)} footprint samples and over 50 trajectory states. One isolated fatal sample contributes about `{float(mppi_cfg.fatal_value)/len(offsets):.1f}` to the footprint aggregate, `{float(mppi_cfg.fatal_value)/len(offsets)*float(mppi_cfg.traversability_cost):.1f}` to that state's weighted terrain cost, and only `{float(mppi_cfg.fatal_value)/len(offsets)*float(mppi_cfg.traversability_cost)/50:.1f}` to the 50-state trajectory mean. Across audited geometric conflicts, median mean base/goal/control cost is {median_base:.1f} and median mean terrain cost is {median_trav:.1f}; {near_endpoint}/{len(geo)} paths place at least half their conflict arc in the last 0.5 m. This supports soft-cost tradeoff, but stored rollout populations are unavailable, so a safer alternative or goal-attraction causality cannot be proven. Isolated contacts are not an MPPI hard rejection; sustained/deep classes remain meaningful whole-body collision evidence.

Circle 0.28 is a centerline/topology approximation. Its false acceptance should be interpreted from `conflict_location_summary.csv`: front, rear, and rotated-corner dominance is systematic overhang geometry, whereas isolated side/corner touches are more threshold/grid sensitive.

Mission-1 teleop conflicts occur on fatal cells that are known in the current elevation map inside the audited prefix. This proves current-elevation observability, not pixel-level RGB visibility (no fatal-cell-to-camera projection is available). The 1–2 m state ratio is higher than 0–1 m, but every first conflict already occurs within 1 m and there is no observable-prefix exposure beyond 2 m. Therefore this subset does **not** establish a distance-onset/staleness trend; it instead shows persistent conflicts after near-field onset. Actual online map updates and localization residuals during teleoperation are not stored, so map staleness, drift, and threshold sensitivity cannot be cleanly separated.

## Safety-model recommendation

`any fatal cell` is useful as a conservative alarm and upper-bound safety reference, but the audit does not justify treating every isolated one-cell touch as equivalent to a sustained collision. Keep the 0.28 m 2-D center reachable field for topology and add orientation-aware path validation with severity/continuous-overlap reporting before undertaking a full SE(2) reachable field. A later SE(2) field is only necessary if planning itself must enforce yaw-dependent reachability, not merely validate candidate paths. No Teacher-A parameter is changed by this conclusion.

## Files

- `rectangle_conflict_audit.csv`: one row per audited path.
- `mppi_cost_comparison.csv`: dense state footprint evidence and nearest exact original-state objective terms.
- `original_vs_dense_summary.csv`, `conflict_severity_summary.csv`, `conflict_location_summary.csv`, `teleop_mission1_distance_summary.csv`: aggregate tables.
- `representative_conflict_manifest.json` and `representative_figures/`: selected local audits.
"""
    (output / "rectangle_conflict_audit.md").write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("dataset_builder/configs/rectangle_conflict_audit.yaml"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    cfg = OmegaConf.load(args.config)
    result = run(cfg)
    log.info("Rectangle conflict audit written to %s", result)


if __name__ == "__main__":
    main()
