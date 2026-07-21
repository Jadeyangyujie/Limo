"""Read-only helpers for auditing yaw-aware rectangular footprint conflicts.

The functions in this module intentionally reproduce the current MPPI grid
sampling semantics (rotated footprint samples, floor quantisation, outside
samples excluded from the aggregate) without changing Teacher-A.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from dataset_builder.reachability.coordinates import MapGeometry


LOCATION_NAMES = ("front_overhang", "rear_overhang", "left_side", "right_side", "rotated_corner")


def largest_connected_component(indices: np.ndarray) -> int:
    cells = {tuple(map(int, item)) for item in np.asarray(indices)}
    best = 0
    while cells:
        seed = cells.pop()
        queue = deque([seed])
        size = 1
        while queue:
            i, j = queue.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    candidate = (i + di, j + dj)
                    if candidate in cells:
                        cells.remove(candidate)
                        queue.append(candidate)
                        size += 1
        best = max(best, size)
    return best


def classify_local_overlap(offsets_xy: np.ndarray, half_length: float, half_width: float) -> str | None:
    offsets = np.asarray(offsets_xy, dtype=np.float64)
    if len(offsets) == 0:
        return None
    counts: Counter[str] = Counter()
    for x, y in offsets:
        nx = abs(float(x)) / max(half_length, 1e-9)
        ny = abs(float(y)) / max(half_width, 1e-9)
        if nx >= 0.70 and ny >= 0.70:
            label = "rotated_corner"
        elif nx >= ny:
            label = "front_overhang" if x >= 0 else "rear_overhang"
        else:
            label = "left_side" if y >= 0 else "right_side"
        counts[label] += 1
    return max(LOCATION_NAMES, key=lambda name: (counts[name], -LOCATION_NAMES.index(name)))


def audit_footprint_states(
    states_xytheta: np.ndarray,
    *,
    geometry: MapGeometry,
    footprint_offsets_xy: np.ndarray,
    risk: np.ndarray,
    trav_cost: np.ndarray,
    fatal_threshold: float,
    half_length: float,
    half_width: float,
) -> tuple[list[dict], list[dict]]:
    """Return scalar state records and plotting payloads for each footprint."""
    records: list[dict] = []
    payloads: list[dict] = []
    # MPPI performs footprint rotation and floor quantisation in float32.
    # Keeping that precision here matters for samples exactly on cell edges.
    offsets = np.asarray(footprint_offsets_xy, dtype=np.float32)
    total = len(offsets)
    origin = np.asarray(geometry.origin_xy, dtype=np.float32)
    resolution = np.float32(geometry.resolution)
    for state_index, (x, y, theta) in enumerate(np.asarray(states_xytheta, dtype=np.float32)):
        c, s = np.float32(np.cos(theta)), np.float32(np.sin(theta))
        world = np.column_stack(
            [x + c * offsets[:, 0] - s * offsets[:, 1], y + s * offsets[:, 0] + c * offsets[:, 1]]
        ).astype(np.float32, copy=False)
        indices = np.floor((world - origin) / resolution).astype(np.int64)
        valid = geometry.valid_indices(indices)
        valid_indices = indices[valid]
        sampled_risk = np.full(total, np.nan, dtype=np.float64)
        sampled_cost = np.zeros(total, dtype=np.float64)
        if valid.any():
            sampled_risk[valid] = risk[valid_indices[:, 0], valid_indices[:, 1]]
            sampled_cost[valid] = np.nan_to_num(
                trav_cost[valid_indices[:, 0], valid_indices[:, 1]], nan=0.0
            )
        known = valid & np.isfinite(sampled_risk)
        unknown = valid & ~np.isfinite(sampled_risk)
        fatal = known & (sampled_risk >= fatal_threshold)
        unique_fatal_indices = np.unique(indices[fatal], axis=0) if fatal.any() else np.empty((0, 2), dtype=np.int64)
        known_values = sampled_risk[known]
        valid_count = int(valid.sum())
        known_count = int(known.sum())
        fatal_count = int(fatal.sum())
        local_fatal = offsets[fatal]
        aggregate = float(sampled_cost[valid].sum() / max(valid_count, 1))
        unknown_fraction = float(unknown.sum() / max(valid_count, 1))
        records.append(
            {
                "state_index": state_index,
                "x": float(x),
                "y": float(y),
                "theta": float(theta),
                "footprint_sample_count": total,
                "inside_sample_count": valid_count,
                "outside_sample_count": int((~valid).sum()),
                "known_sample_count": known_count,
                "unknown_sample_count": int(unknown.sum()),
                "fatal_sample_count": fatal_count,
                "fatal_sample_ratio_known": fatal_count / max(known_count, 1),
                "fatal_sample_ratio_total": fatal_count / max(total, 1),
                "fatal_unique_cell_count": int(len(unique_fatal_indices)),
                "fatal_overlap_area_m2": float(len(unique_fatal_indices) * geometry.resolution**2),
                "largest_fatal_component_cells": largest_connected_component(unique_fatal_indices),
                "isolated_single_cell_touch": bool(len(unique_fatal_indices) == 1),
                "maximum_risk": float(np.max(known_values)) if len(known_values) else np.nan,
                "mean_risk": float(np.mean(known_values)) if len(known_values) else np.nan,
                "p90_risk": float(np.percentile(known_values, 90)) if len(known_values) else np.nan,
                "p95_risk": float(np.percentile(known_values, 95)) if len(known_values) else np.nan,
                "mppi_footprint_aggregate_cost": aggregate,
                "mppi_unknown_fraction": unknown_fraction,
                "teacher_a_hard_blocked": bool(fatal_count > 0),
                "overlaps_unknown": bool(unknown.any()),
                "crosses_domain": bool((~valid).any()),
                "center_to_nearest_fatal_m": float(np.linalg.norm(local_fatal, axis=1).min())
                if len(local_fatal)
                else np.nan,
                "conflict_location": classify_local_overlap(local_fatal, half_length, half_width),
            }
        )
        payloads.append(
            {
                "footprint_world": world,
                "fatal_world": world[fatal],
                "fatal_local": local_fatal,
                "footprint_indices": indices,
            }
        )
    return records, payloads


def arc_weights(arc_length_m: np.ndarray) -> np.ndarray:
    s = np.asarray(arc_length_m, dtype=np.float64)
    output = np.zeros_like(s)
    if len(s) > 1:
        ds = np.diff(s)
        output[:-1] += 0.5 * ds
        output[1:] += 0.5 * ds
    return output


def continuous_runs(mask: np.ndarray, weights: np.ndarray) -> list[float]:
    runs: list[float] = []
    current = 0.0
    for active, weight in zip(np.asarray(mask, dtype=bool), np.asarray(weights, dtype=float)):
        if active:
            current += float(weight)
        elif current > 0:
            runs.append(current)
            current = 0.0
    if current > 0:
        runs.append(current)
    return runs


def classify_path_severity(conflict_records: list[dict], max_run_m: float) -> str:
    if not conflict_records:
        return "no_conflict"
    max_cells = max(int(row["fatal_unique_cell_count"]) for row in conflict_records)
    max_component = max(int(row["largest_fatal_component_cells"]) for row in conflict_records)
    max_ratio = max(float(row["fatal_sample_ratio_known"]) for row in conflict_records)
    if max_ratio >= 0.10 or max_component >= 24:
        return "deep_collision"
    if max_run_m > 0.20 and (max_cells >= 3 or max_ratio >= 0.02 or max_component >= 6):
        return "sustained_overlap"
    if max_cells <= 2 and max_component <= 2 and max_run_m <= 0.06:
        return "isolated_touch"
    return "boundary_graze"


def reconstruct_actions(path_states: np.ndarray, dt: float) -> np.ndarray:
    """Invert MPPI's pre-step-yaw unicycle integration for a stored path."""
    states = np.asarray(path_states, dtype=np.float64)
    previous = np.vstack([np.zeros((1, 3), dtype=np.float64), states[:-1]])
    delta_world = states[:, :2] - previous[:, :2]
    yaw = previous[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = c * delta_world[:, 0] + s * delta_world[:, 1]
    local_y = -s * delta_world[:, 0] + c * delta_world[:, 1]
    delta_yaw = np.arctan2(np.sin(states[:, 2] - previous[:, 2]), np.cos(states[:, 2] - previous[:, 2]))
    return np.column_stack([local_x / dt, local_y / dt, delta_yaw / dt]).astype(np.float32)
