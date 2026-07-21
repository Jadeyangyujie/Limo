from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy import ndimage

from dataset_builder.mppi_planner.mppi_planner import build_robot_footprint
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import (
    ReachabilityState,
    STATE_NAMES,
    TeacherAResult,
    circular_structure,
    strict_dijkstra,
)


@dataclass
class DensePath:
    xytheta: np.ndarray
    arc_length_m: np.ndarray
    arc_weights_m: np.ndarray
    total_length_m: float


@dataclass
class PathEvaluation:
    model: str
    display_state: np.ndarray
    reachable: np.ndarray
    configuration_free: np.ndarray
    locally_blocked: np.ndarray
    clearance_blocked: np.ndarray
    unknown_center: np.ndarray
    unknown_footprint: np.ndarray
    outside_domain: np.ndarray
    disconnected: np.ndarray
    invalid_coordinate: np.ndarray
    overlaps_blocked: np.ndarray
    overlaps_unknown: np.ndarray
    crosses_domain: np.ndarray
    forbidden: np.ndarray
    min_obstacle_clearance_m: float
    min_unknown_clearance_m: float


@dataclass
class RootAnchoringResult:
    raw_valid: bool
    anchored_valid: bool
    anchor_applied: bool
    anchor_reason: Optional[str]
    reachable: np.ndarray
    reachable_area_ratio: float
    seeded_nonfree_count: int
    traversed_nonfree_count_excluding_seed: int
    crosses_unknown_or_forbidden: bool


def densify_path(path_xytheta: np.ndarray, interval_m: float) -> DensePath:
    path = np.asarray(path_xytheta, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) == 0:
        raise ValueError("path_xytheta must have shape (N, 3), N >= 1")
    if interval_m <= 0:
        raise ValueError("interval_m must be positive")
    delta = np.diff(path[:, :2], axis=0)
    segment_lengths = np.linalg.norm(delta, axis=1)
    source_s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(source_s[-1])
    if total == 0.0:
        dense = path[[0]].copy()
        return DensePath(dense, np.array([0.0]), np.array([0.0]), 0.0)

    count = int(np.ceil(total / interval_m)) + 1
    dense_s = np.linspace(0.0, total, count, dtype=np.float64)
    keep = np.concatenate([[True], np.diff(source_s) > 1e-12])
    unique_s = source_s[keep]
    unique_path = path[keep]
    theta_unwrapped = np.unwrap(unique_path[:, 2])
    dense = np.column_stack(
        [
            np.interp(dense_s, unique_s, unique_path[:, 0]),
            np.interp(dense_s, unique_s, unique_path[:, 1]),
            np.interp(dense_s, unique_s, theta_unwrapped),
        ]
    )
    dense[:, 2] = np.arctan2(np.sin(dense[:, 2]), np.cos(dense[:, 2]))
    ds = np.diff(dense_s)
    weights = np.zeros_like(dense_s)
    weights[:-1] += 0.5 * ds
    weights[1:] += 0.5 * ds
    return DensePath(dense.astype(np.float32), dense_s, weights, total)


def weighted_ratio(mask: np.ndarray, dense: DensePath) -> float:
    values = np.asarray(mask, dtype=bool)
    if dense.total_length_m <= 1e-12:
        return float(values[0])
    return float(np.sum(dense.arc_weights_m[values]) / dense.total_length_m)


def weighted_length(mask: np.ndarray, dense: DensePath) -> float:
    values = np.asarray(mask, dtype=bool)
    return float(np.sum(dense.arc_weights_m[values]))


def conditional_weighted_ratio(
    numerator: np.ndarray, denominator: np.ndarray, dense: DensePath
) -> float:
    support = np.asarray(denominator, dtype=bool)
    support_length = weighted_length(support, dense)
    if support_length <= 1e-12:
        return float("nan")
    values = np.asarray(numerator, dtype=bool) & support
    return weighted_length(values, dense) / support_length


def max_consecutive_length(mask: np.ndarray, dense: DensePath) -> float:
    values = np.asarray(mask, dtype=bool)
    best = current = 0.0
    for active, weight in zip(values, dense.arc_weights_m):
        if active:
            current += float(weight)
            best = max(best, current)
        else:
            current = 0.0
    return best


def first_true_index(mask: np.ndarray) -> Optional[int]:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def first_true_arc_length(mask: np.ndarray, dense: DensePath) -> Optional[float]:
    index = first_true_index(mask)
    return float(dense.arc_length_m[index]) if index is not None else None


def prefix_before(mask: np.ndarray, dense: DensePath) -> np.ndarray:
    """Return the continuous prefix strictly before the first true sample."""
    first = first_true_index(mask)
    output = np.ones(len(dense.xytheta), dtype=bool)
    if first is not None:
        output[first:] = False
    return output


def scoped_evaluation_metrics(
    evaluation: PathEvaluation,
    dense: DensePath,
    scope_mask: np.ndarray,
    *,
    tolerance_ratio: float = 0.05,
) -> dict:
    scope = np.asarray(scope_mask, dtype=bool)
    length = weighted_length(scope, dense)
    evaluable = length > 1e-12

    def ratio(values: np.ndarray) -> float:
        return conditional_weighted_ratio(values, scope, dense)

    forbidden_ratio = ratio(evaluation.forbidden) if evaluable else float("nan")
    collision_ratio = ratio(evaluation.overlaps_blocked) if evaluable else float("nan")
    disconnected_ratio = ratio(evaluation.disconnected) if evaluable else float("nan")
    return {
        "evaluable": evaluable,
        "scope_arc_length_m": length,
        "forbidden_ratio": forbidden_ratio,
        "collision_ratio": collision_ratio,
        "locally_blocked_ratio": ratio(evaluation.locally_blocked)
        if evaluable
        else float("nan"),
        "clearance_ratio": ratio(evaluation.clearance_blocked)
        if evaluable
        else float("nan"),
        "unknown_ratio": ratio(evaluation.overlaps_unknown)
        if evaluable
        else float("nan"),
        "outside_ratio": ratio(evaluation.crosses_domain)
        if evaluable
        else float("nan"),
        "disconnected_ratio": disconnected_ratio,
        "strict_pass": bool(evaluable and forbidden_ratio <= 1e-12),
        "tolerant_pass": bool(
            evaluable and forbidden_ratio <= tolerance_ratio + 1e-12
        ),
        "collision_strict_pass": bool(evaluable and collision_ratio <= 1e-12),
        "collision_tolerant_pass": bool(
            evaluable and collision_ratio <= tolerance_ratio + 1e-12
        ),
    }


def conditional_footprint_metrics(
    circle: PathEvaluation, rectangle: PathEvaluation, dense: DensePath
) -> dict:
    """Compare footprint collision geometry only on mutually observable support."""
    eligible = ~(
        circle.crosses_domain
        | rectangle.crosses_domain
        | circle.invalid_coordinate
        | rectangle.invalid_coordinate
        | circle.overlaps_unknown
        | rectangle.overlaps_unknown
    )
    circle_blocked = circle.overlaps_blocked
    rectangle_blocked = rectangle.overlaps_blocked
    rectangle_free = ~rectangle_blocked
    support_length = weighted_length(eligible, dense)
    rectangle_free_length = weighted_length(eligible & rectangle_free, dense)
    rectangle_blocked_length = weighted_length(eligible & rectangle_blocked, dense)
    disagreement = circle_blocked != rectangle_blocked
    return {
        "conditional_support_arc_length_m": support_length,
        "conditional_rectangle_free_arc_length_m": rectangle_free_length,
        "conditional_rectangle_blocked_arc_length_m": rectangle_blocked_length,
        "conditional_circle_rectangle_agreement": conditional_weighted_ratio(
            ~disagreement, eligible, dense
        ),
        "circle_false_reject_ratio_given_rectangle_free": conditional_weighted_ratio(
            circle_blocked, eligible & rectangle_free, dense
        ),
        "circle_false_accept_ratio_given_rectangle_blocked": conditional_weighted_ratio(
            ~circle_blocked, eligible & rectangle_blocked, dense
        ),
        # A clearance conflict is a footprint-level blocked/free disagreement;
        # local center-blocked evidence is retained separately in path metrics.
        "conditional_clearance_conflict_ratio": conditional_weighted_ratio(
            disagreement, eligible, dense
        ),
        "conditional_agreement_arc_length_m": weighted_length(
            eligible & ~disagreement, dense
        ),
        "circle_false_reject_arc_length_m": weighted_length(
            eligible & rectangle_free & circle_blocked, dense
        ),
        "circle_false_accept_arc_length_m": weighted_length(
            eligible & rectangle_blocked & ~circle_blocked, dense
        ),
        "conditional_clearance_conflict_arc_length_m": weighted_length(
            eligible & disagreement, dense
        ),
    }


def diagnostic_root_anchor(
    result: TeacherAResult,
    *,
    rectangle_overlaps_blocked: bool,
    rectangle_overlaps_unknown: bool,
    rectangle_crosses_domain: bool,
) -> RootAnchoringResult:
    """Diagnostic seed-only root anchoring without changing Teacher-A masks."""
    raw_valid = bool(result.root.configuration_valid)
    if raw_valid:
        return RootAnchoringResult(
            raw_valid=True,
            anchored_valid=True,
            anchor_applied=False,
            anchor_reason="raw_valid",
            reachable=result.reachable.copy(),
            reachable_area_ratio=float(result.reachable.mean()),
            seeded_nonfree_count=0,
            traversed_nonfree_count_excluding_seed=0,
            crosses_unknown_or_forbidden=False,
        )

    failure = result.root.failure_reason
    unknown_failure = failure in {"unknown_center", "unknown_footprint"}
    eligible = bool(
        unknown_failure
        and rectangle_overlaps_unknown
        and not rectangle_overlaps_blocked
        and not rectangle_crosses_domain
    )
    if not eligible:
        return RootAnchoringResult(
            raw_valid=False,
            anchored_valid=False,
            anchor_applied=False,
            anchor_reason=f"ineligible:{failure or 'unknown'}",
            reachable=result.reachable.copy(),
            reachable_area_ratio=float(result.reachable.mean()),
            seeded_nonfree_count=0,
            traversed_nonfree_count_excluding_seed=0,
            crosses_unknown_or_forbidden=False,
        )

    seed_free = result.configuration_free.copy()
    ri, rj = result.root.index
    seed_free[ri, rj] = True
    geodesic, _ = strict_dijkstra(seed_free, result.root.index, result.geometry.resolution)
    visited = np.isfinite(geodesic)
    nonfree_visited = visited & ~result.configuration_free
    seed_mask = np.zeros_like(seed_free)
    seed_mask[ri, rj] = True
    traversed_nonfree = int(np.count_nonzero(nonfree_visited & ~seed_mask))
    reachable = result.configuration_free & visited
    return RootAnchoringResult(
        raw_valid=False,
        anchored_valid=True,
        anchor_applied=True,
        anchor_reason=f"seed_only:{failure}",
        reachable=reachable,
        reachable_area_ratio=float(reachable.mean()),
        seeded_nonfree_count=int(np.count_nonzero(nonfree_visited & seed_mask)),
        traversed_nonfree_count_excluding_seed=traversed_nonfree,
        crosses_unknown_or_forbidden=traversed_nonfree > 0,
    )


def _clearance_map(forbidden: np.ndarray, resolution: float) -> np.ndarray:
    if not np.any(forbidden):
        return np.full(forbidden.shape, np.inf, dtype=np.float32)
    return ndimage.distance_transform_edt(~forbidden, sampling=resolution).astype(np.float32)


def _sample_clearance(
    clearance: np.ndarray, indices: np.ndarray, valid: np.ndarray
) -> float:
    if not valid.any():
        return float("nan")
    return float(np.min(clearance[indices[valid, 0], indices[valid, 1]]))


def evaluate_circular_teacher(
    dense: DensePath,
    result: TeacherAResult,
    obstacle_clearance: np.ndarray,
    unknown_clearance: np.ndarray,
) -> PathEvaluation:
    geometry = result.geometry
    indices = geometry.world_to_map_idx(dense.xytheta[:, :2])
    valid = geometry.valid_indices(indices)
    n = len(indices)
    state = np.full(n, -1, dtype=np.int16)
    state[valid] = result.state[indices[valid, 0], indices[valid, 1]]

    def is_state(value: ReachabilityState) -> np.ndarray:
        return valid & (state == int(value))

    reachable = is_state(ReachabilityState.REACHABLE)
    disconnected = is_state(ReachabilityState.TRAVERSABLE_BUT_DISCONNECTED)
    locally_blocked = is_state(ReachabilityState.LOCALLY_BLOCKED)
    clearance_blocked = is_state(ReachabilityState.CLEARANCE_BLOCKED)
    unknown_center = is_state(ReachabilityState.UNKNOWN_CENTER)
    unknown_footprint = is_state(ReachabilityState.UNKNOWN_FOOTPRINT)
    outside = is_state(ReachabilityState.OUTSIDE_DOMAIN)
    invalid = ~valid
    configuration_free = reachable | disconnected
    forbidden = ~reachable
    # Keep the independent raw overlap evidence as well as the mutually
    # exclusive Teacher-A display state.  In particular, Teacher-A gives
    # clearance priority when a footprint overlaps both blocked and unknown
    # cells, but that must not erase the unknown-overlap diagnostic.
    footprint, _ = circular_structure(
        result.inflation_radius_m, result.geometry.resolution
    )
    raw_blocked = obstacle_clearance <= 1e-12
    raw_blocked_overlap = ndimage.binary_dilation(
        raw_blocked, structure=footprint, border_value=0
    )
    raw_unknown_overlap = ndimage.binary_dilation(
        ~result.known_trav, structure=footprint, border_value=1
    )
    overlaps_blocked = np.zeros(n, dtype=bool)
    overlaps_unknown = np.zeros(n, dtype=bool)
    overlaps_blocked[valid] = raw_blocked_overlap[
        indices[valid, 0], indices[valid, 1]
    ]
    overlaps_unknown[valid] = raw_unknown_overlap[
        indices[valid, 0], indices[valid, 1]
    ]
    return PathEvaluation(
        model=f"circle_{result.inflation_radius_m:.2f}",
        display_state=state,
        reachable=reachable,
        configuration_free=configuration_free,
        locally_blocked=locally_blocked,
        clearance_blocked=clearance_blocked,
        unknown_center=unknown_center,
        unknown_footprint=unknown_footprint,
        outside_domain=outside,
        disconnected=disconnected,
        invalid_coordinate=invalid,
        overlaps_blocked=overlaps_blocked,
        overlaps_unknown=overlaps_unknown,
        crosses_domain=outside | invalid,
        forbidden=forbidden,
        min_obstacle_clearance_m=_sample_clearance(obstacle_clearance, indices, valid),
        min_unknown_clearance_m=_sample_clearance(unknown_clearance, indices, valid),
    )


def rectangle_footprint_offsets(
    rectangles, resolution: float, device: str = "cpu"
) -> np.ndarray:
    offsets = build_robot_footprint(rectangles, resolution, torch.device(device))
    return offsets.cpu().numpy().astype(np.float32)


def evaluate_yaw_aware_rectangle(
    dense: DensePath,
    *,
    geometry: MapGeometry,
    footprint_offsets_xy: np.ndarray,
    known_trav: np.ndarray,
    risk: np.ndarray,
    fatal_threshold: float,
    domain_support: np.ndarray,
    obstacle_clearance: np.ndarray,
    unknown_clearance: np.ndarray,
) -> PathEvaluation:
    n = len(dense.xytheta)
    centre_indices = geometry.world_to_map_idx(dense.xytheta[:, :2])
    centre_valid = geometry.valid_indices(centre_indices)
    invalid_coordinate = ~centre_valid
    locally_blocked = np.zeros(n, dtype=bool)
    unknown_center = np.zeros(n, dtype=bool)
    overlaps_blocked = np.zeros(n, dtype=bool)
    overlaps_unknown = np.zeros(n, dtype=bool)
    crosses_domain = np.zeros(n, dtype=bool)
    footprint_min_clearance = np.full(n, np.inf, dtype=np.float32)
    footprint_min_unknown_clearance = np.full(n, np.inf, dtype=np.float32)
    raw_blocked = known_trav & (risk >= fatal_threshold)

    for k, (x, y, theta) in enumerate(dense.xytheta):
        if centre_valid[k]:
            ci, cj = centre_indices[k]
            unknown_center[k] = not known_trav[ci, cj]
            locally_blocked[k] = bool(raw_blocked[ci, cj])
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        ox, oy = footprint_offsets_xy[:, 0], footprint_offsets_xy[:, 1]
        footprint_world = np.column_stack(
            [x + cos_t * ox - sin_t * oy, y + sin_t * ox + cos_t * oy]
        )
        footprint_indices = geometry.world_to_map_idx(footprint_world)
        footprint_valid = geometry.valid_indices(footprint_indices)
        if not footprint_valid.all():
            crosses_domain[k] = True
        if footprint_valid.any():
            valid_indices = footprint_indices[footprint_valid]
            fi, fj = valid_indices[:, 0], valid_indices[:, 1]
            if not domain_support[fi, fj].all():
                crosses_domain[k] = True
            overlaps_blocked[k] = bool(raw_blocked[fi, fj].any())
            overlaps_unknown[k] = bool((~known_trav[fi, fj]).any())
            footprint_min_clearance[k] = float(np.min(obstacle_clearance[fi, fj]))
            footprint_min_unknown_clearance[k] = float(np.min(unknown_clearance[fi, fj]))

    outside = crosses_domain | invalid_coordinate
    clearance_blocked = overlaps_blocked & ~locally_blocked
    unknown_footprint = overlaps_unknown & ~unknown_center
    configuration_free = ~(outside | overlaps_blocked | overlaps_unknown)
    display_state = np.full(n, int(ReachabilityState.REACHABLE), dtype=np.int16)
    display_state[unknown_footprint] = int(ReachabilityState.UNKNOWN_FOOTPRINT)
    display_state[clearance_blocked] = int(ReachabilityState.CLEARANCE_BLOCKED)
    display_state[locally_blocked] = int(ReachabilityState.LOCALLY_BLOCKED)
    display_state[unknown_center] = int(ReachabilityState.UNKNOWN_CENTER)
    display_state[outside] = int(ReachabilityState.OUTSIDE_DOMAIN)
    display_state[invalid_coordinate] = -1
    forbidden = ~configuration_free
    finite_obstacle = footprint_min_clearance[np.isfinite(footprint_min_clearance)]
    finite_unknown = footprint_min_unknown_clearance[
        np.isfinite(footprint_min_unknown_clearance)
    ]
    return PathEvaluation(
        model="yaw_aware_rectangle",
        display_state=display_state,
        reachable=configuration_free,
        configuration_free=configuration_free,
        locally_blocked=locally_blocked,
        clearance_blocked=clearance_blocked,
        unknown_center=unknown_center,
        unknown_footprint=unknown_footprint,
        outside_domain=outside,
        disconnected=np.zeros(n, dtype=bool),
        invalid_coordinate=invalid_coordinate,
        overlaps_blocked=overlaps_blocked,
        overlaps_unknown=overlaps_unknown,
        crosses_domain=crosses_domain,
        forbidden=forbidden,
        min_obstacle_clearance_m=float(np.min(finite_obstacle))
        if len(finite_obstacle)
        else float("nan"),
        min_unknown_clearance_m=float(np.min(finite_unknown))
        if len(finite_unknown)
        else float("nan"),
    )


def build_clearance_maps(
    known_trav: np.ndarray, risk: np.ndarray, fatal_threshold: float, resolution: float
) -> tuple[np.ndarray, np.ndarray]:
    blocked = known_trav & (risk >= fatal_threshold)
    return (
        _clearance_map(blocked, resolution),
        _clearance_map(~known_trav, resolution),
    )


def evaluation_metrics(
    evaluation: PathEvaluation, dense: DensePath, prefix: str
) -> dict:
    forbidden_ratio = weighted_ratio(evaluation.forbidden, dense)
    longest = max_consecutive_length(evaluation.forbidden, dense)
    first = first_true_index(evaluation.forbidden)
    first_disconnected = first_true_index(evaluation.disconnected)
    metrics = {
        f"{prefix}_free_ratio": weighted_ratio(evaluation.configuration_free, dense),
        f"{prefix}_reachable_ratio": weighted_ratio(evaluation.reachable, dense),
        f"{prefix}_blocked_ratio": weighted_ratio(
            evaluation.locally_blocked | evaluation.clearance_blocked, dense
        ),
        f"{prefix}_local_blocked_ratio": weighted_ratio(evaluation.locally_blocked, dense),
        f"{prefix}_clearance_blocked_ratio": weighted_ratio(
            evaluation.clearance_blocked, dense
        ),
        f"{prefix}_unknown_ratio": weighted_ratio(
            evaluation.unknown_center | evaluation.unknown_footprint, dense
        ),
        f"{prefix}_unknown_center_ratio": weighted_ratio(evaluation.unknown_center, dense),
        f"{prefix}_unknown_footprint_ratio": weighted_ratio(
            evaluation.unknown_footprint, dense
        ),
        f"{prefix}_outside_ratio": weighted_ratio(evaluation.outside_domain, dense),
        f"{prefix}_disconnected_ratio": weighted_ratio(evaluation.disconnected, dense),
        f"{prefix}_invalid_coordinate_ratio": weighted_ratio(
            evaluation.invalid_coordinate, dense
        ),
        f"{prefix}_overlaps_blocked_ratio": weighted_ratio(
            evaluation.overlaps_blocked, dense
        ),
        f"{prefix}_overlaps_unknown_ratio": weighted_ratio(
            evaluation.overlaps_unknown, dense
        ),
        f"{prefix}_crosses_domain_ratio": weighted_ratio(
            evaluation.crosses_domain, dense
        ),
        f"{prefix}_forbidden_ratio": forbidden_ratio,
        f"{prefix}_max_consecutive_violation_length_m": longest,
        f"{prefix}_strict_pass": forbidden_ratio <= 1e-12,
        f"{prefix}_tolerance_1_pass": forbidden_ratio <= 0.01 + 1e-12,
        f"{prefix}_tolerance_5_pass": forbidden_ratio <= 0.05 + 1e-12,
        f"{prefix}_continuous_failure": longest > 0.2,
        f"{prefix}_first_violation_path_index": first,
        f"{prefix}_first_violation_arc_length_m": float(dense.arc_length_m[first])
        if first is not None
        else None,
        f"{prefix}_first_disconnected_path_index": first_disconnected,
        f"{prefix}_first_disconnected_arc_length_m": float(
            dense.arc_length_m[first_disconnected]
        )
        if first_disconnected is not None
        else None,
        f"{prefix}_endpoint_state": state_name(evaluation.display_state[-1]),
    }
    return metrics


def state_name(value: int) -> str:
    if value == -1:
        return "invalid_coordinate"
    return STATE_NAMES.get(int(value), f"unknown_state_{value}")


def first_violation_record(
    evaluation: PathEvaluation, dense: DensePath
) -> dict:
    first = first_true_index(evaluation.forbidden)
    if first is None:
        return {
            "type": None,
            "index": None,
            "arc_length_m": None,
            "x": None,
            "y": None,
            "theta": None,
        }
    point = dense.xytheta[first]
    return {
        "type": state_name(evaluation.display_state[first]),
        "index": first,
        "arc_length_m": float(dense.arc_length_m[first]),
        "x": float(point[0]),
        "y": float(point[1]),
        "theta": float(point[2]),
    }


def alignment_checks(
    path: np.ndarray,
    *,
    frame_id: str,
    elevation_row: Optional[int],
    first_position_tolerance_m: float = 0.25,
    first_theta_tolerance_rad: float = 0.35,
) -> dict:
    finite = bool(np.isfinite(path).all())
    first_distance = float(np.linalg.norm(path[0, :2])) if len(path) else float("nan")
    first_theta = (
        float(abs(np.arctan2(np.sin(path[0, 2]), np.cos(path[0, 2]))))
        if len(path)
        else float("nan")
    )
    checks = {
        "frame_id_base": frame_id == "base",
        "elevation_row_found": elevation_row is not None,
        "path_shape_valid": path.ndim == 2 and path.shape[1] == 3 and len(path) >= 2,
        "path_finite": finite,
        "first_point_near_root": first_distance <= first_position_tolerance_m,
        "first_theta_near_zero": first_theta <= first_theta_tolerance_rad,
        "first_point_distance_m": first_distance,
        "first_theta_abs_rad": first_theta,
    }
    checks["alignment_valid"] = all(
        checks[key]
        for key in (
            "frame_id_base",
            "elevation_row_found",
            "path_shape_valid",
            "path_finite",
            "first_point_near_root",
            "first_theta_near_zero",
        )
    )
    return checks
