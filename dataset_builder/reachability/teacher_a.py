from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import IntEnum
from math import ceil, sqrt
from typing import Optional

import numpy as np
from scipy import ndimage

from dataset_builder.reachability.coordinates import MapGeometry


class ReachabilityState(IntEnum):
    OUTSIDE_DOMAIN = 0
    UNKNOWN_CENTER = 1
    LOCALLY_BLOCKED = 2
    CLEARANCE_BLOCKED = 3
    UNKNOWN_FOOTPRINT = 4
    TRAVERSABLE_BUT_DISCONNECTED = 5
    REACHABLE = 6


STATE_NAMES = {int(state): state.name.lower() for state in ReachabilityState}


@dataclass
class RootStatus:
    index: tuple[int, int]
    center_valid: bool
    configuration_valid: bool
    failure_reason: Optional[str]


@dataclass
class TeacherAResult:
    geometry: MapGeometry
    inflation_radius_m: float
    effective_radius_m: float
    domain_support: np.ndarray
    planning_domain: np.ndarray
    outside_domain: np.ndarray
    known_trav: np.ndarray
    local_traversable: np.ndarray
    local_blocked: np.ndarray
    clearance_blocked: np.ndarray
    unknown_center: np.ndarray
    unknown_footprint_overlap: np.ndarray
    unknown_footprint: np.ndarray
    configuration_free: np.ndarray
    reachable: np.ndarray
    traversable_but_disconnected: np.ndarray
    geodesic_m: np.ndarray
    predecessor: np.ndarray
    state: np.ndarray
    root: RootStatus

    def state_counts(self) -> dict[str, int]:
        return {
            STATE_NAMES[value]: int(np.count_nonzero(self.state == value))
            for value in sorted(STATE_NAMES)
        }

    def summary(self) -> dict:
        n = int(self.state.size)
        domain_n = int(self.planning_domain.sum())
        counts = self.state_counts()
        return {
            "inflation_radius_m": self.inflation_radius_m,
            "effective_radius_m": self.effective_radius_m,
            "root_index": list(self.root.index),
            "root_center_valid": self.root.center_valid,
            "root_configuration_valid": self.root.configuration_valid,
            "root_failure_reason": self.root.failure_reason,
            "planning_domain_fraction": domain_n / n,
            "configuration_free_fraction_of_domain": float(self.configuration_free.sum())
            / max(domain_n, 1),
            "reachable_fraction_of_domain": float(self.reachable.sum()) / max(domain_n, 1),
            "disconnected_fraction_of_domain": float(
                self.traversable_but_disconnected.sum()
            )
            / max(domain_n, 1),
            "unknown_footprint_overlap_fraction_of_domain": float(
                self.unknown_footprint_overlap.sum()
            )
            / max(domain_n, 1),
            "max_geodesic_m": float(np.nanmax(self.geodesic_m))
            if np.isfinite(self.geodesic_m).any()
            else None,
            "state_counts": counts,
        }


@dataclass
class DiagnosticTarget:
    category: str
    index: tuple[int, int]
    expected_result: str
    path: Optional[np.ndarray]


def circular_structure(
    radius_m: float, resolution: float
) -> tuple[np.ndarray, float]:
    """Return a conservative circular grid footprint.

    A requested radius is rounded up to whole grid cells.  The returned
    effective radius is therefore 0.28 m for 0.26 m at 4 cm resolution and
    0.64 m for 0.61 m.  The structure remains Euclidean/circular, not square.
    """
    if radius_m < 0:
        raise ValueError("inflation_radius_m must be non-negative")
    radius_cells = int(ceil(radius_m / resolution - 1e-12))
    if radius_cells == 0:
        return np.ones((1, 1), dtype=bool), 0.0
    offsets = np.arange(-radius_cells, radius_cells + 1)
    di, dj = np.meshgrid(offsets, offsets, indexing="ij")
    structure = di * di + dj * dj <= radius_cells * radius_cells
    return structure, radius_cells * resolution


def _erode_domain(domain_support: np.ndarray, structure: np.ndarray) -> np.ndarray:
    return ndimage.binary_erosion(
        domain_support, structure=structure, border_value=0
    ).astype(bool)


def _dilate(
    mask: np.ndarray, structure: np.ndarray, *, border_value: int
) -> np.ndarray:
    return ndimage.binary_dilation(
        mask, structure=structure, border_value=border_value
    ).astype(bool)


_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, sqrt(2.0)),
    (-1, 1, sqrt(2.0)),
    (1, -1, sqrt(2.0)),
    (1, 1, sqrt(2.0)),
)


def strict_dijkstra(
    configuration_free: np.ndarray,
    root_index: tuple[int, int],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """8-neighbour Dijkstra with diagonal corner cutting forbidden."""
    free = np.asarray(configuration_free, dtype=bool)
    height, width = free.shape
    root_i, root_j = root_index
    distances = np.full((height, width), np.inf, dtype=np.float64)
    predecessor = np.full((height, width, 2), -1, dtype=np.int32)
    if not (0 <= root_i < height and 0 <= root_j < width) or not free[root_i, root_j]:
        return np.full((height, width), np.nan, dtype=np.float32), predecessor

    distances[root_i, root_j] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, root_i, root_j)]
    while queue:
        distance, i, j = heapq.heappop(queue)
        if distance != distances[i, j]:
            continue
        for di, dj, scale in _NEIGHBORS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < height and 0 <= nj < width) or not free[ni, nj]:
                continue
            if di != 0 and dj != 0:
                if not free[i + di, j] or not free[i, j + dj]:
                    continue
            candidate = distance + scale * resolution
            if candidate + 1e-12 < distances[ni, nj]:
                distances[ni, nj] = candidate
                predecessor[ni, nj] = (i, j)
                heapq.heappush(queue, (candidate, ni, nj))

    output = distances.astype(np.float32)
    output[~np.isfinite(distances)] = np.nan
    return output, predecessor


def build_teacher_a(
    *,
    geometry: MapGeometry,
    known_trav: np.ndarray,
    risk: np.ndarray,
    fatal_threshold: float,
    inflation_radius_m: float,
    domain_support: Optional[np.ndarray] = None,
    root_xy: tuple[float, float] = (0.0, 0.0),
) -> TeacherAResult:
    """Build a 2-D, footprint-constrained root geometric reachability field.

    State priority, from highest to lowest, is:
      outside_domain -> unknown_center -> locally_blocked ->
      clearance_blocked -> unknown_footprint -> disconnected -> reachable.

    Clearance takes priority over unknown-footprint only when the centre itself
    is known and locally traversable and the footprint overlaps both kinds of
    forbidden evidence.  The raw overlap masks are retained independently.
    """
    shape = (geometry.height, geometry.width)
    known = np.asarray(known_trav, dtype=bool)
    risk_array = np.asarray(risk, dtype=np.float32)
    if known.shape != shape or risk_array.shape != shape:
        raise ValueError(f"Expected masks with shape {shape}")

    support = (
        geometry.fixed_cartesian_domain()
        if domain_support is None
        else np.asarray(domain_support, dtype=bool)
    )
    if support.shape != shape:
        raise ValueError(f"domain_support must have shape {shape}")

    footprint, effective_radius = circular_structure(
        inflation_radius_m, geometry.resolution
    )
    planning_domain = _erode_domain(support, footprint)
    outside = ~planning_domain

    unknown_center = planning_domain & ~known
    local_traversable = planning_domain & known & (risk_array < fatal_threshold)
    local_blocked = planning_domain & known & ~local_traversable

    # Obstacles and unknown observations outside the centre planning domain can
    # still lie under a valid robot footprint, so use the full raw-grid masks.
    raw_local_blocked = known & (risk_array >= fatal_threshold)
    blocked_overlap = _dilate(raw_local_blocked, footprint, border_value=0)
    unknown_overlap = _dilate(~known, footprint, border_value=1)

    candidate = local_traversable
    clearance_blocked = candidate & blocked_overlap
    unknown_footprint_overlap = candidate & unknown_overlap
    unknown_footprint = candidate & ~clearance_blocked & unknown_overlap
    configuration_free = candidate & ~blocked_overlap & ~unknown_overlap

    root_arr = geometry.world_to_map_idx(np.asarray(root_xy, dtype=np.float64))
    root_index = int(root_arr[0]), int(root_arr[1])
    root_in_grid = bool(geometry.valid_indices(root_arr))
    if not root_in_grid:
        center_valid = False
        configuration_valid = False
        failure_reason = "root_outside_grid"
    else:
        ri, rj = root_index
        center_valid = bool(
            planning_domain[ri, rj]
            and known[ri, rj]
            and local_traversable[ri, rj]
        )
        configuration_valid = bool(configuration_free[ri, rj])
        if not planning_domain[ri, rj]:
            failure_reason = "outside_domain"
        elif not known[ri, rj]:
            failure_reason = "unknown_center"
        elif not local_traversable[ri, rj]:
            failure_reason = "locally_blocked"
        elif clearance_blocked[ri, rj]:
            failure_reason = "clearance_blocked"
        elif unknown_footprint_overlap[ri, rj]:
            failure_reason = "unknown_footprint"
        else:
            failure_reason = None

    root = RootStatus(
        index=root_index,
        center_valid=center_valid,
        configuration_valid=configuration_valid,
        failure_reason=failure_reason,
    )
    if configuration_valid:
        geodesic, predecessor = strict_dijkstra(
            configuration_free, root_index, geometry.resolution
        )
    else:
        geodesic = np.full(shape, np.nan, dtype=np.float32)
        predecessor = np.full((*shape, 2), -1, dtype=np.int32)

    reachable = configuration_free & np.isfinite(geodesic)
    disconnected = configuration_free & ~reachable

    state = np.full(shape, int(ReachabilityState.OUTSIDE_DOMAIN), dtype=np.uint8)
    state[unknown_center] = int(ReachabilityState.UNKNOWN_CENTER)
    state[local_blocked] = int(ReachabilityState.LOCALLY_BLOCKED)
    state[clearance_blocked] = int(ReachabilityState.CLEARANCE_BLOCKED)
    state[unknown_footprint] = int(ReachabilityState.UNKNOWN_FOOTPRINT)
    state[disconnected] = int(ReachabilityState.TRAVERSABLE_BUT_DISCONNECTED)
    state[reachable] = int(ReachabilityState.REACHABLE)

    return TeacherAResult(
        geometry=geometry,
        inflation_radius_m=float(inflation_radius_m),
        effective_radius_m=float(effective_radius),
        domain_support=support,
        planning_domain=planning_domain,
        outside_domain=outside,
        known_trav=known,
        local_traversable=local_traversable,
        local_blocked=local_blocked,
        clearance_blocked=clearance_blocked,
        unknown_center=unknown_center,
        unknown_footprint_overlap=unknown_footprint_overlap,
        unknown_footprint=unknown_footprint,
        configuration_free=configuration_free,
        reachable=reachable,
        traversable_but_disconnected=disconnected,
        geodesic_m=geodesic,
        predecessor=predecessor,
        state=state,
        root=root,
    )


def reconstruct_path(
    result: TeacherAResult, target_index: tuple[int, int]
) -> tuple[Optional[np.ndarray], str]:
    i, j = target_index
    if not (0 <= i < result.geometry.height and 0 <= j < result.geometry.width):
        return None, "outside_grid"
    state = ReachabilityState(int(result.state[i, j]))
    if state != ReachabilityState.REACHABLE:
        return None, STATE_NAMES[int(state)]
    if not result.root.configuration_valid:
        return None, f"invalid_root:{result.root.failure_reason}"

    root = result.root.index
    reverse_path: list[tuple[int, int]] = [(i, j)]
    current = (i, j)
    max_steps = result.geometry.height * result.geometry.width
    for _ in range(max_steps):
        if current == root:
            reverse_path.reverse()
            return np.asarray(reverse_path, dtype=np.int32), "success"
        parent = result.predecessor[current[0], current[1]]
        if parent[0] < 0:
            return None, "broken_predecessor"
        current = int(parent[0]), int(parent[1])
        reverse_path.append(current)
    return None, "predecessor_cycle"


def path_grid_length(path: np.ndarray, resolution: float) -> float:
    if path is None or len(path) < 2:
        return 0.0
    steps = np.diff(path.astype(np.int64), axis=0)
    return float(np.sqrt(np.sum(steps * steps, axis=1)).sum() * resolution)


def validate_reconstructed_path(
    result: TeacherAResult, target_index: tuple[int, int], path: np.ndarray
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if path is None or len(path) == 0:
        return False, ["empty_path"]
    if tuple(path[0]) != result.root.index:
        errors.append("path_does_not_start_at_root")
    if tuple(path[-1]) != tuple(target_index):
        errors.append("path_does_not_end_at_target")
    if not result.configuration_free[path[:, 0], path[:, 1]].all():
        errors.append("path_leaves_configuration_free")
    for first, second in zip(path[:-1], path[1:]):
        di, dj = (second - first).tolist()
        if max(abs(di), abs(dj)) != 1 or (di == 0 and dj == 0):
            errors.append("non_neighbor_step")
            break
        if di != 0 and dj != 0:
            i, j = first.tolist()
            if not result.configuration_free[i + di, j] or not result.configuration_free[i, j + dj]:
                errors.append("diagonal_corner_cut")
                break
    expected = float(result.geodesic_m[target_index])
    actual = path_grid_length(path, result.geometry.resolution)
    if not np.isclose(actual, expected, atol=2e-5, rtol=2e-5):
        errors.append(f"length_mismatch:{actual:.6f}!={expected:.6f}")
    return not errors, errors


def _select_mask_target(
    mask: np.ndarray, root: tuple[int, int], preference: str = "far"
) -> Optional[tuple[int, int]]:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return None
    delta = cells - np.asarray(root)[None]
    squared = np.sum(delta * delta, axis=1)
    index = int(np.argmax(squared) if preference == "far" else np.argmin(squared))
    return int(cells[index, 0]), int(cells[index, 1])


def select_diagnostic_targets(result: TeacherAResult) -> list[DiagnosticTarget]:
    targets: list[DiagnosticTarget] = []
    finite = result.geodesic_m[np.isfinite(result.geodesic_m)]
    if len(finite) > 1:
        for category, quantile in (
            ("reachable_near", 0.2),
            ("reachable_mid", 0.55),
            ("reachable_far", 0.9),
        ):
            desired = float(np.quantile(finite, quantile))
            candidates = np.argwhere(result.reachable)
            values = result.geodesic_m[result.reachable]
            target_arr = candidates[int(np.argmin(np.abs(values - desired)))]
            target = int(target_arr[0]), int(target_arr[1])
            path, reason = reconstruct_path(result, target)
            targets.append(DiagnosticTarget(category, target, reason, path))

    diagnostic_masks = (
        ("traversable_but_disconnected", result.traversable_but_disconnected),
        ("clearance_blocked", result.clearance_blocked),
        (
            "unknown",
            result.unknown_center | result.unknown_footprint,
        ),
    )
    for category, mask in diagnostic_masks:
        target = _select_mask_target(mask, result.root.index, preference="far")
        if target is not None:
            path, reason = reconstruct_path(result, target)
            targets.append(DiagnosticTarget(category, target, reason, path))
    return targets
