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
    UNKNOWN = 0
    LOCALLY_BLOCKED = 1
    CLEARANCE_BLOCKED = 2
    TRAVERSABLE_BUT_DISCONNECTED = 3
    REACHABLE = 4


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
    known_trav: np.ndarray
    ego_mask: np.ndarray
    trusted_ego_unknown: np.ndarray
    effective_known: np.ndarray
    local_traversable: np.ndarray
    locally_blocked: np.ndarray
    clearance_blocked: np.ndarray
    unknown: np.ndarray
    configuration_free: np.ndarray
    reachable: np.ndarray
    traversable_but_disconnected: np.ndarray
    geodesic_m: np.ndarray
    predecessor: np.ndarray
    semantic_label: np.ndarray
    ignore_mask: np.ndarray
    root: RootStatus

    def semantic_counts(self) -> dict[str, int]:
        return {
            STATE_NAMES[value]: int(np.count_nonzero(self.semantic_label == value))
            for value in sorted(STATE_NAMES)
        }

    def summary(self) -> dict:
        n = int(self.semantic_label.size)
        counts = self.semantic_counts()
        return {
            "inflation_radius_m": self.inflation_radius_m,
            "effective_radius_m": self.effective_radius_m,
            "root_index": list(self.root.index),
            "root_center_valid": self.root.center_valid,
            "root_configuration_valid": self.root.configuration_valid,
            "root_failure_reason": self.root.failure_reason,
            "configuration_free_fraction": float(self.configuration_free.sum()) / max(n, 1),
            "reachable_fraction": float(self.reachable.sum()) / max(n, 1),
            "disconnected_fraction": float(self.traversable_but_disconnected.sum())
            / max(n, 1),
            "trusted_ego_unknown_count": int(self.trusted_ego_unknown.sum()),
            "ignore_count": int(self.ignore_mask.sum()),
            "max_geodesic_m": float(np.nanmax(self.geodesic_m))
            if np.isfinite(self.geodesic_m).any()
            else None,
            "semantic_counts": counts,
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
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("inflation_radius_m must be finite and non-negative")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("map resolution must be finite and positive")
    radius_cells = int(ceil(radius_m / resolution - 1e-12))
    if radius_cells == 0:
        return np.ones((1, 1), dtype=bool), 0.0
    offsets = np.arange(-radius_cells, radius_cells + 1)
    di, dj = np.meshgrid(offsets, offsets, indexing="ij")
    structure = di * di + dj * dj <= radius_cells * radius_cells
    return structure, radius_cells * resolution


def ego_mask_from_rectangles(
    geometry: MapGeometry,
    rectangles,
    *,
    root_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Build a fixed ego mask using the MPPI rectangle sampling convention.

    Each rectangle is ``[[x_min, y_min], [x_max, y_max]]`` in metres relative
    to the robot root. Only unknown cells inside this fixed mask are trusted;
    known fatal cells are never overridden.
    """
    rects = np.asarray(rectangles, dtype=np.float64)
    if rects.size == 0:
        return np.zeros((geometry.height, geometry.width), dtype=bool)
    if rects.ndim != 3 or rects.shape[1:] != (2, 2):
        raise ValueError("ego rectangles must have shape [N, 2, 2]")

    root = geometry.world_to_map_idx(np.asarray(root_xy, dtype=np.float64))
    mask = np.zeros((geometry.height, geometry.width), dtype=bool)
    for rectangle in rects:
        lower = np.minimum(rectangle[0], rectangle[1])
        upper = np.maximum(rectangle[0], rectangle[1])
        i_offsets = np.arange(
            int(np.floor(lower[0] / geometry.resolution)),
            int(np.ceil(upper[0] / geometry.resolution)),
        )
        j_offsets = np.arange(
            int(np.floor(lower[1] / geometry.resolution)),
            int(np.ceil(upper[1] / geometry.resolution)),
        )
        if len(i_offsets) == 0 or len(j_offsets) == 0:
            continue
        ii, jj = np.meshgrid(root[0] + i_offsets, root[1] + j_offsets, indexing="ij")
        valid = (ii >= 0) & (ii < geometry.height) & (jj >= 0) & (jj < geometry.width)
        mask[ii[valid], jj[valid]] = True
    return mask


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
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
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
    ego_mask: Optional[np.ndarray] = None,
    ignore_mask: Optional[np.ndarray] = None,
    root_xy: tuple[float, float] = (0.0, 0.0),
) -> TeacherAResult:
    """Build five semantic reachability classes plus an independent ignore mask.

    Unknown is determined only at the center cell and is never dilated. Ego
    unknown is trusted before obstacle clearance and Dijkstra; it is never
    overwritten to reachable after the search. Only fatal obstacles are
    inflated, and no planning-domain border erosion is applied.
    """
    shape = (geometry.height, geometry.width)
    if not np.isfinite(fatal_threshold):
        raise ValueError("fatal_threshold must be finite")
    risk_array = np.asarray(risk, dtype=np.float32)
    known_input = np.asarray(known_trav, dtype=bool)
    if known_input.shape != shape or risk_array.shape != shape:
        raise ValueError(f"Expected masks with shape {shape}")
    # A cell cannot be semantically known without a usable risk value.
    known = known_input & np.isfinite(risk_array)

    ego = (
        np.zeros(shape, dtype=bool)
        if ego_mask is None
        else np.asarray(ego_mask, dtype=bool)
    )
    if ego.shape != shape:
        raise ValueError(f"ego_mask must have shape {shape}")
    ignored = (
        np.zeros(shape, dtype=bool)
        if ignore_mask is None
        else np.asarray(ignore_mask, dtype=bool)
    )
    if ignored.shape != shape:
        raise ValueError(f"ignore_mask must have shape {shape}")

    footprint, effective_radius = circular_structure(
        inflation_radius_m, geometry.resolution
    )

    trusted_ego_unknown = ego & ~known
    effective_known = known | trusted_ego_unknown
    unknown = ~effective_known
    local_traversable = (known & (risk_array < fatal_threshold)) | trusted_ego_unknown
    locally_blocked = known & (risk_array >= fatal_threshold)

    # Only fatal obstacles are inflated. Unknown is a center-cell semantic and
    # does not invalidate nearby footprint centers.
    raw_local_blocked = known & (risk_array >= fatal_threshold)
    blocked_overlap = _dilate(raw_local_blocked, footprint, border_value=0)

    candidate = local_traversable
    clearance_blocked = candidate & blocked_overlap
    configuration_free = candidate & ~blocked_overlap

    root_arr = geometry.world_to_map_idx(np.asarray(root_xy, dtype=np.float64))
    root_index = int(root_arr[0]), int(root_arr[1])
    root_in_grid = bool(geometry.valid_indices(root_arr))
    if not root_in_grid:
        center_valid = False
        configuration_valid = False
        failure_reason = "root_outside_grid"
    else:
        ri, rj = root_index
        center_valid = bool(local_traversable[ri, rj])
        configuration_valid = bool(configuration_free[ri, rj])
        if not effective_known[ri, rj]:
            failure_reason = "unknown_center"
        elif locally_blocked[ri, rj]:
            failure_reason = "locally_blocked"
        elif clearance_blocked[ri, rj]:
            failure_reason = "clearance_blocked"
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

    semantic_label = np.full(shape, int(ReachabilityState.UNKNOWN), dtype=np.uint8)
    semantic_label[locally_blocked] = int(ReachabilityState.LOCALLY_BLOCKED)
    semantic_label[clearance_blocked] = int(ReachabilityState.CLEARANCE_BLOCKED)
    semantic_label[disconnected] = int(ReachabilityState.TRAVERSABLE_BUT_DISCONNECTED)
    semantic_label[reachable] = int(ReachabilityState.REACHABLE)

    return TeacherAResult(
        geometry=geometry,
        inflation_radius_m=float(inflation_radius_m),
        effective_radius_m=float(effective_radius),
        known_trav=known,
        ego_mask=ego,
        trusted_ego_unknown=trusted_ego_unknown,
        effective_known=effective_known,
        local_traversable=local_traversable,
        locally_blocked=locally_blocked,
        clearance_blocked=clearance_blocked,
        unknown=unknown,
        configuration_free=configuration_free,
        reachable=reachable,
        traversable_but_disconnected=disconnected,
        geodesic_m=geodesic,
        predecessor=predecessor,
        semantic_label=semantic_label,
        ignore_mask=ignored,
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
