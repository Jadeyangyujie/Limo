from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from dataset_builder.mppi_planner.mppi_planner import (
    GridMap2D,
    MPPIObjective,
    world_to_map_idx as existing_world_to_map_idx,
)
from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import (
    ReachabilityState,
    build_teacher_a,
    reconstruct_path,
    validate_reconstructed_path,
)
from dataset_builder.reachability.traversability import compute_limo_traversability


def geometry(size: int = 21, resolution: float = 1.0) -> MapGeometry:
    centre = size // 2
    return MapGeometry(size, size, resolution, (-centre * resolution, -centre * resolution))


def build_from_masks(
    known: np.ndarray,
    traversable: np.ndarray,
    *,
    radius: float = 0.0,
    root_xy=(0.0, 0.0),
):
    geom = geometry(known.shape[0])
    risk = np.where(traversable, 0.0, 1.0).astype(np.float32)
    risk[~known] = np.nan
    return build_teacher_a(
        geometry=geom,
        known_trav=known,
        risk=risk,
        fatal_threshold=0.9,
        inflation_radius_m=radius,
        root_xy=root_xy,
    )


class CoordinateContractTest(unittest.TestCase):
    def test_cardinal_points_and_roundtrip(self):
        geom = MapGeometry(200, 200, 0.04, (-4.0, -4.0))
        gridmap = GridMap2D(
            elevation=torch.zeros(200, 200),
            resolution=0.04,
            origin_xy=torch.tensor([-4.0, -4.0]),
        )
        expected = {
            (0.0, 0.0): (100, 100),
            (1.0, 0.0): (125, 100),
            (0.0, 1.0): (100, 125),
            (0.0, -1.0): (100, 75),
        }
        for world, index in expected.items():
            actual = geom.world_to_map_idx(world)
            self.assertTupleEqual(tuple(actual), index)
            existing = existing_world_to_map_idx(
                torch.tensor([world], dtype=torch.float32), gridmap
            )[0]
            self.assertTupleEqual(tuple(existing.tolist()), index)
            anchor = geom.map_idx_to_world(actual)
            np.testing.assert_allclose(anchor, world, atol=1e-7)
            np.testing.assert_array_equal(geom.world_to_map_idx(anchor), actual)

    def test_arbitrary_world_quantisation_is_bounded(self):
        geom = MapGeometry(200, 200, 0.04, (-4.0, -4.0))
        points = np.array([[0.013, -0.017], [1.029, 0.071], [-2.991, 2.499]])
        anchors = geom.map_idx_to_world(geom.world_to_map_idx(points))
        error = points - anchors
        self.assertTrue((error >= -1e-10).all())
        self.assertTrue((error < geom.resolution + 1e-10).all())


class TeacherATopologyTest(unittest.TestCase):
    def test_open_map_and_octile_distance(self):
        known = np.ones((21, 21), bool)
        result = build_from_masks(known, known)
        self.assertTrue(result.root.configuration_valid)
        self.assertTrue(result.reachable.all())
        root = np.asarray(result.root.index)
        target = tuple(root + np.array([3, 5]))
        expected = 3 * math.sqrt(2.0) + 2
        self.assertAlmostEqual(float(result.geodesic_m[target]), expected, places=5)
        path, reason = reconstruct_path(result, target)
        self.assertEqual(reason, "success")
        valid, errors = validate_reconstructed_path(result, target, path)
        self.assertTrue(valid, errors)

    def test_full_wall_separates_component(self):
        known = np.ones((21, 21), bool)
        traversable = known.copy()
        traversable[13, :] = False
        result = build_from_masks(known, traversable)
        self.assertTrue(result.reachable[12, 10])
        self.assertTrue(result.traversable_but_disconnected[14, 10])
        self.assertTrue(np.isnan(result.geodesic_m[14, 10]))

    def test_wide_and_narrow_gates_respond_to_inflation(self):
        size = 21
        known = np.ones((size, size), bool)
        narrow = known.copy()
        narrow[13, :] = False
        narrow[13, 10] = True
        no_inflation = build_from_masks(known, narrow, radius=0.0)
        one_cell = build_from_masks(known, narrow, radius=1.0)
        self.assertTrue(no_inflation.reachable[16, 10])
        self.assertFalse(one_cell.reachable[16, 10])

        wide = known.copy()
        wide[13, :] = False
        wide[13, 8:13] = True
        inflated_wide = build_from_masks(known, wide, radius=1.0)
        self.assertTrue(inflated_wide.reachable[16, 10])

    def test_unknown_barrier_does_not_propagate(self):
        known = np.ones((21, 21), bool)
        known[13, :] = False
        result = build_from_masks(known, np.ones_like(known))
        self.assertTrue(result.unknown_center[13, 10])
        self.assertTrue(result.traversable_but_disconnected[14, 10])
        self.assertFalse(result.reachable[14, 10])

    def test_obstacle_ring_encloses_disconnected_free_space(self):
        known = np.ones((21, 21), bool)
        traversable = known.copy()
        traversable[13, 13:18] = False
        traversable[17, 13:18] = False
        traversable[13:18, 13] = False
        traversable[13:18, 17] = False
        result = build_from_masks(known, traversable)
        self.assertTrue(result.traversable_but_disconnected[15, 15])

    def test_blocked_root_is_explicit(self):
        known = np.ones((21, 21), bool)
        traversable = known.copy()
        traversable[10, 10] = False
        result = build_from_masks(known, traversable)
        self.assertFalse(result.root.center_valid)
        self.assertFalse(result.root.configuration_valid)
        self.assertEqual(result.root.failure_reason, "locally_blocked")
        self.assertFalse(result.reachable.any())
        self.assertTrue(np.isnan(result.geodesic_m).all())

    def test_unknown_root_is_explicit(self):
        known = np.ones((21, 21), bool)
        known[10, 10] = False
        result = build_from_masks(known, np.ones_like(known))
        self.assertFalse(result.root.center_valid)
        self.assertEqual(result.root.failure_reason, "unknown_center")
        self.assertFalse(result.reachable.any())

    def test_diagonal_corner_cutting_is_forbidden(self):
        known = np.ones((21, 21), bool)
        traversable = np.zeros((21, 21), bool)
        traversable[10, 10] = True
        traversable[11, 11] = True
        result = build_from_masks(known, traversable)
        self.assertTrue(result.reachable[10, 10])
        self.assertTrue(result.traversable_but_disconnected[11, 11])

    def test_radius_monotonicity(self):
        known = np.ones((31, 31), bool)
        traversable = known.copy()
        traversable[20, :] = False
        traversable[20, 12:19] = True
        results = [
            build_from_masks(known, traversable, radius=radius)
            for radius in (0.0, 1.0, 2.0)
        ]
        for smaller, larger in zip(results[:-1], results[1:]):
            self.assertFalse((larger.configuration_free & ~smaller.configuration_free).any())
            self.assertFalse((larger.reachable & ~smaller.reachable).any())

    def test_states_are_exhaustive_and_mutually_exclusive(self):
        known = np.ones((21, 21), bool)
        known[:, 3] = False
        traversable = known.copy()
        traversable[13, :] = False
        result = build_from_masks(known, traversable, radius=1.0)
        valid_codes = {int(state) for state in ReachabilityState}
        self.assertTrue(set(np.unique(result.state)).issubset(valid_codes))
        self.assertEqual(sum(result.state_counts().values()), result.state.size)


class TraversabilityCompatibilityTest(unittest.TestCase):
    def test_matches_mppi_preprocessing(self):
        cfg = OmegaConf.load("dataset_builder/configs/build.yaml").mppi
        elevation = np.zeros((40, 40), dtype=np.float32)
        elevation[9:12, 20:23] = 0.25
        elevation[:4, :8] = np.nan

        filter_model = get_filter_torch("cpu")
        ours = compute_limo_traversability(elevation, filter_model, cfg, "cpu")

        objective = MPPIObjective(cfg, "cpu")
        gridmap = GridMap2D(
            elevation=torch.from_numpy(elevation),
            resolution=0.04,
            origin_xy=torch.tensor([-0.8, -0.8]),
        )
        objective.set_map(gridmap)
        reference = objective.trav.cpu().numpy()
        np.testing.assert_allclose(ours.trav_cost, reference, rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
