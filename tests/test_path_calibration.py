from __future__ import annotations

import unittest

import numpy as np

from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.path_calibration import (
    alignment_checks,
    build_clearance_maps,
    conditional_footprint_metrics,
    diagnostic_root_anchor,
    densify_path,
    evaluate_circular_teacher,
    evaluate_yaw_aware_rectangle,
    max_consecutive_length,
    rectangle_footprint_offsets,
    scoped_evaluation_metrics,
    weighted_ratio,
)
from dataset_builder.reachability.teacher_a import build_teacher_a


class DenseInterpolationTest(unittest.TestCase):
    def test_interval_endpoint_and_arc_weights(self):
        path = np.array([[0.0, 0.0, 0.0], [0.11, 0.0, 0.3], [0.11, 0.09, 0.6]])
        dense = densify_path(path, 0.02)
        self.assertLessEqual(float(np.diff(dense.arc_length_m).max()), 0.02 + 1e-9)
        np.testing.assert_allclose(dense.xytheta[0], path[0], atol=1e-7)
        np.testing.assert_allclose(dense.xytheta[-1], path[-1], atol=1e-7)
        self.assertAlmostEqual(dense.total_length_m, 0.20, places=7)
        self.assertAlmostEqual(float(dense.arc_weights_m.sum()), 0.20, places=7)

    def test_theta_unwrap_uses_short_rotation(self):
        path = np.array([[0.0, 0.0, np.pi - 0.05], [0.1, 0.0, -np.pi + 0.05]])
        dense = densify_path(path, 0.01)
        theta_unwrapped = np.unwrap(dense.xytheta[:, 2])
        self.assertLess(abs(theta_unwrapped[-1] - theta_unwrapped[0]), 0.11)

    def test_weighted_metrics(self):
        dense = densify_path(np.array([[0, 0, 0], [1, 0, 0]], float), 0.1)
        mask = dense.arc_length_m <= 0.2
        self.assertGreater(weighted_ratio(mask, dense), 0.15)
        self.assertLessEqual(max_consecutive_length(mask, dense), 0.2500001)


class RectangleFootprintTest(unittest.TestCase):
    def setUp(self):
        self.geometry = MapGeometry(101, 101, 0.04, (-2.0, -2.0))
        self.known = np.ones((101, 101), dtype=bool)
        y = self.geometry.map_idx_to_world(
            np.column_stack([np.full(101, 50), np.arange(101)])
        )[:, 1]
        self.risk = np.zeros((101, 101), dtype=np.float32)
        # Leave enough room for the repository's floor-quantised footprint
        # offsets (a nominal -0.28 m sample can quantise to the -0.32 m anchor).
        self.risk[:, np.abs(y) >= 0.36] = 1.0
        self.offsets = rectangle_footprint_offsets(
            [[[-0.55, -0.26], [0.55, 0.26]]], 0.04
        )
        self.obstacle_clearance, self.unknown_clearance = build_clearance_maps(
            self.known, self.risk, 0.9, 0.04
        )

    def evaluate(self, theta):
        dense = densify_path(np.array([[0.0, 0.0, theta], [0.01, 0.0, theta]]), 0.005)
        result = evaluate_yaw_aware_rectangle(
            dense,
            geometry=self.geometry,
            footprint_offsets_xy=self.offsets,
            known_trav=self.known,
            risk=self.risk,
            fatal_threshold=0.9,
            domain_support=np.ones((101, 101), bool),
            obstacle_clearance=self.obstacle_clearance,
            unknown_clearance=self.unknown_clearance,
        )
        return dense, result

    def test_aligned_rectangle_fits_where_circumscribed_circle_does_not(self):
        _, rectangle = self.evaluate(0.0)
        self.assertTrue(rectangle.configuration_free.all())
        circle = build_teacher_a(
            geometry=self.geometry,
            known_trav=self.known,
            risk=self.risk,
            fatal_threshold=0.9,
            inflation_radius_m=0.61,
        )
        self.assertFalse(circle.root.configuration_valid)

    def test_rotated_rectangle_detects_corridor_collision(self):
        _, rectangle = self.evaluate(np.pi / 2)
        self.assertTrue(rectangle.overlaps_blocked.any())
        self.assertFalse(rectangle.configuration_free.all())

    def test_independent_blocked_and_unknown_reasons_are_retained(self):
        known = self.known.copy()
        known[45:48, 45:48] = False
        obstacle_clearance, unknown_clearance = build_clearance_maps(
            known, self.risk, 0.9, 0.04
        )
        dense = densify_path(np.array([[0.0, 0.0, np.pi / 2], [0.01, 0, np.pi / 2]]), 0.005)
        result = evaluate_yaw_aware_rectangle(
            dense,
            geometry=self.geometry,
            footprint_offsets_xy=self.offsets,
            known_trav=known,
            risk=self.risk,
            fatal_threshold=0.9,
            domain_support=np.ones((101, 101), bool),
            obstacle_clearance=obstacle_clearance,
            unknown_clearance=unknown_clearance,
        )
        self.assertTrue(result.overlaps_blocked.any())
        self.assertTrue(result.overlaps_unknown.any())


class CircularDiagnosticMaskTest(unittest.TestCase):
    def test_unknown_overlap_survives_clearance_display_priority(self):
        geometry = MapGeometry(51, 51, 0.04, (-1.0, -1.0))
        known = np.ones((51, 51), dtype=bool)
        risk = np.zeros((51, 51), dtype=np.float32)
        root = np.asarray(geometry.root_index)
        risk[root[0], root[1] + 2] = 1.0
        known[root[0] + 2, root[1]] = False
        teacher = build_teacher_a(
            geometry=geometry,
            known_trav=known,
            risk=risk,
            fatal_threshold=0.9,
            inflation_radius_m=0.12,
        )
        obstacle_clearance, unknown_clearance = build_clearance_maps(
            known, risk, 0.9, geometry.resolution
        )
        dense = densify_path(
            np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]), 0.01
        )
        result = evaluate_circular_teacher(
            dense, teacher, obstacle_clearance, unknown_clearance
        )
        self.assertTrue(result.clearance_blocked.all())
        self.assertTrue(result.overlaps_blocked.all())
        self.assertTrue(result.overlaps_unknown.all())


class ConditionalMetricTest(unittest.TestCase):
    def test_unknown_and_outside_are_excluded_from_geometry_agreement(self):
        geometry = MapGeometry(51, 51, 0.04, (-1.0, -1.0))
        known = np.ones((51, 51), dtype=bool)
        risk = np.zeros((51, 51), dtype=np.float32)
        obstacle_clearance, unknown_clearance = build_clearance_maps(
            known, risk, 0.9, geometry.resolution
        )
        dense = densify_path(np.array([[0, 0, 0], [0.2, 0, 0]], float), 0.02)
        teacher = build_teacher_a(
            geometry=geometry,
            known_trav=known,
            risk=risk,
            fatal_threshold=0.9,
            inflation_radius_m=0.0,
        )
        circle = evaluate_circular_teacher(
            dense, teacher, obstacle_clearance, unknown_clearance
        )
        rectangle = evaluate_yaw_aware_rectangle(
            dense,
            geometry=geometry,
            footprint_offsets_xy=np.array([[0.0, 0.0]], dtype=np.float32),
            known_trav=known,
            risk=risk,
            fatal_threshold=0.9,
            domain_support=np.ones((51, 51), bool),
            obstacle_clearance=obstacle_clearance,
            unknown_clearance=unknown_clearance,
        )
        rectangle.overlaps_unknown[-2:] = True
        metrics = conditional_footprint_metrics(circle, rectangle, dense)
        self.assertAlmostEqual(metrics["conditional_circle_rectangle_agreement"], 1.0)
        scope = np.ones(len(dense.xytheta), dtype=bool)
        scoped = scoped_evaluation_metrics(circle, dense, scope)
        self.assertTrue(scoped["strict_pass"])


class RootAnchoringTest(unittest.TestCase):
    def test_seed_only_anchor_does_not_cross_unknown_ring(self):
        geometry = MapGeometry(51, 51, 0.04, (-1.0, -1.0))
        known = np.ones((51, 51), dtype=bool)
        ri, rj = geometry.root_index
        known[ri - 1 : ri + 2, rj - 1 : rj + 2] = False
        risk = np.zeros((51, 51), dtype=np.float32)
        teacher = build_teacher_a(
            geometry=geometry,
            known_trav=known,
            risk=risk,
            fatal_threshold=0.9,
            inflation_radius_m=0.0,
        )
        anchored = diagnostic_root_anchor(
            teacher,
            rectangle_overlaps_blocked=False,
            rectangle_overlaps_unknown=True,
            rectangle_crosses_domain=False,
        )
        self.assertFalse(anchored.raw_valid)
        self.assertTrue(anchored.anchored_valid)
        self.assertTrue(anchored.anchor_applied)
        self.assertEqual(anchored.traversed_nonfree_count_excluding_seed, 0)
        self.assertFalse(anchored.crosses_unknown_or_forbidden)
        self.assertEqual(int(anchored.reachable.sum()), 0)

    def test_seed_only_anchor_recovers_adjacent_canonical_free_space(self):
        geometry = MapGeometry(51, 51, 0.04, (-1.0, -1.0))
        known = np.ones((51, 51), dtype=bool)
        ri, rj = geometry.root_index
        known[ri, rj] = False
        risk = np.zeros((51, 51), dtype=np.float32)
        teacher = build_teacher_a(
            geometry=geometry,
            known_trav=known,
            risk=risk,
            fatal_threshold=0.9,
            inflation_radius_m=0.0,
        )
        anchored = diagnostic_root_anchor(
            teacher,
            rectangle_overlaps_blocked=False,
            rectangle_overlaps_unknown=True,
            rectangle_crosses_domain=False,
        )
        self.assertGreater(int(anchored.reachable.sum()), 0)
        self.assertEqual(anchored.traversed_nonfree_count_excluding_seed, 0)


class AlignmentTest(unittest.TestCase):
    def test_base_path_near_root_is_valid(self):
        path = np.zeros((50, 3), dtype=np.float32)
        checks = alignment_checks(path, frame_id="base", elevation_row=12)
        self.assertTrue(checks["alignment_valid"])

    def test_missing_elevation_row_is_invalid(self):
        path = np.zeros((50, 3), dtype=np.float32)
        checks = alignment_checks(path, frame_id="base", elevation_row=None)
        self.assertFalse(checks["alignment_valid"])


if __name__ == "__main__":
    unittest.main()
