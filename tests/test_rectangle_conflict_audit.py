import numpy as np

from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.rectangle_conflict_audit import (
    audit_footprint_states,
    classify_local_overlap,
    classify_path_severity,
    continuous_runs,
    largest_connected_component,
    reconstruct_actions,
)


def test_connected_component_uses_eight_neighbours():
    cells = np.array([[0, 0], [1, 1], [5, 5]])
    assert largest_connected_component(cells) == 2


def test_overlap_location_uses_robot_local_axes():
    assert classify_local_overlap(np.array([[0.5, 0.0]]), 0.55, 0.26) == "front_overhang"
    assert classify_local_overlap(np.array([[-0.5, 0.0]]), 0.55, 0.26) == "rear_overhang"
    assert classify_local_overlap(np.array([[0.0, 0.25]]), 0.55, 0.26) == "left_side"
    assert classify_local_overlap(np.array([[0.5, 0.25]]), 0.55, 0.26) == "rotated_corner"


def test_mppi_aggregate_excludes_outside_and_softens_single_fatal_sample():
    geometry = MapGeometry(4, 4, 1.0, (-2.0, -2.0))
    risk = np.zeros((4, 4), dtype=np.float32)
    cost = np.zeros((4, 4), dtype=np.float32)
    risk[2, 2] = 0.95
    cost[2, 2] = 100000.0
    offsets = np.array([[0, 0], [0, 1], [3, 0]], dtype=np.float32)
    records, _ = audit_footprint_states(
        np.array([[0, 0, 0]], dtype=np.float32),
        geometry=geometry,
        footprint_offsets_xy=offsets,
        risk=risk,
        trav_cost=cost,
        fatal_threshold=0.9,
        half_length=3,
        half_width=1,
    )
    row = records[0]
    assert row["inside_sample_count"] == 2
    assert row["outside_sample_count"] == 1
    assert row["fatal_sample_count"] == 1
    assert row["teacher_a_hard_blocked"]
    assert row["mppi_footprint_aggregate_cost"] == 50000.0


def test_reconstruct_actions_round_trip_for_pre_step_yaw():
    dt = 0.1
    states = np.array([[0.1, 0.0, 0.1], [0.1 + np.cos(0.1) * 0.1, np.sin(0.1) * 0.1, 0.2]])
    actions = reconstruct_actions(states, dt)
    np.testing.assert_allclose(actions, np.array([[1, 0, 1], [1, 0, 1]]), atol=1e-6)


def test_severity_categories_and_continuous_runs():
    record = {"fatal_unique_cell_count": 1, "largest_fatal_component_cells": 1, "fatal_sample_ratio_known": 0.01}
    assert classify_path_severity([record], 0.02) == "isolated_touch"
    deep = dict(record, largest_fatal_component_cells=30)
    assert classify_path_severity([deep], 0.5) == "deep_collision"
    assert continuous_runs([1, 1, 0, 1], [0.01, 0.02, 0.01, 0.04]) == [0.03, 0.04]
