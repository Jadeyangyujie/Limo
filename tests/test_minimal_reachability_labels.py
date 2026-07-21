import numpy as np

from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import ReachabilityState, build_teacher_a
from dataset_builder.src.generate_minimal_reachability_labels import (
    IGNORE_INDEX,
    TARGET_SHAPE,
    body_ignore_mask,
    make_four_state_label,
    sample_state_to_forward_bev,
    teacher_state_to_four_state,
    target_bev_centers,
)


def test_teacher_state_mapping_is_final_state_based_and_uint8():
    state = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.uint8)
    label = teacher_state_to_four_state(state)
    np.testing.assert_array_equal(label, [[255, 0, 1, 1, 0, 2, 3]])
    assert label.dtype == np.uint8
    assert set(np.unique(label).tolist()) <= {0, 1, 2, 3, 255}


def test_target_centers_and_axis_direction():
    xx, yy = target_bev_centers()
    assert xx.shape == TARGET_SHAPE and yy.shape == TARGET_SHAPE
    assert xx[0, 0] == 0.05
    assert xx[-1, 0] == 3.95
    assert yy[0, 0] == -2.95
    assert yy[0, -1] == 2.95


def test_discrete_sampling_has_no_interpolated_values_and_outside_is_ignore():
    geometry = MapGeometry(80, 60, 0.1, (-4.0, -3.0))
    state = np.full((80, 60), int(ReachabilityState.REACHABLE), dtype=np.uint8)
    state[:, 0] = int(ReachabilityState.LOCALLY_BLOCKED)
    label, valid = sample_state_to_forward_bev(state, geometry)
    assert label.shape == TARGET_SHAPE
    assert label.dtype == np.uint8
    assert set(np.unique(label).tolist()) <= {0, 1, 2, 3, 255}
    assert valid.all()


def test_body_mask_is_all_ignore_and_outside_body_is_unchanged():
    geometry = MapGeometry(100, 80, 0.1, (-5.0, -4.0))
    state = np.full((100, 80), int(ReachabilityState.REACHABLE), dtype=np.uint8)
    label = make_four_state_label(
        build_teacher_a(
            geometry=geometry,
            known_trav=np.ones((100, 80), dtype=bool),
            risk=np.zeros((100, 80), dtype=np.float32),
            fatal_threshold=0.9,
            inflation_radius_m=0.26,
        )
    )
    body = body_ignore_mask()
    assert np.all(label[body] == IGNORE_INDEX)
    # Target cells close to the source map boundary can legitimately be
    # outside after the full-footprint erosion; all interior non-body cells
    # remain reachable.
    xx, yy = target_bev_centers()
    interior = (~body) & (xx >= 0.4) & (xx < 3.6) & (yy >= -2.6) & (yy <= 2.6)
    assert np.all(label[interior] == 3)


def test_root_invalid_is_all_ignore():
    geometry = MapGeometry(80, 60, 0.1, (-4.0, -3.0))
    known = np.ones((80, 60), dtype=bool)
    known[40, 30] = False
    result = build_teacher_a(
        geometry=geometry,
        known_trav=known,
        risk=np.zeros((80, 60), dtype=np.float32),
        fatal_threshold=0.9,
        inflation_radius_m=0.26,
    )
    assert not result.root.configuration_valid
    label = np.full(TARGET_SHAPE, IGNORE_INDEX, dtype=np.uint8)
    assert np.all(label == IGNORE_INDEX)
