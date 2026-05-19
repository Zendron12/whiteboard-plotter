"""Integration tests for the new quality modules wired into sketch_centerline.

These tests assert that the optional preprocessing, filled-outline split,
and stroke-reorder stages added in the Phase-3 integration round are:

* enabled by the high-quality presets ('detail', 'balanced'),
* surfaced through the plan metadata, and
* actually have the expected effect on the produced plan.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from wall_climber.image_pipeline.sketch_centerline import vectorize_sketch_image_to_plan


def _encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode('.png', image)
    assert ok
    return bytes(buffer)


def _filled_disc_image() -> bytes:
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (0, 0, 0), thickness=-1)
    return _encode(img)


def _scattered_lines_image() -> bytes:
    """Four short horizontal segments at scrambled rows so the reorder
    optimiser has something to improve."""
    img = np.full((400, 400), 255, dtype=np.uint8)
    for (x1, y1), (x2, y2) in (
        ((50, 50), (150, 50)),
        ((50, 350), (150, 350)),
        ((250, 100), (350, 100)),
        ((250, 300), (350, 300)),
    ):
        cv2.line(img, (x1, y1), (x2, y2), 0, thickness=2)
    return _encode(img)


def test_detail_preset_enables_all_quality_stages() -> None:
    """Phase-3 enhancements are auto-enabled by the 'detail' preset (and
    every other non-'raw' preset). Verifies the preset-driven defaults
    so a regression in the wiring is caught immediately."""
    plan = vectorize_sketch_image_to_plan(
        _filled_disc_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='detail',
    )
    metadata = plan.metadata
    assert metadata['preprocessing_enabled'] is True
    assert metadata['filled_outline_enabled'] is True
    assert metadata['stroke_reorder_enabled'] is True
    assert metadata['skeleton_smoothing_enabled'] is True


def test_balanced_preset_also_enables_phase3() -> None:
    plan = vectorize_sketch_image_to_plan(
        _filled_disc_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='balanced',
    )
    metadata = plan.metadata
    assert metadata['preprocessing_enabled'] is True
    assert metadata['filled_outline_enabled'] is True
    assert metadata['stroke_reorder_enabled'] is True
    assert metadata['skeleton_smoothing_enabled'] is True


def test_fast_preset_also_enables_phase3() -> None:
    plan = vectorize_sketch_image_to_plan(
        _filled_disc_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='fast',
    )
    metadata = plan.metadata
    assert metadata['preprocessing_enabled'] is True
    assert metadata['filled_outline_enabled'] is True
    assert metadata['stroke_reorder_enabled'] is True
    assert metadata['skeleton_smoothing_enabled'] is True


def test_raw_preset_disables_all_quality_stages_by_default() -> None:
    plan = vectorize_sketch_image_to_plan(
        _filled_disc_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='raw',
    )
    metadata = plan.metadata
    assert metadata['preprocessing_enabled'] is False
    assert metadata['filled_outline_enabled'] is False
    assert metadata['stroke_reorder_enabled'] is False
    assert metadata['skeleton_smoothing_enabled'] is False


def test_explicit_overrides_take_precedence_over_preset_defaults() -> None:
    plan = vectorize_sketch_image_to_plan(
        _filled_disc_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='detail',
        enable_preprocessing=False,
        enable_filled_outline=False,
        enable_stroke_reorder=False,
        enable_skeleton_smoothing=False,
    )
    metadata = plan.metadata
    assert metadata['preprocessing_enabled'] is False
    assert metadata['filled_outline_enabled'] is False
    assert metadata['stroke_reorder_enabled'] is False
    assert metadata['skeleton_smoothing_enabled'] is False


def test_filled_disc_under_otsu_produces_outline_strokes() -> None:
    """A solid disc should be drawn as an outline, not collapsed to a dot."""
    plan = vectorize_sketch_image_to_plan(
        _filled_disc_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='detail',
        sketch_extraction_method='otsu',
    )
    metadata = plan.metadata
    assert metadata['filled_component_count'] >= 1
    assert metadata['filled_outline_pixel_count'] > 100
    # The drawn perimeter should be substantial (the disc has a visible outline)
    # rather than a degenerate single dot.
    assert plan.metrics.total_drawing_length_m > 1.0


def test_stroke_reorder_does_not_increase_pen_up_travel() -> None:
    plan = vectorize_sketch_image_to_plan(
        _scattered_lines_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='detail',
    )
    metadata = plan.metadata
    travel_before = float(metadata['stroke_reorder_travel_before_px'])
    travel_after = float(metadata['stroke_reorder_travel_after_px'])
    if travel_before > 1.0e-3:
        # Reorder must never make the travel longer; ideally shorter.
        assert travel_after <= travel_before + 1.0e-6


def test_stroke_reorder_metadata_present_when_disabled() -> None:
    plan = vectorize_sketch_image_to_plan(
        _scattered_lines_image(),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='raw',
    )
    metadata = plan.metadata
    # Even when disabled the keys should exist with sensible defaults so
    # the UI/diagnostics don't have to special-case missing fields.
    assert metadata['stroke_reorder_enabled'] is False
    assert metadata['stroke_reorder_iterations'] == 0


def test_preprocessing_does_not_break_simple_line_image() -> None:
    """Sanity check: a thin line image still produces a usable plan when
    the full enhancement chain runs."""
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.line(img, (40, 100), (360, 100), (0, 0, 0), thickness=2)
    plan = vectorize_sketch_image_to_plan(
        _encode(img),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='detail',
    )
    assert plan.metrics.stroke_count >= 1
    assert plan.metrics.total_drawing_length_m > 0.5


def test_skeleton_smoothing_preserves_endpoints() -> None:
    """Smoothing must keep stroke endpoints anchored so that adjacent
    strokes (sharing a junction in the skeleton) still meet exactly.
    """
    from wall_climber.image_pipeline.sketch_centerline import _smooth_pixel_stroke

    stroke = (
        (10, 10), (11, 10), (12, 11), (13, 11), (14, 12),
        (15, 12), (16, 13), (17, 13), (18, 14), (19, 14),
    )
    smoothed = _smooth_pixel_stroke(stroke)
    assert smoothed[0] == stroke[0]
    assert smoothed[-1] == stroke[-1]


def test_skeleton_smoothing_short_stroke_is_noop() -> None:
    """Very short strokes (<5 points) are returned unchanged so that
    deliberately short geometric features aren't accidentally collapsed.
    """
    from wall_climber.image_pipeline.sketch_centerline import _smooth_pixel_stroke

    short = ((1, 1), (2, 2), (3, 3))
    assert _smooth_pixel_stroke(short) == short


def test_skeleton_smoothing_reduces_jitter_on_noisy_diagonal() -> None:
    """A spike pixel in the middle of an otherwise straight stroke must
    be attenuated: the central tap of the binomial kernel is 6/16, so a
    pixel that is +N away from its neighbours comes out at roughly N*6/16
    after smoothing, which is strictly closer to the neighbours."""
    from wall_climber.image_pipeline.sketch_centerline import _smooth_pixel_stroke

    raw = (
        (0, 0), (1, 0), (2, 0), (3, 0),
        (4, 10),
        (5, 0), (6, 0), (7, 0), (8, 0),
    )
    smoothed = _smooth_pixel_stroke(raw)
    # The spike at index 4 must be pulled back toward the neighbouring zeros.
    assert smoothed[4][1] < raw[4][1]
    # Endpoints unchanged.
    assert smoothed[0] == raw[0]
    assert smoothed[-1] == raw[-1]


def test_skeleton_smoothing_returns_subpixel_floats() -> None:
    """The binomial kernel produces fractional coordinates on purpose:
    that is the whole point of smoothing. Rounding the result back to
    integers (the original implementation) discarded the smoothing's
    noise reduction. The helper now keeps floats.

    For the spike pattern (0,0,0,0,10,0,0,0,0) the central output is
    (0*1 + 0*4 + 10*6 + 0*4 + 0*1) / 16 = 3.75 exactly.
    """
    from wall_climber.image_pipeline.sketch_centerline import _smooth_pixel_stroke

    raw = (
        (0, 0), (1, 0), (2, 0), (3, 0),
        (4, 10),
        (5, 0), (6, 0), (7, 0), (8, 0),
    )
    smoothed = _smooth_pixel_stroke(raw)
    assert isinstance(smoothed[4][1], float)
    assert abs(smoothed[4][1] - 3.75) < 1.0e-9


def test_filled_donut_emits_both_outer_and_inner_contours() -> None:
    """A ring-shaped (donut) component must surface both its outer and
    inner edge so the artist's intentional hole is preserved. Earlier
    the splitter used cv2.RETR_EXTERNAL which only produced the outer
    contour, collapsing donuts to filled disks."""
    from wall_climber.image_pipeline._filled_regions import split_filled_and_thin

    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, 255, thickness=-1)
    cv2.circle(img, (200, 200), 30, 0, thickness=-1)  # punch hole

    split = split_filled_and_thin(img)
    outline_pixels = int(np.count_nonzero(split.outline_mask))

    # Outer circle perimeter on the integer pixel grid traces out around
    # 503 px on its own; a combined outer+inner trace lands well above
    # that. We require >550 to assert the inner contour is present without
    # being brittle to grid quantisation.
    assert split.filled_component_count == 1
    assert outline_pixels > 550, (
        f'donut should surface inner+outer contours; '
        f'outer-only would be ~503 px, got {outline_pixels}.'
    )


def test_spur_prune_on_clean_diagonal_does_not_over_prune() -> None:
    """The spur-prune neighbour map is now 8-connected (was 4). Make
    sure that change has not introduced over-pruning on a clean input
    that has no spurs at all: the metric must still report zero
    pruned spurs."""
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.line(img, (40, 100), (360, 100), (0, 0, 0), thickness=2)
    plan = vectorize_sketch_image_to_plan(
        _encode(img),
        board_width_m=2.0,
        board_height_m=1.5,
        optimization_preset='detail',
    )
    assert int(plan.metadata['skeleton_spurs_pruned']) == 0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
