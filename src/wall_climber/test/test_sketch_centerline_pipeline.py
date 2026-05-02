from __future__ import annotations

import builtins
import importlib
import math

import cv2  # type: ignore
import numpy
import pytest

from wall_climber.image_pipeline.adapters import drawing_path_plan_to_canonical
from wall_climber.image_pipeline import sketch_centerline
from wall_climber.image_pipeline.sketch_centerline import vectorize_sketch_image_to_plan
from wall_climber.image_pipeline.types import DrawingPathPlan, PipelineMode


def _encode_png(image: numpy.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return bytes(encoded.tobytes())


def _line_image(*, inverted: bool = False, with_noise: bool = False) -> bytes:
    background = 0 if inverted else 255
    ink = 255 if inverted else 0
    image = numpy.full((100, 180, 3), background, dtype=numpy.uint8)
    cv2.line(image, (20, 50), (160, 50), (ink, ink, ink), 5, lineType=cv2.LINE_AA)
    if with_noise:
        cv2.rectangle(image, (6, 6), (9, 9), (ink, ink, ink), -1)
        cv2.rectangle(image, (168, 88), (171, 91), (ink, ink, ink), -1)
    return _encode_png(image)


def _rectangle_image() -> bytes:
    image = numpy.full((160, 220, 3), 255, dtype=numpy.uint8)
    cv2.rectangle(image, (30, 40), (190, 120), (0, 0, 0), 5, lineType=cv2.LINE_AA)
    return _encode_png(image)


def _faint_line_image() -> bytes:
    image = numpy.full((120, 220, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 35), (200, 35), (0, 0, 0), 3, lineType=cv2.LINE_8)
    cv2.line(image, (20, 85), (200, 85), (180, 180, 180), 3, lineType=cv2.LINE_8)
    return _encode_png(image)


def _broken_line_image() -> bytes:
    image = numpy.full((100, 200, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 50), (85, 50), (0, 0, 0), 3, lineType=cv2.LINE_8)
    cv2.line(image, (94, 50), (160, 50), (0, 0, 0), 3, lineType=cv2.LINE_8)
    return _encode_png(image)


def _short_detail_image() -> bytes:
    image = numpy.full((110, 200, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 45), (170, 45), (0, 0, 0), 3, lineType=cv2.LINE_8)
    cv2.line(image, (45, 75), (48, 75), (0, 0, 0), 1, lineType=cv2.LINE_8)
    cv2.line(image, (95, 78), (98, 78), (0, 0, 0), 1, lineType=cv2.LINE_8)
    return _encode_png(image)


def _perpendicular_nearby_image() -> bytes:
    image = numpy.full((110, 160, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 55), (78, 55), (0, 0, 0), 3, lineType=cv2.LINE_8)
    cv2.line(image, (84, 61), (84, 100), (0, 0, 0), 3, lineType=cv2.LINE_8)
    return _encode_png(image)


def _separate_lines_image() -> bytes:
    image = numpy.full((100, 200, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 25), (160, 25), (0, 0, 0), 3, lineType=cv2.LINE_8)
    cv2.line(image, (20, 75), (160, 75), (0, 0, 0), 3, lineType=cv2.LINE_8)
    return _encode_png(image)


def _all_points(plan: DrawingPathPlan):
    return [point for stroke in plan.strokes for point in stroke.points]


def _bounds(plan: DrawingPathPlan) -> tuple[float, float, float, float]:
    points = _all_points(plan)
    return (
        min(point.x for point in points),
        max(point.x for point in points),
        min(point.y for point in points),
        max(point.y for point in points),
    )


def test_black_line_image_produces_board_drawing_path_plan() -> None:
    plan = vectorize_sketch_image_to_plan(
        _line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        margin_m=0.1,
    )

    assert plan.mode == PipelineMode.SKETCH_CENTERLINE
    assert plan.frame == 'board'
    assert plan.metrics.stroke_count == len(plan.strokes)
    assert plan.metrics.points_after_simplification >= 2
    assert plan.metadata['skeleton_backend'] in {
        'skimage.morphology.skeletonize',
        'cv2.ximgproc.thinning',
    }


def test_timing_metadata_exists_and_is_non_negative() -> None:
    plan = vectorize_sketch_image_to_plan(
        _line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
    )

    timing = plan.metadata['timing']
    expected = {
        'decode_time_ms',
        'resize_time_ms',
        'normalize_time_ms',
        'threshold_time_ms',
        'cleanup_time_ms',
        'skeleton_time_ms',
        'trace_time_ms',
        'simplify_time_ms',
        'merge_time_ms',
        'curve_fit_time_ms',
        'scale_time_ms',
        'preview_total_time_ms',
    }
    assert expected.issubset(timing.keys())
    for key in expected:
        assert timing[key] >= 0.0


def test_max_image_dim_affects_processed_image_size() -> None:
    high = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        max_image_dim=180,
    )
    low = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        max_image_dim=80,
    )

    assert high.metadata['processed_image_size']['width_px'] > low.metadata['processed_image_size']['width_px']
    assert low.metadata['max_image_dim'] == 80


def test_black_on_white_and_white_on_black_both_work() -> None:
    dark_plan = vectorize_sketch_image_to_plan(
        _line_image(inverted=False),
        board_width_m=2.0,
        board_height_m=1.0,
    )
    light_plan = vectorize_sketch_image_to_plan(
        _line_image(inverted=True),
        board_width_m=2.0,
        board_height_m=1.0,
    )

    assert dark_plan.metadata['foreground_polarity'] == 'dark_on_light'
    assert light_plan.metadata['foreground_polarity'] == 'light_on_dark'
    assert dark_plan.strokes
    assert light_plan.strokes


def test_all_output_points_are_finite_and_inside_board_bounds() -> None:
    plan = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=4.0,
        board_height_m=3.0,
        margin_m=0.2,
    )

    for point in _all_points(plan):
        assert math.isfinite(point.x)
        assert math.isfinite(point.y)
        assert 0.0 <= point.x <= 4.0
        assert 0.0 <= point.y <= 3.0


def test_aspect_ratio_is_preserved_reasonably() -> None:
    plan = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=4.0,
        board_height_m=3.0,
        margin_m=0.2,
    )

    min_x, max_x, min_y, max_y = _bounds(plan)
    output_ratio = (max_x - min_x) / (max_y - min_y)

    assert output_ratio == pytest.approx(2.0, rel=0.25)


def test_tiny_noise_is_removed() -> None:
    plan = vectorize_sketch_image_to_plan(
        _line_image(with_noise=True),
        board_width_m=2.0,
        board_height_m=1.0,
        margin_m=0.1,
        min_component_area_px=30,
    )

    assert int(plan.metadata['removed_component_count']) >= 1


def test_line_sensitivity_preserves_faint_gray_lines() -> None:
    low_sensitivity = vectorize_sketch_image_to_plan(
        _faint_line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        margin_m=0.1,
        line_sensitivity=0.0,
        merge_gap_px=0.0,
        min_stroke_length_px=1.0,
        simplify_epsilon_px=0.0,
    )
    high_sensitivity = vectorize_sketch_image_to_plan(
        _faint_line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        margin_m=0.1,
        line_sensitivity=0.6,
        merge_gap_px=0.0,
        min_stroke_length_px=1.0,
        simplify_epsilon_px=0.0,
    )

    assert low_sensitivity.metadata['foreground_pixel_count'] < high_sensitivity.metadata['foreground_pixel_count']
    assert len(low_sensitivity.strokes) == 1
    assert len(high_sensitivity.strokes) >= 2
    assert high_sensitivity.metadata['line_sensitivity'] == pytest.approx(0.6)
    assert high_sensitivity.metadata['effective_threshold_value'] > high_sensitivity.metadata['otsu_threshold_value']


def test_raw_preset_does_not_merge_nearby_broken_strokes() -> None:
    plan = vectorize_sketch_image_to_plan(
        _broken_line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='raw',
    )

    assert len(plan.strokes) == 2
    assert plan.metadata['optimization_preset'] == 'raw'
    assert plan.metadata['merge_enabled'] is False
    assert plan.metadata['merge_count'] == 0


def test_detail_preset_does_not_merge_nearby_broken_strokes() -> None:
    unmerged = vectorize_sketch_image_to_plan(
        _broken_line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='detail',
    )

    assert len(unmerged.strokes) == 2
    assert unmerged.metadata['optimization_preset'] == 'detail'
    assert unmerged.metadata['merge_enabled'] is False
    assert unmerged.metadata['effective_simplify_epsilon_px'] < 0.5


def test_custom_preset_can_merge_broken_line_segments() -> None:
    plan = vectorize_sketch_image_to_plan(
        _broken_line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='custom',
        merge_gap_px=20.0,
        min_stroke_length_px=1.0,
        simplify_epsilon_px=0.0,
    )

    assert len(plan.strokes) == 1
    assert plan.metadata['optimization_preset'] == 'custom'
    assert plan.metadata['effective_merge_gap_px'] == pytest.approx(20.0)
    assert plan.metadata['effective_min_stroke_length_px'] == pytest.approx(1.0)
    assert plan.metadata['effective_simplify_epsilon_px'] == pytest.approx(0.0)
    assert plan.metadata['merge_count'] == 1


def test_merge_cap_returns_warning(monkeypatch) -> None:
    monkeypatch.setattr(sketch_centerline, '_MERGE_MAX_ITERATIONS', 0)

    plan = vectorize_sketch_image_to_plan(
        _broken_line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='custom',
        merge_gap_px=20.0,
        min_stroke_length_px=1.0,
        simplify_epsilon_px=0.0,
    )

    assert plan.metadata['merge_count'] == 0
    assert plan.metadata['merge_warnings']
    assert any('merge stopped' in warning for warning in plan.metadata['warnings'])


def test_fast_preset_reduces_strokes_compared_to_detail() -> None:
    detail = vectorize_sketch_image_to_plan(
        _short_detail_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='detail',
    )
    balanced = vectorize_sketch_image_to_plan(
        _short_detail_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='balanced',
    )
    fast = vectorize_sketch_image_to_plan(
        _short_detail_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='fast',
    )

    assert len(detail.strokes) >= len(balanced.strokes)
    assert len(detail.strokes) > len(fast.strokes)
    assert detail.metadata['effective_simplify_epsilon_px'] < balanced.metadata['effective_simplify_epsilon_px']
    assert detail.metadata['effective_simplify_epsilon_px'] < fast.metadata['effective_simplify_epsilon_px']


def test_merge_does_not_connect_perpendicular_nearby_lines() -> None:
    plan = vectorize_sketch_image_to_plan(
        _perpendicular_nearby_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        optimization_preset='custom',
        merge_gap_px=20.0,
        merge_max_angle_deg=60.0,
        min_stroke_length_px=1.0,
        simplify_epsilon_px=0.0,
    )

    assert len(plan.strokes) == 2
    assert plan.metadata['merge_count'] == 0


def test_merge_does_not_connect_far_unrelated_strokes() -> None:
    plan = vectorize_sketch_image_to_plan(
        _separate_lines_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        merge_gap_px=20.0,
        min_stroke_length_px=1.0,
        simplify_epsilon_px=0.0,
    )

    assert len(plan.strokes) == 2
    assert plan.metadata['merge_count'] == 0


def test_duplicate_adjacent_points_are_not_emitted() -> None:
    plan = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=4.0,
        board_height_m=3.0,
        margin_m=0.2,
    )

    for stroke in plan.strokes:
        for first, second in zip(stroke.points[:-1], stroke.points[1:]):
            assert first != second


def test_scale_percent_changes_output_bounds_size() -> None:
    full_size = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=4.0,
        board_height_m=3.0,
        margin_m=0.2,
        scale_percent=100.0,
    )
    half_size = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=4.0,
        board_height_m=3.0,
        margin_m=0.2,
        scale_percent=50.0,
    )

    full_min_x, full_max_x, _full_min_y, _full_max_y = _bounds(full_size)
    half_min_x, half_max_x, _half_min_y, _half_max_y = _bounds(half_size)

    assert (half_max_x - half_min_x) == pytest.approx((full_max_x - full_min_x) * 0.5, rel=0.05)
    assert half_size.metadata['scale_percent'] == pytest.approx(50.0)


def test_center_coordinates_change_placement() -> None:
    plan = vectorize_sketch_image_to_plan(
        _rectangle_image(),
        board_width_m=4.0,
        board_height_m=3.0,
        margin_m=0.2,
        scale_percent=50.0,
        center_x_m=1.0,
        center_y_m=1.0,
    )
    min_x, max_x, min_y, max_y = _bounds(plan)

    assert ((min_x + max_x) * 0.5) == pytest.approx(1.0, abs=0.02)
    assert ((min_y + max_y) * 0.5) == pytest.approx(1.0, abs=0.02)
    assert plan.metadata['center_x_m'] == pytest.approx(1.0)
    assert plan.metadata['center_y_m'] == pytest.approx(1.0)


def test_invalid_placement_outside_board_raises_clear_error() -> None:
    with pytest.raises(ValueError, match='outside the board bounds'):
        vectorize_sketch_image_to_plan(
            _rectangle_image(),
            board_width_m=4.0,
            board_height_m=3.0,
            margin_m=0.2,
            scale_percent=100.0,
            center_x_m=0.0,
            center_y_m=1.5,
        )


def test_sketch_plan_converts_to_canonical_path_plan() -> None:
    plan = vectorize_sketch_image_to_plan(
        _line_image(),
        board_width_m=2.0,
        board_height_m=1.0,
        margin_m=0.1,
    )

    canonical = drawing_path_plan_to_canonical(plan)

    assert canonical.frame == 'board'
    assert canonical.commands


def test_import_does_not_require_ros_or_webots_runtime(monkeypatch) -> None:
    real_import = builtins.__import__
    forbidden_roots = {
        'controller',
        'rclpy',
        'webots_ros2_driver',
        'webots_ros2_msgs',
    }

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split('.')[0] in forbidden_roots:
            raise AssertionError(f'sketch pipeline imported runtime dependency {name!r}')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    module = importlib.reload(importlib.import_module('wall_climber.image_pipeline.sketch_centerline'))

    assert module.vectorize_sketch_image_to_plan is not None


def test_missing_skeletonization_backend_raises_runtime_error(monkeypatch) -> None:
    module = importlib.import_module('wall_climber.image_pipeline.sketch_centerline')
    monkeypatch.setattr(module, '_skeletonize_with_skimage', lambda _binary: None)
    monkeypatch.setattr(module, '_skeletonize_with_cv2_thinning', lambda _binary: None)

    with pytest.raises(RuntimeError, match='requires a skeletonization backend'):
        module.vectorize_sketch_image_to_plan(
            _line_image(),
            board_width_m=2.0,
            board_height_m=1.0,
        )
