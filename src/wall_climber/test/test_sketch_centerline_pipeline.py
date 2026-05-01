from __future__ import annotations

import builtins
import importlib
import math

import cv2  # type: ignore
import numpy
import pytest

from wall_climber.image_pipeline.adapters import drawing_path_plan_to_canonical
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

