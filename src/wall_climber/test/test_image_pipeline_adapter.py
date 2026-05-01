from __future__ import annotations

import builtins
import importlib

import pytest

from wall_climber.canonical_path import LineSegment, PenDown, PenUp, TravelMove
from wall_climber.image_pipeline.adapters import drawing_path_plan_to_canonical
from wall_climber.image_pipeline.types import (
    DrawingPathPlan,
    PipelineMetrics,
    PipelineMode,
    Point2D,
    Stroke,
)


def _point(x: float, y: float) -> Point2D:
    return Point2D(x=x, y=y)


def _drawing_plan(
    *strokes: Stroke,
    frame: str = 'board',
    metrics: PipelineMetrics | None = None,
    metadata: dict[str, object] | None = None,
) -> DrawingPathPlan:
    return DrawingPathPlan(
        mode=PipelineMode.SKETCH_CENTERLINE,
        strokes=strokes,
        frame=frame,
        metrics=metrics or PipelineMetrics(),
        metadata=metadata or {},
    )


def test_one_stroke_drawing_path_plan_converts_to_canonical() -> None:
    plan = _drawing_plan(
        Stroke(points=(_point(0.0, 0.0), _point(1.0, 0.0))),
    )

    canonical = drawing_path_plan_to_canonical(plan)

    assert canonical.frame == 'board'
    assert canonical.theta_ref == 0.0
    assert [type(command) for command in canonical.commands] == [
        PenDown,
        LineSegment,
        PenUp,
    ]
    line = canonical.commands[1]
    assert isinstance(line, LineSegment)
    assert line.start == (0.0, 0.0)
    assert line.end == (1.0, 0.0)


def test_multiple_strokes_preserve_order_and_insert_travel_move() -> None:
    plan = _drawing_plan(
        Stroke(points=(_point(0.0, 0.0), _point(1.0, 0.0))),
        Stroke(points=(_point(2.0, 0.0), _point(2.0, 1.0))),
    )

    canonical = drawing_path_plan_to_canonical(plan)

    assert [type(command) for command in canonical.commands] == [
        PenDown,
        LineSegment,
        PenUp,
        TravelMove,
        PenDown,
        LineSegment,
        PenUp,
    ]
    first_line = canonical.commands[1]
    travel = canonical.commands[3]
    second_line = canonical.commands[5]
    assert isinstance(first_line, LineSegment)
    assert isinstance(travel, TravelMove)
    assert isinstance(second_line, LineSegment)
    assert first_line.start == (0.0, 0.0)
    assert first_line.end == (1.0, 0.0)
    assert travel.start == (1.0, 0.0)
    assert travel.end == (2.0, 0.0)
    assert second_line.start == (2.0, 0.0)
    assert second_line.end == (2.0, 1.0)


def test_invalid_frame_is_rejected() -> None:
    plan = _drawing_plan(
        Stroke(points=(_point(0.0, 0.0), _point(1.0, 0.0))),
        frame='image',
    )

    with pytest.raises(ValueError, match="frame must be 'board'"):
        drawing_path_plan_to_canonical(plan)


def test_pen_down_false_strokes_are_rejected_explicitly() -> None:
    plan = _drawing_plan(
        Stroke(points=(_point(0.0, 0.0), _point(1.0, 0.0)), pen_down=False),
    )

    with pytest.raises(ValueError, match='pen_down=False'):
        drawing_path_plan_to_canonical(plan)


def test_metrics_and_metadata_are_not_required_for_conversion() -> None:
    plan = DrawingPathPlan(
        mode=PipelineMode.PHOTO_OUTLINE,
        strokes=(Stroke(points=(_point(0.0, 0.0), _point(1.0, 1.0))),),
    )

    canonical = drawing_path_plan_to_canonical(plan)

    assert canonical.frame == 'board'
    assert canonical.theta_ref == 0.0
    assert any(isinstance(command, LineSegment) for command in canonical.commands)


def test_finite_theta_ref_metadata_is_preserved() -> None:
    plan = _drawing_plan(
        Stroke(points=(_point(0.0, 0.0), _point(1.0, 0.0))),
        metadata={'theta_ref': '0.25'},
    )

    canonical = drawing_path_plan_to_canonical(plan)

    assert canonical.theta_ref == 0.25


def test_duplicate_adjacent_points_inside_stroke_are_skipped() -> None:
    plan = _drawing_plan(
        Stroke(
            points=(
                _point(0.0, 0.0),
                _point(0.0, 0.0),
                _point(1.0, 0.0),
                _point(1.0, 0.0),
                _point(1.0, 1.0),
            ),
        ),
    )

    canonical = drawing_path_plan_to_canonical(plan)

    lines = [command for command in canonical.commands if isinstance(command, LineSegment)]
    assert len(lines) == 2
    assert lines[0].start == (0.0, 0.0)
    assert lines[0].end == (1.0, 0.0)
    assert lines[1].start == (1.0, 0.0)
    assert lines[1].end == (1.0, 1.0)


def test_degenerate_stroke_after_duplicate_removal_raises_value_error() -> None:
    plan = _drawing_plan(
        Stroke(points=(_point(0.0, 0.0), _point(0.0, 0.0))),
    )

    with pytest.raises(ValueError, match='becomes degenerate after duplicate removal'):
        drawing_path_plan_to_canonical(plan)


def test_adapter_import_does_not_require_ros_or_webots_runtime(monkeypatch) -> None:
    real_import = builtins.__import__
    forbidden_roots = {
        'controller',
        'rclpy',
        'webots_ros2_driver',
        'webots_ros2_msgs',
    }

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split('.')[0] in forbidden_roots:
            raise AssertionError(f'adapter imported runtime dependency {name!r}')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    module = importlib.reload(importlib.import_module('wall_climber.image_pipeline.adapters'))

    assert module.drawing_path_plan_to_canonical is not None
