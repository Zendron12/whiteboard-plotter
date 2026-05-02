from __future__ import annotations

import math

from wall_climber.canonical_path import CubicBezier, LineSegment, QuadraticBezier
from wall_climber.image_pipeline.curve_fit import drawing_path_plan_to_smooth_canonical
from wall_climber.image_pipeline.types import DrawingPathPlan, PipelineMode, Point2D, Stroke


def _plan(points: list[tuple[float, float]]) -> DrawingPathPlan:
    return DrawingPathPlan(
        mode=PipelineMode.SKETCH_CENTERLINE,
        frame='board',
        strokes=(Stroke(points=tuple(Point2D(x, y) for x, y in points)),),
    )


def test_smooth_curve_conversion_emits_bezier_for_curved_stroke() -> None:
    points = [
        (0.1 + (0.8 * t), 0.7 - (0.45 * math.sin(math.pi * t)))
        for t in [index / 18.0 for index in range(19)]
    ]

    result = drawing_path_plan_to_smooth_canonical(
        _plan(points),
        curve_tolerance_m=0.02,
        max_curve_segment_points=32,
    )

    assert any(isinstance(command, (QuadraticBezier, CubicBezier)) for command in result.plan.commands)
    assert result.metadata['curve_primitive_count'] >= 1
    assert result.metadata['line_primitive_count'] >= 0


def test_smooth_curve_conversion_preserves_sharp_corner() -> None:
    corner = (0.8, 0.2)
    points = [(0.1, 0.2), (0.45, 0.2), corner, (0.8, 0.55), (0.8, 0.9)]

    result = drawing_path_plan_to_smooth_canonical(
        _plan(points),
        curve_tolerance_m=0.5,
        max_curve_segment_points=32,
    )

    primitives = [
        command for command in result.plan.commands
        if isinstance(command, (LineSegment, QuadraticBezier, CubicBezier))
    ]
    assert len(primitives) >= 2
    assert not any(
        isinstance(command, (QuadraticBezier, CubicBezier))
        and command.start == points[0]
        and command.end == points[-1]
        for command in primitives
    )
