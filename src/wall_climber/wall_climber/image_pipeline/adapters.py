from __future__ import annotations

import math
from typing import Any

from wall_climber.canonical_path import (
    CanonicalCommand,
    CanonicalPathPlan,
    LineSegment,
    PenDown,
    PenUp,
    Point2D as CanonicalPoint2D,
    TravelMove,
)
from wall_climber.image_pipeline.types import DrawingPathPlan, Point2D, Stroke


_EPS = 1.0e-9


def _distance(a: CanonicalPoint2D, b: CanonicalPoint2D) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _approximately_equal(a: CanonicalPoint2D, b: CanonicalPoint2D) -> bool:
    return _distance(a, b) <= _EPS


def _point_to_tuple(point: Point2D) -> CanonicalPoint2D:
    return (float(point.x), float(point.y))


def _stroke_points(stroke: Stroke, *, stroke_index: int) -> tuple[CanonicalPoint2D, ...]:
    points: list[CanonicalPoint2D] = []
    for point in stroke.points:
        current = _point_to_tuple(point)
        if points and _approximately_equal(points[-1], current):
            continue
        points.append(current)
    if len(points) < 2:
        raise ValueError(
            f'DrawingPathPlan stroke[{stroke_index}] becomes degenerate after duplicate removal.'
        )
    return tuple(points)


def _theta_ref_from_metadata(metadata: dict[str, Any]) -> float:
    raw = metadata.get('theta_ref')
    if raw is None:
        return 0.0
    try:
        theta_ref = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(theta_ref):
        return 0.0
    return theta_ref


def drawing_path_plan_to_canonical(plan: DrawingPathPlan) -> CanonicalPathPlan:
    """Convert a future-facing DrawingPathPlan into the existing canonical model."""

    if plan.frame != 'board':
        raise ValueError("DrawingPathPlan.frame must be 'board' for canonical conversion.")

    commands: list[CanonicalCommand] = []
    previous_end: CanonicalPoint2D | None = None

    for stroke_index, stroke in enumerate(plan.strokes):
        if not stroke.pen_down:
            raise ValueError(
                'DrawingPathPlan strokes with pen_down=False are not supported by this adapter yet.'
            )

        points = _stroke_points(stroke, stroke_index=stroke_index)
        start = points[0]
        if previous_end is not None and not _approximately_equal(previous_end, start):
            commands.append(TravelMove(start=previous_end, end=start))

        commands.append(PenDown())
        for start_point, end_point in zip(points[:-1], points[1:]):
            commands.append(LineSegment(start=start_point, end=end_point))
        commands.append(PenUp())
        previous_end = points[-1]

    return CanonicalPathPlan(
        frame='board',
        theta_ref=_theta_ref_from_metadata(dict(plan.metadata)),
        commands=tuple(commands),
    )
