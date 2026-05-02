from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy

from wall_climber.canonical_path import (
    CanonicalCommand,
    CanonicalPathPlan,
    CubicBezier,
    LineSegment,
    PenDown,
    PenUp,
    Point2D as CanonicalPoint2D,
    QuadraticBezier,
    TravelMove,
)
from wall_climber.image_pipeline.types import DrawingPathPlan, Point2D, Stroke


_EPS = 1.0e-9
_CORNER_THRESHOLD_DEG = 42.0
_MIN_CURVE_POINTS = 4


@dataclass(frozen=True)
class SmoothCanonicalResult:
    plan: CanonicalPathPlan
    metadata: dict[str, object]


def _point_to_tuple(point: Point2D) -> CanonicalPoint2D:
    return (float(point.x), float(point.y))


def _distance(first: CanonicalPoint2D, second: CanonicalPoint2D) -> float:
    return math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))


def _dedupe_points(points: Iterable[CanonicalPoint2D]) -> tuple[CanonicalPoint2D, ...]:
    output: list[CanonicalPoint2D] = []
    for point in points:
        current = (float(point[0]), float(point[1]))
        if output and _distance(output[-1], current) <= _EPS:
            continue
        output.append(current)
    return tuple(output)


def _stroke_points(stroke: Stroke) -> tuple[CanonicalPoint2D, ...]:
    return _dedupe_points(_point_to_tuple(point) for point in stroke.points)


def _angle_degrees(first: CanonicalPoint2D, middle: CanonicalPoint2D, last: CanonicalPoint2D) -> float:
    incoming = (middle[0] - first[0], middle[1] - first[1])
    outgoing = (last[0] - middle[0], last[1] - middle[1])
    first_len = math.hypot(incoming[0], incoming[1])
    second_len = math.hypot(outgoing[0], outgoing[1])
    if first_len <= _EPS or second_len <= _EPS:
        return 0.0
    dot = ((incoming[0] * outgoing[0]) + (incoming[1] * outgoing[1])) / (first_len * second_len)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _point_line_distance(
    point: CanonicalPoint2D,
    start: CanonicalPoint2D,
    end: CanonicalPoint2D,
) -> float:
    if _distance(start, end) <= _EPS:
        return _distance(point, start)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    t = (((point[0] - start[0]) * dx) + ((point[1] - start[1]) * dy)) / max(_EPS, (dx * dx) + (dy * dy))
    t = max(0.0, min(1.0, t))
    projected = (start[0] + (t * dx), start[1] + (t * dy))
    return _distance(point, projected)


def _rdp(points: tuple[CanonicalPoint2D, ...], *, epsilon_m: float) -> tuple[CanonicalPoint2D, ...]:
    if len(points) <= 2 or epsilon_m <= 0.0:
        return points
    start = points[0]
    end = points[-1]
    max_error = -1.0
    split_index = -1
    for index, point in enumerate(points[1:-1], start=1):
        error = _point_line_distance(point, start, end)
        if error > max_error:
            max_error = error
            split_index = index
    if max_error <= epsilon_m or split_index < 0:
        return (start, end)
    left = _rdp(points[:split_index + 1], epsilon_m=epsilon_m)
    right = _rdp(points[split_index:], epsilon_m=epsilon_m)
    return _dedupe_points((*left[:-1], *right))


def _split_at_corners(points: tuple[CanonicalPoint2D, ...]) -> tuple[tuple[CanonicalPoint2D, ...], ...]:
    if len(points) < 3:
        return (points,)
    spans: list[tuple[CanonicalPoint2D, ...]] = []
    start = 0
    for index in range(1, len(points) - 1):
        if _angle_degrees(points[index - 1], points[index], points[index + 1]) >= _CORNER_THRESHOLD_DEG:
            if index - start >= 1:
                spans.append(points[start:index + 1])
            start = index
    if len(points) - start >= 2:
        spans.append(points[start:])
    return tuple(span for span in spans if len(span) >= 2)


def _chunk_span(
    points: tuple[CanonicalPoint2D, ...],
    *,
    max_curve_segment_points: int,
) -> tuple[tuple[CanonicalPoint2D, ...], ...]:
    max_points = max(_MIN_CURVE_POINTS, int(max_curve_segment_points))
    if len(points) <= max_points:
        return (points,)
    chunks: list[tuple[CanonicalPoint2D, ...]] = []
    start = 0
    while start < len(points) - 1:
        end = min(len(points), start + max_points)
        if end - start >= 2:
            chunks.append(points[start:end])
        if end >= len(points):
            break
        start = end - 1
    return tuple(chunks)


def _chord_parameters(points: tuple[CanonicalPoint2D, ...]) -> tuple[float, ...]:
    distances = [0.0]
    for previous, current in zip(points[:-1], points[1:]):
        distances.append(distances[-1] + _distance(previous, current))
    total = distances[-1]
    if total <= _EPS:
        return tuple(index / max(1, len(points) - 1) for index in range(len(points)))
    return tuple(distance / total for distance in distances)


def _evaluate_quadratic(command: QuadraticBezier, t: float) -> CanonicalPoint2D:
    omt = 1.0 - t
    return (
        (omt * omt * command.start[0]) + (2.0 * omt * t * command.control[0]) + (t * t * command.end[0]),
        (omt * omt * command.start[1]) + (2.0 * omt * t * command.control[1]) + (t * t * command.end[1]),
    )


def _evaluate_cubic(command: CubicBezier, t: float) -> CanonicalPoint2D:
    omt = 1.0 - t
    return (
        (omt ** 3 * command.start[0])
        + (3.0 * omt * omt * t * command.control1[0])
        + (3.0 * omt * t * t * command.control2[0])
        + (t ** 3 * command.end[0]),
        (omt ** 3 * command.start[1])
        + (3.0 * omt * omt * t * command.control1[1])
        + (3.0 * omt * t * t * command.control2[1])
        + (t ** 3 * command.end[1]),
    )


def _fit_error(
    points: tuple[CanonicalPoint2D, ...],
    command: QuadraticBezier | CubicBezier,
) -> tuple[float, int]:
    parameters = _chord_parameters(points)
    evaluator = _evaluate_cubic if isinstance(command, CubicBezier) else _evaluate_quadratic
    errors = [_distance(evaluator(command, t), point) for point, t in zip(points, parameters)]
    if not errors:
        return 0.0, 0
    max_error = max(errors)
    return float(max_error), int(errors.index(max_error))


def _fit_quadratic(points: tuple[CanonicalPoint2D, ...]) -> QuadraticBezier | None:
    if len(points) < 3:
        return None
    start = points[0]
    end = points[-1]
    numerator_x = 0.0
    numerator_y = 0.0
    denominator = 0.0
    for point, t in zip(points[1:-1], _chord_parameters(points)[1:-1]):
        omt = 1.0 - t
        b0 = omt * omt
        b1 = 2.0 * omt * t
        b2 = t * t
        if abs(b1) <= _EPS:
            continue
        numerator_x += b1 * (point[0] - (b0 * start[0]) - (b2 * end[0]))
        numerator_y += b1 * (point[1] - (b0 * start[1]) - (b2 * end[1]))
        denominator += b1 * b1
    if denominator <= _EPS:
        return None
    control = (numerator_x / denominator, numerator_y / denominator)
    if not all(math.isfinite(value) for value in control):
        return None
    return QuadraticBezier(start=start, control=control, end=end)


def _fit_cubic(points: tuple[CanonicalPoint2D, ...]) -> CubicBezier | None:
    if len(points) < 4:
        return None
    start = points[0]
    end = points[-1]
    rows: list[list[float]] = []
    rhs_x: list[float] = []
    rhs_y: list[float] = []
    for point, t in zip(points[1:-1], _chord_parameters(points)[1:-1]):
        omt = 1.0 - t
        b0 = omt ** 3
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t ** 3
        rows.append([b1, b2])
        rhs_x.append(point[0] - (b0 * start[0]) - (b3 * end[0]))
        rhs_y.append(point[1] - (b0 * start[1]) - (b3 * end[1]))
    if len(rows) < 2:
        return None
    matrix = numpy.asarray(rows, dtype=numpy.float64)
    if numpy.linalg.matrix_rank(matrix) < 2:
        return None
    sol_x, *_ = numpy.linalg.lstsq(matrix, numpy.asarray(rhs_x, dtype=numpy.float64), rcond=None)
    sol_y, *_ = numpy.linalg.lstsq(matrix, numpy.asarray(rhs_y, dtype=numpy.float64), rcond=None)
    control1 = (float(sol_x[0]), float(sol_y[0]))
    control2 = (float(sol_x[1]), float(sol_y[1]))
    if not all(math.isfinite(value) for value in (*control1, *control2)):
        return None
    return CubicBezier(start=start, control1=control1, control2=control2, end=end)


def _line_chain(points: tuple[CanonicalPoint2D, ...]) -> tuple[LineSegment, ...]:
    return tuple(LineSegment(start=start, end=end) for start, end in zip(points[:-1], points[1:]))


def _fit_span_recursive(
    points: tuple[CanonicalPoint2D, ...],
    *,
    curve_tolerance_m: float,
    counters: Counter,
    deadline: float,
    warnings: list[str],
    depth: int = 0,
) -> tuple[LineSegment | QuadraticBezier | CubicBezier, ...]:
    if time.perf_counter() >= deadline:
        counters['timeout_line_segments'] += max(0, len(points) - 1)
        if not any('curve fitting time budget' in warning for warning in warnings):
            warnings.append('Sketch curve fitting time budget exceeded; remaining spans were kept as line segments.')
        return _line_chain(points)

    if len(points) < 3:
        counters['line_segments'] += max(0, len(points) - 1)
        return _line_chain(points)

    if len(points) >= 4:
        cubic = _fit_cubic(points)
        if cubic is not None:
            error, worst_index = _fit_error(points, cubic)
            if error <= curve_tolerance_m:
                counters['cubic_beziers'] += 1
                return (cubic,)
        else:
            error = float('inf')
            worst_index = len(points) // 2
    else:
        error = float('inf')
        worst_index = len(points) // 2

    quadratic = _fit_quadratic(points)
    if quadratic is not None:
        quadratic_error, quadratic_worst_index = _fit_error(points, quadratic)
        if quadratic_error <= curve_tolerance_m:
            counters['quadratic_beziers'] += 1
            return (quadratic,)
        if quadratic_error < error:
            error = quadratic_error
            worst_index = quadratic_worst_index

    if depth >= 8 or len(points) <= _MIN_CURVE_POINTS:
        counters['line_segments'] += max(0, len(points) - 1)
        return _line_chain(points)

    split = max(2, min(len(points) - 3, int(worst_index)))
    if split <= 1 or split >= len(points) - 2:
        split = len(points) // 2
    if split <= 1 or split >= len(points) - 1:
        counters['line_segments'] += max(0, len(points) - 1)
        return _line_chain(points)

    left = _fit_span_recursive(
        points[:split + 1],
        curve_tolerance_m=curve_tolerance_m,
        counters=counters,
        deadline=deadline,
        warnings=warnings,
        depth=depth + 1,
    )
    right = _fit_span_recursive(
        points[split:],
        curve_tolerance_m=curve_tolerance_m,
        counters=counters,
        deadline=deadline,
        warnings=warnings,
        depth=depth + 1,
    )
    return left + right


def _theta_ref_from_metadata(plan: DrawingPathPlan) -> float:
    try:
        theta_ref = float(dict(plan.metadata).get('theta_ref', 0.0))
    except (TypeError, ValueError):
        return 0.0
    return theta_ref if math.isfinite(theta_ref) else 0.0


def drawing_path_plan_to_smooth_canonical(
    plan: DrawingPathPlan,
    *,
    curve_tolerance_m: float,
    max_curve_segment_points: int = 32,
    max_fit_time_ms: float = 3000.0,
) -> SmoothCanonicalResult:
    """Convert board-space DrawingPathPlan strokes into curve-aware canonical commands."""

    if plan.frame != 'board':
        raise ValueError("DrawingPathPlan.frame must be 'board' for smooth canonical conversion.")
    tolerance = max(0.0, float(curve_tolerance_m))
    deadline = time.perf_counter() + (max(0.0, float(max_fit_time_ms)) / 1000.0)
    counters: Counter = Counter()
    warnings: list[str] = []
    commands: list[CanonicalCommand] = []
    previous_end: CanonicalPoint2D | None = None

    for stroke in plan.strokes:
        if not stroke.pen_down:
            raise ValueError('Smooth sketch conversion does not support pen_down=False strokes.')
        points = _stroke_points(stroke)
        if len(points) < 2:
            continue
        prepared_points = _rdp(points, epsilon_m=min(max(0.0, tolerance * 0.35), 0.04))
        if len(prepared_points) >= 2:
            points = prepared_points
        start = points[0]
        if previous_end is not None and _distance(previous_end, start) > _EPS:
            commands.append(TravelMove(start=previous_end, end=start))
        commands.append(PenDown())

        for span in _split_at_corners(points):
            fit_span = _rdp(span, epsilon_m=max(0.0, tolerance * 0.35))
            if len(fit_span) < 2:
                fit_span = span
            for chunk in _chunk_span(fit_span, max_curve_segment_points=max_curve_segment_points):
                fitted = _fit_span_recursive(
                    chunk,
                    curve_tolerance_m=tolerance,
                    counters=counters,
                    deadline=deadline,
                    warnings=warnings,
                )
                commands.extend(fitted)
        commands.append(PenUp())
        previous_end = points[-1]

    if not commands:
        raise ValueError('Smooth sketch conversion produced no canonical commands.')

    line_count = int(counters.get('line_segments', 0) + counters.get('timeout_line_segments', 0))
    quadratic_count = int(counters.get('quadratic_beziers', 0))
    cubic_count = int(counters.get('cubic_beziers', 0))
    metadata = {
        'curve_tolerance_m': float(tolerance),
        'max_curve_segment_points': int(max_curve_segment_points),
        'max_fit_time_ms': float(max_fit_time_ms),
        'line_primitive_count': line_count,
        'quadratic_primitive_count': quadratic_count,
        'cubic_primitive_count': cubic_count,
        'curve_primitive_count': quadratic_count + cubic_count,
        'warnings': tuple(warnings),
    }
    return SmoothCanonicalResult(
        plan=CanonicalPathPlan(frame='board', theta_ref=_theta_ref_from_metadata(plan), commands=tuple(commands)),
        metadata=metadata,
    )
