"""Pure geometric helpers used by ``canonical_optimizer``.

These functions are intentionally side-effect free and depend only on the
canonical primitive dataclasses (``LineSegment``, ``ArcSegment``,
``QuadraticBezier``, ``CubicBezier``, ``TravelMove``). They are extracted
from ``canonical_optimizer.py`` so the optimizer module can stay focused on
ordering/merging policy decisions.

Behaviour is identical to the inline helpers that previously lived in the
optimizer; the optimizer now imports these aliases under the same private
names it used before, so callers see no API change.
"""

from __future__ import annotations

import math

from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalCommand,
    CubicBezier,
    LineSegment,
    Point2D,
    QuadraticBezier,
    TravelMove,
)


EPS = 1.0e-9


# ------------------------------------------------------------------
# Vector helpers
# ------------------------------------------------------------------

def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def approximately_equal(a: Point2D, b: Point2D, *, eps: float = EPS) -> bool:
    return distance(a, b) <= eps


def angle_delta_deg(first: float, second: float) -> float:
    delta = math.atan2(math.sin(second - first), math.cos(second - first))
    return abs(math.degrees(delta))


def vector_angle(vector: Point2D) -> float:
    return math.atan2(vector[1], vector[0])


# ------------------------------------------------------------------
# Primitive endpoint / length helpers
# ------------------------------------------------------------------

def primitive_start(command: CanonicalCommand) -> Point2D:
    if isinstance(command, TravelMove):
        return command.start
    if isinstance(command, LineSegment):
        return command.start
    if isinstance(command, QuadraticBezier):
        return command.start
    if isinstance(command, CubicBezier):
        return command.start
    if isinstance(command, ArcSegment):
        return (
            command.center[0] + command.radius * math.cos(command.start_angle_rad),
            command.center[1] + command.radius * math.sin(command.start_angle_rad),
        )
    raise ValueError(f'Unsupported primitive {type(command)!r}.')


def primitive_end(command: CanonicalCommand) -> Point2D:
    if isinstance(command, TravelMove):
        return command.end
    if isinstance(command, LineSegment):
        return command.end
    if isinstance(command, QuadraticBezier):
        return command.end
    if isinstance(command, CubicBezier):
        return command.end
    if isinstance(command, ArcSegment):
        angle = command.start_angle_rad + command.sweep_angle_rad
        return (
            command.center[0] + command.radius * math.cos(angle),
            command.center[1] + command.radius * math.sin(angle),
        )
    raise ValueError(f'Unsupported primitive {type(command)!r}.')


def primitive_length(command: CanonicalCommand) -> float:
    if isinstance(command, (TravelMove, LineSegment)):
        return distance(command.start, command.end)
    if isinstance(command, ArcSegment):
        return abs(command.radius * command.sweep_angle_rad)
    if isinstance(command, QuadraticBezier):
        return (
            distance(command.start, command.control)
            + distance(command.control, command.end)
        )
    if isinstance(command, CubicBezier):
        return (
            distance(command.start, command.control1)
            + distance(command.control1, command.control2)
            + distance(command.control2, command.end)
        )
    raise ValueError(f'Unsupported primitive {type(command)!r}.')


def is_draw_primitive(command: CanonicalCommand) -> bool:
    return isinstance(
        command,
        (LineSegment, ArcSegment, QuadraticBezier, CubicBezier),
    )


# ------------------------------------------------------------------
# Reversal of draw commands (used when reordering "units")
# ------------------------------------------------------------------

def reverse_draw_command(command: CanonicalCommand) -> CanonicalCommand:
    if isinstance(command, LineSegment):
        return LineSegment(start=command.end, end=command.start)
    if isinstance(command, QuadraticBezier):
        return QuadraticBezier(
            start=command.end, control=command.control, end=command.start,
        )
    if isinstance(command, CubicBezier):
        return CubicBezier(
            start=command.end,
            control1=command.control2,
            control2=command.control1,
            end=command.start,
        )
    if isinstance(command, ArcSegment):
        return ArcSegment(
            center=command.center,
            radius=command.radius,
            start_angle_rad=command.start_angle_rad + command.sweep_angle_rad,
            sweep_angle_rad=-command.sweep_angle_rad,
        )
    raise ValueError(f'Unsupported draw command {type(command)!r}.')


# ------------------------------------------------------------------
# Line / arc heading helpers
# ------------------------------------------------------------------

def line_heading(line: LineSegment) -> float:
    return math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0])


def arc_start(command: ArcSegment) -> Point2D:
    return (
        command.center[0] + command.radius * math.cos(command.start_angle_rad),
        command.center[1] + command.radius * math.sin(command.start_angle_rad),
    )


def arc_end(command: ArcSegment) -> Point2D:
    angle = command.start_angle_rad + command.sweep_angle_rad
    return (
        command.center[0] + command.radius * math.cos(angle),
        command.center[1] + command.radius * math.sin(angle),
    )


# ------------------------------------------------------------------
# Bezier evaluation and tangents
# ------------------------------------------------------------------

def evaluate_quadratic(command: QuadraticBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        (omt * omt * command.start[0])
        + (2.0 * omt * t * command.control[0])
        + (t * t * command.end[0]),
        (omt * omt * command.start[1])
        + (2.0 * omt * t * command.control[1])
        + (t * t * command.end[1]),
    )


def evaluate_cubic(command: CubicBezier, t: float) -> Point2D:
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


def quadratic_derivative(command: QuadraticBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        2.0 * omt * (command.control[0] - command.start[0])
        + 2.0 * t * (command.end[0] - command.control[0]),
        2.0 * omt * (command.control[1] - command.start[1])
        + 2.0 * t * (command.end[1] - command.control[1]),
    )


def cubic_derivative(command: CubicBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        3.0 * omt * omt * (command.control1[0] - command.start[0])
        + 6.0 * omt * t * (command.control2[0] - command.control1[0])
        + 3.0 * t * t * (command.end[0] - command.control2[0]),
        3.0 * omt * omt * (command.control1[1] - command.start[1])
        + 6.0 * omt * t * (command.control2[1] - command.control1[1])
        + 3.0 * t * t * (command.end[1] - command.control2[1]),
    )


def tangent_angle(command: CanonicalCommand, *, at_end: bool) -> float | None:
    derivative: Point2D | None = None
    if isinstance(command, LineSegment):
        derivative = (
            command.end[0] - command.start[0],
            command.end[1] - command.start[1],
        )
    elif isinstance(command, ArcSegment):
        angle = command.start_angle_rad + (command.sweep_angle_rad if at_end else 0.0)
        offset = math.pi / 2.0 if command.sweep_angle_rad >= 0.0 else -math.pi / 2.0
        derivative = (math.cos(angle + offset), math.sin(angle + offset))
    elif isinstance(command, QuadraticBezier):
        derivative = quadratic_derivative(command, 1.0 if at_end else 0.0)
    elif isinstance(command, CubicBezier):
        derivative = cubic_derivative(command, 1.0 if at_end else 0.0)
    if derivative is None or math.hypot(*derivative) <= EPS:
        return None
    return vector_angle(derivative)


def sample_curve_command(
    command: CanonicalCommand, *, segments: int,
) -> tuple[Point2D, ...]:
    segments = max(2, int(segments))
    if isinstance(command, LineSegment):
        return (command.start, command.end)
    if isinstance(command, ArcSegment):
        return tuple(
            (
                command.center[0]
                + command.radius
                * math.cos(
                    command.start_angle_rad
                    + (command.sweep_angle_rad * (index / segments))
                ),
                command.center[1]
                + command.radius
                * math.sin(
                    command.start_angle_rad
                    + (command.sweep_angle_rad * (index / segments))
                ),
            )
            for index in range(segments + 1)
        )
    if isinstance(command, QuadraticBezier):
        return tuple(
            evaluate_quadratic(command, index / segments)
            for index in range(segments + 1)
        )
    if isinstance(command, CubicBezier):
        return tuple(
            evaluate_cubic(command, index / segments)
            for index in range(segments + 1)
        )
    raise ValueError(f'Unsupported curve command {type(command)!r}.')


def chord_length_parameters(points: tuple[Point2D, ...]) -> tuple[float, ...]:
    if len(points) <= 1:
        return (0.0,)
    distances = [0.0]
    for index in range(1, len(points)):
        distances.append(distances[-1] + distance(points[index - 1], points[index]))
    total = distances[-1]
    if total <= EPS:
        return tuple(
            index / max(1, len(points) - 1) for index in range(len(points))
        )
    return tuple(d / total for d in distances)


# ------------------------------------------------------------------
# Circle and arc fitting helpers
# ------------------------------------------------------------------

def circle_from_points(
    first: Point2D,
    middle: Point2D,
    last: Point2D,
) -> tuple[Point2D, float] | None:
    ax, ay = first
    bx, by = middle
    cx, cy = last
    determinant = 2.0 * (
        ax * (by - cy)
        + bx * (cy - ay)
        + cx * (ay - by)
    )
    if abs(determinant) <= EPS:
        return None
    first_sq = (ax * ax) + (ay * ay)
    middle_sq = (bx * bx) + (by * by)
    last_sq = (cx * cx) + (cy * cy)
    center_x = (
        first_sq * (by - cy)
        + middle_sq * (cy - ay)
        + last_sq * (ay - by)
    ) / determinant
    center_y = (
        first_sq * (cx - bx)
        + middle_sq * (ax - cx)
        + last_sq * (bx - ax)
    ) / determinant
    center = (center_x, center_y)
    radius = distance(center, first)
    if not math.isfinite(radius) or radius <= EPS:
        return None
    return center, radius


def unwrap_angles(
    points: tuple[Point2D, ...],
    *,
    center: Point2D,
    clockwise: bool,
) -> tuple[float, ...] | None:
    if len(points) < 2:
        return None
    angles = [
        math.atan2(point[1] - center[1], point[0] - center[0])
        for point in points
    ]
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        candidate = angle
        if clockwise:
            while candidate >= unwrapped[-1]:
                candidate -= 2.0 * math.pi
        else:
            while candidate <= unwrapped[-1]:
                candidate += 2.0 * math.pi
        if abs(candidate - unwrapped[-1]) <= EPS:
            return None
        unwrapped.append(candidate)
    return tuple(unwrapped)


def polyline_points_from_lines(
    lines: tuple[LineSegment, ...],
) -> tuple[Point2D, ...]:
    """Return the joined polyline of ``lines`` if they are end-to-end.

    Returns an empty tuple if any consecutive pair is not approximately
    connected (within 1.0e-5 m). Mirrors the behaviour of the original
    helper in ``canonical_optimizer.py``.
    """
    points: list[Point2D] = [lines[0].start]
    for line in lines:
        if not approximately_equal(points[-1], line.start, eps=1.0e-5):
            return ()
        points.append(line.end)
    return tuple(points)


__all__ = [
    'EPS',
    'distance',
    'approximately_equal',
    'angle_delta_deg',
    'vector_angle',
    'primitive_start',
    'primitive_end',
    'primitive_length',
    'is_draw_primitive',
    'reverse_draw_command',
    'line_heading',
    'arc_start',
    'arc_end',
    'evaluate_quadratic',
    'evaluate_cubic',
    'quadratic_derivative',
    'cubic_derivative',
    'tangent_angle',
    'sample_curve_command',
    'chord_length_parameters',
    'circle_from_points',
    'unwrap_angles',
    'polyline_points_from_lines',
]
