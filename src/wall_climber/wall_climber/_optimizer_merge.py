"""Primitive merge / fit helpers used by ``canonical_optimizer``.

These helpers decide when two consecutive primitives can be combined into a
single one, and produce the merged result. They are intentionally pure: they
operate on ``CanonicalCommand`` instances and tolerance scalars only, so they
can be tested in isolation.

Behaviour is identical to the inline helpers that previously lived in
``canonical_optimizer.py``; the optimizer keeps thin wrapper aliases that
delegate here.
"""

from __future__ import annotations

import math

from wall_climber import _optimizer_geometry as _geom
from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalCommand,
    CubicBezier,
    LineSegment,
    Point2D,
    QuadraticBezier,
)


EPS = _geom.EPS


# ------------------------------------------------------------------
# Lines
# ------------------------------------------------------------------

def can_merge_lines(
    first: LineSegment,
    second: LineSegment,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    if not _geom.approximately_equal(first.end, second.start, eps=distance_tolerance_m):
        return False
    if _geom.distance(first.start, first.end) <= distance_tolerance_m:
        return False
    if _geom.distance(second.start, second.end) <= distance_tolerance_m:
        return False
    return (
        _geom.angle_delta_deg(_geom.line_heading(first), _geom.line_heading(second))
        <= angle_tolerance_deg
    )


def merge_lines(first: LineSegment, second: LineSegment) -> LineSegment:
    return LineSegment(start=first.start, end=second.end)


# ------------------------------------------------------------------
# Arcs
# ------------------------------------------------------------------

def can_merge_arcs(
    first: ArcSegment,
    second: ArcSegment,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    if not _geom.approximately_equal(
        first.center, second.center, eps=distance_tolerance_m,
    ):
        return False
    if abs(first.radius - second.radius) > distance_tolerance_m:
        return False
    if not _geom.approximately_equal(
        _geom.arc_end(first), _geom.arc_start(second), eps=distance_tolerance_m,
    ):
        return False
    if (first.sweep_angle_rad >= 0.0) != (second.sweep_angle_rad >= 0.0):
        return False
    tangent_first = first.start_angle_rad + first.sweep_angle_rad + (
        math.pi / 2.0 if first.sweep_angle_rad >= 0.0 else -math.pi / 2.0
    )
    tangent_second = second.start_angle_rad + (
        math.pi / 2.0 if second.sweep_angle_rad >= 0.0 else -math.pi / 2.0
    )
    return (
        _geom.angle_delta_deg(tangent_first, tangent_second) <= angle_tolerance_deg
    )


def merge_arcs(first: ArcSegment, second: ArcSegment) -> ArcSegment:
    return ArcSegment(
        center=first.center,
        radius=first.radius,
        start_angle_rad=first.start_angle_rad,
        sweep_angle_rad=first.sweep_angle_rad + second.sweep_angle_rad,
    )


# ------------------------------------------------------------------
# Cubic / quadratic curve fitting
# ------------------------------------------------------------------

def solve_cubic_controls(
    points: tuple[Point2D, ...],
    *,
    parameters: tuple[float, ...] | None = None,
) -> CubicBezier | None:
    if len(points) < 4:
        return None
    start = points[0]
    end = points[-1]
    if parameters is None or len(parameters) != len(points):
        parameters = _geom.chord_length_parameters(points)
    rows: list[list[float]] = []
    rhs_x: list[float] = []
    rhs_y: list[float] = []
    for point, t in zip(points[1:-1], parameters[1:-1]):
        omt = 1.0 - t
        basis0 = omt ** 3
        basis1 = 3.0 * omt * omt * t
        basis2 = 3.0 * omt * t * t
        basis3 = t ** 3
        rows.append([basis1, basis2])
        rhs_x.append(point[0] - ((basis0 * start[0]) + (basis3 * end[0])))
        rhs_y.append(point[1] - ((basis0 * start[1]) + (basis3 * end[1])))
    if len(rows) < 2:
        return None

    matrix_a = (
        (0.0, 0.0),
        (0.0, 0.0),
    )
    vec_x = [0.0, 0.0]
    vec_y = [0.0, 0.0]
    for (basis1, basis2), x_value, y_value in zip(rows, rhs_x, rhs_y):
        matrix_a = (
            (matrix_a[0][0] + (basis1 * basis1), matrix_a[0][1] + (basis1 * basis2)),
            (matrix_a[1][0] + (basis2 * basis1), matrix_a[1][1] + (basis2 * basis2)),
        )
        vec_x[0] += basis1 * x_value
        vec_x[1] += basis2 * x_value
        vec_y[0] += basis1 * y_value
        vec_y[1] += basis2 * y_value

    determinant = (matrix_a[0][0] * matrix_a[1][1]) - (matrix_a[0][1] * matrix_a[1][0])
    if abs(determinant) <= EPS:
        return None

    def solve(rhs: list[float]) -> tuple[float, float]:
        return (
            ((rhs[0] * matrix_a[1][1]) - (matrix_a[0][1] * rhs[1])) / determinant,
            ((matrix_a[0][0] * rhs[1]) - (rhs[0] * matrix_a[1][0])) / determinant,
        )

    control_x = solve(vec_x)
    control_y = solve(vec_y)
    control1 = (float(control_x[0]), float(control_y[0]))
    control2 = (float(control_x[1]), float(control_y[1]))
    if not all(math.isfinite(value) for value in (*control1, *control2)):
        return None
    return CubicBezier(
        start=start,
        control1=control1,
        control2=control2,
        end=end,
    )


def reduce_cubic_to_quadratic(
    cubic: CubicBezier,
    *,
    fit_tolerance_m: float,
    sampled_points: tuple[Point2D, ...],
) -> QuadraticBezier | None:
    q1 = (
        (3.0 * cubic.control1[0] - cubic.start[0]) * 0.5,
        (3.0 * cubic.control1[1] - cubic.start[1]) * 0.5,
    )
    q2 = (
        (3.0 * cubic.control2[0] - cubic.end[0]) * 0.5,
        (3.0 * cubic.control2[1] - cubic.end[1]) * 0.5,
    )
    if _geom.distance(q1, q2) > max(fit_tolerance_m * 0.6, 1.0e-4):
        return None
    quadratic = QuadraticBezier(
        start=cubic.start,
        control=((q1[0] + q2[0]) * 0.5, (q1[1] + q2[1]) * 0.5),
        end=cubic.end,
    )
    parameters = _geom.chord_length_parameters(sampled_points)
    max_error = 0.0
    for point, t in zip(sampled_points, parameters):
        max_error = max(
            max_error, _geom.distance(_geom.evaluate_quadratic(quadratic, t), point),
        )
    if max_error > fit_tolerance_m * 0.6:
        return None
    return quadratic


def can_merge_curve_pair(
    first: CanonicalCommand,
    second: CanonicalCommand,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    if not _geom.approximately_equal(
        _geom.primitive_end(first),
        _geom.primitive_start(second),
        eps=distance_tolerance_m,
    ):
        return False
    first_tangent = _geom.tangent_angle(first, at_end=True)
    second_tangent = _geom.tangent_angle(second, at_end=False)
    if first_tangent is None or second_tangent is None:
        return False
    return (
        _geom.angle_delta_deg(first_tangent, second_tangent) <= angle_tolerance_deg
    )


def merge_curve_pair(
    first: CanonicalCommand,
    second: CanonicalCommand,
    *,
    fit_tolerance_m: float,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> CanonicalCommand | None:
    if not isinstance(first, (QuadraticBezier, CubicBezier)) or not isinstance(
        second, (QuadraticBezier, CubicBezier),
    ):
        return None
    if not can_merge_curve_pair(
        first,
        second,
        angle_tolerance_deg=angle_tolerance_deg,
        distance_tolerance_m=distance_tolerance_m,
    ):
        return None

    sampled = list(_geom.sample_curve_command(first, segments=12))
    second_sampled = _geom.sample_curve_command(second, segments=12)
    sampled.extend(second_sampled[1:])
    sampled_points = tuple(sampled)
    first_length = max(_geom.primitive_length(first), distance_tolerance_m)
    second_length = max(_geom.primitive_length(second), distance_tolerance_m)
    split_ratio = first_length / (first_length + second_length)
    sampled_parameters = tuple(
        (split_ratio * (index / 12.0)) for index in range(13)
    ) + tuple(
        split_ratio + ((1.0 - split_ratio) * (index / 12.0))
        for index in range(1, 13)
    )
    cubic = solve_cubic_controls(sampled_points, parameters=sampled_parameters)
    if cubic is None:
        return None
    max_error = 0.0
    for point, t in zip(sampled_points, sampled_parameters):
        max_error = max(
            max_error, _geom.distance(_geom.evaluate_cubic(cubic, t), point),
        )
    if max_error > fit_tolerance_m:
        return None

    start_tangent = _geom.tangent_angle(cubic, at_end=False)
    end_tangent = _geom.tangent_angle(cubic, at_end=True)
    first_outer = _geom.tangent_angle(first, at_end=False)
    second_outer = _geom.tangent_angle(second, at_end=True)
    if (
        start_tangent is None
        or end_tangent is None
        or first_outer is None
        or second_outer is None
        or _geom.angle_delta_deg(start_tangent, first_outer) > angle_tolerance_deg
        or _geom.angle_delta_deg(end_tangent, second_outer) > angle_tolerance_deg
    ):
        return None

    quadratic = reduce_cubic_to_quadratic(
        cubic,
        fit_tolerance_m=fit_tolerance_m,
        sampled_points=sampled_points,
    )
    return quadratic or cubic


# ------------------------------------------------------------------
# Arc fitting from a chain of lines
# ------------------------------------------------------------------

def fit_arc_from_line_chain(
    lines: tuple[LineSegment, ...],
    *,
    tolerance_m: float,
) -> ArcSegment | None:
    if len(lines) < 3:
        return None
    polyline = _geom.polyline_points_from_lines(lines)
    if len(polyline) < 4:
        return None

    midpoint = polyline[len(polyline) // 2]
    circle = _geom.circle_from_points(polyline[0], midpoint, polyline[-1])
    if circle is None:
        return None
    center, radius = circle

    radial_error = max(
        abs(_geom.distance(point, center) - radius) for point in polyline
    )
    if radial_error > tolerance_m:
        return None

    signed_turn = 0.0
    for index in range(1, len(polyline) - 1):
        first = (
            polyline[index][0] - polyline[index - 1][0],
            polyline[index][1] - polyline[index - 1][1],
        )
        second = (
            polyline[index + 1][0] - polyline[index][0],
            polyline[index + 1][1] - polyline[index][1],
        )
        signed_turn += math.atan2(
            (first[0] * second[1]) - (first[1] * second[0]),
            (first[0] * second[0]) + (first[1] * second[1]),
        )
    if abs(signed_turn) <= math.radians(10.0):
        return None

    clockwise = signed_turn < 0.0
    unwrapped_angles = _geom.unwrap_angles(
        polyline, center=center, clockwise=clockwise,
    )
    if unwrapped_angles is None:
        return None

    sweep = unwrapped_angles[-1] - unwrapped_angles[0]
    if clockwise and sweep >= -EPS:
        return None
    if not clockwise and sweep <= EPS:
        return None

    arc_length = abs(radius * sweep)
    polyline_length = sum(
        _geom.distance(polyline[index - 1], polyline[index])
        for index in range(1, len(polyline))
    )
    if abs(polyline_length - arc_length) > max(
        tolerance_m * len(polyline), tolerance_m * 2.0,
    ):
        return None

    return ArcSegment(
        center=center,
        radius=radius,
        start_angle_rad=unwrapped_angles[0],
        sweep_angle_rad=sweep,
    )


__all__ = [
    'can_merge_lines',
    'merge_lines',
    'can_merge_arcs',
    'merge_arcs',
    'solve_cubic_controls',
    'reduce_cubic_to_quadratic',
    'can_merge_curve_pair',
    'merge_curve_pair',
    'fit_arc_from_line_chain',
]
