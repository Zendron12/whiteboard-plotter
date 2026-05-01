from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalCommand,
    CanonicalPathPlan,
    CubicBezier,
    LineSegment,
    PenDown,
    PenUp,
    Point2D,
    QuadraticBezier,
    TravelMove,
)


_EPS = 1.0e-9


@dataclass(frozen=True)
class CanonicalOptimizationPolicy:
    label: str = 'draw'
    merge_collinear_lines: bool = True
    reorder_units: bool = True
    cluster_units: bool = False
    merge_travel_moves: bool = True
    remove_duplicate_units: bool = True
    prune_tiny_primitives: bool = True
    fit_arcs: bool = False
    enable_hatch_ordering: bool = False
    tiny_primitive_m: float = 0.0008
    arc_fit_tolerance_m: float = 0.0015
    merge_angle_tolerance_deg: float = 6.0
    merge_distance_tolerance_m: float = 1.0e-5
    dedupe_precision_m: float = 1.0e-5
    cluster_cell_size_m: float = 0.28


@dataclass(frozen=True)
class CanonicalOptimizationStats:
    policy_label: str
    original_command_count: int
    optimized_command_count: int
    original_unit_count: int
    optimized_unit_count: int
    original_travel_length_m: float
    optimized_travel_length_m: float
    merged_line_segments: int
    merged_curve_segments: int
    merged_travel_moves: int
    pruned_primitives: int
    fitted_arc_segments: int
    removed_duplicate_units: int
    joined_units: int
    hatch_reordered_units: bool
    reordered_units: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            'policy_label': self.policy_label,
            'original_command_count': int(self.original_command_count),
            'optimized_command_count': int(self.optimized_command_count),
            'original_unit_count': int(self.original_unit_count),
            'optimized_unit_count': int(self.optimized_unit_count),
            'original_travel_length_m': float(self.original_travel_length_m),
            'optimized_travel_length_m': float(self.optimized_travel_length_m),
            'travel_reduction_m': float(
                max(0.0, self.original_travel_length_m - self.optimized_travel_length_m)
            ),
            'merged_line_segments': int(self.merged_line_segments),
            'merged_curve_segments': int(self.merged_curve_segments),
            'merged_travel_moves': int(self.merged_travel_moves),
            'pruned_primitives': int(self.pruned_primitives),
            'fitted_arc_segments': int(self.fitted_arc_segments),
            'removed_duplicate_units': int(self.removed_duplicate_units),
            'joined_units': int(self.joined_units),
            'hatch_reordered_units': bool(self.hatch_reordered_units),
            'reordered_units': bool(self.reordered_units),
        }


@dataclass(frozen=True)
class CanonicalOptimizationResult:
    plan: CanonicalPathPlan
    stats: CanonicalOptimizationStats


@dataclass(frozen=True)
class _DrawUnit:
    commands: tuple[CanonicalCommand, ...]
    original_index: int

    @property
    def start(self) -> Point2D:
        return _primitive_start(self.commands[0])

    @property
    def end(self) -> Point2D:
        return _primitive_end(self.commands[-1])

    @property
    def draw_length_m(self) -> float:
        return sum(_primitive_length(command) for command in self.commands)


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _approximately_equal(a: Point2D, b: Point2D, *, eps: float = _EPS) -> bool:
    return _distance(a, b) <= eps


def _angle_delta_deg(first: float, second: float) -> float:
    delta = math.atan2(math.sin(second - first), math.cos(second - first))
    return abs(math.degrees(delta))


def _primitive_start(command: CanonicalCommand) -> Point2D:
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


def _primitive_end(command: CanonicalCommand) -> Point2D:
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


def _primitive_length(command: CanonicalCommand) -> float:
    if isinstance(command, (TravelMove, LineSegment)):
        return _distance(command.start, command.end)
    if isinstance(command, ArcSegment):
        return abs(command.radius * command.sweep_angle_rad)
    if isinstance(command, QuadraticBezier):
        return (
            _distance(command.start, command.control)
            + _distance(command.control, command.end)
        )
    if isinstance(command, CubicBezier):
        return (
            _distance(command.start, command.control1)
            + _distance(command.control1, command.control2)
            + _distance(command.control2, command.end)
        )
    raise ValueError(f'Unsupported primitive {type(command)!r}.')


def _is_draw_primitive(command: CanonicalCommand) -> bool:
    return isinstance(
        command,
        (LineSegment, ArcSegment, QuadraticBezier, CubicBezier),
    )


def _reverse_draw_command(command: CanonicalCommand) -> CanonicalCommand:
    if isinstance(command, LineSegment):
        return LineSegment(start=command.end, end=command.start)
    if isinstance(command, QuadraticBezier):
        return QuadraticBezier(start=command.end, control=command.control, end=command.start)
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


def _reverse_unit(unit: _DrawUnit) -> _DrawUnit:
    return _DrawUnit(
        commands=tuple(_reverse_draw_command(command) for command in reversed(unit.commands)),
        original_index=unit.original_index,
    )


def _line_heading(line: LineSegment) -> float:
    return math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0])


def _can_merge_lines(
    first: LineSegment,
    second: LineSegment,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    if not _approximately_equal(first.end, second.start, eps=distance_tolerance_m):
        return False
    if _distance(first.start, first.end) <= distance_tolerance_m:
        return False
    if _distance(second.start, second.end) <= distance_tolerance_m:
        return False
    return _angle_delta_deg(_line_heading(first), _line_heading(second)) <= angle_tolerance_deg


def _merge_lines(first: LineSegment, second: LineSegment) -> LineSegment:
    return LineSegment(start=first.start, end=second.end)


def _arc_start(command: ArcSegment) -> Point2D:
    return (
        command.center[0] + command.radius * math.cos(command.start_angle_rad),
        command.center[1] + command.radius * math.sin(command.start_angle_rad),
    )


def _arc_end(command: ArcSegment) -> Point2D:
    angle = command.start_angle_rad + command.sweep_angle_rad
    return (
        command.center[0] + command.radius * math.cos(angle),
        command.center[1] + command.radius * math.sin(angle),
    )


def _can_merge_arcs(
    first: ArcSegment,
    second: ArcSegment,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    if not _approximately_equal(first.center, second.center, eps=distance_tolerance_m):
        return False
    if abs(first.radius - second.radius) > distance_tolerance_m:
        return False
    if not _approximately_equal(_arc_end(first), _arc_start(second), eps=distance_tolerance_m):
        return False
    if (first.sweep_angle_rad >= 0.0) != (second.sweep_angle_rad >= 0.0):
        return False
    tangent_first = first.start_angle_rad + first.sweep_angle_rad + (math.pi / 2.0 if first.sweep_angle_rad >= 0.0 else -math.pi / 2.0)
    tangent_second = second.start_angle_rad + (math.pi / 2.0 if second.sweep_angle_rad >= 0.0 else -math.pi / 2.0)
    return _angle_delta_deg(tangent_first, tangent_second) <= angle_tolerance_deg


def _merge_arcs(first: ArcSegment, second: ArcSegment) -> ArcSegment:
    return ArcSegment(
        center=first.center,
        radius=first.radius,
        start_angle_rad=first.start_angle_rad,
        sweep_angle_rad=first.sweep_angle_rad + second.sweep_angle_rad,
    )


def _vector_angle(vector: Point2D) -> float:
    return math.atan2(vector[1], vector[0])


def _quadratic_derivative(command: QuadraticBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        2.0 * omt * (command.control[0] - command.start[0]) + 2.0 * t * (command.end[0] - command.control[0]),
        2.0 * omt * (command.control[1] - command.start[1]) + 2.0 * t * (command.end[1] - command.control[1]),
    )


def _cubic_derivative(command: CubicBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        3.0 * omt * omt * (command.control1[0] - command.start[0])
        + 6.0 * omt * t * (command.control2[0] - command.control1[0])
        + 3.0 * t * t * (command.end[0] - command.control2[0]),
        3.0 * omt * omt * (command.control1[1] - command.start[1])
        + 6.0 * omt * t * (command.control2[1] - command.control1[1])
        + 3.0 * t * t * (command.end[1] - command.control2[1]),
    )


def _tangent_angle(command: CanonicalCommand, *, at_end: bool) -> float | None:
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
        derivative = _quadratic_derivative(command, 1.0 if at_end else 0.0)
    elif isinstance(command, CubicBezier):
        derivative = _cubic_derivative(command, 1.0 if at_end else 0.0)
    if derivative is None or math.hypot(*derivative) <= _EPS:
        return None
    return _vector_angle(derivative)


def _evaluate_quadratic(command: QuadraticBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        (omt * omt * command.start[0]) + (2.0 * omt * t * command.control[0]) + (t * t * command.end[0]),
        (omt * omt * command.start[1]) + (2.0 * omt * t * command.control[1]) + (t * t * command.end[1]),
    )


def _evaluate_cubic(command: CubicBezier, t: float) -> Point2D:
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


def _sample_curve_command(command: CanonicalCommand, *, segments: int) -> tuple[Point2D, ...]:
    segments = max(2, int(segments))
    if isinstance(command, LineSegment):
        return (
            command.start,
            command.end,
        )
    if isinstance(command, ArcSegment):
        return tuple(
            (
                command.center[0] + command.radius * math.cos(command.start_angle_rad + (command.sweep_angle_rad * (index / segments))),
                command.center[1] + command.radius * math.sin(command.start_angle_rad + (command.sweep_angle_rad * (index / segments))),
            )
            for index in range(segments + 1)
        )
    if isinstance(command, QuadraticBezier):
        return tuple(_evaluate_quadratic(command, index / segments) for index in range(segments + 1))
    if isinstance(command, CubicBezier):
        return tuple(_evaluate_cubic(command, index / segments) for index in range(segments + 1))
    raise ValueError(f'Unsupported curve command {type(command)!r}.')


def _chord_length_parameters(points: tuple[Point2D, ...]) -> tuple[float, ...]:
    if len(points) <= 1:
        return (0.0,)
    distances = [0.0]
    for index in range(1, len(points)):
        distances.append(distances[-1] + _distance(points[index - 1], points[index]))
    total = distances[-1]
    if total <= _EPS:
        return tuple(index / max(1, len(points) - 1) for index in range(len(points)))
    return tuple(distance / total for distance in distances)


def _solve_cubic_controls(
    points: tuple[Point2D, ...],
    *,
    parameters: tuple[float, ...] | None = None,
) -> CubicBezier | None:
    if len(points) < 4:
        return None
    start = points[0]
    end = points[-1]
    if parameters is None or len(parameters) != len(points):
        parameters = _chord_length_parameters(points)
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
    if abs(determinant) <= _EPS:
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


def _reduce_cubic_to_quadratic(
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
    if _distance(q1, q2) > max(fit_tolerance_m * 0.6, 1.0e-4):
        return None
    quadratic = QuadraticBezier(
        start=cubic.start,
        control=((q1[0] + q2[0]) * 0.5, (q1[1] + q2[1]) * 0.5),
        end=cubic.end,
    )
    parameters = _chord_length_parameters(sampled_points)
    max_error = 0.0
    for point, t in zip(sampled_points, parameters):
        max_error = max(max_error, _distance(_evaluate_quadratic(quadratic, t), point))
    if max_error > fit_tolerance_m * 0.6:
        return None
    return quadratic


def _can_merge_curve_pair(
    first: CanonicalCommand,
    second: CanonicalCommand,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    if not _approximately_equal(_primitive_end(first), _primitive_start(second), eps=distance_tolerance_m):
        return False
    first_tangent = _tangent_angle(first, at_end=True)
    second_tangent = _tangent_angle(second, at_end=False)
    if first_tangent is None or second_tangent is None:
        return False
    return _angle_delta_deg(first_tangent, second_tangent) <= angle_tolerance_deg


def _merge_curve_pair(
    first: CanonicalCommand,
    second: CanonicalCommand,
    *,
    fit_tolerance_m: float,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> CanonicalCommand | None:
    if not isinstance(first, (QuadraticBezier, CubicBezier)) or not isinstance(second, (QuadraticBezier, CubicBezier)):
        return None
    if not _can_merge_curve_pair(
        first,
        second,
        angle_tolerance_deg=angle_tolerance_deg,
        distance_tolerance_m=distance_tolerance_m,
    ):
        return None

    sampled = list(_sample_curve_command(first, segments=12))
    second_sampled = _sample_curve_command(second, segments=12)
    sampled.extend(second_sampled[1:])
    sampled_points = tuple(sampled)
    first_length = max(_primitive_length(first), distance_tolerance_m)
    second_length = max(_primitive_length(second), distance_tolerance_m)
    split_ratio = first_length / (first_length + second_length)
    sampled_parameters = tuple(
        (split_ratio * (index / 12.0)) for index in range(13)
    ) + tuple(
        split_ratio + ((1.0 - split_ratio) * (index / 12.0)) for index in range(1, 13)
    )
    cubic = _solve_cubic_controls(sampled_points, parameters=sampled_parameters)
    if cubic is None:
        return None
    max_error = 0.0
    for point, t in zip(sampled_points, sampled_parameters):
        max_error = max(max_error, _distance(_evaluate_cubic(cubic, t), point))
    if max_error > fit_tolerance_m:
        return None

    start_tangent = _tangent_angle(cubic, at_end=False)
    end_tangent = _tangent_angle(cubic, at_end=True)
    first_outer = _tangent_angle(first, at_end=False)
    second_outer = _tangent_angle(second, at_end=True)
    if (
        start_tangent is None
        or end_tangent is None
        or first_outer is None
        or second_outer is None
        or _angle_delta_deg(start_tangent, first_outer) > angle_tolerance_deg
        or _angle_delta_deg(end_tangent, second_outer) > angle_tolerance_deg
    ):
        return None

    quadratic = _reduce_cubic_to_quadratic(
        cubic,
        fit_tolerance_m=fit_tolerance_m,
        sampled_points=sampled_points,
    )
    return quadratic or cubic


def _circle_from_points(
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
    if abs(determinant) <= _EPS:
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
    radius = _distance(center, first)
    if not math.isfinite(radius) or radius <= _EPS:
        return None
    return center, radius


def _unwrap_angles(
    points: tuple[Point2D, ...],
    *,
    center: Point2D,
    clockwise: bool,
) -> tuple[float, ...] | None:
    if len(points) < 2:
        return None
    angles = [math.atan2(point[1] - center[1], point[0] - center[0]) for point in points]
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        candidate = angle
        if clockwise:
            while candidate >= unwrapped[-1]:
                candidate -= 2.0 * math.pi
        else:
            while candidate <= unwrapped[-1]:
                candidate += 2.0 * math.pi
        if abs(candidate - unwrapped[-1]) <= _EPS:
            return None
        unwrapped.append(candidate)
    return tuple(unwrapped)


def _polyline_points_from_lines(lines: tuple[LineSegment, ...]) -> tuple[Point2D, ...]:
    points: list[Point2D] = [lines[0].start]
    for line in lines:
        if not _approximately_equal(points[-1], line.start, eps=1.0e-5):
            return ()
        points.append(line.end)
    return tuple(points)


def _fit_arc_from_line_chain(
    lines: tuple[LineSegment, ...],
    *,
    tolerance_m: float,
) -> ArcSegment | None:
    if len(lines) < 3:
        return None
    polyline = _polyline_points_from_lines(lines)
    if len(polyline) < 4:
        return None

    midpoint = polyline[len(polyline) // 2]
    circle = _circle_from_points(polyline[0], midpoint, polyline[-1])
    if circle is None:
        return None
    center, radius = circle

    radial_error = max(abs(_distance(point, center) - radius) for point in polyline)
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
    unwrapped_angles = _unwrap_angles(polyline, center=center, clockwise=clockwise)
    if unwrapped_angles is None:
        return None

    sweep = unwrapped_angles[-1] - unwrapped_angles[0]
    if clockwise and sweep >= -_EPS:
        return None
    if not clockwise and sweep <= _EPS:
        return None

    arc_length = abs(radius * sweep)
    polyline_length = sum(_distance(polyline[index - 1], polyline[index]) for index in range(1, len(polyline)))
    if abs(polyline_length - arc_length) > max(tolerance_m * len(polyline), tolerance_m * 2.0):
        return None

    return ArcSegment(
        center=center,
        radius=radius,
        start_angle_rad=unwrapped_angles[0],
        sweep_angle_rad=sweep,
    )


def _primitive_descriptor(command: CanonicalCommand, *, precision_m: float) -> tuple[Any, ...]:
    scale = max(1.0e-9, float(precision_m))

    def pack_point(point: Point2D) -> tuple[int, int]:
        return (
            int(round(point[0] / scale)),
            int(round(point[1] / scale)),
        )

    def pack_scalar(value: float) -> int:
        return int(round(float(value) / scale))

    if isinstance(command, LineSegment):
        return ('L', pack_point(command.start), pack_point(command.end))
    if isinstance(command, QuadraticBezier):
        return ('Q', pack_point(command.start), pack_point(command.control), pack_point(command.end))
    if isinstance(command, CubicBezier):
        return (
            'C',
            pack_point(command.start),
            pack_point(command.control1),
            pack_point(command.control2),
            pack_point(command.end),
        )
    if isinstance(command, ArcSegment):
        return (
            'A',
            pack_point(command.center),
            pack_scalar(command.radius),
            pack_scalar(command.start_angle_rad),
            pack_scalar(command.sweep_angle_rad),
        )
    raise ValueError(f'Unsupported draw command {type(command)!r}.')


def _unit_signature(unit: _DrawUnit, *, precision_m: float) -> tuple[tuple[Any, ...], ...]:
    forward = tuple(
        _primitive_descriptor(command, precision_m=precision_m)
        for command in unit.commands
    )
    reversed_signature = tuple(
        _primitive_descriptor(command, precision_m=precision_m)
        for command in _reverse_unit(unit).commands
    )
    return min(forward, reversed_signature)


def _extract_draw_units(plan: CanonicalPathPlan) -> tuple[_DrawUnit, ...]:
    units: list[_DrawUnit] = []
    current: list[CanonicalCommand] = []
    unit_index = 0

    for command in plan.commands:
        if isinstance(command, PenDown):
            if current:
                units.append(_DrawUnit(commands=tuple(current), original_index=unit_index))
                unit_index += 1
                current = []
            continue
        if isinstance(command, PenUp):
            if current:
                units.append(_DrawUnit(commands=tuple(current), original_index=unit_index))
                unit_index += 1
                current = []
            continue
        if isinstance(command, TravelMove):
            if current:
                units.append(_DrawUnit(commands=tuple(current), original_index=unit_index))
                unit_index += 1
                current = []
            continue
        if _is_draw_primitive(command):
            current.append(command)

    if current:
        units.append(_DrawUnit(commands=tuple(current), original_index=unit_index))
    return tuple(units)


def _optimize_unit(
    unit: _DrawUnit,
    *,
    policy: CanonicalOptimizationPolicy,
) -> tuple[_DrawUnit | None, int, int, int, int]:
    optimized: list[CanonicalCommand] = []
    pruned_count = 0
    merged_count = 0
    merged_curve_count = 0
    fitted_arc_count = 0

    for command in unit.commands:
        if policy.prune_tiny_primitives and _primitive_length(command) <= policy.tiny_primitive_m:
            pruned_count += 1
            continue
        if (
            policy.merge_collinear_lines
            and optimized
            and isinstance(optimized[-1], LineSegment)
            and isinstance(command, LineSegment)
            and _can_merge_lines(
                optimized[-1],
                command,
                angle_tolerance_deg=policy.merge_angle_tolerance_deg,
                distance_tolerance_m=policy.merge_distance_tolerance_m,
            )
        ):
            optimized[-1] = _merge_lines(optimized[-1], command)
            merged_count += 1
            continue
        if (
            optimized
            and isinstance(optimized[-1], ArcSegment)
            and isinstance(command, ArcSegment)
            and _can_merge_arcs(
                optimized[-1],
                command,
                angle_tolerance_deg=policy.merge_angle_tolerance_deg,
                distance_tolerance_m=policy.merge_distance_tolerance_m,
            )
        ):
            optimized[-1] = _merge_arcs(optimized[-1], command)
            merged_curve_count += 1
            continue
        if (
            optimized
            and isinstance(optimized[-1], (QuadraticBezier, CubicBezier))
            and isinstance(command, (QuadraticBezier, CubicBezier))
        ):
            merged_curve = _merge_curve_pair(
                optimized[-1],
                command,
                fit_tolerance_m=max(policy.arc_fit_tolerance_m, policy.tiny_primitive_m * 2.0),
                angle_tolerance_deg=policy.merge_angle_tolerance_deg,
                distance_tolerance_m=policy.merge_distance_tolerance_m,
            )
            if merged_curve is not None:
                optimized[-1] = merged_curve
                merged_curve_count += 1
                continue
        optimized.append(command)

    if policy.fit_arcs and optimized:
        arc_ready: list[CanonicalCommand] = []
        line_chain: list[LineSegment] = []

        def flush_line_chain() -> None:
            nonlocal fitted_arc_count, line_chain
            if len(line_chain) >= 3:
                fitted = _fit_arc_from_line_chain(
                    tuple(line_chain),
                    tolerance_m=policy.arc_fit_tolerance_m,
                )
                if fitted is not None:
                    arc_ready.append(fitted)
                    fitted_arc_count += 1
                    line_chain = []
                    return
            arc_ready.extend(line_chain)
            line_chain = []

        for command in optimized:
            if isinstance(command, LineSegment):
                if not line_chain or _approximately_equal(
                    line_chain[-1].end,
                    command.start,
                    eps=policy.merge_distance_tolerance_m,
                ):
                    line_chain.append(command)
                    continue
                flush_line_chain()
                line_chain.append(command)
                continue
            flush_line_chain()
            arc_ready.append(command)
        flush_line_chain()
        optimized = arc_ready

    if not optimized:
        return None, pruned_count, merged_count, merged_curve_count, fitted_arc_count
    return (
        _DrawUnit(commands=tuple(optimized), original_index=unit.original_index),
        pruned_count,
        merged_count,
        merged_curve_count,
        fitted_arc_count,
    )


def _path_order_key(unit: _DrawUnit) -> tuple[float, float, float, int]:
    start = unit.start
    end = unit.end
    return (
        min(start[0], end[0]),
        min(start[1], end[1]),
        -unit.draw_length_m,
        unit.original_index,
    )


def _travel_length(units: tuple[_DrawUnit, ...]) -> float:
    if len(units) <= 1:
        return 0.0
    total = 0.0
    for previous, current in zip(units[:-1], units[1:]):
        total += _distance(previous.end, current.start)
    return total


def _unit_midpoint(unit: _DrawUnit) -> Point2D:
    return (
        0.5 * (unit.start[0] + unit.end[0]),
        0.5 * (unit.start[1] + unit.end[1]),
    )


def _dedupe_units(
    units: tuple[_DrawUnit, ...],
    *,
    precision_m: float,
) -> tuple[tuple[_DrawUnit, ...], int]:
    seen: set[tuple[tuple[Any, ...], ...]] = set()
    deduped: list[_DrawUnit] = []
    removed = 0
    for unit in units:
        signature = _unit_signature(unit, precision_m=precision_m)
        if signature in seen:
            removed += 1
            continue
        seen.add(signature)
        deduped.append(unit)
    return tuple(deduped), removed


def _reorder_units(
    units: tuple[_DrawUnit, ...],
    *,
    start_point: Point2D | None = None,
) -> tuple[_DrawUnit, ...]:
    if len(units) <= 1:
        return units

    remaining = list(units)

    def oriented_start_key(unit: _DrawUnit) -> tuple[float, float, float, int]:
        return _path_order_key(unit)

    seeded_candidates = []
    for unit in remaining:
        seeded_candidates.append(unit)
        seeded_candidates.append(_reverse_unit(unit))
    if start_point is None:
        current = min(seeded_candidates, key=oriented_start_key)
    else:
        current = min(
            seeded_candidates,
            key=lambda candidate: (
                _distance(start_point, candidate.start),
                *oriented_start_key(candidate),
            ),
        )

    for index, candidate in enumerate(remaining):
        if candidate.original_index == current.original_index:
            remaining.pop(index)
            break

    ordered = [current]
    current_end = current.end

    while remaining:
        best_choice: tuple[int, _DrawUnit, float, tuple[float, float, float, int]] | None = None
        for list_index, unit in enumerate(remaining):
            for candidate in (unit, _reverse_unit(unit)):
                travel_cost = _distance(current_end, candidate.start)
                candidate_key = _path_order_key(candidate)
                if (
                    best_choice is None
                    or travel_cost < best_choice[2] - _EPS
                    or (
                        abs(travel_cost - best_choice[2]) <= _EPS
                        and candidate_key < best_choice[3]
                    )
                ):
                    best_choice = (list_index, candidate, travel_cost, candidate_key)
        if best_choice is None:
            break
        chosen_index, chosen_unit, _, _ = best_choice
        remaining.pop(chosen_index)
        ordered.append(chosen_unit)
        current_end = chosen_unit.end

    return tuple(ordered)


def _cluster_reorder_units(
    units: tuple[_DrawUnit, ...],
    *,
    cell_size_m: float,
) -> tuple[_DrawUnit, ...]:
    if len(units) <= 2:
        return units

    cell_size = max(0.05, float(cell_size_m))
    buckets: dict[tuple[int, int], list[_DrawUnit]] = {}
    for unit in units:
        midpoint = _unit_midpoint(unit)
        key = (
            int(math.floor(midpoint[0] / cell_size)),
            int(math.floor(midpoint[1] / cell_size)),
        )
        buckets.setdefault(key, []).append(unit)

    if len(buckets) <= 1:
        return _reorder_units(units)

    def cluster_centroid(cluster_units: list[_DrawUnit]) -> Point2D:
        xs = []
        ys = []
        for unit in cluster_units:
            midpoint = _unit_midpoint(unit)
            xs.append(midpoint[0])
            ys.append(midpoint[1])
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    remaining_clusters = [
        {
            'centroid': cluster_centroid(cluster_units),
            'units': tuple(cluster_units),
        }
        for cluster_units in buckets.values()
    ]
    remaining_clusters.sort(
        key=lambda item: (
            item['centroid'][1],
            item['centroid'][0],
            min(unit.original_index for unit in item['units']),
        )
    )

    ordered: list[_DrawUnit] = []
    current_point: Point2D | None = None
    while remaining_clusters:
        if current_point is None:
            chosen_index = 0
        else:
            chosen_index = min(
                range(len(remaining_clusters)),
                key=lambda index: (
                    _distance(current_point, remaining_clusters[index]['centroid']),
                    remaining_clusters[index]['centroid'][1],
                    remaining_clusters[index]['centroid'][0],
                ),
            )
        cluster = remaining_clusters.pop(chosen_index)
        cluster_order = _reorder_units(cluster['units'], start_point=current_point)
        ordered.extend(cluster_order)
        current_point = cluster_order[-1].end

    return tuple(ordered)


def _classify_hatch_orientation(unit: _DrawUnit) -> tuple[str, float] | None:
    if not unit.commands or not all(isinstance(command, LineSegment) for command in unit.commands):
        return None
    heading = math.atan2(unit.end[1] - unit.start[1], unit.end[0] - unit.start[0])
    heading_abs = abs(math.degrees(math.atan2(math.sin(heading), math.cos(heading))))
    heading_abs = min(heading_abs, abs(180.0 - heading_abs))
    if heading_abs <= 18.0:
        return 'horizontal', 0.5 * (unit.start[1] + unit.end[1])
    if abs(heading_abs - 90.0) <= 18.0:
        return 'vertical', 0.5 * (unit.start[0] + unit.end[0])
    return None


def _orient_unit_for_axis(unit: _DrawUnit, *, axis: str, forward: bool) -> _DrawUnit:
    if axis == 'horizontal':
        left_to_right = unit.start[0] <= unit.end[0]
        return unit if left_to_right == forward else _reverse_unit(unit)
    top_to_bottom = unit.start[1] <= unit.end[1]
    return unit if top_to_bottom == forward else _reverse_unit(unit)


def _hatch_order_units(units: tuple[_DrawUnit, ...]) -> tuple[tuple[_DrawUnit, ...], bool]:
    if len(units) < 3:
        return units, False
    classified: list[tuple[_DrawUnit, str, float]] = []
    for unit in units:
        classification = _classify_hatch_orientation(unit)
        if classification is None:
            continue
        axis, coord = classification
        classified.append((unit, axis, coord))
    if len(classified) < 3:
        return units, False

    horizontal = [item for item in classified if item[1] == 'horizontal']
    vertical = [item for item in classified if item[1] == 'vertical']
    dominant = horizontal if len(horizontal) >= len(vertical) else vertical
    if len(dominant) < 3:
        return units, False

    dominant_axis = dominant[0][1]
    dominant_ordered = sorted(dominant, key=lambda item: (item[2], min(item[0].start[0], item[0].end[0]), item[0].original_index))
    hatch_units: list[_DrawUnit] = []
    for index, (unit, _, _) in enumerate(dominant_ordered):
        hatch_units.append(_orient_unit_for_axis(unit, axis=dominant_axis, forward=(index % 2 == 0)))

    dominant_indices = {item[0].original_index for item in dominant_ordered}
    remaining = tuple(unit for unit in units if unit.original_index not in dominant_indices)
    if remaining:
        hatch_units.extend(_reorder_units(remaining))
    return tuple(hatch_units), True


def _join_touching_units(
    units: tuple[_DrawUnit, ...],
    *,
    join_tolerance_m: float,
) -> tuple[tuple[_DrawUnit, ...], int]:
    if len(units) <= 1:
        return units, 0
    joined: list[_DrawUnit] = [units[0]]
    join_count = 0
    for unit in units[1:]:
        if _approximately_equal(joined[-1].end, unit.start, eps=join_tolerance_m):
            joined[-1] = _DrawUnit(
                commands=joined[-1].commands + unit.commands,
                original_index=joined[-1].original_index,
            )
            join_count += 1
            continue
        joined.append(unit)
    return tuple(joined), join_count


def _cleanup_transport_commands(
    plan: CanonicalPathPlan,
    *,
    merge_travel_moves: bool,
    travel_eps_m: float,
) -> tuple[CanonicalPathPlan, int]:
    commands: list[CanonicalCommand] = []
    merged_travel_moves = 0
    for command in plan.commands:
        if isinstance(command, TravelMove):
            if _primitive_length(command) <= travel_eps_m:
                merged_travel_moves += 1
                continue
            if (
                merge_travel_moves
                and commands
                and isinstance(commands[-1], TravelMove)
                and _approximately_equal(commands[-1].end, command.start, eps=travel_eps_m)
            ):
                commands[-1] = TravelMove(start=commands[-1].start, end=command.end)
                merged_travel_moves += 1
                continue
        if isinstance(command, PenUp) and commands and isinstance(commands[-1], PenUp):
            continue
        if isinstance(command, PenDown) and commands and isinstance(commands[-1], PenDown):
            continue
        commands.append(command)
    if not commands:
        return plan, merged_travel_moves
    return (
        CanonicalPathPlan(
            frame=plan.frame,
            theta_ref=plan.theta_ref,
            commands=tuple(commands),
        ),
        merged_travel_moves,
    )


def _rebuild_plan(
    units: tuple[_DrawUnit, ...],
    *,
    frame: str,
    theta_ref: float,
) -> CanonicalPathPlan:
    commands: list[CanonicalCommand] = []
    previous_end: Point2D | None = None
    pen_is_down = False

    for unit in units:
        if previous_end is not None and not _approximately_equal(previous_end, unit.start):
            if pen_is_down:
                commands.append(PenUp())
                pen_is_down = False
            commands.append(TravelMove(start=previous_end, end=unit.start))
        if not pen_is_down:
            commands.append(PenDown())
            pen_is_down = True
        commands.extend(unit.commands)
        previous_end = unit.end

    if pen_is_down:
        commands.append(PenUp())
    if not commands:
        raise ValueError('No drawable units remain after canonical optimization.')
    return CanonicalPathPlan(
        frame=frame,
        theta_ref=float(theta_ref),
        commands=tuple(commands),
    )


def optimize_canonical_plan(
    plan: CanonicalPathPlan,
    *,
    policy: CanonicalOptimizationPolicy | None = None,
) -> CanonicalOptimizationResult:
    active_policy = policy or CanonicalOptimizationPolicy()
    original_units = _extract_draw_units(plan)
    if not original_units:
        return CanonicalOptimizationResult(
            plan=plan,
            stats=CanonicalOptimizationStats(
                policy_label=active_policy.label,
                original_command_count=plan.command_count,
                optimized_command_count=plan.command_count,
                original_unit_count=0,
                optimized_unit_count=0,
                original_travel_length_m=0.0,
                optimized_travel_length_m=0.0,
                merged_line_segments=0,
                merged_curve_segments=0,
                merged_travel_moves=0,
                pruned_primitives=0,
                fitted_arc_segments=0,
                removed_duplicate_units=0,
                joined_units=0,
                hatch_reordered_units=False,
                reordered_units=False,
            ),
        )

    optimized_units: list[_DrawUnit] = []
    pruned_primitives = 0
    merged_line_segments = 0
    merged_curve_segments = 0
    fitted_arc_segments = 0
    for unit in original_units:
        optimized_unit, pruned_count, merged_count, merged_curve_count, fitted_arc_count = _optimize_unit(
            unit,
            policy=active_policy,
        )
        pruned_primitives += pruned_count
        merged_line_segments += merged_count
        merged_curve_segments += merged_curve_count
        fitted_arc_segments += fitted_arc_count
        if optimized_unit is not None:
            optimized_units.append(optimized_unit)

    if not optimized_units:
        return CanonicalOptimizationResult(
            plan=plan,
            stats=CanonicalOptimizationStats(
                policy_label=active_policy.label,
                original_command_count=plan.command_count,
                optimized_command_count=plan.command_count,
                original_unit_count=len(original_units),
                optimized_unit_count=len(original_units),
                original_travel_length_m=_travel_length(original_units),
                optimized_travel_length_m=_travel_length(original_units),
                merged_line_segments=merged_line_segments,
                merged_curve_segments=merged_curve_segments,
                merged_travel_moves=0,
                pruned_primitives=pruned_primitives,
                fitted_arc_segments=fitted_arc_segments,
                removed_duplicate_units=0,
                joined_units=0,
                hatch_reordered_units=False,
                reordered_units=False,
            ),
        )

    candidate_units = tuple(optimized_units)
    removed_duplicate_units = 0
    if active_policy.remove_duplicate_units:
        candidate_units, removed_duplicate_units = _dedupe_units(
            candidate_units,
            precision_m=active_policy.dedupe_precision_m,
        )

    reordered_units = candidate_units
    did_reorder = False
    hatch_reordered = False
    if active_policy.enable_hatch_ordering:
        hatch_candidate, hatch_applied = _hatch_order_units(candidate_units)
        if hatch_applied and _travel_length(hatch_candidate) < _travel_length(reordered_units) - _EPS:
            reordered_units = hatch_candidate
            hatch_reordered = True
            did_reorder = hatch_candidate != candidate_units
    if active_policy.reorder_units:
        candidate_travel_length = _travel_length(reordered_units)
        if active_policy.cluster_units and not hatch_reordered:
            reordered_candidate = _cluster_reorder_units(
                reordered_units,
                cell_size_m=active_policy.cluster_cell_size_m,
            )
        else:
            reordered_candidate = _reorder_units(reordered_units)
        reordered_travel_length = _travel_length(reordered_candidate)
        if reordered_travel_length < candidate_travel_length - _EPS:
            reordered_units = reordered_candidate
            did_reorder = reordered_units != candidate_units

    joined_units, joined_count = _join_touching_units(
        reordered_units,
        join_tolerance_m=active_policy.merge_distance_tolerance_m,
    )
    rebuilt_plan = _rebuild_plan(
        joined_units,
        frame=plan.frame,
        theta_ref=plan.theta_ref,
    )
    optimized_plan, merged_travel_moves = _cleanup_transport_commands(
        rebuilt_plan,
        merge_travel_moves=active_policy.merge_travel_moves,
        travel_eps_m=max(active_policy.merge_distance_tolerance_m, active_policy.tiny_primitive_m),
    )
    return CanonicalOptimizationResult(
        plan=optimized_plan,
        stats=CanonicalOptimizationStats(
            policy_label=active_policy.label,
            original_command_count=plan.command_count,
            optimized_command_count=optimized_plan.command_count,
            original_unit_count=len(original_units),
            optimized_unit_count=len(joined_units),
            original_travel_length_m=_travel_length(original_units),
            optimized_travel_length_m=_travel_length(joined_units),
            merged_line_segments=merged_line_segments,
            merged_curve_segments=merged_curve_segments,
            merged_travel_moves=merged_travel_moves,
            pruned_primitives=pruned_primitives,
            fitted_arc_segments=fitted_arc_segments,
            removed_duplicate_units=removed_duplicate_units,
            joined_units=joined_count,
            hatch_reordered_units=hatch_reordered,
            reordered_units=did_reorder,
        ),
    )
