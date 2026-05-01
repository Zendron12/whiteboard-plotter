from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


Point2D = tuple[float, float]


PEN_UP = 0
PEN_DOWN = 1
TRAVEL_MOVE = 2
LINE_SEGMENT = 3
ARC_SEGMENT = 4
QUADRATIC_BEZIER = 5
CUBIC_BEZIER = 6


@dataclass(frozen=True)
class SamplingPolicy:
    draw_step_m: float = 0.020
    travel_step_m: float = 0.040
    curve_step_m: float = 0.020
    max_arc_angle_step_rad: float = 0.12


@dataclass(frozen=True)
class SampledPrimitivePath:
    draw: bool
    points: tuple[Point2D, ...]
    primitive_index: int


def _primitive_type_codes() -> dict[str, int]:
    try:
        from wall_climber_interfaces.msg import PathPrimitive
    except ImportError:
        return {
            'PEN_UP': PEN_UP,
            'PEN_DOWN': PEN_DOWN,
            'TRAVEL_MOVE': TRAVEL_MOVE,
            'LINE_SEGMENT': LINE_SEGMENT,
            'ARC_SEGMENT': ARC_SEGMENT,
            'QUADRATIC_BEZIER': QUADRATIC_BEZIER,
            'CUBIC_BEZIER': CUBIC_BEZIER,
        }
    return {
        'PEN_UP': int(PathPrimitive.PEN_UP),
        'PEN_DOWN': int(PathPrimitive.PEN_DOWN),
        'TRAVEL_MOVE': int(PathPrimitive.TRAVEL_MOVE),
        'LINE_SEGMENT': int(PathPrimitive.LINE_SEGMENT),
        'ARC_SEGMENT': int(PathPrimitive.ARC_SEGMENT),
        'QUADRATIC_BEZIER': int(PathPrimitive.QUADRATIC_BEZIER),
        'CUBIC_BEZIER': int(PathPrimitive.CUBIC_BEZIER),
    }


TYPE_CODES = _primitive_type_codes()


def _point(point_msg: Any, label: str) -> Point2D:
    try:
        point = (float(point_msg.x), float(point_msg.y))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f'{label} must expose finite x/y floats') from exc
    if not math.isfinite(point[0]) or not math.isfinite(point[1]):
        raise ValueError(f'{label} must expose finite x/y floats')
    return point


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _append_point(points: list[Point2D], point: Point2D, *, eps: float = 1.0e-9) -> None:
    if points and _distance(points[-1], point) <= eps:
        return
    points.append((float(point[0]), float(point[1])))


def _sample_line(start: Point2D, end: Point2D, step_m: float) -> tuple[Point2D, ...]:
    length = _distance(start, end)
    if length <= 1.0e-9:
        return ()
    subdivisions = max(1, int(math.ceil(length / max(1.0e-4, float(step_m)))))
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        ratio = index / subdivisions
        _append_point(
            points,
            (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            ),
        )
    return tuple(points)


def _normalized_sweep(primitive: Any) -> float:
    sweep = float(primitive.sweep_angle_rad)
    clockwise = bool(primitive.clockwise)
    if clockwise and sweep > 0.0:
        return -sweep
    if not clockwise and sweep < 0.0:
        return -sweep
    return sweep


def _sample_arc(primitive: Any, policy: SamplingPolicy) -> tuple[Point2D, ...]:
    center = _point(primitive.center, 'ArcSegment.center')
    radius = float(primitive.radius)
    start_angle = float(primitive.start_angle_rad)
    sweep = _normalized_sweep(primitive)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError('ArcSegment.radius must be finite and > 0')
    if not math.isfinite(start_angle) or not math.isfinite(sweep):
        raise ValueError('ArcSegment angles must be finite')
    arc_length = abs(radius * sweep)
    by_length = int(math.ceil(arc_length / max(1.0e-4, float(policy.curve_step_m))))
    by_angle = int(math.ceil(abs(sweep) / max(1.0e-4, float(policy.max_arc_angle_step_rad))))
    subdivisions = max(1, by_length, by_angle)
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        ratio = index / subdivisions
        angle = start_angle + sweep * ratio
        _append_point(
            points,
            (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            ),
        )
    return tuple(points)


def _sample_quadratic(primitive: Any, policy: SamplingPolicy) -> tuple[Point2D, ...]:
    start = _point(primitive.start, 'QuadraticBezier.start')
    control = _point(primitive.control1, 'QuadraticBezier.control1')
    end = _point(primitive.end, 'QuadraticBezier.end')
    estimated_length = _distance(start, control) + _distance(control, end)
    subdivisions = max(2, int(math.ceil(estimated_length / max(1.0e-4, float(policy.curve_step_m)))))
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        t = index / subdivisions
        omt = 1.0 - t
        _append_point(
            points,
            (
                omt * omt * start[0] + 2.0 * omt * t * control[0] + t * t * end[0],
                omt * omt * start[1] + 2.0 * omt * t * control[1] + t * t * end[1],
            ),
        )
    return tuple(points)


def _sample_cubic(primitive: Any, policy: SamplingPolicy) -> tuple[Point2D, ...]:
    start = _point(primitive.start, 'CubicBezier.start')
    control1 = _point(primitive.control1, 'CubicBezier.control1')
    control2 = _point(primitive.control2, 'CubicBezier.control2')
    end = _point(primitive.end, 'CubicBezier.end')
    estimated_length = _distance(start, control1) + _distance(control1, control2) + _distance(control2, end)
    subdivisions = max(3, int(math.ceil(estimated_length / max(1.0e-4, float(policy.curve_step_m)))))
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        t = index / subdivisions
        omt = 1.0 - t
        _append_point(
            points,
            (
                (omt ** 3 * start[0])
                + (3.0 * omt * omt * t * control1[0])
                + (3.0 * omt * t * t * control2[0])
                + (t ** 3 * end[0]),
                (omt ** 3 * start[1])
                + (3.0 * omt * omt * t * control1[1])
                + (3.0 * omt * t * t * control2[1])
                + (t ** 3 * end[1]),
            ),
        )
    return tuple(points)


def sample_primitive_path_plan(
    plan: Any,
    *,
    policy: SamplingPolicy | None = None,
) -> tuple[SampledPrimitivePath, ...]:
    active_policy = policy or SamplingPolicy()
    frame = str(getattr(plan, 'frame', ''))
    if frame != 'board':
        raise ValueError("PrimitivePathPlan.frame must be 'board'")
    primitives = tuple(getattr(plan, 'primitives', ()))
    if not primitives:
        raise ValueError('PrimitivePathPlan.primitives must not be empty')

    sampled: list[SampledPrimitivePath] = []
    pen_down = False
    for primitive_index, primitive in enumerate(primitives):
        primitive_type = int(getattr(primitive, 'type'))
        if primitive_type == TYPE_CODES['PEN_UP']:
            pen_down = False
            continue
        if primitive_type == TYPE_CODES['PEN_DOWN']:
            pen_down = True
            continue
        if primitive_type == TYPE_CODES['TRAVEL_MOVE']:
            pen_down = False
            points = _sample_line(
                _point(primitive.start, 'TravelMove.start'),
                _point(primitive.end, 'TravelMove.end'),
                active_policy.travel_step_m,
            )
            draw = False
        elif primitive_type == TYPE_CODES['LINE_SEGMENT']:
            pen_down = bool(primitive.pen_down)
            points = _sample_line(
                _point(primitive.start, 'LineSegment.start'),
                _point(primitive.end, 'LineSegment.end'),
                active_policy.draw_step_m if pen_down else active_policy.travel_step_m,
            )
            draw = pen_down
        elif primitive_type == TYPE_CODES['ARC_SEGMENT']:
            pen_down = bool(primitive.pen_down)
            points = _sample_arc(primitive, active_policy)
            draw = pen_down
        elif primitive_type == TYPE_CODES['QUADRATIC_BEZIER']:
            pen_down = bool(primitive.pen_down)
            points = _sample_quadratic(primitive, active_policy)
            draw = pen_down
        elif primitive_type == TYPE_CODES['CUBIC_BEZIER']:
            pen_down = bool(primitive.pen_down)
            points = _sample_cubic(primitive, active_policy)
            draw = pen_down
        else:
            raise ValueError(f'Unsupported PathPrimitive.type: {primitive_type}')

        if len(points) >= 2:
            sampled.append(SampledPrimitivePath(draw=draw, points=points, primitive_index=primitive_index))
    if not sampled:
        raise ValueError('PrimitivePathPlan produced no sampled paths')
    return tuple(sampled)

