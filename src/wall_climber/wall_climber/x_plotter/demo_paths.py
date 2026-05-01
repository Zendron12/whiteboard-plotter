from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from wall_climber.x_plotter.frame_config import BoardFrameConfig, load_board_frame_config
from wall_climber.x_plotter.primitive_sampler import (
    CUBIC_BEZIER,
    LINE_SEGMENT,
    PEN_DOWN,
    PEN_UP,
    QUADRATIC_BEZIER,
    TRAVEL_MOVE,
)


@dataclass
class _FallbackBoardPoint:
    x: float = 0.0
    y: float = 0.0


@dataclass
class _FallbackPathPrimitive:
    PEN_UP = PEN_UP
    PEN_DOWN = PEN_DOWN
    TRAVEL_MOVE = TRAVEL_MOVE
    LINE_SEGMENT = LINE_SEGMENT
    ARC_SEGMENT = 4
    QUADRATIC_BEZIER = QUADRATIC_BEZIER
    CUBIC_BEZIER = CUBIC_BEZIER

    type: int = PEN_UP
    start: _FallbackBoardPoint = field(default_factory=_FallbackBoardPoint)
    end: _FallbackBoardPoint = field(default_factory=_FallbackBoardPoint)
    control1: _FallbackBoardPoint = field(default_factory=_FallbackBoardPoint)
    control2: _FallbackBoardPoint = field(default_factory=_FallbackBoardPoint)
    center: _FallbackBoardPoint = field(default_factory=_FallbackBoardPoint)
    radius: float = 0.0
    start_angle_rad: float = 0.0
    sweep_angle_rad: float = 0.0
    clockwise: bool = False
    pen_down: bool = False


@dataclass
class _FallbackPrimitivePathPlan:
    frame: str = 'board'
    theta_ref: float = 0.0
    primitives: list[_FallbackPathPrimitive] = field(default_factory=list)


def _message_types():
    try:
        from wall_climber_interfaces.msg import BoardPoint, PathPrimitive, PrimitivePathPlan
    except ImportError:
        return _FallbackBoardPoint, _FallbackPathPrimitive, _FallbackPrimitivePathPlan
    return BoardPoint, PathPrimitive, PrimitivePathPlan


def _point(x: float, y: float):
    BoardPoint, _, _ = _message_types()
    return BoardPoint(x=float(x), y=float(y))


def _primitive(primitive_type: int):
    _, PathPrimitive, _ = _message_types()
    primitive = PathPrimitive()
    primitive.type = int(primitive_type)
    return primitive


def _new_plan():
    _, _, PrimitivePathPlan = _message_types()
    plan = PrimitivePathPlan()
    plan.frame = 'board'
    plan.theta_ref = 0.0
    return plan


def _append_pen(plan, down: bool) -> None:
    _, PathPrimitive, _ = _message_types()
    plan.primitives.append(_primitive(PathPrimitive.PEN_DOWN if down else PathPrimitive.PEN_UP))


def _append_travel(plan, start: tuple[float, float], end: tuple[float, float]) -> None:
    _, PathPrimitive, _ = _message_types()
    primitive = _primitive(PathPrimitive.TRAVEL_MOVE)
    primitive.start = _point(*start)
    primitive.end = _point(*end)
    primitive.pen_down = False
    plan.primitives.append(primitive)


def _append_line(plan, start: tuple[float, float], end: tuple[float, float]) -> None:
    _, PathPrimitive, _ = _message_types()
    primitive = _primitive(PathPrimitive.LINE_SEGMENT)
    primitive.start = _point(*start)
    primitive.end = _point(*end)
    primitive.pen_down = True
    plan.primitives.append(primitive)


def _safe_point(point: tuple[float, float], frame: BoardFrameConfig) -> tuple[float, float]:
    return frame.clamp_point(point[0], point[1])


def _safe_points(points: Iterable[tuple[float, float]], frame: BoardFrameConfig) -> tuple[tuple[float, float], ...]:
    return tuple(_safe_point(point, frame) for point in points)


def _append_stroke(plan, points: tuple[tuple[float, float], ...], *, cursor: tuple[float, float] | None) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError('A demo stroke needs at least two points.')
    if cursor is not None and cursor != points[0]:
        _append_travel(plan, cursor, points[0])
    _append_pen(plan, True)
    for start, end in zip(points[:-1], points[1:]):
        _append_line(plan, start, end)
    _append_pen(plan, False)
    return points[-1]


def demo_line_points(frame: BoardFrameConfig) -> tuple[tuple[float, float], ...]:
    y = frame.drawable_y_min + 0.28 * (frame.drawable_y_max - frame.drawable_y_min)
    return _safe_points(
        (
            (frame.drawable_x_min + 0.20 * (frame.drawable_x_max - frame.drawable_x_min), y),
            (frame.drawable_x_min + 0.43 * (frame.drawable_x_max - frame.drawable_x_min), y),
        ),
        frame,
    )


def demo_square_points(frame: BoardFrameConfig) -> tuple[tuple[float, float], ...]:
    x0 = frame.drawable_x_min + 0.50 * (frame.drawable_x_max - frame.drawable_x_min)
    y0 = frame.drawable_y_min + 0.22 * (frame.drawable_y_max - frame.drawable_y_min)
    size = min(0.55, 0.22 * min(frame.drawable_x_max - frame.drawable_x_min, frame.drawable_y_max - frame.drawable_y_min))
    return _safe_points(
        (
            (x0, y0),
            (x0 + size, y0),
            (x0 + size, y0 + size),
            (x0, y0 + size),
            (x0, y0),
        ),
        frame,
    )


def demo_triangle_points(frame: BoardFrameConfig) -> tuple[tuple[float, float], ...]:
    cx = frame.drawable_x_min + 0.78 * (frame.drawable_x_max - frame.drawable_x_min)
    cy = frame.drawable_y_min + 0.43 * (frame.drawable_y_max - frame.drawable_y_min)
    w = min(0.70, 0.26 * (frame.drawable_x_max - frame.drawable_x_min))
    h = min(0.62, 0.25 * (frame.drawable_y_max - frame.drawable_y_min))
    return _safe_points(
        (
            (cx, cy - h * 0.50),
            (cx + w * 0.50, cy + h * 0.50),
            (cx - w * 0.50, cy + h * 0.50),
            (cx, cy - h * 0.50),
        ),
        frame,
    )


def build_demo_path(name: str, *, frame: BoardFrameConfig | None = None):
    frame = frame or load_board_frame_config()
    normalized = str(name or 'line_square_triangle').strip().lower()
    plan = _new_plan()
    cursor: tuple[float, float] | None = None

    if normalized == 'line':
        cursor = _append_stroke(plan, demo_line_points(frame), cursor=cursor)
    elif normalized == 'square':
        cursor = _append_stroke(plan, demo_square_points(frame), cursor=cursor)
    elif normalized == 'triangle':
        cursor = _append_stroke(plan, demo_triangle_points(frame), cursor=cursor)
    elif normalized in {'line_square_triangle', 'all', 'demo'}:
        for points in (demo_line_points(frame), demo_square_points(frame), demo_triangle_points(frame)):
            cursor = _append_stroke(plan, points, cursor=cursor)
    elif normalized == 'off':
        raise ValueError("demo path 'off' cannot be built")
    else:
        raise ValueError(f'Unsupported X plotter demo path: {name!r}')

    return plan


VALID_DEMO_PATHS = ('off', 'line', 'square', 'triangle', 'line_square_triangle')

