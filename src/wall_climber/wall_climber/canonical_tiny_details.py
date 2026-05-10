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
_DRAW_COMMAND_TYPES = (LineSegment, ArcSegment, QuadraticBezier, CubicBezier)


@dataclass(frozen=True)
class TinyDetailExpansion:
    plan: CanonicalPathPlan
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _PassthroughChunk:
    commands: tuple[CanonicalCommand, ...]


@dataclass(frozen=True)
class _DrawUnit:
    commands: tuple[CanonicalCommand, ...]


@dataclass(frozen=True)
class _DrawUnitGeometry:
    length_m: float
    bounds: tuple[float, float, float, float]
    center: Point2D
    approach_start: Point2D
    first_point: Point2D
    last_point: Point2D
    max_span_m: float


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _bounds_gap_m(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_x_min, first_x_max, first_y_min, first_y_max = first
    second_x_min, second_x_max, second_y_min, second_y_max = second
    dx = max(0.0, max(second_x_min - first_x_max, first_x_min - second_x_max))
    dy = max(0.0, max(second_y_min - first_y_max, first_y_min - second_y_max))
    return math.hypot(dx, dy)


def _arc_point(segment: ArcSegment, ratio: float) -> Point2D:
    angle = float(segment.start_angle_rad) + float(segment.sweep_angle_rad) * float(ratio)
    return (
        float(segment.center[0]) + float(segment.radius) * math.cos(angle),
        float(segment.center[1]) + float(segment.radius) * math.sin(angle),
    )


def _command_points(command: CanonicalCommand) -> tuple[Point2D, ...]:
    if isinstance(command, (PenUp, PenDown)):
        return ()
    if isinstance(command, TravelMove):
        return (command.start, command.end)
    if isinstance(command, LineSegment):
        return (command.start, command.end)
    if isinstance(command, ArcSegment):
        samples = max(4, min(24, int(math.ceil(abs(float(command.sweep_angle_rad)) / 0.35))))
        return tuple(_arc_point(command, index / samples) for index in range(samples + 1))
    if isinstance(command, QuadraticBezier):
        return (command.start, command.control, command.end)
    if isinstance(command, CubicBezier):
        return (command.start, command.control1, command.control2, command.end)
    return ()


def _draw_command_length(command: CanonicalCommand) -> float:
    if isinstance(command, LineSegment):
        return _distance(command.start, command.end)
    if isinstance(command, ArcSegment):
        return abs(float(command.radius) * float(command.sweep_angle_rad))
    if isinstance(command, QuadraticBezier):
        return _distance(command.start, command.control) + _distance(command.control, command.end)
    if isinstance(command, CubicBezier):
        return (
            _distance(command.start, command.control1)
            + _distance(command.control1, command.control2)
            + _distance(command.control2, command.end)
        )
    return 0.0


def _unit_geometry(unit: _DrawUnit) -> _DrawUnitGeometry | None:
    draw_commands = tuple(command for command in unit.commands if isinstance(command, _DRAW_COMMAND_TYPES))
    if not draw_commands:
        return None

    points: list[Point2D] = []
    length_m = 0.0
    first_point: Point2D | None = None
    last_point: Point2D | None = None
    approach_start: Point2D | None = None
    for command in unit.commands:
        if isinstance(command, TravelMove):
            approach_start = command.start
        if isinstance(command, PenDown):
            break
    for command in draw_commands:
        command_points = _command_points(command)
        if command_points and first_point is None:
            first_point = command_points[0]
        if command_points:
            last_point = command_points[-1]
        points.extend(command_points)
        length_m += _draw_command_length(command)

    if not points or first_point is None or last_point is None:
        return None

    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    width = x_max - x_min
    height = y_max - y_min
    return _DrawUnitGeometry(
        length_m=float(length_m),
        bounds=(x_min, x_max, y_min, y_max),
        center=((x_min + x_max) * 0.5, (y_min + y_max) * 0.5),
        approach_start=approach_start or first_point,
        first_point=first_point,
        last_point=last_point,
        max_span_m=max(width, height),
    )


def _split_units(commands: tuple[CanonicalCommand, ...]) -> tuple[_PassthroughChunk | _DrawUnit, ...]:
    chunks: list[_PassthroughChunk | _DrawUnit] = []
    pending: list[CanonicalCommand] = []
    index = 0

    while index < len(commands):
        command = commands[index]
        if not isinstance(command, PenDown):
            pending.append(command)
            index += 1
            continue

        lead_travel: TravelMove | None = None
        if pending and isinstance(pending[-1], TravelMove):
            lead_travel = pending.pop()  # keep the pen-up approach travel with the draw unit
        if pending:
            chunks.append(_PassthroughChunk(commands=tuple(pending)))
            pending = []

        unit_commands: list[CanonicalCommand] = []
        if lead_travel is not None:
            unit_commands.append(lead_travel)
        unit_commands.append(command)
        index += 1
        while index < len(commands):
            unit_commands.append(commands[index])
            if isinstance(commands[index], PenUp):
                index += 1
                break
            index += 1
        chunks.append(_DrawUnit(commands=tuple(unit_commands)))

    if pending:
        chunks.append(_PassthroughChunk(commands=tuple(pending)))
    return tuple(chunks)


def _clamp_center(center: Point2D, *, half_size: float, bounds: dict[str, float] | None) -> Point2D:
    if not bounds:
        return (float(center[0]), float(center[1]))
    x_min = float(bounds.get('x_min', center[0] - half_size))
    x_max = float(bounds.get('x_max', center[0] + half_size))
    y_min = float(bounds.get('y_min', center[1] - half_size))
    y_max = float(bounds.get('y_max', center[1] + half_size))
    if x_max - x_min < 2.0 * half_size or y_max - y_min < 2.0 * half_size:
        return (float(center[0]), float(center[1]))
    return (
        min(max(float(center[0]), x_min + half_size), x_max - half_size),
        min(max(float(center[1]), y_min + half_size), y_max - half_size),
    )


def _maybe_travel(start: Point2D, end: Point2D) -> tuple[TravelMove, ...]:
    if _distance(start, end) <= _EPS:
        return ()
    return (TravelMove(start=start, end=end),)


def _micro_cross_commands(geometry: _DrawUnitGeometry, *, size_m: float, bounds: dict[str, float] | None) -> tuple[CanonicalCommand, ...]:
    half = max(1.0e-5, float(size_m) * 0.5)
    cx, cy = _clamp_center(geometry.center, half_size=half, bounds=bounds)
    h_start = (cx - half, cy)
    h_end = (cx + half, cy)
    v_start = (cx, cy - half)
    v_end = (cx, cy + half)
    commands: list[CanonicalCommand] = []
    commands.extend(_maybe_travel(geometry.approach_start, h_start))
    commands.extend((PenDown(), LineSegment(start=h_start, end=h_end), PenUp()))
    commands.extend(_maybe_travel(h_end, v_start))
    commands.extend((PenDown(), LineSegment(start=v_start, end=v_end), PenUp()))
    commands.extend(_maybe_travel(v_end, geometry.last_point))
    return tuple(commands)


def _micro_loop_commands(geometry: _DrawUnitGeometry, *, size_m: float, bounds: dict[str, float] | None) -> tuple[CanonicalCommand, ...]:
    half = max(1.0e-5, float(size_m) * 0.5)
    cx, cy = _clamp_center(geometry.center, half_size=half, bounds=bounds)
    p1 = (cx - half, cy - half)
    p2 = (cx + half, cy - half)
    p3 = (cx + half, cy + half)
    p4 = (cx - half, cy + half)
    commands: list[CanonicalCommand] = []
    commands.extend(_maybe_travel(geometry.approach_start, p1))
    commands.append(PenDown())
    commands.extend(
        (
            LineSegment(start=p1, end=p2),
            LineSegment(start=p2, end=p3),
            LineSegment(start=p3, end=p4),
            LineSegment(start=p4, end=p1),
        )
    )
    commands.append(PenUp())
    commands.extend(_maybe_travel(p1, geometry.last_point))
    return tuple(commands)


def expand_tiny_details_in_canonical_plan(
    plan: CanonicalPathPlan,
    *,
    preserve: bool = True,
    minimum_drawable_feature_m: float = 0.0045,
    candidate_max_feature_m: float | None = None,
    expand_mode: str = 'micro_cross',
    max_expansions: int = 512,
    context_radius_m: float | None = 0.08,
    bounds: dict[str, float] | None = None,
) -> TinyDetailExpansion:
    """Expand very small drawable units into geometry the robot can physically draw."""

    min_feature = max(1.0e-5, float(minimum_drawable_feature_m))
    candidate_limit = (
        max(1.0e-5, float(candidate_max_feature_m))
        if candidate_max_feature_m is not None
        else min_feature * 0.75
    )
    expansion_limit = max(0, int(max_expansions))
    context_radius = (
        None
        if context_radius_m is None
        else max(0.0, float(context_radius_m))
    )
    normalized_mode = str(expand_mode or 'micro_cross').strip().lower()
    if normalized_mode not in {'micro_cross', 'micro_loop'}:
        raise ValueError("expand_mode must be 'micro_cross' or 'micro_loop'.")

    metrics: dict[str, Any] = {
        'preserve_tiny_details': bool(preserve),
        'tiny_detail_expand_mode': normalized_mode,
        'minimum_drawable_feature_m': float(min_feature),
        'tiny_detail_candidate_max_feature_m': float(candidate_limit),
        'tiny_detail_max_expansions': int(expansion_limit),
        'tiny_detail_context_radius_m': None if context_radius is None else float(context_radius),
        'tiny_details_detected': 0,
        'tiny_details_preserved': 0,
        'tiny_details_expanded': 0,
        'tiny_details_skipped_by_limit': 0,
        'tiny_details_skipped_as_isolated': 0,
        'tiny_details_expansion_added_commands': 0,
    }
    if not preserve:
        return TinyDetailExpansion(plan=plan, metrics=metrics)

    chunks = _split_units(tuple(plan.commands))
    unit_geometries = {
        index: _unit_geometry(chunk)
        for index, chunk in enumerate(chunks)
        if isinstance(chunk, _DrawUnit)
    }
    context_geometries = tuple(
        geometry
        for geometry in unit_geometries.values()
        if geometry is not None and geometry.max_span_m > candidate_limit
    )
    rebuilt: list[CanonicalCommand] = []
    expanded_count = 0
    detected_count = 0
    isolated_count = 0

    for index, chunk in enumerate(chunks):
        if isinstance(chunk, _PassthroughChunk):
            rebuilt.extend(chunk.commands)
            continue

        geometry = unit_geometries.get(index)
        is_tiny = geometry is not None and geometry.max_span_m <= candidate_limit
        if not is_tiny or geometry is None:
            rebuilt.extend(chunk.commands)
            continue

        detected_count += 1
        if context_radius is not None and context_geometries:
            nearest_context_gap = min(
                _bounds_gap_m(geometry.bounds, context_geometry.bounds)
                for context_geometry in context_geometries
            )
            if nearest_context_gap > context_radius:
                isolated_count += 1
                rebuilt.extend(chunk.commands)
                continue
        if expanded_count >= expansion_limit:
            rebuilt.extend(chunk.commands)
            continue

        if normalized_mode == 'micro_loop':
            replacement = _micro_loop_commands(geometry, size_m=min_feature, bounds=bounds)
        else:
            replacement = _micro_cross_commands(geometry, size_m=min_feature, bounds=bounds)
        rebuilt.extend(replacement)
        expanded_count += 1
        metrics['tiny_details_expansion_added_commands'] = int(
            metrics['tiny_details_expansion_added_commands']
        ) + max(0, len(replacement) - len(chunk.commands))

    if expanded_count == 0:
        metrics['tiny_details_detected'] = int(detected_count)
        metrics['tiny_details_skipped_as_isolated'] = int(isolated_count)
        metrics['tiny_details_skipped_by_limit'] = max(0, detected_count - isolated_count - expanded_count)
        return TinyDetailExpansion(plan=plan, metrics=metrics)

    metrics['tiny_details_detected'] = int(detected_count)
    metrics['tiny_details_preserved'] = int(expanded_count)
    metrics['tiny_details_expanded'] = int(expanded_count)
    metrics['tiny_details_skipped_as_isolated'] = int(isolated_count)
    metrics['tiny_details_skipped_by_limit'] = max(0, detected_count - isolated_count - expanded_count)
    return TinyDetailExpansion(
        plan=CanonicalPathPlan(
            frame=plan.frame,
            theta_ref=plan.theta_ref,
            commands=tuple(rebuilt),
        ),
        metrics=metrics,
    )
