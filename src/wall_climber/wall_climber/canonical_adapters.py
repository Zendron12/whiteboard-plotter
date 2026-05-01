from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

try:
    import wall_climber_geometry_cpp as _geometry_cpp
except ImportError:
    _geometry_cpp = None

from wall_climber.canonical_path import (
    ArcSegment,
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
class SampledPath:
    draw: bool
    points: tuple[Point2D, ...]


@dataclass(frozen=True)
class SamplingPolicy:
    curve_tolerance_m: float = 0.01
    draw_step_m: float | None = None
    travel_step_m: float | None = None
    max_heading_delta_rad: float | None = None
    label: str = 'custom'


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _approximately_equal(a: Point2D, b: Point2D, *, eps: float = _EPS) -> bool:
    return _distance(a, b) <= eps


def _rotate_point(point: Point2D, theta: float) -> Point2D:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return (
        point[0] * cos_theta - point[1] * sin_theta,
        point[0] * sin_theta + point[1] * cos_theta,
    )


def _append_sampled_point(points: list[Point2D], point: Point2D) -> None:
    if points and _approximately_equal(points[-1], point):
        return
    points.append((float(point[0]), float(point[1])))


def _sanitize_sampling_policy(
    sampling_policy: SamplingPolicy | None = None,
    *,
    curve_tolerance_m: float = 0.01,
) -> SamplingPolicy:
    policy = sampling_policy or SamplingPolicy(curve_tolerance_m=float(curve_tolerance_m))
    curve = max(1.0e-4, float(policy.curve_tolerance_m))
    draw_step = None if policy.draw_step_m is None else max(1.0e-4, float(policy.draw_step_m))
    travel_step = None if policy.travel_step_m is None else max(1.0e-4, float(policy.travel_step_m))
    heading = (
        None
        if policy.max_heading_delta_rad is None
        else max(1.0e-4, float(policy.max_heading_delta_rad))
    )
    return SamplingPolicy(
        curve_tolerance_m=curve,
        draw_step_m=draw_step,
        travel_step_m=travel_step,
        max_heading_delta_rad=heading,
        label=str(policy.label or 'custom'),
    )


def _resample_linear_path(
    start: Point2D,
    end: Point2D,
    *,
    step_m: float | None,
) -> tuple[Point2D, ...]:
    if step_m is None or _distance(start, end) <= step_m + _EPS:
        return (start, end)
    subdivisions = max(1, int(math.ceil(_distance(start, end) / step_m)))
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        ratio = index / subdivisions
        _append_sampled_point(
            points,
            (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            ),
        )
    return tuple(points)


def _sample_line(
    segment: LineSegment,
    *,
    step_m: float | None,
) -> tuple[Point2D, ...]:
    return _resample_linear_path(segment.start, segment.end, step_m=step_m)


def _sample_travel(
    segment: TravelMove,
    *,
    step_m: float | None,
) -> tuple[Point2D, ...]:
    return _resample_linear_path(segment.start, segment.end, step_m=step_m)


def _sample_arc(
    segment: ArcSegment,
    *,
    curve_tolerance_m: float,
    max_heading_delta_rad: float | None,
) -> tuple[Point2D, ...]:
    sweep = abs(float(segment.sweep_angle_rad))
    arc_length = float(segment.radius) * sweep
    step = max(float(curve_tolerance_m), 1.0e-4)
    subdivisions = max(1, int(math.ceil(arc_length / step)))
    if max_heading_delta_rad is not None and max_heading_delta_rad > 0.0:
        subdivisions = max(subdivisions, int(math.ceil(max(sweep, 1.0e-9) / max_heading_delta_rad)))
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        ratio = index / subdivisions
        angle = float(segment.start_angle_rad) + float(segment.sweep_angle_rad) * ratio
        _append_sampled_point(
            points,
            (
                float(segment.center[0]) + float(segment.radius) * math.cos(angle),
                float(segment.center[1]) + float(segment.radius) * math.sin(angle),
            ),
        )
    return tuple(points)


def _sample_quadratic(
    segment: QuadraticBezier,
    *,
    curve_tolerance_m: float,
    max_heading_delta_rad: float | None,
) -> tuple[Point2D, ...]:
    chord = _distance(segment.start, segment.end)
    control_span = _distance(segment.start, segment.control) + _distance(segment.control, segment.end)
    estimated = max(chord, control_span)
    step = max(float(curve_tolerance_m), 1.0e-4)
    subdivisions = max(2, int(math.ceil(estimated / step)))
    if max_heading_delta_rad is not None and max_heading_delta_rad > 0.0:
        start_heading = math.atan2(
            segment.control[1] - segment.start[1],
            segment.control[0] - segment.start[0],
        )
        end_heading = math.atan2(
            segment.end[1] - segment.control[1],
            segment.end[0] - segment.control[0],
        )
        heading_delta = abs(math.atan2(math.sin(end_heading - start_heading), math.cos(end_heading - start_heading)))
        subdivisions = max(subdivisions, int(math.ceil(max(heading_delta, 1.0e-9) / max_heading_delta_rad)))
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        t = index / subdivisions
        omt = 1.0 - t
        _append_sampled_point(
            points,
            (
                (omt * omt * segment.start[0])
                + (2.0 * omt * t * segment.control[0])
                + (t * t * segment.end[0]),
                (omt * omt * segment.start[1])
                + (2.0 * omt * t * segment.control[1])
                + (t * t * segment.end[1]),
            ),
        )
    return tuple(points)


def _sample_cubic(
    segment: CubicBezier,
    *,
    curve_tolerance_m: float,
    max_heading_delta_rad: float | None,
) -> tuple[Point2D, ...]:
    control_span = (
        _distance(segment.start, segment.control1)
        + _distance(segment.control1, segment.control2)
        + _distance(segment.control2, segment.end)
    )
    step = max(float(curve_tolerance_m), 1.0e-4)
    subdivisions = max(3, int(math.ceil(control_span / step)))
    if max_heading_delta_rad is not None and max_heading_delta_rad > 0.0:
        start_heading = math.atan2(
            segment.control1[1] - segment.start[1],
            segment.control1[0] - segment.start[0],
        )
        end_heading = math.atan2(
            segment.end[1] - segment.control2[1],
            segment.end[0] - segment.control2[0],
        )
        heading_delta = abs(math.atan2(math.sin(end_heading - start_heading), math.cos(end_heading - start_heading)))
        subdivisions = max(subdivisions, int(math.ceil(max(heading_delta, 1.0e-9) / max_heading_delta_rad)))
    points: list[Point2D] = []
    for index in range(subdivisions + 1):
        t = index / subdivisions
        omt = 1.0 - t
        _append_sampled_point(
            points,
            (
                (omt ** 3 * segment.start[0])
                + (3.0 * omt * omt * t * segment.control1[0])
                + (3.0 * omt * t * t * segment.control2[0])
                + (t ** 3 * segment.end[0]),
                (omt ** 3 * segment.start[1])
                + (3.0 * omt * omt * t * segment.control1[1])
                + (3.0 * omt * t * t * segment.control2[1])
                + (t ** 3 * segment.end[1]),
            ),
        )
    return tuple(points)


def _command_to_cpp_descriptor(command: object) -> dict[str, object]:
    if isinstance(command, PenUp):
        return {'type': 'pen_up'}
    if isinstance(command, PenDown):
        return {'type': 'pen_down'}
    if isinstance(command, TravelMove):
        return {
            'type': 'travel',
            'start': [float(command.start[0]), float(command.start[1])],
            'end': [float(command.end[0]), float(command.end[1])],
        }
    if isinstance(command, LineSegment):
        return {
            'type': 'line',
            'start': [float(command.start[0]), float(command.start[1])],
            'end': [float(command.end[0]), float(command.end[1])],
        }
    if isinstance(command, ArcSegment):
        return {
            'type': 'arc',
            'center': [float(command.center[0]), float(command.center[1])],
            'radius': float(command.radius),
            'start_angle_rad': float(command.start_angle_rad),
            'sweep_angle_rad': float(command.sweep_angle_rad),
        }
    if isinstance(command, QuadraticBezier):
        return {
            'type': 'quadratic',
            'start': [float(command.start[0]), float(command.start[1])],
            'control': [float(command.control[0]), float(command.control[1])],
            'end': [float(command.end[0]), float(command.end[1])],
        }
    if isinstance(command, CubicBezier):
        return {
            'type': 'cubic',
            'start': [float(command.start[0]), float(command.start[1])],
            'control1': [float(command.control1[0]), float(command.control1[1])],
            'control2': [float(command.control2[0]), float(command.control2[1])],
            'end': [float(command.end[0]), float(command.end[1])],
        }
    raise ValueError(f'Unsupported canonical command {type(command)!r}.')


def _zero_point_payload() -> dict[str, float]:
    return {'x': 0.0, 'y': 0.0}


def _point_payload(point: Point2D) -> dict[str, float]:
    return {'x': float(point[0]), 'y': float(point[1])}


def _arc_endpoint_payloads(segment: ArcSegment) -> tuple[dict[str, float], dict[str, float]]:
    start_angle = float(segment.start_angle_rad)
    end_angle = start_angle + float(segment.sweep_angle_rad)
    center_x = float(segment.center[0])
    center_y = float(segment.center[1])
    radius = float(segment.radius)
    return (
        {
            'x': center_x + radius * math.cos(start_angle),
            'y': center_y + radius * math.sin(start_angle),
        },
        {
            'x': center_x + radius * math.cos(end_angle),
            'y': center_y + radius * math.sin(end_angle),
        },
    )


def _plan_to_cpp_descriptor(plan: CanonicalPathPlan) -> dict[str, object]:
    return {
        'frame': str(plan.frame),
        'theta_ref': float(plan.theta_ref),
        'commands': [_command_to_cpp_descriptor(command) for command in plan.commands],
    }


def canonical_plan_to_primitive_path_plan(plan: CanonicalPathPlan) -> dict[str, object]:
    primitives: list[dict[str, object]] = []
    pen_down = False
    for command in plan.commands:
        primitive = {
            'type': '',
            'start': _zero_point_payload(),
            'end': _zero_point_payload(),
            'control1': _zero_point_payload(),
            'control2': _zero_point_payload(),
            'center': _zero_point_payload(),
            'radius': 0.0,
            'start_angle_rad': 0.0,
            'sweep_angle_rad': 0.0,
            'clockwise': False,
            'pen_down': False,
        }
        if isinstance(command, PenUp):
            pen_down = False
            primitive['type'] = 'PEN_UP'
        elif isinstance(command, PenDown):
            pen_down = True
            primitive['type'] = 'PEN_DOWN'
            primitive['pen_down'] = True
        elif isinstance(command, TravelMove):
            primitive['type'] = 'TRAVEL_MOVE'
            primitive['start'] = _point_payload(command.start)
            primitive['end'] = _point_payload(command.end)
        elif isinstance(command, LineSegment):
            primitive['type'] = 'LINE_SEGMENT'
            primitive['start'] = _point_payload(command.start)
            primitive['end'] = _point_payload(command.end)
            primitive['pen_down'] = bool(pen_down)
        elif isinstance(command, ArcSegment):
            start, end = _arc_endpoint_payloads(command)
            primitive['type'] = 'ARC_SEGMENT'
            primitive['start'] = start
            primitive['end'] = end
            primitive['center'] = _point_payload(command.center)
            primitive['radius'] = float(command.radius)
            primitive['start_angle_rad'] = float(command.start_angle_rad)
            primitive['sweep_angle_rad'] = float(command.sweep_angle_rad)
            primitive['clockwise'] = float(command.sweep_angle_rad) < 0.0
            primitive['pen_down'] = bool(pen_down)
        elif isinstance(command, QuadraticBezier):
            primitive['type'] = 'QUADRATIC_BEZIER'
            primitive['start'] = _point_payload(command.start)
            primitive['end'] = _point_payload(command.end)
            primitive['control1'] = _point_payload(command.control)
            primitive['pen_down'] = bool(pen_down)
        elif isinstance(command, CubicBezier):
            primitive['type'] = 'CUBIC_BEZIER'
            primitive['start'] = _point_payload(command.start)
            primitive['end'] = _point_payload(command.end)
            primitive['control1'] = _point_payload(command.control1)
            primitive['control2'] = _point_payload(command.control2)
            primitive['pen_down'] = bool(pen_down)
        else:
            raise ValueError(f'Unsupported canonical command {type(command)!r}.')
        primitives.append(primitive)
    return {
        'frame': str(plan.frame),
        'theta_ref': float(plan.theta_ref),
        'primitives': primitives,
    }


def _command_path_length(command: object, *, tolerance_m: float = 1.0e-4) -> float:
    descriptor = _command_to_cpp_descriptor(command)
    if _geometry_cpp is not None:
        return float(_geometry_cpp.path_length(descriptor, tolerance_m=tolerance_m))
    if isinstance(command, (PenUp, PenDown)):
        return 0.0
    if isinstance(command, (TravelMove, LineSegment)):
        return _distance(command.start, command.end)
    if isinstance(command, ArcSegment):
        return abs(float(command.radius) * float(command.sweep_angle_rad))
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
    return 0.0


def _command_curvature(command: object, *, t: float = 0.5) -> float:
    descriptor = _command_to_cpp_descriptor(command)
    if _geometry_cpp is not None:
        return float(_geometry_cpp.curvature(descriptor, t=t))
    if isinstance(command, ArcSegment):
        if abs(float(command.sweep_angle_rad)) <= _EPS:
            return 0.0
        return (1.0 if float(command.sweep_angle_rad) >= 0.0 else -1.0) / float(command.radius)
    return 0.0


def canonical_command_to_debug_dict(command: object) -> dict[str, object]:
    if isinstance(command, PenUp):
        return {'type': 'pen_up'}
    if isinstance(command, PenDown):
        return {'type': 'pen_down'}
    if isinstance(command, TravelMove):
        return {
            'type': 'travel',
            'start': list(command.start),
            'end': list(command.end),
            'length_m': _command_path_length(command),
        }
    if isinstance(command, LineSegment):
        return {
            'type': 'line',
            'start': list(command.start),
            'end': list(command.end),
            'length_m': _command_path_length(command),
            'curvature': 0.0,
        }
    if isinstance(command, ArcSegment):
        return {
            'type': 'arc',
            'center': list(command.center),
            'radius': float(command.radius),
            'start_angle_rad': float(command.start_angle_rad),
            'sweep_angle_rad': float(command.sweep_angle_rad),
            'length_m': _command_path_length(command),
            'curvature': _command_curvature(command),
        }
    if isinstance(command, QuadraticBezier):
        return {
            'type': 'quadratic',
            'start': list(command.start),
            'control': list(command.control),
            'end': list(command.end),
            'length_m': _command_path_length(command),
            'curvature_midpoint': _command_curvature(command),
        }
    if isinstance(command, CubicBezier):
        return {
            'type': 'cubic',
            'start': list(command.start),
            'control1': list(command.control1),
            'control2': list(command.control2),
            'end': list(command.end),
            'length_m': _command_path_length(command),
            'curvature_midpoint': _command_curvature(command),
        }
    raise ValueError(f'Unsupported canonical command {type(command)!r}.')


def canonical_plan_debug_payload(
    plan: CanonicalPathPlan,
    *,
    sampling_policy: SamplingPolicy | None = None,
    command_metadata: tuple[dict[str, object] | None, ...] | None = None,
) -> dict[str, object]:
    active_policy = _sanitize_sampling_policy(sampling_policy)
    sampled_paths = sampled_paths_from_canonical_plan(plan, sampling_policy=active_policy)
    metrics = _sampled_path_metrics(sampled_paths)
    commands: list[dict[str, object]] = []
    metadata_items = command_metadata if command_metadata is not None else ()
    for index, command in enumerate(plan.commands):
        payload = canonical_command_to_debug_dict(command)
        if index < len(metadata_items) and metadata_items[index]:
            payload.update(
                {
                    str(key): value
                    for key, value in metadata_items[index].items()
                    if value is not None
                }
            )
        commands.append(payload)
    return {
        'frame': str(plan.frame),
        'theta_ref': float(plan.theta_ref),
        'command_count': int(plan.command_count),
        'primitive_counts': _primitive_counts(plan),
        'sampled_bounds': metrics['all_bounds'],
        'commands': commands,
    }


def _sampled_paths_from_cpp(
    plan: CanonicalPathPlan,
    *,
    sampling_policy: SamplingPolicy,
) -> tuple[SampledPath, ...]:
    if _geometry_cpp is None:
        raise RuntimeError('wall_climber_geometry_cpp is not available.')
    sampled = _geometry_cpp.sample_canonical_plan(
        _plan_to_cpp_descriptor(plan),
        curve_tolerance_m=float(sampling_policy.curve_tolerance_m),
        draw_step_m=sampling_policy.draw_step_m,
        travel_step_m=sampling_policy.travel_step_m,
        max_heading_delta_rad=sampling_policy.max_heading_delta_rad,
    )
    return tuple(
        SampledPath(
            draw=bool(item['draw']),
            points=tuple(
                (float(point[0]), float(point[1]))
                for point in item['points']
            ),
        )
        for item in sampled
        if len(item['points']) >= 2
    )


def _sampled_paths_from_python(
    plan: CanonicalPathPlan,
    *,
    sampling_policy: SamplingPolicy,
) -> tuple[SampledPath, ...]:
    sampled_paths: list[SampledPath] = []
    active_points: list[Point2D] = []
    active_draw: bool | None = None
    pen_down = False

    def flush_active() -> None:
        nonlocal active_points, active_draw
        if active_draw is None or len(active_points) < 2:
            active_points = []
            active_draw = None
            return
        sampled_paths.append(SampledPath(draw=active_draw, points=tuple(active_points)))
        active_points = []
        active_draw = None

    def append_geometry(points: tuple[Point2D, ...], *, draw: bool) -> None:
        nonlocal active_draw
        if len(points) < 2:
            return
        if active_draw is None:
            active_draw = draw
        elif active_draw != draw:
            flush_active()
            active_draw = draw
        for point in points:
            _append_sampled_point(active_points, point)

    for command in plan.commands:
        if isinstance(command, PenUp):
            flush_active()
            pen_down = False
            continue
        if isinstance(command, PenDown):
            flush_active()
            pen_down = True
            continue
        if isinstance(command, TravelMove):
            append_geometry(
                _sample_travel(command, step_m=sampling_policy.travel_step_m),
                draw=False,
            )
            continue
        if isinstance(command, LineSegment):
            append_geometry(
                _sample_line(command, step_m=sampling_policy.draw_step_m),
                draw=pen_down,
            )
            continue
        if isinstance(command, ArcSegment):
            append_geometry(
                _sample_arc(
                    command,
                    curve_tolerance_m=sampling_policy.curve_tolerance_m,
                    max_heading_delta_rad=sampling_policy.max_heading_delta_rad,
                ),
                draw=pen_down,
            )
            continue
        if isinstance(command, QuadraticBezier):
            append_geometry(
                _sample_quadratic(
                    command,
                    curve_tolerance_m=sampling_policy.curve_tolerance_m,
                    max_heading_delta_rad=sampling_policy.max_heading_delta_rad,
                ),
                draw=pen_down,
            )
            continue
        if isinstance(command, CubicBezier):
            append_geometry(
                _sample_cubic(
                    command,
                    curve_tolerance_m=sampling_policy.curve_tolerance_m,
                    max_heading_delta_rad=sampling_policy.max_heading_delta_rad,
                ),
                draw=pen_down,
            )
            continue
        raise ValueError(f'Unsupported canonical command {type(command)!r}.')

    flush_active()
    return tuple(sampled_paths)


def sampled_paths_from_canonical_plan(
    plan: CanonicalPathPlan,
    *,
    curve_tolerance_m: float = 0.01,
    sampling_policy: SamplingPolicy | None = None,
) -> tuple[SampledPath, ...]:
    policy = _sanitize_sampling_policy(
        sampling_policy,
        curve_tolerance_m=curve_tolerance_m,
    )
    if _geometry_cpp is not None:
        return _sampled_paths_from_cpp(
            plan,
            sampling_policy=policy,
        )
    return _sampled_paths_from_python(
        plan,
        sampling_policy=policy,
    )


def canonical_plan_to_draw_strokes(
    plan: CanonicalPathPlan,
    *,
    curve_tolerance_m: float = 0.01,
    sampling_policy: SamplingPolicy | None = None,
) -> tuple[tuple[Point2D, ...], ...]:
    return tuple(
        sampled.points
        for sampled in sampled_paths_from_canonical_plan(
            plan,
            curve_tolerance_m=curve_tolerance_m,
            sampling_policy=sampling_policy,
        )
        if sampled.draw
    )


def canonical_plan_to_sampled_paths(
    plan: CanonicalPathPlan,
    *,
    curve_tolerance_m: float = 0.01,
    sampling_policy: SamplingPolicy | None = None,
) -> tuple[SampledPath, ...]:
    return sampled_paths_from_canonical_plan(
        plan,
        curve_tolerance_m=curve_tolerance_m,
        sampling_policy=sampling_policy,
    )


def canonical_plan_to_segment_payload(
    plan: CanonicalPathPlan,
    *,
    curve_tolerance_m: float = 0.01,
    sampling_policy: SamplingPolicy | None = None,
) -> dict[str, object]:
    segments = []
    for sampled in sampled_paths_from_canonical_plan(
        plan,
        curve_tolerance_m=curve_tolerance_m,
        sampling_policy=sampling_policy,
    ):
        if len(sampled.points) < 2:
            continue
        segments.append(
            {
                'draw': bool(sampled.draw),
                'type': 'line' if len(sampled.points) == 2 else 'polyline',
                'points': [[float(point[0]), float(point[1])] for point in sampled.points],
            }
        )
    return {
        'frame': str(plan.frame),
        'theta_ref': float(plan.theta_ref),
        'segments': segments,
    }


def canonical_plan_to_legacy_draw_plan(
    plan: CanonicalPathPlan,
    *,
    curve_tolerance_m: float = 0.01,
    sampling_policy: SamplingPolicy | None = None,
) -> dict[str, object]:
    return canonical_plan_to_segment_payload(
        plan,
        curve_tolerance_m=curve_tolerance_m,
        sampling_policy=sampling_policy,
    )


def canonical_plan_to_legacy_strokes(
    plan: CanonicalPathPlan,
    *,
    curve_tolerance_m: float = 0.01,
    sampling_policy: SamplingPolicy | None = None,
) -> dict[str, object]:
    strokes = []
    for sampled in sampled_paths_from_canonical_plan(
        plan,
        curve_tolerance_m=curve_tolerance_m,
        sampling_policy=sampling_policy,
    ):
        if not sampled.draw or len(sampled.points) < 2:
            continue
        strokes.append(
            {
                'draw': True,
                'type': 'line' if len(sampled.points) == 2 else 'polyline',
                'points': [[float(point[0]), float(point[1])] for point in sampled.points],
            }
        )
    if not strokes:
        raise ValueError('No drawable strokes available after canonical export.')
    return {'frame': str(plan.frame), 'strokes': strokes}


def _legacy_segment_type_counts(segments: list[dict[str, object]]) -> dict[str, int]:
    line_count = sum(1 for segment in segments if segment.get('type') == 'line')
    polyline_count = sum(1 for segment in segments if segment.get('type') == 'polyline')
    return {
        'line': int(line_count),
        'polyline': int(polyline_count),
    }


def _legacy_contract_summary(
    plan: CanonicalPathPlan,
    *,
    preview_sampling_policy: SamplingPolicy,
    runtime_sampling_policy: SamplingPolicy,
) -> dict[str, object]:
    preview_draw_plan = canonical_plan_to_segment_payload(
        plan,
        sampling_policy=preview_sampling_policy,
    )
    runtime_draw_plan = canonical_plan_to_segment_payload(
        plan,
        sampling_policy=runtime_sampling_policy,
    )
    preview_strokes = canonical_plan_to_legacy_strokes(
        plan,
        sampling_policy=preview_sampling_policy,
    )
    runtime_strokes = canonical_plan_to_legacy_strokes(
        plan,
        sampling_policy=runtime_sampling_policy,
    )
    preview_segments = list(preview_draw_plan['segments'])
    runtime_segments = list(runtime_draw_plan['segments'])
    preview_stroke_items = list(preview_strokes['strokes'])
    runtime_stroke_items = list(runtime_strokes['strokes'])
    return {
        'internal_truth': 'canonical_path_plan',
        'draw_plan_role': 'diagnostic_export_only',
        'stroke_payload_role': 'preview_payload_only',
        'runtime_transport': 'primitive_path_plan_only',
        'raw_draw_plan_endpoint_enabled': False,
        'preview_export': {
            'segment_count': len(preview_segments),
            'stroke_count': len(preview_stroke_items),
            'segment_type_counts': _legacy_segment_type_counts(preview_segments),
        },
        'runtime_export': {
            'segment_count': len(runtime_segments),
            'stroke_count': len(runtime_stroke_items),
            'segment_type_counts': _legacy_segment_type_counts(runtime_segments),
        },
    }


def _sampled_path_bounds(sampled_paths: tuple[SampledPath, ...]) -> dict[str, float] | None:
    points = [point for sampled in sampled_paths for point in sampled.points]
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return {
        'x_min': min(xs),
        'x_max': max(xs),
        'y_min': min(ys),
        'y_max': max(ys),
        'width': max(xs) - min(xs),
        'height': max(ys) - min(ys),
    }


def _sampled_path_metrics(sampled_paths: tuple[SampledPath, ...]) -> dict[str, object]:
    draw_paths = tuple(sampled for sampled in sampled_paths if sampled.draw)
    travel_paths = tuple(sampled for sampled in sampled_paths if not sampled.draw)
    draw_points = sum(len(sampled.points) for sampled in draw_paths)
    travel_points = sum(len(sampled.points) for sampled in travel_paths)
    return {
        'draw_path_count': len(draw_paths),
        'travel_path_count': len(travel_paths),
        'draw_point_count': draw_points,
        'travel_point_count': travel_points,
        'total_point_count': draw_points + travel_points,
        'draw_bounds': _sampled_path_bounds(draw_paths),
        'all_bounds': _sampled_path_bounds(sampled_paths),
    }


def _sampling_policy_payload(policy: SamplingPolicy) -> dict[str, object]:
    return {
        'label': str(policy.label),
        'curve_tolerance_m': float(policy.curve_tolerance_m),
        'draw_step_m': None if policy.draw_step_m is None else float(policy.draw_step_m),
        'travel_step_m': None if policy.travel_step_m is None else float(policy.travel_step_m),
        'max_heading_delta_rad': (
            None if policy.max_heading_delta_rad is None else float(policy.max_heading_delta_rad)
        ),
    }


def _primitive_counts(plan: CanonicalPathPlan) -> dict[str, int]:
    counts = Counter(type(command).__name__ for command in plan.commands)
    return {
        'PenUp': int(counts.get('PenUp', 0)),
        'PenDown': int(counts.get('PenDown', 0)),
        'TravelMove': int(counts.get('TravelMove', 0)),
        'LineSegment': int(counts.get('LineSegment', 0)),
        'ArcSegment': int(counts.get('ArcSegment', 0)),
        'QuadraticBezier': int(counts.get('QuadraticBezier', 0)),
        'CubicBezier': int(counts.get('CubicBezier', 0)),
    }


def canonical_plan_diagnostics(
    plan: CanonicalPathPlan,
    *,
    preview_sampling_policy: SamplingPolicy,
    runtime_sampling_policy: SamplingPolicy,
) -> dict[str, object]:
    preview_policy = _sanitize_sampling_policy(preview_sampling_policy)
    runtime_policy = _sanitize_sampling_policy(runtime_sampling_policy)

    preview_paths = sampled_paths_from_canonical_plan(plan, sampling_policy=preview_policy)
    runtime_paths = sampled_paths_from_canonical_plan(plan, sampling_policy=runtime_policy)
    preview_metrics = _sampled_path_metrics(preview_paths)
    runtime_metrics = _sampled_path_metrics(runtime_paths)

    preview_bounds = preview_metrics['draw_bounds']
    runtime_bounds = runtime_metrics['draw_bounds']
    if preview_bounds is None or runtime_bounds is None:
        bounds_delta_max_m = 0.0
    else:
        bounds_delta_max_m = max(
            abs(float(preview_bounds['x_min']) - float(runtime_bounds['x_min'])),
            abs(float(preview_bounds['x_max']) - float(runtime_bounds['x_max'])),
            abs(float(preview_bounds['y_min']) - float(runtime_bounds['y_min'])),
            abs(float(preview_bounds['y_max']) - float(runtime_bounds['y_max'])),
        )
    bounds_tolerance_m = max(
        float(preview_policy.curve_tolerance_m),
        float(runtime_policy.curve_tolerance_m),
    ) * 2.0
    path_count_match = (
        int(preview_metrics['draw_path_count']) == int(runtime_metrics['draw_path_count'])
        and int(preview_metrics['travel_path_count']) == int(runtime_metrics['travel_path_count'])
    )
    parity_status = 'ok' if path_count_match and bounds_delta_max_m <= bounds_tolerance_m + _EPS else 'warning'
    preview_total_points = int(preview_metrics['total_point_count'])
    runtime_total_points = int(runtime_metrics['total_point_count'])
    runtime_to_preview_ratio = (
        float(runtime_total_points) / float(preview_total_points)
        if preview_total_points > 0 else 1.0
    )

    return {
        'canonical_plan': {
            'frame': str(plan.frame),
            'theta_ref': float(plan.theta_ref),
            'command_count': len(plan.commands),
            'primitive_counts': _primitive_counts(plan),
        },
        'legacy_contract': _legacy_contract_summary(
            plan,
            preview_sampling_policy=preview_policy,
            runtime_sampling_policy=runtime_policy,
        ),
        'preview_sampling': {
            'policy': _sampling_policy_payload(preview_policy),
            **preview_metrics,
        },
        'runtime_sampling': {
            'policy': _sampling_policy_payload(runtime_policy),
            **runtime_metrics,
        },
        'point_budget': {
            'preview_total_points': preview_total_points,
            'runtime_total_points': runtime_total_points,
            'delta_points': runtime_total_points - preview_total_points,
            'runtime_to_preview_ratio': runtime_to_preview_ratio,
        },
        'parity': {
            'status': parity_status,
            'draw_path_count_match': path_count_match,
            'bounds_delta_max_m': bounds_delta_max_m,
            'bounds_tolerance_m': bounds_tolerance_m,
        },
    }


def pen_strokes_to_canonical_plan(
    pen_strokes: tuple[tuple[Point2D, ...], ...],
    *,
    theta_ref: float,
    pen_offset_x_m: float,
    pen_offset_y_m: float,
    frame: str = 'board',
) -> CanonicalPathPlan:
    if not pen_strokes:
        raise ValueError('No draw strokes available to convert into a canonical plan.')

    rotated_offset = _rotate_point((pen_offset_x_m, pen_offset_y_m), theta_ref)

    def to_body_points(stroke: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
        return tuple(
            (
                float(point[0]) - rotated_offset[0],
                float(point[1]) - rotated_offset[1],
            )
            for point in stroke
        )

    commands: list[object] = []
    previous_end: Point2D | None = None
    pen_is_down = False

    for stroke in pen_strokes:
        if len(stroke) < 2:
            continue
        body_points = to_body_points(stroke)
        start_point = body_points[0]
        if previous_end is not None and not _approximately_equal(previous_end, start_point):
            if pen_is_down:
                commands.append(PenUp())
                pen_is_down = False
            commands.append(TravelMove(start=previous_end, end=start_point))
        if not pen_is_down:
            commands.append(PenDown())
            pen_is_down = True
        for index in range(1, len(body_points)):
            start = body_points[index - 1]
            end = body_points[index]
            if _approximately_equal(start, end):
                continue
            commands.append(LineSegment(start=start, end=end))
        previous_end = body_points[-1]

    if not commands:
        raise ValueError('No drawable strokes available to convert into a canonical plan.')
    if pen_is_down:
        commands.append(PenUp())

    return CanonicalPathPlan(
        frame=frame,
        theta_ref=float(theta_ref),
        commands=tuple(commands),
    )
