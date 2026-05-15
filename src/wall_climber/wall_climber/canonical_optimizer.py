from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from wall_climber import _optimizer_geometry as _geom
from wall_climber import _optimizer_merge as _merge
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


_EPS = _geom.EPS


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
    return _geom.distance(a, b)


def _approximately_equal(a: Point2D, b: Point2D, *, eps: float = _EPS) -> bool:
    return _geom.approximately_equal(a, b, eps=eps)


def _angle_delta_deg(first: float, second: float) -> float:
    return _geom.angle_delta_deg(first, second)


def _primitive_start(command: CanonicalCommand) -> Point2D:
    return _geom.primitive_start(command)


def _primitive_end(command: CanonicalCommand) -> Point2D:
    return _geom.primitive_end(command)


def _primitive_length(command: CanonicalCommand) -> float:
    return _geom.primitive_length(command)


def _is_draw_primitive(command: CanonicalCommand) -> bool:
    return _geom.is_draw_primitive(command)


def _reverse_draw_command(command: CanonicalCommand) -> CanonicalCommand:
    return _geom.reverse_draw_command(command)


def _reverse_unit(unit: _DrawUnit) -> _DrawUnit:
    return _DrawUnit(
        commands=tuple(_reverse_draw_command(command) for command in reversed(unit.commands)),
        original_index=unit.original_index,
    )


def _line_heading(line: LineSegment) -> float:
    return _geom.line_heading(line)


def _can_merge_lines(
    first: LineSegment,
    second: LineSegment,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    return _merge.can_merge_lines(
        first,
        second,
        angle_tolerance_deg=angle_tolerance_deg,
        distance_tolerance_m=distance_tolerance_m,
    )


def _merge_lines(first: LineSegment, second: LineSegment) -> LineSegment:
    return _merge.merge_lines(first, second)


def _arc_start(command: ArcSegment) -> Point2D:
    return _geom.arc_start(command)


def _arc_end(command: ArcSegment) -> Point2D:
    return _geom.arc_end(command)


def _can_merge_arcs(
    first: ArcSegment,
    second: ArcSegment,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    return _merge.can_merge_arcs(
        first,
        second,
        angle_tolerance_deg=angle_tolerance_deg,
        distance_tolerance_m=distance_tolerance_m,
    )


def _merge_arcs(first: ArcSegment, second: ArcSegment) -> ArcSegment:
    return _merge.merge_arcs(first, second)


def _vector_angle(vector: Point2D) -> float:
    return _geom.vector_angle(vector)


def _quadratic_derivative(command: QuadraticBezier, t: float) -> Point2D:
    return _geom.quadratic_derivative(command, t)


def _cubic_derivative(command: CubicBezier, t: float) -> Point2D:
    return _geom.cubic_derivative(command, t)


def _tangent_angle(command: CanonicalCommand, *, at_end: bool) -> float | None:
    return _geom.tangent_angle(command, at_end=at_end)


def _evaluate_quadratic(command: QuadraticBezier, t: float) -> Point2D:
    return _geom.evaluate_quadratic(command, t)


def _evaluate_cubic(command: CubicBezier, t: float) -> Point2D:
    return _geom.evaluate_cubic(command, t)


def _sample_curve_command(command: CanonicalCommand, *, segments: int) -> tuple[Point2D, ...]:
    return _geom.sample_curve_command(command, segments=segments)


def _chord_length_parameters(points: tuple[Point2D, ...]) -> tuple[float, ...]:
    return _geom.chord_length_parameters(points)


def _solve_cubic_controls(
    points: tuple[Point2D, ...],
    *,
    parameters: tuple[float, ...] | None = None,
) -> CubicBezier | None:
    return _merge.solve_cubic_controls(points, parameters=parameters)


def _reduce_cubic_to_quadratic(
    cubic: CubicBezier,
    *,
    fit_tolerance_m: float,
    sampled_points: tuple[Point2D, ...],
) -> QuadraticBezier | None:
    return _merge.reduce_cubic_to_quadratic(
        cubic,
        fit_tolerance_m=fit_tolerance_m,
        sampled_points=sampled_points,
    )


def _can_merge_curve_pair(
    first: CanonicalCommand,
    second: CanonicalCommand,
    *,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> bool:
    return _merge.can_merge_curve_pair(
        first,
        second,
        angle_tolerance_deg=angle_tolerance_deg,
        distance_tolerance_m=distance_tolerance_m,
    )


def _merge_curve_pair(
    first: CanonicalCommand,
    second: CanonicalCommand,
    *,
    fit_tolerance_m: float,
    angle_tolerance_deg: float,
    distance_tolerance_m: float,
) -> CanonicalCommand | None:
    return _merge.merge_curve_pair(
        first,
        second,
        fit_tolerance_m=fit_tolerance_m,
        angle_tolerance_deg=angle_tolerance_deg,
        distance_tolerance_m=distance_tolerance_m,
    )


def _circle_from_points(
    first: Point2D,
    middle: Point2D,
    last: Point2D,
) -> tuple[Point2D, float] | None:
    return _geom.circle_from_points(first, middle, last)


def _unwrap_angles(
    points: tuple[Point2D, ...],
    *,
    center: Point2D,
    clockwise: bool,
) -> tuple[float, ...] | None:
    return _geom.unwrap_angles(points, center=center, clockwise=clockwise)


def _polyline_points_from_lines(lines: tuple[LineSegment, ...]) -> tuple[Point2D, ...]:
    return _geom.polyline_points_from_lines(lines)


def _fit_arc_from_line_chain(
    lines: tuple[LineSegment, ...],
    *,
    tolerance_m: float,
) -> ArcSegment | None:
    return _merge.fit_arc_from_line_chain(lines, tolerance_m=tolerance_m)


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
