from __future__ import annotations

import math

from wall_climber.canonical_optimizer import (
    CanonicalOptimizationPolicy,
    optimize_canonical_plan,
)
from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalPathPlan,
    CubicBezier,
    LineSegment,
    PenDown,
    PenUp,
    QuadraticBezier,
    TravelMove,
)


def test_optimizer_merges_collinear_lines_and_prunes_tiny_primitives() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(1.0, 0.0)),
            LineSegment(start=(1.0, 0.0), end=(2.0, 0.0)),
            LineSegment(start=(2.0, 0.0), end=(2.0002, 0.0)),
            PenUp(),
        ),
    )

    result = optimize_canonical_plan(
        plan,
        policy=CanonicalOptimizationPolicy(
            label='test',
            reorder_units=False,
            remove_duplicate_units=False,
            tiny_primitive_m=0.001,
        ),
    )

    assert [type(command) for command in result.plan.commands] == [PenDown, LineSegment, PenUp]
    merged = result.plan.commands[1]
    assert isinstance(merged, LineSegment)
    assert merged.start == (0.0, 0.0)
    assert merged.end == (2.0, 0.0)
    assert result.stats.merged_line_segments == 1
    assert result.stats.pruned_primitives == 1


def test_optimizer_preserves_tiny_primitives_when_pruning_disabled() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(0.0004, 0.0)),
            PenUp(),
        ),
    )

    result = optimize_canonical_plan(
        plan,
        policy=CanonicalOptimizationPolicy(
            label='preserve_tiny',
            reorder_units=False,
            merge_collinear_lines=False,
            prune_tiny_primitives=False,
            tiny_primitive_m=0.001,
        ),
    )

    lines = [command for command in result.plan.commands if isinstance(command, LineSegment)]
    assert len(lines) == 1
    assert lines[0].start == (0.0, 0.0)
    assert lines[0].end == (0.0004, 0.0)
    assert result.stats.pruned_primitives == 0


def test_optimizer_removes_duplicate_units_and_reorders_for_shorter_travel() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(10.0, 0.0), end=(11.0, 0.0)),
            PenUp(),
            TravelMove(start=(11.0, 0.0), end=(0.0, 0.0)),
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(1.0, 0.0)),
            PenUp(),
            TravelMove(start=(1.0, 0.0), end=(2.0, 0.0)),
            PenDown(),
            LineSegment(start=(2.0, 0.0), end=(3.0, 0.0)),
            PenUp(),
            TravelMove(start=(3.0, 0.0), end=(2.0, 0.0)),
            PenDown(),
            LineSegment(start=(2.0, 0.0), end=(3.0, 0.0)),
            PenUp(),
        ),
    )

    result = optimize_canonical_plan(plan)

    draw_lines = [command for command in result.plan.commands if isinstance(command, LineSegment)]
    assert len(draw_lines) == 3
    assert draw_lines[0].start == (0.0, 0.0)
    assert draw_lines[0].end == (1.0, 0.0)
    assert result.stats.removed_duplicate_units == 1
    assert result.stats.optimized_travel_length_m < result.stats.original_travel_length_m
    assert result.stats.reordered_units is True


def test_optimizer_can_reverse_quadratic_unit_when_it_reduces_travel() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(0.0, 0.0), end=(1.0, 0.0)),
            PenUp(),
            TravelMove(start=(1.0, 0.0), end=(5.0, 0.0)),
            PenDown(),
            QuadraticBezier(
                start=(5.0, 0.0),
                control=(4.0, 1.0),
                end=(2.0, 0.0),
            ),
            PenUp(),
        ),
    )

    result = optimize_canonical_plan(plan)

    quadratics = [command for command in result.plan.commands if isinstance(command, QuadraticBezier)]
    assert len(quadratics) == 1
    assert quadratics[0].start == (2.0, 0.0)
    assert quadratics[0].end == (5.0, 0.0)
    assert math.isclose(result.stats.optimized_travel_length_m, 1.0, rel_tol=1.0e-6)
    assert result.stats.optimized_travel_length_m < result.stats.original_travel_length_m


def test_optimizer_fits_arc_from_line_chain_when_enabled() -> None:
    points = (
        (1.0, 0.0),
        (0.9238795, 0.3826834),
        (0.7071068, 0.7071068),
        (0.3826834, 0.9238795),
        (0.0, 1.0),
    )
    commands = [PenDown()]
    for start, end in zip(points[:-1], points[1:]):
        commands.append(LineSegment(start=start, end=end))
    commands.append(PenUp())
    plan = CanonicalPathPlan(frame='board', theta_ref=0.0, commands=tuple(commands))

    result = optimize_canonical_plan(
        plan,
        policy=CanonicalOptimizationPolicy(
            label='svg',
            reorder_units=False,
            fit_arcs=True,
            arc_fit_tolerance_m=0.02,
        ),
    )

    arcs = [command for command in result.plan.commands if isinstance(command, ArcSegment)]
    assert len(arcs) == 1
    assert result.stats.fitted_arc_segments == 1


def test_optimizer_hatch_ordering_reduces_travel_for_scrambled_lines() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(3.0, 2.0), end=(0.0, 2.0)),
            PenUp(),
            TravelMove(start=(0.0, 2.0), end=(2.5, 0.0)),
            PenDown(),
            LineSegment(start=(2.5, 0.0), end=(0.0, 0.0)),
            PenUp(),
            TravelMove(start=(0.0, 0.0), end=(3.0, 1.0)),
            PenDown(),
            LineSegment(start=(3.0, 1.0), end=(0.0, 1.0)),
            PenUp(),
        ),
    )

    result = optimize_canonical_plan(
        plan,
        policy=CanonicalOptimizationPolicy(
            label='image',
            reorder_units=False,
            enable_hatch_ordering=True,
        ),
    )

    assert result.stats.hatch_reordered_units is True
    assert result.stats.optimized_travel_length_m < result.stats.original_travel_length_m


def test_optimizer_merges_adjacent_curve_spans_when_tangents_match() -> None:
    full_curve = CubicBezier(
        start=(0.0, 0.0),
        control1=(0.5, 1.0),
        control2=(1.5, 1.0),
        end=(2.0, 0.0),
    )

    def lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
        return (a[0] + ((b[0] - a[0]) * t), a[1] + ((b[1] - a[1]) * t))

    p01 = lerp(full_curve.start, full_curve.control1, 0.5)
    p12 = lerp(full_curve.control1, full_curve.control2, 0.5)
    p23 = lerp(full_curve.control2, full_curve.end, 0.5)
    p012 = lerp(p01, p12, 0.5)
    p123 = lerp(p12, p23, 0.5)
    split_point = lerp(p012, p123, 0.5)

    first_half = CubicBezier(
        start=full_curve.start,
        control1=p01,
        control2=p012,
        end=split_point,
    )
    second_half = CubicBezier(
        start=split_point,
        control1=p123,
        control2=p23,
        end=full_curve.end,
    )

    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(PenDown(), first_half, second_half, PenUp()),
    )

    result = optimize_canonical_plan(
        plan,
        policy=CanonicalOptimizationPolicy(
            label='image',
            reorder_units=False,
            fit_arcs=False,
            arc_fit_tolerance_m=0.02,
        ),
    )

    curves = [command for command in result.plan.commands if isinstance(command, (QuadraticBezier, CubicBezier))]
    assert len(curves) == 1
    assert isinstance(curves[0], CubicBezier)
    assert result.stats.merged_curve_segments == 1
