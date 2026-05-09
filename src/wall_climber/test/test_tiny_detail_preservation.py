from __future__ import annotations

import math

import pytest

from wall_climber.canonical_path import CanonicalPathPlan, LineSegment, PenDown, PenUp, TravelMove
from wall_climber.canonical_tiny_details import expand_tiny_details_in_canonical_plan


def _line_length(segment: LineSegment) -> float:
    return math.hypot(segment.end[0] - segment.start[0], segment.end[1] - segment.start[1])


def test_tiny_detail_expands_to_micro_cross_geometry() -> None:
    tiny_start = (1.0, 1.0)
    tiny_end = (1.0004, 1.0002)
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            TravelMove(start=(0.5, 0.5), end=tiny_start),
            PenDown(),
            LineSegment(start=tiny_start, end=tiny_end),
            PenUp(),
            TravelMove(start=tiny_end, end=(2.0, 2.0)),
            PenDown(),
            LineSegment(start=(2.0, 2.0), end=(2.05, 2.0)),
            PenUp(),
        ),
    )

    result = expand_tiny_details_in_canonical_plan(
        plan,
        minimum_drawable_feature_m=0.004,
        candidate_max_feature_m=0.002,
        expand_mode='micro_cross',
        bounds={'x_min': 0.0, 'x_max': 6.3, 'y_min': 0.0, 'y_max': 3.0},
    )

    assert result.metrics['tiny_details_detected'] == 1
    assert result.metrics['tiny_details_expanded'] == 1
    assert result.metrics['tiny_details_preserved'] == 1
    assert result.plan != plan

    lines = [command for command in result.plan.commands if isinstance(command, LineSegment)]
    cross_lines = [segment for segment in lines if _line_length(segment) == pytest.approx(0.004)]
    assert len(cross_lines) == 2
    assert any(_line_length(segment) == pytest.approx(0.05) for segment in lines)


def test_tiny_detail_preservation_can_be_disabled() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(1.0, 1.0), end=(1.0002, 1.0002)),
            PenUp(),
        ),
    )

    result = expand_tiny_details_in_canonical_plan(
        plan,
        preserve=False,
        minimum_drawable_feature_m=0.004,
    )

    assert result.plan is plan
    assert result.metrics['preserve_tiny_details'] is False
    assert result.metrics['tiny_details_expanded'] == 0
