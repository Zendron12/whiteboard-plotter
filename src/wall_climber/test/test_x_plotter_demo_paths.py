from __future__ import annotations

import math

import pytest

from wall_climber.x_plotter.demo_paths import build_demo_path
from wall_climber.x_plotter.frame_config import load_board_frame_config
from wall_climber.x_plotter.primitive_sampler import sample_primitive_path_plan


@pytest.mark.parametrize('demo_name', ['line', 'square', 'triangle', 'line_square_triangle'])
def test_x_plotter_demo_path_samples_inside_board(demo_name: str) -> None:
    frame = load_board_frame_config()
    plan = build_demo_path(demo_name, frame=frame)

    sampled_paths = sample_primitive_path_plan(plan)

    assert plan.frame == 'board'
    assert sampled_paths
    assert any(path.draw for path in sampled_paths)
    for path in sampled_paths:
        assert len(path.points) >= 2
        for x, y in path.points:
            assert math.isfinite(x)
            assert math.isfinite(y)
            assert frame.in_drawable_bounds(x, y)


def test_x_plotter_demo_path_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match='Unsupported X plotter demo path'):
        build_demo_path('hexagon')

