from __future__ import annotations

import pytest

from wall_climber.x_plotter.corexy_kinematics import (
    delta_motors_to_delta_xy,
    delta_xy_to_delta_motors,
    motors_to_xy,
    xy_to_motors,
)


@pytest.mark.parametrize(
    ('x', 'y'),
    [
        (0.0, 0.0),
        (1.25, 0.5),
        (3.2, 1.4),
        (-0.3, 2.7),
    ],
)
def test_xy_to_motors_round_trip(x: float, y: float) -> None:
    a, b = xy_to_motors(x, y)
    round_trip_x, round_trip_y = motors_to_xy(a, b)

    assert round_trip_x == pytest.approx(x)
    assert round_trip_y == pytest.approx(y)


@pytest.mark.parametrize(
    ('a', 'b'),
    [
        (0.0, 0.0),
        (1.75, 0.25),
        (4.6, 1.8),
        (-0.4, 2.2),
    ],
)
def test_motors_to_xy_round_trip(a: float, b: float) -> None:
    x, y = motors_to_xy(a, b)
    round_trip_a, round_trip_b = xy_to_motors(x, y)

    assert round_trip_a == pytest.approx(a)
    assert round_trip_b == pytest.approx(b)


@pytest.mark.parametrize(
    ('dx', 'dy'),
    [
        (0.0, 0.0),
        (0.05, -0.02),
        (-0.3, 0.4),
        (1.2, 0.8),
    ],
)
def test_delta_round_trip(dx: float, dy: float) -> None:
    da, db = delta_xy_to_delta_motors(dx, dy)
    round_trip_dx, round_trip_dy = delta_motors_to_delta_xy(da, db)

    assert round_trip_dx == pytest.approx(dx)
    assert round_trip_dy == pytest.approx(dy)

