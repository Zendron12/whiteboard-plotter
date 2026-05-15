"""Smoke tests for the per-step pose clamp added to CableSupervisorPlugin.

The supervisor itself depends on Webots runtime (controller, ``Supervisor.getRoot``,
``rclpy``, etc.) so importing it directly is not feasible from ``pytest``.
Instead we exercise the math that backs the rate limit in isolation:

    if distance > _MAX_STEP_TELEPORT_M:
        scale = _MAX_STEP_TELEPORT_M / distance
        center_x = cur_x + dx * scale
        center_y = cur_y + dy * scale
        reached_target = False

This test guards against accidental regressions in the cap value or the
linear-interpolation formula. It does not test integration with Webots.
"""

import math


_MAX_STEP_TELEPORT_M = 0.120


def _clamped_step(cur_x, cur_y, target_x, target_y):
    dx = target_x - cur_x
    dy = target_y - cur_y
    distance = math.hypot(dx, dy)
    if distance <= _MAX_STEP_TELEPORT_M or distance == 0.0:
        return (target_x, target_y, True)
    scale = _MAX_STEP_TELEPORT_M / distance
    return (cur_x + dx * scale, cur_y + dy * scale, False)


def test_clamp_short_move_reaches_target():
    nx, ny, reached = _clamped_step(1.0, 1.0, 1.02, 1.04)
    assert reached is True
    assert math.isclose(nx, 1.02, abs_tol=1e-9)
    assert math.isclose(ny, 1.04, abs_tol=1e-9)


def test_clamp_long_move_advances_at_cap():
    nx, ny, reached = _clamped_step(0.0, 0.0, 4.0, 0.0)
    assert reached is False
    # 4.0m target, cap 0.12 → exactly 0.12m forward, 0.0 sideways.
    assert math.isclose(nx, 0.120, abs_tol=1e-9)
    assert math.isclose(ny, 0.000, abs_tol=1e-9)


def test_clamp_diagonal_move_preserves_direction():
    nx, ny, reached = _clamped_step(0.0, 0.0, 3.0, 4.0)
    assert reached is False
    # Distance 5.0, cap 0.12 → scale 0.024.
    assert math.isclose(nx, 0.072, abs_tol=1e-9)
    assert math.isclose(ny, 0.096, abs_tol=1e-9)
    # Direction preserved: ratio (ny/nx) == (4/3).
    assert math.isclose(ny / nx, 4.0 / 3.0, rel_tol=1e-9)


def test_clamp_zero_distance_is_no_op():
    nx, ny, reached = _clamped_step(2.5, 1.5, 2.5, 1.5)
    assert reached is True
    assert (nx, ny) == (2.5, 1.5)


def test_clamp_at_exact_cap_treated_as_reached():
    # When distance == cap, the requested target is allowed in one step.
    nx, ny, reached = _clamped_step(0.0, 0.0, _MAX_STEP_TELEPORT_M, 0.0)
    assert reached is True
    assert math.isclose(nx, _MAX_STEP_TELEPORT_M, abs_tol=1e-12)
    assert math.isclose(ny, 0.0, abs_tol=1e-12)


def test_clamp_just_above_cap_is_clamped():
    target = _MAX_STEP_TELEPORT_M + 1.0e-3
    nx, ny, reached = _clamped_step(0.0, 0.0, target, 0.0)
    assert reached is False
    assert math.isclose(nx, _MAX_STEP_TELEPORT_M, abs_tol=1e-9)
    assert math.isclose(ny, 0.0, abs_tol=1e-9)
