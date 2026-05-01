from __future__ import annotations


def xy_to_motors(x: float, y: float) -> tuple[float, float]:
    """Convert board/cartesian X/Y position to CoreXY motor coordinates."""
    return float(x) + float(y), float(x) - float(y)


def motors_to_xy(a: float, b: float) -> tuple[float, float]:
    """Convert CoreXY motor coordinates back to board/cartesian X/Y."""
    return (float(a) + float(b)) / 2.0, (float(a) - float(b)) / 2.0


def delta_xy_to_delta_motors(dx: float, dy: float) -> tuple[float, float]:
    """Convert a board/cartesian X/Y delta to CoreXY motor deltas."""
    return xy_to_motors(dx, dy)


def delta_motors_to_delta_xy(da: float, db: float) -> tuple[float, float]:
    """Convert CoreXY motor deltas back to board/cartesian X/Y deltas."""
    return motors_to_xy(da, db)

