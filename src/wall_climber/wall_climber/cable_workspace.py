from __future__ import annotations

from typing import Iterable

from wall_climber.shared_config import SharedConfig


def point_in_safe_workspace(x: float, y: float, config: SharedConfig) -> bool:
    return config.point_in_safe_workspace(float(x), float(y))


def count_points_outside_safe_workspace(
    strokes: Iterable[Iterable[tuple[float, float]]],
    config: SharedConfig,
) -> int:
    outside = 0
    for stroke in strokes:
        for x, y in stroke:
            if not point_in_safe_workspace(x, y, config):
                outside += 1
    return outside


def safe_workspace_metadata(config: SharedConfig) -> dict[str, float]:
    safe = config.safe_bounds()
    return {
        'safe_x_min': safe['x_min'],
        'safe_x_max': safe['x_max'],
        'safe_y_min': safe['y_min'],
        'safe_y_max': safe['y_max'],
        'corner_keepout_radius': config.workspace.corner_keepout_radius,
    }
