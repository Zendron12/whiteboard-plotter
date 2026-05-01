from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
except ImportError:  # pragma: no cover - used outside a sourced ROS environment
    class PackageNotFoundError(Exception):
        pass

    def get_package_share_directory(_package_name: str) -> str:
        raise PackageNotFoundError(_package_name)


PACKAGE_NAME = 'wall_climber'
CONFIG_RELATIVE_PATH = Path('config') / 'x_plotter.yaml'


@dataclass(frozen=True)
class BoardFrameConfig:
    width: float
    height: float
    center_x: float
    center_z: float
    surface_y: float
    carriage_plane_y: float
    margin_left: float
    margin_right: float
    margin_top: float
    margin_bottom: float
    origin: str = 'top_left'
    x_direction: str = 'right'
    y_direction: str = 'down'

    @property
    def board_left(self) -> float:
        return self.center_x - (self.width * 0.5)

    @property
    def board_top_z(self) -> float:
        return self.center_z + (self.height * 0.5)

    @property
    def drawable_x_min(self) -> float:
        return self.margin_left

    @property
    def drawable_x_max(self) -> float:
        return self.width - self.margin_right

    @property
    def drawable_y_min(self) -> float:
        return self.margin_top

    @property
    def drawable_y_max(self) -> float:
        return self.height - self.margin_bottom

    def board_to_world(self, x: float, y: float, *, plane_y: float | None = None) -> tuple[float, float, float]:
        """Map board coordinates to Webots coordinates.

        Board frame convention: origin at top-left, +X right, +Y down.
        Webots frame convention used by this world: X right, Y depth, Z up.
        """
        return (
            self.board_left + float(x),
            self.carriage_plane_y if plane_y is None else float(plane_y),
            self.board_top_z - float(y),
        )

    def in_drawable_bounds(self, x: float, y: float, *, eps: float = 1.0e-9) -> bool:
        return (
            self.drawable_x_min - eps <= float(x) <= self.drawable_x_max + eps
            and self.drawable_y_min - eps <= float(y) <= self.drawable_y_max + eps
        )

    def clamp_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(float(x), self.drawable_x_min), self.drawable_x_max),
            min(max(float(y), self.drawable_y_min), self.drawable_y_max),
        )

    def board_info_payload(self) -> dict[str, Any]:
        return {
            'frame_origin': self.origin,
            'frame_x_axis': self.x_direction,
            'frame_y_axis': self.y_direction,
            'width': self.width,
            'height': self.height,
            'writable_x_min': self.drawable_x_min,
            'writable_x_max': self.drawable_x_max,
            'writable_y_min': self.drawable_y_min,
            'writable_y_max': self.drawable_y_max,
            'safe_x_min': self.drawable_x_min,
            'safe_x_max': self.drawable_x_max,
            'safe_y_min': self.drawable_y_min,
            'safe_y_max': self.drawable_y_max,
            'corexy': {
                'kinematics': 'a=x+y,b=x-y',
                'execution': 'primitive_direct_foundation',
            },
        }


def _source_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        candidates.append(Path(get_package_share_directory(PACKAGE_NAME)) / CONFIG_RELATIVE_PATH)
    except PackageNotFoundError:
        pass
    candidates.append(Path(__file__).resolve().parents[2] / CONFIG_RELATIVE_PATH)
    return candidates


def _read_yaml_config() -> dict[str, Any]:
    for path in _source_candidates():
        if path.is_file():
            with path.open('r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle)
            if not isinstance(data, dict):
                raise RuntimeError(f'Config file {path} must contain a YAML object.')
            return data
    raise FileNotFoundError(f'Unable to locate {CONFIG_RELATIVE_PATH} for {PACKAGE_NAME}.')


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise RuntimeError(f'Missing or invalid X plotter config section: {name}')
    return section


def board_frame_from_dict(data: dict[str, Any]) -> BoardFrameConfig:
    board = _section(data, 'board')
    carriage = _section(data, 'carriage')
    return BoardFrameConfig(
        width=float(board['width']),
        height=float(board['height']),
        center_x=float(board['center_x']),
        center_z=float(board['center_z']),
        surface_y=float(board['surface_y']),
        carriage_plane_y=float(carriage['plane_y']),
        margin_left=float(board['margin_left']),
        margin_right=float(board['margin_right']),
        margin_top=float(board['margin_top']),
        margin_bottom=float(board['margin_bottom']),
        origin=str(board.get('origin', 'top_left')),
        x_direction=str(board.get('x_direction', 'right')),
        y_direction=str(board.get('y_direction', 'down')),
    )


@lru_cache(maxsize=1)
def load_board_frame_config() -> BoardFrameConfig:
    return board_frame_from_dict(_read_yaml_config())

