from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
except ImportError:
    class PackageNotFoundError(Exception):
        pass

    def get_package_share_directory(_package_name: str) -> str:
        raise PackageNotFoundError(_package_name)


PACKAGE_NAME = 'wall_climber'
CONFIG_RELATIVE_PATH = Path('config') / 'cable_robot.yaml'


@dataclass(frozen=True)
class BoardDefaults:
    width: float
    height: float
    center_x: float
    center_z: float
    surface_y: float
    margin_left: float
    margin_right: float
    margin_top: float
    margin_bottom: float
    line_height: float


@dataclass(frozen=True)
class AnchorDefaults:
    left_x: float
    left_y: float
    right_x: float
    right_y: float


@dataclass(frozen=True)
class WorkspaceDefaults:
    safety_margin_side: float
    safety_margin_bottom: float
    top_clearance: float
    corner_keepout_radius: float


@dataclass(frozen=True)
class CarriageDefaults:
    width: float
    height: float
    depth: float
    mass: float
    root_to_chassis_z: float
    plane_y: float
    attachment_left_x: float
    attachment_left_y: float
    attachment_right_x: float
    attachment_right_y: float
    pen_offset_x: float
    pen_offset_y: float
    initial_center_x: float
    initial_center_y: float


@dataclass(frozen=True)
class PenDefaults:
    mount_length: float
    radius: float
    tip_radius: float
    tip_local_z: float
    support_radius: float
    slide_velocity: float
    up_position: float
    down_position: float
    gripper_velocity: float
    gripper_closed_position: float
    gripper_open_position: float
    settle_sec: float
    contact_engage_gap: float
    contact_release_gap: float


@dataclass(frozen=True)
class TextLayoutDefaults:
    left_margin: float
    right_margin: float
    top_margin: float
    bottom_margin: float
    glyph_height: float
    letter_spacing: float
    word_spacing: float
    uppercase_advance_scale: float
    glyph_cluster_max_chars: int
    text_upward_bias_em: float


@dataclass(frozen=True)
class TextVectorDefaults:
    simplify_epsilon_m: float
    resample_step_m: float
    min_feature_len_m: float
    max_points_per_stroke: int
    max_total_points: int


@dataclass(frozen=True)
class DrawExecutionDefaults:
    fixed_draw_theta_rad: float
    draw_scale_fit_margin_m: float
    draw_path_simplify_tolerance_m: float
    draw_resample_step_m: float
    travel_resample_step_m: float
    publish_period_sec: float


@dataclass(frozen=True)
class VisualDefaults:
    trail_enabled_default: bool
    trail_half_width: float
    trail_round_segments: int
    trail_min_spacing: float
    trail_max: int


@dataclass(frozen=True)
class SharedConfig:
    board: BoardDefaults
    anchors: AnchorDefaults
    workspace: WorkspaceDefaults
    carriage: CarriageDefaults
    pen: PenDefaults
    text_layout: TextLayoutDefaults
    text_vector: TextVectorDefaults
    draw_execution: DrawExecutionDefaults
    visuals: VisualDefaults

    @property
    def board_left(self) -> float:
        return self.board.center_x - (self.board.width * 0.5)

    @property
    def board_top_z(self) -> float:
        return self.board.center_z + (self.board.height * 0.5)

    @property
    def writable_x_min(self) -> float:
        return self.board.margin_left

    @property
    def writable_x_max(self) -> float:
        return self.board.width - self.board.margin_right

    @property
    def writable_y_min(self) -> float:
        return self.board.margin_top

    @property
    def writable_y_max(self) -> float:
        return self.board.height - self.board.margin_bottom

    @property
    def safe_x_min(self) -> float:
        return self.writable_x_min + self.workspace.safety_margin_side

    @property
    def safe_x_max(self) -> float:
        return self.writable_x_max - self.workspace.safety_margin_side

    @property
    def safe_y_min(self) -> float:
        return self.writable_y_min + self.workspace.top_clearance

    @property
    def safe_y_max(self) -> float:
        return self.writable_y_max - self.workspace.safety_margin_bottom

    @property
    def carriage_center_x_min(self) -> float:
        return self.carriage.width * 0.5

    @property
    def carriage_center_x_max(self) -> float:
        return self.board.width - (self.carriage.width * 0.5)

    @property
    def carriage_center_y_min(self) -> float:
        return self.carriage.height * 0.5

    @property
    def carriage_center_y_max(self) -> float:
        return self.board.height - (self.carriage.height * 0.5)

    @property
    def carriage_safe_pen_x_min(self) -> float:
        return self.carriage_center_x_min + self.carriage.pen_offset_x

    @property
    def carriage_safe_pen_x_max(self) -> float:
        return self.carriage_center_x_max + self.carriage.pen_offset_x

    @property
    def carriage_safe_pen_y_min(self) -> float:
        return self.carriage_center_y_min + self.carriage.pen_offset_y

    @property
    def carriage_safe_pen_y_max(self) -> float:
        return self.carriage_center_y_max + self.carriage.pen_offset_y

    def writable_bounds(self) -> dict[str, float]:
        return {
            'x_min': self.writable_x_min,
            'x_max': self.writable_x_max,
            'y_min': self.writable_y_min,
            'y_max': self.writable_y_max,
        }

    def safe_bounds(self) -> dict[str, float]:
        if self.safe_x_max <= self.safe_x_min or self.safe_y_max <= self.safe_y_min:
            raise RuntimeError('Configured safe workspace collapsed after margins were applied.')
        return {
            'x_min': self.safe_x_min,
            'x_max': self.safe_x_max,
            'y_min': self.safe_y_min,
            'y_max': self.safe_y_max,
        }

    def carriage_safe_pen_bounds(self) -> dict[str, float]:
        if (
            self.carriage_center_x_max <= self.carriage_center_x_min or
            self.carriage_center_y_max <= self.carriage_center_y_min
        ):
            raise RuntimeError('Configured carriage dimensions collapse the board-safe center region.')
        return {
            'x_min': self.carriage_safe_pen_x_min,
            'x_max': self.carriage_safe_pen_x_max,
            'y_min': self.carriage_safe_pen_y_min,
            'y_max': self.carriage_safe_pen_y_max,
        }

    def carriage_safe_writable_bounds(self) -> dict[str, float]:
        writable = self.writable_bounds()
        carriage_pen = self.carriage_safe_pen_bounds()
        bounds = {
            'x_min': max(writable['x_min'], carriage_pen['x_min']),
            'x_max': min(writable['x_max'], carriage_pen['x_max']),
            'y_min': max(writable['y_min'], carriage_pen['y_min']),
            'y_max': min(writable['y_max'], carriage_pen['y_max']),
        }
        if bounds['x_max'] <= bounds['x_min'] or bounds['y_max'] <= bounds['y_min']:
            raise RuntimeError('Configured carriage-safe writable region collapsed after intersections.')
        return bounds

    def carriage_safe_workspace_bounds(self) -> dict[str, float]:
        safe = self.safe_bounds()
        carriage_pen = self.carriage_safe_pen_bounds()
        bounds = {
            'x_min': max(safe['x_min'], carriage_pen['x_min']),
            'x_max': min(safe['x_max'], carriage_pen['x_max']),
            'y_min': max(safe['y_min'], carriage_pen['y_min']),
            'y_max': min(safe['y_max'], carriage_pen['y_max']),
        }
        if bounds['x_max'] <= bounds['x_min'] or bounds['y_max'] <= bounds['y_min']:
            raise RuntimeError('Configured carriage-safe workspace region collapsed after intersections.')
        return bounds

    def point_keeps_carriage_on_board(self, x: float, y: float) -> bool:
        center_x = x - self.carriage.pen_offset_x
        center_y = y - self.carriage.pen_offset_y
        return (
            self.carriage_center_x_min <= center_x <= self.carriage_center_x_max and
            self.carriage_center_y_min <= center_y <= self.carriage_center_y_max
        )

    def point_in_safe_workspace(self, x: float, y: float) -> bool:
        if not (self.safe_x_min <= x <= self.safe_x_max and self.safe_y_min <= y <= self.safe_y_max):
            return False
        left_dist_sq = (x - self.anchors.left_x) ** 2 + (y - self.anchors.left_y) ** 2
        right_dist_sq = (x - self.anchors.right_x) ** 2 + (y - self.anchors.right_y) ** 2
        min_dist_sq = self.workspace.corner_keepout_radius ** 2
        return left_dist_sq >= min_dist_sq and right_dist_sq >= min_dist_sq

    def initial_spawn_translation(self) -> tuple[float, float, float]:
        world_x = self.board_left + self.carriage.initial_center_x
        world_z = self.board_top_z - self.carriage.initial_center_y
        return (world_x, self.carriage.plane_y, world_z)

    def initial_spawn_translation_str(self) -> str:
        x, y, z = self.initial_spawn_translation()
        return f'{x:.4f} {y:.4f} {z:.4f}'

    def cable_executor_params(self) -> dict[str, Any]:
        safe = self.safe_bounds()
        writable = self.writable_bounds()
        body_safe_writable = self.carriage_safe_writable_bounds()
        body_safe_safe = self.carriage_safe_workspace_bounds()
        return {
            'anchor_left_x': self.anchors.left_x,
            'anchor_left_y': self.anchors.left_y,
            'anchor_right_x': self.anchors.right_x,
            'anchor_right_y': self.anchors.right_y,
            'carriage_attachment_left_x': self.carriage.attachment_left_x,
            'carriage_attachment_left_y': self.carriage.attachment_left_y,
            'carriage_attachment_right_x': self.carriage.attachment_right_x,
            'carriage_attachment_right_y': self.carriage.attachment_right_y,
            'pen_offset_x': self.carriage.pen_offset_x,
            'pen_offset_y': self.carriage.pen_offset_y,
            'initial_pen_x': self.carriage.initial_center_x + self.carriage.pen_offset_x,
            'initial_pen_y': self.carriage.initial_center_y + self.carriage.pen_offset_y,
            'fixed_theta_rad': self.draw_execution.fixed_draw_theta_rad,
            'draw_resample_step_m': self.draw_execution.draw_resample_step_m,
            'travel_resample_step_m': self.draw_execution.travel_resample_step_m,
            'publish_period_sec': self.draw_execution.publish_period_sec,
            'writable_x_min': writable['x_min'],
            'writable_x_max': writable['x_max'],
            'writable_y_min': writable['y_min'],
            'writable_y_max': writable['y_max'],
            'safe_x_min': safe['x_min'],
            'safe_x_max': safe['x_max'],
            'safe_y_min': safe['y_min'],
            'safe_y_max': safe['y_max'],
            'body_safe_writable_x_min': body_safe_writable['x_min'],
            'body_safe_writable_x_max': body_safe_writable['x_max'],
            'body_safe_writable_y_min': body_safe_writable['y_min'],
            'body_safe_writable_y_max': body_safe_writable['y_max'],
            'body_safe_safe_x_min': body_safe_safe['x_min'],
            'body_safe_safe_x_max': body_safe_safe['x_max'],
            'body_safe_safe_y_min': body_safe_safe['y_min'],
            'body_safe_safe_y_max': body_safe_safe['y_max'],
            'corner_keepout_radius': self.workspace.corner_keepout_radius,
            'pen_down_settle_sec': self.pen.settle_sec,
        }


def _source_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        candidates.append(Path(get_package_share_directory(PACKAGE_NAME)) / CONFIG_RELATIVE_PATH)
    except PackageNotFoundError:
        pass
    candidates.append(Path(__file__).resolve().parents[1] / CONFIG_RELATIVE_PATH)
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
        raise RuntimeError(f'Missing or invalid config section: {name}')
    return section


@lru_cache(maxsize=1)
def load_shared_config() -> SharedConfig:
    data = _read_yaml_config()
    return SharedConfig(
        board=BoardDefaults(**_section(data, 'board')),
        anchors=AnchorDefaults(**_section(data, 'anchors')),
        workspace=WorkspaceDefaults(**_section(data, 'workspace')),
        carriage=CarriageDefaults(**_section(data, 'carriage')),
        pen=PenDefaults(**_section(data, 'pen')),
        text_layout=TextLayoutDefaults(**_section(data, 'text_layout')),
        text_vector=TextVectorDefaults(**_section(data, 'text_vector')),
        draw_execution=DrawExecutionDefaults(**_section(data, 'draw_execution')),
        visuals=VisualDefaults(**_section(data, 'visuals')),
    )
