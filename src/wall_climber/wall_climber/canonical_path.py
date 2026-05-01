from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias


Point2D: TypeAlias = tuple[float, float]


def _validate_point(point: Point2D, *, field_name: str) -> None:
    if len(point) != 2:
        raise ValueError(f'{field_name} must contain exactly two coordinates.')
    x = float(point[0])
    y = float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f'{field_name} coordinates must be finite.')


def _validate_scalar(value: float, *, field_name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f'{field_name} must be finite.')


@dataclass(frozen=True)
class PenUp:
    pass


@dataclass(frozen=True)
class PenDown:
    pass


@dataclass(frozen=True)
class TravelMove:
    start: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        _validate_point(self.start, field_name='TravelMove.start')
        _validate_point(self.end, field_name='TravelMove.end')


@dataclass(frozen=True)
class LineSegment:
    start: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        _validate_point(self.start, field_name='LineSegment.start')
        _validate_point(self.end, field_name='LineSegment.end')


@dataclass(frozen=True)
class ArcSegment:
    center: Point2D
    radius: float
    start_angle_rad: float
    sweep_angle_rad: float

    def __post_init__(self) -> None:
        _validate_point(self.center, field_name='ArcSegment.center')
        _validate_scalar(self.radius, field_name='ArcSegment.radius')
        _validate_scalar(self.start_angle_rad, field_name='ArcSegment.start_angle_rad')
        _validate_scalar(self.sweep_angle_rad, field_name='ArcSegment.sweep_angle_rad')
        if float(self.radius) <= 0.0:
            raise ValueError('ArcSegment.radius must be > 0.')


@dataclass(frozen=True)
class QuadraticBezier:
    start: Point2D
    control: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        _validate_point(self.start, field_name='QuadraticBezier.start')
        _validate_point(self.control, field_name='QuadraticBezier.control')
        _validate_point(self.end, field_name='QuadraticBezier.end')


@dataclass(frozen=True)
class CubicBezier:
    start: Point2D
    control1: Point2D
    control2: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        _validate_point(self.start, field_name='CubicBezier.start')
        _validate_point(self.control1, field_name='CubicBezier.control1')
        _validate_point(self.control2, field_name='CubicBezier.control2')
        _validate_point(self.end, field_name='CubicBezier.end')


CanonicalCommand: TypeAlias = (
    PenUp
    | PenDown
    | TravelMove
    | LineSegment
    | ArcSegment
    | QuadraticBezier
    | CubicBezier
)


@dataclass(frozen=True)
class CanonicalPathPlan:
    frame: str
    theta_ref: float
    commands: tuple[CanonicalCommand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError('CanonicalPathPlan.frame must be a non-empty string.')
        _validate_scalar(self.theta_ref, field_name='CanonicalPathPlan.theta_ref')
        if not isinstance(self.commands, tuple) or not self.commands:
            raise ValueError('CanonicalPathPlan.commands must be a non-empty tuple.')
        for index, command in enumerate(self.commands):
            if not isinstance(
                command,
                (
                    PenUp,
                    PenDown,
                    TravelMove,
                    LineSegment,
                    ArcSegment,
                    QuadraticBezier,
                    CubicBezier,
                ),
            ):
                raise ValueError(
                    f'CanonicalPathPlan.commands[{index}] has unsupported type {type(command)!r}.'
                )

    @property
    def command_count(self) -> int:
        return len(self.commands)

    @property
    def primitive_count(self) -> int:
        return sum(
            isinstance(
                command,
                (
                    TravelMove,
                    LineSegment,
                    ArcSegment,
                    QuadraticBezier,
                    CubicBezier,
                ),
            )
            for command in self.commands
        )
