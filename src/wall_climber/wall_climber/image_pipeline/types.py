from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Mapping


class PipelineMode(str, Enum):
    """Future image/vector pipeline mode identifiers."""

    SKETCH_CENTERLINE = 'sketch_centerline'
    SVG_VECTOR = 'svg_vector'
    PHOTO_OUTLINE = 'photo_outline'
    HATCHING = 'hatching'
    TEXT = 'text'


@dataclass(frozen=True)
class Point2D:
    """Board-space point in meters."""

    x: float
    y: float

    def __post_init__(self) -> None:
        x = float(self.x)
        y = float(self.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError('Point2D coordinates must be finite.')
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'y', y)


def _coerce_point(point: Point2D | Iterable[float]) -> Point2D:
    if isinstance(point, Point2D):
        return point
    values = tuple(point)
    if len(values) != 2:
        raise ValueError('Point values must contain exactly two coordinates.')
    return Point2D(float(values[0]), float(values[1]))


@dataclass(frozen=True)
class Stroke:
    """One ordered stroke in board coordinates."""

    points: tuple[Point2D, ...]
    pen_down: bool = True
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(_coerce_point(point) for point in self.points)
        if len(points) < 2:
            raise ValueError('Stroke.points must contain at least two points.')
        object.__setattr__(self, 'points', points)
        object.__setattr__(self, 'pen_down', bool(self.pen_down))
        object.__setattr__(self, 'metadata', dict(self.metadata))


@dataclass(frozen=True)
class PipelineMetrics:
    """Source-neutral metrics produced by future pipeline stages."""

    stroke_count: int = 0
    points_before_simplification: int = 0
    points_after_simplification: int = 0
    total_drawing_length_m: float = 0.0
    pen_up_travel_length_m: float = 0.0
    pen_lift_count: int = 0
    estimated_drawing_time_sec: float | None = None
    processing_time_ms: float | None = None
    similarity_score: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            'stroke_count',
            'points_before_simplification',
            'points_after_simplification',
            'pen_lift_count',
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f'PipelineMetrics.{field_name} must be >= 0.')
            object.__setattr__(self, field_name, value)

        for field_name in ('total_drawing_length_m', 'pen_up_travel_length_m'):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'PipelineMetrics.{field_name} must be finite and >= 0.')
            object.__setattr__(self, field_name, value)

        for field_name in ('estimated_drawing_time_sec', 'processing_time_ms', 'similarity_score'):
            value = getattr(self, field_name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f'PipelineMetrics.{field_name} must be finite when provided.')
            if field_name != 'similarity_score' and numeric < 0.0:
                raise ValueError(f'PipelineMetrics.{field_name} must be >= 0.')
            object.__setattr__(self, field_name, numeric)

        object.__setattr__(self, 'warnings', tuple(str(item) for item in self.warnings))


@dataclass(frozen=True)
class DrawingPathPlan:
    """Future source-neutral path plan emitted by image, SVG, text, or library pipelines."""

    mode: PipelineMode
    strokes: tuple[Stroke, ...]
    frame: str = 'board'
    source_id: str | None = None
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, PipelineMode) else PipelineMode(str(self.mode))
        strokes = tuple(self.strokes)
        if not strokes:
            raise ValueError('DrawingPathPlan.strokes must contain at least one stroke.')
        for index, stroke in enumerate(strokes):
            if not isinstance(stroke, Stroke):
                raise ValueError(f'DrawingPathPlan.strokes[{index}] must be a Stroke.')
        frame = str(self.frame).strip()
        if not frame:
            raise ValueError('DrawingPathPlan.frame must be a non-empty string.')

        object.__setattr__(self, 'mode', mode)
        object.__setattr__(self, 'strokes', strokes)
        object.__setattr__(self, 'frame', frame)
        object.__setattr__(self, 'metadata', dict(self.metadata))
