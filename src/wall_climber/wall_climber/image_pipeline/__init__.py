"""Neutral image-pipeline type definitions for future plan builders."""

from wall_climber.image_pipeline.adapters import drawing_path_plan_to_canonical
from wall_climber.image_pipeline.types import (
    DrawingPathPlan,
    PipelineMetrics,
    PipelineMode,
    Point2D,
    Stroke,
)

__all__ = [
    'DrawingPathPlan',
    'PipelineMetrics',
    'PipelineMode',
    'Point2D',
    'Stroke',
    'drawing_path_plan_to_canonical',
    'vectorize_sketch_image_to_plan',
]


def __getattr__(name: str):
    if name == 'vectorize_sketch_image_to_plan':
        from wall_climber.image_pipeline.sketch_centerline import vectorize_sketch_image_to_plan

        return vectorize_sketch_image_to_plan
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
