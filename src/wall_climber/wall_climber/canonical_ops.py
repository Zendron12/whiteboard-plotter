from __future__ import annotations

from wall_climber.vector_pipeline import (
    PlacementResult,
    VectorPlacement,
    cleanup_canonical_plan,
    default_image_placement,
    normalize_placement,
    place_canonical_plan_on_board,
    place_grouped_text_on_board,
    stroke_stats,
)

__all__ = [
    'PlacementResult',
    'VectorPlacement',
    'cleanup_canonical_plan',
    'default_image_placement',
    'normalize_placement',
    'place_canonical_plan_on_board',
    'place_grouped_text_on_board',
    'stroke_stats',
]
