from __future__ import annotations

from typing import Any

from wall_climber.canonical_builders import draw_strokes_to_canonical_plan
from wall_climber.canonical_path import CanonicalPathPlan
from wall_climber.vector_pipeline import vectorize_svg


def svg_text_to_canonical_plan(
    svg_text: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> CanonicalPathPlan:
    """Parse SVG paths into a CanonicalPathPlan in the SVG coordinate frame."""

    strokes = vectorize_svg(svg_text)
    plan = draw_strokes_to_canonical_plan(strokes, frame='svg', theta_ref=0.0)
    return plan
