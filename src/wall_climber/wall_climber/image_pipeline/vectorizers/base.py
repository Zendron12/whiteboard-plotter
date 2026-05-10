from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wall_climber.canonical_path import CanonicalPathPlan


@dataclass(frozen=True)
class VectorizationEngineResult:
    engine_name: str
    available: bool
    canonical_plan: CanonicalPathPlan | None = None
    svg_output: str | None = None
    preview_svg: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def unavailable_reason(self) -> str:
        return self.warnings[0] if self.warnings else f'{self.engine_name} is unavailable.'
