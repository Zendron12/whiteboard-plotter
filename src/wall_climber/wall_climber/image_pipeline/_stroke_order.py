"""Stroke-order optimisation (nearest-neighbour seed + bounded 2-opt).

Reorders strokes so that the end of stroke N is close to the start of
stroke N+1, optionally reversing strokes when the reverse is closer.
This minimises pen-up travel between strokes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]


@dataclass(frozen=True)
class Stroke:
    points: tuple[Point, ...]

    @property
    def start(self) -> Point:
        return self.points[0]

    @property
    def end(self) -> Point:
        return self.points[-1]


@dataclass(frozen=True)
class OrderingResult:
    strokes: tuple[Stroke, ...]
    travel_length_m: float
    iterations: int


def optimise_stroke_order(
    strokes: list[Stroke] | tuple[Stroke, ...],
    *,
    allow_reverse: bool = True,
    max_2opt_iterations: int = 200,
) -> OrderingResult:
    """Reorder ``strokes`` so per-stroke pen-up travel is minimised."""
    materialised = list(strokes)
    if len(materialised) < 2:
        return OrderingResult(
            strokes=tuple(materialised),
            travel_length_m=0.0,
            iterations=0,
        )

    # Step 1: nearest-neighbour seed.
    ordered: list[Stroke] = []
    remaining = list(materialised)
    current = remaining.pop(0)
    ordered.append(current)
    while remaining:
        best_index = 0
        best_cost = float('inf')
        best_reversed = False
        for index, candidate in enumerate(remaining):
            cost = _distance(current.end, candidate.start)
            reversed_cost = (
                _distance(current.end, candidate.end) if allow_reverse else float('inf')
            )
            if cost <= reversed_cost and cost < best_cost:
                best_cost = cost
                best_index = index
                best_reversed = False
            elif allow_reverse and reversed_cost < best_cost:
                best_cost = reversed_cost
                best_index = index
                best_reversed = True
        chosen = remaining.pop(best_index)
        if best_reversed:
            chosen = Stroke(points=tuple(reversed(chosen.points)))
        ordered.append(chosen)
        current = chosen

    # Step 2: bounded 2-opt refinement.
    iteration = 0
    improved = True
    while improved and iteration < max_2opt_iterations:
        improved = False
        iteration += 1
        for i in range(0, len(ordered) - 1):
            for j in range(i + 2, len(ordered)):
                a_end = ordered[i].end
                b_start = ordered[i + 1].start
                c_end = ordered[j].end
                d_start = ordered[j + 1].start if j + 1 < len(ordered) else None

                old_cost = _distance(a_end, b_start)
                if d_start is not None:
                    old_cost += _distance(c_end, d_start)
                new_cost = _distance(a_end, c_end)
                if d_start is not None:
                    new_cost += _distance(b_start, d_start)

                if new_cost + 1.0e-12 < old_cost:
                    reversed_segment = [
                        Stroke(points=tuple(reversed(s.points)))
                        for s in reversed(ordered[i + 1 : j + 1])
                    ]
                    ordered[i + 1 : j + 1] = reversed_segment
                    improved = True
                    break
            if improved:
                break

    travel = _total_travel(ordered)
    return OrderingResult(
        strokes=tuple(ordered),
        travel_length_m=travel,
        iterations=iteration,
    )


def _total_travel(ordered: list[Stroke]) -> float:
    if len(ordered) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(ordered)):
        total += _distance(ordered[index - 1].end, ordered[index].start)
    return total


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


__all__ = ['Stroke', 'OrderingResult', 'optimise_stroke_order']
