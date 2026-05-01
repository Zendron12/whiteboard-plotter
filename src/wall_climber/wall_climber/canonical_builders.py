from __future__ import annotations

from typing import TYPE_CHECKING

from wall_climber.canonical_path import (
    CanonicalPathPlan,
    LineSegment,
    PenDown,
    PenUp,
    Point2D,
    TravelMove,
)


_EPS = 1.0e-9

if TYPE_CHECKING:
    from wall_climber.vector_pipeline import TextGlyphOutline


def _distance(a: Point2D, b: Point2D) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _approximately_equal(a: Point2D, b: Point2D, *, eps: float = _EPS) -> bool:
    return _distance(a, b) <= eps


def _append_draw_stroke_commands(
    commands: list[object],
    stroke: tuple[Point2D, ...],
    *,
    previous_end: Point2D | None,
    pen_is_down: bool,
) -> tuple[Point2D | None, bool]:
    if len(stroke) < 2:
        return previous_end, pen_is_down

    start_point = stroke[0]
    if previous_end is not None and not _approximately_equal(previous_end, start_point):
        if pen_is_down:
            commands.append(PenUp())
            pen_is_down = False
        commands.append(TravelMove(start=previous_end, end=start_point))
    if not pen_is_down:
        commands.append(PenDown())
        pen_is_down = True

    last_point = start_point
    for index in range(1, len(stroke)):
        current_point = stroke[index]
        if _approximately_equal(last_point, current_point):
            continue
        commands.append(LineSegment(start=last_point, end=current_point))
        last_point = current_point
    return last_point, pen_is_down


def draw_strokes_to_canonical_plan(
    strokes: tuple[tuple[Point2D, ...], ...],
    *,
    theta_ref: float,
    frame: str = 'board',
) -> CanonicalPathPlan:
    if not strokes:
        raise ValueError('No draw strokes available to convert into a canonical plan.')

    commands: list[object] = []
    previous_end: Point2D | None = None
    pen_is_down = False

    for stroke in strokes:
        previous_end, pen_is_down = _append_draw_stroke_commands(
            commands,
            tuple((float(point[0]), float(point[1])) for point in stroke),
            previous_end=previous_end,
            pen_is_down=pen_is_down,
        )

    if not commands:
        raise ValueError('No drawable strokes available to convert into a canonical plan.')
    if pen_is_down:
        commands.append(PenUp())

    return CanonicalPathPlan(
        frame=frame,
        theta_ref=float(theta_ref),
        commands=tuple(commands),
    )


def text_glyph_outlines_to_canonical_plan(
    glyphs: tuple['TextGlyphOutline', ...],
    *,
    theta_ref: float,
    frame: str = 'board',
) -> CanonicalPathPlan:
    if not glyphs:
        raise ValueError('No grouped text glyphs available to convert into a canonical plan.')

    commands: list[object] = []
    previous_end: Point2D | None = None
    pen_is_down = False

    for glyph in glyphs:
        for stroke in glyph.strokes:
            previous_end, pen_is_down = _append_draw_stroke_commands(
                commands,
                tuple((float(point[0]), float(point[1])) for point in stroke),
                previous_end=previous_end,
                pen_is_down=pen_is_down,
            )

    if not commands:
        raise ValueError('Grouped text glyphs produced no drawable canonical commands.')
    if pen_is_down:
        commands.append(PenUp())

    return CanonicalPathPlan(
        frame=frame,
        theta_ref=float(theta_ref),
        commands=tuple(commands),
    )
