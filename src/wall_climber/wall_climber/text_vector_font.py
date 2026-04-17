from __future__ import annotations

from dataclasses import dataclass
import math


_AXIS_EPS = 1.0e-9


@dataclass(frozen=True)
class Glyph:
    strokes: tuple[tuple[tuple[float, float], ...], ...]
    advance: float


def _segments(*names: str) -> tuple[tuple[tuple[float, float], ...], ...]:
    strokes: list[tuple[tuple[float, float], ...]] = []
    for name in names:
        strokes.extend(_SEGMENTS[name])
    return tuple(strokes)


def _poly(*points: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    return tuple(points)


_SEGMENTS: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {
    'top': (_poly((0.15, 1.00), (0.85, 1.00)),),
    'middle': (_poly((0.15, 0.50), (0.85, 0.50)),),
    'bottom': (_poly((0.15, 0.00), (0.85, 0.00)),),
    'left_full': (_poly((0.15, 1.00), (0.15, 0.00)),),
    'left_upper': (_poly((0.15, 1.00), (0.15, 0.50)),),
    'left_lower': (_poly((0.15, 0.50), (0.15, 0.00)),),
    'center_full': (_poly((0.50, 1.00), (0.50, 0.00)),),
    'center_upper': (_poly((0.50, 1.00), (0.50, 0.50)),),
    'center_lower': (_poly((0.50, 0.50), (0.50, 0.00)),),
    'right_full': (_poly((0.85, 1.00), (0.85, 0.00)),),
    'right_upper': (_poly((0.85, 1.00), (0.85, 0.50)),),
    'right_lower': (_poly((0.85, 0.50), (0.85, 0.00)),),
    'inner_left_full': (_poly((0.35, 1.00), (0.35, 0.00)),),
    'inner_right_full': (_poly((0.65, 1.00), (0.65, 0.00)),),
    'middle_left': (_poly((0.15, 0.50), (0.50, 0.50)),),
    'middle_right': (_poly((0.50, 0.50), (0.85, 0.50)),),
    'bottom_left': (_poly((0.15, 0.00), (0.50, 0.00)),),
    'bottom_right': (_poly((0.50, 0.00), (0.85, 0.00)),),
    'dot_tick': (_poly((0.50, 0.10), (0.50, 0.00)),),
    'comma_tick': (_poly((0.50, 0.04), (0.50, -0.10)),),
    'colon_top_tick': (_poly((0.50, 0.74), (0.50, 0.64)),),
    'colon_bottom_tick': (_poly((0.50, 0.26), (0.50, 0.16)),),
    'question_stem': (_poly((0.50, 0.50), (0.50, 0.22)),),
    'lc_top': (_poly((0.15, 0.72), (0.85, 0.72)),),
    'lc_middle': (_poly((0.15, 0.36), (0.85, 0.36)),),
    'lc_bottom': (_poly((0.15, 0.00), (0.85, 0.00)),),
    'lc_left_full': (_poly((0.15, 0.72), (0.15, 0.00)),),
    'lc_left_upper': (_poly((0.15, 0.72), (0.15, 0.36)),),
    'lc_left_lower': (_poly((0.15, 0.36), (0.15, 0.00)),),
    'lc_center_full': (_poly((0.50, 0.72), (0.50, 0.00)),),
    'lc_center_upper': (_poly((0.50, 0.72), (0.50, 0.36)),),
    'lc_center_lower': (_poly((0.50, 0.36), (0.50, 0.00)),),
    'lc_right_full': (_poly((0.85, 0.72), (0.85, 0.00)),),
    'lc_right_upper': (_poly((0.85, 0.72), (0.85, 0.36)),),
    'lc_right_lower': (_poly((0.85, 0.36), (0.85, 0.00)),),
    'lc_inner_left_full': (_poly((0.35, 0.72), (0.35, 0.00)),),
    'lc_inner_right_full': (_poly((0.65, 0.72), (0.65, 0.00)),),
    'lc_middle_left': (_poly((0.15, 0.36), (0.50, 0.36)),),
    'lc_middle_right': (_poly((0.50, 0.36), (0.85, 0.36)),),
    'lc_left_desc_full': (_poly((0.15, 0.72), (0.15, -0.20)),),
    'lc_center_desc_full': (_poly((0.50, 0.72), (0.50, -0.20)),),
    'lc_right_desc_full': (_poly((0.85, 0.72), (0.85, -0.20)),),
    'lc_dot': (_poly((0.50, 0.96), (0.50, 0.84)),),
    'j_hook': (_poly((0.50, 0.72), (0.50, -0.20), (0.15, -0.20)),),
}


_GLYPHS: dict[str, Glyph] = {
    ' ': Glyph(strokes=tuple(), advance=0.45),
    'A': Glyph(strokes=_segments('left_full', 'right_full', 'top', 'middle'), advance=1.00),
    'B': Glyph(strokes=_segments('left_full', 'top', 'middle', 'bottom', 'right_upper', 'right_lower'), advance=1.00),
    'C': Glyph(strokes=_segments('top', 'left_full', 'bottom'), advance=1.00),
    'D': Glyph(strokes=_segments('left_full', 'top', 'bottom', 'right_full'), advance=1.00),
    'E': Glyph(strokes=_segments('left_full', 'top', 'middle', 'bottom'), advance=1.00),
    'F': Glyph(strokes=_segments('left_full', 'top', 'middle'), advance=1.00),
    'G': Glyph(strokes=_segments('top', 'left_full', 'bottom', 'middle_right', 'right_lower'), advance=1.00),
    'H': Glyph(strokes=_segments('left_full', 'right_full', 'middle'), advance=1.00),
    'I': Glyph(strokes=_segments('top', 'center_full', 'bottom'), advance=0.80),
    'J': Glyph(strokes=(
        _poly((0.15, 1.00), (0.85, 1.00), (0.85, 0.00), (0.15, 0.00), (0.15, 0.20)),
    ), advance=1.00),
    'K': Glyph(strokes=_segments('left_full', 'middle', 'right_upper', 'right_lower'), advance=1.00),
    'L': Glyph(strokes=_segments('left_full', 'bottom'), advance=0.95),
    'M': Glyph(strokes=_segments('left_full', 'inner_left_full', 'inner_right_full', 'right_full', 'top'), advance=1.10),
    'N': Glyph(strokes=_segments('left_full', 'center_full', 'right_full'), advance=1.05),
    'O': Glyph(strokes=_segments('top', 'bottom', 'left_full', 'right_full'), advance=1.00),
    'P': Glyph(strokes=_segments('left_full', 'top', 'middle', 'right_upper'), advance=1.00),
    'Q': Glyph(strokes=_segments('top', 'bottom', 'left_full', 'right_full', 'center_lower'), advance=1.00),
    'R': Glyph(strokes=_segments('left_full', 'top', 'middle', 'right_upper', 'right_lower'), advance=1.00),
    'S': Glyph(strokes=_segments('top', 'left_upper', 'middle', 'right_lower', 'bottom'), advance=1.00),
    'T': Glyph(strokes=_segments('top', 'center_full'), advance=1.00),
    'U': Glyph(strokes=_segments('left_full', 'right_full', 'bottom'), advance=1.00),
    'V': Glyph(strokes=_segments('left_full', 'right_full', 'bottom'), advance=1.00),
    'W': Glyph(strokes=_segments('left_full', 'inner_left_full', 'inner_right_full', 'right_full', 'bottom'), advance=1.20),
    'X': Glyph(strokes=_segments('left_full', 'right_full', 'middle'), advance=1.00),
    'Y': Glyph(strokes=_segments('top', 'center_full'), advance=1.00),
    'Z': Glyph(strokes=_segments('top', 'middle', 'bottom', 'right_upper', 'left_lower'), advance=1.00),
    'a': Glyph(strokes=_segments('lc_top', 'lc_middle', 'lc_bottom', 'lc_left_lower', 'lc_right_full'), advance=0.90),
    'b': Glyph(strokes=_segments('left_full', 'lc_top', 'lc_middle', 'lc_bottom', 'lc_right_full'), advance=0.95),
    'c': Glyph(strokes=_segments('lc_top', 'lc_left_full', 'lc_bottom'), advance=0.88),
    'd': Glyph(strokes=_segments('right_full', 'lc_top', 'lc_middle', 'lc_bottom', 'lc_left_full'), advance=0.95),
    'e': Glyph(strokes=_segments('lc_top', 'lc_left_full', 'lc_middle', 'lc_bottom'), advance=0.90),
    'f': Glyph(strokes=_segments('center_full', 'lc_top', 'lc_middle'), advance=0.75),
    'g': Glyph(strokes=_segments('lc_top', 'lc_left_full', 'lc_middle', 'lc_bottom', 'lc_right_desc_full'), advance=0.95),
    'h': Glyph(strokes=_segments('left_full', 'lc_right_full', 'lc_middle'), advance=0.95),
    'i': Glyph(strokes=_segments('lc_center_full', 'lc_dot'), advance=0.50),
    'j': Glyph(strokes=_segments('j_hook', 'lc_dot'), advance=0.58),
    'k': Glyph(strokes=_segments('left_full', 'lc_middle', 'lc_right_upper', 'lc_right_lower'), advance=0.90),
    'l': Glyph(strokes=_segments('left_full'), advance=0.55),
    'm': Glyph(strokes=_segments('lc_left_full', 'lc_inner_left_full', 'lc_inner_right_full', 'lc_right_full', 'lc_top'), advance=1.10),
    'n': Glyph(strokes=_segments('lc_left_full', 'lc_right_full', 'lc_top'), advance=0.95),
    'o': Glyph(strokes=_segments('lc_top', 'lc_bottom', 'lc_left_full', 'lc_right_full'), advance=0.92),
    'p': Glyph(strokes=_segments('lc_left_desc_full', 'lc_top', 'lc_middle', 'lc_right_upper'), advance=0.95),
    'q': Glyph(strokes=_segments('lc_right_desc_full', 'lc_top', 'lc_middle', 'lc_left_upper', 'lc_bottom'), advance=0.95),
    'r': Glyph(strokes=_segments('lc_left_full', 'lc_top', 'lc_middle_right'), advance=0.75),
    's': Glyph(strokes=_segments('lc_top', 'lc_left_upper', 'lc_middle', 'lc_right_lower', 'lc_bottom'), advance=0.88),
    't': Glyph(strokes=_segments('center_full', 'lc_middle'), advance=0.78),
    'u': Glyph(strokes=_segments('lc_left_full', 'lc_right_full', 'lc_bottom'), advance=0.95),
    'v': Glyph(strokes=_segments('lc_left_lower', 'lc_right_lower', 'lc_bottom'), advance=0.90),
    'w': Glyph(strokes=_segments('lc_left_full', 'lc_inner_left_full', 'lc_inner_right_full', 'lc_right_full', 'lc_bottom'), advance=1.10),
    'x': Glyph(strokes=_segments('lc_left_full', 'lc_right_full', 'lc_middle'), advance=0.90),
    'y': Glyph(strokes=_segments('lc_left_full', 'lc_right_desc_full', 'lc_bottom'), advance=0.95),
    'z': Glyph(strokes=_segments('lc_top', 'lc_middle', 'lc_bottom', 'lc_right_upper', 'lc_left_lower'), advance=0.90),
    '0': Glyph(strokes=_segments('top', 'bottom', 'left_full', 'right_full'), advance=1.00),
    '1': Glyph(strokes=_segments('center_full', 'bottom'), advance=0.80),
    '2': Glyph(strokes=_segments('top', 'right_upper', 'middle', 'left_lower', 'bottom'), advance=1.00),
    '3': Glyph(strokes=_segments('top', 'middle', 'bottom', 'right_full'), advance=1.00),
    '4': Glyph(strokes=_segments('left_upper', 'middle', 'right_full'), advance=1.00),
    '5': Glyph(strokes=_segments('top', 'left_upper', 'middle', 'right_lower', 'bottom'), advance=1.00),
    '6': Glyph(strokes=_segments('top', 'middle', 'bottom', 'left_full', 'right_lower'), advance=1.00),
    '7': Glyph(strokes=_segments('top', 'right_full'), advance=1.00),
    '8': Glyph(strokes=_segments('top', 'middle', 'bottom', 'left_full', 'right_full'), advance=1.00),
    '9': Glyph(strokes=_segments('top', 'middle', 'bottom', 'left_upper', 'right_full'), advance=1.00),
    '.': Glyph(strokes=_segments('dot_tick'), advance=0.40),
    ',': Glyph(strokes=_segments('comma_tick'), advance=0.40),
    '-': Glyph(strokes=_segments('middle'), advance=0.70),
    ':': Glyph(strokes=_segments('colon_top_tick', 'colon_bottom_tick'), advance=0.45),
    '/': Glyph(strokes=_segments('center_full'), advance=0.80),
    '?': Glyph(strokes=_segments('top', 'right_upper', 'middle', 'question_stem', 'dot_tick'), advance=1.00),
    '!': Glyph(strokes=_segments('center_upper', 'dot_tick'), advance=0.50),
}


def normalize_char(char: str) -> str:
    return char


def _validate_glyph_table(glyphs: dict[str, Glyph] | None = None) -> None:
    table = _GLYPHS if glyphs is None else glyphs
    for char, glyph in table.items():
        if not math.isfinite(glyph.advance) or glyph.advance <= 0.0:
            raise RuntimeError(f'Invalid glyph advance for {char!r}: {glyph.advance!r}')
        for stroke_index, stroke in enumerate(glyph.strokes):
            if len(stroke) < 2:
                raise RuntimeError(
                    f'Glyph {char!r} stroke {stroke_index} must contain at least 2 points.'
                )
            for point_index, (x, y) in enumerate(stroke):
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise RuntimeError(
                        f'Glyph {char!r} stroke {stroke_index} point {point_index} is not finite: {(x, y)!r}'
                    )
            for segment_index in range(len(stroke) - 1):
                start = stroke[segment_index]
                end = stroke[segment_index + 1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if abs(dx) <= _AXIS_EPS and abs(dy) <= _AXIS_EPS:
                    raise RuntimeError(
                        f'Glyph {char!r} stroke {stroke_index} segment {segment_index} is zero-length: '
                        f'{start!r} -> {end!r}'
                    )
                if abs(dx) > _AXIS_EPS and abs(dy) > _AXIS_EPS:
                    raise RuntimeError(
                        f'Glyph {char!r} stroke {stroke_index} segment {segment_index} is not axis-aligned: '
                        f'{start!r} -> {end!r}'
                    )


_validate_glyph_table()


def get_glyph(char: str) -> Glyph:
    normalized = normalize_char(char)
    if normalized not in _GLYPHS:
        raise KeyError(normalized)
    return _GLYPHS[normalized]


def supported_chars() -> set[str]:
    return set(_GLYPHS.keys())
