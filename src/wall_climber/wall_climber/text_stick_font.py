from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Glyph:
    strokes: tuple[tuple[tuple[float, float], ...], ...]
    advance: float


_AXIS_EPS = 1.0e-9


def _poly(*points: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    return tuple(points)


def _ellipse(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    *,
    segments: int = 18,
    start_deg: float = 0.0,
    end_deg: float = 360.0,
    close: bool = True,
) -> tuple[tuple[float, float], ...]:
    if segments < 4:
        raise ValueError('segments must be >= 4')
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)
    points = []
    step_count = segments if close else max(1, segments - 1)
    for index in range(step_count):
        t = index / float(segments if close else max(1, segments - 1))
        angle = start_rad + ((end_rad - start_rad) * t)
        points.append((cx + (rx * math.cos(angle)), cy + (ry * math.sin(angle))))
    if not close:
        angle = end_rad
        points.append((cx + (rx * math.cos(angle)), cy + (ry * math.sin(angle))))
    elif points[0] != points[-1]:
        points.append(points[0])
    return tuple(points)


def _validate_glyph_table(glyphs: dict[str, Glyph]) -> None:
    closed_loop_strokes = {
        'O': (0,),
        'Q': (0,),
        '0': (0,),
        '8': (0, 1),
    }
    for char, glyph in glyphs.items():
        if glyph.advance <= 0.0 or not math.isfinite(glyph.advance):
            raise RuntimeError(f'invalid stick glyph advance for {char!r}')
        for stroke_index, stroke in enumerate(glyph.strokes):
            if len(stroke) < 2:
                raise RuntimeError(f'stick glyph {char!r} stroke {stroke_index} is too short')
            for point_index, (x, y) in enumerate(stroke):
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise RuntimeError(
                        f'stick glyph {char!r} stroke {stroke_index} point {point_index} is not finite'
                    )
            for segment_index in range(len(stroke) - 1):
                dx = stroke[segment_index + 1][0] - stroke[segment_index][0]
                dy = stroke[segment_index + 1][1] - stroke[segment_index][1]
                if abs(dx) <= _AXIS_EPS and abs(dy) <= _AXIS_EPS:
                    raise RuntimeError(
                        f'stick glyph {char!r} stroke {stroke_index} segment {segment_index} is zero-length'
                    )
        for stroke_index in closed_loop_strokes.get(char, ()):
            if stroke_index >= len(glyph.strokes):
                raise RuntimeError(
                    f'stick glyph {char!r} is missing expected closed stroke {stroke_index}'
                )
            stroke = glyph.strokes[stroke_index]
            if abs(stroke[0][0] - stroke[-1][0]) > 1.0e-6 or abs(stroke[0][1] - stroke[-1][1]) > 1.0e-6:
                raise RuntimeError(
                    f'stick glyph {char!r} stroke {stroke_index} must close explicitly'
                )


_GLYPHS: dict[str, Glyph] = {
    ' ': Glyph(strokes=tuple(), advance=0.42),
    'A': Glyph(strokes=(
        _poly((0.15, 0.00), (0.50, 1.00), (0.85, 0.00)),
        _poly((0.28, 0.45), (0.72, 0.45)),
    ), advance=1.00),
    'B': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00)),
        _poly((0.18, 1.00), (0.62, 1.00), (0.80, 0.84), (0.80, 0.66), (0.62, 0.52), (0.18, 0.52)),
        _poly((0.18, 0.52), (0.65, 0.52), (0.84, 0.36), (0.84, 0.14), (0.65, 0.00), (0.18, 0.00)),
    ), advance=1.04),
    'C': Glyph(strokes=(
        _ellipse(0.53, 0.50, 0.34, 0.50, segments=18, start_deg=48.0, end_deg=312.0, close=False),
    ), advance=1.02),
    'D': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00)),
        _poly((0.18, 1.00), (0.58, 1.00), (0.84, 0.78), (0.84, 0.22), (0.58, 0.00), (0.18, 0.00)),
    ), advance=1.05),
    'E': Glyph(strokes=(
        _poly((0.82, 1.00), (0.18, 1.00), (0.18, 0.00), (0.82, 0.00)),
        _poly((0.18, 0.50), (0.68, 0.50)),
    ), advance=0.98),
    'F': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00), (0.82, 1.00)),
        _poly((0.18, 0.50), (0.68, 0.50)),
    ), advance=0.96),
    'G': Glyph(strokes=(
        _ellipse(0.53, 0.50, 0.34, 0.50, segments=18, start_deg=48.0, end_deg=330.0, close=False),
        _poly((0.54, 0.50), (0.82, 0.50), (0.82, 0.24)),
    ), advance=1.05),
    'H': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00)),
        _poly((0.82, 0.00), (0.82, 1.00)),
        _poly((0.18, 0.50), (0.82, 0.50)),
    ), advance=1.04),
    'I': Glyph(strokes=(
        _poly((0.50, 0.00), (0.50, 1.00)),
    ), advance=0.58),
    'J': Glyph(strokes=(
        _poly((0.82, 1.00), (0.82, 0.18), (0.66, 0.00), (0.36, 0.00), (0.18, 0.18)),
    ), advance=0.92),
    'K': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00)),
        _poly((0.82, 1.00), (0.18, 0.50), (0.84, 0.00)),
    ), advance=1.02),
    'L': Glyph(strokes=(
        _poly((0.18, 1.00), (0.18, 0.00), (0.82, 0.00)),
    ), advance=0.94),
    'M': Glyph(strokes=(
        _poly((0.15, 0.00), (0.15, 1.00), (0.50, 0.45), (0.85, 1.00), (0.85, 0.00)),
    ), advance=1.12),
    'N': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00), (0.82, 0.00), (0.82, 1.00)),
    ), advance=1.06),
    'O': Glyph(strokes=(
        _ellipse(0.50, 0.50, 0.34, 0.50, segments=22, close=True),
    ), advance=1.04),
    'P': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00)),
        _poly((0.18, 1.00), (0.62, 1.00), (0.80, 0.84), (0.80, 0.63), (0.62, 0.50), (0.18, 0.50)),
    ), advance=1.00),
    'Q': Glyph(strokes=(
        _ellipse(0.50, 0.50, 0.34, 0.50, segments=22, close=True),
        _poly((0.58, 0.18), (0.86, -0.06)),
    ), advance=1.08),
    'R': Glyph(strokes=(
        _poly((0.18, 0.00), (0.18, 1.00)),
        _poly((0.18, 1.00), (0.62, 1.00), (0.80, 0.84), (0.80, 0.63), (0.62, 0.50), (0.18, 0.50)),
        _poly((0.18, 0.50), (0.84, 0.00)),
    ), advance=1.03),
    'S': Glyph(strokes=(
        _poly((0.82, 0.86), (0.66, 1.00), (0.35, 1.00), (0.18, 0.82), (0.18, 0.64), (0.34, 0.50), (0.66, 0.50), (0.82, 0.36), (0.82, 0.18), (0.66, 0.00), (0.34, 0.00), (0.18, 0.14)),
    ), advance=0.98),
    'T': Glyph(strokes=(
        _poly((0.15, 1.00), (0.85, 1.00)),
        _poly((0.50, 1.00), (0.50, 0.00)),
    ), advance=1.00),
    'U': Glyph(strokes=(
        _poly((0.18, 1.00), (0.18, 0.18), (0.35, 0.00), (0.65, 0.00), (0.82, 0.18), (0.82, 1.00)),
    ), advance=1.05),
    'V': Glyph(strokes=(
        _poly((0.15, 1.00), (0.50, 0.00), (0.85, 1.00)),
    ), advance=1.02),
    'W': Glyph(strokes=(
        _poly((0.10, 1.00), (0.28, 0.00), (0.50, 0.54), (0.72, 0.00), (0.90, 1.00)),
    ), advance=1.22),
    'X': Glyph(strokes=(
        _poly((0.15, 1.00), (0.85, 0.00)),
        _poly((0.15, 0.00), (0.85, 1.00)),
    ), advance=1.00),
    'Y': Glyph(strokes=(
        _poly((0.15, 1.00), (0.50, 0.55), (0.85, 1.00)),
        _poly((0.50, 0.55), (0.50, 0.00)),
    ), advance=1.00),
    'Z': Glyph(strokes=(
        _poly((0.15, 1.00), (0.85, 1.00), (0.15, 0.00), (0.85, 0.00)),
    ), advance=1.00),
    '0': Glyph(strokes=(
        _ellipse(0.50, 0.50, 0.32, 0.50, segments=22, close=True),
    ), advance=1.00),
    '1': Glyph(strokes=(
        _poly((0.40, 0.80), (0.50, 1.00), (0.50, 0.00)),
    ), advance=0.70),
    '2': Glyph(strokes=(
        _poly((0.18, 0.82), (0.34, 1.00), (0.64, 1.00), (0.82, 0.82), (0.82, 0.62), (0.18, 0.00), (0.82, 0.00)),
    ), advance=0.98),
    '3': Glyph(strokes=(
        _poly((0.22, 1.00), (0.74, 1.00), (0.56, 0.55), (0.76, 0.55), (0.82, 0.42), (0.82, 0.18), (0.64, 0.00), (0.24, 0.00)),
    ), advance=0.98),
    '4': Glyph(strokes=(
        _poly((0.76, 0.00), (0.76, 1.00)),
        _poly((0.76, 0.50), (0.18, 0.50), (0.62, 1.00)),
    ), advance=0.98),
    '5': Glyph(strokes=(
        _poly((0.82, 1.00), (0.25, 1.00), (0.18, 0.55), (0.62, 0.55), (0.82, 0.38), (0.82, 0.18), (0.64, 0.00), (0.22, 0.00)),
    ), advance=0.98),
    '6': Glyph(strokes=(
        _poly((0.76, 0.92), (0.62, 1.00), (0.36, 1.00), (0.18, 0.76), (0.18, 0.20), (0.36, 0.00), (0.66, 0.00), (0.82, 0.18), (0.82, 0.38), (0.66, 0.55), (0.28, 0.55)),
    ), advance=1.00),
    '7': Glyph(strokes=(
        _poly((0.15, 1.00), (0.85, 1.00), (0.34, 0.00)),
    ), advance=0.98),
    '8': Glyph(strokes=(
        _ellipse(0.50, 0.74, 0.24, 0.24, segments=16, close=True),
        _ellipse(0.50, 0.24, 0.28, 0.24, segments=16, close=True),
    ), advance=1.00),
    '9': Glyph(strokes=(
        _poly((0.82, 0.45), (0.64, 0.55), (0.34, 0.55), (0.18, 0.74), (0.18, 0.88), (0.35, 1.00), (0.64, 1.00), (0.82, 0.82), (0.82, 0.18), (0.64, 0.00), (0.24, 0.00)),
    ), advance=1.00),
    '.': Glyph(strokes=(
        _poly((0.50, 0.08), (0.50, 0.00)),
    ), advance=0.34),
    ',': Glyph(strokes=(
        _poly((0.52, 0.06), (0.46, -0.12)),
    ), advance=0.34),
    '-': Glyph(strokes=(
        _poly((0.22, 0.45), (0.78, 0.45)),
    ), advance=0.68),
    ':': Glyph(strokes=(
        _poly((0.50, 0.74), (0.50, 0.66)),
        _poly((0.50, 0.18), (0.50, 0.10)),
    ), advance=0.38),
    '/': Glyph(strokes=(
        _poly((0.18, 0.00), (0.82, 1.00)),
    ), advance=0.84),
    '?': Glyph(strokes=(
        _poly((0.22, 0.82), (0.38, 1.00), (0.66, 1.00), (0.82, 0.84), (0.82, 0.64), (0.50, 0.44), (0.50, 0.24)),
        _poly((0.50, 0.08), (0.50, 0.00)),
    ), advance=0.96),
    '!': Glyph(strokes=(
        _poly((0.50, 0.20), (0.50, 1.00)),
        _poly((0.50, 0.08), (0.50, 0.00)),
    ), advance=0.42),
}


_validate_glyph_table(_GLYPHS)


def get_glyph(char: str) -> Glyph:
    normalized = normalize_char(char)
    if normalized not in _GLYPHS:
        raise KeyError(normalized)
    return _GLYPHS[normalized]


def normalize_char(char: str) -> str:
    return char


def supported_chars() -> set[str]:
    return set(_GLYPHS.keys())
