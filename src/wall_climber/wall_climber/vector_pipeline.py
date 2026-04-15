from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import cv2  # type: ignore
import numpy
try:
    from ament_index_python.packages import (  # type: ignore
        PackageNotFoundError,
        get_package_share_directory,
    )
except ImportError:
    class PackageNotFoundError(Exception):
        pass

    def get_package_share_directory(_package_name: str) -> str:
        raise PackageNotFoundError(_package_name)
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, TextToPath

from wall_climber.shared_config import load_shared_config
from wall_climber.text_stick_font import get_glyph as get_stick_glyph
from wall_climber.text_vector_font import get_glyph as get_legacy_glyph


_Point = tuple[float, float]
_EPS = 1.0e-9
_PACKAGE_NAME = 'wall_climber'
_BUNDLED_FONT_NAME = 'DejaVuSans.ttf'
_RELIEF_SVG_FONT_NAME = 'ReliefSingleLineSVG-Regular.svg'
_RELIEF_SVG_MIN_CURVE_TOLERANCE = 0.005
_DEFAULT_TEXT_CURVE_TOLERANCE = 0.008
_MAX_TEXT_CONTOUR_POINTS = 512
_MAX_TEXT_TOTAL_POINTS = 12000
_LOG = logging.getLogger(__name__)
_LOGGED_TEXT_FONT_SOURCES: set[str] = set()
_LOGGED_TEXT_SOURCE_POLICIES: set[str] = set()
_LOGGED_TEXT_FALLBACK = False
_LOGGED_TEXT_NORMALIZATION = False
_LOGGED_TEXT_UPWARD_BIAS: set[float] = set()


@dataclass(frozen=True)
class ReliefSvgFont:
    glyphs: dict[str, tuple[str | None, float]]
    default_advance: float
    cap_height: float
    source: str


@dataclass(frozen=True)
class VectorPlacement:
    x: float
    y: float
    scale: float


@dataclass(frozen=True)
class VectorBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class PlacementResult:
    placement: VectorPlacement
    base_fit_scale: float
    final_scale: float
    bounds: VectorBounds
    outside_points: int


@dataclass(frozen=True)
class DrawPathSegment:
    draw: bool
    points: tuple[_Point, ...]


@dataclass(frozen=True)
class TextGlyphTemplate:
    text: str
    strokes: tuple[tuple[_Point, ...], ...]
    bbox: VectorBounds | None
    advance: float
    source: str


@dataclass(frozen=True)
class TextGlyphOutline:
    line_index: int
    word_index: int
    text: str
    strokes: tuple[tuple[_Point, ...], ...]
    bbox: VectorBounds
    advance: float


class TextDensityError(ValueError):
    pass


@lru_cache(maxsize=1)
def _text_vector_cleanup_defaults():
    return load_shared_config().text_vector


@lru_cache(maxsize=1)
def _text_layout_defaults():
    return load_shared_config().text_layout


def _normalized_text_spacing_defaults() -> tuple[float, float, float]:
    defaults = _text_layout_defaults()
    glyph_height = max(float(defaults.glyph_height), 1.0e-6)
    letter_spacing_em = max(0.0, float(defaults.letter_spacing) / glyph_height)
    word_spacing_em = max(letter_spacing_em, float(defaults.word_spacing) / glyph_height)
    uppercase_advance_scale = max(0.1, float(defaults.uppercase_advance_scale))
    return letter_spacing_em, word_spacing_em, uppercase_advance_scale


def _effective_text_advance_em(
    char: str,
    *,
    base_advance: float,
    letter_spacing_em: float,
    word_spacing_em: float,
    uppercase_advance_scale: float,
) -> float:
    advance = max(0.0, float(base_advance))
    if char.isspace():
        return max(advance, word_spacing_em)
    if char.isalpha() and char.upper() == char and char.lower() != char:
        advance *= uppercase_advance_scale
    return advance + letter_spacing_em


def _distance(a: _Point, b: _Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: _Point, b: _Point) -> _Point:
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _point_line_distance(point: _Point, start: _Point, end: _Point) -> float:
    if _distance(start, end) <= _EPS:
        return _distance(point, start)
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj = (x1 + t * dx, y1 + t * dy)
    return _distance(point, proj)


def _rdp(points: list[_Point], epsilon: float) -> list[_Point]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_dist = -1.0
    split_index = -1
    for index in range(1, len(points) - 1):
        dist = _point_line_distance(points[index], start, end)
        if dist > max_dist:
            max_dist = dist
            split_index = index
    if max_dist <= epsilon or split_index < 0:
        return [start, end]
    left = _rdp(points[: split_index + 1], epsilon)
    right = _rdp(points[split_index:], epsilon)
    return left[:-1] + right


def _quadratic_flatness(p0: _Point, p1: _Point, p2: _Point) -> float:
    return _point_line_distance(p1, p0, p2)


def _cubic_flatness(p0: _Point, p1: _Point, p2: _Point, p3: _Point) -> float:
    return max(_point_line_distance(p1, p0, p3), _point_line_distance(p2, p0, p3))


def _flatten_quadratic(p0: _Point, p1: _Point, p2: _Point, tol: float, depth: int = 0) -> list[_Point]:
    if depth >= 12 or _quadratic_flatness(p0, p1, p2) <= tol:
        return [p0, p2]
    p01 = _midpoint(p0, p1)
    p12 = _midpoint(p1, p2)
    p012 = _midpoint(p01, p12)
    left = _flatten_quadratic(p0, p01, p012, tol, depth + 1)
    right = _flatten_quadratic(p012, p12, p2, tol, depth + 1)
    return left[:-1] + right


def _flatten_cubic(
    p0: _Point,
    p1: _Point,
    p2: _Point,
    p3: _Point,
    tol: float,
    depth: int = 0,
) -> list[_Point]:
    if depth >= 12 or _cubic_flatness(p0, p1, p2, p3) <= tol:
        return [p0, p3]
    p01 = _midpoint(p0, p1)
    p12 = _midpoint(p1, p2)
    p23 = _midpoint(p2, p3)
    p012 = _midpoint(p01, p12)
    p123 = _midpoint(p12, p23)
    p0123 = _midpoint(p012, p123)
    left = _flatten_cubic(p0, p01, p012, p0123, tol, depth + 1)
    right = _flatten_cubic(p0123, p123, p23, p3, tol, depth + 1)
    return left[:-1] + right


def _sanitize_stroke(points: list[_Point]) -> tuple[_Point, ...] | None:
    deduped: list[_Point] = []
    for point in points:
        if not deduped:
            deduped.append(point)
            continue
        if _distance(deduped[-1], point) <= _EPS:
            continue
        deduped.append(point)
    if len(deduped) < 2:
        return None
    return tuple(deduped)


def _strokes_bounds(strokes: tuple[tuple[_Point, ...], ...]) -> VectorBounds:
    if not strokes:
        raise ValueError('No strokes available to compute bounds.')
    xs = [point[0] for stroke in strokes for point in stroke]
    ys = [point[1] for stroke in strokes for point in stroke]
    if not xs or not ys:
        raise ValueError('No points available to compute bounds.')
    return VectorBounds(min(xs), max(xs), min(ys), max(ys))


def _stroke_length(points: tuple[_Point, ...]) -> float:
    return sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))


def _stroke_heading(points: tuple[_Point, ...], *, from_start: bool) -> float | None:
    if len(points) < 2:
        return None
    pairs = zip(points[:-1], points[1:]) if from_start else zip(reversed(points[1:]), reversed(points[:-1]))
    for start, end in pairs:
        if _distance(start, end) <= _EPS:
            continue
        return math.atan2(end[1] - start[1], end[0] - start[0])
    return None


def _heading_delta_deg(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return 0.0
    delta = math.atan2(math.sin(second - first), math.cos(second - first))
    return abs(math.degrees(delta))


def _heading_change_deg(prev_point: _Point, point: _Point, next_point: _Point) -> float:
    before = math.atan2(point[1] - prev_point[1], point[0] - prev_point[0])
    after = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
    delta = math.atan2(math.sin(after - before), math.cos(after - before))
    return abs(math.degrees(delta))


def _stroke_has_preserved_short_feature(points: tuple[_Point, ...], min_feature_len: float) -> bool:
    if len(points) <= 2:
        return True
    if _stroke_length(points) <= max(min_feature_len * 3.0, 1.0e-6):
        return True
    first_seg = _distance(points[0], points[1])
    last_seg = _distance(points[-2], points[-1])
    return first_seg <= min_feature_len or last_seg <= min_feature_len


def _protected_stroke_indices(
    points: tuple[_Point, ...],
    min_feature_len: float,
    *,
    corner_threshold_deg: float = 22.0,
) -> tuple[int, ...]:
    if len(points) <= 2:
        return tuple(range(len(points)))
    protected = {0, len(points) - 1}
    if _stroke_has_preserved_short_feature(points, min_feature_len):
        protected.update(range(len(points)))
        return tuple(sorted(protected))
    for index in range(1, len(points) - 1):
        if _heading_change_deg(points[index - 1], points[index], points[index + 1]) >= corner_threshold_deg:
            protected.add(index)
    for index in range(1, len(points)):
        if _distance(points[index - 1], points[index]) <= min_feature_len:
            protected.add(index - 1)
            protected.add(index)
    return tuple(sorted(protected))


def _simplify_stroke_preserving_features(
    points: tuple[_Point, ...],
    simplify_epsilon: float,
    min_feature_len: float,
) -> tuple[_Point, ...]:
    if simplify_epsilon <= 0.0 or len(points) <= 2:
        return points
    protected = _protected_stroke_indices(points, min_feature_len)
    if len(protected) >= len(points):
        return points
    simplified: list[_Point] = [points[0]]
    for start_index, end_index in zip(protected[:-1], protected[1:]):
        span = list(points[start_index:end_index + 1])
        if len(span) <= 2:
            reduced = span
        else:
            reduced = _rdp(span, simplify_epsilon)
        simplified.extend(reduced[1:])
    sanitized = _sanitize_stroke(simplified)
    return sanitized if sanitized is not None else points


def _interpolate_along_stroke(points: tuple[_Point, ...], distance_along: float) -> _Point:
    if distance_along <= 0.0:
        return points[0]
    traveled = 0.0
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        segment_length = _distance(start, end)
        if segment_length <= _EPS:
            continue
        next_traveled = traveled + segment_length
        if distance_along <= next_traveled + _EPS:
            ratio = (distance_along - traveled) / segment_length
            return (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        traveled = next_traveled
    return points[-1]


def _resample_stroke_preserving_features(
    points: tuple[_Point, ...],
    resample_step_m: float,
    min_feature_len: float,
) -> tuple[_Point, ...]:
    if resample_step_m <= 0.0 or len(points) <= 2:
        return points
    if _stroke_has_preserved_short_feature(points, min_feature_len):
        return points
    protected = _protected_stroke_indices(points, min_feature_len)
    if len(protected) >= len(points):
        return points
    resampled: list[_Point] = [points[0]]
    for start_index, end_index in zip(protected[:-1], protected[1:]):
        span = points[start_index:end_index + 1]
        span_length = _stroke_length(span)
        if span_length <= resample_step_m + _EPS:
            if _distance(resampled[-1], span[-1]) > _EPS:
                resampled.append(span[-1])
            continue
        target = resample_step_m
        while target < span_length - _EPS:
            point = _interpolate_along_stroke(span, target)
            if _distance(resampled[-1], point) > _EPS:
                resampled.append(point)
            target += resample_step_m
        if _distance(resampled[-1], span[-1]) > _EPS:
            resampled.append(span[-1])
    sanitized = _sanitize_stroke(resampled)
    return sanitized if sanitized is not None else points


def _apply_text_vector_cleanup(
    strokes: tuple[tuple[_Point, ...], ...],
) -> tuple[tuple[_Point, ...], ...]:
    defaults = _text_vector_cleanup_defaults()
    simplify_epsilon = max(0.0, float(defaults.simplify_epsilon_m))
    resample_step_m = max(0.0, float(defaults.resample_step_m))
    min_feature_len = max(0.0, float(defaults.min_feature_len_m))
    max_points_per_stroke = max(2, int(defaults.max_points_per_stroke))
    max_total_points = max(2, int(defaults.max_total_points))
    if simplify_epsilon <= 0.0 and resample_step_m <= 0.0:
        return strokes
    cleaned: list[tuple[_Point, ...]] = []
    total_points = 0
    for stroke in strokes:
        sanitized = _sanitize_stroke(list(stroke))
        base_stroke = sanitized if sanitized is not None else stroke
        candidate = _simplify_stroke_preserving_features(
            base_stroke,
            simplify_epsilon,
            min_feature_len,
        )
        candidate = _resample_stroke_preserving_features(
            candidate,
            resample_step_m,
            min_feature_len,
        )
        if len(candidate) > max_points_per_stroke or len(candidate) > _MAX_TEXT_CONTOUR_POINTS:
            candidate = base_stroke
        total_points += len(candidate)
        if total_points > max_total_points or total_points > _MAX_TEXT_TOTAL_POINTS:
            return strokes
        cleaned.append(candidate)
    cleaned_tuple = tuple(cleaned)
    if not cleaned_tuple:
        return strokes
    return cleaned_tuple


def _simplify_strokes(
    strokes: tuple[tuple[_Point, ...], ...],
    simplify_epsilon: float,
) -> tuple[tuple[_Point, ...], ...]:
    if simplify_epsilon <= 0.0:
        return strokes
    simplified: list[tuple[_Point, ...]] = []
    for stroke in strokes:
        reduced = _rdp(list(stroke), simplify_epsilon)
        sanitized = _sanitize_stroke(reduced)
        if sanitized is not None:
            simplified.append(sanitized)
    if not simplified:
        raise ValueError('All strokes were removed during simplification.')
    return tuple(simplified)


def _log_text_source_policy_once(policy: str) -> None:
    if policy in _LOGGED_TEXT_SOURCE_POLICIES:
        return
    _LOGGED_TEXT_SOURCE_POLICIES.add(policy)
    if policy == 'relief_svg':
        _LOG.info('Text vector source policy: Relief SingleLine SVG font.')
    elif policy == 'stick_font':
        _LOG.info('Text vector source policy: stick font fallback.')
    elif policy == 'outline_font':
        _LOG.info('Text vector source policy: bundled outline font.')
    elif policy == 'legacy_fallback':
        _LOG.info('Text vector source policy: legacy fallback.')


def _log_text_upward_bias_once(text_upward_bias_em: float) -> None:
    if text_upward_bias_em in _LOGGED_TEXT_UPWARD_BIAS:
        return
    _LOGGED_TEXT_UPWARD_BIAS.add(text_upward_bias_em)
    _LOG.info(
        'Text upward bias applied: text_upward_bias_em=%.4f.',
        text_upward_bias_em,
    )


def normalize_text_for_text_mode(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError('text must be a string.')
    # Preserve case to support lowercase letters from the Relief SVG font.
    return text.replace('\r\n', '\n').replace('\r', '\n')


def normalize_text_plan_input(
    text: str,
    *,
    decode_escaped_line_breaks: bool = False,
) -> str:
    normalized = normalize_text_for_text_mode(text)
    if decode_escaped_line_breaks:
        normalized = (
            normalized.replace('\\r\\n', '\n')
            .replace('\\n', '\n')
            .replace('\\r', '\n')
        )
        normalized = normalize_text_for_text_mode(normalized)

    invalid_controls = sorted(
        {f'0x{ord(char):02x}' for char in normalized if ord(char) < 0x20 and char != '\n'}
    )
    if invalid_controls:
        raise ValueError(
            'text plan contains unsupported control characters: '
            + ', '.join(invalid_controls)
        )
    return normalized


def _log_text_font_source_once(source: str) -> None:
    if source in _LOGGED_TEXT_FONT_SOURCES:
        return
    _LOGGED_TEXT_FONT_SOURCES.add(source)
    if source == 'relief_svg_installed':
        _LOG.info('Text vector font source: Relief SingleLine SVG (installed).')
    elif source == 'relief_svg_source':
        _LOG.info('Text vector font source: Relief SingleLine SVG (source tree).')
    elif source == 'bundled_installed':
        _LOG.info('Text vector font source: bundled installed font.')
    elif source == 'bundled_source':
        _LOG.info('Text vector font source: bundled source-tree font.')
    elif source.startswith('family_override:'):
        _LOG.info(
            'Text vector font source: explicit family override %s.',
            source.split(':', 1)[1],
        )


def _log_text_fallback_once(reason: str) -> None:
    global _LOGGED_TEXT_FALLBACK
    if _LOGGED_TEXT_FALLBACK:
        return
    _LOGGED_TEXT_FALLBACK = True
    _LOG.warning(
        'Text vector fallback active: using text_vector_font.py because TextPath font loading failed (%s).',
        reason,
    )


def _bundled_font_candidates() -> tuple[tuple[str, Path], ...]:
    candidates: list[tuple[str, Path]] = []
    try:
        installed = (
            Path(get_package_share_directory(_PACKAGE_NAME))
            / 'fonts'
            / _BUNDLED_FONT_NAME
        )
        candidates.append(('bundled_installed', installed))
    except PackageNotFoundError:
        pass
    source_tree = Path(__file__).resolve().parents[1] / 'fonts' / _BUNDLED_FONT_NAME
    candidates.append(('bundled_source', source_tree))
    return tuple(candidates)


def _relief_svg_font_candidates() -> tuple[tuple[str, Path], ...]:
    candidates: list[tuple[str, Path]] = []
    try:
        installed = (
            Path(get_package_share_directory(_PACKAGE_NAME))
            / 'fonts'
            / _RELIEF_SVG_FONT_NAME
        )
        candidates.append(('relief_svg_installed', installed))
    except PackageNotFoundError:
        pass
    source_tree = Path(__file__).resolve().parents[1] / 'fonts' / _RELIEF_SVG_FONT_NAME
    candidates.append(('relief_svg_source', source_tree))
    return tuple(candidates)


def _svg_tag_name(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def _parse_svg_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    cleaned = value.strip()
    if not cleaned:
        return default
    cleaned = re.sub(r'[^0-9eE+\-.]', '', cleaned)
    if not cleaned:
        return default
    return float(cleaned)


@lru_cache(maxsize=1)
def _load_relief_svg_font() -> ReliefSvgFont:
    """Load the Relief SingleLine SVG font and cache it."""
    svg_path: Path | None = None
    source = 'relief_svg_source'
    for candidate_source, candidate_path in _relief_svg_font_candidates():
        if candidate_path.is_file():
            svg_path = candidate_path
            source = candidate_source
            break
    if svg_path is None:
        raise FileNotFoundError(
            f'Unable to locate Relief SVG font {_RELIEF_SVG_FONT_NAME!r} '
            'in package share or source tree.'
        )
    root = ET.parse(svg_path).getroot()
    font = next(
        (elem for elem in root.iter() if _svg_tag_name(elem.tag) == 'font'),
        None,
    )
    if font is None:
        raise ValueError('Relief SVG font file does not contain a <font> definition.')
    face = next(
        (elem for elem in font if _svg_tag_name(elem.tag) == 'font-face'),
        None,
    )
    if face is None:
        raise ValueError('Relief SVG font file does not contain a <font-face> definition.')
    cap_height = _parse_svg_float(face.attrib.get('cap-height'), 0.0)
    if cap_height <= _EPS:
        cap_height = _parse_svg_float(face.attrib.get('x-height'), 0.0)
    if cap_height <= _EPS:
        cap_height = _parse_svg_float(face.attrib.get('units-per-em'), 1000.0)
    default_advance = _parse_svg_float(font.attrib.get('horiz-adv-x'), cap_height)
    glyphs: dict[str, tuple[str | None, float]] = {}
    for glyph in font:
        if _svg_tag_name(glyph.tag) != 'glyph':
            continue
        unicode_value = glyph.attrib.get('unicode')
        if not unicode_value or len(unicode_value) != 1:
            continue
        glyphs[unicode_value] = (
            glyph.attrib.get('d'),
            _parse_svg_float(glyph.attrib.get('horiz-adv-x'), default_advance),
        )
    _log_text_font_source_once(source)
    _LOG.info(
        'Relief SVG font loaded: %d glyphs, cap_height=%.1f, source=%s',
        len(glyphs), cap_height, source,
    )
    return ReliefSvgFont(
        glyphs=glyphs,
        default_advance=default_advance,
        cap_height=cap_height,
        source=source,
    )


def _parse_svg_path_d(
    path_data: str,
    cap_height: float,
    curve_tolerance: float,
) -> tuple[tuple[_Point, ...], ...]:
    """Parse an SVG path 'd' attribute into polyline strokes.

    SVG font glyphs use an inverted Y axis (Y grows upward), so we
    flip the Y coordinate by dividing by cap_height and negating.
    The result is in normalized coordinates where cap-height = 1.0
    and Y grows downward (matching our stroke convention where
    (0,0) is top-left and (1,1) is bottom-right of the em square).
    """
    token_pattern = re.compile(
        r'[MmLlHhVvCcSsQqZz]|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
    )
    tokens = token_pattern.findall(path_data)
    if not tokens:
        return tuple()

    def is_command(token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    strokes: list[list[_Point]] = []
    stroke: list[_Point] = []
    command: str | None = None
    cursor: _Point = (0.0, 0.0)
    subpath_start: _Point | None = None
    last_cubic_control: _Point | None = None

    def norm(x: float, y: float) -> _Point:
        return (x / cap_height, 1.0 - (y / cap_height))

    def flush_stroke() -> None:
        nonlocal stroke
        sanitized = _sanitize_stroke(stroke)
        if sanitized is not None:
            strokes.append(list(sanitized))
        stroke = []

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if is_command(token):
            command = token
            index += 1
        elif command is None:
            raise ValueError('Invalid SVG path data: missing command.')

        if command is None:
            continue

        absolute = command.isupper()
        cmd = command.upper()

        def read_number() -> float:
            nonlocal index
            if index >= len(tokens) or is_command(tokens[index]):
                raise ValueError(f'SVG path command {command} is missing parameters.')
            value = float(tokens[index])
            index += 1
            return value

        if cmd == 'M':
            x = read_number()
            y = read_number()
            if not absolute:
                x += cursor[0]
                y += cursor[1]
            flush_stroke()
            cursor = (x, y)
            subpath_start = cursor
            stroke = [norm(x, y)]
            last_cubic_control = None
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_number()
                y = read_number()
                if not absolute:
                    x += cursor[0]
                    y += cursor[1]
                cursor = (x, y)
                stroke.append(norm(x, y))
            command = 'L' if absolute else 'l'
            continue

        if cmd == 'L':
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_number()
                y = read_number()
                if not absolute:
                    x += cursor[0]
                    y += cursor[1]
                cursor = (x, y)
                if not stroke:
                    stroke = [norm(x, y)]
                    subpath_start = cursor
                else:
                    stroke.append(norm(x, y))
            last_cubic_control = None
            continue

        if cmd == 'H':
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_number()
                if not absolute:
                    x += cursor[0]
                cursor = (x, cursor[1])
                if not stroke:
                    stroke = [norm(cursor[0], cursor[1])]
                    subpath_start = cursor
                else:
                    stroke.append(norm(cursor[0], cursor[1]))
            last_cubic_control = None
            continue

        if cmd == 'V':
            while index < len(tokens) and not is_command(tokens[index]):
                y = read_number()
                if not absolute:
                    y += cursor[1]
                cursor = (cursor[0], y)
                if not stroke:
                    stroke = [norm(cursor[0], cursor[1])]
                    subpath_start = cursor
                else:
                    stroke.append(norm(cursor[0], cursor[1]))
            last_cubic_control = None
            continue

        if cmd == 'C':
            while index < len(tokens) and not is_command(tokens[index]):
                c1x = read_number()
                c1y = read_number()
                c2x = read_number()
                c2y = read_number()
                ex = read_number()
                ey = read_number()
                if not absolute:
                    c1x += cursor[0]
                    c1y += cursor[1]
                    c2x += cursor[0]
                    c2y += cursor[1]
                    ex += cursor[0]
                    ey += cursor[1]
                if not stroke:
                    stroke = [norm(cursor[0], cursor[1])]
                    subpath_start = cursor
                flattened = _flatten_cubic(
                    norm(cursor[0], cursor[1]),
                    norm(c1x, c1y),
                    norm(c2x, c2y),
                    norm(ex, ey),
                    curve_tolerance,
                )
                stroke.extend(flattened[1:])
                last_cubic_control = (c2x, c2y)
                cursor = (ex, ey)
            continue

        if cmd == 'S':
            while index < len(tokens) and not is_command(tokens[index]):
                c2x = read_number()
                c2y = read_number()
                ex = read_number()
                ey = read_number()
                if not absolute:
                    c2x += cursor[0]
                    c2y += cursor[1]
                    ex += cursor[0]
                    ey += cursor[1]
                if last_cubic_control is not None:
                    c1x = 2.0 * cursor[0] - last_cubic_control[0]
                    c1y = 2.0 * cursor[1] - last_cubic_control[1]
                else:
                    c1x, c1y = cursor
                if not stroke:
                    stroke = [norm(cursor[0], cursor[1])]
                    subpath_start = cursor
                flattened = _flatten_cubic(
                    norm(cursor[0], cursor[1]),
                    norm(c1x, c1y),
                    norm(c2x, c2y),
                    norm(ex, ey),
                    curve_tolerance,
                )
                stroke.extend(flattened[1:])
                last_cubic_control = (c2x, c2y)
                cursor = (ex, ey)
            continue

        if cmd == 'Q':
            while index < len(tokens) and not is_command(tokens[index]):
                c1x = read_number()
                c1y = read_number()
                ex = read_number()
                ey = read_number()
                if not absolute:
                    c1x += cursor[0]
                    c1y += cursor[1]
                    ex += cursor[0]
                    ey += cursor[1]
                if not stroke:
                    stroke = [norm(cursor[0], cursor[1])]
                    subpath_start = cursor
                flattened = _flatten_quadratic(
                    norm(cursor[0], cursor[1]),
                    norm(c1x, c1y),
                    norm(ex, ey),
                    curve_tolerance,
                )
                stroke.extend(flattened[1:])
                cursor = (ex, ey)
            last_cubic_control = None
            continue

        if cmd == 'Z':
            if stroke and subpath_start is not None:
                end_pt = norm(subpath_start[0], subpath_start[1])
                if stroke and _distance(stroke[-1], end_pt) > _EPS:
                    stroke.append(end_pt)
            flush_stroke()
            if subpath_start is not None:
                cursor = subpath_start
            subpath_start = None
            last_cubic_control = None
            continue

        index += 1

    flush_stroke()
    return tuple(tuple(s) for s in strokes)


def _glyph_template_from_relief_svg(
    char: str,
    *,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> TextGlyphTemplate:
    font = _load_relief_svg_font()
    if char not in font.glyphs:
        raise KeyError(char)
    path_data, raw_advance = font.glyphs[char]
    normalized_advance = float(raw_advance) / font.cap_height
    if char.isspace() or not path_data:
        return TextGlyphTemplate(
            text=char,
            strokes=tuple(),
            bbox=None,
            advance=normalized_advance,
            source='relief_svg',
        )
    effective_curve_tolerance = max(curve_tolerance, _RELIEF_SVG_MIN_CURVE_TOLERANCE)
    raw_strokes = _parse_svg_path_d(
        path_data,
        font.cap_height,
        effective_curve_tolerance,
    )
    if simplify_epsilon > 0.0:
        raw_strokes = _simplify_strokes(raw_strokes, simplify_epsilon)
    return TextGlyphTemplate(
        text=char,
        strokes=raw_strokes,
        bbox=_strokes_bounds(raw_strokes) if raw_strokes else None,
        advance=normalized_advance,
        source='relief_svg',
    )


def _resolve_text_font_properties(
    font_family: str | None = None,
) -> tuple[FontProperties, str]:
    if font_family is not None and font_family.strip():
        source = f'family_override:{font_family.strip()}'
        _log_text_font_source_once(source)
        return FontProperties(family=font_family.strip(), size=1.0), source

    for source, path in _bundled_font_candidates():
        if path.is_file():
            _log_text_font_source_once(source)
            return FontProperties(fname=str(path), size=1.0), source
    raise FileNotFoundError(
        f'Unable to locate bundled text font {_BUNDLED_FONT_NAME!r} in package share or source tree.'
    )


@lru_cache(maxsize=16)
def _reference_font_height(font_family: str | None) -> float:
    prop, _ = _resolve_text_font_properties(font_family)
    ttp = TextToPath()
    _, height, _ = ttp.get_text_width_height_descent('H', prop, ismath=False)
    if height <= _EPS:
        fallback_path = TextPath((0.0, 0.0), 'H', size=1.0, prop=prop)
        height = max(float(fallback_path.get_extents().height), 0.0)
    if height <= _EPS:
        raise ValueError('Resolved text font has zero reference height.')
    return float(height)


def _glyph_template_from_textpath(
    char: str,
    *,
    font_family: str | None,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> TextGlyphTemplate:
    prop, source = _resolve_text_font_properties(font_family)
    ttp = TextToPath()
    reference_height = _reference_font_height(font_family)
    advance, _, _ = ttp.get_text_width_height_descent(char, prop, ismath=False)
    normalized_advance = float(advance) / reference_height
    if char.isspace():
        return TextGlyphTemplate(
            text=char,
            strokes=tuple(),
            bbox=None,
            advance=normalized_advance,
            source=source,
        )

    path = TextPath((0.0, 0.0), char, size=1.0, prop=prop)
    strokes = _path_to_strokes(
        path.vertices / reference_height,
        path.codes,
        curve_tolerance=curve_tolerance,
        simplify_epsilon=simplify_epsilon,
    )
    for stroke in strokes:
        if len(stroke) > _MAX_TEXT_CONTOUR_POINTS:
            raise TextDensityError(
                f'text glyph {char!r} produced a contour with {len(stroke)} points, exceeding the limit of {_MAX_TEXT_CONTOUR_POINTS}.'
            )
    return TextGlyphTemplate(
        text=char,
        strokes=strokes,
        bbox=_strokes_bounds(strokes),
        advance=normalized_advance,
        source=source,
    )


def _glyph_template_from_stick_font(char: str) -> TextGlyphTemplate:
    glyph = get_stick_glyph(char)
    strokes = tuple(tuple((point[0], point[1]) for point in stroke) for stroke in glyph.strokes)
    bbox = _strokes_bounds(strokes) if strokes else None
    return TextGlyphTemplate(
        text=char,
        strokes=strokes,
        bbox=bbox,
        advance=float(glyph.advance),
        source='stick_font',
    )


def _glyph_template_from_legacy_fallback(char: str) -> TextGlyphTemplate:
    glyph = get_legacy_glyph(char)
    strokes = tuple(tuple((point[0], point[1]) for point in stroke) for stroke in glyph.strokes)
    bbox = _strokes_bounds(strokes) if strokes else None
    return TextGlyphTemplate(
        text=char,
        strokes=strokes,
        bbox=bbox,
        advance=float(glyph.advance),
        source='text_vector_font_fallback',
    )


@lru_cache(maxsize=256)
def _cached_text_glyph_template(
    char: str,
    font_family: str | None,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> TextGlyphTemplate:
    if font_family is not None:
        try:
            return _glyph_template_from_textpath(
                char,
                font_family=font_family,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
            )
        except TextDensityError:
            raise
        except Exception as exc:
            _log_text_fallback_once(str(exc))
            return _glyph_template_from_legacy_fallback(char)

    # 1. Try Relief SingleLine SVG font first (best quality single-line strokes)
    try:
        return _glyph_template_from_relief_svg(
            char,
            curve_tolerance=curve_tolerance,
            simplify_epsilon=simplify_epsilon,
        )
    except KeyError:
        pass
    except Exception as exc:
        _LOG.debug('Relief SVG glyph %r load failed: %s', char, exc)

    # 2. Stick font fallback for basic characters
    try:
        return _glyph_template_from_stick_font(char)
    except KeyError:
        pass

    # 3. DejaVu outline font via matplotlib TextPath
    try:
        return _glyph_template_from_textpath(
            char,
            font_family=None,
            curve_tolerance=curve_tolerance,
            simplify_epsilon=simplify_epsilon,
        )
    except TextDensityError:
        raise
    except Exception as exc:
        _log_text_fallback_once(str(exc))
        return _glyph_template_from_legacy_fallback(char)


def get_text_glyph_template(
    char: str,
    *,
    font_family: str | None = None,
    curve_tolerance: float = _DEFAULT_TEXT_CURVE_TOLERANCE,
    simplify_epsilon: float = 0.0,
) -> TextGlyphTemplate:
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError('char must be a single-character string.')
    template = _cached_text_glyph_template(
        normalize_text_for_text_mode(char),
        font_family.strip() if isinstance(font_family, str) and font_family.strip() else None,
        float(curve_tolerance),
        float(simplify_epsilon),
    )
    if template.source == 'relief_svg':
        _log_text_source_policy_once('relief_svg')
    elif template.source == 'stick_font':
        _log_text_source_policy_once('stick_font')
    elif template.source == 'text_vector_font_fallback':
        _log_text_source_policy_once('legacy_fallback')
    else:
        _log_text_source_policy_once('outline_font')
    return template


def _path_to_strokes(
    vertices: numpy.ndarray,
    codes: numpy.ndarray | None,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> tuple[tuple[_Point, ...], ...]:
    strokes: list[tuple[_Point, ...]] = []
    current: list[_Point] = []
    start_point: _Point | None = None

    def flush_current() -> None:
        nonlocal current
        sanitized = _sanitize_stroke(current)
        if sanitized is not None:
            strokes.append(sanitized)
        current = []

    index = 0
    total = len(vertices)
    while index < total:
        code = MplPath.LINETO if codes is None else int(codes[index])
        vertex = (float(vertices[index][0]), float(vertices[index][1]))

        if code == MplPath.MOVETO:
            flush_current()
            current = [vertex]
            start_point = vertex
            index += 1
            continue

        if code == MplPath.LINETO:
            if not current:
                current = [vertex]
                start_point = vertex
            else:
                current.append(vertex)
            index += 1
            continue

        if code == MplPath.CURVE3:
            if not current:
                current = [vertex]
                start_point = vertex
                index += 1
                continue
            if index + 1 >= total:
                break
            control = vertex
            end = (float(vertices[index + 1][0]), float(vertices[index + 1][1]))
            flattened = _flatten_quadratic(current[-1], control, end, curve_tolerance)
            current.extend(flattened[1:])
            index += 2
            continue

        if code == MplPath.CURVE4:
            if not current:
                current = [vertex]
                start_point = vertex
                index += 1
                continue
            if index + 2 >= total:
                break
            control1 = vertex
            control2 = (float(vertices[index + 1][0]), float(vertices[index + 1][1]))
            end = (float(vertices[index + 2][0]), float(vertices[index + 2][1]))
            flattened = _flatten_cubic(current[-1], control1, control2, end, curve_tolerance)
            current.extend(flattened[1:])
            index += 3
            continue

        if code == MplPath.CLOSEPOLY:
            if current and start_point is not None and _distance(current[-1], start_point) > _EPS:
                current.append(start_point)
            flush_current()
            start_point = None
            index += 1
            continue

        if code == MplPath.STOP:
            break

        index += 1

    flush_current()
    if not strokes:
        raise ValueError('No drawable strokes extracted from path.')
    return _simplify_strokes(tuple(strokes), simplify_epsilon)


def _guard_grouped_text_density(
    glyphs: tuple[TextGlyphOutline, ...],
) -> None:
    total_points = 0
    for glyph in glyphs:
        for stroke in glyph.strokes:
            if len(stroke) > _MAX_TEXT_CONTOUR_POINTS:
                raise TextDensityError(
                    f'text glyph {glyph.text!r} produced a contour with {len(stroke)} points, exceeding the limit of {_MAX_TEXT_CONTOUR_POINTS}.'
                )
            total_points += len(stroke)
            if total_points > _MAX_TEXT_TOTAL_POINTS:
                raise TextDensityError(
                    f'text produced {total_points} points, exceeding the limit of {_MAX_TEXT_TOTAL_POINTS}.'
                )
    if total_points <= 0:
        raise ValueError('text produced no drawable glyph strokes.')


def _vectorize_text_grouped_with_templates(
    text: str,
    *,
    font_family: str | None,
    line_height: float,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> tuple[TextGlyphOutline, ...]:
    letter_spacing_em, word_spacing_em, uppercase_advance_scale = _normalized_text_spacing_defaults()
    glyphs: list[TextGlyphOutline] = []
    y_offset = 0.0
    for line_index, raw_line in enumerate(text.split('\n')):
        cursor_x = 0.0
        word_index = -1
        in_word = False
        for char in raw_line:
            template = get_text_glyph_template(
                char,
                font_family=font_family,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
            )
            advance_em = _effective_text_advance_em(
                char,
                base_advance=template.advance,
                letter_spacing_em=letter_spacing_em,
                word_spacing_em=word_spacing_em,
                uppercase_advance_scale=uppercase_advance_scale,
            )
            if char.isspace():
                cursor_x += advance_em
                in_word = False
                continue
            if not in_word:
                word_index += 1
                in_word = True
            translated_strokes = tuple(
                tuple((point[0] + cursor_x, y_offset - point[1]) for point in stroke)
                for stroke in template.strokes
            )
            if translated_strokes:
                glyphs.append(
                    TextGlyphOutline(
                        line_index=line_index,
                        word_index=max(word_index, 0),
                        text=char,
                        strokes=translated_strokes,
                        bbox=_strokes_bounds(translated_strokes),
                        advance=advance_em,
                    )
                )
            cursor_x += advance_em
        y_offset += line_height
    glyph_tuple = tuple(glyphs)
    _guard_grouped_text_density(glyph_tuple)
    return glyph_tuple


def vectorize_text_grouped(
    text: str,
    *,
    font_family: str | None = None,
    line_height: float = 1.35,
    curve_tolerance: float = _DEFAULT_TEXT_CURVE_TOLERANCE,
    simplify_epsilon: float = 0.0,
) -> tuple[TextGlyphOutline, ...]:
    normalized_text = normalize_text_for_text_mode(text)
    if not text.strip():
        raise ValueError('text must not be empty.')
    return _vectorize_text_grouped_with_templates(
        normalized_text,
        font_family=font_family.strip() if isinstance(font_family, str) and font_family.strip() else None,
        line_height=line_height,
        curve_tolerance=curve_tolerance,
        simplify_epsilon=simplify_epsilon,
    )


def vectorize_text(
    text: str,
    *,
    font_family: str | None = None,
    line_height: float = 1.35,
    curve_tolerance: float = _DEFAULT_TEXT_CURVE_TOLERANCE,
    simplify_epsilon: float = 0.0,
) -> tuple[tuple[_Point, ...], ...]:
    glyphs = vectorize_text_grouped(
        text,
        font_family=font_family,
        line_height=line_height,
        curve_tolerance=curve_tolerance,
        simplify_epsilon=simplify_epsilon,
    )
    all_strokes = tuple(stroke for glyph in glyphs for stroke in glyph.strokes)
    if not all_strokes:
        raise ValueError('text produced no drawable glyph strokes.')
    return all_strokes


def _svg_tag_name(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    cleaned = value.strip()
    if not cleaned:
        return default
    cleaned = re.sub(r'[^0-9eE+\-.]', '', cleaned)
    if not cleaned:
        return default
    return float(cleaned)


def _parse_points_attr(raw: str) -> list[_Point]:
    values = re.findall(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', raw)
    if len(values) < 4 or len(values) % 2 != 0:
        raise ValueError('Invalid SVG points attribute.')
    points: list[_Point] = []
    for index in range(0, len(values), 2):
        points.append((float(values[index]), float(values[index + 1])))
    return points


def _parse_svg_path_d(
    path_data: str,
    *,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> tuple[tuple[_Point, ...], ...]:
    token_pattern = re.compile(
        r'[MmLlHhVvCcQqZz]|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
    )
    tokens = token_pattern.findall(path_data)
    if not tokens:
        raise ValueError('SVG path data is empty.')

    def is_command(token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    strokes: list[list[_Point]] = []
    stroke: list[_Point] = []
    command: str | None = None
    cursor: _Point = (0.0, 0.0)
    subpath_start: _Point | None = None

    def flush_stroke() -> None:
        nonlocal stroke
        sanitized = _sanitize_stroke(stroke)
        if sanitized is not None:
            strokes.append(list(sanitized))
        stroke = []

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if is_command(token):
            command = token
            index += 1
        elif command is None:
            raise ValueError('Invalid SVG path data: missing command.')

        if command is None:
            continue

        absolute = command.isupper()
        cmd = command.upper()

        def read_number() -> float:
            nonlocal index
            if index >= len(tokens) or is_command(tokens[index]):
                raise ValueError(f'SVG path command {command} is missing parameters.')
            value = float(tokens[index])
            index += 1
            return value

        if cmd == 'M':
            x = read_number()
            y = read_number()
            if not absolute:
                x += cursor[0]
                y += cursor[1]
            flush_stroke()
            cursor = (x, y)
            subpath_start = cursor
            stroke = [cursor]
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_number()
                y = read_number()
                if not absolute:
                    x += cursor[0]
                    y += cursor[1]
                cursor = (x, y)
                stroke.append(cursor)
            command = 'L' if absolute else 'l'
            continue

        if cmd == 'L':
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_number()
                y = read_number()
                if not absolute:
                    x += cursor[0]
                    y += cursor[1]
                cursor = (x, y)
                if not stroke:
                    stroke = [cursor]
                    subpath_start = cursor
                else:
                    stroke.append(cursor)
            continue

        if cmd == 'H':
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_number()
                if not absolute:
                    x += cursor[0]
                cursor = (x, cursor[1])
                if not stroke:
                    stroke = [cursor]
                    subpath_start = cursor
                else:
                    stroke.append(cursor)
            continue

        if cmd == 'V':
            while index < len(tokens) and not is_command(tokens[index]):
                y = read_number()
                if not absolute:
                    y += cursor[1]
                cursor = (cursor[0], y)
                if not stroke:
                    stroke = [cursor]
                    subpath_start = cursor
                else:
                    stroke.append(cursor)
            continue

        if cmd == 'Q':
            while index < len(tokens) and not is_command(tokens[index]):
                c1x = read_number()
                c1y = read_number()
                ex = read_number()
                ey = read_number()
                if not absolute:
                    c1x += cursor[0]
                    c1y += cursor[1]
                    ex += cursor[0]
                    ey += cursor[1]
                if not stroke:
                    stroke = [cursor]
                    subpath_start = cursor
                flattened = _flatten_quadratic(cursor, (c1x, c1y), (ex, ey), curve_tolerance)
                stroke.extend(flattened[1:])
                cursor = (ex, ey)
            continue

        if cmd == 'C':
            while index < len(tokens) and not is_command(tokens[index]):
                c1x = read_number()
                c1y = read_number()
                c2x = read_number()
                c2y = read_number()
                ex = read_number()
                ey = read_number()
                if not absolute:
                    c1x += cursor[0]
                    c1y += cursor[1]
                    c2x += cursor[0]
                    c2y += cursor[1]
                    ex += cursor[0]
                    ey += cursor[1]
                if not stroke:
                    stroke = [cursor]
                    subpath_start = cursor
                flattened = _flatten_cubic(
                    cursor,
                    (c1x, c1y),
                    (c2x, c2y),
                    (ex, ey),
                    curve_tolerance,
                )
                stroke.extend(flattened[1:])
                cursor = (ex, ey)
            continue

        if cmd == 'Z':
            if stroke and subpath_start is not None and _distance(stroke[-1], subpath_start) > _EPS:
                stroke.append(subpath_start)
            flush_stroke()
            subpath_start = None
            continue

        raise ValueError(f'Unsupported SVG path command: {command}')

    flush_stroke()
    if not strokes:
        raise ValueError('No drawable path segments found in SVG.')
    return _simplify_strokes(tuple(tuple(point for point in stroke) for stroke in strokes), simplify_epsilon)


def _has_svg_transform(element: ET.Element) -> bool:
    transform = element.attrib.get('transform')
    return isinstance(transform, str) and bool(transform.strip())


def vectorize_svg(
    svg_text: str,
    *,
    curve_tolerance: float = 0.015,
    simplify_epsilon: float = 0.0,
    circle_segments: int = 48,
) -> tuple[tuple[_Point, ...], ...]:
    if not isinstance(svg_text, str) or not svg_text.strip():
        raise ValueError('SVG payload must be a non-empty string.')

    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError(f'Invalid SVG XML: {exc}') from exc

    strokes: list[tuple[_Point, ...]] = []

    for element in root.iter():
        tag = _svg_tag_name(element.tag)
        if tag == 'g' and _has_svg_transform(element):
            raise ValueError('SVG groups with transforms are not supported in draw mode v1.')
        if tag in {'line', 'polyline', 'polygon', 'rect', 'circle', 'ellipse', 'path'} and _has_svg_transform(element):
            raise ValueError(f'SVG transformed {tag} elements are not supported in draw mode v1.')
        if tag == 'line':
            x1 = _parse_float(element.attrib.get('x1'))
            y1 = _parse_float(element.attrib.get('y1'))
            x2 = _parse_float(element.attrib.get('x2'))
            y2 = _parse_float(element.attrib.get('y2'))
            sanitized = _sanitize_stroke([(x1, y1), (x2, y2)])
            if sanitized is not None:
                strokes.append(sanitized)
        elif tag in ('polyline', 'polygon'):
            points = _parse_points_attr(element.attrib.get('points', ''))
            if tag == 'polygon' and points and _distance(points[0], points[-1]) > _EPS:
                points.append(points[0])
            sanitized = _sanitize_stroke(points)
            if sanitized is not None:
                strokes.append(sanitized)
        elif tag == 'rect':
            x = _parse_float(element.attrib.get('x'))
            y = _parse_float(element.attrib.get('y'))
            width = _parse_float(element.attrib.get('width'))
            height = _parse_float(element.attrib.get('height'))
            if width <= 0.0 or height <= 0.0:
                continue
            rect = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
                (x, y),
            ]
            sanitized = _sanitize_stroke(rect)
            if sanitized is not None:
                strokes.append(sanitized)
        elif tag == 'circle':
            cx = _parse_float(element.attrib.get('cx'))
            cy = _parse_float(element.attrib.get('cy'))
            r = _parse_float(element.attrib.get('r'))
            if r <= 0.0:
                continue
            points = [
                (
                    cx + r * math.cos(2.0 * math.pi * i / float(circle_segments)),
                    cy + r * math.sin(2.0 * math.pi * i / float(circle_segments)),
                )
                for i in range(circle_segments)
            ]
            points.append(points[0])
            sanitized = _sanitize_stroke(points)
            if sanitized is not None:
                strokes.append(sanitized)
        elif tag == 'ellipse':
            cx = _parse_float(element.attrib.get('cx'))
            cy = _parse_float(element.attrib.get('cy'))
            rx = _parse_float(element.attrib.get('rx'))
            ry = _parse_float(element.attrib.get('ry'))
            if rx <= 0.0 or ry <= 0.0:
                continue
            points = [
                (
                    cx + rx * math.cos(2.0 * math.pi * i / float(circle_segments)),
                    cy + ry * math.sin(2.0 * math.pi * i / float(circle_segments)),
                )
                for i in range(circle_segments)
            ]
            points.append(points[0])
            sanitized = _sanitize_stroke(points)
            if sanitized is not None:
                strokes.append(sanitized)
        elif tag == 'path':
            raw_d = element.attrib.get('d', '')
            if raw_d.strip():
                path_strokes = _parse_svg_path_d(
                    raw_d,
                    curve_tolerance=curve_tolerance,
                    simplify_epsilon=simplify_epsilon,
                )
                strokes.extend(path_strokes)

    if not strokes:
        raise ValueError('SVG payload produced no drawable strokes.')
    return _simplify_strokes(tuple(strokes), simplify_epsilon)


def trace_line_art_image(
    image_bytes: bytes,
    *,
    min_perimeter_px: float = 24.0,
    contour_simplify_ratio: float = 0.005,
    max_strokes: int = 512,
) -> tuple[tuple[tuple[_Point, ...], ...], tuple[int, int]]:
    if not image_bytes:
        raise ValueError('Image payload is empty.')

    array = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Failed to decode image payload.')

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = numpy.ones((3, 3), dtype=numpy.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes: list[tuple[_Point, ...]] = []

    for contour in contours:
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < min_perimeter_px:
            continue
        epsilon = max(0.5, contour_simplify_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [(float(point[0][0]), float(point[0][1])) for point in approx]
        if len(points) < 2:
            continue
        if _distance(points[0], points[-1]) > _EPS:
            points.append(points[0])
        sanitized = _sanitize_stroke(points)
        if sanitized is not None:
            strokes.append(sanitized)

    if not strokes:
        raise ValueError('No drawable contours were extracted from image.')

    strokes.sort(
        key=lambda stroke: sum(_distance(stroke[index], stroke[index + 1]) for index in range(len(stroke) - 1)),
        reverse=True,
    )
    strokes = strokes[:max_strokes]
    return tuple(strokes), (int(image.shape[1]), int(image.shape[0]))


def default_placement(writable_bounds: dict[str, float]) -> VectorPlacement:
    width = writable_bounds['x_max'] - writable_bounds['x_min']
    height = writable_bounds['y_max'] - writable_bounds['y_min']
    return VectorPlacement(
        x=writable_bounds['x_min'] + (width * 0.22),
        y=writable_bounds['y_min'] + (height * 0.14),
        scale=1.0,
    )


def normalize_placement(raw: Any, writable_bounds: dict[str, float]) -> VectorPlacement:
    base = default_placement(writable_bounds)
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError('placement must be an object with x, y, and scale.')

    def read_float(key: str, fallback: float) -> float:
        value = raw.get(key, fallback)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'placement.{key} must be numeric.')
        if not math.isfinite(numeric):
            raise ValueError(f'placement.{key} must be finite.')
        return numeric

    x = read_float('x', base.x)
    y = read_float('y', base.y)
    scale = read_float('scale', base.scale)
    if scale <= 0.0:
        raise ValueError('placement.scale must be > 0.')
    return VectorPlacement(x=x, y=y, scale=scale)


def _count_outside_points(
    strokes: tuple[tuple[_Point, ...], ...],
    writable_bounds: dict[str, float],
) -> int:
    outside_points = 0
    for stroke in strokes:
        for point in stroke:
            if not (
                writable_bounds['x_min'] <= point[0] <= writable_bounds['x_max']
                and writable_bounds['y_min'] <= point[1] <= writable_bounds['y_max']
            ):
                outside_points += 1
    return outside_points


def place_strokes_on_board(
    strokes: tuple[tuple[_Point, ...], ...],
    *,
    writable_bounds: dict[str, float],
    placement: VectorPlacement,
    fit_padding: float = 0.9,
) -> tuple[tuple[tuple[_Point, ...], ...], PlacementResult]:
    source_bounds = _strokes_bounds(strokes)
    source_width = max(source_bounds.width, _EPS)
    source_height = max(source_bounds.height, _EPS)

    board_width = writable_bounds['x_max'] - writable_bounds['x_min']
    board_height = writable_bounds['y_max'] - writable_bounds['y_min']
    if board_width <= _EPS or board_height <= _EPS:
        raise ValueError('Writable board bounds are invalid.')

    fit_scale = min((board_width * fit_padding) / source_width, (board_height * fit_padding) / source_height)
    final_scale = fit_scale * placement.scale

    source_center_x = 0.5 * (source_bounds.x_min + source_bounds.x_max)
    source_center_y = 0.5 * (source_bounds.y_min + source_bounds.y_max)

    placed: list[tuple[_Point, ...]] = []
    outside_points = 0
    for stroke in strokes:
        transformed = tuple(
            (
                placement.x + (point[0] - source_center_x) * final_scale,
                placement.y + (point[1] - source_center_y) * final_scale,
            )
            for point in stroke
        )
        for point in transformed:
            if not (
                writable_bounds['x_min'] <= point[0] <= writable_bounds['x_max']
                and writable_bounds['y_min'] <= point[1] <= writable_bounds['y_max']
            ):
                outside_points += 1
        placed.append(transformed)

    placed_tuple = tuple(placed)
    placed_bounds = _strokes_bounds(placed_tuple)
    return (
        placed_tuple,
        PlacementResult(
            placement=placement,
            base_fit_scale=fit_scale,
            final_scale=final_scale,
            bounds=placed_bounds,
            outside_points=outside_points,
        ),
    )


def place_grouped_text_on_board(
    glyphs: tuple[TextGlyphOutline, ...],
    *,
    writable_bounds: dict[str, float],
    placement: VectorPlacement,
    fit_padding: float = 0.9,
    text_upward_bias_em: float = 0.0,
) -> tuple[tuple[TextGlyphOutline, ...], PlacementResult]:
    if not glyphs:
        raise ValueError('No grouped text glyphs available to place on board.')
    flat_strokes = tuple(stroke for glyph in glyphs for stroke in glyph.strokes)
    if not flat_strokes:
        raise ValueError('Grouped text glyphs did not contain drawable strokes.')
    stroke_counts = tuple(len(glyph.strokes) for glyph in glyphs)
    placed_strokes, placement_result = place_strokes_on_board(
        flat_strokes,
        writable_bounds=writable_bounds,
        placement=placement,
        fit_padding=fit_padding,
    )
    if text_upward_bias_em != 0.0:
        _log_text_upward_bias_once(text_upward_bias_em)
        y_bias = -placement_result.final_scale * text_upward_bias_em
        placed_strokes = tuple(
            tuple((point[0], point[1] + y_bias) for point in stroke)
            for stroke in placed_strokes
        )
    cleaned_strokes = _apply_text_vector_cleanup(placed_strokes)
    placed_glyphs: list[TextGlyphOutline] = []
    offset = 0
    for glyph, stroke_count in zip(glyphs, stroke_counts):
        glyph_strokes = tuple(cleaned_strokes[offset:offset + stroke_count])
        offset += stroke_count
        placed_glyphs.append(
            TextGlyphOutline(
                line_index=glyph.line_index,
                word_index=glyph.word_index,
                text=glyph.text,
                strokes=glyph_strokes,
                bbox=_strokes_bounds(glyph_strokes),
                advance=glyph.advance,
            )
        )
    placed_tuple = tuple(placed_glyphs)
    adjusted_strokes = tuple(stroke for glyph in placed_tuple for stroke in glyph.strokes)
    adjusted_bounds = _strokes_bounds(adjusted_strokes)
    adjusted_result = PlacementResult(
        placement=placement_result.placement,
        base_fit_scale=placement_result.base_fit_scale,
        final_scale=placement_result.final_scale,
        bounds=adjusted_bounds,
        outside_points=_count_outside_points(adjusted_strokes, writable_bounds),
    )
    return placed_tuple, adjusted_result


def _simplify_draw_paths(
    strokes: tuple[tuple[_Point, ...], ...],
    tolerance_m: float,
) -> tuple[tuple[_Point, ...], ...]:
    if tolerance_m <= 0.0:
        return tuple(
            sanitized
            for stroke in strokes
            if (sanitized := _sanitize_stroke(list(stroke))) is not None
        )
    simplified: list[tuple[_Point, ...]] = []
    for stroke in strokes:
        reduced = _rdp(list(stroke), tolerance_m)
        sanitized = _sanitize_stroke(reduced)
        if sanitized is not None:
            simplified.append(sanitized)
    return tuple(simplified)


def _merge_draw_paths(
    strokes: tuple[tuple[_Point, ...], ...],
    *,
    gap_tolerance_m: float,
    heading_tolerance_deg: float = 30.0,
) -> tuple[tuple[_Point, ...], ...]:
    if len(strokes) <= 1 or gap_tolerance_m <= 0.0:
        return strokes
    remaining = list(strokes)
    merged: list[tuple[_Point, ...]] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed and remaining:
            changed = False
            current_end = current[-1]
            current_heading = _stroke_heading(current, from_start=False)
            for index, candidate in enumerate(remaining):
                candidate_start = candidate[0]
                gap = _distance(current_end, candidate_start)
                if gap > gap_tolerance_m:
                    continue
                candidate_heading = _stroke_heading(candidate, from_start=True)
                if _heading_delta_deg(current_heading, candidate_heading) > heading_tolerance_deg:
                    continue
                joined = list(current)
                if gap > _EPS:
                    joined.append(candidate_start)
                joined.extend(candidate[1:])
                sanitized = _sanitize_stroke(joined)
                if sanitized is None:
                    continue
                current = sanitized
                del remaining[index]
                changed = True
                break
        merged.append(current)
    return tuple(merged)


def _path_sort_key(stroke: tuple[_Point, ...], index: int) -> tuple[float, float, float, int]:
    xs = [point[0] for point in stroke]
    ys = [point[1] for point in stroke]
    return (min(xs), min(ys), -_stroke_length(stroke), index)


def _order_draw_paths(
    strokes: tuple[tuple[_Point, ...], ...],
) -> tuple[tuple[_Point, ...], ...]:
    if len(strokes) <= 1:
        return strokes
    indexed = list(enumerate(strokes))
    indexed.sort(key=lambda item: _path_sort_key(item[1], item[0]))
    _, current = indexed.pop(0)
    ordered = [current]
    current_point = current[-1]
    while indexed:
        best_choice: tuple[int, tuple[_Point, ...], float, tuple[float, float, float, int]] | None = None
        for list_index, (original_index, stroke) in enumerate(indexed):
            forward_cost = _distance(current_point, stroke[0])
            reverse_cost = _distance(current_point, stroke[-1])
            candidate_key = _path_sort_key(stroke, original_index)
            if best_choice is None or forward_cost < best_choice[2] - _EPS or (
                abs(forward_cost - best_choice[2]) <= _EPS and candidate_key < best_choice[3]
            ):
                best_choice = (list_index, stroke, forward_cost, candidate_key)
            reversed_stroke = tuple(reversed(stroke))
            reversed_key = _path_sort_key(reversed_stroke, original_index)
            if best_choice is None or reverse_cost < best_choice[2] - _EPS or (
                abs(reverse_cost - best_choice[2]) <= _EPS and reversed_key < best_choice[3]
            ):
                best_choice = (list_index, reversed_stroke, reverse_cost, reversed_key)
        if best_choice is None:
            break
        chosen_index, chosen_stroke, _, _ = best_choice
        indexed.pop(chosen_index)
        ordered.append(chosen_stroke)
        current_point = chosen_stroke[-1]
    return tuple(ordered)


def cleanup_draw_strokes(
    strokes: tuple[tuple[_Point, ...], ...],
    *,
    simplify_tolerance_m: float,
    preserve_order: bool = False,
) -> tuple[tuple[_Point, ...], ...]:
    simplified = _simplify_draw_paths(strokes, simplify_tolerance_m)
    merged = _merge_draw_paths(
        simplified,
        gap_tolerance_m=simplify_tolerance_m,
    )
    if preserve_order:
        return merged
    return _order_draw_paths(merged)


def _shrink_bounds(bounds: dict[str, float], margin_m: float) -> dict[str, float]:
    shrink = max(0.0, float(margin_m))
    shrunk = {
        'x_min': bounds['x_min'] + shrink,
        'x_max': bounds['x_max'] - shrink,
        'y_min': bounds['y_min'] + shrink,
        'y_max': bounds['y_max'] - shrink,
    }
    if shrunk['x_max'] <= shrunk['x_min'] or shrunk['y_max'] <= shrunk['y_min']:
        raise ValueError('Writable bounds are too small after applying draw fit margin.')
    return shrunk


def place_draw_strokes_on_board(
    strokes: tuple[tuple[_Point, ...], ...],
    *,
    writable_bounds: dict[str, float],
    placement: VectorPlacement,
    fit_margin_m: float,
) -> tuple[tuple[tuple[_Point, ...], ...], PlacementResult]:
    fit_bounds = _shrink_bounds(writable_bounds, fit_margin_m)
    source_bounds = _strokes_bounds(strokes)
    source_width = max(source_bounds.width, _EPS)
    source_height = max(source_bounds.height, _EPS)
    fit_width = fit_bounds['x_max'] - fit_bounds['x_min']
    fit_height = fit_bounds['y_max'] - fit_bounds['y_min']
    fit_scale = min(fit_width / source_width, fit_height / source_height)
    final_scale = fit_scale * placement.scale
    source_center_x = 0.5 * (source_bounds.x_min + source_bounds.x_max)
    source_center_y = 0.5 * (source_bounds.y_min + source_bounds.y_max)

    placed: list[tuple[_Point, ...]] = []
    outside_points = 0
    for stroke in strokes:
        transformed = tuple(
            (
                placement.x + (point[0] - source_center_x) * final_scale,
                placement.y + (point[1] - source_center_y) * final_scale,
            )
            for point in stroke
        )
        for point in transformed:
            if not (
                writable_bounds['x_min'] <= point[0] <= writable_bounds['x_max']
                and writable_bounds['y_min'] <= point[1] <= writable_bounds['y_max']
            ):
                outside_points += 1
        placed.append(transformed)

    placed_tuple = tuple(placed)
    return (
        placed_tuple,
        PlacementResult(
            placement=placement,
            base_fit_scale=fit_scale,
            final_scale=final_scale,
            bounds=_strokes_bounds(placed_tuple),
            outside_points=outside_points,
        ),
    )


def _rotate_point(point: _Point, theta: float) -> _Point:
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return (
        point[0] * cos_theta - point[1] * sin_theta,
        point[0] * sin_theta + point[1] * cos_theta,
    )


def draw_segments_from_pen_strokes(
    pen_strokes: tuple[tuple[_Point, ...], ...],
    *,
    theta_ref: float,
    pen_offset_x_m: float,
    pen_offset_y_m: float,
) -> tuple[DrawPathSegment, ...]:
    if not pen_strokes:
        raise ValueError('No draw strokes available to convert into plan segments.')
    rotated_offset = _rotate_point((pen_offset_x_m, pen_offset_y_m), theta_ref)

    def to_body_points(stroke: tuple[_Point, ...]) -> tuple[_Point, ...]:
        return tuple(
            (
                point[0] - rotated_offset[0],
                point[1] - rotated_offset[1],
            )
            for point in stroke
        )

    segments: list[DrawPathSegment] = []
    previous_end: _Point | None = None
    for stroke in pen_strokes:
        body_points = to_body_points(stroke)
        if previous_end is not None:
            start_point = body_points[0]
            if _distance(previous_end, start_point) > _EPS:
                segments.append(
                    DrawPathSegment(
                        draw=False,
                        points=(previous_end, start_point),
                    )
                )
        segments.append(DrawPathSegment(draw=True, points=body_points))
        previous_end = body_points[-1]
    return tuple(segments)


def draw_plan_to_dict(
    segments: tuple[DrawPathSegment, ...],
    *,
    theta_ref: float,
) -> dict[str, Any]:
    return {
        'frame': 'board',
        'theta_ref': float(theta_ref),
        'segments': [
            {
                'draw': bool(segment.draw),
                'type': 'line' if len(segment.points) == 2 else 'polyline',
                'points': [[float(point[0]), float(point[1])] for point in segment.points],
            }
            for segment in segments
        ],
    }


def strokes_to_draw_plan(strokes: tuple[tuple[_Point, ...], ...]) -> dict[str, Any]:
    serialized_strokes = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        stroke_type = 'line' if len(stroke) == 2 else 'polyline'
        serialized_strokes.append(
            {
                'type': stroke_type,
                'draw': True,
                'points': [[float(point[0]), float(point[1])] for point in stroke],
            }
        )
    if not serialized_strokes:
        raise ValueError('No drawable strokes available after serialization.')
    return {'frame': 'board', 'strokes': serialized_strokes}


def stroke_stats(strokes: tuple[tuple[_Point, ...], ...]) -> dict[str, Any]:
    bounds = _strokes_bounds(strokes)
    point_count = sum(len(stroke) for stroke in strokes)
    return {
        'stroke_count': len(strokes),
        'point_count': point_count,
        'bounds': {
            'x_min': bounds.x_min,
            'x_max': bounds.x_max,
            'y_min': bounds.y_min,
            'y_max': bounds.y_max,
            'width': bounds.width,
            'height': bounds.height,
        },
    }
