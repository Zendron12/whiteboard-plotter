from __future__ import annotations

# Compatibility façade:
# New mainline imports should prefer wall_climber.ingestion.* and wall_climber.canonical_ops.

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

from wall_climber.canonical_adapters import (
    canonical_plan_to_draw_strokes,
    canonical_plan_to_segment_payload,
    canonical_plan_to_legacy_strokes,
    canonical_plan_to_sampled_paths,
    pen_strokes_to_canonical_plan,
)
from wall_climber.canonical_builders import draw_strokes_to_canonical_plan
from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalPathPlan,
    CubicBezier,
    LineSegment,
    PenDown,
    PenUp,
    QuadraticBezier,
    TravelMove,
)
from wall_climber.shared_config import load_shared_config
from wall_climber.text_vector_font import get_glyph as get_legacy_glyph


_Point = tuple[float, float]
_EPS = 1.0e-9
_PACKAGE_NAME = 'wall_climber'
_BUNDLED_FONT_NAME = 'DejaVuSans.ttf'
_RELIEF_SVG_FONT_NAME = 'ReliefSingleLineSVG-Regular.svg'
_HERSHEY_SVG_FONT_NAME = 'HersheySans1.svg'
_RELIEF_SVG_MIN_CURVE_TOLERANCE = 0.005
_DEFAULT_TEXT_CURVE_TOLERANCE = 0.008
_MAX_TEXT_CONTOUR_POINTS = 512
_MAX_TEXT_TOTAL_POINTS = 12000
_TEXT_WRAP_RIGHT_MARGIN_EPS_M = 0.003
_GLYPH_LOCAL_PADDING_EM = 0.012
_TEXT_FONT_SOURCE_RELIEF = 'relief_singleline'
_TEXT_FONT_SOURCE_HERSHEY = 'hershey_sans_1'
_TEXT_FONT_SOURCE_DEJAVU = 'dejavu_sans'
_DEJAVU_LETTER_SPACING_SCALE = 0.84
_VALID_TEXT_FONT_SOURCES = frozenset(
    {
        _TEXT_FONT_SOURCE_RELIEF,
        _TEXT_FONT_SOURCE_HERSHEY,
        _TEXT_FONT_SOURCE_DEJAVU,
    }
)
_LOG = logging.getLogger(__name__)
_LOGGED_TEXT_FONT_SOURCES: set[str] = set()
_LOGGED_TEXT_SOURCE_POLICIES: set[str] = set()
_LOGGED_TEXT_FALLBACK = False
_LOGGED_TEXT_NORMALIZATION = False
_LOGGED_TEXT_UPWARD_BIAS: set[float] = set()


@dataclass(frozen=True)
class SvgStrokeFont:
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
class ImageRoutingMetrics:
    approximate_color_count: int
    contrast: float
    edge_density: float
    contour_count: int
    tonal_variation: float
    entropy: float
    background_whiteness: float
    texture_score: float
    dark_pixel_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            'approximate_color_count': int(self.approximate_color_count),
            'contrast': float(self.contrast),
            'edge_density': float(self.edge_density),
            'contour_count': int(self.contour_count),
            'tonal_variation': float(self.tonal_variation),
            'entropy': float(self.entropy),
            'background_whiteness': float(self.background_whiteness),
            'texture_score': float(self.texture_score),
            'dark_pixel_ratio': float(self.dark_pixel_ratio),
        }


@dataclass(frozen=True)
class ImageRouteDecision:
    route: str
    metrics: ImageRoutingMetrics
    simple_outline_score: float
    complex_tonal_score: float
    colored_illustration_score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'route': self.route,
            'metrics': self.metrics.to_dict(),
            'scores': {
                'simple_outline': float(self.simple_outline_score),
                'complex_tonal': float(self.complex_tonal_score),
                'colored_illustration': float(self.colored_illustration_score),
            },
            'rationale': self.rationale,
        }


@dataclass(frozen=True)
class ImageVectorizationResult:
    plan: CanonicalPathPlan
    image_size: tuple[int, int]
    route_decision: ImageRouteDecision
    branch_details: dict[str, Any]
    command_metadata: tuple[dict[str, Any] | None, ...] = ()
    curve_fit_debug: dict[str, Any] | None = None
    raw_contours: tuple[tuple[_Point, ...], ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        pipeline = {
            **self.route_decision.to_dict(),
            'branch_details': dict(self.branch_details),
        }
        if self.curve_fit_debug:
            pipeline['curve_fit_summary'] = dict(self.curve_fit_debug.get('fit_summary') or {})
        return {
            'width_px': int(self.image_size[0]),
            'height_px': int(self.image_size[1]),
            'pipeline': pipeline,
        }


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
    source: str


class TextDensityError(ValueError):
    pass


_IMAGE_ROUTE_SIMPLE_OUTLINE = 'simple_outline'
_IMAGE_ROUTE_COMPLEX_TONAL = 'complex_tonal'
_IMAGE_ROUTE_COLORED_ILLUSTRATION = 'colored_illustration'


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


def _normalize_text_font_source(font_source: str | None) -> str | None:
    if font_source is None:
        return None
    normalized = str(font_source).strip().lower()
    if not normalized:
        return None
    if normalized not in _VALID_TEXT_FONT_SOURCES:
        raise ValueError(
            f'font_source must be one of {sorted(_VALID_TEXT_FONT_SOURCES)}.'
        )
    return normalized


def _normalize_text_glyph_template(template: TextGlyphTemplate) -> TextGlyphTemplate:
    global _LOGGED_TEXT_NORMALIZATION
    if template.bbox is None or not template.strokes:
        return template
    if template.source not in {'relief_svg', 'hershey_svg', 'text_vector_font_fallback'}:
        return template

    x_shift = _GLYPH_LOCAL_PADDING_EM - float(template.bbox.x_min)
    if abs(x_shift) <= _EPS:
        return template

    normalized_strokes = tuple(
        tuple((point[0] + x_shift, point[1]) for point in stroke)
        for stroke in template.strokes
    )
    normalized_bbox = _strokes_bounds(normalized_strokes)
    preserved_advance = max(float(normalized_bbox.x_max), float(template.advance) + x_shift)
    letter_spacing_em, _, uppercase_advance_scale = _normalized_text_spacing_defaults()
    is_uppercase_letter = (
        len(template.text) == 1
        and template.text.isalpha()
        and template.text.upper() == template.text
        and template.text.lower() != template.text
    )
    normalized_advance = preserved_advance
    if not is_uppercase_letter:
        tight_advance = float(normalized_bbox.x_max) + _GLYPH_LOCAL_PADDING_EM
        normalized_advance = max(float(normalized_bbox.x_max), min(preserved_advance, tight_advance))
    elif uppercase_advance_scale > _EPS:
        min_uppercase_advance = (
            float(normalized_bbox.x_max)
            + _GLYPH_LOCAL_PADDING_EM
            - letter_spacing_em
        ) / uppercase_advance_scale
        normalized_advance = max(normalized_advance, min_uppercase_advance)
    if not _LOGGED_TEXT_NORMALIZATION:
        _LOGGED_TEXT_NORMALIZATION = True
        _LOG.info(
            'Text glyph normalization enabled for line-font sources with glyph_local_padding_em=%.4f and advance compensation.',
            _GLYPH_LOCAL_PADDING_EM,
        )
    return TextGlyphTemplate(
        text=template.text,
        strokes=normalized_strokes,
        bbox=normalized_bbox,
        advance=normalized_advance,
        source=template.source,
    )


@lru_cache(maxsize=1)
def _text_wrap_line_width_units() -> float:
    shared = load_shared_config()
    glyph_height = max(float(shared.text_layout.glyph_height), 1.0e-6)

    # Keep only the left-side margin for text wrapping.
    # Do not reserve an artificial right text margin anymore.
    usable_width_m = (
        float(shared.board.width)
        - float(shared.text_layout.left_margin)
    )
    if usable_width_m <= _EPS:
        raise ValueError('Configured text layout width collapsed after margins were applied.')
    return usable_width_m / glyph_height


def _effective_text_advance_em(
    char: str,
    *,
    base_advance: float,
    letter_spacing_em: float,
    word_spacing_em: float,
    uppercase_advance_scale: float,
    font_source: str | None = None,
) -> float:
    advance = max(0.0, float(base_advance))
    if char.isspace():
        return max(advance, word_spacing_em)
    if char.isalpha() and char.upper() == char and char.lower() != char:
        advance *= uppercase_advance_scale
    if _normalize_text_font_source(font_source) == _TEXT_FONT_SOURCE_DEJAVU:
        letter_spacing_em *= _DEJAVU_LETTER_SPACING_SCALE
    return advance + letter_spacing_em


def _text_token_layout_entries(
    token: str,
    *,
    font_family: str | None,
    font_source: str | None,
    curve_tolerance: float,
    simplify_epsilon: float,
    letter_spacing_em: float,
    word_spacing_em: float,
    uppercase_advance_scale: float,
) -> tuple[tuple[str, TextGlyphTemplate, float], ...]:
    entries: list[tuple[str, TextGlyphTemplate, float]] = []
    for char in token:
        template = get_text_glyph_template(
            char,
            font_family=font_family,
            font_source=font_source,
            curve_tolerance=curve_tolerance,
            simplify_epsilon=simplify_epsilon,
        )
        advance_em = _effective_text_advance_em(
            char,
            base_advance=template.advance,
            letter_spacing_em=letter_spacing_em,
            word_spacing_em=word_spacing_em,
            uppercase_advance_scale=uppercase_advance_scale,
            font_source=font_source,
        )
        entries.append((char, template, advance_em))
    return tuple(entries)


def _measure_text_token_width(
    token: str,
    *,
    font_family: str | None,
    font_source: str | None,
    curve_tolerance: float,
    simplify_epsilon: float,
    letter_spacing_em: float,
    word_spacing_em: float,
    uppercase_advance_scale: float,
) -> float:
    return sum(
        entry[2]
        for entry in _text_token_layout_entries(
            token,
            font_family=font_family,
            font_source=font_source,
            curve_tolerance=curve_tolerance,
            simplify_epsilon=simplify_epsilon,
            letter_spacing_em=letter_spacing_em,
            word_spacing_em=word_spacing_em,
            uppercase_advance_scale=uppercase_advance_scale,
        )
    )


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
    *,
    glyph_source: str | None = None,
) -> tuple[tuple[_Point, ...], ...]:
    if glyph_source in {'relief_svg', 'hershey_svg'}:
        return strokes
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
    elif policy == 'hershey_svg':
        _LOG.info('Text vector source policy: Hershey Sans 1-stroke SVG font.')
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
    elif source == 'hershey_svg_installed':
        _LOG.info('Text vector font source: Hershey Sans 1-stroke SVG (installed).')
    elif source == 'hershey_svg_source':
        _LOG.info('Text vector font source: Hershey Sans 1-stroke SVG (source tree).')
    elif source == 'bundled_installed':
        _LOG.info('Text vector font source: bundled DejaVu Sans (installed).')
    elif source == 'bundled_source':
        _LOG.info('Text vector font source: bundled DejaVu Sans (source tree).')
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


def _is_svg_stroke_font_source(font_source: str | None) -> bool:
    normalized_source = _normalize_text_font_source(font_source)
    return normalized_source in {
        _TEXT_FONT_SOURCE_RELIEF,
        _TEXT_FONT_SOURCE_HERSHEY,
    }


def _svg_stroke_font_candidates(
    *,
    font_name: str,
    installed_source: str,
    source_tree_source: str,
) -> tuple[tuple[str, Path], ...]:
    candidates: list[tuple[str, Path]] = []
    try:
        installed = (
            Path(get_package_share_directory(_PACKAGE_NAME))
            / 'fonts'
            / font_name
        )
        candidates.append((installed_source, installed))
    except PackageNotFoundError:
        pass
    source_tree = Path(__file__).resolve().parents[1] / 'fonts' / font_name
    candidates.append((source_tree_source, source_tree))
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


@lru_cache(maxsize=4)
def _load_svg_stroke_font(font_source: str) -> SvgStrokeFont:
    normalized_source = _normalize_text_font_source(font_source)
    if normalized_source == _TEXT_FONT_SOURCE_RELIEF:
        font_name = _RELIEF_SVG_FONT_NAME
        installed_source = 'relief_svg_installed'
        source_tree_source = 'relief_svg_source'
    elif normalized_source == _TEXT_FONT_SOURCE_HERSHEY:
        font_name = _HERSHEY_SVG_FONT_NAME
        installed_source = 'hershey_svg_installed'
        source_tree_source = 'hershey_svg_source'
    else:
        raise ValueError(f'Unsupported SVG stroke font source: {font_source!r}')

    svg_path: Path | None = None
    source = source_tree_source
    for candidate_source, candidate_path in _svg_stroke_font_candidates(
        font_name=font_name,
        installed_source=installed_source,
        source_tree_source=source_tree_source,
    ):
        if candidate_path.is_file():
            svg_path = candidate_path
            source = candidate_source
            break
    if svg_path is None:
        raise FileNotFoundError(
            f'Unable to locate SVG stroke font {font_name!r} '
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
        'SVG stroke font loaded: %d glyphs, cap_height=%.1f, source=%s',
        len(glyphs), cap_height, source,
    )
    return SvgStrokeFont(
        glyphs=glyphs,
        default_advance=default_advance,
        cap_height=cap_height,
        source=source,
    )


def _parse_svg_font_path_d(
    path_data: str,
    cap_height: float,
    curve_tolerance: float,
) -> tuple[tuple[_Point, ...], ...]:
    """Parse an SVG path 'd' attribute into polyline strokes.

    SVG font glyphs should be normalized into the same glyph space used by the
    legacy line-font templates:
      - X grows to the right
      - Y grows upward
      - cap height = 1.0

    The line layout stage later applies the single Y inversion needed for
    top-to-bottom text placement on the board, so we must NOT flip Y here.
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
        return (x / cap_height, y / cap_height)

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


@lru_cache(maxsize=4)
def _svg_stroke_font_render_scale(font_source: str) -> float:
    normalized_source = _normalize_text_font_source(font_source)
    if normalized_source != _TEXT_FONT_SOURCE_HERSHEY:
        return 1.0

    font = _load_svg_stroke_font(normalized_source)
    reference = font.glyphs.get('H')
    if reference is None or not reference[0]:
        return 1.0

    try:
        raw_strokes = _parse_svg_font_path_d(
            reference[0],
            1.0,
            _RELIEF_SVG_MIN_CURVE_TOLERANCE,
        )
        reference_height = _strokes_bounds(raw_strokes).height
    except Exception as exc:
        _LOG.debug('Hershey SVG reference-height detection failed: %s', exc)
        return 1.0

    if reference_height <= _EPS:
        return 1.0
    return min(1.0, max(0.1, float(font.cap_height) / float(reference_height)))


def _glyph_template_from_svg_stroke_font(
    char: str,
    *,
    font_source: str,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> TextGlyphTemplate:
    normalized_source = _normalize_text_font_source(font_source)
    font = _load_svg_stroke_font(normalized_source or _TEXT_FONT_SOURCE_RELIEF)
    if char not in font.glyphs:
        raise KeyError(char)
    path_data, raw_advance = font.glyphs[char]
    normalized_advance = float(raw_advance) / font.cap_height
    glyph_source = (
        'relief_svg' if normalized_source == _TEXT_FONT_SOURCE_RELIEF else 'hershey_svg'
    )
    render_scale = _svg_stroke_font_render_scale(normalized_source or _TEXT_FONT_SOURCE_RELIEF)
    if char.isspace() or not path_data:
        return TextGlyphTemplate(
            text=char,
            strokes=tuple(),
            bbox=None,
            advance=normalized_advance * render_scale,
            source=glyph_source,
        )
    effective_curve_tolerance = max(curve_tolerance, _RELIEF_SVG_MIN_CURVE_TOLERANCE)
    raw_strokes = _parse_svg_font_path_d(
        path_data,
        font.cap_height,
        effective_curve_tolerance,
    )
    if render_scale != 1.0:
        raw_strokes = tuple(
            tuple((point[0] * render_scale, point[1] * render_scale) for point in stroke)
            for stroke in raw_strokes
        )
    if simplify_epsilon > 0.0:
        raw_strokes = _simplify_strokes(raw_strokes, simplify_epsilon)
    return TextGlyphTemplate(
        text=char,
        strokes=raw_strokes,
        bbox=_strokes_bounds(raw_strokes) if raw_strokes else None,
        advance=normalized_advance * render_scale,
        source=glyph_source,
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
    font_source: str | None,
    curve_tolerance: float,
    simplify_epsilon: float,
) -> TextGlyphTemplate:
    def _is_problematic_uppercase(candidate: str) -> bool:
        return (
            len(candidate) == 1
            and candidate.isalpha()
            and candidate.upper() == candidate
            and candidate.lower() != candidate
        )

    normalized_font_source = _normalize_text_font_source(font_source)
    if normalized_font_source is not None:
        if _is_svg_stroke_font_source(normalized_font_source):
            try:
                return _glyph_template_from_svg_stroke_font(
                    char,
                    font_source=normalized_font_source,
                    curve_tolerance=curve_tolerance,
                    simplify_epsilon=simplify_epsilon,
                )
            except KeyError as exc:
                raise ValueError(
                    f'text glyph {char!r} is not available in font_source {normalized_font_source!r}.'
                ) from exc
        if normalized_font_source == _TEXT_FONT_SOURCE_DEJAVU:
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
                raise ValueError(
                    f'text glyph {char!r} could not be rendered from font_source {normalized_font_source!r}: {exc}'
                ) from exc

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

    # 1. Try Relief SingleLine SVG font first
    try:
        return _glyph_template_from_svg_stroke_font(
            char,
            font_source=_TEXT_FONT_SOURCE_RELIEF,
            curve_tolerance=curve_tolerance,
            simplify_epsilon=simplify_epsilon,
        )
    except KeyError:
        pass
    except Exception as exc:
        _LOG.debug('Relief SVG glyph %r load failed: %s', char, exc)

    # 2. For uppercase letters, prefer TextPath before the legacy fallback.
    if _is_problematic_uppercase(char):
        try:
            template = _glyph_template_from_textpath(
                char,
                font_family=None,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
            )
            _LOG.info(
                'Uppercase glyph %r: using TextPath fallback before legacy fallback.',
                char,
            )
            return template
        except TextDensityError:
            raise
        except Exception as exc:
            _LOG.debug('TextPath uppercase glyph %r load failed: %s', char, exc)

    # 3. For non-uppercase cases, or if uppercase TextPath failed, try TextPath here
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
    font_source: str | None = None,
    curve_tolerance: float = _DEFAULT_TEXT_CURVE_TOLERANCE,
    simplify_epsilon: float = 0.0,
) -> TextGlyphTemplate:
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError('char must be a single-character string.')
    normalized_font_source = _normalize_text_font_source(font_source)
    template = _cached_text_glyph_template(
        normalize_text_for_text_mode(char),
        font_family.strip() if isinstance(font_family, str) and font_family.strip() else None,
        normalized_font_source,
        float(curve_tolerance),
        float(simplify_epsilon),
    )
    template = _normalize_text_glyph_template(template)
    if template.source == 'relief_svg':
        _log_text_source_policy_once('relief_svg')
    elif template.source == 'hershey_svg':
        _log_text_source_policy_once('hershey_svg')
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
    font_source: str | None,
    line_height: float,
    curve_tolerance: float,
    simplify_epsilon: float,
    max_line_width_units: float | None,
) -> tuple[TextGlyphOutline, ...]:
    letter_spacing_em, word_spacing_em, uppercase_advance_scale = _normalized_text_spacing_defaults()
    usable_line_width_em = (
        float(max_line_width_units)
        if max_line_width_units is not None
        else _text_wrap_line_width_units()
    )
    if usable_line_width_em <= _EPS:
        raise ValueError('Configured text line width must be positive.')
    glyphs: list[TextGlyphOutline] = []
    y_offset = 0.0
    line_index = 0
    for raw_line in text.split('\n'):
        cursor_x = 0.0
        word_index = -1
        line_has_content = False
        pending_space_width = 0.0
        tokens = re.findall(r'\S+|\s+', raw_line)
        for token in tokens:
            if token.isspace():
                if not line_has_content:
                    continue
                pending_space_width += _measure_text_token_width(
                    token,
                    font_family=font_family,
                    font_source=font_source,
                    curve_tolerance=curve_tolerance,
                    simplify_epsilon=simplify_epsilon,
                    letter_spacing_em=letter_spacing_em,
                    word_spacing_em=word_spacing_em,
                    uppercase_advance_scale=uppercase_advance_scale,
                )
                continue

            word_entries = _text_token_layout_entries(
                token,
                font_family=font_family,
                font_source=font_source,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
                letter_spacing_em=letter_spacing_em,
                word_spacing_em=word_spacing_em,
                uppercase_advance_scale=uppercase_advance_scale,
            )
            word_width = sum(entry[2] for entry in word_entries)
            pending_width = word_width + (pending_space_width if line_has_content else 0.0)
            if line_has_content and (cursor_x + pending_width) > usable_line_width_em:
                line_index += 1
                y_offset += line_height
                cursor_x = 0.0
                word_index = -1
                line_has_content = False
                pending_space_width = 0.0
            elif line_has_content and pending_space_width > 0.0:
                cursor_x += pending_space_width
                pending_space_width = 0.0

            word_index += 1
            for char, template, advance_em in word_entries:
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
                            source=template.source,
                        )
                    )
                cursor_x += advance_em
            line_has_content = True
            pending_space_width = 0.0
        line_index += 1
        y_offset += line_height
    glyph_tuple = tuple(glyphs)
    _guard_grouped_text_density(glyph_tuple)
    return glyph_tuple


def vectorize_text_grouped(
    text: str,
    *,
    font_family: str | None = None,
    font_source: str | None = None,
    line_height: float = 1.35,
    curve_tolerance: float = _DEFAULT_TEXT_CURVE_TOLERANCE,
    simplify_epsilon: float = 0.0,
    max_line_width_units: float | None = None,
) -> tuple[TextGlyphOutline, ...]:
    normalized_text = normalize_text_for_text_mode(text)
    if not text.strip():
        raise ValueError('text must not be empty.')
    return _vectorize_text_grouped_with_templates(
        normalized_text,
        font_family=font_family.strip() if isinstance(font_family, str) and font_family.strip() else None,
        font_source=_normalize_text_font_source(font_source),
        line_height=line_height,
        curve_tolerance=curve_tolerance,
        simplify_epsilon=simplify_epsilon,
        max_line_width_units=max_line_width_units,
    )


def vectorize_text(
    text: str,
    *,
    font_family: str | None = None,
    font_source: str | None = None,
    line_height: float = 1.35,
    curve_tolerance: float = _DEFAULT_TEXT_CURVE_TOLERANCE,
    simplify_epsilon: float = 0.0,
    max_line_width_units: float | None = None,
) -> tuple[tuple[_Point, ...], ...]:
    glyphs = vectorize_text_grouped(
        text,
        font_family=font_family,
        font_source=font_source,
        line_height=line_height,
        curve_tolerance=curve_tolerance,
        simplify_epsilon=simplify_epsilon,
        max_line_width_units=max_line_width_units,
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


def _decode_image_mats(image_bytes: bytes) -> tuple[numpy.ndarray, numpy.ndarray]:
    if not image_bytes:
        raise ValueError('Image payload is empty.')

    array = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    color = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError('Failed to decode image payload.')
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray


def _remove_tiny_binary_components(binary: numpy.ndarray, *, min_area_px: int) -> numpy.ndarray:
    if min_area_px <= 1:
        return binary
    mask = (binary > 0).astype(numpy.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = numpy.zeros_like(binary)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        cleaned[labels == label] = 255
    return cleaned


def _line_art_binary_from_gray(gray: numpy.ndarray) -> numpy.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    blurred = cv2.medianBlur(normalized, 3)

    _, otsu_binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    adaptive_binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    binary = cv2.max(otsu_binary, adaptive_binary)

    connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, numpy.ones((2, 2), dtype=numpy.uint8))

    image_area = int(gray.shape[0] * gray.shape[1])
    min_component_area = max(4, int(round(image_area * 0.000025)))
    return _remove_tiny_binary_components(binary, min_area_px=min_component_area)


def _trace_binary_contours(
    binary: numpy.ndarray,
    *,
    min_perimeter_px: float,
    contour_simplify_ratio: float,
    max_strokes: int,
    close_contours: bool,
) -> tuple[tuple[tuple[_Point, ...], ...], dict[str, Any]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes: list[tuple[_Point, ...]] = []
    kept_contours = 0

    for contour in contours:
        perimeter = float(cv2.arcLength(contour, close_contours))
        if perimeter < min_perimeter_px:
            continue
        epsilon = max(0.5, contour_simplify_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, close_contours)
        points = [(float(point[0][0]), float(point[0][1])) for point in approx]
        if len(points) < 2:
            continue
        if close_contours and _distance(points[0], points[-1]) > _EPS:
            points.append(points[0])
        sanitized = _sanitize_stroke(points)
        if sanitized is not None:
            strokes.append(sanitized)
            kept_contours += 1

    if not strokes:
        raise ValueError('No drawable contours were extracted from image.')

    strokes.sort(key=_stroke_length, reverse=True)
    limited = tuple(strokes[:max_strokes])
    stats = {
        'candidate_contours': int(len(contours)),
        'kept_contours': int(kept_contours),
        'stroke_count': int(len(limited)),
    }
    return limited, stats


def _approximate_color_count(color: numpy.ndarray) -> int:
    height, width = color.shape[:2]
    max_side = max(height, width)
    if max_side > 96:
        scale = 96.0 / float(max_side)
        resized = cv2.resize(
            color,
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = color
    quantized = (resized // 32).reshape(-1, 3)
    return int(numpy.unique(quantized, axis=0).shape[0])


def _normalized_entropy(gray: numpy.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    total = float(hist.sum())
    if total <= _EPS:
        return 0.0
    probabilities = hist / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(numpy.sum(probabilities * numpy.log2(probabilities)))
    return min(1.0, entropy / math.log2(32.0))


def _image_routing_metrics(
    color: numpy.ndarray,
    gray: numpy.ndarray,
) -> ImageRoutingMetrics:
    line_binary = _line_art_binary_from_gray(gray)
    edges = cv2.Canny(gray, 80, 160)
    contrast = float(numpy.percentile(gray, 95) - numpy.percentile(gray, 5)) / 255.0
    edge_density = float(numpy.count_nonzero(edges)) / float(edges.size)
    contour_count = int(len(cv2.findContours(line_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]))
    hist32 = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    occupied_bins = int(numpy.count_nonzero(hist32))
    tonal_variation = float(occupied_bins) / 32.0
    entropy = _normalized_entropy(gray)
    background_whiteness = float(numpy.mean(gray >= 220))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    texture_score = min(1.0, laplacian_var / 1200.0)
    dark_pixel_ratio = float(numpy.mean(gray <= 120))
    return ImageRoutingMetrics(
        approximate_color_count=_approximate_color_count(color),
        contrast=min(1.0, max(0.0, contrast)),
        edge_density=min(1.0, max(0.0, edge_density)),
        contour_count=max(0, contour_count),
        tonal_variation=min(1.0, max(0.0, tonal_variation)),
        entropy=min(1.0, max(0.0, entropy)),
        background_whiteness=min(1.0, max(0.0, background_whiteness)),
        texture_score=min(1.0, max(0.0, texture_score)),
        dark_pixel_ratio=min(1.0, max(0.0, dark_pixel_ratio)),
    )


def _is_sparse_line_art_metrics(metrics: ImageRoutingMetrics) -> bool:
    normalized_color_count = min(1.0, float(metrics.approximate_color_count) / 24.0)
    normalized_edges = min(1.0, metrics.edge_density / 0.12)
    return (
        metrics.background_whiteness >= 0.90
        and metrics.entropy <= 0.22
        and metrics.dark_pixel_ratio <= 0.08
        and normalized_color_count <= 0.35
        and normalized_edges <= 0.68
    )


def _route_from_metrics(metrics: ImageRoutingMetrics) -> ImageRouteDecision:
    normalized_color_count = min(1.0, float(metrics.approximate_color_count) / 24.0)
    normalized_contours = min(1.0, float(metrics.contour_count) / 180.0)
    normalized_edges = min(1.0, metrics.edge_density / 0.12)
    sparse_line_art = _is_sparse_line_art_metrics(metrics)

    simple_outline_score = (
        0.30 * metrics.background_whiteness
        + 0.20 * metrics.contrast
        + 0.15 * (1.0 - metrics.entropy)
        + 0.15 * (1.0 - metrics.tonal_variation)
        + 0.10 * (1.0 - normalized_color_count)
        + 0.10 * (1.0 - metrics.texture_score)
    )
    complex_tonal_score = (
        0.25 * metrics.entropy
        + 0.20 * metrics.tonal_variation
        + 0.15 * metrics.texture_score
        + 0.15 * (1.0 - metrics.background_whiteness)
        + 0.10 * normalized_color_count
        + 0.10 * normalized_edges
        + 0.05 * normalized_contours
    )
    colored_illustration_score = (
        0.24 * metrics.background_whiteness
        + 0.20 * normalized_color_count
        + 0.16 * metrics.contrast
        + 0.14 * normalized_edges
        + 0.10 * (1.0 - metrics.texture_score)
        + 0.10 * (1.0 - metrics.dark_pixel_ratio)
        + 0.06 * (1.0 - abs(metrics.tonal_variation - 0.35))
    )

    if (
        metrics.background_whiteness >= 0.82
        and metrics.entropy <= 0.38
        and metrics.tonal_variation <= 0.35
    ):
        simple_outline_score += 0.15
    if sparse_line_art:
        simple_outline_score += 0.24
        complex_tonal_score -= 0.10
        colored_illustration_score -= 0.06
    if (
        metrics.entropy >= 0.48
        or metrics.tonal_variation >= 0.42
        or metrics.texture_score >= 0.35
    ):
        complex_tonal_score += 0.12
    if (
        metrics.background_whiteness >= 0.68
        and normalized_color_count >= 0.28
        and metrics.texture_score <= 0.48
        and metrics.entropy <= 0.68
    ):
        colored_illustration_score += 0.18
    if metrics.texture_score >= 0.62 or metrics.entropy >= 0.78:
        colored_illustration_score -= 0.08

    if (
        colored_illustration_score > simple_outline_score
        and colored_illustration_score > complex_tonal_score
    ):
        route = _IMAGE_ROUTE_COLORED_ILLUSTRATION
        rationale = 'Moderate color variation on a light background favors foreground isolation plus outline-led illustration tracing.'
    elif simple_outline_score >= complex_tonal_score:
        route = _IMAGE_ROUTE_SIMPLE_OUTLINE
        if sparse_line_art:
            rationale = 'Sparse grayscale line-art on a clean background favors direct line-art tracing over tonal hatching.'
        else:
            rationale = 'High background whiteness and low tonal complexity favor direct outline extraction.'
    else:
        route = _IMAGE_ROUTE_COMPLEX_TONAL
        rationale = 'Higher tonal complexity and texture favor hatch-based tonal rendering.'

    return ImageRouteDecision(
        route=route,
        metrics=metrics,
        simple_outline_score=float(simple_outline_score),
        complex_tonal_score=float(complex_tonal_score),
        colored_illustration_score=float(colored_illustration_score),
        rationale=rationale,
    )


def route_image_vector_pipeline(image_bytes: bytes) -> ImageRouteDecision:
    color, gray = _decode_image_mats(image_bytes)
    return _route_from_metrics(_image_routing_metrics(color, gray))


def _hatch_strokes_from_mask(
    mask: numpy.ndarray,
    *,
    axis: str,
    spacing_px: int,
    min_run_px: int,
) -> list[tuple[_Point, ...]]:
    if axis not in {'horizontal', 'vertical'}:
        raise ValueError(f'Unsupported hatch axis {axis!r}.')

    height, width = mask.shape[:2]
    strokes: list[tuple[_Point, ...]] = []
    positions = range(max(0, spacing_px // 2), height if axis == 'horizontal' else width, max(1, spacing_px))
    for position in positions:
        line = mask[position, :] if axis == 'horizontal' else mask[:, position]
        start_index: int | None = None
        for index, value in enumerate(line):
            if bool(value):
                if start_index is None:
                    start_index = index
                continue
            if start_index is None:
                continue
            if index - start_index >= min_run_px:
                if axis == 'horizontal':
                    strokes.append(((float(start_index), float(position)), (float(index - 1), float(position))))
                else:
                    strokes.append(((float(position), float(start_index)), (float(position), float(index - 1))))
            start_index = None
        if start_index is not None and len(line) - start_index >= min_run_px:
            if axis == 'horizontal':
                strokes.append(((float(start_index), float(position)), (float(len(line) - 1), float(position))))
            else:
                strokes.append(((float(position), float(start_index)), (float(position), float(len(line) - 1))))
    return strokes


def _vectorize_simple_outline_image(
    gray: numpy.ndarray,
    *,
    min_perimeter_px: float,
    contour_simplify_ratio: float,
    max_strokes: int,
) -> tuple[tuple[tuple[_Point, ...], ...], dict[str, Any]]:
    binary = _line_art_binary_from_gray(gray)
    strokes, contour_stats = _trace_binary_contours(
        binary,
        min_perimeter_px=min_perimeter_px,
        contour_simplify_ratio=contour_simplify_ratio,
        max_strokes=max_strokes,
        close_contours=True,
    )
    return strokes, {
        'mode': _IMAGE_ROUTE_SIMPLE_OUTLINE,
        'binary_nonzero_ratio': float(numpy.count_nonzero(binary)) / float(binary.size),
        **contour_stats,
    }


def _vectorize_complex_tonal_image(
    gray: numpy.ndarray,
    *,
    min_perimeter_px: float,
    contour_simplify_ratio: float,
    max_strokes: int,
) -> tuple[tuple[tuple[_Point, ...], ...], dict[str, Any]]:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
    darkness = 1.0 - (smoothed.astype(numpy.float32) / 255.0)

    height, width = gray.shape[:2]
    base_spacing = int(
        max(
            6,
            min(
                28,
                round(math.sqrt(max(1.0, (width * height) / float(max(1, max_strokes)))) * 0.85),
            ),
        )
    )
    primary_spacing = max(6, base_spacing)
    secondary_spacing = max(primary_spacing + 2, primary_spacing * 2)
    min_run_px = max(4, primary_spacing // 2)

    primary_mask = cv2.morphologyEx(
        (darkness >= 0.34).astype(numpy.uint8) * 255,
        cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=numpy.uint8),
    ) > 0
    secondary_mask = cv2.morphologyEx(
        (darkness >= 0.58).astype(numpy.uint8) * 255,
        cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=numpy.uint8),
    ) > 0

    primary_strokes = _hatch_strokes_from_mask(
        primary_mask,
        axis='horizontal',
        spacing_px=primary_spacing,
        min_run_px=min_run_px,
    )
    secondary_strokes = _hatch_strokes_from_mask(
        secondary_mask,
        axis='vertical',
        spacing_px=secondary_spacing,
        min_run_px=min_run_px,
    )

    edge_binary = cv2.dilate(cv2.Canny(smoothed, 70, 150), numpy.ones((3, 3), dtype=numpy.uint8), iterations=1)
    outline_budget = max(16, min(max_strokes // 4, max_strokes))
    try:
        outline_strokes, outline_stats = _trace_binary_contours(
            edge_binary,
            min_perimeter_px=max(8.0, min_perimeter_px * 0.5),
            contour_simplify_ratio=min(0.03, max(0.003, contour_simplify_ratio * 1.5)),
            max_strokes=outline_budget,
            close_contours=False,
        )
    except ValueError:
        outline_strokes = ()
        outline_stats = {
            'candidate_contours': 0,
            'kept_contours': 0,
            'stroke_count': 0,
        }

    primary_strokes.sort(key=_stroke_length, reverse=True)
    secondary_strokes.sort(key=_stroke_length, reverse=True)
    outline_list = list(outline_strokes)
    outline_list.sort(key=_stroke_length, reverse=True)

    secondary_budget = max(1, max_strokes // 5)
    primary_budget = max(1, max_strokes - outline_budget - secondary_budget)
    limited_primary = primary_strokes[:primary_budget]
    limited_secondary = secondary_strokes[:secondary_budget]
    limited_outline = outline_list[:outline_budget]
    combined = limited_primary + limited_secondary + limited_outline
    if not combined:
        raise ValueError('Complex tonal routing produced no drawable strokes.')

    return tuple(combined[:max_strokes]), {
        'mode': _IMAGE_ROUTE_COMPLEX_TONAL,
        'primary_hatch_count': int(len(limited_primary)),
        'secondary_hatch_count': int(len(limited_secondary)),
        'outline_overlay_count': int(len(limited_outline)),
        'primary_spacing_px': int(primary_spacing),
        'secondary_spacing_px': int(secondary_spacing),
        'min_run_px': int(min_run_px),
        'outline_overlay_stats': outline_stats,
    }


def vectorize_image_to_canonical_plan(
    image_bytes: bytes,
    *,
    theta_ref: float,
    frame: str = 'board',
    min_perimeter_px: float = 8.0,
    contour_simplify_ratio: float = 0.001,
    max_strokes: int = 4096,
) -> ImageVectorizationResult:
    from wall_climber.ingestion.image_curve_fitting import (
        vectorize_image_to_canonical_plan as _vectorize_image_to_canonical_plan,
    )

    return _vectorize_image_to_canonical_plan(
        image_bytes,
        theta_ref=theta_ref,
        frame=frame,
        min_perimeter_px=min_perimeter_px,
        contour_simplify_ratio=contour_simplify_ratio,
        max_strokes=max_strokes,
    )


def trace_line_art_image(
    image_bytes: bytes,
    *,
    min_perimeter_px: float = 8.0,
    contour_simplify_ratio: float = 0.001,
    max_strokes: int = 4096,
) -> tuple[tuple[tuple[_Point, ...], ...], tuple[int, int]]:
    from wall_climber.ingestion.image_curve_fitting import (
        trace_line_art_image as _trace_line_art_image,
    )

    return _trace_line_art_image(
        image_bytes,
        min_perimeter_px=min_perimeter_px,
        contour_simplify_ratio=contour_simplify_ratio,
        max_strokes=max_strokes,
    )


def default_placement(writable_bounds: dict[str, float]) -> VectorPlacement:
    width = writable_bounds['x_max'] - writable_bounds['x_min']
    height = writable_bounds['y_max'] - writable_bounds['y_min']
    return VectorPlacement(
        x=writable_bounds['x_min'] + (width * 0.22),
        y=writable_bounds['y_min'] + (height * 0.14),
        scale=1.0,
    )


def default_image_placement(
    writable_bounds: dict[str, float],
    *,
    safe_bounds: dict[str, float] | None = None,
    safety_scale_padding: float = 0.99,
) -> VectorPlacement:
    width = writable_bounds['x_max'] - writable_bounds['x_min']
    height = writable_bounds['y_max'] - writable_bounds['y_min']
    if width <= _EPS or height <= _EPS:
        raise ValueError('Writable board bounds are invalid.')

    if safe_bounds is None:
        return VectorPlacement(
            x=writable_bounds['x_min'] + (width * 0.5),
            y=writable_bounds['y_min'] + (height * 0.5),
            scale=1.0,
        )

    safe_width = safe_bounds['x_max'] - safe_bounds['x_min']
    safe_height = safe_bounds['y_max'] - safe_bounds['y_min']
    if safe_width <= _EPS or safe_height <= _EPS:
        raise ValueError('Safe board bounds are invalid.')

    fit_ratio = min(safe_width / width, safe_height / height)
    scale = min(1.0, max(0.1, fit_ratio * max(0.5, float(safety_scale_padding))))
    return VectorPlacement(
        x=safe_bounds['x_min'] + (safe_width * 0.5),
        y=safe_bounds['y_min'] + (safe_height * 0.5),
        scale=scale,
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
    placed_glyphs: list[TextGlyphOutline] = []
    offset = 0
    for glyph, stroke_count in zip(glyphs, stroke_counts):
        glyph_strokes = tuple(placed_strokes[offset:offset + stroke_count])
        offset += stroke_count
        glyph_strokes = _apply_text_vector_cleanup(
            glyph_strokes,
            glyph_source=glyph.source,
        )
        placed_glyphs.append(
            TextGlyphOutline(
                line_index=glyph.line_index,
                word_index=glyph.word_index,
                text=glyph.text,
                strokes=glyph_strokes,
                bbox=_strokes_bounds(glyph_strokes),
                advance=glyph.advance,
                source=glyph.source,
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


def _canonical_plan_draw_strokes(
    plan: CanonicalPathPlan,
    *,
    curve_tolerance_m: float = 0.01,
) -> tuple[tuple[_Point, ...], ...]:
    strokes = canonical_plan_to_draw_strokes(
        plan,
        curve_tolerance_m=curve_tolerance_m,
    )
    if not strokes:
        raise ValueError('Canonical plan did not contain drawable geometry.')
    return tuple(
        tuple((float(point[0]), float(point[1])) for point in stroke)
        for stroke in strokes
    )


def _transform_canonical_point(
    point: _Point,
    *,
    source_center_x: float,
    source_center_y: float,
    placement: VectorPlacement,
    final_scale: float,
) -> _Point:
    return (
        placement.x + (float(point[0]) - source_center_x) * final_scale,
        placement.y + (float(point[1]) - source_center_y) * final_scale,
    )


def _transform_canonical_command(
    command: object,
    *,
    source_center_x: float,
    source_center_y: float,
    placement: VectorPlacement,
    final_scale: float,
) -> object:
    if isinstance(command, (PenUp, PenDown)):
        return command
    if isinstance(command, TravelMove):
        return TravelMove(
            start=_transform_canonical_point(
                command.start,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            end=_transform_canonical_point(
                command.end,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
        )
    if isinstance(command, LineSegment):
        return LineSegment(
            start=_transform_canonical_point(
                command.start,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            end=_transform_canonical_point(
                command.end,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
        )
    if isinstance(command, ArcSegment):
        return ArcSegment(
            center=_transform_canonical_point(
                command.center,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            radius=float(command.radius) * final_scale,
            start_angle_rad=float(command.start_angle_rad),
            sweep_angle_rad=float(command.sweep_angle_rad),
        )
    if isinstance(command, QuadraticBezier):
        return QuadraticBezier(
            start=_transform_canonical_point(
                command.start,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            control=_transform_canonical_point(
                command.control,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            end=_transform_canonical_point(
                command.end,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
        )
    if isinstance(command, CubicBezier):
        return CubicBezier(
            start=_transform_canonical_point(
                command.start,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            control1=_transform_canonical_point(
                command.control1,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            control2=_transform_canonical_point(
                command.control2,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
            end=_transform_canonical_point(
                command.end,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            ),
        )
    raise ValueError(f'Unsupported canonical command {type(command)!r}.')


def place_canonical_plan_on_board(
    plan: CanonicalPathPlan,
    *,
    writable_bounds: dict[str, float],
    placement: VectorPlacement,
    fit_padding: float | None = None,
    fit_margin_m: float | None = None,
    curve_tolerance_m: float = 0.01,
) -> tuple[CanonicalPathPlan, PlacementResult]:
    if fit_padding is not None and fit_margin_m is not None:
        raise ValueError('Specify either fit_padding or fit_margin_m, not both.')
    if fit_padding is None and fit_margin_m is None:
        fit_padding = 0.9

    source_strokes = _canonical_plan_draw_strokes(
        plan,
        curve_tolerance_m=curve_tolerance_m,
    )
    source_bounds = _strokes_bounds(source_strokes)
    source_width = max(source_bounds.width, _EPS)
    source_height = max(source_bounds.height, _EPS)

    fit_bounds = (
        _shrink_bounds(writable_bounds, fit_margin_m)
        if fit_margin_m is not None
        else writable_bounds
    )
    fit_width = fit_bounds['x_max'] - fit_bounds['x_min']
    fit_height = fit_bounds['y_max'] - fit_bounds['y_min']
    if fit_width <= _EPS or fit_height <= _EPS:
        raise ValueError('Writable board bounds are invalid.')

    fit_scale = min(
        (fit_width * float(fit_padding)) / source_width,
        (fit_height * float(fit_padding)) / source_height,
    ) if fit_padding is not None else min(fit_width / source_width, fit_height / source_height)
    final_scale = fit_scale * placement.scale
    source_center_x = 0.5 * (source_bounds.x_min + source_bounds.x_max)
    source_center_y = 0.5 * (source_bounds.y_min + source_bounds.y_max)

    transformed_plan = CanonicalPathPlan(
        frame=plan.frame,
        theta_ref=plan.theta_ref,
        commands=tuple(
            _transform_canonical_command(
                command,
                source_center_x=source_center_x,
                source_center_y=source_center_y,
                placement=placement,
                final_scale=final_scale,
            )
            for command in plan.commands
        ),
    )
    placed_strokes = _canonical_plan_draw_strokes(
        transformed_plan,
        curve_tolerance_m=curve_tolerance_m,
    )
    placed_bounds = _strokes_bounds(placed_strokes)
    return (
        transformed_plan,
        PlacementResult(
            placement=placement,
            base_fit_scale=fit_scale,
            final_scale=final_scale,
            bounds=placed_bounds,
            outside_points=_count_outside_points(placed_strokes, writable_bounds),
        ),
    )


def cleanup_canonical_plan(
    plan: CanonicalPathPlan,
    *,
    simplify_tolerance_m: float,
    preserve_order: bool = False,
    curve_tolerance_m: float = 0.01,
) -> CanonicalPathPlan:
    draw_strokes = _canonical_plan_draw_strokes(
        plan,
        curve_tolerance_m=curve_tolerance_m,
    )
    cleaned_strokes = cleanup_draw_strokes(
        draw_strokes,
        simplify_tolerance_m=simplify_tolerance_m,
        preserve_order=preserve_order,
    )
    return draw_strokes_to_canonical_plan(
        cleaned_strokes,
        theta_ref=plan.theta_ref,
        frame=plan.frame,
    )


def canonical_plan_from_pen_strokes(
    pen_strokes: tuple[tuple[_Point, ...], ...],
    *,
    theta_ref: float,
    pen_offset_x_m: float,
    pen_offset_y_m: float,
) -> CanonicalPathPlan:
    # Legacy-only compatibility helper for old stroke-first call sites.
    return pen_strokes_to_canonical_plan(
        pen_strokes,
        theta_ref=theta_ref,
        pen_offset_x_m=pen_offset_x_m,
        pen_offset_y_m=pen_offset_y_m,
    )


def draw_segments_from_pen_strokes(
    pen_strokes: tuple[tuple[_Point, ...], ...],
    *,
    theta_ref: float,
    pen_offset_x_m: float,
    pen_offset_y_m: float,
) -> tuple[DrawPathSegment, ...]:
    canonical_plan = canonical_plan_from_pen_strokes(
        pen_strokes,
        theta_ref=theta_ref,
        pen_offset_x_m=pen_offset_x_m,
        pen_offset_y_m=pen_offset_y_m,
    )
    return tuple(
        DrawPathSegment(draw=sampled.draw, points=sampled.points)
        for sampled in canonical_plan_to_sampled_paths(canonical_plan)
        if len(sampled.points) >= 2
    )


def draw_plan_to_dict(
    segments: tuple[DrawPathSegment, ...],
    *,
    theta_ref: float,
) -> dict[str, Any]:
    if not segments:
        return {
            'frame': 'board',
            'theta_ref': float(theta_ref),
            'segments': [],
        }
    commands = []
    pen_is_down = False
    for segment in segments:
        if segment.draw and not pen_is_down:
            commands.append(PenDown())
            pen_is_down = True
        if not segment.draw and pen_is_down:
            commands.append(PenUp())
            pen_is_down = False
        if segment.draw:
            for index in range(1, len(segment.points)):
                start = segment.points[index - 1]
                end = segment.points[index]
                if _distance(start, end) <= _EPS:
                    continue
                commands.append(LineSegment(start=start, end=end))
        else:
            for index in range(1, len(segment.points)):
                start = segment.points[index - 1]
                end = segment.points[index]
                if _distance(start, end) <= _EPS:
                    continue
                commands.append(TravelMove(start=start, end=end))
    if pen_is_down:
        commands.append(PenUp())
    if not commands:
        return {
            'frame': 'board',
            'theta_ref': float(theta_ref),
            'segments': [],
        }
    canonical_plan = CanonicalPathPlan(
        frame='board',
        theta_ref=float(theta_ref),
        commands=tuple(commands),
    )
    return canonical_plan_to_segment_payload(canonical_plan)


def strokes_to_draw_plan(strokes: tuple[tuple[_Point, ...], ...]) -> dict[str, Any]:
    canonical_plan = canonical_plan_from_pen_strokes(
        strokes,
        theta_ref=0.0,
        pen_offset_x_m=0.0,
        pen_offset_y_m=0.0,
    )
    return canonical_plan_to_legacy_strokes(canonical_plan)


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
