from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Any

import cv2  # type: ignore
import numpy

from wall_climber.canonical_adapters import (
    SamplingPolicy,
    canonical_command_to_debug_dict,
    canonical_plan_to_draw_strokes,
)
from wall_climber.canonical_optimizer import _fit_arc_from_line_chain
from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalCommand,
    CanonicalPathPlan,
    CubicBezier,
    LineSegment,
    PenDown,
    PenUp,
    Point2D,
    QuadraticBezier,
    TravelMove,
)
from wall_climber.vector_pipeline import (
    ImageVectorizationResult,
    VectorPlacement,
    _IMAGE_ROUTE_COLORED_ILLUSTRATION,
    _IMAGE_ROUTE_COMPLEX_TONAL,
    _IMAGE_ROUTE_SIMPLE_OUTLINE,
    _decode_image_mats,
    _distance,
    _hatch_strokes_from_mask,
    _image_routing_metrics,
    _is_sparse_line_art_metrics,
    _line_art_binary_from_gray,
    _route_from_metrics,
    _sanitize_stroke,
    _stroke_length,
    _strokes_bounds,
    _trace_binary_contours,
    _transform_canonical_command,
    draw_strokes_to_canonical_plan,
    stroke_stats,
)


_EPS = 1.0e-9
_MIN_SPAN_POINTS = 4
_MAX_RECURSION_DEPTH = 8
_MAX_IMAGE_PROCESSING_DIM_PX = 1800


@dataclass(frozen=True)
class _FitAttempt:
    command: CanonicalCommand | None
    error_px: float
    worst_index: int
    candidate_kind: str
    accepted: bool


def _point_line_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    if _distance(start, end) <= _EPS:
        return _distance(point, start)
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    t = ((px - x1) * dx + (py - y1) * dy) / max(_EPS, (dx * dx + dy * dy))
    t = max(0.0, min(1.0, t))
    proj = (x1 + t * dx, y1 + t * dy)
    return _distance(point, proj)


def _heading(a: Point2D, b: Point2D) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _angle_delta_deg(first: float, second: float) -> float:
    delta = math.atan2(math.sin(second - first), math.cos(second - first))
    return abs(math.degrees(delta))


def _signed_turn_deg(prev_point: Point2D, point: Point2D, next_point: Point2D) -> float:
    first = (point[0] - prev_point[0], point[1] - prev_point[1])
    second = (next_point[0] - point[0], next_point[1] - point[1])
    if math.hypot(*first) <= _EPS or math.hypot(*second) <= _EPS:
        return 0.0
    cross = (first[0] * second[1]) - (first[1] * second[0])
    dot = (first[0] * second[0]) + (first[1] * second[1])
    return math.degrees(math.atan2(cross, dot))


def _primitive_start(command: CanonicalCommand) -> Point2D:
    if isinstance(command, TravelMove):
        return command.start
    if isinstance(command, LineSegment):
        return command.start
    if isinstance(command, QuadraticBezier):
        return command.start
    if isinstance(command, CubicBezier):
        return command.start
    if isinstance(command, ArcSegment):
        return (
            command.center[0] + command.radius * math.cos(command.start_angle_rad),
            command.center[1] + command.radius * math.sin(command.start_angle_rad),
        )
    raise ValueError(f'Unsupported primitive {type(command)!r}.')


def _primitive_end(command: CanonicalCommand) -> Point2D:
    if isinstance(command, TravelMove):
        return command.end
    if isinstance(command, LineSegment):
        return command.end
    if isinstance(command, QuadraticBezier):
        return command.end
    if isinstance(command, CubicBezier):
        return command.end
    if isinstance(command, ArcSegment):
        angle = command.start_angle_rad + command.sweep_angle_rad
        return (
            command.center[0] + command.radius * math.cos(angle),
            command.center[1] + command.radius * math.sin(angle),
        )
    raise ValueError(f'Unsupported primitive {type(command)!r}.')


def _command_descriptor(command: CanonicalCommand | None, *, precision: float = 1.0e-6) -> tuple[Any, ...] | None:
    if command is None or isinstance(command, (PenUp, PenDown)):
        return None

    scale = max(precision, 1.0e-9)

    def pack_point(point: Point2D) -> tuple[int, int]:
        return (
            int(round(point[0] / scale)),
            int(round(point[1] / scale)),
        )

    def pack_scalar(value: float) -> int:
        return int(round(float(value) / scale))

    if isinstance(command, TravelMove):
        return ('T', pack_point(command.start), pack_point(command.end))
    if isinstance(command, LineSegment):
        return ('L', pack_point(command.start), pack_point(command.end))
    if isinstance(command, ArcSegment):
        return (
            'A',
            pack_point(command.center),
            pack_scalar(command.radius),
            pack_scalar(command.start_angle_rad),
            pack_scalar(command.sweep_angle_rad),
        )
    if isinstance(command, QuadraticBezier):
        return ('Q', pack_point(command.start), pack_point(command.control), pack_point(command.end))
    if isinstance(command, CubicBezier):
        return (
            'C',
            pack_point(command.start),
            pack_point(command.control1),
            pack_point(command.control2),
            pack_point(command.end),
        )
    raise ValueError(f'Unsupported command type {type(command)!r}.')


def _dedupe_points(points: list[Point2D]) -> tuple[Point2D, ...]:
    sanitized = _sanitize_stroke(points)
    if sanitized is None:
        return ()
    return tuple((float(point[0]), float(point[1])) for point in sanitized)


def _downscale_processing_mats(
    color: numpy.ndarray,
    gray: numpy.ndarray,
    *,
    max_dim_px: int = _MAX_IMAGE_PROCESSING_DIM_PX,
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    height = int(gray.shape[0])
    width = int(gray.shape[1])
    longest_edge = max(width, height)
    limit = max(256, int(max_dim_px))
    if longest_edge <= limit:
        return color, gray, 1.0

    scale = float(limit) / float(longest_edge)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_color = cv2.resize(color, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    resized_gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
    return resized_color, resized_gray, scale


def _resample_polyline(points: tuple[Point2D, ...], *, spacing_px: float) -> tuple[Point2D, ...]:
    if len(points) < 2:
        return points
    spacing = max(0.5, float(spacing_px))
    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + _distance(points[index - 1], points[index]))
    total = cumulative[-1]
    if total <= spacing + _EPS:
        return points

    targets = [0.0]
    cursor = spacing
    while cursor < total:
        targets.append(cursor)
        cursor += spacing
    targets.append(total)

    sampled: list[Point2D] = []
    segment_index = 1
    for target in targets:
        while segment_index < len(cumulative) and cumulative[segment_index] < target - _EPS:
            segment_index += 1
        if segment_index >= len(cumulative):
            sampled.append(points[-1])
            continue
        previous_distance = cumulative[segment_index - 1]
        next_distance = cumulative[segment_index]
        start = points[segment_index - 1]
        end = points[segment_index]
        if next_distance - previous_distance <= _EPS:
            sampled.append(end)
            continue
        ratio = (target - previous_distance) / (next_distance - previous_distance)
        sampled.append(
            (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        )
    return _dedupe_points(sampled)


def _smooth_polyline(points: tuple[Point2D, ...], *, window: int = 7) -> tuple[Point2D, ...]:
    if len(points) < max(3, window):
        return points
    radius = max(1, int(window) // 2)
    is_closed = _distance(points[0], points[-1]) <= 1.5
    source = list(points[:-1] if is_closed else points)
    if len(source) < 3:
        return points

    smoothed: list[Point2D] = []
    count = len(source)
    for index in range(count):
        sample_x = 0.0
        sample_y = 0.0
        sample_count = 0
        for offset in range(-radius, radius + 1):
            if is_closed:
                point = source[(index + offset) % count]
            else:
                clamped_index = max(0, min(count - 1, index + offset))
                point = source[clamped_index]
            sample_x += point[0]
            sample_y += point[1]
            sample_count += 1
        smoothed.append((sample_x / sample_count, sample_y / sample_count))

    if is_closed and smoothed:
        smoothed.append(smoothed[0])
    return _dedupe_points(smoothed)


def _open_contour(points: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    if len(points) < 4 or _distance(points[0], points[-1]) > 1.5:
        return points

    closed = list(points[:-1])
    if len(closed) < 3:
        return points

    corner_strengths: list[tuple[float, int]] = []
    for index in range(len(closed)):
        prev_point = closed[index - 1]
        point = closed[index]
        next_point = closed[(index + 1) % len(closed)]
        corner_strengths.append((abs(_signed_turn_deg(prev_point, point, next_point)), index))
    corner_strengths.sort(reverse=True)
    cut_index = corner_strengths[0][1] if corner_strengths else 0
    rotated = closed[cut_index:] + closed[:cut_index]
    rotated.append(rotated[0])
    return _dedupe_points(rotated)


def _split_polyline_into_spans(
    points: tuple[Point2D, ...],
    *,
    corner_threshold_deg: float,
    heading_continuity_deg: float,
) -> tuple[tuple[Point2D, ...], ...]:
    if len(points) <= _MIN_SPAN_POINTS:
        return (points,)

    segment_lengths = [
        _distance(points[index - 1], points[index])
        for index in range(1, len(points))
    ]
    median_segment = float(numpy.median(segment_lengths)) if segment_lengths else 0.0
    split_indices = [0]
    previous_sign = 0
    previous_turn = 0.0

    for index in range(1, len(points) - 1):
        prev_point = points[index - 1]
        point = points[index]
        next_point = points[index + 1]
        turn = _signed_turn_deg(prev_point, point, next_point)
        abs_turn = abs(turn)
        sign = 1 if turn > 0.0 else -1 if turn < 0.0 else 0
        before = _distance(prev_point, point)
        after = _distance(point, next_point)
        gap_split = median_segment > _EPS and max(before, after) > (median_segment * 2.5)
        should_split = False
        if abs_turn >= corner_threshold_deg:
            should_split = True
        elif (
            previous_sign
            and sign
            and previous_sign != sign
            and abs_turn >= heading_continuity_deg
            and abs(previous_turn) >= heading_continuity_deg
        ):
            should_split = True
        elif gap_split:
            should_split = True
        if should_split and index - split_indices[-1] >= 2:
            split_indices.append(index)
        if sign:
            previous_sign = sign
        if abs_turn > 0.0:
            previous_turn = turn

    if split_indices[-1] != len(points) - 1:
        split_indices.append(len(points) - 1)

    spans: list[tuple[Point2D, ...]] = []
    for start_index, end_index in zip(split_indices[:-1], split_indices[1:]):
        span = _dedupe_points(list(points[start_index:end_index + 1]))
        if len(span) >= 2:
            spans.append(span)
    return tuple(spans) if spans else (points,)


_SKELETON_NEIGHBORS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),            (1, 0),
    (-1, 1),  (0, 1),   (1, 1),
)


def _thin_binary(binary: numpy.ndarray) -> numpy.ndarray:
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return binary


def _skeleton_component_prune(binary: numpy.ndarray, *, min_area_px: int) -> numpy.ndarray:
    mask = (binary > 0).astype(numpy.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pruned = numpy.zeros_like(binary)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        pruned[labels == label] = 255
    return pruned


def _pixel_neighbors(
    pixel: tuple[int, int],
    pixels: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    x, y = pixel
    return tuple(
        (x + dx, y + dy)
        for dx, dy in _SKELETON_NEIGHBORS
        if (x + dx, y + dy) in pixels
    )


def _edge_key(first: tuple[int, int], second: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (first, second) if first <= second else (second, first)


def _choose_skeleton_neighbor(
    previous: tuple[int, int],
    current: tuple[int, int],
    candidates: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    if len(candidates) == 1:
        return candidates[0]
    prev_vec = (current[0] - previous[0], current[1] - previous[1])
    prev_len = math.hypot(prev_vec[0], prev_vec[1])
    best = candidates[0]
    best_score = -float('inf')
    for candidate in candidates:
        next_vec = (candidate[0] - current[0], candidate[1] - current[1])
        next_len = math.hypot(next_vec[0], next_vec[1])
        if prev_len <= _EPS or next_len <= _EPS:
            score = 0.0
        else:
            score = ((prev_vec[0] * next_vec[0]) + (prev_vec[1] * next_vec[1])) / (prev_len * next_len)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _trace_centerline_strokes(
    skeleton_binary: numpy.ndarray,
    *,
    min_length_px: float,
) -> tuple[tuple[Point2D, ...], ...]:
    ys, xs = numpy.nonzero(skeleton_binary > 0)
    pixels = {(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())}
    if not pixels:
        return ()

    neighbor_map = {
        pixel: _pixel_neighbors(pixel, pixels)
        for pixel in pixels
    }
    nodes = {
        pixel for pixel, neighbors in neighbor_map.items()
        if len(neighbors) != 2
    }
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    traced: list[tuple[Point2D, ...]] = []

    def follow(start: tuple[int, int], next_pixel: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        path = [start, next_pixel]
        previous = start
        current = next_pixel
        visited_edges.add(_edge_key(previous, current))
        while True:
            if current in nodes and current != start:
                break
            candidates = tuple(
                neighbor for neighbor in neighbor_map[current]
                if neighbor != previous and _edge_key(current, neighbor) not in visited_edges
            )
            if not candidates:
                break
            chosen = _choose_skeleton_neighbor(previous, current, candidates)
            path.append(chosen)
            visited_edges.add(_edge_key(current, chosen))
            previous, current = current, chosen
            if current == start:
                break
        return tuple(path)

    def record_path(path: tuple[tuple[int, int], ...]) -> None:
        if len(path) < 2:
            return
        stroke = _dedupe_points([(float(point[0]), float(point[1])) for point in path])
        if len(stroke) < 2 or _stroke_length(stroke) < min_length_px:
            return
        traced.append(stroke)

    for node in sorted(nodes):
        for neighbor in neighbor_map[node]:
            key = _edge_key(node, neighbor)
            if key in visited_edges:
                continue
            record_path(follow(node, neighbor))

    for pixel in sorted(pixels):
        for neighbor in neighbor_map[pixel]:
            key = _edge_key(pixel, neighbor)
            if key in visited_edges:
                continue
            record_path(follow(pixel, neighbor))

    traced.sort(key=_stroke_length, reverse=True)
    return tuple(traced)


def _stroke_endpoint_heading(
    stroke: tuple[Point2D, ...],
    *,
    at_start: bool,
    sample_count: int = 4,
) -> float | None:
    if len(stroke) < 2:
        return None
    window = min(len(stroke) - 1, max(1, sample_count))
    if at_start:
        anchor = stroke[0]
        target = stroke[window]
        if _distance(anchor, target) <= _EPS:
            return None
        return _heading(anchor, target)
    anchor = stroke[-1]
    target = stroke[-1 - window]
    if _distance(anchor, target) <= _EPS:
        return None
    return _heading(target, anchor)


def _merge_centerline_strokes(
    strokes: tuple[tuple[Point2D, ...], ...],
    *,
    max_gap_px: float,
    max_heading_delta_deg: float,
) -> tuple[tuple[tuple[Point2D, ...], ...], dict[str, int]]:
    working = [tuple(stroke) for stroke in strokes if len(stroke) >= 2]
    merges = 0
    if len(working) < 2:
        return tuple(working), {
            'input_centerlines': int(len(working)),
            'merged_centerlines': int(merges),
            'output_centerlines': int(len(working)),
        }

    def oriented(stroke: tuple[Point2D, ...], reverse: bool) -> tuple[Point2D, ...]:
        return tuple(reversed(stroke)) if reverse else stroke

    while len(working) > 1:
        best_pair: tuple[int, int, bool, bool] | None = None
        best_score = float('inf')
        for first_index in range(len(working)):
            first = working[first_index]
            for second_index in range(first_index + 1, len(working)):
                second = working[second_index]
                for reverse_first in (False, True):
                    left = oriented(first, reverse_first)
                    left_heading = _stroke_endpoint_heading(left, at_start=False)
                    if left_heading is None:
                        continue
                    for reverse_second in (False, True):
                        right = oriented(second, reverse_second)
                        right_heading = _stroke_endpoint_heading(right, at_start=True)
                        if right_heading is None:
                            continue
                        gap = _distance(left[-1], right[0])
                        if gap > max_gap_px:
                            continue
                        heading_delta = _angle_delta_deg(left_heading, right_heading)
                        if heading_delta > max_heading_delta_deg:
                            continue
                        score = gap + (heading_delta * 0.05)
                        if score < best_score:
                            best_score = score
                            best_pair = (first_index, second_index, reverse_first, reverse_second)
        if best_pair is None:
            break

        first_index, second_index, reverse_first, reverse_second = best_pair
        left = oriented(working[first_index], reverse_first)
        right = oriented(working[second_index], reverse_second)
        merged = _dedupe_points(list(left) + list(right))
        if len(merged) < 2:
            break
        working[first_index] = merged
        del working[second_index]
        merges += 1

    working.sort(key=_stroke_length, reverse=True)
    return tuple(working), {
        'input_centerlines': int(len(strokes)),
        'merged_centerlines': int(merges),
        'output_centerlines': int(len(working)),
    }


def _prune_centerline_spurs(
    strokes: tuple[tuple[Point2D, ...], ...],
    *,
    max_spur_length_px: float,
    attach_distance_px: float,
) -> tuple[tuple[tuple[Point2D, ...], ...], dict[str, int]]:
    if len(strokes) <= 1:
        return strokes, {
            'input_centerlines': int(len(strokes)),
            'pruned_spurs': 0,
            'output_centerlines': int(len(strokes)),
        }

    kept: list[tuple[Point2D, ...]] = []
    pruned = 0
    for index, stroke in enumerate(strokes):
        length = _stroke_length(stroke)
        if length > max_spur_length_px:
            kept.append(stroke)
            continue
        endpoints = (stroke[0], stroke[-1])
        attached = False
        for other_index, other in enumerate(strokes):
            if index == other_index:
                continue
            for endpoint in endpoints:
                if any(_distance(endpoint, point) <= attach_distance_px for point in other):
                    attached = True
                    break
            if attached:
                break
        if attached:
            pruned += 1
            continue
        kept.append(stroke)

    kept_tuple = tuple(kept)
    return kept_tuple, {
        'input_centerlines': int(len(strokes)),
        'pruned_spurs': int(pruned),
        'output_centerlines': int(len(kept_tuple)),
    }


def _line_fit(points: tuple[Point2D, ...], fit_tol_px: float) -> _FitAttempt:
    if len(points) < 2:
        return _FitAttempt(None, 0.0, 0, 'line', False)
    start = points[0]
    end = points[-1]
    max_error = 0.0
    worst_index = 0
    for index, point in enumerate(points[1:-1], start=1):
        error = _point_line_distance(point, start, end)
        if error > max_error:
            max_error = error
            worst_index = index
    accepted = max_error <= fit_tol_px + _EPS
    return _FitAttempt(
        LineSegment(start=start, end=end) if accepted else None,
        max_error,
        worst_index,
        'line',
        accepted,
    )


def _arc_fit(points: tuple[Point2D, ...], fit_tol_px: float, *, min_span_length_px: float) -> _FitAttempt:
    if len(points) < 5:
        return _FitAttempt(None, float('inf'), max(1, len(points) // 2), 'arc', False)
    polyline_length = sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))
    if polyline_length < min_span_length_px:
        return _FitAttempt(None, polyline_length, max(1, len(points) // 2), 'arc', False)
    lines = tuple(
        LineSegment(start=points[index - 1], end=points[index])
        for index in range(1, len(points))
    )
    fitted = _fit_arc_from_line_chain(lines, tolerance_m=fit_tol_px)
    if fitted is None or fitted.radius <= 3.0:
        return _FitAttempt(None, float('inf'), max(1, len(points) // 2), 'arc', False)
    center = fitted.center
    radial_errors = [abs(_distance(point, center) - fitted.radius) for point in points]
    max_error = max(radial_errors) if radial_errors else 0.0
    worst_index = radial_errors.index(max_error) if radial_errors else 0
    accepted = max_error <= fit_tol_px + _EPS
    return _FitAttempt(fitted if accepted else None, max_error, worst_index, 'arc', accepted)


def _chord_length_parameters(points: tuple[Point2D, ...]) -> tuple[float, ...]:
    if len(points) <= 1:
        return (0.0,)
    distances = [0.0]
    for index in range(1, len(points)):
        distances.append(distances[-1] + _distance(points[index - 1], points[index]))
    total = distances[-1]
    if total <= _EPS:
        return tuple(index / max(1, len(points) - 1) for index in range(len(points)))
    return tuple(distance / total for distance in distances)


def _evaluate_quadratic(segment: QuadraticBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        (omt * omt * segment.start[0]) + (2.0 * omt * t * segment.control[0]) + (t * t * segment.end[0]),
        (omt * omt * segment.start[1]) + (2.0 * omt * t * segment.control[1]) + (t * t * segment.end[1]),
    )


def _evaluate_cubic(segment: CubicBezier, t: float) -> Point2D:
    omt = 1.0 - t
    return (
        (omt ** 3 * segment.start[0])
        + (3.0 * omt * omt * t * segment.control1[0])
        + (3.0 * omt * t * t * segment.control2[0])
        + (t ** 3 * segment.end[0]),
        (omt ** 3 * segment.start[1])
        + (3.0 * omt * omt * t * segment.control1[1])
        + (3.0 * omt * t * t * segment.control2[1])
        + (t ** 3 * segment.end[1]),
    )


def _solve_cubic_controls(points: tuple[Point2D, ...]) -> CubicBezier | None:
    if len(points) < 4:
        return None
    start = points[0]
    end = points[-1]
    parameters = _chord_length_parameters(points)
    rows: list[list[float]] = []
    rhs_x: list[float] = []
    rhs_y: list[float] = []
    for point, t in zip(points[1:-1], parameters[1:-1]):
        omt = 1.0 - t
        b0 = omt ** 3
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t ** 3
        rows.append([b1, b2])
        rhs_x.append(point[0] - ((b0 * start[0]) + (b3 * end[0])))
        rhs_y.append(point[1] - ((b0 * start[1]) + (b3 * end[1])))
    if len(rows) < 2:
        return None
    matrix = numpy.asarray(rows, dtype=numpy.float64)
    if numpy.linalg.matrix_rank(matrix) < 2:
        return None
    sol_x, *_ = numpy.linalg.lstsq(matrix, numpy.asarray(rhs_x, dtype=numpy.float64), rcond=None)
    sol_y, *_ = numpy.linalg.lstsq(matrix, numpy.asarray(rhs_y, dtype=numpy.float64), rcond=None)
    control1 = (float(sol_x[0]), float(sol_y[0]))
    control2 = (float(sol_x[1]), float(sol_y[1]))
    if not all(math.isfinite(value) for value in (*control1, *control2)):
        return None
    return CubicBezier(start=start, control1=control1, control2=control2, end=end)


def _cubic_fit(points: tuple[Point2D, ...], fit_tol_px: float, *, min_span_length_px: float) -> _FitAttempt:
    if len(points) < 4:
        return _FitAttempt(None, float('inf'), max(1, len(points) // 2), 'cubic', False)
    polyline_length = sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))
    if polyline_length < min_span_length_px:
        return _FitAttempt(None, polyline_length, max(1, len(points) // 2), 'cubic', False)
    fitted = _solve_cubic_controls(points)
    if fitted is None:
        return _FitAttempt(None, float('inf'), max(1, len(points) // 2), 'cubic', False)
    parameters = _chord_length_parameters(points)
    errors = [_distance(_evaluate_cubic(fitted, t), point) for point, t in zip(points, parameters)]
    max_error = max(errors) if errors else 0.0
    worst_index = errors.index(max_error) if errors else 0
    accepted = max_error <= fit_tol_px + _EPS
    return _FitAttempt(fitted if accepted else None, max_error, worst_index, 'cubic', accepted)


def _reduce_cubic_to_quadratic(cubic: CubicBezier, fit_tol_px: float, points: tuple[Point2D, ...]) -> _FitAttempt:
    q1 = (
        (3.0 * cubic.control1[0] - cubic.start[0]) * 0.5,
        (3.0 * cubic.control1[1] - cubic.start[1]) * 0.5,
    )
    q2 = (
        (3.0 * cubic.control2[0] - cubic.end[0]) * 0.5,
        (3.0 * cubic.control2[1] - cubic.end[1]) * 0.5,
    )
    if _distance(q1, q2) > max(fit_tol_px * 0.6, 0.75):
        return _FitAttempt(None, float('inf'), max(1, len(points) // 2), 'quadratic', False)
    quadratic = QuadraticBezier(
        start=cubic.start,
        control=((q1[0] + q2[0]) * 0.5, (q1[1] + q2[1]) * 0.5),
        end=cubic.end,
    )
    parameters = _chord_length_parameters(points)
    errors = [_distance(_evaluate_quadratic(quadratic, t), point) for point, t in zip(points, parameters)]
    max_error = max(errors) if errors else 0.0
    worst_index = errors.index(max_error) if errors else 0
    accepted = max_error <= (fit_tol_px * 0.6) + _EPS
    return _FitAttempt(quadratic if accepted else None, max_error, worst_index, 'quadratic', accepted)


def _fallback_line_chain(
    points: tuple[Point2D, ...],
    *,
    span_id: int,
    error_px: float,
    reason: str,
) -> tuple[tuple[CanonicalCommand, dict[str, Any]], ...]:
    if len(points) < 2:
        return ()
    simplified = _dedupe_points(points if len(points) <= 4 else _dedupe_points(_rdp_local(points, epsilon=max(0.75, error_px * 0.5))))
    if len(simplified) < 2:
        simplified = points
    output: list[tuple[CanonicalCommand, dict[str, Any]]] = []
    for index in range(1, len(simplified)):
        command = LineSegment(start=simplified[index - 1], end=simplified[index])
        output.append(
            (
                command,
                {
                    'fit_source': 'fallback_line_chain',
                    'fit_error_px': float(error_px),
                    'span_id': int(span_id),
                    'fallback_reason': str(reason),
                },
            )
        )
    return tuple(output)


def _rdp_local(points: tuple[Point2D, ...], epsilon: float) -> list[Point2D]:
    if len(points) <= 2:
        return list(points)
    start = points[0]
    end = points[-1]
    max_error = -1.0
    split_index = -1
    for index in range(1, len(points) - 1):
        error = _point_line_distance(points[index], start, end)
        if error > max_error:
            max_error = error
            split_index = index
    if max_error <= epsilon or split_index < 0:
        return [start, end]
    left = _rdp_local(points[:split_index + 1], epsilon)
    right = _rdp_local(points[split_index:], epsilon)
    return left[:-1] + right


def _fit_span_recursive(
    points: tuple[Point2D, ...],
    *,
    fit_tol_px: float,
    min_span_length_px: float,
    span_id: int,
    counters: Counter,
    worst_spans: list[dict[str, Any]],
    depth: int = 0,
) -> tuple[tuple[CanonicalCommand, dict[str, Any]], ...]:
    line_attempt = _line_fit(points, fit_tol_px)
    if line_attempt.accepted and line_attempt.command is not None:
        counters['accepted_lines'] += 1
        worst_spans.append(
            {
                'span_id': int(span_id),
                'chosen_primitive': 'line',
                'fit_error_px': float(line_attempt.error_px),
                'fallback_reason': None,
            }
        )
        return ((line_attempt.command, {'fit_source': 'span_line_fit', 'fit_error_px': float(line_attempt.error_px), 'span_id': int(span_id)}),)
    counters['rejected_lines'] += 1

    arc_attempt = _arc_fit(points, fit_tol_px, min_span_length_px=min_span_length_px)
    if arc_attempt.accepted and arc_attempt.command is not None:
        counters['accepted_arcs'] += 1
        worst_spans.append(
            {
                'span_id': int(span_id),
                'chosen_primitive': 'arc',
                'fit_error_px': float(arc_attempt.error_px),
                'fallback_reason': None,
            }
        )
        return ((arc_attempt.command, {'fit_source': 'span_arc_fit', 'fit_error_px': float(arc_attempt.error_px), 'span_id': int(span_id)}),)
    counters['rejected_arcs'] += 1

    cubic_attempt = _cubic_fit(points, fit_tol_px, min_span_length_px=min_span_length_px)
    if cubic_attempt.accepted and isinstance(cubic_attempt.command, CubicBezier):
        quadratic_attempt = _reduce_cubic_to_quadratic(cubic_attempt.command, fit_tol_px, points)
        if quadratic_attempt.accepted and quadratic_attempt.command is not None:
            counters['accepted_quadratics'] += 1
            worst_spans.append(
                {
                    'span_id': int(span_id),
                    'chosen_primitive': 'quadratic',
                    'fit_error_px': float(quadratic_attempt.error_px),
                    'fallback_reason': None,
                }
            )
            return (
                (
                    quadratic_attempt.command,
                    {
                        'fit_source': 'span_quadratic_reduction',
                        'fit_error_px': float(quadratic_attempt.error_px),
                        'span_id': int(span_id),
                    },
                ),
            )
        counters['accepted_cubics'] += 1
        worst_spans.append(
            {
                'span_id': int(span_id),
                'chosen_primitive': 'cubic',
                'fit_error_px': float(cubic_attempt.error_px),
                'fallback_reason': None,
            }
        )
        return ((cubic_attempt.command, {'fit_source': 'span_cubic_fit', 'fit_error_px': float(cubic_attempt.error_px), 'span_id': int(span_id)}),)
    counters['rejected_cubics'] += 1
    counters['rejected_quadratics'] += 1

    if depth >= _MAX_RECURSION_DEPTH or len(points) <= _MIN_SPAN_POINTS:
        counters['fallback_line_spans'] += 1
        worst_spans.append(
            {
                'span_id': int(span_id),
                'chosen_primitive': 'fallback_line',
                'fit_error_px': float(line_attempt.error_px),
                'fallback_reason': 'terminal_split',
            }
        )
        return _fallback_line_chain(points, span_id=span_id, error_px=line_attempt.error_px, reason='terminal_split')

    split_index = cubic_attempt.worst_index if math.isfinite(cubic_attempt.error_px) else line_attempt.worst_index
    split_index = max(2, min(len(points) - 3, int(split_index)))
    if split_index <= 1 or split_index >= len(points) - 2:
        split_index = len(points) // 2
    if split_index <= 1 or split_index >= len(points) - 2:
        counters['fallback_line_spans'] += 1
        worst_spans.append(
            {
                'span_id': int(span_id),
                'chosen_primitive': 'fallback_line',
                'fit_error_px': float(line_attempt.error_px),
                'fallback_reason': 'unsplittable',
            }
        )
        return _fallback_line_chain(points, span_id=span_id, error_px=line_attempt.error_px, reason='unsplittable')

    left = _fit_span_recursive(
        points[:split_index + 1],
        fit_tol_px=fit_tol_px,
        min_span_length_px=min_span_length_px,
        span_id=span_id,
        counters=counters,
        worst_spans=worst_spans,
        depth=depth + 1,
    )
    right = _fit_span_recursive(
        points[split_index:],
        fit_tol_px=fit_tol_px,
        min_span_length_px=min_span_length_px,
        span_id=span_id,
        counters=counters,
        worst_spans=worst_spans,
        depth=depth + 1,
    )
    return left + right


def _build_plan_from_units(
    units: tuple[tuple[tuple[CanonicalCommand, dict[str, Any]], ...], ...],
    *,
    theta_ref: float,
    frame: str,
) -> tuple[CanonicalPathPlan, tuple[dict[str, Any] | None, ...]]:
    commands: list[CanonicalCommand] = []
    metadata: list[dict[str, Any] | None] = []
    last_point: Point2D | None = None

    for unit in units:
        if not unit:
            continue
        first_command = unit[0][0]
        start = _primitive_start(first_command)
        if last_point is not None and _distance(last_point, start) > _EPS:
            commands.append(PenUp())
            metadata.append(None)
            commands.append(TravelMove(start=last_point, end=start))
            metadata.append(None)
        commands.append(PenDown())
        metadata.append(None)
        for command, command_meta in unit:
            commands.append(command)
            metadata.append(dict(command_meta))
        last_point = _primitive_end(unit[-1][0])

    if commands:
        commands.append(PenUp())
        metadata.append(None)
    return CanonicalPathPlan(frame=frame, theta_ref=theta_ref, commands=tuple(commands)), tuple(metadata)


def _fit_outline_units(
    binary: numpy.ndarray,
    *,
    fit_tol_px: float,
    min_perimeter_px: float,
    max_strokes: int,
    close_contours: bool,
    contour_simplify_ratio: float,
) -> tuple[tuple[tuple[CanonicalCommand, dict[str, Any]], ...], tuple[tuple[Point2D, ...], ...], dict[str, Any]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(
        contours,
        key=lambda contour: float(cv2.arcLength(contour, close_contours)),
        reverse=True,
    )
    candidate_contours = len(contours)
    raw_contours: list[tuple[Point2D, ...]] = []
    units: list[tuple[tuple[CanonicalCommand, dict[str, Any]], ...]] = []
    counters: Counter = Counter()
    worst_spans: list[dict[str, Any]] = []
    raw_contour_count = 0
    span_id = 0
    heading_continuity_deg = 18.0
    corner_threshold_deg = 28.0
    min_span_length_px = 12.0
    smoothing_window = 7

    for contour in contours:
        perimeter = float(cv2.arcLength(contour, close_contours))
        if perimeter < min_perimeter_px:
            continue
        points = [(float(point[0][0]), float(point[0][1])) for point in contour]
        if close_contours and points and _distance(points[0], points[-1]) > _EPS:
            points.append(points[0])
        sanitized = _dedupe_points(points)
        if len(sanitized) < 2:
            continue
        opened = _open_contour(sanitized)
        raw_contours.append(opened)
        raw_contour_count += 1
        resampled = _resample_polyline(opened, spacing_px=max(0.8, fit_tol_px * 0.75))
        resampled = _smooth_polyline(resampled, window=smoothing_window)
        if len(resampled) < 2:
            continue
        spans = _split_polyline_into_spans(
            resampled,
            corner_threshold_deg=corner_threshold_deg,
            heading_continuity_deg=heading_continuity_deg,
        )
        for span in spans:
            span_id += 1
            fitted = _fit_span_recursive(
                span,
                fit_tol_px=fit_tol_px,
                min_span_length_px=min_span_length_px,
                span_id=span_id,
                counters=counters,
                worst_spans=worst_spans,
            )
            if fitted:
                units.append(fitted)
            if len(units) >= max_strokes:
                break
        if len(units) >= max_strokes:
            break

    if not units:
        raise ValueError('No drawable contours were extracted from image.')

    contour_stats = {
        'candidate_contours': int(candidate_contours),
        'kept_contours': int(raw_contour_count),
        'stroke_count': int(len(units)),
    }
    fit_summary = {
        'accepted_lines': int(counters.get('accepted_lines', 0)),
        'accepted_arcs': int(counters.get('accepted_arcs', 0)),
        'accepted_quadratics': int(counters.get('accepted_quadratics', 0)),
        'accepted_cubics': int(counters.get('accepted_cubics', 0)),
        'fallback_line_spans': int(counters.get('fallback_line_spans', 0)),
        'rejected_candidates': {
            'line': int(counters.get('rejected_lines', 0)),
            'arc': int(counters.get('rejected_arcs', 0)),
            'quadratic': int(counters.get('rejected_quadratics', 0)),
            'cubic': int(counters.get('rejected_cubics', 0)),
        },
    }
    curve_fit_debug = {
        'raw_contour_count': int(raw_contour_count),
        'span_count': int(span_id),
        'fit_summary': fit_summary,
        'fit_tolerances': {
            'fit_tol_px': float(fit_tol_px),
            'corner_threshold_deg': float(corner_threshold_deg),
            'heading_continuity_deg': float(heading_continuity_deg),
            'min_span_length_px': float(min_span_length_px),
            'smoothing_window': int(smoothing_window),
            'contour_simplify_ratio_legacy': float(contour_simplify_ratio),
        },
        'worst_spans': sorted(worst_spans, key=lambda item: float(item.get('fit_error_px') or 0.0), reverse=True)[:8],
    }
    return tuple(units[:max_strokes]), tuple(raw_contours), {
        **contour_stats,
        'curve_fit_summary': fit_summary,
        'curve_fit_debug': curve_fit_debug,
    }


def _fit_centerline_units(
    binary: numpy.ndarray,
    *,
    fit_tol_px: float,
    min_perimeter_px: float,
    max_strokes: int,
) -> tuple[tuple[tuple[CanonicalCommand, dict[str, Any]], ...], tuple[tuple[Point2D, ...], ...], dict[str, Any]]:
    min_area_px = max(6, int(round(min_perimeter_px * 0.35)))
    prepared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, numpy.ones((3, 3), dtype=numpy.uint8))
    prepared = _skeleton_component_prune(prepared, min_area_px=min_area_px)
    skeleton = _thin_binary(prepared)
    raw_centerlines = _trace_centerline_strokes(
        skeleton,
        min_length_px=max(6.0, min_perimeter_px * 0.6),
    )
    if not raw_centerlines:
        raise ValueError('No drawable centerlines were extracted from image.')
    raw_centerlines, spur_stats = _prune_centerline_spurs(
        raw_centerlines,
        max_spur_length_px=max(7.0, min_perimeter_px * 0.8),
        attach_distance_px=max(1.5, fit_tol_px * 1.2),
    )
    merged_centerlines, merge_stats = _merge_centerline_strokes(
        raw_centerlines,
        max_gap_px=max(3.0, fit_tol_px * 2.2),
        max_heading_delta_deg=38.0,
    )
    if merged_centerlines:
        raw_centerlines = merged_centerlines
    merge_stats['pruned_spurs'] = int(spur_stats.get('pruned_spurs', 0))

    units: list[tuple[tuple[CanonicalCommand, dict[str, Any]], ...]] = []
    counters: Counter = Counter()
    worst_spans: list[dict[str, Any]] = []
    span_id = 0
    raw_contours: list[tuple[Point2D, ...]] = []
    heading_continuity_deg = 16.0
    corner_threshold_deg = 30.0
    min_span_length_px = 10.0
    smoothing_window = 5

    for stroke in raw_centerlines:
        raw_contours.append(stroke)
        simplified = _dedupe_points(
            _rdp_local(
                stroke,
                epsilon=max(0.45, fit_tol_px * 0.35),
            )
        )
        if len(simplified) >= 2:
            stroke = simplified
        resampled = _resample_polyline(stroke, spacing_px=max(0.7, fit_tol_px * 0.55))
        resampled = _smooth_polyline(resampled, window=smoothing_window)
        if len(resampled) < 2:
            continue
        spans = _split_polyline_into_spans(
            resampled,
            corner_threshold_deg=corner_threshold_deg,
            heading_continuity_deg=heading_continuity_deg,
        )
        for span in spans:
            span_id += 1
            fitted = _fit_span_recursive(
                span,
                fit_tol_px=fit_tol_px,
                min_span_length_px=min_span_length_px,
                span_id=span_id,
                counters=counters,
                worst_spans=worst_spans,
            )
            if fitted:
                units.append(fitted)
            if len(units) >= max_strokes:
                break
        if len(units) >= max_strokes:
            break

    if not units:
        raise ValueError('Centerline tracing produced no drawable units.')

    fit_summary = {
        'accepted_lines': int(counters.get('accepted_lines', 0)),
        'accepted_arcs': int(counters.get('accepted_arcs', 0)),
        'accepted_quadratics': int(counters.get('accepted_quadratics', 0)),
        'accepted_cubics': int(counters.get('accepted_cubics', 0)),
        'fallback_line_spans': int(counters.get('fallback_line_spans', 0)),
        'rejected_candidates': {
            'line': int(counters.get('rejected_lines', 0)),
            'arc': int(counters.get('rejected_arcs', 0)),
            'quadratic': int(counters.get('rejected_quadratics', 0)),
            'cubic': int(counters.get('rejected_cubics', 0)),
        },
    }
    curve_fit_debug = {
        'trace_mode': 'centerline',
        'raw_contour_count': int(len(raw_contours)),
        'span_count': int(span_id),
        'fit_summary': fit_summary,
        'fit_tolerances': {
            'fit_tol_px': float(fit_tol_px),
            'corner_threshold_deg': float(corner_threshold_deg),
            'heading_continuity_deg': float(heading_continuity_deg),
            'min_span_length_px': float(min_span_length_px),
            'smoothing_window': int(smoothing_window),
            'component_min_area_px': int(min_area_px),
        },
        'merge_stats': merge_stats,
        'worst_spans': sorted(
            worst_spans,
            key=lambda item: float(item.get('fit_error_px') or 0.0),
            reverse=True,
        )[:8],
    }
    return tuple(units[:max_strokes]), tuple(raw_contours), {
        'candidate_contours': int(len(raw_centerlines)),
        'kept_contours': int(len(raw_contours)),
        'stroke_count': int(len(units)),
        'skeleton_pixel_count': int(numpy.count_nonzero(skeleton)),
        'centerline_count': int(len(raw_centerlines)),
        'merge_stats': merge_stats,
        'curve_fit_summary': fit_summary,
        'curve_fit_debug': curve_fit_debug,
    }


def _fit_quality_score(stats: dict[str, Any]) -> float:
    fit_summary = dict(stats.get('curve_fit_summary') or {})
    merge_stats = dict(stats.get('merge_stats') or {})
    accepted_lines = int(fit_summary.get('accepted_lines', 0))
    accepted_arcs = int(fit_summary.get('accepted_arcs', 0))
    accepted_quadratics = int(fit_summary.get('accepted_quadratics', 0))
    accepted_cubics = int(fit_summary.get('accepted_cubics', 0))
    fallback_line_spans = int(fit_summary.get('fallback_line_spans', 0))
    stroke_count = int(stats.get('stroke_count', 0))
    merged_centerlines = int(merge_stats.get('merged_centerlines', 0))
    return (
        (accepted_arcs * 5.0)
        + (accepted_quadratics * 4.0)
        + (accepted_cubics * 4.0)
        + (merged_centerlines * 2.0)
        - (accepted_lines * 0.25)
        - (fallback_line_spans * 1.5)
        - (stroke_count * 0.15)
    )


def _filled_feature_rescue_gate(
    route_decision: Any,
    metrics: Any,
) -> bool:
    return (
        route_decision.route == _IMAGE_ROUTE_SIMPLE_OUTLINE
        and metrics.background_whiteness >= 0.80
        and metrics.entropy <= 0.35
        and metrics.contour_count <= 320
    )


def _extract_filled_feature_mask(
    gray: numpy.ndarray,
) -> tuple[numpy.ndarray, dict[str, Any]]:
    height, width = gray.shape[:2]
    rescue_mask = numpy.zeros_like(gray)
    _, dark_feature_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_feature_core = cv2.erode(dark_feature_mask, core_kernel, iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dark_feature_core > 0).astype(numpy.uint8),
        connectivity=8,
    )

    candidate_feature_count = 0
    rescued_feature_count = 0
    rescued_feature_area_ratio_sum = 0.0
    image_area = float(max(1, height * width))

    for label in range(1, component_count):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area <= 0:
            continue
        candidate_feature_count += 1
        if x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height:
            continue
        area_ratio_image = float(area) / image_area
        if not (0.00004 <= area_ratio_image <= 0.08):
            continue
        bbox_area = max(1, w * h)
        bbox_aspect_ratio = float(w) / max(float(h), 1.0)
        if not (0.15 <= bbox_aspect_ratio <= 8.0):
            continue
        fill_ratio = float(area) / float(bbox_area)
        if fill_ratio < 0.15:
            continue

        component_mask = ((labels == label).astype(numpy.uint8) * 255)
        expanded_component = cv2.dilate(component_mask, core_kernel, iterations=2)
        component_region = cv2.bitwise_and(expanded_component, dark_feature_mask)
        contours, _ = cv2.findContours(component_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= _EPS:
            continue
        compactness = (perimeter * perimeter) / max(_EPS, (4.0 * math.pi * float(area)))
        if compactness > 18.0:
            continue

        rescue_mask[component_region > 0] = 255
        rescued_feature_count += 1
        rescued_feature_area_ratio_sum += area_ratio_image

    return rescue_mask, {
        'rescued_feature_count': int(rescued_feature_count),
        'rescued_feature_area_ratio_sum': float(rescued_feature_area_ratio_sum),
        'candidate_feature_count': int(candidate_feature_count),
    }


def _merge_outline_stats(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    if not secondary:
        return dict(primary)

    merged = dict(primary)
    merged['candidate_contours'] = int(primary.get('candidate_contours', 0)) + int(secondary.get('candidate_contours', 0))
    merged['kept_contours'] = int(primary.get('kept_contours', 0)) + int(secondary.get('kept_contours', 0))
    merged['stroke_count'] = int(primary.get('stroke_count', 0)) + int(secondary.get('stroke_count', 0))

    primary_fit_summary = dict(primary.get('curve_fit_summary') or {})
    secondary_fit_summary = dict(secondary.get('curve_fit_summary') or {})
    merged_fit_summary = {
        'accepted_lines': int(primary_fit_summary.get('accepted_lines', 0)) + int(secondary_fit_summary.get('accepted_lines', 0)),
        'accepted_arcs': int(primary_fit_summary.get('accepted_arcs', 0)) + int(secondary_fit_summary.get('accepted_arcs', 0)),
        'accepted_quadratics': int(primary_fit_summary.get('accepted_quadratics', 0)) + int(secondary_fit_summary.get('accepted_quadratics', 0)),
        'accepted_cubics': int(primary_fit_summary.get('accepted_cubics', 0)) + int(secondary_fit_summary.get('accepted_cubics', 0)),
        'fallback_line_spans': int(primary_fit_summary.get('fallback_line_spans', 0)) + int(secondary_fit_summary.get('fallback_line_spans', 0)),
        'rejected_candidates': {
            'line': int((primary_fit_summary.get('rejected_candidates') or {}).get('line', 0))
            + int((secondary_fit_summary.get('rejected_candidates') or {}).get('line', 0)),
            'arc': int((primary_fit_summary.get('rejected_candidates') or {}).get('arc', 0))
            + int((secondary_fit_summary.get('rejected_candidates') or {}).get('arc', 0)),
            'quadratic': int((primary_fit_summary.get('rejected_candidates') or {}).get('quadratic', 0))
            + int((secondary_fit_summary.get('rejected_candidates') or {}).get('quadratic', 0)),
            'cubic': int((primary_fit_summary.get('rejected_candidates') or {}).get('cubic', 0))
            + int((secondary_fit_summary.get('rejected_candidates') or {}).get('cubic', 0)),
        },
    }
    merged['curve_fit_summary'] = merged_fit_summary

    primary_debug = dict(primary.get('curve_fit_debug') or {})
    secondary_debug = dict(secondary.get('curve_fit_debug') or {})
    merged_debug = dict(primary_debug)
    merged_debug['raw_contour_count'] = int(primary_debug.get('raw_contour_count', 0)) + int(secondary_debug.get('raw_contour_count', 0))
    merged_debug['span_count'] = int(primary_debug.get('span_count', 0)) + int(secondary_debug.get('span_count', 0))
    merged_debug['fit_summary'] = merged_fit_summary
    primary_worst = list(primary_debug.get('worst_spans') or [])
    secondary_worst = list(secondary_debug.get('worst_spans') or [])
    merged_debug['worst_spans'] = sorted(
        primary_worst + secondary_worst,
        key=lambda item: float(item.get('fit_error_px') or 0.0),
        reverse=True,
    )[:8]
    merged['curve_fit_debug'] = merged_debug
    return merged


def _quantize_color_image(color: numpy.ndarray, *, cluster_count: int = 5) -> numpy.ndarray:
    pixels = color.reshape((-1, 3)).astype(numpy.float32)
    if pixels.size == 0:
        return color
    k = max(3, min(int(cluster_count), len(pixels)))
    if k < 2:
        return color
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        16,
        1.0,
    )
    _compactness, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(numpy.uint8)
    quantized = centers[labels.flatten()].reshape(color.shape)
    return quantized


def _foreground_mask_from_color(color: numpy.ndarray, gray: numpy.ndarray) -> numpy.ndarray:
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(numpy.float32)
    border = numpy.concatenate(
        [
            lab[0, :, :],
            lab[-1, :, :],
            lab[:, 0, :],
            lab[:, -1, :],
        ],
        axis=0,
    )
    background = numpy.median(border, axis=0)
    distance_map = numpy.linalg.norm(lab - background[None, None, :], axis=2)
    edge_map = cv2.Canny(gray, 48, 128)
    mask = (
        (distance_map >= 18.0)
        | (gray <= 242)
        | (edge_map > 0)
    ).astype(numpy.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, numpy.ones((5, 5), dtype=numpy.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((3, 3), dtype=numpy.uint8))
    mask = _skeleton_component_prune(mask, min_area_px=max(20, int(mask.size * 0.00035)))
    return mask


def _vectorize_simple_outline_image(
    gray: numpy.ndarray,
    *,
    min_perimeter_px: float,
    contour_simplify_ratio: float,
    max_strokes: int,
    theta_ref: float,
    frame: str,
) -> ImageVectorizationResult:
    binary = _line_art_binary_from_gray(gray)
    metrics = _image_routing_metrics(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray)
    route_decision = _route_from_metrics(metrics)
    sparse_line_art = _is_sparse_line_art_metrics(metrics)
    image_diagonal_px = math.hypot(gray.shape[1], gray.shape[0])
    fit_tol_px = max(1.25, 0.0018 * image_diagonal_px)
    trace_mode = 'centerline_outline'
    rescue_mask = numpy.zeros_like(binary)
    rescue_stats = {
        'rescued_feature_count': 0,
        'rescued_feature_area_ratio_sum': 0.0,
        'candidate_feature_count': 0,
    }
    filled_feature_rescue_applied = False
    filled_feature_rescue_eligible = _filled_feature_rescue_gate(route_decision, metrics)
    if filled_feature_rescue_eligible:
        rescue_mask, rescue_stats = _extract_filled_feature_mask(gray)
        filled_feature_rescue_applied = int(rescue_stats['rescued_feature_count']) >= 1
    detailed_sparse_line_art = (
        sparse_line_art
        and image_diagonal_px >= 1200.0
        and metrics.contour_count >= 180
    )
    if detailed_sparse_line_art or filled_feature_rescue_applied:
        trace_mode = _IMAGE_ROUTE_SIMPLE_OUTLINE
        rescue_units: tuple[tuple[tuple[CanonicalCommand, dict[str, Any]], ...], ...] = ()
        rescue_raw_contours: tuple[tuple[Point2D, ...], ...] = ()
        rescue_outline_stats: dict[str, Any] = {}
        outline_binary = binary
        if filled_feature_rescue_applied:
            outline_binary = cv2.bitwise_and(binary, cv2.bitwise_not(rescue_mask))
            rescue_units, rescue_raw_contours, rescue_outline_stats = _fit_outline_units(
                rescue_mask,
                fit_tol_px=fit_tol_px,
                min_perimeter_px=max(4.0, min_perimeter_px * 0.45),
                max_strokes=max(2, min(32, max_strokes // 8)),
                close_contours=True,
                contour_simplify_ratio=contour_simplify_ratio,
            )
        outline_units, outline_raw_contours, outline_stats = _fit_outline_units(
            outline_binary,
            fit_tol_px=fit_tol_px,
            min_perimeter_px=min_perimeter_px,
            max_strokes=max(1, max_strokes - len(rescue_units)),
            close_contours=True,
            contour_simplify_ratio=contour_simplify_ratio,
        )
        units = outline_units + rescue_units
        raw_contours = outline_raw_contours + rescue_raw_contours
        if filled_feature_rescue_applied:
            outline_stats = _merge_outline_stats(outline_stats, rescue_outline_stats)
        outline_stats['rescued_feature_unit_count'] = int(len(rescue_units))
    else:
        try:
            units, raw_contours, outline_stats = _fit_centerline_units(
                binary,
                fit_tol_px=fit_tol_px,
                min_perimeter_px=min_perimeter_px,
                max_strokes=max_strokes,
            )
            fit_summary = dict(outline_stats.get('curve_fit_summary') or {})
            merge_stats = dict(outline_stats.get('merge_stats') or {})
            accepted_curve_count = (
                int(fit_summary.get('accepted_arcs', 0))
                + int(fit_summary.get('accepted_quadratics', 0))
                + int(fit_summary.get('accepted_cubics', 0))
            )
            accepted_line_count = int(fit_summary.get('accepted_lines', 0))
            output_centerlines = int(merge_stats.get('output_centerlines', accepted_line_count))
            centerline_useful = (
                accepted_curve_count > 0
                or int(merge_stats.get('merged_centerlines', 0)) > 0
                or (
                    accepted_line_count <= 4
                    and output_centerlines <= 4
                )
            )
            if not centerline_useful:
                raise ValueError('centerline tracing did not simplify line-art geometry')
            outline_units, outline_raw_contours, outline_candidate_stats = _fit_outline_units(
                binary,
                fit_tol_px=fit_tol_px,
                min_perimeter_px=min_perimeter_px,
                max_strokes=max_strokes,
                close_contours=True,
                contour_simplify_ratio=contour_simplify_ratio,
            )
            keep_centerline_for_sparse_line_art = (
                sparse_line_art
                and (
                    int(merge_stats.get('merged_centerlines', 0)) > 0
                    or output_centerlines > 8
                    or accepted_line_count > 48
                )
            )
            if (
                not keep_centerline_for_sparse_line_art
                and _fit_quality_score(outline_candidate_stats) > (_fit_quality_score(outline_stats) + 0.5)
            ):
                trace_mode = _IMAGE_ROUTE_SIMPLE_OUTLINE
                units = outline_units
                raw_contours = outline_raw_contours
                outline_stats = outline_candidate_stats
        except ValueError:
            trace_mode = _IMAGE_ROUTE_SIMPLE_OUTLINE
            units, raw_contours, outline_stats = _fit_outline_units(
                binary,
                fit_tol_px=fit_tol_px,
                min_perimeter_px=min_perimeter_px,
                max_strokes=max_strokes,
                close_contours=True,
                contour_simplify_ratio=contour_simplify_ratio,
            )
    plan, command_metadata = _build_plan_from_units(units, theta_ref=theta_ref, frame=frame)
    draw_strokes = canonical_plan_to_draw_strokes(plan, sampling_policy=SamplingPolicy(curve_tolerance_m=max(0.5, fit_tol_px * 0.35)))
    branch_details = {
        'mode': trace_mode,
        'binary_nonzero_ratio': float(numpy.count_nonzero(binary)) / float(binary.size),
        'sparse_line_art': bool(sparse_line_art),
        'detailed_sparse_line_art': bool(detailed_sparse_line_art),
        'filled_feature_rescue_applied': bool(filled_feature_rescue_applied),
        'rescued_feature_count': int(rescue_stats['rescued_feature_count']),
        'rescued_feature_unit_count': int(outline_stats.get('rescued_feature_unit_count', 0)),
        'rescued_feature_area_ratio_sum': float(rescue_stats['rescued_feature_area_ratio_sum']),
        **outline_stats,
        'stroke_stats': stroke_stats(draw_strokes),
    }
    curve_fit_debug = dict(outline_stats.get('curve_fit_debug') or {})
    curve_fit_debug['filled_feature_rescue_applied'] = bool(filled_feature_rescue_applied)
    curve_fit_debug['rescued_feature_count'] = int(rescue_stats['rescued_feature_count'])
    curve_fit_debug['rescued_feature_unit_count'] = int(outline_stats.get('rescued_feature_unit_count', 0))
    curve_fit_debug['rescued_feature_area_ratio_sum'] = float(rescue_stats['rescued_feature_area_ratio_sum'])
    return ImageVectorizationResult(
        plan=plan,
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
        route_decision=route_decision,
        branch_details=branch_details,
        command_metadata=command_metadata,
        curve_fit_debug=curve_fit_debug,
        raw_contours=raw_contours,
    )


def _vectorize_colored_illustration_image(
    color: numpy.ndarray,
    gray: numpy.ndarray,
    *,
    min_perimeter_px: float,
    contour_simplify_ratio: float,
    max_strokes: int,
    theta_ref: float,
    frame: str,
) -> ImageVectorizationResult:
    quantized = _quantize_color_image(color, cluster_count=5)
    quantized_gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    foreground_mask = _foreground_mask_from_color(quantized, quantized_gray)
    line_binary = _line_art_binary_from_gray(quantized_gray)
    edge_binary = cv2.Canny(quantized_gray, 42, 128)
    combined_binary = cv2.max(line_binary, edge_binary)
    combined_binary = cv2.bitwise_and(combined_binary, foreground_mask)
    combined_binary = cv2.morphologyEx(
        combined_binary,
        cv2.MORPH_CLOSE,
        numpy.ones((3, 3), dtype=numpy.uint8),
    )

    fit_tol_px = max(1.25, 0.0018 * math.hypot(gray.shape[1], gray.shape[0]))
    outline_budget = max(24, min(max_strokes - max(8, max_strokes // 5), max_strokes))
    outline_units, raw_contours, outline_stats = _fit_outline_units(
        combined_binary,
        fit_tol_px=fit_tol_px,
        min_perimeter_px=max(6.0, min_perimeter_px * 0.8),
        max_strokes=outline_budget,
        close_contours=False,
        contour_simplify_ratio=min(0.02, max(0.0015, contour_simplify_ratio * 1.2)),
    )

    darkness = 1.0 - (quantized_gray.astype(numpy.float32) / 255.0)
    hatch_mask = (
        (darkness >= 0.52)
        & (foreground_mask > 0)
    )
    hatch_spacing = max(
        5,
        min(
            18,
            int(round(math.sqrt(max(1.0, (gray.shape[0] * gray.shape[1]) / float(max(1, max_strokes)))) * 0.62)),
        ),
    )
    hatch_strokes = _hatch_strokes_from_mask(
        hatch_mask,
        axis='horizontal',
        spacing_px=hatch_spacing,
        min_run_px=max(4, hatch_spacing // 2),
    )
    hatch_budget = max(0, max_strokes - len(outline_units))
    hatch_strokes.sort(key=_stroke_length, reverse=True)

    units: list[tuple[tuple[CanonicalCommand, dict[str, Any]], ...]] = list(outline_units)
    hatch_added = 0
    for stroke in hatch_strokes[:hatch_budget]:
        if len(stroke) < 2:
            continue
        primitives: list[tuple[CanonicalCommand, dict[str, Any]]] = []
        for index in range(1, len(stroke)):
            primitives.append(
                (
                    LineSegment(start=stroke[index - 1], end=stroke[index]),
                    {
                        'fit_source': 'illustration_hatch',
                        'fit_error_px': 0.0,
                        'span_id': None,
                    },
                )
            )
        if primitives:
            units.append(tuple(primitives))
            hatch_added += 1

    if not units:
        raise ValueError('Colored illustration routing produced no drawable geometry.')

    plan, command_metadata = _build_plan_from_units(tuple(units[:max_strokes]), theta_ref=theta_ref, frame=frame)
    draw_strokes = canonical_plan_to_draw_strokes(plan)
    curve_fit_debug = dict(outline_stats.get('curve_fit_debug') or {})
    fit_summary = dict(curve_fit_debug.get('fit_summary') or {})
    fit_summary['illustration_hatch_units'] = int(hatch_added)
    curve_fit_debug['fit_summary'] = fit_summary
    branch_details = {
        'mode': _IMAGE_ROUTE_COLORED_ILLUSTRATION,
        'foreground_ratio': float(numpy.count_nonzero(foreground_mask)) / float(max(1, foreground_mask.size)),
        'outline_overlay_count': int(len(outline_units)),
        'illustration_hatch_count': int(hatch_added),
        'hatch_spacing_px': int(hatch_spacing),
        'curve_fit_summary': fit_summary,
        'outline_overlay_stats': {
            key: value for key, value in outline_stats.items() if key != 'curve_fit_debug'
        },
        'stroke_stats': stroke_stats(draw_strokes),
    }
    return ImageVectorizationResult(
        plan=plan,
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
        route_decision=_route_from_metrics(_image_routing_metrics(color, gray)),
        branch_details=branch_details,
        command_metadata=command_metadata,
        curve_fit_debug=curve_fit_debug,
        raw_contours=raw_contours,
    )


def _vectorize_complex_tonal_image(
    gray: numpy.ndarray,
    *,
    min_perimeter_px: float,
    contour_simplify_ratio: float,
    max_strokes: int,
    theta_ref: float,
    frame: str,
) -> ImageVectorizationResult:
    metrics = _image_routing_metrics(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray)
    sparse_line_art = _is_sparse_line_art_metrics(metrics)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
    darkness = 1.0 - (smoothed.astype(numpy.float32) / 255.0)

    height, width = gray.shape[:2]
    base_spacing = int(
        max(
            5,
            min(
                24,
                round(math.sqrt(max(1.0, (width * height) / float(max(1, max_strokes)))) * 0.78),
            ),
        )
    )
    band_a_spacing = max(5, int(round(base_spacing * 1.45)))
    band_b_spacing = max(4, int(round(base_spacing * 1.05)))
    band_c_spacing = max(5, int(round(base_spacing * 1.20)))
    min_run_px = max(4, base_spacing // 2)

    light_mask = cv2.morphologyEx(
        (darkness >= 0.26).astype(numpy.uint8) * 255,
        cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=numpy.uint8),
    ) > 0
    mid_mask = cv2.morphologyEx(
        (darkness >= 0.48).astype(numpy.uint8) * 255,
        cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=numpy.uint8),
    ) > 0
    dark_mask = cv2.morphologyEx(
        (darkness >= 0.68).astype(numpy.uint8) * 255,
        cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=numpy.uint8),
    ) > 0

    if sparse_line_art:
        light_hatch = []
        mid_hatch = []
        dark_hatch = []
        edge_binary = _line_art_binary_from_gray(gray)
    else:
        light_hatch = _hatch_strokes_from_mask(light_mask, axis='horizontal', spacing_px=band_a_spacing, min_run_px=min_run_px)
        mid_hatch = _hatch_strokes_from_mask(mid_mask, axis='horizontal', spacing_px=band_b_spacing, min_run_px=max(3, min_run_px - 1))
        dark_hatch = _hatch_strokes_from_mask(dark_mask, axis='vertical', spacing_px=band_c_spacing, min_run_px=min_run_px)
        edge_binary = cv2.dilate(cv2.Canny(smoothed, 60, 150), numpy.ones((3, 3), dtype=numpy.uint8), iterations=1)
    outline_budget = max(24, min(max_strokes // 3, max_strokes))
    fit_tol_px = max(1.25, 0.0018 * math.hypot(width, height))
    outline_units, raw_contours, outline_stats = _fit_outline_units(
        edge_binary,
        fit_tol_px=fit_tol_px,
        min_perimeter_px=max(8.0, min_perimeter_px * 0.5),
        max_strokes=outline_budget,
        close_contours=bool(sparse_line_art),
        contour_simplify_ratio=min(0.03, max(0.003, contour_simplify_ratio * 1.5)),
    )

    light_hatch.sort(key=_stroke_length, reverse=True)
    mid_hatch.sort(key=_stroke_length, reverse=True)
    dark_hatch.sort(key=_stroke_length, reverse=True)

    dark_budget = max(1, max_strokes // 6)
    mid_budget = max(1, max_strokes // 4)
    light_budget = max(1, max_strokes - outline_budget - mid_budget - dark_budget)

    units: list[tuple[tuple[CanonicalCommand, dict[str, Any]], ...]] = []

    def append_hatch_units(strokes: list[tuple[Point2D, ...]], budget: int, fit_source: str) -> int:
        added = 0
        for stroke in strokes[:budget]:
            if len(stroke) < 2:
                continue
            primitives: list[tuple[CanonicalCommand, dict[str, Any]]] = []
            for index in range(1, len(stroke)):
                primitives.append(
                    (
                        LineSegment(start=stroke[index - 1], end=stroke[index]),
                        {
                            'fit_source': fit_source,
                            'fit_error_px': 0.0,
                            'span_id': None,
                        },
                    )
                )
            if primitives:
                units.append(tuple(primitives))
                added += 1
        return added

    light_added = append_hatch_units(light_hatch, light_budget, 'hatch_light')
    mid_added = append_hatch_units(mid_hatch, mid_budget, 'hatch_mid')
    dark_added = append_hatch_units(dark_hatch, dark_budget, 'hatch_dark')
    units.extend(outline_units)

    if not units:
        raise ValueError('Complex tonal routing produced no drawable geometry.')

    plan, command_metadata = _build_plan_from_units(tuple(units[:max_strokes]), theta_ref=theta_ref, frame=frame)
    draw_strokes = canonical_plan_to_draw_strokes(plan)
    curve_fit_debug = dict(outline_stats.get('curve_fit_debug') or {})
    fit_summary = dict(curve_fit_debug.get('fit_summary') or {})
    fit_summary['tonal_hatch_units'] = int(light_added + mid_added + dark_added)
    curve_fit_debug['fit_summary'] = fit_summary
    branch_details = {
        'mode': _IMAGE_ROUTE_COMPLEX_TONAL,
        'sparse_line_art': bool(sparse_line_art),
        'light_hatch_count': int(light_added),
        'mid_hatch_count': int(mid_added),
        'dark_hatch_count': int(dark_added),
        'outline_overlay_count': int(len(outline_units)),
        'band_spacing_px': {
            'light': int(band_a_spacing),
            'mid': int(band_b_spacing),
            'dark': int(band_c_spacing),
        },
        'min_run_px': int(min_run_px),
        'curve_fit_summary': fit_summary,
        'outline_overlay_stats': {
            key: value for key, value in outline_stats.items() if key != 'curve_fit_debug'
        },
        'stroke_stats': stroke_stats(draw_strokes),
    }
    return ImageVectorizationResult(
        plan=plan,
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
        route_decision=_route_from_metrics(metrics),
        branch_details=branch_details,
        command_metadata=command_metadata,
        curve_fit_debug=curve_fit_debug,
        raw_contours=raw_contours,
    )


def vectorize_image_to_canonical_plan(
    image_bytes: bytes,
    *,
    theta_ref: float,
    frame: str = 'board',
    min_perimeter_px: float = 8.0,
    contour_simplify_ratio: float = 0.001,
    max_strokes: int = 4096,
) -> ImageVectorizationResult:
    original_color, original_gray = _decode_image_mats(image_bytes)
    original_image_size = (
        int(original_gray.shape[1]),
        int(original_gray.shape[0]),
    )
    color, gray, processing_scale = _downscale_processing_mats(
        original_color,
        original_gray,
    )
    processing_image_size = (
        int(gray.shape[1]),
        int(gray.shape[0]),
    )
    route_decision = _route_from_metrics(_image_routing_metrics(color, gray))
    if route_decision.route == _IMAGE_ROUTE_SIMPLE_OUTLINE:
        result = _vectorize_simple_outline_image(
            gray,
            min_perimeter_px=min_perimeter_px,
            contour_simplify_ratio=contour_simplify_ratio,
            max_strokes=max_strokes,
            theta_ref=theta_ref,
            frame=frame,
        )
    elif route_decision.route == _IMAGE_ROUTE_COLORED_ILLUSTRATION:
        result = _vectorize_colored_illustration_image(
            color,
            gray,
            min_perimeter_px=min_perimeter_px,
            contour_simplify_ratio=contour_simplify_ratio,
            max_strokes=max_strokes,
            theta_ref=theta_ref,
            frame=frame,
        )
    else:
        result = _vectorize_complex_tonal_image(
            gray,
            min_perimeter_px=min_perimeter_px,
            contour_simplify_ratio=contour_simplify_ratio,
            max_strokes=max_strokes,
            theta_ref=theta_ref,
            frame=frame,
        )
    branch_details = dict(result.branch_details)
    branch_details['original_image_size'] = {
        'width_px': int(original_image_size[0]),
        'height_px': int(original_image_size[1]),
    }
    branch_details['processing_image_size'] = {
        'width_px': int(processing_image_size[0]),
        'height_px': int(processing_image_size[1]),
    }
    branch_details['processing_scale'] = float(processing_scale)
    curve_fit_debug = dict(result.curve_fit_debug or {})
    curve_fit_debug['original_image_size'] = {
        'width_px': int(original_image_size[0]),
        'height_px': int(original_image_size[1]),
    }
    curve_fit_debug['processing_image_size'] = {
        'width_px': int(processing_image_size[0]),
        'height_px': int(processing_image_size[1]),
    }
    curve_fit_debug['processing_scale'] = float(processing_scale)
    return ImageVectorizationResult(
        plan=result.plan,
        image_size=original_image_size,
        route_decision=route_decision,
        branch_details=branch_details,
        command_metadata=result.command_metadata,
        curve_fit_debug=curve_fit_debug,
        raw_contours=result.raw_contours,
    )


def trace_line_art_image(
    image_bytes: bytes,
    *,
    min_perimeter_px: float = 8.0,
    contour_simplify_ratio: float = 0.001,
    max_strokes: int = 4096,
) -> tuple[tuple[tuple[Point2D, ...], ...], tuple[int, int]]:
    result = vectorize_image_to_canonical_plan(
        image_bytes,
        theta_ref=0.0,
        min_perimeter_px=min_perimeter_px,
        contour_simplify_ratio=contour_simplify_ratio,
        max_strokes=max_strokes,
    )
    return canonical_plan_to_draw_strokes(result.plan), result.image_size


def map_curve_fit_command_metadata(
    source_plan: CanonicalPathPlan,
    source_metadata: tuple[dict[str, Any] | None, ...],
    target_plan: CanonicalPathPlan,
) -> tuple[dict[str, Any] | None, ...]:
    source_lookup: dict[tuple[Any, ...], list[dict[str, Any] | None]] = {}
    for command, metadata in zip(source_plan.commands, source_metadata):
        descriptor = _command_descriptor(command)
        if descriptor is None:
            continue
        source_lookup.setdefault(descriptor, []).append(dict(metadata) if metadata else None)

    mapped: list[dict[str, Any] | None] = []
    for command in target_plan.commands:
        descriptor = _command_descriptor(command)
        if descriptor is None:
            mapped.append(None)
            continue
        bucket = source_lookup.get(descriptor)
        if bucket:
            mapped.append(bucket.pop(0))
            continue
        if isinstance(command, ArcSegment):
            mapped.append({'fit_source': 'optimizer_arc_fit', 'fit_error_px': None, 'span_id': None})
        elif isinstance(command, (QuadraticBezier, CubicBezier)):
            mapped.append({'fit_source': 'optimizer_curve_merge', 'fit_error_px': None, 'span_id': None})
        else:
            mapped.append(None)
    return tuple(mapped)


def curve_fit_debug_to_board(
    *,
    source_plan: CanonicalPathPlan,
    placed_plan: CanonicalPathPlan,
    command_metadata: tuple[dict[str, Any] | None, ...],
    raw_contours: tuple[tuple[Point2D, ...], ...],
    curve_fit_debug: dict[str, Any] | None,
    placement: VectorPlacement,
    final_scale: float,
    source_type: str = 'image',
) -> dict[str, Any]:
    if curve_fit_debug is None:
        return {'available': False}

    source_draw_strokes = canonical_plan_to_draw_strokes(
        source_plan,
        sampling_policy=SamplingPolicy(curve_tolerance_m=0.5, label='curve-fit-debug-source'),
    )
    source_bounds = _strokes_bounds(
        tuple(
            tuple((float(point[0]), float(point[1])) for point in stroke)
            for stroke in source_draw_strokes
        )
    )
    source_center_x = 0.5 * (source_bounds.x_min + source_bounds.x_max)
    source_center_y = 0.5 * (source_bounds.y_min + source_bounds.y_max)

    raw_contours_board = [
        [
            [
                float(placement.x + (point[0] - source_center_x) * final_scale),
                float(placement.y + (point[1] - source_center_y) * final_scale),
            ]
            for point in contour
        ]
        for contour in raw_contours
        if len(contour) >= 2
    ]

    fitted_primitives: list[dict[str, Any]] = []
    fallback_lines: list[list[list[float]]] = []
    for command, metadata in zip(placed_plan.commands, command_metadata):
        if isinstance(command, (PenUp, PenDown, TravelMove)):
            continue
        payload = canonical_command_to_debug_dict(command)
        if metadata:
            payload.update({key: value for key, value in metadata.items() if value is not None})
        fitted_primitives.append(payload)
        if payload.get('type') == 'line' and payload.get('fit_source') == 'fallback_line_chain':
            fallback_lines.append([
                [float(command.start[0]), float(command.start[1])],
                [float(command.end[0]), float(command.end[1])],
            ])

    payload = {
        'available': True,
        'source_type': source_type,
        'trace_mode': curve_fit_debug.get('trace_mode'),
        'raw_contour_count': int(curve_fit_debug.get('raw_contour_count', 0)),
        'span_count': int(curve_fit_debug.get('span_count', 0)),
        'merge_stats': dict(curve_fit_debug.get('merge_stats') or {}),
        'fit_summary': dict(curve_fit_debug.get('fit_summary') or {}),
        'fit_tolerances': dict(curve_fit_debug.get('fit_tolerances') or {}),
        'worst_spans': list(curve_fit_debug.get('worst_spans') or []),
        'overlay_geometry': {
            'raw_contours': raw_contours_board,
            'fitted_primitives': fitted_primitives,
            'fallback_line_spans': fallback_lines,
        },
    }
    return payload
