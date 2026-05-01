from __future__ import annotations

from pathlib import Path
import math
import time
from typing import Iterable

import cv2  # type: ignore
import numpy

from wall_climber.image_pipeline.types import (
    DrawingPathPlan,
    PipelineMetrics,
    PipelineMode,
    Point2D,
    Stroke,
)


_EPS = 1.0e-9
_NEIGHBORS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),            (1, 0),
    (-1, 1),  (0, 1),   (1, 1),
)


Pixel = tuple[int, int]
PixelStroke = tuple[Pixel, ...]


def _read_image_payload(image_bytes_or_path: bytes | bytearray | str | Path) -> tuple[bytes, str | None]:
    if isinstance(image_bytes_or_path, (bytes, bytearray)):
        return bytes(image_bytes_or_path), None
    if isinstance(image_bytes_or_path, (str, Path)):
        path = Path(image_bytes_or_path)
        return path.read_bytes(), str(path)
    raise TypeError('image_bytes_or_path must be bytes, str, or pathlib.Path.')


def _decode_grayscale(
    image_bytes_or_path: bytes | bytearray | str | Path,
) -> tuple[numpy.ndarray, tuple[int, int], str | None]:
    payload, source_path = _read_image_payload(image_bytes_or_path)
    if not payload:
        raise ValueError('Image payload is empty.')
    array = numpy.frombuffer(payload, dtype=numpy.uint8)
    color = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError('Failed to decode PNG/JPG image payload.')
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    return gray, (int(width), int(height)), source_path


def _resize_for_processing(gray: numpy.ndarray, *, max_image_dim: int) -> tuple[numpy.ndarray, float]:
    if max_image_dim <= 0:
        raise ValueError('max_image_dim must be > 0.')
    height, width = gray.shape[:2]
    longest = max(width, height)
    if longest <= max_image_dim:
        return gray, 1.0
    scale = float(max_image_dim) / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA), scale


def _normalize_grayscale(gray: numpy.ndarray) -> numpy.ndarray:
    low, high = numpy.percentile(gray, (2.0, 98.0))
    if high - low <= 1.0:
        return gray.copy()
    stretched = (gray.astype(numpy.float32) - float(low)) * (255.0 / float(high - low))
    return numpy.clip(stretched, 0.0, 255.0).astype(numpy.uint8)


def _border_pixels(gray: numpy.ndarray) -> numpy.ndarray:
    if gray.size == 0:
        return gray.reshape((0,))
    return numpy.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))


def _threshold_foreground(gray: numpy.ndarray) -> tuple[numpy.ndarray, dict[str, object]]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold, _unused = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    border_median = float(numpy.median(_border_pixels(blurred)))
    dark_foreground = border_median >= float(threshold)
    threshold_type = cv2.THRESH_BINARY_INV if dark_foreground else cv2.THRESH_BINARY
    _threshold, binary = cv2.threshold(blurred, float(threshold), 255, threshold_type)
    return binary.astype(numpy.uint8), {
        'threshold_method': 'otsu',
        'threshold_value': float(threshold),
        'foreground_polarity': 'dark_on_light' if dark_foreground else 'light_on_dark',
        'border_median': border_median,
    }


def _remove_small_components(binary: numpy.ndarray, *, min_component_area_px: int) -> tuple[numpy.ndarray, dict[str, int]]:
    min_area = max(1, int(min_component_area_px))
    mask = (binary > 0).astype(numpy.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = numpy.zeros_like(binary, dtype=numpy.uint8)
    kept = 0
    removed = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            removed += 1
            continue
        cleaned[labels == label] = 255
        kept += 1
    return cleaned, {
        'component_count': max(0, int(component_count) - 1),
        'kept_component_count': kept,
        'removed_component_count': removed,
    }


def _skeletonize_with_skimage(binary: numpy.ndarray) -> tuple[numpy.ndarray, str] | None:
    try:
        from skimage.morphology import skeletonize  # type: ignore
    except ImportError:
        return None
    skeleton = skeletonize(binary > 0)
    return (skeleton.astype(numpy.uint8) * 255), 'skimage.morphology.skeletonize'


def _skeletonize_with_cv2_thinning(binary: numpy.ndarray) -> tuple[numpy.ndarray, str] | None:
    if not (hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning')):
        return None
    skeleton = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return skeleton.astype(numpy.uint8), 'cv2.ximgproc.thinning'


def _skeletonize_foreground(binary: numpy.ndarray) -> tuple[numpy.ndarray, str]:
    for backend in (_skeletonize_with_skimage, _skeletonize_with_cv2_thinning):
        result = backend(binary)
        if result is not None:
            return result
    raise RuntimeError(
        'Sketch Centerline Mode requires a skeletonization backend: '
        'install scikit-image or OpenCV ximgproc thinning.'
    )


def _pixel_neighbors(pixel: Pixel, pixels: set[Pixel]) -> tuple[Pixel, ...]:
    x, y = pixel
    return tuple(
        (x + dx, y + dy)
        for dx, dy in _NEIGHBORS
        if (x + dx, y + dy) in pixels
    )


def _edge_key(first: Pixel, second: Pixel) -> tuple[Pixel, Pixel]:
    return (first, second) if first <= second else (second, first)


def _choose_next_pixel(previous: Pixel, current: Pixel, candidates: Iterable[Pixel]) -> Pixel:
    candidates = tuple(candidates)
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


def _dedupe_pixels(points: Iterable[Pixel]) -> PixelStroke:
    deduped: list[Pixel] = []
    for point in points:
        if deduped and deduped[-1] == point:
            continue
        deduped.append(point)
    return tuple(deduped)


def _stroke_length_px(stroke: PixelStroke) -> float:
    return sum(
        math.hypot(float(end[0] - start[0]), float(end[1] - start[1]))
        for start, end in zip(stroke[:-1], stroke[1:])
    )


def _trace_skeleton_pixels(
    skeleton: numpy.ndarray,
    *,
    min_stroke_length_px: float,
) -> tuple[PixelStroke, ...]:
    ys, xs = numpy.nonzero(skeleton > 0)
    pixels = {(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())}
    if not pixels:
        return ()

    neighbor_map = {pixel: _pixel_neighbors(pixel, pixels) for pixel in pixels}
    nodes = {
        pixel for pixel, neighbors in neighbor_map.items()
        if len(neighbors) != 2
    }
    visited_edges: set[tuple[Pixel, Pixel]] = set()
    traced: list[PixelStroke] = []

    def follow(start: Pixel, next_pixel: Pixel) -> PixelStroke:
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
            chosen = _choose_next_pixel(previous, current, candidates)
            path.append(chosen)
            visited_edges.add(_edge_key(current, chosen))
            previous, current = current, chosen
        return _dedupe_pixels(path)

    ordered_nodes = sorted(nodes, key=lambda pixel: (len(neighbor_map[pixel]) != 1, pixel[1], pixel[0]))
    for start in ordered_nodes:
        for neighbor in sorted(neighbor_map[start], key=lambda pixel: (pixel[1], pixel[0])):
            if _edge_key(start, neighbor) in visited_edges:
                continue
            stroke = follow(start, neighbor)
            if len(stroke) >= 2 and _stroke_length_px(stroke) >= min_stroke_length_px:
                traced.append(stroke)

    for start in sorted(pixels, key=lambda pixel: (pixel[1], pixel[0])):
        for neighbor in sorted(neighbor_map[start], key=lambda pixel: (pixel[1], pixel[0])):
            if _edge_key(start, neighbor) in visited_edges:
                continue
            stroke = follow(start, neighbor)
            if len(stroke) >= 2 and _stroke_length_px(stroke) >= min_stroke_length_px:
                traced.append(stroke)

    return tuple(traced)


def _point_line_distance(point: Pixel, start: Pixel, end: Pixel) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= _EPS:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, (((px - x1) * dx) + ((py - y1) * dy)) / denom))
    proj_x = x1 + (t * dx)
    proj_y = y1 + (t * dy)
    return math.hypot(px - proj_x, py - proj_y)


def _simplify_pixels(points: PixelStroke, *, epsilon_px: float) -> PixelStroke:
    if len(points) <= 2 or epsilon_px <= 0.0:
        return points

    def rdp(span: PixelStroke) -> list[Pixel]:
        if len(span) <= 2:
            return [span[0], span[-1]]
        start = span[0]
        end = span[-1]
        max_distance = -1.0
        split_index = 0
        for index, point in enumerate(span[1:-1], start=1):
            distance = _point_line_distance(point, start, end)
            if distance > max_distance:
                max_distance = distance
                split_index = index
        if max_distance <= epsilon_px:
            return [start, end]
        left = rdp(span[:split_index + 1])
        right = rdp(span[split_index:])
        return left[:-1] + right

    return _dedupe_pixels(rdp(points))


def _scale_strokes_to_board(
    pixel_strokes: tuple[PixelStroke, ...],
    *,
    board_width_m: float,
    board_height_m: float,
    margin_m: float,
) -> tuple[tuple[Stroke, ...], dict[str, object]]:
    if board_width_m <= 0.0 or board_height_m <= 0.0:
        raise ValueError('board_width_m and board_height_m must be > 0.')
    if margin_m < 0.0:
        raise ValueError('margin_m must be >= 0.')
    available_width = float(board_width_m) - (2.0 * float(margin_m))
    available_height = float(board_height_m) - (2.0 * float(margin_m))
    if available_width <= 0.0 or available_height <= 0.0:
        raise ValueError('margin_m leaves no drawable board area.')

    points = [point for stroke in pixel_strokes for point in stroke]
    if not points:
        raise ValueError('No skeleton strokes were traced from the image.')
    min_x = min(float(point[0]) for point in points)
    max_x = max(float(point[0]) for point in points)
    min_y = min(float(point[1]) for point in points)
    max_y = max(float(point[1]) for point in points)
    source_width = max_x - min_x
    source_height = max_y - min_y
    if source_width <= _EPS and source_height <= _EPS:
        raise ValueError('Traced sketch geometry is degenerate.')

    scale_candidates = []
    if source_width > _EPS:
        scale_candidates.append(available_width / source_width)
    if source_height > _EPS:
        scale_candidates.append(available_height / source_height)
    scale = min(scale_candidates)
    offset_x = float(margin_m) + ((available_width - (source_width * scale)) * 0.5) - (min_x * scale)
    offset_y = float(margin_m) + ((available_height - (source_height * scale)) * 0.5) - (min_y * scale)

    strokes: list[Stroke] = []
    for index, pixel_stroke in enumerate(pixel_strokes):
        board_points: list[Point2D] = []
        for point in pixel_stroke:
            board_point = Point2D(
                x=(float(point[0]) * scale) + offset_x,
                y=(float(point[1]) * scale) + offset_y,
            )
            if board_points and board_points[-1] == board_point:
                continue
            board_points.append(board_point)
        if len(board_points) >= 2:
            strokes.append(Stroke(points=tuple(board_points), pen_down=True, label=f'sketch_stroke_{index}'))

    if not strokes:
        raise ValueError('No non-degenerate sketch strokes remained after scaling.')

    return tuple(strokes), {
        'source_bounds_px': {
            'x_min': min_x,
            'x_max': max_x,
            'y_min': min_y,
            'y_max': max_y,
        },
        'scale_m_per_px': float(scale),
        'offset_m': {'x': float(offset_x), 'y': float(offset_y)},
        'board_size_m': {'width': float(board_width_m), 'height': float(board_height_m)},
        'margin_m': float(margin_m),
    }


def _drawing_length(points: tuple[Point2D, ...]) -> float:
    return sum(
        math.hypot(end.x - start.x, end.y - start.y)
        for start, end in zip(points[:-1], points[1:])
    )


def _metrics(
    strokes: tuple[Stroke, ...],
    *,
    points_before_simplification: int,
    processing_time_ms: float,
) -> PipelineMetrics:
    total_drawing_length = sum(_drawing_length(stroke.points) for stroke in strokes)
    pen_up_travel = 0.0
    for previous, current in zip(strokes[:-1], strokes[1:]):
        pen_up_travel += math.hypot(
            current.points[0].x - previous.points[-1].x,
            current.points[0].y - previous.points[-1].y,
        )
    return PipelineMetrics(
        stroke_count=len(strokes),
        points_before_simplification=points_before_simplification,
        points_after_simplification=sum(len(stroke.points) for stroke in strokes),
        total_drawing_length_m=total_drawing_length,
        pen_up_travel_length_m=pen_up_travel,
        pen_lift_count=len(strokes),
        processing_time_ms=processing_time_ms,
    )


def vectorize_sketch_image_to_plan(
    image_bytes_or_path: bytes | bytearray | str | Path,
    *,
    board_width_m: float,
    board_height_m: float,
    margin_m: float = 0.05,
    max_image_dim: int = 1200,
    min_component_area_px: int = 8,
    min_stroke_length_px: float = 4.0,
    simplify_epsilon_px: float = 1.0,
) -> DrawingPathPlan:
    """Convert a high-contrast sketch image into a board-space DrawingPathPlan."""

    started = time.perf_counter()
    gray, original_size, source_path = _decode_grayscale(image_bytes_or_path)
    processed_gray, resize_scale = _resize_for_processing(gray, max_image_dim=max_image_dim)
    normalized_gray = _normalize_grayscale(processed_gray)
    binary, threshold_metadata = _threshold_foreground(normalized_gray)
    cleaned_binary, component_metadata = _remove_small_components(
        binary,
        min_component_area_px=min_component_area_px,
    )
    skeleton, skeleton_backend = _skeletonize_foreground(cleaned_binary)
    raw_strokes = _trace_skeleton_pixels(
        skeleton,
        min_stroke_length_px=max(0.0, float(min_stroke_length_px)),
    )
    if not raw_strokes:
        raise ValueError('No drawable centerline strokes were extracted from the sketch image.')
    points_before_simplification = sum(len(stroke) for stroke in raw_strokes)
    simplified_strokes = tuple(
        stroke for stroke in (
            _simplify_pixels(stroke, epsilon_px=max(0.0, float(simplify_epsilon_px)))
            for stroke in raw_strokes
        )
        if len(stroke) >= 2 and _stroke_length_px(stroke) >= max(0.0, float(min_stroke_length_px))
    )
    if not simplified_strokes:
        raise ValueError('No drawable centerline strokes remained after simplification.')

    board_strokes, placement_metadata = _scale_strokes_to_board(
        simplified_strokes,
        board_width_m=float(board_width_m),
        board_height_m=float(board_height_m),
        margin_m=float(margin_m),
    )
    processing_time_ms = (time.perf_counter() - started) * 1000.0
    processed_height, processed_width = processed_gray.shape[:2]
    metadata = {
        'source_path': source_path,
        'original_image_size': {'width_px': original_size[0], 'height_px': original_size[1]},
        'processed_image_size': {'width_px': int(processed_width), 'height_px': int(processed_height)},
        'resize_scale': float(resize_scale),
        'max_image_dim': int(max_image_dim),
        'min_component_area_px': int(min_component_area_px),
        'min_stroke_length_px': float(min_stroke_length_px),
        'simplify_epsilon_px': float(simplify_epsilon_px),
        'foreground_pixel_count': int(numpy.count_nonzero(binary)),
        'cleaned_foreground_pixel_count': int(numpy.count_nonzero(cleaned_binary)),
        'skeleton_pixel_count': int(numpy.count_nonzero(skeleton)),
        'raw_stroke_count': len(raw_strokes),
        'simplified_stroke_count': len(simplified_strokes),
        'skeleton_backend': skeleton_backend,
        **threshold_metadata,
        **component_metadata,
        **placement_metadata,
    }
    return DrawingPathPlan(
        mode=PipelineMode.SKETCH_CENTERLINE,
        frame='board',
        strokes=board_strokes,
        metrics=_metrics(
            board_strokes,
            points_before_simplification=points_before_simplification,
            processing_time_ms=processing_time_ms,
        ),
        metadata=metadata,
    )

