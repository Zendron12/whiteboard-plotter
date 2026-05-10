from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import cv2  # type: ignore
import numpy


@dataclass(frozen=True)
class ColorToSketchResult:
    sketch_png: bytes
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop('sketch_png', None)
        return payload


def _decode_bgr(image_bytes: bytes) -> numpy.ndarray:
    encoded = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('colored image could not be decoded')
    return image


def _resize_to_max_dim(image: numpy.ndarray, *, max_image_dim: int) -> tuple[numpy.ndarray, float]:
    height, width = image.shape[:2]
    max_dim = max(1, int(max_image_dim))
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0
    scale = float(max_dim) / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _remove_small_components(mask: numpy.ndarray, *, min_area_px: int) -> tuple[numpy.ndarray, int]:
    binary = (mask > 0).astype(numpy.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = numpy.zeros_like(binary)
    removed = 0
    min_area = max(1, int(min_area_px))
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            removed += 1
            continue
        kept[labels == label] = 255
    return kept.astype(numpy.uint8), removed


def convert_color_image_to_sketch(
    image_bytes: bytes,
    *,
    method: str = 'opencv_pencil',
    max_image_dim: int = 1000,
) -> ColorToSketchResult:
    normalized_method = str(method or 'opencv_pencil').strip().lower()
    if normalized_method not in {'opencv_pencil', 'opencv_edge'}:
        raise ValueError('color_to_sketch_method must be one of: opencv_pencil, opencv_edge')

    started = time.perf_counter()
    original = _decode_bgr(image_bytes)
    resized, resize_scale = _resize_to_max_dim(original, max_image_dim=max_image_dim)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=55, sigmaSpace=55)

    median_value = float(numpy.median(denoised))
    lower = int(max(20, min(140, median_value * 0.50)))
    upper = int(max(lower + 20, min(220, median_value * 1.15)))
    edge_mask = cv2.Canny(denoised, lower, upper)

    if normalized_method == 'opencv_pencil':
        inverted = 255 - denoised
        blurred_inverted = cv2.GaussianBlur(inverted, (0, 0), sigmaX=9.0, sigmaY=9.0)
        pencil = cv2.divide(denoised, 255 - blurred_inverted, scale=256)
        pencil = cv2.GaussianBlur(pencil, (3, 3), 0)
        block_size = max(15, min(71, (min(pencil.shape[:2]) // 18) | 1))
        pencil_binary = cv2.adaptiveThreshold(
            pencil,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            int(block_size),
            5,
        )
        line_mask = cv2.max(edge_mask, pencil_binary)
    else:
        line_mask = edge_mask

    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, numpy.ones((2, 2), dtype=numpy.uint8), iterations=1)
    line_mask, removed_components = _remove_small_components(line_mask, min_area_px=3)

    foreground_ratio = float(numpy.count_nonzero(line_mask)) / float(max(1, line_mask.size))
    if foreground_ratio > 0.32:
        # Fall back to the cleaner edge-only mask if pencil shading turned too much texture into ink.
        line_mask = edge_mask
        line_mask, removed_components = _remove_small_components(line_mask, min_area_px=3)
        foreground_ratio = float(numpy.count_nonzero(line_mask)) / float(max(1, line_mask.size))

    line_mask = cv2.dilate(line_mask, numpy.ones((2, 2), dtype=numpy.uint8), iterations=1)
    sketch = 255 - line_mask
    ok, encoded = cv2.imencode('.png', sketch)
    if not ok:
        raise ValueError('failed to encode OpenCV sketch output')

    metadata = {
        'color_to_sketch_method': normalized_method,
        'color_to_sketch_backend': 'opencv',
        'original_image_size': {'width_px': int(original.shape[1]), 'height_px': int(original.shape[0])},
        'processed_image_size': {'width_px': int(sketch.shape[1]), 'height_px': int(sketch.shape[0])},
        'resize_scale': float(resize_scale),
        'canny_lower_threshold': int(lower),
        'canny_upper_threshold': int(upper),
        'foreground_ratio': float(foreground_ratio),
        'removed_small_component_count': int(removed_components),
        'processing_time_ms': (time.perf_counter() - started) * 1000.0,
    }
    return ColorToSketchResult(sketch_png=bytes(encoded.tobytes()), metadata=metadata)
