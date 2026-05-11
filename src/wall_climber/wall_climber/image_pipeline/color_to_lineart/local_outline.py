from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import cv2  # type: ignore
import numpy


_COMPLEX_WARNING = (
    'Local outline conversion may be noisy for complex painted images; '
    'AI line-art backend recommended later.'
)


@dataclass(frozen=True)
class ColorLineArtResult:
    line_art_png: bytes
    metadata: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop('line_art_png', None)
        return payload


def _decode_bgr(image_bytes: bytes) -> numpy.ndarray:
    encoded = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('colored image could not be decoded')
    return image


def _resize_to_max_dim(image: numpy.ndarray, *, max_image_dim: int) -> tuple[numpy.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    max_dim = max(1, int(max_image_dim))
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
    kept = numpy.zeros_like(binary, dtype=numpy.uint8)
    removed = 0
    min_area = max(1, int(min_area_px))
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            removed += 1
            continue
        kept[labels == label] = 255
    return kept, removed


def _component_count(mask: numpy.ndarray) -> int:
    component_count, _labels = cv2.connectedComponents((mask > 0).astype(numpy.uint8), connectivity=8)
    return max(0, int(component_count) - 1)


def _quantized_region_boundaries(image: numpy.ndarray) -> numpy.ndarray:
    try:
        flattened = cv2.pyrMeanShiftFiltering(image, sp=12, sr=36)
    except cv2.error:
        flattened = cv2.medianBlur(image, 5)
    quantized = (flattened.astype(numpy.uint16) // 32).astype(numpy.uint16)
    labels = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]
    boundaries = numpy.zeros(labels.shape, dtype=numpy.uint8)
    boundaries[:, 1:] |= (labels[:, 1:] != labels[:, :-1])
    boundaries[1:, :] |= (labels[1:, :] != labels[:-1, :])
    boundaries = cv2.dilate(boundaries * 255, numpy.ones((2, 2), dtype=numpy.uint8), iterations=1)
    return boundaries.astype(numpy.uint8)


def _strong_edge_mask(gray: numpy.ndarray, *, diagnostic: bool = False) -> numpy.ndarray:
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
    median_value = float(numpy.median(denoised))
    if diagnostic:
        lower = int(max(20, min(130, median_value * 0.45)))
        upper = int(max(lower + 20, min(230, median_value * 1.10)))
    else:
        lower = int(max(45, min(160, median_value * 0.72)))
        upper = int(max(lower + 35, min(245, median_value * 1.35)))
    return cv2.Canny(denoised, lower, upper)


def _photo_diagram_edge_mask(image: numpy.ndarray) -> tuple[numpy.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=55, sigmaSpace=55)
    median_value = float(numpy.median(denoised))
    lower = int(max(20, min(150, median_value * 0.58)))
    upper = int(max(lower + 35, min(245, median_value * 1.22)))
    edges = cv2.Canny(denoised, lower, upper)
    edge_pixel_ratio = float(numpy.count_nonzero(edges)) / float(max(1, edges.size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, numpy.ones((3, 3), dtype=numpy.uint8), iterations=1)
    metadata = {
        'photo_diagram_preprocessing': 'lab_lightness_clahe_bilateral',
        'canny_lower_threshold': int(lower),
        'canny_upper_threshold': int(upper),
        'edge_pixel_ratio': edge_pixel_ratio,
    }
    return closed.astype(numpy.uint8), metadata


def _image_complexity_metrics(image: numpy.ndarray) -> dict[str, float | int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    texture_score = float(numpy.mean(numpy.abs(laplacian)))
    edges = _strong_edge_mask(gray, diagnostic=True)
    edge_density = float(numpy.count_nonzero(edges)) / float(max(1, edges.size))
    quantized = (image.astype(numpy.uint16) // 32).reshape(-1, 3)
    unique_quantized_colors = int(numpy.unique(quantized, axis=0).shape[0])
    saturation = hsv[:, :, 1].astype(numpy.float32)
    return {
        'texture_score': texture_score,
        'edge_density': edge_density,
        'unique_quantized_colors': unique_quantized_colors,
        'mean_saturation': float(numpy.mean(saturation)),
        'p90_saturation': float(numpy.percentile(saturation, 90.0)),
    }


def _classify_profile(metrics: dict[str, float | int]) -> tuple[str, str, tuple[str, ...]]:
    texture_score = float(metrics['texture_score'])
    edge_density = float(metrics['edge_density'])
    unique_colors = int(metrics['unique_quantized_colors'])
    if texture_score <= 9.5 and edge_density <= 0.13 and unique_colors <= 120:
        return 'simple_cartoon_diagram', 'good', ()
    if texture_score <= 14.0 and edge_density <= 0.18 and unique_colors <= 180:
        return 'illustration_moderate', 'noisy', ('Local outline quality is moderate; inspect executable preview before drawing.',)
    return 'complex_artwork_photo', 'complex', (_COMPLEX_WARNING,)


def _build_outline_mask(image: numpy.ndarray, *, method: str, profile: str) -> numpy.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'opencv_edge_diagnostic':
        return _strong_edge_mask(gray, diagnostic=True)

    region_boundaries = _quantized_region_boundaries(image)
    dark_or_strong_edges = _strong_edge_mask(gray, diagnostic=False)
    if profile == 'complex_artwork_photo':
        mask = dark_or_strong_edges
    else:
        mask = cv2.max(region_boundaries, dark_or_strong_edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, numpy.ones((2, 2), dtype=numpy.uint8), iterations=1)
    return mask.astype(numpy.uint8)


def convert_color_image_to_lineart(
    image_bytes: bytes,
    *,
    method: str = 'auto_outline',
    max_image_dim: int = 1000,
) -> ColorLineArtResult:
    normalized_method = str(method or 'auto_outline').strip().lower()
    if normalized_method not in {'auto_outline', 'photo_diagram_edges', 'simple_cartoon', 'opencv_edge_diagnostic'}:
        raise ValueError(
            'color_lineart_method must be one of: auto_outline, photo_diagram_edges, simple_cartoon, opencv_edge_diagnostic'
        )

    started = time.perf_counter()
    original = _decode_bgr(image_bytes)
    resized, resize_scale = _resize_to_max_dim(original, max_image_dim=max_image_dim)
    metrics = _image_complexity_metrics(resized)
    profile, quality, warnings = _classify_profile(metrics)
    effective_method = normalized_method
    if normalized_method == 'simple_cartoon':
        profile = 'simple_cartoon_diagram'
        quality = 'good' if quality != 'complex' else 'noisy'
        warnings = tuple(warning for warning in warnings if warning != _COMPLEX_WARNING)
        if quality == 'noisy':
            warnings = warnings + ('Simple cartoon mode was forced on a complex image; inspect preview before drawing.',)
    lineart_metadata: dict[str, Any] = {}
    if normalized_method == 'photo_diagram_edges':
        effective_method = 'photo_diagram_edges'
        if profile == 'complex_artwork_photo':
            warnings = warnings + ('Complex colored artwork may require AI line-art backend later.',)

    if normalized_method == 'photo_diagram_edges':
        mask, lineart_metadata = _photo_diagram_edge_mask(resized)
    else:
        mask = _build_outline_mask(resized, method=effective_method, profile=profile)
    mask, removed_components = _remove_small_components(mask, min_area_px=4 if quality == 'good' else 8)
    foreground_ratio = float(numpy.count_nonzero(mask)) / float(max(1, mask.size))
    component_count = _component_count(mask)
    if foreground_ratio > 0.22 or component_count > 2500:
        quality = 'complex' if quality == 'complex' else 'noisy'
        if _COMPLEX_WARNING not in warnings and foreground_ratio > 0.22:
            warnings = warnings + ('Converted outline is dense; robot drawing may look noisy.',)

    line_art = 255 - mask
    ok, encoded = cv2.imencode('.png', line_art)
    if not ok:
        raise ValueError('failed to encode local line-art output')

    metadata = {
        'color_lineart_method': normalized_method,
        'effective_color_lineart_method': effective_method,
        'color_lineart_backend': 'local_opencv_outline',
        'color_lineart_profile': profile,
        'color_lineart_quality': quality,
        'original_image_size': {'width_px': int(original.shape[1]), 'height_px': int(original.shape[0])},
        'processed_image_size': {'width_px': int(line_art.shape[1]), 'height_px': int(line_art.shape[0])},
        'resize_scale': float(resize_scale),
        'foreground_ratio': foreground_ratio,
        'component_count': int(component_count),
        'removed_small_component_count': int(removed_components),
        'processing_time_ms': (time.perf_counter() - started) * 1000.0,
        **metrics,
        **lineart_metadata,
    }
    return ColorLineArtResult(
        line_art_png=bytes(encoded.tobytes()),
        metadata=metadata,
        warnings=tuple(str(warning) for warning in warnings),
    )
