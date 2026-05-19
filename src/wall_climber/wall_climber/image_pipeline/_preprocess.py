"""Pre-processing stage for sketch / line-art input.

Enhances raw raster input before the threshold + skeleton stages so that:
  1. Low-contrast scans don't break the skeleton.
  2. JPEG/compression noise doesn't grow into spurious skeleton spurs.
  3. Uneven background lighting doesn't make adaptive thresholding flicker.

The pipeline is conservative: only intensity is touched, geometry is never
altered, and a uint8 array of the same shape as the input is returned.
"""

from __future__ import annotations

import cv2  # type: ignore
import numpy


def enhance_for_extraction(
    gray: numpy.ndarray,
    *,
    use_clahe: bool = True,
    use_bilateral: bool = True,
    use_unsharp: bool = True,
    normalize_background: bool = True,
) -> numpy.ndarray:
    """Run the preprocessing chain. Returns a fresh uint8 array."""
    if gray.dtype != numpy.uint8:
        working = numpy.clip(gray, 0, 255).astype(numpy.uint8)
    else:
        working = gray.copy()

    if normalize_background:
        working = _normalize_background(working)
    if use_clahe:
        working = _apply_clahe(working)
    if use_bilateral:
        working = _apply_bilateral(working)
    if use_unsharp:
        working = _apply_unsharp(working)
    return working


def _normalize_background(gray: numpy.ndarray) -> numpy.ndarray:
    short_side = max(1, min(gray.shape[:2]))
    kernel_size = max(15, min(101, short_side // 24))
    if kernel_size % 2 == 0:
        kernel_size += 1
    background = cv2.medianBlur(gray, kernel_size)
    safe_bg = numpy.maximum(background, 1).astype(numpy.float32)
    flattened = (gray.astype(numpy.float32) / safe_bg) * 192.0
    return numpy.clip(flattened, 0, 255).astype(numpy.uint8)


def _apply_clahe(gray: numpy.ndarray) -> numpy.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _apply_bilateral(gray: numpy.ndarray) -> numpy.ndarray:
    return cv2.bilateralFilter(gray, d=5, sigmaColor=50.0, sigmaSpace=50.0)


def _apply_unsharp(gray: numpy.ndarray) -> numpy.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    sharpened = cv2.addWeighted(gray, 1.4, blurred, -0.4, 0)
    return numpy.clip(sharpened, 0, 255).astype(numpy.uint8)


__all__ = ['enhance_for_extraction']
