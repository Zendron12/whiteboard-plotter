"""Filled vs thin region classification.

The previous pipeline ran skeletonization on every foreground component,
which collapses solid filled shapes into a single point. This module
classifies each connected component using the compactness ratio
``4*pi*area / perimeter^2`` (1.0 for a circle, ~0 for a thin line).
Filled components are replaced with their 1-pixel outline; thin ones
pass through unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2  # type: ignore
import numpy


@dataclass(frozen=True)
class FilledRegionSplit:
    thin_mask: numpy.ndarray
    outline_mask: numpy.ndarray
    filled_component_count: int
    thin_component_count: int


def split_filled_and_thin(
    binary: numpy.ndarray,
    *,
    compactness_threshold: float = 0.55,
    min_filled_area_px: int = 64,
) -> FilledRegionSplit:
    """Split a binary foreground mask into thin and filled-region masks."""
    if binary.dtype != numpy.uint8:
        binary = (binary > 0).astype(numpy.uint8) * 255
    foreground = (binary > 0).astype(numpy.uint8)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground, connectivity=8,
    )
    thin_mask = numpy.zeros_like(foreground)
    outline_mask = numpy.zeros_like(foreground)
    filled_count = 0
    thin_count = 0

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component_mask = (labels == label).astype(numpy.uint8)
        # Use RETR_CCOMP so that components with internal holes (donuts,
        # rings, the inside of a 'P' or 'A' character) emit BOTH outer and
        # hole boundaries instead of only the outer one. RETR_EXTERNAL
        # would draw a filled donut as just the outside circle, losing the
        # inner edge that the artist intentionally drew. The hierarchy
        # tells us which contours are outer (parent==-1) vs holes; we use
        # only the outer one for the compactness check so a donut with a
        # large hole still classifies the same way it did before.
        contours, hierarchy = cv2.findContours(
            component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE,
        )
        if not contours or hierarchy is None:
            continue
        outer_contours = [
            contours[index]
            for index, info in enumerate(hierarchy[0])
            if int(info[3]) == -1
        ]
        if not outer_contours:
            continue
        outer_perimeter = sum(cv2.arcLength(c, True) for c in outer_contours)
        if outer_perimeter <= 0.0 or area < min_filled_area_px:
            thin_mask |= component_mask
            thin_count += 1
            continue
        compactness = (4.0 * math.pi * float(area)) / (
            float(outer_perimeter) * float(outer_perimeter)
        )
        if compactness >= compactness_threshold:
            # Draw all contours (outer + holes) so the inner edge of a
            # donut is preserved as part of the outline mask.
            cv2.drawContours(outline_mask, contours, -1, color=1, thickness=1)
            filled_count += 1
        else:
            thin_mask |= component_mask
            thin_count += 1

    return FilledRegionSplit(
        thin_mask=(thin_mask > 0).astype(numpy.uint8) * 255,
        outline_mask=(outline_mask > 0).astype(numpy.uint8) * 255,
        filled_component_count=filled_count,
        thin_component_count=thin_count,
    )


__all__ = ['FilledRegionSplit', 'split_filled_and_thin']
