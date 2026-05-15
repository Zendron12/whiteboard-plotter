"""Shared SVG parsing primitives used by ``vector_pipeline``.

These helpers parse SVG XML fragments without holding any module state. They
are extracted so the vector pipeline can stay focused on text/SVG/image
orchestration. Parsing of the full SVG document, font glyphs, and
``path/d`` strings stays in ``vector_pipeline.py`` because those routines
also produce the project-specific stroke representations.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET


Point = tuple[float, float]


def svg_tag_name(tag: str) -> str:
    """Return the local element name without an XML namespace prefix."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def parse_float(value: str | None, default: float = 0.0) -> float:
    """Parse a CSS-style numeric value (``0.5``, ``12px``, ``1e-3``, ...).

    Strips any non-numeric characters before conversion so suffixes like
    ``px`` or ``mm`` are tolerated. Returns ``default`` for empty/None input
    or strings that fail to parse.
    """
    if value is None:
        return default
    cleaned = value.strip()
    if not cleaned:
        return default
    cleaned = re.sub(r'[^0-9eE+\-.]', '', cleaned)
    if not cleaned:
        return default
    return float(cleaned)


def parse_points_attr(raw: str) -> list[Point]:
    """Parse the ``points`` attribute of ``<polyline>`` / ``<polygon>``.

    The attribute is a flat whitespace/comma-separated list of numbers; pairs
    are interpreted as ``(x, y)``. Raises ``ValueError`` when the count is
    odd or fewer than two pairs are present.
    """
    values = re.findall(
        r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?',
        raw,
    )
    if len(values) < 4 or len(values) % 2 != 0:
        raise ValueError('Invalid SVG points attribute.')
    points: list[Point] = []
    for index in range(0, len(values), 2):
        points.append((float(values[index]), float(values[index + 1])))
    return points


def has_svg_transform(element: ET.Element) -> bool:
    """Return True when the element carries a non-empty ``transform`` attribute."""
    transform = element.attrib.get('transform')
    return isinstance(transform, str) and bool(transform.strip())


__all__ = [
    'Point',
    'svg_tag_name',
    'parse_float',
    'parse_points_attr',
    'has_svg_transform',
]
