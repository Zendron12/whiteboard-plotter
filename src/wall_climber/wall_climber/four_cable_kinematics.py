from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


CableName = str

FOUR_CABLE_NAMES: tuple[CableName, ...] = (
    'top_left',
    'top_right',
    'bottom_left',
    'bottom_right',
)


@dataclass(frozen=True)
class CablePoint:
    x: float
    y: float


@dataclass(frozen=True)
class FourCableLayout:
    anchors: Mapping[CableName, CablePoint]
    attachments: Mapping[CableName, CablePoint]


def _finite_point(name: str, point: CablePoint) -> CablePoint:
    x = float(point.x)
    y = float(point.y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f'{name} must contain finite x/y values.')
    return CablePoint(x=x, y=y)


def _point_from_pair(name: str, value: CablePoint | tuple[float, float]) -> CablePoint:
    if isinstance(value, CablePoint):
        return _finite_point(name, value)
    if len(value) != 2:
        raise ValueError(f'{name} must contain exactly two values.')
    return _finite_point(name, CablePoint(float(value[0]), float(value[1])))


def normalize_layout(
    anchors: Mapping[CableName, CablePoint | tuple[float, float]],
    attachments: Mapping[CableName, CablePoint | tuple[float, float]],
) -> FourCableLayout:
    normalized_anchors: dict[CableName, CablePoint] = {}
    normalized_attachments: dict[CableName, CablePoint] = {}
    for cable_name in FOUR_CABLE_NAMES:
        if cable_name not in anchors:
            raise ValueError(f'Missing four-cable anchor: {cable_name}')
        if cable_name not in attachments:
            raise ValueError(f'Missing four-cable attachment: {cable_name}')
        normalized_anchors[cable_name] = _point_from_pair(f'anchor {cable_name}', anchors[cable_name])
        normalized_attachments[cable_name] = _point_from_pair(
            f'attachment {cable_name}',
            attachments[cable_name],
        )
    return FourCableLayout(
        anchors=normalized_anchors,
        attachments=normalized_attachments,
    )


def compute_four_cable_lengths(
    carriage_center: CablePoint | tuple[float, float],
    anchors: Mapping[CableName, CablePoint | tuple[float, float]],
    attachments: Mapping[CableName, CablePoint | tuple[float, float]],
) -> dict[CableName, float]:
    center = _point_from_pair('carriage center', carriage_center)
    layout = normalize_layout(anchors, attachments)
    lengths: dict[CableName, float] = {}
    for cable_name in FOUR_CABLE_NAMES:
        anchor = layout.anchors[cable_name]
        attachment = layout.attachments[cable_name]
        attachment_x = center.x + attachment.x
        attachment_y = center.y + attachment.y
        length = math.hypot(anchor.x - attachment_x, anchor.y - attachment_y)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f'Computed invalid cable length for {cable_name}: {length!r}')
        lengths[cable_name] = length
    return lengths
