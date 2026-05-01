from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2  # type: ignore
import numpy


_SVG_CONTENT_TYPES = {
    'application/xml',
    'image/svg+xml',
    'text/xml',
}
_IMAGE_CONTENT_TYPES = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
}
_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}


@dataclass(frozen=True)
class UploadedVectorFile:
    source_type: str
    extension: str
    normalized_content_type: str
    svg_text: str | None = None
    image_size: tuple[int, int] | None = None


def _normalized_content_type(content_type: str | None) -> str:
    return str(content_type or '').split(';', 1)[0].strip().lower()


def _decode_svg_text(content: bytes) -> str:
    last_error: Exception | None = None
    for encoding in ('utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be'):
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            last_error = exc
            continue
        if not root.tag.lower().endswith('svg'):
            raise ValueError('uploaded svg root element must be <svg>')
        return text
    if last_error is not None:
        raise ValueError(f'uploaded svg is invalid: {last_error}')
    raise ValueError('uploaded svg is invalid')


def _decode_image(content: bytes) -> tuple[int, int]:
    array = numpy.frombuffer(content, dtype=numpy.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError('uploaded file is not a decodable image')
    height, width = decoded.shape[:2]
    return (int(width), int(height))


def classify_uploaded_vector_file(
    filename: str | None,
    content_type: str | None,
    content: bytes,
) -> UploadedVectorFile:
    suffix = Path(filename or '').suffix.lower()
    normalized_type = _normalized_content_type(content_type)

    looks_like_svg = (
        suffix == '.svg'
        or normalized_type in _SVG_CONTENT_TYPES
        or content.lstrip().startswith(b'<?xml')
        or content.lstrip().startswith(b'<svg')
    )
    if looks_like_svg:
        svg_text = _decode_svg_text(content)
        return UploadedVectorFile(
            source_type='svg',
            extension='.svg',
            normalized_content_type=normalized_type or 'image/svg+xml',
            svg_text=svg_text,
            image_size=None,
        )

    looks_like_image = (
        normalized_type in _IMAGE_CONTENT_TYPES
        or suffix in _IMAGE_SUFFIXES
    )
    if looks_like_image:
        image_size = _decode_image(content)
        extension = _IMAGE_CONTENT_TYPES.get(normalized_type)
        if extension is None:
            extension = '.jpg' if suffix == '.jpeg' else (suffix or '.png')
        return UploadedVectorFile(
            source_type='image',
            extension=extension,
            normalized_content_type=normalized_type or 'application/octet-stream',
            svg_text=None,
            image_size=image_size,
        )

    raise ValueError('unsupported upload content type')


def infer_uploaded_source_type(
    *,
    stored_filename: str | None,
    original_filename: str | None,
    content_type: str | None,
    source_type: str | None,
) -> str:
    normalized_source = str(source_type or '').strip().lower()
    if normalized_source in {'image', 'svg'}:
        return normalized_source

    normalized_type = _normalized_content_type(content_type)
    if normalized_type in _SVG_CONTENT_TYPES:
        return 'svg'
    if normalized_type in _IMAGE_CONTENT_TYPES:
        return 'image'

    for name in (stored_filename, original_filename):
        suffix = Path(name or '').suffix.lower()
        if suffix == '.svg':
            return 'svg'
        if suffix in _IMAGE_SUFFIXES:
            return 'image'

    return 'image'


__all__ = [
    'UploadedVectorFile',
    'classify_uploaded_vector_file',
    'infer_uploaded_source_type',
]
