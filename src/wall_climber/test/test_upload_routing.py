from __future__ import annotations

import cv2
import numpy

from wall_climber.ingestion.upload_routing import (
    classify_uploaded_vector_file,
    infer_uploaded_source_type,
)


def _encode_png(image: numpy.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return bytes(encoded.tobytes())


def test_classify_uploaded_vector_file_detects_svg_markup() -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M1 1 L9 9"/></svg>'

    details = classify_uploaded_vector_file('rose.svg', 'image/svg+xml', svg)

    assert details.source_type == 'svg'
    assert details.extension == '.svg'
    assert details.svg_text is not None
    assert '<svg' in details.svg_text


def test_classify_uploaded_vector_file_detects_png_image() -> None:
    image = numpy.full((24, 24, 3), 255, dtype=numpy.uint8)
    cv2.circle(image, (12, 12), 6, (0, 0, 0), 2)
    payload = _encode_png(image)

    details = classify_uploaded_vector_file('flower.png', 'image/png', payload)

    assert details.source_type == 'image'
    assert details.extension == '.png'
    assert details.svg_text is None


def test_infer_uploaded_source_type_uses_metadata_and_suffix_fallback() -> None:
    assert infer_uploaded_source_type(
        stored_filename='abc.svg',
        original_filename='flower.svg',
        content_type='image/svg+xml',
        source_type=None,
    ) == 'svg'
    assert infer_uploaded_source_type(
        stored_filename='abc.png',
        original_filename='flower.png',
        content_type='image/png',
        source_type=None,
    ) == 'image'
