from __future__ import annotations

import cv2  # type: ignore
import numpy

from wall_climber.image_pipeline.color_to_lineart import convert_color_image_to_lineart


def _encode_jpeg(image: numpy.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    return bytes(encoded.tobytes())


def _cartoon_diagram_image() -> bytes:
    image = numpy.full((240, 360, 3), (230, 245, 250), dtype=numpy.uint8)
    cv2.rectangle(image, (20, 180), (160, 230), (220, 190, 140), -1)
    cv2.rectangle(image, (20, 180), (160, 230), (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.circle(image, (70, 65), 30, (0, 220, 255), -1)
    cv2.circle(image, (70, 65), 30, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (220, 80), (58, 24), 0, 0, 360, (245, 245, 245), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (220, 80), (58, 24), 0, 0, 360, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.line(image, (205, 142), (270, 205), (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.putText(image, 'Rain', (190, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return _encode_jpeg(image)


def _complex_painted_image() -> bytes:
    rng = numpy.random.default_rng(7)
    base = rng.integers(0, 255, size=(220, 340, 3), dtype=numpy.uint8)
    gradient = numpy.linspace(0, 120, 340, dtype=numpy.uint8)
    base[:, :, 1] = numpy.clip(base[:, :, 1].astype(numpy.uint16) + gradient[None, :], 0, 255).astype(numpy.uint8)
    for radius in range(12, 92, 10):
        color = tuple(int(value) for value in rng.integers(20, 235, size=3))
        cv2.circle(base, (170, 110), radius, color, 2, lineType=cv2.LINE_AA)
    return _encode_jpeg(base)


def _foreground_ratio(line_art_png: bytes) -> float:
    decoded = cv2.imdecode(numpy.frombuffer(line_art_png, dtype=numpy.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded is not None
    return float(numpy.count_nonzero(decoded < 128)) / float(decoded.size)


def test_auto_outline_converts_cartoon_diagram_to_clean_foreground() -> None:
    result = convert_color_image_to_lineart(_cartoon_diagram_image(), method='auto_outline', max_image_dim=500)

    assert result.metadata['color_lineart_backend'] == 'local_opencv_outline'
    assert result.metadata['color_lineart_method'] == 'auto_outline'
    assert result.metadata['color_lineart_quality'] in {'good', 'noisy'}
    ratio = _foreground_ratio(result.line_art_png)
    assert 0.005 < ratio < 0.20


def test_complex_colored_image_returns_warning_without_failing() -> None:
    result = convert_color_image_to_lineart(_complex_painted_image(), method='auto_outline', max_image_dim=500)

    assert result.line_art_png
    assert result.metadata['color_lineart_quality'] in {'noisy', 'complex'}
    assert result.warnings


def test_simple_cartoon_mode_still_returns_line_art() -> None:
    result = convert_color_image_to_lineart(_cartoon_diagram_image(), method='simple_cartoon', max_image_dim=500)

    assert result.line_art_png
    assert result.metadata['color_lineart_method'] == 'simple_cartoon'
    assert _foreground_ratio(result.line_art_png) > 0.005
