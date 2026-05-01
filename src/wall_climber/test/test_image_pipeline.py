from __future__ import annotations

from collections import Counter

import cv2
import numpy

from wall_climber.canonical_adapters import canonical_plan_to_draw_strokes
from wall_climber.vector_pipeline import (
    route_image_vector_pipeline,
    vectorize_image_to_canonical_plan,
)


def _encode_png(image: numpy.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return bytes(encoded.tobytes())


def _simple_line_art_image() -> bytes:
    image = numpy.full((160, 160, 3), 255, dtype=numpy.uint8)
    cv2.rectangle(image, (20, 20), (120, 120), (0, 0, 0), 3)
    cv2.circle(image, (80, 80), 25, (0, 0, 0), 3)
    return _encode_png(image)


def _circle_line_art_image() -> bytes:
    image = numpy.full((200, 200, 3), 255, dtype=numpy.uint8)
    cv2.circle(image, (100, 100), 55, (0, 0, 0), 3)
    return _encode_png(image)


def _petal_line_art_image() -> bytes:
    image = numpy.full((240, 240, 3), 255, dtype=numpy.uint8)

    def cubic_points(
        start: tuple[float, float],
        control1: tuple[float, float],
        control2: tuple[float, float],
        end: tuple[float, float],
        *,
        steps: int = 120,
    ) -> list[list[int]]:
        points: list[list[int]] = []
        for index in range(steps + 1):
            t = index / steps
            omt = 1.0 - t
            x = (
                (omt ** 3 * start[0])
                + (3.0 * omt * omt * t * control1[0])
                + (3.0 * omt * t * t * control2[0])
                + (t ** 3 * end[0])
            )
            y = (
                (omt ** 3 * start[1])
                + (3.0 * omt * omt * t * control1[1])
                + (3.0 * omt * t * t * control2[1])
                + (t ** 3 * end[1])
            )
            points.append([int(round(x)), int(round(y))])
        return points

    contour = cubic_points((120, 24), (182, 52), (178, 186), (120, 214))
    contour.extend(cubic_points((120, 214), (62, 186), (58, 52), (120, 24))[1:])
    cv2.polylines(image, [numpy.asarray(contour, dtype=numpy.int32)], True, (0, 0, 0), 3)
    return _encode_png(image)


def _broken_line_art_image() -> bytes:
    image = numpy.full((120, 220, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 60), (74, 60), (0, 0, 0), 3)
    cv2.line(image, (78, 60), (142, 60), (0, 0, 0), 3)
    cv2.line(image, (146, 60), (200, 60), (0, 0, 0), 3)
    return _encode_png(image)


def _faint_line_art_image() -> bytes:
    image = numpy.full((180, 180, 3), 255, dtype=numpy.uint8)
    ink = (110, 110, 110)
    cv2.circle(image, (90, 90), 46, ink, 2, lineType=cv2.LINE_AA)
    cv2.line(image, (50, 128), (130, 52), ink, 2, lineType=cv2.LINE_AA)
    return _encode_png(image)


def _complex_tonal_image() -> bytes:
    base = numpy.tile(numpy.linspace(30, 220, 160, dtype=numpy.uint8), (160, 1))
    noise = numpy.random.default_rng(0).integers(0, 35, size=(160, 160), dtype=numpy.uint8)
    gray = cv2.add(base, noise)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(image, (80, 80), 30, (10, 10, 10), -1)
    return _encode_png(image)


def _colored_illustration_image() -> bytes:
    image = numpy.full((220, 220, 3), 255, dtype=numpy.uint8)
    cv2.circle(image, (78, 88), 46, (64, 136, 236), -1)
    cv2.circle(image, (142, 122), 38, (84, 184, 112), -1)
    cv2.ellipse(image, (110, 110), (70, 52), 0, 0, 360, (20, 20, 20), 3)
    cv2.line(image, (32, 174), (192, 48), (42, 42, 42), 4)
    cv2.line(image, (28, 34), (194, 188), (212, 88, 96), 3)
    return _encode_png(image)


def _large_line_art_image() -> bytes:
    image = numpy.full((2400, 1800, 3), 255, dtype=numpy.uint8)
    cv2.circle(image, (900, 1200), 520, (0, 0, 0), 12, lineType=cv2.LINE_AA)
    cv2.line(image, (260, 1840), (1540, 580), (0, 0, 0), 10, lineType=cv2.LINE_AA)
    return _encode_png(image)


def _detailed_grayscale_line_art_image() -> bytes:
    image = numpy.full((480, 360, 3), 255, dtype=numpy.uint8)
    ink = (168, 168, 168)
    cv2.ellipse(image, (160, 150), (72, 104), 8, 0, 360, ink, 2, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (160, 150), (38, 54), 8, 0, 360, ink, 2, lineType=cv2.LINE_AA)
    cv2.line(image, (112, 120), (182, 118), ink, 2, lineType=cv2.LINE_AA)
    cv2.line(image, (122, 184), (178, 196), ink, 2, lineType=cv2.LINE_AA)
    cv2.polylines(
        image,
        [numpy.asarray([(196, 60), (250, 112), (278, 216), (242, 344)], dtype=numpy.int32)],
        False,
        ink,
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        image,
        [numpy.asarray([(124, 266), (88, 318), (62, 396)], dtype=numpy.int32)],
        False,
        ink,
        2,
        lineType=cv2.LINE_AA,
    )
    for offset in range(10):
        cv2.line(
            image,
            (120 + offset * 4, 66),
            (138 + offset * 6, 258),
            ink,
            1,
            lineType=cv2.LINE_AA,
    )
    return _encode_png(image)


def _portrait_filled_feature_line_art_image() -> bytes:
    image = numpy.full((480, 360, 3), 255, dtype=numpy.uint8)
    cv2.ellipse(image, (180, 118), (60, 82), 0, 0, 360, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(image, (145, 78), (120, 34), (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(image, (154, 76), (136, 28), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(image, (166, 78), (158, 32), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(image, (215, 78), (241, 36), (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(image, (206, 76), (223, 28), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(image, (194, 78), (202, 32), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(image, (140, 92), (166, 88), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(image, (194, 88), (220, 92), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (158, 106), (17, 7), 0, 0, 360, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(image, (158, 106), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (202, 106), (17, 7), 0, 0, 360, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(image, (202, 106), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.line(image, (180, 114), (174, 140), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(image, (174, 140), (186, 144), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (174, 140), (2, 1), 0, 0, 360, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (186, 140), (2, 1), 0, 0, 360, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (180, 164), (22, 8), 0, 10, 170, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.line(image, (156, 200), (132, 236), (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.line(image, (204, 200), (232, 244), (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return _encode_png(image)


def _dense_decorative_line_art_image() -> bytes:
    image = numpy.full((520, 360, 3), 255, dtype=numpy.uint8)
    cv2.ellipse(image, (250, 165), (72, 126), 8, 0, 360, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.polylines(
        image,
        [numpy.asarray([(210, 80), (156, 128), (126, 236), (160, 370)], dtype=numpy.int32)],
        False,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.polylines(
        image,
        [numpy.asarray([(116, 404), (188, 328), (248, 286), (308, 188), (336, 90)], dtype=numpy.int32)],
        False,
        (0, 0, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    for index in range(34):
        y = 78 + (index * 12)
        cv2.line(image, (40, y), (150, max(30, y - 18)), (0, 0, 0), 1, lineType=cv2.LINE_AA)
        if index % 3 == 0:
            cv2.line(image, (150, y + 10), (300, y + 14), (0, 0, 0), 1, lineType=cv2.LINE_AA)
        if index % 4 == 0:
            cv2.ellipse(
                image,
                (76 + ((index * 17) % 180), 98 + ((index * 21) % 280)),
                (4, 2),
                float((index * 19) % 180),
                0,
                360,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )
    return _encode_png(image)


def _full_body_portrait_filled_feature_line_art_image() -> bytes:
    image = numpy.full((900, 480, 3), 255, dtype=numpy.uint8)
    cv2.ellipse(image, (240, 145), (66, 88), 0, 0, 360, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    for cx in (216, 264):
        cv2.ellipse(image, (cx, 132), (18, 8), 0, 0, 360, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(image, (cx, 132), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (232, 170), (2, 1), 0, 0, 360, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (248, 170), (2, 1), 0, 0, 360, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (250, 560), (140, 260), 0, 0, 360, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    for index in range(280):
        x0 = 40 + ((index * 7) % 380)
        y0 = 230 + ((index * 5) % 620)
        x1 = min(470, x0 + 18 + ((index % 9) * 6))
        y1 = min(890, y0 + ((index % 11) * 5))
        cv2.line(image, (x0, y0), (x1, y1), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return _encode_png(image)


def test_image_router_prefers_simple_outline_for_clean_line_art() -> None:
    decision = route_image_vector_pipeline(_simple_line_art_image())

    assert decision.route == 'simple_outline'
    assert decision.metrics.background_whiteness > 0.8
    assert decision.simple_outline_score > decision.complex_tonal_score


def test_image_router_prefers_complex_tonal_for_shaded_input() -> None:
    decision = route_image_vector_pipeline(_complex_tonal_image())

    assert decision.route == 'complex_tonal'
    assert decision.metrics.entropy > 0.7
    assert decision.complex_tonal_score > decision.simple_outline_score


def test_image_router_prefers_colored_illustration_for_colored_subject() -> None:
    decision = route_image_vector_pipeline(_colored_illustration_image())

    assert decision.route == 'colored_illustration'
    assert decision.metrics.background_whiteness > 0.55
    assert decision.colored_illustration_score > decision.simple_outline_score


def test_image_router_prefers_simple_outline_for_sparse_grayscale_line_art() -> None:
    decision = route_image_vector_pipeline(_detailed_grayscale_line_art_image())

    assert decision.route == 'simple_outline'
    assert decision.metrics.background_whiteness > 0.85
    assert decision.metrics.dark_pixel_ratio < 0.1


def test_simple_outline_branch_emits_canonical_plan_directly() -> None:
    result = vectorize_image_to_canonical_plan(
        _simple_line_art_image(),
        theta_ref=0.0,
        max_strokes=128,
    )

    strokes = canonical_plan_to_draw_strokes(result.plan)
    assert result.route_decision.route == 'simple_outline'
    assert result.branch_details['mode'] in {'simple_outline', 'centerline_outline'}
    assert result.branch_details['stroke_stats']['stroke_count'] > 0
    assert len(strokes) > 0


def test_simple_outline_chooses_a_quality_branch_for_clean_line_art() -> None:
    result = vectorize_image_to_canonical_plan(
        _simple_line_art_image(),
        theta_ref=0.0,
        max_strokes=128,
    )

    assert result.route_decision.route == 'simple_outline'
    assert result.branch_details['mode'] in {'simple_outline', 'centerline_outline'}
    assert result.branch_details['stroke_stats']['stroke_count'] > 0
    assert result.curve_fit_debug is not None
    if result.branch_details['mode'] == 'centerline_outline':
        assert result.branch_details['centerline_count'] > 0
        assert result.curve_fit_debug['trace_mode'] == 'centerline'


def test_simple_outline_prefers_arcs_for_clean_circle_line_art() -> None:
    result = vectorize_image_to_canonical_plan(
        _circle_line_art_image(),
        theta_ref=0.0,
        max_strokes=256,
    )

    primitive_counts = Counter(type(command).__name__ for command in result.plan.commands)
    fit_summary = result.branch_details['curve_fit_summary']
    assert result.route_decision.route == 'simple_outline'
    assert (
        fit_summary['accepted_arcs'] > 0
        or fit_summary['accepted_cubics'] > 0
        or fit_summary['accepted_quadratics'] > 0
    )
    assert (
        primitive_counts['ArcSegment'] > 0
        or primitive_counts['CubicBezier'] > 0
        or primitive_counts['QuadraticBezier'] > 0
    )


def test_simple_outline_emits_curve_primitives_for_curved_petal_line_art() -> None:
    result = vectorize_image_to_canonical_plan(
        _petal_line_art_image(),
        theta_ref=0.0,
        max_strokes=256,
    )

    primitive_counts = Counter(type(command).__name__ for command in result.plan.commands)
    fit_summary = result.branch_details['curve_fit_summary']
    assert result.route_decision.route == 'simple_outline'
    assert (
        fit_summary['accepted_arcs'] > 0
        or fit_summary['accepted_cubics'] > 0
        or fit_summary['accepted_quadratics'] > 0
    )
    assert (
        primitive_counts['ArcSegment'] > 0
        or primitive_counts['CubicBezier'] > 0
        or primitive_counts['QuadraticBezier'] > 0
    )


def test_centerline_branch_merges_fragmented_line_art_strokes() -> None:
    result = vectorize_image_to_canonical_plan(
        _broken_line_art_image(),
        theta_ref=0.0,
        max_strokes=128,
    )

    primitive_counts = Counter(type(command).__name__ for command in result.plan.commands)
    merge_stats = result.branch_details['merge_stats']
    assert result.route_decision.route == 'simple_outline'
    assert result.branch_details['mode'] == 'centerline_outline'
    assert merge_stats['input_centerlines'] >= merge_stats['output_centerlines']
    assert merge_stats['output_centerlines'] == 1
    assert primitive_counts['LineSegment'] == 1
    assert result.curve_fit_debug is not None
    assert result.curve_fit_debug['trace_mode'] == 'centerline'


def test_simple_outline_preserves_faint_line_art_as_drawable_geometry() -> None:
    result = vectorize_image_to_canonical_plan(
        _faint_line_art_image(),
        theta_ref=0.0,
        max_strokes=256,
    )

    primitive_counts = Counter(type(command).__name__ for command in result.plan.commands)
    assert result.route_decision.route == 'simple_outline'
    assert result.branch_details['stroke_stats']['stroke_count'] > 0
    assert (
        primitive_counts['ArcSegment'] > 0
        or primitive_counts['LineSegment'] > 0
        or primitive_counts['CubicBezier'] > 0
    )
    assert result.branch_details['filled_feature_rescue_applied'] is False


def test_simple_outline_rescues_filled_face_features_for_portrait_line_art() -> None:
    result = vectorize_image_to_canonical_plan(
        _portrait_filled_feature_line_art_image(),
        theta_ref=0.0,
        max_strokes=512,
    )

    assert result.route_decision.route == 'simple_outline'
    assert result.branch_details['filled_feature_rescue_applied'] is True
    assert result.branch_details['mode'] == 'simple_outline'
    assert result.branch_details['rescued_feature_count'] >= 2
    assert result.branch_details['rescued_feature_unit_count'] >= 2
    assert result.curve_fit_debug is not None
    assert result.curve_fit_debug['filled_feature_rescue_applied'] is True


def test_simple_outline_rescues_filled_face_features_for_full_body_portrait_line_art() -> None:
    result = vectorize_image_to_canonical_plan(
        _full_body_portrait_filled_feature_line_art_image(),
        theta_ref=0.0,
        max_strokes=1024,
    )

    assert result.route_decision.route == 'simple_outline'
    assert result.branch_details['filled_feature_rescue_applied'] is True
    assert result.branch_details['mode'] == 'simple_outline'
    assert result.branch_details['rescued_feature_count'] >= 2


def test_simple_outline_does_not_enable_filled_feature_rescue_for_dense_decorative_line_art() -> None:
    result = vectorize_image_to_canonical_plan(
        _dense_decorative_line_art_image(),
        theta_ref=0.0,
        max_strokes=768,
    )

    assert result.route_decision.route == 'simple_outline'
    assert result.branch_details['filled_feature_rescue_applied'] is False


def test_complex_tonal_branch_emits_hatch_based_canonical_plan_directly() -> None:
    result = vectorize_image_to_canonical_plan(
        _complex_tonal_image(),
        theta_ref=0.0,
        max_strokes=128,
    )

    strokes = canonical_plan_to_draw_strokes(result.plan)
    assert result.route_decision.route == 'complex_tonal'
    assert result.branch_details['mode'] == 'complex_tonal'
    assert result.branch_details['light_hatch_count'] > 0
    assert result.branch_details['mid_hatch_count'] > 0 or result.branch_details['dark_hatch_count'] > 0
    assert result.branch_details['outline_overlay_count'] >= 0
    assert 'curve_fit_summary' in result.branch_details
    assert len(strokes) > 0


def test_colored_illustration_branch_emits_outline_led_canonical_plan() -> None:
    result = vectorize_image_to_canonical_plan(
        _colored_illustration_image(),
        theta_ref=0.0,
        max_strokes=192,
    )

    primitive_counts = Counter(type(command).__name__ for command in result.plan.commands)
    assert result.route_decision.route == 'colored_illustration'
    assert result.branch_details['mode'] == 'colored_illustration'
    assert result.branch_details['outline_overlay_count'] > 0
    assert primitive_counts['LineSegment'] > 0 or primitive_counts['ArcSegment'] > 0


def test_large_images_are_downscaled_for_processing_but_keep_original_metadata() -> None:
    result = vectorize_image_to_canonical_plan(
        _large_line_art_image(),
        theta_ref=0.0,
        max_strokes=256,
    )

    processing_size = result.branch_details['processing_image_size']
    original_size = result.branch_details['original_image_size']
    assert result.image_size == (1800, 2400)
    assert original_size == {'width_px': 1800, 'height_px': 2400}
    assert processing_size['height_px'] == 1800
    assert processing_size['width_px'] < original_size['width_px']
    assert 0.0 < float(result.branch_details['processing_scale']) < 1.0
