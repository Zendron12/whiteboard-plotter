from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy
from fastapi.testclient import TestClient

from wall_climber import web_server


def _encode_png(image: numpy.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return bytes(encoded.tobytes())


def _simple_sketch_png() -> bytes:
    image = numpy.full((100, 180, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 50), (160, 50), (0, 0, 0), 5, lineType=cv2.LINE_AA)
    return _encode_png(image)


def _curved_sketch_png() -> bytes:
    image = numpy.full((140, 220, 3), 255, dtype=numpy.uint8)
    points = numpy.array(
        [[20, 105], [45, 60], [85, 35], [135, 35], [180, 70], [205, 110]],
        dtype=numpy.int32,
    )
    cv2.polylines(image, [points], False, (0, 0, 0), 5, lineType=cv2.LINE_AA)
    return _encode_png(image)


class _FakeNode:
    def __init__(self) -> None:
        self.publish_count = 0

    def publish_execution_plan(self, *_args, **_kwargs):
        self.publish_count += 1
        raise AssertionError('preview endpoint must not publish robot commands')


class _FakeRuntime:
    def __init__(self) -> None:
        self.node = _FakeNode()
        self.web_dir = Path(__file__).resolve().parents[1] / 'web'


def _client_and_runtime() -> tuple[TestClient, _FakeRuntime]:
    runtime = _FakeRuntime()
    return TestClient(web_server.create_app(runtime)), runtime


def test_sketch_centerline_preview_endpoint_accepts_png_upload() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('line.png', _simple_sketch_png(), 'image/png')},
        data={
            'optimization_preset': 'custom',
            'margin_m': '0.1',
            'line_sensitivity': '0.35',
            'min_stroke_length_px': '1.5',
            'merge_gap_px': '5.0',
            'merge_max_angle_deg': '75',
            'scale_percent': '80',
            'center_x_m': '3.0',
            'center_y_m': '1.5',
            'preview_geometry_mode': 'smooth_curves',
            'curve_tolerance_px': '1.25',
            'max_image_dim': '1000',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['ok'] is True
    assert payload['mode'] == 'sketch_centerline'
    assert payload['stroke_count'] >= 1
    assert payload['point_count'] >= 2
    assert payload['canonical_command_count'] >= 3
    assert payload['metrics']['stroke_count'] == payload['stroke_count']
    assert payload['metadata']['optimization_preset'] == 'custom'
    assert payload['metadata']['line_sensitivity'] == 0.35
    assert payload['metadata']['min_stroke_length_px'] == 1.5
    assert payload['metadata']['merge_gap_px'] == 5.0
    assert payload['metadata']['merge_max_angle_deg'] == 75.0
    assert payload['metadata']['effective_min_stroke_length_px'] == 1.5
    assert payload['metadata']['effective_merge_gap_px'] == 5.0
    assert payload['metadata']['effective_simplify_epsilon_px'] >= 0.0
    assert payload['metadata']['scale_percent'] == 80.0
    assert payload['metadata']['center_x_m'] == 3.0
    assert payload['metadata']['center_y_m'] == 1.5
    assert payload['metadata']['preview_geometry_mode'] == 'smooth_curves'
    assert payload['metadata']['curve_tolerance_px'] == 1.25
    assert payload['metadata']['max_image_dim'] == 1000
    assert payload['metadata']['timing']['curve_fit_time_ms'] >= 0.0
    assert payload['metadata']['line_primitive_count'] >= 1
    assert 'effective_threshold_value' in payload['metadata']
    assert 'timing' in payload['metadata']
    assert 'merge_count' in payload['metadata']
    assert 'removed_short_stroke_count' in payload['metadata']
    assert 'fitted_width_m' in payload['metadata']
    assert payload['metadata']['skeleton_backend'] in {
        'skimage.morphology.skeletonize',
        'cv2.ximgproc.thinning',
    }
    assert payload['bounds']['x_min'] >= 0.0
    assert payload['bounds']['x_max'] <= 6.3
    assert payload['preview_svg'].startswith('<svg')
    assert 'viewBox="0 0 6.3 3"' in payload['preview_svg']
    assert payload['preview']['strokes']
    assert payload['preview']['max_points'] > 0
    assert payload['preview']['returned_point_count'] <= payload['preview']['max_points']
    assert payload['preview']['original_point_count'] == payload['point_count']
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_returns_smooth_curve_svg() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('curve.png', _curved_sketch_png(), 'image/png')},
        data={
            'preview_geometry_mode': 'smooth_curves',
            'curve_tolerance_px': '3.0',
            'optimization_preset': 'detail',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['metadata']['preview_geometry_mode'] == 'smooth_curves'
    assert payload['metadata']['curve_primitive_count'] >= 1
    assert ' Q ' in payload['preview_svg'] or ' C ' in payload['preview_svg']
    assert '<polyline' not in payload['preview_svg']
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_keeps_polyline_debug() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('curve.png', _curved_sketch_png(), 'image/png')},
        data={'preview_geometry_mode': 'polyline'},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['metadata']['preview_geometry_mode'] == 'polyline'
    assert payload['metadata']['curve_primitive_count'] == 0
    assert '<polyline' in payload['preview_svg']
    assert ' Q ' not in payload['preview_svg']
    assert ' C ' not in payload['preview_svg']
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_rejects_unknown_geometry_mode() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('line.png', _simple_sketch_png(), 'image/png')},
        data={'preview_geometry_mode': 'magic'},
    )

    assert response.status_code == 422
    assert 'preview_geometry_mode' in response.json()['detail']
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_accepts_optimization_preset() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('line.png', _simple_sketch_png(), 'image/png')},
        data={'optimization_preset': 'fast', 'merge_gap_px': '0', 'simplify_epsilon_px': '0'},
    )

    assert response.status_code == 200
    metadata = response.json()['metadata']
    assert metadata['optimization_preset'] == 'fast'
    assert metadata['merge_enabled'] is True
    assert metadata['effective_merge_gap_px'] == 5.0
    assert metadata['effective_simplify_epsilon_px'] == 1.5
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_rejects_unknown_preset() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('line.png', _simple_sketch_png(), 'image/png')},
        data={'optimization_preset': 'turbo'},
    )

    assert response.status_code == 422
    assert 'optimization_preset' in response.json()['detail']
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_rejects_outside_placement() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('line.png', _simple_sketch_png(), 'image/png')},
        data={'scale_percent': '100', 'center_x_m': '0', 'center_y_m': '1.5'},
    )

    assert response.status_code == 422
    assert 'outside the board bounds' in response.json()['detail']
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_caps_preview_points(monkeypatch) -> None:
    monkeypatch.setattr(web_server, '_SKETCH_PREVIEW_MAX_POINTS', 2)
    client, _runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('curve.png', _curved_sketch_png(), 'image/png')},
        data={'preview_geometry_mode': 'smooth_curves', 'curve_tolerance_px': '3.0'},
    )

    assert response.status_code == 200
    payload = response.json()
    preview = payload['preview']
    assert preview['max_points'] == 2
    assert preview['returned_point_count'] <= 2
    assert preview['truncated'] == (preview['original_point_count'] > preview['returned_point_count'])
    assert payload['metadata']['preview_geometry_mode'] == 'smooth_curves'
    assert payload['metadata']['curve_primitive_count'] >= 1
    assert ' Q ' in payload['preview_svg'] or ' C ' in payload['preview_svg']
    assert '<polyline' not in payload['preview_svg']


def test_sketch_centerline_preview_endpoint_rejects_invalid_file_type() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('notes.txt', b'not an image', 'text/plain')},
    )

    assert response.status_code == 415
    assert 'PNG or JPG' in response.json()['detail']
    assert runtime.node.publish_count == 0


def test_sketch_centerline_preview_endpoint_rejects_invalid_image_bytes() -> None:
    client, runtime = _client_and_runtime()

    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('bad.png', b'not an image', 'image/png')},
    )

    assert response.status_code == 422
    assert 'decode' in response.json()['detail']
    assert runtime.node.publish_count == 0
