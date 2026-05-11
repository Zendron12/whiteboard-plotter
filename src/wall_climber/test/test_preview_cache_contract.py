from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from wall_climber import web_server
from wall_climber.runtime_topics import MODE_DRAW, MODE_TEXT, PEN_MODE_AUTO


class _FakeBoardPoint:
    def __init__(self, *, x: float = 0.0, y: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)


class _FakePathPrimitive:
    PEN_UP = 1
    PEN_DOWN = 2
    TRAVEL_MOVE = 3
    LINE_SEGMENT = 4
    ARC_SEGMENT = 5
    QUADRATIC_BEZIER = 6
    CUBIC_BEZIER = 7

    def __init__(self) -> None:
        self.type = 0
        self.start = _FakeBoardPoint()
        self.end = _FakeBoardPoint()
        self.control1 = _FakeBoardPoint()
        self.control2 = _FakeBoardPoint()
        self.center = _FakeBoardPoint()
        self.radius = 0.0
        self.start_angle_rad = 0.0
        self.sweep_angle_rad = 0.0
        self.clockwise = False
        self.pen_down = False


class _FakePrimitivePathPlan:
    def __init__(self) -> None:
        self.frame = ''
        self.theta_ref = 0.0
        self.primitives: list[_FakePathPrimitive] = []


class _FakeNode:
    def __init__(self) -> None:
        self.active_mode = MODE_DRAW
        self.manual_pen_mode = PEN_MODE_AUTO
        self.publish_count = 0
        self.published_plans: list[_FakePrimitivePathPlan] = []

    def carriage_safe_writable_bounds(self) -> dict[str, float]:
        return {'x_min': 0.348, 'x_max': 6.2, 'y_min': 0.12, 'y_max': 2.9}

    def carriage_safe_safe_bounds(self) -> dict[str, float]:
        return {'x_min': 0.348, 'x_max': 6.14, 'y_min': 0.22, 'y_max': 2.82}

    def publish_execution_plan(self, primitive_plan, *, allowed_modes):
        if self.active_mode not in allowed_modes:
            raise HTTPException(status_code=409, detail='active mode must be draw')
        if self.manual_pen_mode != PEN_MODE_AUTO:
            raise HTTPException(status_code=409, detail='manual arm test must be auto')
        self.publish_count += 1
        self.published_plans.append(primitive_plan)
        return {
            'published': 'primitive_path_plan',
            'preferred_transport': 'primitive_path_plan',
            'primitive_transport_published': True,
            'topics': {'primitive_path_plan': '/wall_climber/primitive_path_plan'},
        }


class _FakeRuntime:
    def __init__(self) -> None:
        self.node = _FakeNode()
        self.web_dir = Path(__file__).resolve().parents[1] / 'web'
        self.last_plan_debug = None
        self.last_execution_debug = None
        self.last_curve_fit_debug = None
        self.uploads: dict[str, tuple[dict, bytes]] = {}

    def record_last_plan_debug(self, payload: dict) -> None:
        self.last_plan_debug = dict(payload)

    def record_last_execution_debug(self, payload: dict) -> None:
        self.last_execution_debug = dict(payload)

    def record_last_curve_fit_debug(self, payload: dict) -> None:
        self.last_curve_fit_debug = dict(payload)

    def load_upload(self, upload_id: str) -> tuple[dict, bytes]:
        try:
            metadata, payload = self.uploads[upload_id]
        except KeyError:
            raise HTTPException(status_code=404, detail='upload_id was not found')
        return dict(metadata), payload

    def upload_processing_snapshot(self, upload_id: str, *, metadata=None, payload=None) -> dict:
        metadata = dict(metadata or self.uploads[upload_id][0])
        return {
            'upload_id': upload_id,
            'source_type': metadata.get('source_type', 'image'),
            'state': 'ready',
            'stage': 'ready',
            'progress': 1.0,
            'message': 'Vector preview is ready.',
            'image_size': metadata.get('image_size'),
            'route': None,
            'timings_ms': {},
            'curve_fit_summary': {},
        }

    def prepared_image_artifact(self, upload_id: str, *, metadata=None, payload=None):
        return None


@pytest.fixture(autouse=True)
def _fake_ros_messages(monkeypatch):
    monkeypatch.setattr(web_server, 'BoardPoint', _FakeBoardPoint)
    monkeypatch.setattr(web_server, 'PathPrimitive', _FakePathPrimitive)
    monkeypatch.setattr(web_server, 'PrimitivePathPlan', _FakePrimitivePathPlan)


def _client_and_runtime() -> tuple[TestClient, _FakeRuntime]:
    runtime = _FakeRuntime()
    return TestClient(web_server.create_app(runtime)), runtime


def _encode_png(image: numpy.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return bytes(encoded.tobytes())


def _simple_line_art_png() -> bytes:
    image = numpy.full((120, 160, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (20, 60), (140, 60), (0, 0, 0), 4, lineType=cv2.LINE_AA)
    cv2.circle(image, (80, 60), 22, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return _encode_png(image)


def _simple_colored_diagram_png() -> bytes:
    image = numpy.full((160, 220, 3), (225, 245, 250), dtype=numpy.uint8)
    cv2.rectangle(image, (16, 104), (110, 150), (215, 190, 130), -1)
    cv2.rectangle(image, (16, 104), (110, 150), (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.circle(image, (44, 42), 24, (0, 220, 255), -1)
    cv2.circle(image, (44, 42), 24, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (150, 55), (42, 20), 0, 0, 360, (245, 245, 245), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(image, (150, 55), (42, 20), 0, 0, 360, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.line(image, (134, 95), (194, 138), (0, 0, 0), 3, lineType=cv2.LINE_AA)
    return _encode_png(image)


def _preview_svg(client: TestClient) -> dict:
    response = client.post(
        '/api/preview',
        json={
            'input_type': 'svg',
            'svg': '<svg viewBox="0 0 100 60"><path d="M10 30 C30 5 70 55 90 30"/></svg>',
            'placement': {'x': 3.2, 'y': 1.5, 'scale': 0.7},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['preview_id']
    assert payload['canonical_hash']
    assert payload['preview']['canonical_hash'] == payload['canonical_hash']
    assert payload['primitive_hash']
    assert payload['execution_hash']
    return payload


def _preview_text(client: TestClient) -> dict:
    response = client.post(
        '/api/preview',
        json={
            'input_type': 'text',
            'text': 'HELLO',
            'placement': {'x': 0.55, 'y': 0.45, 'scale': 0.8},
            'settings': {'font_source': 'relief_singleline'},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['preview_id']
    assert payload['canonical_hash']
    assert payload['primitive_hash']
    assert payload['execution_hash']
    return payload


def _preview_image(client: TestClient, runtime: _FakeRuntime) -> dict:
    payload = _simple_line_art_png()
    response = client.post(
        '/api/preview',
        files={'file': ('line.png', payload, 'image/png')},
        data={
            'input_type': 'auto',
            'settings_json': '{"preview_geometry_mode":"polyline","max_image_dim":600}',
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['preview_id']
    assert body['canonical_hash']
    assert body['pipeline_mode'] == 'sketch_centerline'
    assert body['metadata']['sketch_extraction_method'] == 'adaptive'
    assert body['primitive_hash']
    assert body['execution_hash']
    return body


def _preview_colored_image(client: TestClient, *, color_lineart_method: str = 'auto_outline') -> dict:
    payload = _simple_colored_diagram_png()
    response = client.post(
        '/api/preview',
        files={'file': ('diagram.png', payload, 'image/png')},
        data={
            'input_type': 'colored_image',
            'settings_json': (
                '{"preview_geometry_mode":"polyline","max_image_dim":600,'
                f'"color_lineart_method":"{color_lineart_method}"' + '}'
            ),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['preview_id']
    assert body['pipeline_mode'] == 'local_outline_adaptive_centerline'
    assert body['input_type'] == 'colored_image'
    assert body['metadata']['color_lineart_method'] == color_lineart_method
    assert body['converted_lineart_preview']['data_url'].startswith('data:image/png;base64,')
    assert body['primitive_hash']
    assert body['execution_hash']
    return body


def test_auto_colored_raster_uses_local_outline_pipeline() -> None:
    client, _runtime = _client_and_runtime()
    response = client.post(
        '/api/preview',
        files={'file': ('diagram.png', _simple_colored_diagram_png(), 'image/png')},
        data={
            'input_type': 'auto',
            'settings_json': '{"preview_geometry_mode":"polyline","max_image_dim":600,"color_lineart_method":"auto_outline"}',
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['input_type'] == 'colored_image'
    assert body['pipeline_mode'] == 'local_outline_adaptive_centerline'
    assert body['metadata']['input_detection']['input_type'] == 'colored_image'
    assert body['converted_lineart_preview']['quality'] in {'good', 'noisy', 'complex'}


def test_photo_diagram_edges_preview_contract_and_settings_hash() -> None:
    client, _runtime = _client_and_runtime()
    auto = _preview_colored_image(client, color_lineart_method='auto_outline')
    photo_edges = _preview_colored_image(client, color_lineart_method='photo_diagram_edges')

    assert photo_edges['metadata']['color_lineart_method'] == 'photo_diagram_edges'
    assert photo_edges['metadata']['effective_color_lineart_method'] == 'photo_diagram_edges'
    assert photo_edges['metadata']['canny_lower_threshold'] < photo_edges['metadata']['canny_upper_threshold']
    assert photo_edges['metadata']['edge_pixel_ratio'] > 0
    assert photo_edges['converted_lineart_preview']['method'] == 'photo_diagram_edges'
    assert photo_edges['converted_lineart_preview']['metadata']['canny_lower_threshold'] < (
        photo_edges['converted_lineart_preview']['metadata']['canny_upper_threshold']
    )
    assert photo_edges['metrics']['color_lineart']['color_lineart_method'] == 'photo_diagram_edges'
    assert photo_edges['metrics']['color_lineart']['edge_pixel_ratio'] > 0
    assert photo_edges['execution_preview_svg'].startswith('<svg')
    assert photo_edges['settings_hash'] != auto['settings_hash']


def _preview_sketch(client: TestClient) -> dict:
    response = client.post(
        '/api/preview',
        files={'file': ('line.png', _simple_line_art_png(), 'image/png')},
        data={
            'input_type': 'auto',
            'preview_geometry_mode': 'polyline',
            'optimization_preset': 'detail',
            'max_image_dim': '600',
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['preview_id']
    assert body['canonical_hash']
    assert body['pipeline_mode'] == 'sketch_centerline'
    assert body['metadata']['sketch_extraction_method'] == 'adaptive'
    assert body['primitive_hash']
    assert body['execution_hash']
    return body


def test_svg_preview_draw_uses_same_cached_canonical_hash(monkeypatch) -> None:
    client, runtime = _client_and_runtime()
    preview = _preview_svg(client)

    def _reject_rebuild(*_args, **_kwargs):
        raise AssertionError('draw must use the cached CanonicalPathPlan')

    monkeypatch.setattr(web_server, 'vectorize_svg', _reject_rebuild)

    response = client.post('/api/draw', json={'preview_id': preview['preview_id']})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['published'] is True
    assert body['used_cached_preview_plan'] is True
    assert body['canonical_hash'] == preview['canonical_hash']
    assert body['preview_draw_hash_match'] is True
    assert runtime.node.publish_count == 1


def test_text_preview_draw_uses_same_cached_canonical_hash(monkeypatch) -> None:
    client, runtime = _client_and_runtime()
    preview = _preview_text(client)
    runtime.node.active_mode = MODE_TEXT

    def _reject_rebuild(*_args, **_kwargs):
        raise AssertionError('draw must use the cached text CanonicalPathPlan')

    monkeypatch.setattr(web_server, 'vectorize_text_grouped', _reject_rebuild)

    response = client.post('/api/draw', json={'preview_id': preview['preview_id']})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['source_type'] == 'text'
    assert body['canonical_hash'] == preview['canonical_hash']
    assert body['preview_draw_hash_match'] is True
    assert runtime.node.publish_count == 1


def test_image_preview_draw_uses_same_cached_canonical_hash(monkeypatch) -> None:
    client, runtime = _client_and_runtime()
    preview = _preview_image(client, runtime)

    def _reject_rebuild(*_args, **_kwargs):
        raise AssertionError('draw must use the cached image CanonicalPathPlan')

    monkeypatch.setattr(web_server, 'vectorize_image_to_canonical_plan', _reject_rebuild)

    response = client.post('/api/draw', json={'preview_id': preview['preview_id']})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['source_type'] == 'sketch_centerline'
    assert body['canonical_hash'] == preview['canonical_hash']
    assert body['preview_draw_hash_match'] is True
    assert runtime.node.publish_count == 1


def test_sketch_preview_draw_uses_same_cached_canonical_hash(monkeypatch) -> None:
    client, runtime = _client_and_runtime()
    preview = _preview_sketch(client)

    def _reject_rebuild(*_args, **_kwargs):
        raise AssertionError('draw must use the cached sketch CanonicalPathPlan')

    monkeypatch.setattr(web_server, 'vectorize_sketch_image_to_plan', _reject_rebuild)

    response = client.post('/api/draw', json={'preview_id': preview['preview_id']})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['source_type'] == 'sketch_centerline'
    assert body['canonical_hash'] == preview['canonical_hash']
    assert body['preview_draw_hash_match'] is True
    assert runtime.node.publish_count == 1


def test_colored_preview_uses_local_outline_and_draw_uses_cached_payload(monkeypatch) -> None:
    client, runtime = _client_and_runtime()
    preview = _preview_colored_image(client)

    def _reject_color_conversion(*_args, **_kwargs):
        raise AssertionError('draw must not rerun local outline conversion')

    def _reject_vectorize(*_args, **_kwargs):
        raise AssertionError('draw must not rerun sketch vectorization')

    monkeypatch.setattr(web_server, 'convert_color_image_to_lineart', _reject_color_conversion)
    monkeypatch.setattr(web_server, 'vectorize_sketch_image_to_plan', _reject_vectorize)

    response = client.post('/api/draw', json={'preview_id': preview['preview_id']})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['primitive_hash'] == preview['primitive_hash']
    assert body['execution_hash'] == preview['execution_hash']
    assert body['preview_draw_hash_match'] is True
    assert runtime.node.publish_count == 1


def test_forced_sketch_image_bypasses_color_lineart_conversion(monkeypatch) -> None:
    client, _runtime = _client_and_runtime()

    def _reject_color_conversion(*_args, **_kwargs):
        raise AssertionError('forced sketch_image must not call color conversion')

    monkeypatch.setattr(web_server, 'convert_color_image_to_lineart', _reject_color_conversion)
    response = client.post(
        '/api/preview',
        files={'file': ('diagram.png', _simple_colored_diagram_png(), 'image/png')},
        data={
            'input_type': 'sketch_image',
            'settings_json': '{"preview_geometry_mode":"polyline","max_image_dim":600,"color_lineart_method":"auto_outline"}',
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['pipeline_mode'] == 'sketch_centerline'
    assert body['input_type'] == 'sketch_image'
    assert 'converted_lineart_preview' not in body


def test_invalid_raster_upload_returns_clear_bad_request() -> None:
    client, _runtime = _client_and_runtime()
    response = client.post(
        '/api/preview',
        files={'file': ('bad.png', b'not-a-real-image', 'image/png')},
        data={
            'input_type': 'auto',
            'settings_json': '{"preview_geometry_mode":"polyline","max_image_dim":600}',
        },
    )

    assert response.status_code == 400
    assert 'Unable to decode uploaded image' in response.json()['detail']


def test_preview_metrics_split_canonical_and_executable_geometry() -> None:
    client, _runtime = _client_and_runtime()
    preview = _preview_sketch(client)
    metrics = preview['metrics']

    assert 'canonical_geometry' in metrics
    assert 'executable_geometry' in metrics
    assert set(metrics['canonical_geometry']).issuperset(
        {'line_count', 'quadratic_count', 'cubic_count', 'arc_count', 'total_curve_count'}
    )
    assert set(metrics['executable_geometry']).issuperset(
        {'draw_path_count', 'sampled_point_count', 'sampled_segment_count'}
    )
    assert metrics['executable_geometry']['sampled_point_count'] == metrics['draw_sample_count']


def test_compare_methods_returns_per_engine_results_without_selecting_active_preview() -> None:
    client, _runtime = _client_and_runtime()

    response = client.post(
        '/api/preview/compare',
        files={'file': ('line.png', _simple_line_art_png(), 'image/png')},
        data={
            'settings_json': '{"preview_geometry_mode":"polyline","max_image_dim":600}',
            'engines_json': '["internal_centerline","autotrace_centerline"]',
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['active_preview_mutated'] is False
    results = payload['results']
    assert [result['engine_name'] for result in results] == ['internal_centerline', 'autotrace_centerline']
    assert results[0]['available'] is True
    assert results[0]['preview_id']
    assert results[0]['execution_preview_svg'].startswith('<svg')
    assert 'canonical_geometry' in results[0]
    assert 'executable_geometry' in results[0]
    assert 'available' in results[1]


def test_draw_with_preview_id_uses_cached_svg_without_rebuilding(monkeypatch) -> None:
    client, runtime = _client_and_runtime()
    preview = _preview_svg(client)

    def _reject_rebuild(*_args, **_kwargs):
        raise AssertionError('draw with preview_id must use the cached plan')

    monkeypatch.setattr(web_server, 'vectorize_svg', _reject_rebuild)

    response = client.post('/api/draw', json={'preview_id': preview['preview_id']})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['canonical_hash'] == preview['canonical_hash']
    assert body['preview_draw_hash_match'] is True
    assert runtime.node.publish_count == 1


def test_cached_preview_draw_rejects_missing_and_expired_preview_id(monkeypatch) -> None:
    client, runtime = _client_and_runtime()

    missing = client.post('/api/draw', json={})
    assert missing.status_code == 400

    preview = _preview_svg(client)
    monkeypatch.setattr(web_server, '_PREVIEW_CACHE_TTL_SECONDS', -1)

    expired = client.post('/api/draw', json={'preview_id': preview['preview_id']})

    assert expired.status_code == 410
    assert 'expired' in expired.json()['detail']
    assert runtime.node.publish_count == 0
