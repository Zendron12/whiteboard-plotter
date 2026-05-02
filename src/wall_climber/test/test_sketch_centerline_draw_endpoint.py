from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from wall_climber import web_server
from wall_climber.canonical_optimizer import optimize_canonical_plan
from wall_climber.canonical_path import CanonicalPathPlan, LineSegment, PenDown, PenUp, TravelMove
from wall_climber.runtime_topics import MODE_DRAW, PEN_MODE_AUTO
from wall_climber.shared_config import load_shared_config


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
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.active_mode = MODE_DRAW
        self.manual_pen_mode = PEN_MODE_AUTO
        self.publish_count = 0
        self.published_plans: list[_FakePrimitivePathPlan] = []

    def runtime_snapshot(self) -> dict:
        return {
            'ready': self.ready,
            'active_mode': self.active_mode,
            'manual_pen_mode': self.manual_pen_mode,
            'statuses': {'cable_executor_status': 'idle'},
            'observed_statuses': {
                'cable_executor_status': True,
                'cable_supervisor_status': True,
            },
        }

    def writable_bounds(self) -> dict[str, float]:
        return {'x_min': 0.0, 'x_max': 6.3, 'y_min': 0.0, 'y_max': 3.0}

    def carriage_safe_writable_bounds(self) -> dict[str, float]:
        return {'x_min': 0.348, 'x_max': 6.2, 'y_min': 0.12, 'y_max': 2.9}

    def carriage_safe_safe_bounds(self) -> dict[str, float]:
        return {'x_min': 0.348, 'x_max': 6.14, 'y_min': 0.22, 'y_max': 2.82}

    def publish_execution_plan(self, primitive_plan, *, allowed_modes):
        if not self.ready:
            raise HTTPException(status_code=503, detail='runtime is not ready')
        if self.active_mode not in allowed_modes:
            raise HTTPException(status_code=409, detail='active mode must be draw')
        self.publish_count += 1
        self.published_plans.append(primitive_plan)
        return {
            'published': 'primitive_path_plan',
            'preferred_transport': 'primitive_path_plan',
            'primitive_transport_published': True,
            'topics': {'primitive_path_plan': '/wall_climber/primitive_path_plan'},
        }


class _FakeRuntime:
    def __init__(self, *, ready: bool = True) -> None:
        self.node = _FakeNode(ready=ready)
        self.web_dir = Path(__file__).resolve().parents[1] / 'web'
        self.last_plan_debug = None
        self.last_execution_debug = None

    def record_last_plan_debug(self, payload: dict) -> None:
        self.last_plan_debug = dict(payload)

    def record_last_execution_debug(self, payload: dict) -> None:
        self.last_execution_debug = dict(payload)

    def record_last_curve_fit_debug(self, _payload: dict) -> None:
        pass


@pytest.fixture(autouse=True)
def _fake_ros_messages(monkeypatch):
    monkeypatch.setattr(web_server, 'BoardPoint', _FakeBoardPoint)
    monkeypatch.setattr(web_server, 'PathPrimitive', _FakePathPrimitive)
    monkeypatch.setattr(web_server, 'PrimitivePathPlan', _FakePrimitivePathPlan)


def _client_and_runtime(*, ready: bool = True) -> tuple[TestClient, _FakeRuntime]:
    runtime = _FakeRuntime(ready=ready)
    return TestClient(web_server.create_app(runtime)), runtime


def _preview(
    client: TestClient,
    *,
    image: bytes | None = None,
    preview_geometry_mode: str = 'smooth_curves',
    curve_tolerance_px: str = '3.0',
    extra_data: dict[str, str] | None = None,
) -> dict:
    data = {
        'preview_geometry_mode': preview_geometry_mode,
        'curve_tolerance_px': curve_tolerance_px,
        'optimization_preset': 'detail',
    }
    if extra_data:
        data.update(extra_data)
    response = client.post(
        '/api/sketch-centerline/preview',
        files={'file': ('sketch.png', image or _curved_sketch_png(), 'image/png')},
        data=data,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['preview_id']
    return payload


def _draw(client: TestClient, preview_id: str, *, optimize_stroke_order: bool | None = None):
    payload = {'preview_id': preview_id}
    if optimize_stroke_order is not None:
        payload['optimize_stroke_order'] = optimize_stroke_order
    return client.post(
        '/api/sketch-centerline/draw',
        json=payload,
    )


def test_preview_response_includes_preview_id() -> None:
    client, runtime = _client_and_runtime()

    payload = _preview(client)

    assert isinstance(payload['preview_id'], str)
    assert payload['preview_id']
    assert runtime.node.publish_count == 0


def test_draw_rejects_missing_unknown_expired_and_extra_preview_id(monkeypatch) -> None:
    client, runtime = _client_and_runtime()

    missing = client.post('/api/sketch-centerline/draw', json={})
    assert missing.status_code == 422

    unknown = _draw(client, 'does-not-exist')
    assert unknown.status_code == 404

    payload = _preview(client)
    extra = client.post(
        '/api/sketch-centerline/draw',
        json={
            'preview_id': payload['preview_id'],
            'svg': '<svg/>',
            'strokes': [[[0, 0], [1, 1]]],
            'optimize_stroke_order': True,
        },
    )
    assert extra.status_code == 422
    assert runtime.node.publish_count == 0

    monkeypatch.setattr(web_server, '_SKETCH_PREVIEW_CACHE_TTL_SECONDS', -1)
    expired = _draw(client, payload['preview_id'])
    assert expired.status_code == 410
    assert runtime.node.publish_count == 0


def test_draw_uses_cached_smooth_canonical_plan() -> None:
    client, runtime = _client_and_runtime()
    payload = _preview(client, preview_geometry_mode='smooth_curves')
    assert payload['metadata']['curve_primitive_count'] >= 1

    response = _draw(client, payload['preview_id'])

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['ok'] is True
    assert body['preview_geometry_mode'] == 'smooth_curves'
    assert body['used_full_cached_plan'] is True
    assert body['optimized'] is True
    assert body['canonical_command_count'] >= 1
    assert body['primitive_count'] >= 1
    assert 'travel_reduction_m' in body['optimization']
    assert runtime.node.publish_count == 1
    primitive_types = [primitive.type for primitive in runtime.node.published_plans[0].primitives]
    assert _FakePathPrimitive.QUADRATIC_BEZIER in primitive_types or _FakePathPrimitive.CUBIC_BEZIER in primitive_types


def test_draw_uses_cached_polyline_canonical_plan() -> None:
    client, runtime = _client_and_runtime()
    payload = _preview(client, preview_geometry_mode='polyline')
    assert payload['metadata']['curve_primitive_count'] == 0

    response = _draw(client, payload['preview_id'])

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['preview_geometry_mode'] == 'polyline'
    assert body['used_full_cached_plan'] is True
    primitive_types = [primitive.type for primitive in runtime.node.published_plans[0].primitives]
    assert _FakePathPrimitive.LINE_SEGMENT in primitive_types
    assert _FakePathPrimitive.QUADRATIC_BEZIER not in primitive_types
    assert _FakePathPrimitive.CUBIC_BEZIER not in primitive_types


def test_oversized_cached_plan_returns_structured_413(monkeypatch) -> None:
    client, runtime = _client_and_runtime()
    payload = _preview(client, image=_simple_sketch_png(), preview_geometry_mode='polyline')
    monkeypatch.setattr(web_server, '_SKETCH_DRAW_MAX_PRIMITIVES', 1)

    response = _draw(client, payload['preview_id'])

    assert response.status_code == 413
    detail = response.json()['detail']
    assert detail['error'] == 'sketch preview plan is too large for the existing execution transport'
    assert detail['counts']['canonical_command_count'] >= 1
    assert detail['counts']['primitive_count'] > 1
    assert detail['limits']['max_primitive_count'] == 1
    assert runtime.node.publish_count == 0


def test_default_safe_fit_preview_draws_without_bounds_failure() -> None:
    client, runtime = _client_and_runtime()
    payload = _preview(client, preview_geometry_mode='polyline')
    bounds = payload['bounds']
    metadata = payload['metadata']
    assert bounds['x_min'] >= metadata['safe_x_min']
    assert bounds['x_max'] <= metadata['safe_x_max']
    assert bounds['y_min'] >= metadata['safe_y_min']
    assert bounds['y_max'] <= metadata['safe_y_max']

    response = _draw(client, payload['preview_id'])

    assert response.status_code == 200, response.text
    assert runtime.node.publish_count == 1


def test_draw_uses_full_cached_plan_when_canvas_preview_is_truncated(monkeypatch) -> None:
    monkeypatch.setattr(web_server, '_SKETCH_PREVIEW_MAX_POINTS', 2)
    client, runtime = _client_and_runtime()
    payload = _preview(client, preview_geometry_mode='polyline')
    assert payload['preview']['truncated'] is True
    assert payload['preview']['returned_point_count'] <= 2

    response = _draw(client, payload['preview_id'], optimize_stroke_order=False)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body['used_full_cached_plan'] is True
    assert body['cached_canonical_command_count'] == payload['canonical_command_count']
    assert body['primitive_count'] > payload['preview']['returned_point_count']
    assert runtime.node.publish_count == 1


def test_sketch_draw_optimization_reduces_travel_without_changing_segments() -> None:
    plan = CanonicalPathPlan(
        frame='board',
        theta_ref=0.0,
        commands=(
            PenDown(),
            LineSegment(start=(0.4, 0.4), end=(0.6, 0.4)),
            PenUp(),
            TravelMove(start=(0.6, 0.4), end=(5.6, 2.4)),
            PenDown(),
            LineSegment(start=(5.6, 2.4), end=(5.8, 2.4)),
            PenUp(),
            TravelMove(start=(5.8, 2.4), end=(0.7, 0.5)),
            PenDown(),
            LineSegment(start=(0.7, 0.5), end=(0.9, 0.5)),
            PenUp(),
        ),
    )

    result = optimize_canonical_plan(
        plan,
        policy=web_server._sketch_draw_optimization_policy(load_shared_config()),
    )
    original_segments = sorted(
        (command.start, command.end)
        for command in plan.commands
        if isinstance(command, LineSegment)
    )
    optimized_segments = sorted(
        tuple(sorted((command.start, command.end)))
        for command in result.plan.commands
        if isinstance(command, LineSegment)
    )
    original_segments_unoriented = sorted(tuple(sorted(segment)) for segment in original_segments)

    assert result.stats.optimized_travel_length_m < result.stats.original_travel_length_m
    assert result.stats.reordered_units is True
    assert optimized_segments == original_segments_unoriented


def test_runtime_not_ready_does_not_discard_cached_preview() -> None:
    client, runtime = _client_and_runtime(ready=False)
    payload = _preview(client, preview_geometry_mode='polyline')

    first = _draw(client, payload['preview_id'])
    assert first.status_code == 503
    assert runtime.node.publish_count == 0

    runtime.node.ready = True
    second = _draw(client, payload['preview_id'])
    assert second.status_code == 200, second.text
    assert runtime.node.publish_count == 1
