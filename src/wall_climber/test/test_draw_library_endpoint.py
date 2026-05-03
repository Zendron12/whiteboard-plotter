from __future__ import annotations

from pathlib import Path
import json

import cv2  # type: ignore
import numpy
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from wall_climber import web_server
from wall_climber.runtime_topics import MODE_DRAW, PEN_MODE_AUTO


def _encode_png(image: numpy.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return bytes(encoded.tobytes())


def _simple_sketch_png() -> bytes:
    image = numpy.full((96, 160, 3), 255, dtype=numpy.uint8)
    cv2.line(image, (18, 48), (142, 48), (0, 0, 0), 5, lineType=cv2.LINE_AA)
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
    def __init__(self, share_dir: Path, *, ready: bool = True) -> None:
        self.node = _FakeNode(ready=ready)
        self.share_dir = share_dir
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


def _client_and_runtime(tmp_path: Path, *, ready: bool = True) -> tuple[TestClient, _FakeRuntime, Path]:
    share_dir = tmp_path / 'share' / 'wall_climber'
    library_dir = share_dir / 'assets' / 'draw_library'
    library_dir.mkdir(parents=True)
    runtime = _FakeRuntime(share_dir, ready=ready)
    return TestClient(web_server.create_app(runtime)), runtime, library_dir


def test_draw_library_unknown_id_returns_clear_error(tmp_path: Path) -> None:
    client, runtime, _library_dir = _client_and_runtime(tmp_path)

    response = client.post('/api/draw-library/draw', json={'id': 99})

    assert response.status_code == 404
    assert 'was not found' in response.json()['detail']
    assert runtime.node.publish_count == 0


def test_draw_library_unsupported_manifest_file_type(tmp_path: Path) -> None:
    client, runtime, library_dir = _client_and_runtime(tmp_path)
    (library_dir / '1.svg').write_text('<svg/>', encoding='utf-8')
    (library_dir / 'manifest.json').write_text(
        json.dumps({'entries': [{'id': 1, 'file': '1.svg'}]}),
        encoding='utf-8',
    )

    response = client.post('/api/draw-library/draw', json={'id': 1})

    assert response.status_code == 415
    assert 'PNG or JPG' in response.json()['detail']
    assert runtime.node.publish_count == 0


def test_draw_library_valid_png_publishes_backend_owned_plan(tmp_path: Path) -> None:
    client, runtime, library_dir = _client_and_runtime(tmp_path)
    (library_dir / '1.png').write_bytes(_simple_sketch_png())

    response = client.post('/api/draw-library/draw', json={'id': 1})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['ok'] is True
    assert payload['published'] is True
    assert payload['source_type'] == 'draw_library'
    assert payload['id'] == 1
    assert payload['file'] == '1.png'
    assert payload['preview_geometry_mode'] == 'smooth_curves'
    assert payload['used_backend_owned_plan'] is True
    assert payload['primitive_count'] > 0
    assert runtime.node.publish_count == 1
    assert runtime.node.published_plans[0].primitives


def test_draw_library_runtime_not_ready_is_clear_and_does_not_publish(tmp_path: Path) -> None:
    client, runtime, library_dir = _client_and_runtime(tmp_path, ready=False)
    (library_dir / '1.png').write_bytes(_simple_sketch_png())

    response = client.post('/api/draw-library/draw', json={'id': 1})

    assert response.status_code == 503
    assert 'runtime is not ready' in response.json()['detail']
    assert runtime.node.publish_count == 0

