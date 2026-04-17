from __future__ import annotations

import json
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy
import rclpy
import uvicorn
from ament_index_python.packages import get_package_share_directory
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from geometry_msgs.msg import Point
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from wall_climber_interfaces.msg import DrawPlan, DrawPolyline

from wall_climber.runtime_topics import (
    ACTIVE_MODE_TOPIC,
    CABLE_EXECUTOR_STATUS_TOPIC,
    CABLE_SUPERVISOR_STATUS_TOPIC,
    DRAW_PLAN_TOPIC,
    MANUAL_PEN_MODE_TOPIC,
    MODE_DRAW,
    MODE_OFF,
    MODE_TEXT,
    PEN_MODE_AUTO,
    PEN_MODE_DOWN,
    PEN_MODE_UP,
    VALID_MODES,
    VALID_MANUAL_PEN_MODES,
)
from wall_climber.shared_config import load_shared_config
from wall_climber.vector_pipeline import (
    DrawPathSegment,
    TextGlyphOutline,
    VectorPlacement,
    cleanup_draw_strokes,
    default_placement,
    draw_plan_to_dict,
    draw_segments_from_pen_strokes,
    normalize_placement,
    normalize_text_plan_input,
    place_grouped_text_on_board,
    place_draw_strokes_on_board,
    stroke_stats,
    strokes_to_draw_plan,
    trace_line_art_image,
    vectorize_svg,
    vectorize_text_grouped,
)


_ACTIVE_MODE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
_STATUS_TOPIC_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

_MAX_TEXT_CHARS = 400
_MAX_TEXT_BYTES = 4096
_MAX_DRAW_PLAN_BYTES = 256 * 1024
_MAX_DRAW_STROKES = 256
_MAX_POINTS_PER_STROKE = 2048
_MAX_TOTAL_POINTS = 8192
_SEGMENT_EPS_M = 1.0e-4
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_MAX_SVG_BYTES = 256 * 1024
_MAX_VECTOR_REQUEST_BYTES = 512 * 1024
_ALLOWED_UPLOAD_TYPES = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/webp': '.webp',
}
_ALLOWED_UPLOAD_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
_REQUIRED_STATUS_KEYS = (
    'cable_executor_status',
    'cable_supervisor_status',
)


class WebBackendNode(Node):
    def __init__(self) -> None:
        super().__init__('web_ui_server')
        self._shared = load_shared_config()
        self.declare_parameter('port', 8080)
        self.declare_parameter('initial_mode', MODE_OFF)
        self.declare_parameter('enable_webots_trail', False)
        self.declare_parameter('open_browser', False)

        self.port = int(self.get_parameter('port').value)
        self.enable_webots_trail = bool(self.get_parameter('enable_webots_trail').value)
        self.open_browser = bool(self.get_parameter('open_browser').value)
        requested_mode = str(self.get_parameter('initial_mode').value).strip().lower()
        if requested_mode not in VALID_MODES:
            self.get_logger().warn(
                f'Invalid initial_mode {requested_mode!r}; defaulting to {MODE_OFF!r}.'
            )
            requested_mode = MODE_OFF
        self._active_mode = requested_mode
        self._manual_pen_mode = PEN_MODE_AUTO

        self._lock = threading.Lock()
        self._observed_statuses = {key: False for key in _REQUIRED_STATUS_KEYS}
        self._statuses = {key: None for key in _REQUIRED_STATUS_KEYS}
        self._board_info: dict[str, Any] | None = None
        self._board_bounds: dict[str, float] | None = None

        self._active_mode_pub = self.create_publisher(
            String,
            ACTIVE_MODE_TOPIC,
            _ACTIVE_MODE_QOS,
        )
        self._draw_plan_pub = self.create_publisher(DrawPlan, DRAW_PLAN_TOPIC, 10)
        self._manual_pen_mode_pub = self.create_publisher(
            String,
            MANUAL_PEN_MODE_TOPIC,
            _ACTIVE_MODE_QOS,
        )

        self.create_subscription(
            String,
            CABLE_EXECUTOR_STATUS_TOPIC,
            self._status_cb('cable_executor_status'),
            _STATUS_TOPIC_QOS,
        )
        self.create_subscription(
            String,
            CABLE_SUPERVISOR_STATUS_TOPIC,
            self._status_cb('cable_supervisor_status'),
            _STATUS_TOPIC_QOS,
        )
        self.create_subscription(String, '/wall_climber/board_info', self._board_info_cb, 10)
        self.create_subscription(
            String,
            MANUAL_PEN_MODE_TOPIC,
            self._manual_pen_mode_cb,
            _ACTIVE_MODE_QOS,
        )

        self._publish_active_mode(self._active_mode)
        self._publish_manual_pen_mode(self._manual_pen_mode)
        self.get_logger().info(
            f'Web backend ready on port {self.port} with initial mode {self._active_mode!r}.'
        )

    def _status_cb(self, key: str):
        def _callback(msg: String) -> None:
            value = str(msg.data).strip().lower()
            with self._lock:
                self._statuses[key] = value
                self._observed_statuses[key] = True

        return _callback

    def _board_info_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        needed = (
            'width',
            'height',
            'writable_x_min',
            'writable_x_max',
            'writable_y_min',
            'writable_y_max',
        )
        if not all(key in data for key in needed):
            return
        try:
            board_info = dict(data)
            for key, value in tuple(board_info.items()):
                if isinstance(value, (int, float)):
                    board_info[key] = float(value)
            for key in needed:
                board_info[key] = float(data[key])
        except (TypeError, ValueError):
            return
        with self._lock:
            self._board_info = board_info
            self._board_bounds = {
                'x_min': board_info['writable_x_min'],
                'x_max': board_info['writable_x_max'],
                'y_min': board_info['writable_y_min'],
                'y_max': board_info['writable_y_max'],
            }

    def _manual_pen_mode_cb(self, msg: String) -> None:
        mode = str(msg.data).strip().lower()
        if mode not in VALID_MANUAL_PEN_MODES:
            return
        with self._lock:
            self._manual_pen_mode = mode

    def _publish_active_mode(self, mode: str) -> None:
        msg = String()
        msg.data = mode
        self._active_mode_pub.publish(msg)

    def _publish_manual_pen_mode(self, mode: str) -> None:
        msg = String()
        msg.data = mode
        self._manual_pen_mode_pub.publish(msg)

    def runtime_snapshot(self) -> dict[str, Any]:
        with self._lock:
            statuses = dict(self._statuses)
            observed = dict(self._observed_statuses)
            board_info = dict(self._board_info) if self._board_info is not None else None
            active_mode = self._active_mode
            manual_pen_mode = self._manual_pen_mode
        ready = all(observed.values())
        return {
            'active_mode': active_mode,
            'manual_pen_mode': manual_pen_mode,
            'ready': ready,
            'not_ready_reason': None if ready else 'waiting_for_status_topics',
            'observed_statuses': observed,
            'statuses': statuses,
            'board_info': board_info,
            'enable_webots_trail': self.enable_webots_trail,
        }

    def ensure_ready(self) -> dict[str, Any]:
        snapshot = self.runtime_snapshot()
        if not snapshot['ready']:
            raise HTTPException(status_code=503, detail='runtime status topics are not ready yet')
        return snapshot

    def switch_mode(self, mode: str) -> dict[str, Any]:
        snapshot = self.ensure_ready()
        statuses = snapshot['statuses']
        if statuses['cable_executor_status'] == 'running':
            raise HTTPException(status_code=409, detail='runtime is busy; mode switch rejected')
        with self._lock:
            self._active_mode = mode
        self._publish_active_mode(mode)
        return self.runtime_snapshot()

    def set_manual_pen_mode(self, mode: str) -> dict[str, Any]:
        if mode not in VALID_MANUAL_PEN_MODES:
            raise HTTPException(status_code=400, detail='invalid manual pen mode')
        snapshot = self.ensure_ready()
        statuses = snapshot['statuses']
        if statuses['cable_executor_status'] == 'running':
            raise HTTPException(status_code=409, detail='runtime is busy; manual pen control rejected')
        with self._lock:
            self._manual_pen_mode = mode
        self._publish_manual_pen_mode(mode)
        return self.runtime_snapshot()

    def publish_draw_plan(self, plan: DrawPlan, *, allowed_modes: tuple[str, ...]) -> None:
        snapshot = self.ensure_ready()
        if snapshot['active_mode'] not in allowed_modes:
            allowed = ', '.join(allowed_modes)
            raise HTTPException(status_code=409, detail=f'active mode must be one of: {allowed}')
        if snapshot['manual_pen_mode'] != PEN_MODE_AUTO:
            raise HTTPException(status_code=409, detail='manual arm test must be set to auto before drawing')
        if snapshot['statuses']['cable_executor_status'] == 'running':
            raise HTTPException(status_code=409, detail='cable executor is busy')
        self._draw_plan_pub.publish(plan)

    def writable_bounds(self) -> dict[str, float]:
        with self._lock:
            if self._board_bounds is None:
                raise HTTPException(status_code=503, detail='board metadata is not ready yet')
            return dict(self._board_bounds)

    def carriage_safe_writable_bounds(self) -> dict[str, float]:
        try:
            return self._shared.carriage_safe_writable_bounds()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    def carriage_safe_safe_bounds(self) -> dict[str, float]:
        try:
            return self._shared.carriage_safe_workspace_bounds()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))


class BackendRuntime:
    def __init__(self, node: WebBackendNode) -> None:
        self._node = node
        self._executor = SingleThreadedExecutor()
        self._executor_thread: threading.Thread | None = None
        self._shutdown_lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._server: uvicorn.Server | None = None
        self._share_dir = Path(get_package_share_directory('wall_climber'))
        self._web_dir = self._share_dir / 'web'
        self._uploads_dir = Path.home() / '.ros' / 'wall_climber' / 'uploads'
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    @property
    def node(self) -> WebBackendNode:
        return self._node

    @property
    def web_dir(self) -> Path:
        return self._web_dir

    def attach_server(self, server: uvicorn.Server) -> None:
        self._server = server

    def start(self) -> None:
        with self._shutdown_lock:
            if self._started:
                return
            self._executor.add_node(self._node)
            self._executor_thread = threading.Thread(
                target=self._executor.spin,
                name='web_ui_server_ros_executor',
                daemon=True,
            )
            self._executor_thread.start()
            self._started = True

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._stopped:
                return
            self._stopped = True
            server = self._server
            if server is not None:
                server.should_exit = True
            try:
                self._executor.shutdown(timeout_sec=2.0)
            except TypeError:
                self._executor.shutdown()
            if self._executor_thread is not None:
                self._executor_thread.join(timeout=5.0)
                if self._executor_thread.is_alive():
                    self._node.get_logger().warn('ROS executor thread did not stop cleanly before timeout.')
            try:
                self._executor.remove_node(self._node)
            except Exception:
                pass
            try:
                self._node.destroy_node()
            except Exception:
                pass
            if rclpy.ok():
                rclpy.shutdown()

    def store_upload(self, upload: UploadFile, content: bytes) -> dict[str, Any]:
        upload_id = uuid.uuid4().hex
        extension = _ALLOWED_UPLOAD_TYPES[upload.content_type]
        payload_path = self._uploads_dir / f'{upload_id}{extension}'
        metadata_path = self._uploads_dir / f'{upload_id}.json'
        payload_path.write_bytes(content)
        metadata = {
            'upload_id': upload_id,
            'stored_filename': payload_path.name,
            'metadata_filename': metadata_path.name,
            'original_filename': upload.filename,
            'content_type': upload.content_type,
            'size_bytes': len(content),
            'stored_only': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, separators=(',', ':'), indent=2), encoding='utf-8')
        return metadata

    def load_upload(self, upload_id: str) -> tuple[dict[str, Any], bytes]:
        metadata_path = self._uploads_dir / f'{upload_id}.json'
        if not metadata_path.is_file():
            raise HTTPException(status_code=404, detail='upload_id was not found')
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'failed to read upload metadata: {exc}')
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=500, detail='upload metadata is invalid')
        stored_filename = metadata.get('stored_filename')
        if not isinstance(stored_filename, str) or not stored_filename:
            raise HTTPException(status_code=500, detail='upload metadata is missing stored filename')
        payload_path = self._uploads_dir / stored_filename
        if not payload_path.is_file():
            raise HTTPException(status_code=404, detail='stored upload payload is missing')
        try:
            payload = payload_path.read_bytes()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'failed to read upload payload: {exc}')
        return metadata, payload


def _require_json_object(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail=f'{name} body must be a JSON object')
    return raw


async def _load_json_request(
    request: Request,
    *,
    name: str,
    max_bytes: int,
) -> dict[str, Any]:
    raw_body = await request.body()
    if len(raw_body) > max_bytes:
        raise HTTPException(status_code=413, detail=f'{name} exceeds the maximum allowed payload size')
    try:
        raw_json = json.loads(raw_body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f'invalid {name} JSON: {exc}')
    return _require_json_object(raw_json, name)


def _reject_extra_fields(payload: dict[str, Any], allowed: set[str], name: str) -> None:
    extras = sorted(set(payload.keys()) - allowed)
    if extras:
        raise HTTPException(status_code=422, detail=f'{name} contains unsupported fields: {extras}')


def _validate_text_value(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail=f'{field_name} must be a string')
    if not value.strip():
        raise HTTPException(status_code=422, detail=f'{field_name} must not be empty')
    if len(value) > _MAX_TEXT_CHARS or len(value.encode('utf-8')) > _MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail=f'{field_name} exceeds the maximum allowed size')
    return value


def _validate_text_request(raw: Any) -> str:
    payload = _require_json_object(raw, 'text request')
    _reject_extra_fields(payload, {'text'}, 'text request')
    return _validate_text_value(payload.get('text'), 'text request.text')


def _coerce_float(
    value: Any,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f'{field_name} must be numeric')
    if not numpy.isfinite(numeric):
        raise HTTPException(status_code=422, detail=f'{field_name} must be finite')
    if minimum is not None and numeric < minimum:
        raise HTTPException(status_code=422, detail=f'{field_name} must be >= {minimum}')
    if maximum is not None and numeric > maximum:
        raise HTTPException(status_code=422, detail=f'{field_name} must be <= {maximum}')
    return numeric


def _coerce_int(
    value: Any,
    *,
    field_name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f'{field_name} must be an integer')
    if minimum is not None and numeric < minimum:
        raise HTTPException(status_code=422, detail=f'{field_name} must be >= {minimum}')
    if maximum is not None and numeric > maximum:
        raise HTTPException(status_code=422, detail=f'{field_name} must be <= {maximum}')
    return numeric


def _validate_upload_id(raw_upload_id: Any) -> str:
    if not isinstance(raw_upload_id, str):
        raise HTTPException(status_code=422, detail='upload_id must be a string')
    upload_id = raw_upload_id.strip().lower()
    if len(upload_id) != 32 or any(ch not in '0123456789abcdef' for ch in upload_id):
        raise HTTPException(status_code=422, detail='upload_id must be a 32-char lowercase hex string')
    return upload_id


def _resolve_text_start_placement(
    raw_placement: Any,
    *,
    request_name: str,
    writable_bounds: dict[str, float],
    safe_bounds: dict[str, float],
    text_layout_defaults,
) -> VectorPlacement:
    min_x = safe_bounds['x_min'] + float(text_layout_defaults.left_margin)

    # Keep left protection, but do not shrink the right side with a text right-margin.
    max_x = safe_bounds['x_max']

    min_y = safe_bounds['y_min'] + float(text_layout_defaults.top_margin)
    max_y = safe_bounds['y_max'] - float(text_layout_defaults.bottom_margin)

    default_x = min_x
    default_y = min_y
    default_scale = 1.0

    if raw_placement is None:
        return VectorPlacement(x=default_x, y=default_y, scale=default_scale)

    if not isinstance(raw_placement, dict):
        raise HTTPException(
            status_code=422,
            detail=f'{request_name}.placement must be an object with x, y, and scale',
        )

    _reject_extra_fields(raw_placement, {'x', 'y', 'scale'}, f'{request_name}.placement')

    x = _coerce_float(
        raw_placement.get('x', default_x),
        field_name=f'{request_name}.placement.x',
    )
    y = _coerce_float(
        raw_placement.get('y', default_y),
        field_name=f'{request_name}.placement.y',
    )
    scale = _coerce_float(
        raw_placement.get('scale', default_scale),
        field_name=f'{request_name}.placement.scale',
        minimum=0.05,
        maximum=10.0,
    )

    x = min(max(x, min_x), max_x)
    y = min(max(y, min_y), max_y)
    return VectorPlacement(x=x, y=y, scale=scale)


def _grouped_text_bounds(glyphs: tuple[TextGlyphOutline, ...]) -> dict[str, float]:
    if not glyphs:
        raise ValueError('text produced no drawable glyphs')
    x_min = min(glyph.bbox.x_min for glyph in glyphs)
    x_max = max(glyph.bbox.x_max for glyph in glyphs)
    y_min = min(glyph.bbox.y_min for glyph in glyphs)
    y_max = max(glyph.bbox.y_max for glyph in glyphs)
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': x_max - x_min,
        'height': y_max - y_min,
    }


def _normalize_text_font_source(font_source: Any) -> str:
    normalized = str(font_source or 'relief_singleline').strip().lower()
    if normalized not in {'relief_singleline', 'hershey_sans_1'}:
        raise ValueError('font_source must be one of ["relief_singleline", "hershey_sans_1"]')
    return normalized


def _expand_preview_bounds(
    bounds: dict[str, float],
    *,
    pad_m: float,
) -> dict[str, float]:
    pad = max(1.0e-4, float(pad_m))
    x_min = float(bounds['x_min']) - pad
    x_max = float(bounds['x_max']) + pad
    y_min = float(bounds['y_min']) - pad
    y_max = float(bounds['y_max']) + pad
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': x_max - x_min,
        'height': y_max - y_min,
    }


def _preview_payload_from_strokes(
    placed_strokes: tuple[tuple[tuple[float, float], ...], ...],
    placement_result,
    *,
    outside_safe_points: int,
    normalized_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    draw_plan = strokes_to_draw_plan(placed_strokes)
    preview_strokes = [stroke['points'] for stroke in draw_plan['strokes']]
    stats = stroke_stats(placed_strokes)

    # Add preview-only padding so letters touching the text bounds
    # do not appear visually clipped in the browser.
    preview_pad_m = max(0.003, float(placement_result.final_scale) * 0.10)
    padded_bounds = _expand_preview_bounds(stats['bounds'], pad_m=preview_pad_m)

    can_commit = placement_result.outside_points == 0 and outside_safe_points == 0
    validation_error = None
    if placement_result.outside_points != 0:
        validation_error = 'geometry exceeds carriage-safe writable bounds'
    elif outside_safe_points != 0:
        validation_error = 'geometry exits the configured safe cable workspace'

    return {
        'strokes': preview_strokes,
        'stroke_count': stats['stroke_count'],
        'point_count': stats['point_count'],
        'bounds': padded_bounds,
        'placement': {
            'x': placement_result.placement.x,
            'y': placement_result.placement.y,
            'scale': placement_result.placement.scale,
            'base_fit_scale': placement_result.base_fit_scale,
            'final_scale': placement_result.final_scale,
        },
        'outside_points': placement_result.outside_points,
        'outside_safe_points': outside_safe_points,
        'can_commit': can_commit,
        'validation_error': validation_error,
        'normalized_plan': normalized_plan,
    }


def _interpolated_outside_safe_workspace_count(
    polylines: tuple[tuple[tuple[float, float], ...], ...],
    shared_config,
    *,
    step_m: float = 0.01,
) -> int:
    outside = 0
    step = max(1.0e-4, float(step_m))
    for stroke in polylines:
        if len(stroke) < 2:
            continue
        for index in range(1, len(stroke)):
            start = stroke[index - 1]
            end = stroke[index]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = float(numpy.hypot(dx, dy))
            subdivisions = max(1, int(numpy.ceil(length / step)))
            for sample_index in range(subdivisions + 1):
                t = sample_index / subdivisions
                x = start[0] + dx * t
                y = start[1] + dy * t
                if not shared_config.point_in_safe_workspace(x, y):
                    outside += 1
    return outside


def _build_draw_plan_message(
    segments: tuple[DrawPathSegment, ...],
    *,
    theta_ref: float,
    writable_bounds: dict[str, float],
    shared_config,
) -> DrawPlan:
    if not segments:
        raise HTTPException(status_code=422, detail='draw plan has no drawable segments')
    for index, segment in enumerate(segments):
        if len(segment.points) < 2:
            raise HTTPException(status_code=422, detail=f'draw segment[{index}] is degenerate')
        for point in segment.points:
            if not (
                writable_bounds['x_min'] <= point[0] <= writable_bounds['x_max']
                and writable_bounds['y_min'] <= point[1] <= writable_bounds['y_max']
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f'draw segment[{index}] extends outside carriage-safe writable bounds',
                )
        if _interpolated_outside_safe_workspace_count((segment.points,), shared_config) != 0:
            raise HTTPException(
                status_code=422,
                detail=f'draw segment[{index}] exits the configured safe cable workspace',
            )
    plan = DrawPlan()
    plan.frame_id = 'board'
    plan.theta_ref = float(theta_ref)
    for segment in segments:
        msg_segment = DrawPolyline()
        msg_segment.draw = bool(segment.draw)
        msg_segment.points = [
            Point(x=float(point[0]), y=float(point[1]), z=0.0)
            for point in segment.points
        ]
        plan.segments.append(msg_segment)
    return plan


def _sanitize_points(raw_points: Any, stroke_index: int) -> list[tuple[float, float]]:
    if not isinstance(raw_points, list):
        raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}].points must be a list')
    if len(raw_points) > _MAX_POINTS_PER_STROKE:
        raise HTTPException(status_code=413, detail=f'stroke[{stroke_index}] exceeds the maximum points per stroke')
    points: list[tuple[float, float]] = []
    for point_index, point in enumerate(raw_points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}].points[{point_index}] must be [x, y]')
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}].points[{point_index}] must be numeric')
        if not (numpy.isfinite(x) and numpy.isfinite(y)):
            raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}].points[{point_index}] must be finite')
        current = (x, y)
        if points and abs(points[-1][0] - current[0]) <= _SEGMENT_EPS_M and abs(points[-1][1] - current[1]) <= _SEGMENT_EPS_M:
            continue
        points.append(current)
    sanitized: list[tuple[float, float]] = []
    for point in points:
        if not sanitized:
            sanitized.append(point)
            continue
        previous = sanitized[-1]
        if abs(previous[0] - point[0]) <= _SEGMENT_EPS_M and abs(previous[1] - point[1]) <= _SEGMENT_EPS_M:
            continue
        sanitized.append(point)
    return sanitized


def _normalize_draw_plan(raw: Any, writable_bounds: dict[str, float]) -> str:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail='draw plan body must be a JSON object')
    _reject_extra_fields(raw, {'frame', 'strokes'}, 'draw plan')
    if raw.get('frame') != 'board':
        raise HTTPException(status_code=422, detail='draw plan.frame must be exactly "board"')
    strokes = raw.get('strokes')
    if not isinstance(strokes, list) or not strokes:
        raise HTTPException(status_code=422, detail='draw plan.strokes must be a non-empty list')
    if len(strokes) > _MAX_DRAW_STROKES:
        raise HTTPException(status_code=413, detail='draw plan exceeds the maximum number of strokes')

    normalized_strokes: list[dict[str, Any]] = []
    total_points = 0
    for stroke_index, stroke in enumerate(strokes):
        if not isinstance(stroke, dict):
            raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}] must be an object')
        _reject_extra_fields(stroke, {'type', 'draw', 'points'}, f'stroke[{stroke_index}]')
        stroke_type = stroke.get('type')
        if stroke_type not in ('line', 'polyline'):
            raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}].type must be "line" or "polyline"')
        draw_flag = stroke.get('draw')
        if not isinstance(draw_flag, bool):
            raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}].draw must be boolean')
        points = _sanitize_points(stroke.get('points'), stroke_index)
        if len(points) < 2:
            raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}] is degenerate after sanitization')
        total_points += len(points)
        if total_points > _MAX_TOTAL_POINTS:
            raise HTTPException(status_code=413, detail='draw plan exceeds the maximum total point budget')
        for point in points:
            if not (
                writable_bounds['x_min'] <= point[0] <= writable_bounds['x_max']
                and writable_bounds['y_min'] <= point[1] <= writable_bounds['y_max']
            ):
                raise HTTPException(status_code=422, detail=f'stroke[{stroke_index}] contains points outside writable board bounds')
        normalized_strokes.append(
            {
                'type': 'line' if len(points) == 2 else 'polyline',
                'draw': draw_flag,
                'points': [[point[0], point[1]] for point in points],
            }
        )
    payload = {'frame': 'board', 'strokes': normalized_strokes}
    encoded = json.dumps(payload, separators=(',', ':'))
    if len(encoded.encode('utf-8')) > _MAX_DRAW_PLAN_BYTES:
        raise HTTPException(status_code=413, detail='draw plan exceeds the maximum allowed payload size')
    return encoded


def _validate_upload(upload: UploadFile, content: bytes) -> None:
    if upload.content_type not in _ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=415, detail='unsupported upload content type')
    suffix = Path(upload.filename or '').suffix.lower()
    if suffix and suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415, detail='unsupported upload filename extension')
    if not content:
        raise HTTPException(status_code=422, detail='uploaded image is empty')
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='uploaded image exceeds the maximum allowed size')
    array = numpy.frombuffer(content, dtype=numpy.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=422, detail='uploaded file is not a decodable image')


def _resolve_web_asset_path(web_dir: Path, asset_path: str) -> Path:
    normalized = asset_path.strip('/')
    rel_path = Path(normalized)
    if not normalized or rel_path.is_absolute():
        raise HTTPException(status_code=404, detail='asset not found')
    if any(part in ('', '.', '..') for part in rel_path.parts):
        raise HTTPException(status_code=404, detail='asset not found')
    candidate = web_dir / rel_path
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail='asset not found')
    return candidate


def create_app(runtime: BackendRuntime) -> FastAPI:
    app = FastAPI(title='Two-Cable Drawing Robot UI Backend', version='1.0.0')
    shared = load_shared_config()
    text_layout_defaults = shared.text_layout
    draw_execution_defaults = shared.draw_execution

    @app.get('/assets/{asset_path:path}')
    async def assets(asset_path: str) -> FileResponse:
        return FileResponse(_resolve_web_asset_path(runtime.web_dir, asset_path))

    @app.get('/vendor/{asset_path:path}')
    async def vendor_compat(asset_path: str) -> FileResponse:
        # Backward-compatible alias for older index.html versions.
        return FileResponse(
            _resolve_web_asset_path(runtime.web_dir, f'vendor/{asset_path}')
        )

    @app.get('/')
    async def index() -> FileResponse:
        return FileResponse(runtime.web_dir / 'index.html')

    @app.get('/api/health')
    async def health() -> JSONResponse:
        snapshot = runtime.node.runtime_snapshot()
        return JSONResponse(
            {
                'ok': True,
                'ready': snapshot['ready'],
                'active_mode': snapshot['active_mode'],
                'observed_statuses': snapshot['observed_statuses'],
            }
        )

    @app.get('/api/runtime')
    async def runtime_state() -> JSONResponse:
        return JSONResponse(runtime.node.runtime_snapshot())

    @app.post('/api/mode')
    async def set_mode(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='mode request',
            max_bytes=1024,
        )
        _reject_extra_fields(raw, {'mode'}, 'mode request')
        mode = raw.get('mode')
        if mode not in VALID_MODES:
            raise HTTPException(status_code=422, detail=f'mode must be one of {VALID_MODES}')
        snapshot = runtime.node.switch_mode(mode)
        return JSONResponse({'ok': True, 'active_mode': snapshot['active_mode'], 'runtime': snapshot})

    @app.post('/api/manual/pen')
    async def set_manual_pen_mode(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='manual pen request',
            max_bytes=1024,
        )
        _reject_extra_fields(raw, {'mode'}, 'manual pen request')
        mode = raw.get('mode')
        if mode not in VALID_MANUAL_PEN_MODES:
            raise HTTPException(
                status_code=422,
                detail=f'mode must be one of {VALID_MANUAL_PEN_MODES}',
            )
        snapshot = runtime.node.set_manual_pen_mode(mode)
        return JSONResponse({'ok': True, 'manual_pen_mode': mode, 'runtime': snapshot})

    def _build_text_vector(
        raw: dict[str, Any],
        *,
        request_name: str,
    ) -> tuple[
        tuple[TextGlyphOutline, ...],
        tuple[tuple[tuple[float, float], ...], ...],
        Any,
        dict[str, float],
        dict[str, Any],
        DrawPlan,
        dict[str, Any],
        int,
    ]:
        allowed = {
            'text',
            'placement',
            'font_source',
            'line_height',
            'curve_tolerance',
            'simplify_epsilon',
            'fit_padding',
        }
        _reject_extra_fields(raw, allowed, request_name)
        text = _validate_text_value(raw.get('text'), f'{request_name}.text')
        try:
            normalized_text = normalize_text_plan_input(
                text,
                decode_escaped_line_breaks=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name}.text invalid: {exc}')
        try:
            font_source = _normalize_text_font_source(raw.get('font_source', 'relief_singleline'))
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f'{request_name}.font_source invalid: {exc}',
            )
        default_line_height = 1.75
        line_height = _coerce_float(
            raw.get('line_height', default_line_height),
            field_name=f'{request_name}.line_height',
            minimum=0.3,
            maximum=4.0,
        )
        curve_tolerance = _coerce_float(
            raw.get('curve_tolerance', 0.008),
            field_name=f'{request_name}.curve_tolerance',
            minimum=0.0005,
            maximum=0.2,
        )
        simplify_epsilon = _coerce_float(
            raw.get('simplify_epsilon', 0.0),
            field_name=f'{request_name}.simplify_epsilon',
            minimum=0.0,
            maximum=0.2,
        )
        fit_padding = _coerce_float(
            raw.get('fit_padding', 0.9),
            field_name=f'{request_name}.fit_padding',
            minimum=0.1,
            maximum=1.0,
        )
        writable_bounds = runtime.node.carriage_safe_writable_bounds()
        safe_bounds = runtime.node.carriage_safe_safe_bounds()
        text_start = _resolve_text_start_placement(
            raw.get('placement'),
            request_name=request_name,
            writable_bounds=writable_bounds,
            safe_bounds=safe_bounds,
            text_layout_defaults=text_layout_defaults,
        )
        glyph_scale_m = float(text_layout_defaults.glyph_height) * text_start.scale
        if glyph_scale_m <= 0.0:
            raise HTTPException(status_code=422, detail=f'{request_name}.placement.scale must be > 0')
        available_width_m = (
            safe_bounds['x_max'] - text_start.x
        )
        if available_width_m <= 0.0:
            raise HTTPException(
                status_code=422,
                detail=f'{request_name}.placement.x leaves no carriage-safe width for text',
            )
        max_line_width_units = available_width_m / glyph_scale_m
        if max_line_width_units <= 0.25:
            raise HTTPException(
                status_code=422,
                detail=f'{request_name}.placement.scale is too large for the remaining carriage-safe width',
            )
        try:
            grouped_source = vectorize_text_grouped(
                normalized_text,
                font_source=font_source,
                line_height=line_height,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
                max_line_width_units=max_line_width_units,
            )
            source_bounds = _grouped_text_bounds(grouped_source)
            board_width = writable_bounds['x_max'] - writable_bounds['x_min']
            board_height = writable_bounds['y_max'] - writable_bounds['y_min']
            fit_scale = min(
                (board_width * fit_padding) / max(source_bounds['width'], 1.0e-9),
                (board_height * fit_padding) / max(source_bounds['height'], 1.0e-9),
            )
            if fit_scale <= 0.0:
                raise ValueError('invalid fit scale for text placement')
            placement = VectorPlacement(
                x=text_start.x + (0.5 * source_bounds['width'] * glyph_scale_m),
                y=text_start.y + (0.5 * source_bounds['height'] * glyph_scale_m),
                scale=glyph_scale_m / fit_scale,
            )
            placed_groups, placement_result = place_grouped_text_on_board(
                grouped_source,
                writable_bounds=writable_bounds,
                placement=placement,
                fit_padding=fit_padding,
                text_upward_bias_em=0.0,
            )
            placed_strokes = tuple(
                stroke for glyph in placed_groups for stroke in glyph.strokes
            )
            cleaned_strokes = cleanup_draw_strokes(
                placed_strokes,
                simplify_tolerance_m=draw_execution_defaults.draw_path_simplify_tolerance_m,
                preserve_order=True,
            )
            segments = draw_segments_from_pen_strokes(
                cleaned_strokes,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
                pen_offset_x_m=0.0,
                pen_offset_y_m=0.0,
            )
            plan_msg = _build_draw_plan_message(
                segments,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
                writable_bounds=writable_bounds,
                shared_config=shared,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: {exc}')
        outside_safe_points = _interpolated_outside_safe_workspace_count(cleaned_strokes, shared)
        plan_preview = draw_plan_to_dict(
            segments,
            theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
        )
        commit_request = {
            'text': normalized_text,
            'placement': {'x': text_start.x, 'y': text_start.y, 'scale': text_start.scale},
            'font_source': font_source,
            'line_height': line_height,
            'curve_tolerance': curve_tolerance,
            'simplify_epsilon': simplify_epsilon,
            'fit_padding': fit_padding,
            'glyph_height_m': float(text_layout_defaults.glyph_height),
        }
        return (
            placed_groups,
            cleaned_strokes,
            placement_result,
            writable_bounds,
            commit_request,
            plan_msg,
            plan_preview,
            outside_safe_points,
        )

    def _text_clusters_payload(
        text_clusters: tuple[TextGlyphOutline, ...],
        normalized_strokes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cluster_payload: list[dict[str, Any]] = []
        stroke_index = 0
        for cluster in text_clusters:
            stroke_count = len(cluster.strokes)
            cluster_payload.append(
                {
                    'line_index': int(cluster.line_index),
                    'word_index': int(cluster.word_index),
                    'text': cluster.text,
                    'strokes': normalized_strokes[
                        stroke_index:stroke_index + stroke_count
                    ],
                }
            )
            stroke_index += stroke_count
        return cluster_payload

    def _normalize_text_plan(
        text_clusters: tuple[TextGlyphOutline, ...],
        writable_bounds: dict[str, float],
    ) -> str:
        draw_plan = strokes_to_draw_plan(
            tuple(stroke for cluster in text_clusters for stroke in cluster.strokes)
        )
        normalized = json.loads(_normalize_draw_plan(draw_plan, writable_bounds))
        normalized['text_clusters'] = _text_clusters_payload(
            text_clusters,
            normalized['strokes'],
        )
        return json.dumps(normalized, separators=(',', ':'))

    def _build_svg_vector(
        raw: dict[str, Any],
        *,
        request_name: str,
    ) -> tuple[
        tuple[tuple[tuple[float, float], ...], ...],
        tuple[DrawPathSegment, ...],
        Any,
        dict[str, float],
        dict[str, Any],
        DrawPlan,
        dict[str, Any],
        int,
    ]:
        allowed = {
            'svg',
            'placement',
            'curve_tolerance',
            'simplify_epsilon',
        }
        _reject_extra_fields(raw, allowed, request_name)
        svg_payload = raw.get('svg')
        if not isinstance(svg_payload, str):
            raise HTTPException(status_code=422, detail=f'{request_name}.svg must be a string')
        if not svg_payload.strip():
            raise HTTPException(status_code=422, detail=f'{request_name}.svg must not be empty')
        if len(svg_payload.encode('utf-8')) > _MAX_SVG_BYTES:
            raise HTTPException(status_code=413, detail=f'{request_name}.svg exceeds max size')
        curve_tolerance = _coerce_float(
            raw.get('curve_tolerance', 0.015),
            field_name=f'{request_name}.curve_tolerance',
            minimum=0.0005,
            maximum=0.2,
        )
        simplify_epsilon = _coerce_float(
            raw.get('simplify_epsilon', 0.0),
            field_name=f'{request_name}.simplify_epsilon',
            minimum=0.0,
            maximum=2.0,
        )
        writable_bounds = runtime.node.carriage_safe_writable_bounds()
        try:
            placement = normalize_placement(raw.get('placement'), writable_bounds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name}.placement invalid: {exc}')
        try:
            source_strokes = vectorize_svg(
                svg_payload,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
            )
            placed_strokes, placement_result = place_draw_strokes_on_board(
                source_strokes,
                writable_bounds=writable_bounds,
                placement=placement,
                fit_margin_m=draw_execution_defaults.draw_scale_fit_margin_m,
            )
            cleaned_strokes = cleanup_draw_strokes(
                placed_strokes,
                simplify_tolerance_m=draw_execution_defaults.draw_path_simplify_tolerance_m,
            )
            segments = draw_segments_from_pen_strokes(
                cleaned_strokes,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
                pen_offset_x_m=0.0,
                pen_offset_y_m=0.0,
            )
            plan_msg = _build_draw_plan_message(
                segments,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
                writable_bounds=writable_bounds,
                shared_config=shared,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: {exc}')
        outside_safe_points = _interpolated_outside_safe_workspace_count(placed_strokes, shared)
        plan_preview = draw_plan_to_dict(
            segments,
            theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
        )
        commit_request = {
            'svg': svg_payload,
            'placement': {'x': placement.x, 'y': placement.y, 'scale': placement.scale},
            'curve_tolerance': curve_tolerance,
            'simplify_epsilon': simplify_epsilon,
            'draw_scale_fit_margin_m': draw_execution_defaults.draw_scale_fit_margin_m,
        }
        return (
            cleaned_strokes,
            segments,
            placement_result,
            writable_bounds,
            commit_request,
            plan_msg,
            plan_preview,
            outside_safe_points,
        )

    def _build_image_vector(
        content: bytes,
        raw: dict[str, Any],
        *,
        request_name: str,
    ) -> tuple[
        tuple[tuple[tuple[float, float], ...], ...],
        tuple[DrawPathSegment, ...],
        Any,
        dict[str, float],
        dict[str, Any],
        dict[str, int],
        DrawPlan,
        dict[str, Any],
        int,
    ]:
        allowed = {
            'placement',
            'min_perimeter_px',
            'contour_simplify_ratio',
            'max_strokes',
        }
        _reject_extra_fields(raw, allowed, request_name)
        min_perimeter_px = _coerce_float(
            raw.get('min_perimeter_px', 24.0),
            field_name=f'{request_name}.min_perimeter_px',
            minimum=1.0,
            maximum=1000.0,
        )
        contour_simplify_ratio = _coerce_float(
            raw.get('contour_simplify_ratio', 0.005),
            field_name=f'{request_name}.contour_simplify_ratio',
            minimum=0.0001,
            maximum=0.2,
        )
        max_strokes = _coerce_int(
            raw.get('max_strokes', 512),
            field_name=f'{request_name}.max_strokes',
            minimum=1,
            maximum=2048,
        )
        writable_bounds = runtime.node.carriage_safe_writable_bounds()
        try:
            placement = normalize_placement(raw.get('placement'), writable_bounds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name}.placement invalid: {exc}')
        try:
            source_strokes, image_size = trace_line_art_image(
                content,
                min_perimeter_px=min_perimeter_px,
                contour_simplify_ratio=contour_simplify_ratio,
                max_strokes=max_strokes,
            )
            placed_strokes, placement_result = place_draw_strokes_on_board(
                source_strokes,
                writable_bounds=writable_bounds,
                placement=placement,
                fit_margin_m=draw_execution_defaults.draw_scale_fit_margin_m,
            )
            cleaned_strokes = cleanup_draw_strokes(
                placed_strokes,
                simplify_tolerance_m=draw_execution_defaults.draw_path_simplify_tolerance_m,
            )
            segments = draw_segments_from_pen_strokes(
                cleaned_strokes,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
                pen_offset_x_m=0.0,
                pen_offset_y_m=0.0,
            )
            plan_msg = _build_draw_plan_message(
                segments,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
                writable_bounds=writable_bounds,
                shared_config=shared,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: {exc}')
        outside_safe_points = _interpolated_outside_safe_workspace_count(placed_strokes, shared)
        plan_preview = draw_plan_to_dict(
            segments,
            theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
        )
        commit_tail = {
            'placement': {'x': placement.x, 'y': placement.y, 'scale': placement.scale},
            'min_perimeter_px': min_perimeter_px,
            'contour_simplify_ratio': contour_simplify_ratio,
            'max_strokes': max_strokes,
            'draw_scale_fit_margin_m': draw_execution_defaults.draw_scale_fit_margin_m,
        }
        image_info = {'width_px': int(image_size[0]), 'height_px': int(image_size[1])}
        return (
            cleaned_strokes,
            segments,
            placement_result,
            writable_bounds,
            commit_tail,
            image_info,
            plan_msg,
            plan_preview,
            outside_safe_points,
        )

    @app.post('/api/text')
    async def submit_text(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='text request',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        placed_groups, placed_strokes, placement_result, writable_bounds, commit_request, plan_msg, plan_preview, outside_safe_points = _build_text_vector(
            raw,
            request_name='text request',
        )
        normalized = _normalize_text_plan(placed_groups, writable_bounds)
        runtime.node.publish_draw_plan(plan_msg, allowed_modes=(MODE_TEXT,))
        return JSONResponse(
            {
                'ok': True,
                'published': True,
                'active_mode': MODE_TEXT,
                'preview': _preview_payload_from_strokes(
                    placed_strokes,
                    placement_result,
                    outside_safe_points=outside_safe_points,
                    normalized_plan=plan_preview,
                ),
                'commit_request': commit_request,
                'normalized_plan': json.loads(normalized),
            }
        )

    @app.post('/api/draw/plan')
    async def submit_draw_plan(request: Request) -> JSONResponse:
        raise HTTPException(
            status_code=409,
            detail='raw /api/draw/plan is disabled for draw mode v1; use /api/vector/svg/commit or /api/vector/image/commit',
        )

    @app.post('/api/vector/text/preview')
    async def preview_text_vector(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='vector text request',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        _, placed_strokes, placement_result, _, commit_request, _, plan_preview, outside_safe_points = _build_text_vector(
            raw,
            request_name='vector text request',
        )
        return JSONResponse(
            {
                'ok': True,
                'source_type': 'text',
                'preview': _preview_payload_from_strokes(
                    placed_strokes,
                    placement_result,
                    outside_safe_points=outside_safe_points,
                    normalized_plan=plan_preview,
                ),
                'commit_request': commit_request,
            }
        )

    @app.post('/api/vector/text/commit')
    async def commit_text_vector(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='vector text commit',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        placed_groups, placed_strokes, placement_result, writable_bounds, commit_request, plan_msg, plan_preview, outside_safe_points = _build_text_vector(
            raw,
            request_name='vector text commit',
        )
        normalized = _normalize_text_plan(placed_groups, writable_bounds)
        runtime.node.publish_draw_plan(plan_msg, allowed_modes=(MODE_TEXT,))
        return JSONResponse(
            {
                'ok': True,
                'published': True,
                'active_mode': MODE_TEXT,
                'source_type': 'text',
                'preview': _preview_payload_from_strokes(
                    placed_strokes,
                    placement_result,
                    outside_safe_points=outside_safe_points,
                    normalized_plan=plan_preview,
                ),
                'commit_request': commit_request,
                'normalized_plan': json.loads(normalized),
            }
        )

    @app.post('/api/vector/svg/preview')
    async def preview_svg_vector(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='vector svg request',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        placed_strokes, _, placement_result, _, commit_request, _, plan_preview, outside_safe_points = _build_svg_vector(
            raw,
            request_name='vector svg request',
        )
        return JSONResponse(
            {
                'ok': True,
                'source_type': 'svg',
                'preview': _preview_payload_from_strokes(
                    placed_strokes,
                    placement_result,
                    outside_safe_points=outside_safe_points,
                    normalized_plan=plan_preview,
                ),
                'commit_request': commit_request,
            }
        )

    @app.post('/api/vector/svg/commit')
    async def commit_svg_vector(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='vector svg commit',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        placed_strokes, _, placement_result, _, commit_request, plan_msg, plan_preview, outside_safe_points = _build_svg_vector(
            raw,
            request_name='vector svg commit',
        )
        runtime.node.publish_draw_plan(plan_msg, allowed_modes=(MODE_DRAW,))
        return JSONResponse(
            {
                'ok': True,
                'published': True,
                'active_mode': MODE_DRAW,
                'source_type': 'svg',
                'preview': _preview_payload_from_strokes(
                    placed_strokes,
                    placement_result,
                    outside_safe_points=outside_safe_points,
                    normalized_plan=plan_preview,
                ),
                'commit_request': commit_request,
                'normalized_plan': plan_preview,
            }
        )

    @app.post('/api/vector/image/preview')
    async def preview_uploaded_image_vector(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='vector image request',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        _reject_extra_fields(
            raw,
            {
                'upload_id',
                'placement',
                'min_perimeter_px',
                'contour_simplify_ratio',
                'max_strokes',
            },
            'vector image request',
        )
        upload_id = _validate_upload_id(raw.get('upload_id'))
        metadata, payload = runtime.load_upload(upload_id)
        image_raw = dict(raw)
        image_raw.pop('upload_id', None)
        placed_strokes, _, placement_result, _, commit_tail, image_info, _, plan_preview, outside_safe_points = _build_image_vector(
            payload,
            image_raw,
            request_name='vector image request',
        )
        commit_request = {'upload_id': upload_id, **commit_tail}
        return JSONResponse(
            {
                'ok': True,
                'source_type': 'image',
                'upload': metadata,
                'image_info': image_info,
                'preview': _preview_payload_from_strokes(
                    placed_strokes,
                    placement_result,
                    outside_safe_points=outside_safe_points,
                    normalized_plan=plan_preview,
                ),
                'commit_request': commit_request,
            }
        )

    @app.post('/api/vector/image/commit')
    async def commit_uploaded_image_vector(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='vector image commit',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        _reject_extra_fields(
            raw,
            {
                'upload_id',
                'placement',
                'min_perimeter_px',
                'contour_simplify_ratio',
                'max_strokes',
            },
            'vector image commit',
        )
        upload_id = _validate_upload_id(raw.get('upload_id'))
        metadata, payload = runtime.load_upload(upload_id)
        image_raw = dict(raw)
        image_raw.pop('upload_id', None)
        placed_strokes, _, placement_result, _, commit_tail, image_info, plan_msg, plan_preview, outside_safe_points = _build_image_vector(
            payload,
            image_raw,
            request_name='vector image commit',
        )
        runtime.node.publish_draw_plan(plan_msg, allowed_modes=(MODE_DRAW,))
        return JSONResponse(
            {
                'ok': True,
                'published': True,
                'active_mode': MODE_DRAW,
                'source_type': 'image',
                'upload': metadata,
                'image_info': image_info,
                'preview': _preview_payload_from_strokes(
                    placed_strokes,
                    placement_result,
                    outside_safe_points=outside_safe_points,
                    normalized_plan=plan_preview,
                ),
                'commit_request': {'upload_id': upload_id, **commit_tail},
                'normalized_plan': plan_preview,
            }
        )

    @app.post('/api/draw/image')
    async def upload_draw_image(file: UploadFile) -> JSONResponse:
        try:
            content = await file.read(_MAX_UPLOAD_BYTES + 1)
            _validate_upload(file, content)
            metadata = runtime.store_upload(file, content)
            try:
                writable_bounds = runtime.node.carriage_safe_writable_bounds()
            except HTTPException as exc:
                return JSONResponse(
                    {
                        'ok': True,
                        'stored_only': True,
                        'upload': metadata,
                        'preview_error': str(exc.detail),
                    }
                )
            placement = default_placement(writable_bounds)
            placed_strokes, _, placement_result, _, commit_tail, image_info, _, plan_preview, outside_safe_points = _build_image_vector(
                content,
                {'placement': {'x': placement.x, 'y': placement.y, 'scale': placement.scale}},
                request_name='draw image upload',
            )
            return JSONResponse(
                {
                    'ok': True,
                    'stored_only': False,
                    'source_type': 'image',
                    'upload': metadata,
                    'image_info': image_info,
                    'preview': _preview_payload_from_strokes(
                        placed_strokes,
                        placement_result,
                        outside_safe_points=outside_safe_points,
                        normalized_plan=plan_preview,
                    ),
                    'commit_request': {'upload_id': metadata['upload_id'], **commit_tail},
                }
            )
        finally:
            await file.close()

    return app


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebBackendNode()
    runtime = BackendRuntime(node)
    app = create_app(runtime)
    config = uvicorn.Config(app, host='0.0.0.0', port=node.port, log_level='info')
    server = uvicorn.Server(config)
    runtime.attach_server(server)

    if node.open_browser:
        threading.Timer(0.75, lambda: webbrowser.open(f'http://localhost:{node.port}')).start()

    runtime.start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()


if __name__ == '__main__':
    main()
