from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import base64
import hashlib
import io
import json
import socket
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Optional

import numpy
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

try:
    import rclpy
    from ament_index_python.packages import get_package_share_directory
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from wall_climber_interfaces.msg import BoardPoint, PathPrimitive, PrimitivePathPlan
except ImportError as exc:
    rclpy = None
    _ROS_IMPORT_ERROR = exc

    def get_package_share_directory(_package_name: str) -> str:
        raise RuntimeError('ROS 2 Python dependencies are required for package share lookup.') from _ROS_IMPORT_ERROR

    class SingleThreadedExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError('ROS 2 Python dependencies are required for WebBackendNode.') from _ROS_IMPORT_ERROR

    class Node:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError('ROS 2 Python dependencies are required for WebBackendNode.') from _ROS_IMPORT_ERROR

    class ReliabilityPolicy:
        RELIABLE = 'reliable'

    class DurabilityPolicy:
        TRANSIENT_LOCAL = 'transient_local'

    class QoSProfile:
        def __init__(self, *, depth: int, reliability: Any, durability: Any) -> None:
            self.depth = depth
            self.reliability = reliability
            self.durability = durability

    class String:
        data: str

    class BoardPoint:
        x: float
        y: float

    class PathPrimitive:
        pass

    class PrimitivePathPlan:
        pass
else:
    _ROS_IMPORT_ERROR = None

from wall_climber import _http_helpers as _http
from wall_climber._debug_snapshots import DebugSnapshotStore
from wall_climber._ttl_cache import TTLCache
from wall_climber._uploads_store import (
    DEFAULT_IMAGE_PREP_OPTIONS as _DEFAULT_IMAGE_PREP_OPTIONS,
    PreparedImageArtifact,
    UPLOAD_GC_MIN_INTERVAL_SECONDS as _UPLOAD_GC_MIN_INTERVAL_SECONDS,
    UPLOAD_RETENTION_SECONDS as _UPLOAD_RETENTION_SECONDS,
    UploadStore,
)
from wall_climber import canonical_adapters as _canonical_adapters
from wall_climber.canonical_adapters import (
    SamplingPolicy,
    canonical_plan_debug_payload,
    canonical_plan_diagnostics,
    canonical_plan_to_draw_strokes,
    canonical_plan_to_legacy_strokes,
    canonical_plan_to_primitive_path_plan,
    canonical_plan_to_sampled_paths,
)
from wall_climber.canonical_builders import (
    draw_strokes_to_canonical_plan,
    text_glyph_outlines_to_canonical_plan,
)
from wall_climber.canonical_optimizer import (
    CanonicalOptimizationPolicy,
    optimize_canonical_plan,
)
from wall_climber.canonical_path import (
    ArcSegment,
    CanonicalCommand,
    CanonicalPathPlan,
    CubicBezier,
    LineSegment,
    PenDown,
    PenUp,
    QuadraticBezier,
    TravelMove,
)
from wall_climber.canonical_tiny_details import expand_tiny_details_in_canonical_plan
from wall_climber.canonical_ops import (
    cleanup_canonical_plan,
    default_image_placement,
    normalize_placement,
    place_canonical_plan_on_board,
    place_grouped_text_on_board,
    stroke_stats,
)
from wall_climber.ingestion.image import vectorize_image_to_canonical_plan
from wall_climber.ingestion.image_curve_fitting import (
    curve_fit_debug_to_board,
    map_curve_fit_command_metadata,
)
from wall_climber.ingestion.svg import vectorize_svg
from wall_climber.ingestion.text import (
    TextGlyphOutline,
    normalize_text_plan_input,
    vectorize_text_grouped,
)
from wall_climber.ingestion.upload_routing import (
    UploadedVectorFile,
    classify_uploaded_vector_file,
    infer_uploaded_source_type,
)
from wall_climber.image_pipeline.adapters import drawing_path_plan_to_canonical
from wall_climber.image_pipeline.color_to_lineart import convert_color_image_to_lineart
from wall_climber.image_pipeline.curve_fit import drawing_path_plan_to_smooth_canonical
from wall_climber.image_pipeline.input_detection import detect_raster_input_type
from wall_climber.image_pipeline.sketch_centerline import vectorize_sketch_image_to_plan
from wall_climber.image_pipeline.types import DrawingPathPlan
from wall_climber.image_pipeline.vectorizers.autotrace_centerline import (
    vectorize_autotrace_centerline,
)
from wall_climber.image_pipeline.vectorizers.base import VectorizationEngineResult
from wall_climber.image_pipeline.vectorizers.potrace_backend import vectorize_potrace_bw
from wall_climber.image_pipeline.vectorizers.vtracer_backend import vectorize_vtracer_svg
from wall_climber.optimizers import vpype_optimizer
from wall_climber.runtime_topics import (
    ACTIVE_MODE_TOPIC,
    CABLE_EXECUTOR_STATUS_TOPIC,
    CABLE_SUPERVISOR_STATUS_TOPIC,
    EXECUTION_DIAGNOSTICS_TOPIC,
    MANUAL_PEN_MODE_TOPIC,
    MODE_DRAW,
    MODE_OFF,
    MODE_TEXT,
    PEN_MODE_AUTO,
    PEN_MODE_DOWN,
    PEN_MODE_UP,
    PRIMITIVE_PATH_PLAN_TOPIC,
    VALID_MODES,
    VALID_MANUAL_PEN_MODES,
)
from wall_climber.shared_config import load_shared_config
from wall_climber.vector_pipeline import VectorPlacement


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
_SKETCH_PREVIEW_MAX_POINTS = 2400
_SKETCH_PREVIEW_CACHE_TTL_SECONDS = 10 * 60
_SKETCH_PREVIEW_CACHE_MAX_ENTRIES = 16
_PREVIEW_CACHE_TTL_SECONDS = 30 * 60
_PREVIEW_CACHE_MAX_ENTRIES = 48
_SKETCH_DRAW_MAX_CANONICAL_COMMANDS = 200_000
_SKETCH_DRAW_MAX_PRIMITIVES = 200_000
_SKETCH_DRAW_MAX_PRIMITIVE_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_REQUIRED_STATUS_KEYS = (
    'cable_executor_status',
    'cable_supervisor_status',
)
_FACE_TEXT_TOPIC = '/wall_climber/face/text'
_FACE_EXPRESSION_TOPIC = '/wall_climber/face/expression'
_FACE_VALID_EXPRESSIONS = {
    'happy',
    'smile',
    'ready',
    'neutral',
    'sleep',
    'sleepy',
    'closed',
    'angry',
    'focus',
    'focused',
    'sad',
    'error',
}
_FACE_MAX_TEXT_CHARS = 18


@dataclass(frozen=True)
class PreviewCacheEntry:
    preview_id: str
    source_type: str
    canonical_plan: CanonicalPathPlan
    canonical_hash: str
    executable_canonical_plan: CanonicalPathPlan
    executable_canonical_hash: str
    primitive_descriptor: dict[str, Any]
    primitive_plan: PrimitivePathPlan
    primitive_hash: str
    execution_preview_svg: str
    execution_hash: str
    settings_hash: str
    metrics: dict[str, Any]
    preview_payload: dict[str, Any]
    commit_request: dict[str, Any]
    created_at_unix: float
    input_type: str
    pipeline_mode: str
    source_hash: str | None
    settings: dict[str, Any]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]
    source_filename: str
    drawing_plan: DrawingPathPlan | None
    command_metadata: tuple[dict[str, Any] | None, ...] | None
    optimizer_stats: dict[str, Any] | None
    route_metadata: dict[str, Any] | None
    curve_fit_payload: dict[str, Any] | None


@dataclass(frozen=True)
class SketchPreviewCacheEntry:
    preview_id: str
    drawing_plan: DrawingPathPlan
    canonical_plan: CanonicalPathPlan
    canonical_hash: str
    preview_geometry_mode: str
    metadata: dict[str, Any]
    warnings: tuple[str, ...]
    created_at_unix: float
    source_filename: str
    parameters: dict[str, Any]
    stroke_count: int
    point_count: int
    canonical_command_count: int


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
        self._executor_diagnostics: dict[str, Any] | None = None

        self._active_mode_pub = self.create_publisher(
            String,
            ACTIVE_MODE_TOPIC,
            _ACTIVE_MODE_QOS,
        )
        self._primitive_path_plan_pub = self.create_publisher(
            PrimitivePathPlan,
            PRIMITIVE_PATH_PLAN_TOPIC,
            10,
        )
        self._manual_pen_mode_pub = self.create_publisher(
            String,
            MANUAL_PEN_MODE_TOPIC,
            _ACTIVE_MODE_QOS,
        )
        self._face_text_pub = self.create_publisher(
            String,
            _FACE_TEXT_TOPIC,
            _ACTIVE_MODE_QOS,
        )
        self._face_expression_pub = self.create_publisher(
            String,
            _FACE_EXPRESSION_TOPIC,
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
        self.create_subscription(
            String,
            EXECUTION_DIAGNOSTICS_TOPIC,
            self._executor_diagnostics_cb,
            _STATUS_TOPIC_QOS,
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

    def _executor_diagnostics_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        with self._lock:
            self._executor_diagnostics = payload

    def _publish_active_mode(self, mode: str) -> None:
        msg = String()
        msg.data = mode
        self._active_mode_pub.publish(msg)

    def _publish_manual_pen_mode(self, mode: str) -> None:
        msg = String()
        msg.data = mode
        self._manual_pen_mode_pub.publish(msg)

    def publish_face_text(self, text: str) -> str:
        value = str(text).strip()[:_FACE_MAX_TEXT_CHARS]
        msg = String()
        msg.data = value
        self._face_text_pub.publish(msg)
        return value

    def publish_face_expression(self, expression: str) -> str:
        value = str(expression).strip().lower()
        if value not in _FACE_VALID_EXPRESSIONS:
            allowed = ', '.join(sorted(_FACE_VALID_EXPRESSIONS))
            raise HTTPException(status_code=422, detail=f'expression must be one of: {allowed}')
        msg = String()
        msg.data = value
        self._face_expression_pub.publish(msg)
        return value

    def runtime_snapshot(self) -> dict[str, Any]:
        with self._lock:
            statuses = dict(self._statuses)
            observed = dict(self._observed_statuses)
            board_info = dict(self._board_info) if self._board_info is not None else None
            active_mode = self._active_mode
            manual_pen_mode = self._manual_pen_mode
            executor_diagnostics = (
                dict(self._executor_diagnostics)
                if self._executor_diagnostics is not None else None
            )
        ready = all(observed.values())
        return {
            'active_mode': active_mode,
            'manual_pen_mode': manual_pen_mode,
            'ready': ready,
            'not_ready_reason': None if ready else 'waiting_for_status_topics',
            'observed_statuses': observed,
            'statuses': statuses,
            'board_info': board_info,
            'executor_diagnostics': executor_diagnostics,
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

    def publish_execution_plan(
        self,
        primitive_plan: PrimitivePathPlan,
        *,
        allowed_modes: tuple[str, ...],
    ) -> dict[str, Any]:
        snapshot = self.ensure_ready()
        if snapshot['active_mode'] not in allowed_modes:
            allowed = ', '.join(allowed_modes)
            raise HTTPException(status_code=409, detail=f'active mode must be one of: {allowed}')
        if snapshot['manual_pen_mode'] != PEN_MODE_AUTO:
            raise HTTPException(status_code=409, detail='manual arm test must be set to auto before drawing')
        if snapshot['statuses']['cable_executor_status'] == 'running':
            raise HTTPException(status_code=409, detail='cable executor is busy')
        self._primitive_path_plan_pub.publish(primitive_plan)
        return {
            'published': 'primitive_path_plan',
            'preferred_transport': 'primitive_path_plan',
            'primitive_transport_published': True,
            'topics': {
                'primitive_path_plan': PRIMITIVE_PATH_PLAN_TOPIC,
            },
        }

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

    def executor_diagnostics_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._executor_diagnostics) if self._executor_diagnostics is not None else None


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
        self._uploads = UploadStore(
            Path.home() / '.ros' / 'wall_climber' / 'uploads',
            theta_ref_provider=lambda: node._shared.draw_execution.fixed_draw_theta_rad,
            max_workers=1,
        )
        self._debug = DebugSnapshotStore()

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
            try:
                self._uploads.shutdown()
            except Exception:
                pass
            if rclpy.ok():
                rclpy.shutdown()

    # ------------------------------------------------------------------
    # Uploads (façade over UploadStore)
    # ------------------------------------------------------------------

    def store_upload(
        self,
        upload: UploadFile,
        content: bytes,
        *,
        upload_details: UploadedVectorFile,
    ) -> dict[str, Any]:
        return self._uploads.store_upload(upload, content, upload_details=upload_details)

    def load_upload(self, upload_id: str) -> tuple[dict[str, Any], bytes]:
        return self._uploads.load_upload(upload_id)

    def ensure_upload_processing(
        self,
        upload_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        payload: bytes | None = None,
    ) -> None:
        self._uploads.ensure_processing(upload_id, metadata=metadata, payload=payload)

    def upload_processing_snapshot(
        self,
        upload_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        payload: bytes | None = None,
    ) -> dict[str, Any]:
        return self._uploads.processing_snapshot(
            upload_id, metadata=metadata, payload=payload,
        )

    def prepared_image_artifact(
        self,
        upload_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        payload: bytes | None = None,
    ) -> PreparedImageArtifact:
        return self._uploads.prepared_image_artifact(
            upload_id, metadata=metadata, payload=payload,
        )

    def record_last_plan_debug(self, payload: dict[str, Any]) -> None:
        self._debug.record_plan(payload)

    def record_last_execution_debug(self, payload: dict[str, Any]) -> None:
        self._debug.record_execution(payload)

    def record_last_curve_fit_debug(self, payload: dict[str, Any]) -> None:
        self._debug.record_curve_fit(payload)

    def last_plan_debug_snapshot(self) -> dict[str, Any] | None:
        return self._debug.plan_snapshot()

    def last_execution_debug_snapshot(self) -> dict[str, Any] | None:
        return self._debug.execution_snapshot()

    def last_curve_fit_debug_snapshot(self) -> dict[str, Any] | None:
        return self._debug.curve_fit_snapshot()


def _require_json_object(raw: Any, name: str) -> dict[str, Any]:
    return _http.require_json_object(raw, name)


async def _load_json_request(
    request: Request,
    *,
    name: str,
    max_bytes: int,
) -> dict[str, Any]:
    return await _http.load_json_request(request, name=name, max_bytes=max_bytes)


def _reject_extra_fields(payload: dict[str, Any], allowed: set[str], name: str) -> None:
    _http.reject_extra_fields(payload, allowed, name)


def _validate_text_value(value: Any, field_name: str) -> str:
    return _http.validate_text_value(
        value,
        field_name,
        max_chars=_MAX_TEXT_CHARS,
        max_bytes=_MAX_TEXT_BYTES,
    )


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
    return _http.coerce_float(
        value, field_name=field_name, minimum=minimum, maximum=maximum,
    )


def _coerce_int(
    value: Any,
    *,
    field_name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    return _http.coerce_int(
        value, field_name=field_name, minimum=minimum, maximum=maximum,
    )


def _coerce_bool(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    return _http.coerce_bool(value, field_name=field_name, default=default)


def _validate_upload_id(raw_upload_id: Any) -> str:
    return _http.validate_upload_id(raw_upload_id)


def _validate_preview_id(raw_preview_id: Any) -> str:
    return _http.validate_preview_id(raw_preview_id)


def _stable_float(value: float, *, precision: int = 7) -> float:
    return _http.stable_float(value, precision=precision)


def _stable_point_payload(point: tuple[float, float]) -> list[float]:
    return _http.stable_point_payload(point)


def _stable_payload(value: Any) -> Any:
    return _http.stable_payload(value)


def _stable_hash(value: Any) -> str:
    return _http.stable_hash(value)


def settings_hash(settings: dict[str, Any]) -> str:
    return _http.settings_hash(settings)


def _canonical_command_payload(command: CanonicalCommand) -> dict[str, Any]:
    if isinstance(command, PenUp):
        return {'type': 'pen_up'}
    if isinstance(command, PenDown):
        return {'type': 'pen_down'}
    if isinstance(command, TravelMove):
        return {
            'type': 'travel',
            'start': _stable_point_payload(command.start),
            'end': _stable_point_payload(command.end),
        }
    if isinstance(command, LineSegment):
        return {
            'type': 'line',
            'start': _stable_point_payload(command.start),
            'end': _stable_point_payload(command.end),
        }
    if isinstance(command, ArcSegment):
        return {
            'type': 'arc',
            'center': _stable_point_payload(command.center),
            'radius': _stable_float(command.radius),
            'start_angle_rad': _stable_float(command.start_angle_rad),
            'sweep_angle_rad': _stable_float(command.sweep_angle_rad),
        }
    if isinstance(command, QuadraticBezier):
        return {
            'type': 'quadratic',
            'start': _stable_point_payload(command.start),
            'control': _stable_point_payload(command.control),
            'end': _stable_point_payload(command.end),
        }
    if isinstance(command, CubicBezier):
        return {
            'type': 'cubic',
            'start': _stable_point_payload(command.start),
            'control1': _stable_point_payload(command.control1),
            'control2': _stable_point_payload(command.control2),
            'end': _stable_point_payload(command.end),
        }
    raise ValueError(f'Unsupported canonical command {type(command)!r}.')


def canonical_plan_stable_payload(plan: CanonicalPathPlan) -> dict[str, Any]:
    return {
        'frame': str(plan.frame),
        'theta_ref': _stable_float(plan.theta_ref),
        'commands': [
            _canonical_command_payload(command)
            for command in plan.commands
        ],
    }


def canonical_plan_hash(plan: CanonicalPathPlan) -> str:
    payload = canonical_plan_stable_payload(plan)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _content_hash(content: bytes | str | dict[str, Any] | None) -> str | None:
    if content is None:
        return None
    if isinstance(content, bytes):
        payload = content
    elif isinstance(content, str):
        payload = content.encode('utf-8')
    else:
        payload = json.dumps(content, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


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
    if normalized not in {'relief_singleline', 'hershey_sans_1', 'dejavu_sans'}:
        raise ValueError(
            'font_source must be one of ["relief_singleline", "hershey_sans_1", "dejavu_sans"]'
        )
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


def _preview_sampling_policy(shared_config) -> SamplingPolicy:
    draw_defaults = shared_config.draw_execution
    return SamplingPolicy(
        curve_tolerance_m=max(0.006, float(draw_defaults.draw_resample_step_m) * 1.75),
        draw_step_m=max(0.006, float(draw_defaults.draw_resample_step_m) * 1.75),
        travel_step_m=max(0.006, float(draw_defaults.travel_resample_step_m) * 1.35),
        max_heading_delta_rad=0.28,
        label='preview',
    )


def _runtime_sampling_policy(shared_config) -> SamplingPolicy:
    draw_defaults = shared_config.draw_execution
    return SamplingPolicy(
        curve_tolerance_m=max(1.0e-4, float(draw_defaults.draw_resample_step_m)),
        draw_step_m=max(1.0e-4, float(draw_defaults.draw_resample_step_m)),
        travel_step_m=max(1.0e-4, float(draw_defaults.travel_resample_step_m)),
        max_heading_delta_rad=0.16,
        label='runtime',
    )


def _draw_optimization_policy(
    shared_config,
    *,
    label: str,
    reorder_units: bool,
    fit_arcs: bool = False,
    enable_hatch_ordering: bool = False,
    cluster_units: bool = False,
) -> CanonicalOptimizationPolicy:
    draw_defaults = shared_config.draw_execution
    tiny = max(2.0e-4, float(draw_defaults.draw_path_simplify_tolerance_m) * 2.0)
    return CanonicalOptimizationPolicy(
        label=label,
        reorder_units=bool(reorder_units),
        cluster_units=bool(cluster_units),
        merge_travel_moves=True,
        fit_arcs=bool(fit_arcs),
        enable_hatch_ordering=bool(enable_hatch_ordering),
        tiny_primitive_m=tiny,
        arc_fit_tolerance_m=max(tiny, float(draw_defaults.draw_path_simplify_tolerance_m) * 2.5),
        merge_distance_tolerance_m=max(1.0e-5, tiny * 0.25),
        dedupe_precision_m=max(1.0e-5, tiny * 0.5),
        cluster_cell_size_m=0.26,
    )


def _sketch_draw_optimization_policy(shared_config) -> CanonicalOptimizationPolicy:
    draw_defaults = shared_config.draw_execution
    tiny = max(2.0e-4, float(draw_defaults.draw_path_simplify_tolerance_m) * 2.0)
    return CanonicalOptimizationPolicy(
        label='sketch_centerline_draw',
        merge_collinear_lines=False,
        reorder_units=True,
        cluster_units=True,
        merge_travel_moves=True,
        remove_duplicate_units=True,
        prune_tiny_primitives=False,
        fit_arcs=False,
        enable_hatch_ordering=False,
        tiny_primitive_m=tiny,
        merge_distance_tolerance_m=max(1.0e-5, tiny * 0.25),
        dedupe_precision_m=max(1.0e-5, tiny * 0.5),
        cluster_cell_size_m=0.26,
    )


def _sampling_validation_step_m(policy: SamplingPolicy) -> float:
    candidates = [
        float(policy.curve_tolerance_m),
        float(policy.draw_step_m) if policy.draw_step_m is not None else None,
        float(policy.travel_step_m) if policy.travel_step_m is not None else None,
    ]
    steps = [max(1.0e-4, value) for value in candidates if value is not None]
    return min(steps) if steps else 0.01


def _validated_runtime_sampled_paths(
    canonical_plan: CanonicalPathPlan,
    *,
    writable_bounds: dict[str, float],
    shared_config,
    sampling_policy: SamplingPolicy,
):
    segments = canonical_plan_to_sampled_paths(
        canonical_plan,
        sampling_policy=sampling_policy,
    )
    if not segments:
        raise HTTPException(status_code=422, detail='execution payload has no drawable segments')
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
        if _interpolated_outside_safe_workspace_count(
            (segment.points,),
            shared_config,
            step_m=_sampling_validation_step_m(sampling_policy),
        ) != 0:
            raise HTTPException(
                status_code=422,
                detail=f'draw segment[{index}] exits the configured safe cable workspace',
            )
    return segments


def _preview_payload_from_strokes(
    placed_strokes: tuple[tuple[tuple[float, float], ...], ...],
    placement_result,
    *,
    outside_safe_points: int,
    normalized_plan: dict[str, Any] | None = None,
    canonical_plan: CanonicalPathPlan | None = None,
    preview_sampling_policy: SamplingPolicy | None = None,
    runtime_sampling_policy: SamplingPolicy | None = None,
) -> dict[str, Any]:
    if canonical_plan is None:
        preview_strokes = [
            [[float(point[0]), float(point[1])] for point in stroke]
            for stroke in placed_strokes
        ]
        diagnostics = None
    else:
        preview_policy = preview_sampling_policy or SamplingPolicy(label='preview')
        runtime_policy = runtime_sampling_policy or preview_policy
        preview_draw_strokes = canonical_plan_to_draw_strokes(
            canonical_plan,
            sampling_policy=preview_policy,
        )
        preview_strokes = [
            [[float(point[0]), float(point[1])] for point in stroke]
            for stroke in preview_draw_strokes
        ]
        diagnostics = canonical_plan_diagnostics(
            canonical_plan,
            preview_sampling_policy=preview_policy,
            runtime_sampling_policy=runtime_policy,
        )
    stats = stroke_stats(
        tuple(
            tuple((float(point[0]), float(point[1])) for point in stroke)
            for stroke in (
                tuple(tuple(tuple(point) for point in stroke) for stroke in placed_strokes)
                if canonical_plan is None else preview_draw_strokes
            )
        )
    )

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
        'diagnostics': diagnostics,
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


def _build_primitive_path_plan_message(
    canonical_plan: CanonicalPathPlan,
) -> PrimitivePathPlan:
    descriptor = canonical_plan_to_primitive_path_plan(canonical_plan)
    return _primitive_path_plan_message_from_descriptor(descriptor)


def _primitive_path_plan_message_from_descriptor(
    descriptor: dict[str, Any],
) -> PrimitivePathPlan:
    def board_point_from_payload(point: dict[str, Any]):
        try:
            return BoardPoint(
                x=float(point['x']),
                y=float(point['y']),
            )
        except TypeError:
            board_point = BoardPoint()
            board_point.x = float(point['x'])
            board_point.y = float(point['y'])
            return board_point

    plan = PrimitivePathPlan()
    plan.frame = str(descriptor['frame'])
    plan.theta_ref = float(descriptor['theta_ref'])
    if not hasattr(plan, 'primitives'):
        plan.primitives = []
    type_codes = {
        'PEN_UP': getattr(PathPrimitive, 'PEN_UP', 1),
        'PEN_DOWN': getattr(PathPrimitive, 'PEN_DOWN', 2),
        'TRAVEL_MOVE': getattr(PathPrimitive, 'TRAVEL_MOVE', 3),
        'LINE_SEGMENT': getattr(PathPrimitive, 'LINE_SEGMENT', 4),
        'ARC_SEGMENT': getattr(PathPrimitive, 'ARC_SEGMENT', 5),
        'QUADRATIC_BEZIER': getattr(PathPrimitive, 'QUADRATIC_BEZIER', 6),
        'CUBIC_BEZIER': getattr(PathPrimitive, 'CUBIC_BEZIER', 7),
    }
    for primitive_descriptor in descriptor['primitives']:
        primitive = PathPrimitive()
        primitive.type = int(type_codes[str(primitive_descriptor['type'])])
        for field_name in ('start', 'end', 'control1', 'control2', 'center'):
            point = primitive_descriptor[field_name]
            setattr(
                primitive,
                field_name,
                board_point_from_payload(point),
            )
        primitive.radius = float(primitive_descriptor['radius'])
        primitive.start_angle_rad = float(primitive_descriptor['start_angle_rad'])
        primitive.sweep_angle_rad = float(primitive_descriptor['sweep_angle_rad'])
        primitive.clockwise = bool(primitive_descriptor['clockwise'])
        primitive.pen_down = bool(primitive_descriptor['pen_down'])
        plan.primitives.append(primitive)
    return plan


def _build_execution_transport_message(
    canonical_plan: CanonicalPathPlan,
    *,
    writable_bounds: dict[str, float],
    shared_config,
    sampling_policy: SamplingPolicy,
) -> PrimitivePathPlan:
    _validated_runtime_sampled_paths(
        canonical_plan,
        writable_bounds=writable_bounds,
        shared_config=shared_config,
        sampling_policy=sampling_policy,
    )
    return _build_primitive_path_plan_message(canonical_plan)


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


def _normalize_stroke_payload(raw: Any, writable_bounds: dict[str, float]) -> str:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail='stroke payload body must be a JSON object')
    _reject_extra_fields(raw, {'frame', 'strokes'}, 'stroke payload')
    if raw.get('frame') != 'board':
        raise HTTPException(status_code=422, detail='stroke payload.frame must be exactly "board"')
    strokes = raw.get('strokes')
    if not isinstance(strokes, list) or not strokes:
        raise HTTPException(status_code=422, detail='stroke payload.strokes must be a non-empty list')
    if len(strokes) > _MAX_DRAW_STROKES:
        raise HTTPException(status_code=413, detail='stroke payload exceeds the maximum number of strokes')

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
            raise HTTPException(status_code=413, detail='stroke payload exceeds the maximum total point budget')
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
        raise HTTPException(status_code=413, detail='stroke payload exceeds the maximum allowed payload size')
    return encoded


def _validate_upload(upload: UploadFile, content: bytes) -> UploadedVectorFile:
    if not content:
        raise HTTPException(status_code=422, detail='uploaded file is empty')
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='uploaded file exceeds the maximum allowed size')
    try:
        return classify_uploaded_vector_file(upload.filename, upload.content_type, content)
    except ValueError as exc:
        detail = str(exc)
        status_code = 415 if 'unsupported upload content type' in detail else 422
        raise HTTPException(status_code=status_code, detail=detail)


def _validate_sketch_upload(upload: UploadFile, content: bytes) -> None:
    if not content:
        raise HTTPException(status_code=422, detail='uploaded file is empty')
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='uploaded file exceeds the maximum allowed size')

    suffix = Path(upload.filename or '').suffix.lower()
    normalized_type = str(upload.content_type or '').split(';', 1)[0].strip().lower()
    if suffix not in {'.png', '.jpg', '.jpeg', '.webp'} and normalized_type not in {'image/png', 'image/jpeg', 'image/webp'}:
        raise HTTPException(status_code=415, detail='sketch preview accepts PNG, JPG, or WebP uploads only')


def _image_value_error_to_http(exc: ValueError) -> HTTPException:
    detail = str(exc)
    lowered = detail.lower()
    if (
        'could not be decoded' in lowered
        or 'failed to decode' in lowered
        or 'image payload is empty' in lowered
    ):
        return HTTPException(
            status_code=400,
            detail='Unable to decode uploaded image. Please upload a valid PNG, JPG, or WebP image.',
        )
    return HTTPException(status_code=422, detail=detail)


def _drawing_plan_bounds(plan) -> dict[str, float]:
    points = [point for stroke in plan.strokes for point in stroke.points]
    if not points:
        raise ValueError('DrawingPathPlan has no points.')
    x_min = min(float(point.x) for point in points)
    x_max = max(float(point.x) for point in points)
    y_min = min(float(point.y) for point in points)
    y_max = max(float(point.y) for point in points)
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'width': x_max - x_min,
        'height': y_max - y_min,
    }


def _downsample_points_for_preview(points, *, stride: int) -> list:
    selected = list(points[::stride])
    if selected and selected[-1] != points[-1]:
        selected.append(points[-1])
    if len(selected) < 2 and len(points) >= 2:
        selected = [points[0], points[-1]]
    return selected


def _sketch_preview_strokes(plan, *, max_points: int | None = None) -> dict[str, Any]:
    max_points = _SKETCH_PREVIEW_MAX_POINTS if max_points is None else max_points
    max_points = max(2, int(max_points))
    original_point_count = sum(len(stroke.points) for stroke in plan.strokes)
    stride = max(1, int(numpy.ceil(original_point_count / float(max_points)))) if original_point_count else 1
    preview_strokes: list[list[list[float]]] = []
    returned_point_count = 0

    for stroke in plan.strokes:
        selected = _downsample_points_for_preview(tuple(stroke.points), stride=stride)
        remaining = max_points - returned_point_count
        if remaining <= 1:
            break
        if len(selected) > remaining:
            selected = selected[:remaining]
        if len(selected) < 2:
            continue
        preview_strokes.append([[float(point.x), float(point.y)] for point in selected])
        returned_point_count += len(selected)

    return {
        'strokes': preview_strokes,
        'max_points': max_points,
        'returned_point_count': returned_point_count,
        'original_point_count': original_point_count,
        'truncated': returned_point_count < original_point_count,
    }


def _svg_number(value: float) -> str:
    return f'{float(value):.6g}'


def _sketch_preview_svg(
    preview_strokes: list[list[list[float]]],
    *,
    board_width_m: float,
    board_height_m: float,
) -> str:
    stroke_width = max(float(board_width_m), float(board_height_m)) / 360.0
    polylines: list[str] = []
    for stroke in preview_strokes:
        if len(stroke) < 2:
            continue
        points = ' '.join(
            f'{_svg_number(point[0])},{_svg_number(point[1])}'
            for point in stroke
        )
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="#111827" '
            f'stroke-width="{_svg_number(stroke_width)}" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    body = ''.join(polylines)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {_svg_number(board_width_m)} {_svg_number(board_height_m)}" '
        f'width="{_svg_number(board_width_m)}" height="{_svg_number(board_height_m)}" '
        'role="img" aria-label="Sketch centerline preview">'
        f'<rect x="0" y="0" width="{_svg_number(board_width_m)}" '
        f'height="{_svg_number(board_height_m)}" fill="white"/>'
        f'{body}</svg>'
    )


def _command_start(command: CanonicalCommand) -> tuple[float, float] | None:
    if isinstance(command, LineSegment):
        return command.start
    if isinstance(command, QuadraticBezier):
        return command.start
    if isinstance(command, CubicBezier):
        return command.start
    return None


def _smooth_sketch_preview_svg(
    canonical_plan: CanonicalPathPlan,
    *,
    board_width_m: float,
    board_height_m: float,
) -> str:
    stroke_width = max(float(board_width_m), float(board_height_m)) / 360.0
    paths: list[str] = []
    current: list[str] = []
    pen_down = False

    def flush_current() -> None:
        if current:
            paths.append(
                '<path d="'
                + ' '.join(current)
                + f'" fill="none" stroke="#111827" stroke-width="{_svg_number(stroke_width)}" '
                + 'stroke-linecap="round" stroke-linejoin="round"/>'
            )
            current.clear()

    for command in canonical_plan.commands:
        if isinstance(command, PenDown):
            flush_current()
            pen_down = True
            continue
        if isinstance(command, PenUp):
            flush_current()
            pen_down = False
            continue
        if isinstance(command, TravelMove):
            continue
        if not pen_down:
            continue
        start = _command_start(command)
        if start is None:
            continue
        if not current:
            current.append(f'M {_svg_number(start[0])} {_svg_number(start[1])}')
        if isinstance(command, LineSegment):
            current.append(f'L {_svg_number(command.end[0])} {_svg_number(command.end[1])}')
        elif isinstance(command, QuadraticBezier):
            current.append(
                f'Q {_svg_number(command.control[0])} {_svg_number(command.control[1])} '
                f'{_svg_number(command.end[0])} {_svg_number(command.end[1])}'
            )
        elif isinstance(command, CubicBezier):
            current.append(
                f'C {_svg_number(command.control1[0])} {_svg_number(command.control1[1])} '
                f'{_svg_number(command.control2[0])} {_svg_number(command.control2[1])} '
                f'{_svg_number(command.end[0])} {_svg_number(command.end[1])}'
            )
    flush_current()
    body = ''.join(paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {_svg_number(board_width_m)} {_svg_number(board_height_m)}" '
        f'width="{_svg_number(board_width_m)}" height="{_svg_number(board_height_m)}" '
        'role="img" aria-label="Sketch centerline smooth curve preview">'
        f'<rect x="0" y="0" width="{_svg_number(board_width_m)}" '
        f'height="{_svg_number(board_height_m)}" fill="white"/>'
        f'{body}</svg>'
    )


def _sampled_paths_bounds(sampled_paths) -> dict[str, float] | None:
    points = [
        point
        for sampled in sampled_paths
        if sampled.draw
        for point in sampled.points
    ]
    if not points:
        return None
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    return {
        'x_min': min(x_values),
        'x_max': max(x_values),
        'y_min': min(y_values),
        'y_max': max(y_values),
        'width': max(x_values) - min(x_values),
        'height': max(y_values) - min(y_values),
    }


def _sampled_paths_length(sampled_paths, *, draw: bool) -> float:
    total = 0.0
    for sampled in sampled_paths:
        if bool(sampled.draw) != bool(draw):
            continue
        for index in range(1, len(sampled.points)):
            start = sampled.points[index - 1]
            end = sampled.points[index]
            total += float(numpy.hypot(end[0] - start[0], end[1] - start[1]))
    return total


def _sampled_paths_stable_payload(sampled_paths) -> list[dict[str, Any]]:
    return [
        {
            'draw': bool(sampled.draw),
            'points': [
                _stable_point_payload((float(point[0]), float(point[1])))
                for point in sampled.points
            ],
        }
        for sampled in sampled_paths
    ]


def _execution_preview_svg_from_sampled_paths(
    sampled_paths,
    *,
    board_width_m: float,
    board_height_m: float,
) -> str:
    draw_strokes = [
        [
            [float(point[0]), float(point[1])]
            for point in sampled.points
        ]
        for sampled in sampled_paths
        if sampled.draw and len(sampled.points) >= 2
    ]
    return _sketch_preview_svg(
        draw_strokes,
        board_width_m=board_width_m,
        board_height_m=board_height_m,
    )


def _canonical_primitive_counts(plan: CanonicalPathPlan) -> dict[str, int]:
    line_count = sum(isinstance(command, LineSegment) for command in plan.commands)
    quadratic_count = sum(isinstance(command, QuadraticBezier) for command in plan.commands)
    cubic_count = sum(isinstance(command, CubicBezier) for command in plan.commands)
    arc_count = sum(isinstance(command, ArcSegment) for command in plan.commands)
    return {
        'line_primitive_count': int(line_count),
        'quadratic_primitive_count': int(quadratic_count),
        'cubic_primitive_count': int(cubic_count),
        'arc_primitive_count': int(arc_count),
        'curve_primitive_count': int(quadratic_count + cubic_count + arc_count),
    }


def _canonical_geometry_metrics(plan: CanonicalPathPlan) -> dict[str, int]:
    counts = _canonical_primitive_counts(plan)
    return {
        'line_count': int(counts['line_primitive_count']),
        'quadratic_count': int(counts['quadratic_primitive_count']),
        'cubic_count': int(counts['cubic_primitive_count']),
        'arc_count': int(counts['arc_primitive_count']),
        'total_curve_count': int(counts['curve_primitive_count']),
    }


def _executable_geometry_metrics(sampled_paths) -> dict[str, int]:
    draw_paths = [sampled for sampled in sampled_paths if sampled.draw]
    sampled_point_count = sum(len(sampled.points) for sampled in draw_paths)
    sampled_segment_count = sum(max(0, len(sampled.points) - 1) for sampled in draw_paths)
    return {
        'draw_path_count': int(len(draw_paths)),
        'sampled_point_count': int(sampled_point_count),
        'sampled_segment_count': int(sampled_segment_count),
    }


def _canonical_transport_size_summary(plan: CanonicalPathPlan) -> dict[str, int]:
    descriptor = canonical_plan_to_primitive_path_plan(plan)
    primitive_count = len(descriptor['primitives'])
    descriptor_bytes = len(
        json.dumps(descriptor, separators=(',', ':'), sort_keys=True).encode('utf-8')
    )
    return {
        'canonical_command_count': int(len(plan.commands)),
        'primitive_count': int(primitive_count),
        'primitive_descriptor_bytes': int(descriptor_bytes),
    }


def _enforce_sketch_draw_size_limits(summary: dict[str, int]) -> None:
    limits = {
        'max_canonical_command_count': int(_SKETCH_DRAW_MAX_CANONICAL_COMMANDS),
        'max_primitive_count': int(_SKETCH_DRAW_MAX_PRIMITIVES),
        'max_primitive_descriptor_bytes': int(_SKETCH_DRAW_MAX_PRIMITIVE_DESCRIPTOR_BYTES),
    }
    violations: list[str] = []
    if int(summary['canonical_command_count']) > limits['max_canonical_command_count']:
        violations.append('canonical_command_count')
    if int(summary['primitive_count']) > limits['max_primitive_count']:
        violations.append('primitive_count')
    if int(summary['primitive_descriptor_bytes']) > limits['max_primitive_descriptor_bytes']:
        violations.append('primitive_descriptor_bytes')
    if violations:
        raise HTTPException(
            status_code=413,
            detail={
                'error': 'sketch preview plan is too large for the existing execution transport',
                'violations': violations,
                'counts': summary,
                'limits': limits,
            },
        )


def _bounds_payload(bounds: dict[str, float]) -> dict[str, float]:
    return {
        'x_min': float(bounds['x_min']),
        'x_max': float(bounds['x_max']),
        'y_min': float(bounds['y_min']),
        'y_max': float(bounds['y_max']),
        'width': float(bounds['x_max']) - float(bounds['x_min']),
        'height': float(bounds['y_max']) - float(bounds['y_min']),
    }


def _slowest_timing_stage(timing: dict[str, Any]) -> dict[str, Any] | None:
    candidates = {
        key: float(value)
        for key, value in timing.items()
        if key.endswith('_time_ms') and key != 'preview_total_time_ms'
    }
    if not candidates:
        return None
    key, value = max(candidates.items(), key=lambda item: item[1])
    return {'stage': key.removesuffix('_time_ms'), 'time_ms': float(value)}


def _sketch_preview_response(
    plan,
    *,
    preview_id: str | None = None,
    canonical_plan: CanonicalPathPlan,
    board_width_m: float,
    board_height_m: float,
    preview_geometry_mode: str,
    use_smooth_svg: bool,
    curve_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preview = _sketch_preview_strokes(plan)
    metadata = dict(plan.metadata)
    base_metadata_warnings = tuple(str(item) for item in metadata.get('warnings') or ())
    curve_metadata = dict(curve_metadata or {})
    metadata.update(curve_metadata)
    metadata['preview_geometry_mode'] = preview_geometry_mode
    timing = dict(metadata.get('timing') or {})
    timing['slowest_stage'] = _slowest_timing_stage(timing)
    metadata['timing'] = timing
    if 'curve_fit_time_ms' in timing:
        metadata['curve_fit_time_ms'] = float(timing['curve_fit_time_ms'])
    primitive_counts = _canonical_primitive_counts(canonical_plan)
    metadata.update(primitive_counts)
    warnings = list(plan.metrics.warnings)
    warnings.extend(base_metadata_warnings)
    warnings.extend(str(item) for item in curve_metadata.get('warnings') or ())
    deduped_warnings = list(dict.fromkeys(warnings))
    metadata['warnings'] = tuple(deduped_warnings)
    preview_svg = (
        _smooth_sketch_preview_svg(
            canonical_plan,
            board_width_m=board_width_m,
            board_height_m=board_height_m,
        )
        if use_smooth_svg
        else _sketch_preview_svg(
            preview['strokes'],
            board_width_m=board_width_m,
            board_height_m=board_height_m,
        )
    )
    return {
        'ok': True,
        'preview_id': preview_id,
        'mode': plan.mode.value,
        'stroke_count': len(plan.strokes),
        'point_count': sum(len(stroke.points) for stroke in plan.strokes),
        'canonical_command_count': len(canonical_plan.commands),
        'metrics': asdict(plan.metrics),
        'metadata': metadata,
        'bounds': _drawing_plan_bounds(plan),
        'warnings': deduped_warnings,
        'preview_svg': preview_svg,
        'preview': preview,
    }


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
    # Defence in depth: ensure the candidate path itself never contains traversal
    # parts after lexical normalisation. We intentionally do NOT call
    # ``candidate.resolve()`` because colcon installs Webots/web assets as
    # symlinks pointing back into ``src/``; resolving and checking against
    # web_dir would reject those legitimate symlinks. The ``..``/``.`` parts
    # check above is what stops path traversal regardless of symlinks.
    return candidate


def create_app(runtime: BackendRuntime) -> FastAPI:
    app = FastAPI(title='Four-Cable Drawing Robot UI Backend', version='1.0.0')
    shared = load_shared_config()
    text_layout_defaults = shared.text_layout
    draw_execution_defaults = shared.draw_execution
    preview_sampling_policy = _preview_sampling_policy(shared)
    runtime_sampling_policy = _runtime_sampling_policy(shared)
    svg_optimization_policy = _draw_optimization_policy(
        shared,
        label='svg',
        reorder_units=True,
        fit_arcs=True,
    )
    image_optimization_policy = _draw_optimization_policy(
        shared,
        label='image',
        reorder_units=True,
        fit_arcs=True,
        enable_hatch_ordering=True,
        cluster_units=True,
    )
    sketch_draw_optimization_policy = _sketch_draw_optimization_policy(shared)
    preview_cache: TTLCache[PreviewCacheEntry] = TTLCache(
        max_entries=int(_PREVIEW_CACHE_MAX_ENTRIES),
        ttl_seconds=float(_PREVIEW_CACHE_TTL_SECONDS),
    )
    sketch_preview_cache: TTLCache[SketchPreviewCacheEntry] = TTLCache(
        max_entries=int(_SKETCH_PREVIEW_CACHE_MAX_ENTRIES),
        ttl_seconds=float(_SKETCH_PREVIEW_CACHE_TTL_SECONDS),
    )

    def _preview_optimization_policy(source_type: str) -> CanonicalOptimizationPolicy:
        if source_type == 'sketch_centerline':
            return sketch_draw_optimization_policy
        if source_type == 'image':
            return image_optimization_policy
        if source_type == 'svg':
            return svg_optimization_policy
        return _draw_optimization_policy(
            shared,
            label=f'{source_type}_preview',
            reorder_units=True,
            fit_arcs=False,
        )

    def _preview_allowed_modes(source_type: str) -> tuple[str, ...]:
        return (MODE_TEXT,) if source_type == 'text' else (MODE_DRAW,)

    def _carriage_safe_writable_bounds_for_sketch() -> dict[str, float]:
        try:
            writable_bounds = runtime.node.carriage_safe_writable_bounds()
        except Exception:
            writable_bounds = shared.carriage_safe_writable_bounds()
        try:
            safe_workspace_bounds = runtime.node.carriage_safe_safe_bounds()
        except Exception:
            safe_workspace_bounds = shared.carriage_safe_workspace_bounds()
        bounds = {
            'x_min': max(float(writable_bounds['x_min']), float(safe_workspace_bounds['x_min'])),
            'x_max': min(float(writable_bounds['x_max']), float(safe_workspace_bounds['x_max'])),
            'y_min': max(float(writable_bounds['y_min']), float(safe_workspace_bounds['y_min'])),
            'y_max': min(float(writable_bounds['y_max']), float(safe_workspace_bounds['y_max'])),
        }
        return _bounds_payload(bounds)

    def _board_bounds_for_sketch() -> dict[str, float]:
        return {
            'x_min': 0.0,
            'x_max': float(shared.board.width),
            'y_min': 0.0,
            'y_max': float(shared.board.height),
            'width': float(shared.board.width),
            'height': float(shared.board.height),
        }

    def _preview_writable_bounds_for_source(source_type: str) -> dict[str, float]:
        if source_type == 'sketch_centerline':
            return _carriage_safe_writable_bounds_for_sketch()
        return runtime.node.carriage_safe_writable_bounds()

    def _normalize_path_optimizer(value: Any, *, field_name: str = 'path_optimizer') -> str:
        optimizer = str('internal' if value in (None, '') else value).strip().lower()
        if optimizer in {'off', 'false', 'disabled'}:
            optimizer = 'none'
        if optimizer not in {'internal', 'vpype', 'none'}:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name} must be one of: internal, vpype, none",
            )
        return optimizer

    def _tiny_detail_policy_for_preview(
        source_type: str,
        settings_payload: dict[str, Any],
    ) -> dict[str, Any]:
        raw_settings = settings_payload.get('settings')
        settings = raw_settings if isinstance(raw_settings, dict) else {}
        def setting_value(key: str, default: Any) -> Any:
            value = settings.get(key, default)
            return default if value is None else value

        eligible = str(source_type) == 'sketch_centerline'
        if eligible:
            preserve = _coerce_bool(
                settings.get('preserve_tiny_details'),
                field_name='preserve_tiny_details',
                default=True,
            )
        else:
            preserve = False
        runtime_draw_step = runtime_sampling_policy.draw_step_m
        default_min_feature = max(
            0.0035,
            float(runtime_draw_step) * 1.5 if runtime_draw_step is not None else 0.0045,
        )
        minimum_feature = _coerce_float(
            setting_value('minimum_drawable_feature_m', default_min_feature),
            field_name='minimum_drawable_feature_m',
            minimum=0.0005,
            maximum=0.03,
        )
        candidate_max = _coerce_float(
            setting_value('tiny_detail_candidate_max_feature_m', minimum_feature * 0.75),
            field_name='tiny_detail_candidate_max_feature_m',
            minimum=0.0001,
            maximum=0.03,
        )
        expand_mode = str(setting_value('tiny_detail_expand_mode', 'micro_cross')).strip().lower()
        if expand_mode not in {'micro_cross', 'micro_loop'}:
            raise HTTPException(
                status_code=422,
                detail="tiny_detail_expand_mode must be 'micro_cross' or 'micro_loop'",
            )
        max_expansions = _coerce_int(
            setting_value('tiny_detail_max_expansions', 512),
            field_name='tiny_detail_max_expansions',
            minimum=0,
            maximum=10_000,
        )
        context_radius = _coerce_float(
            setting_value('tiny_detail_context_radius_m', 0.08),
            field_name='tiny_detail_context_radius_m',
            minimum=0.0,
            maximum=0.5,
        )
        return {
            'eligible': bool(eligible),
            'preserve_tiny_details': bool(preserve),
            'minimum_drawable_feature_m': float(minimum_feature),
            'tiny_detail_candidate_max_feature_m': float(candidate_max),
            'tiny_detail_expand_mode': expand_mode,
            'tiny_detail_max_expansions': int(max_expansions),
            'tiny_detail_context_radius_m': float(context_radius),
        }

    def _build_executable_preview_payload(
        canonical_plan: CanonicalPathPlan,
        *,
        source_type: str,
        settings_payload: dict[str, Any],
        writable_bounds: dict[str, float],
        optimize_stroke_order: bool,
        path_optimizer: str = 'internal',
        existing_optimizer_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        executable_plan = canonical_plan
        path_optimizer = _normalize_path_optimizer(path_optimizer)
        tiny_detail_policy = _tiny_detail_policy_for_preview(source_type, settings_payload)
        effective_settings_payload = dict(settings_payload)
        effective_settings_payload['tiny_detail_policy'] = tiny_detail_policy
        effective_settings_payload['path_optimizer'] = path_optimizer
        tiny_detail_metrics = {
            'preserve_tiny_details': bool(tiny_detail_policy['preserve_tiny_details']),
            'tiny_detail_expand_mode': tiny_detail_policy['tiny_detail_expand_mode'],
            'minimum_drawable_feature_m': float(tiny_detail_policy['minimum_drawable_feature_m']),
            'tiny_detail_candidate_max_feature_m': float(
                tiny_detail_policy['tiny_detail_candidate_max_feature_m']
            ),
            'tiny_detail_max_expansions': int(tiny_detail_policy['tiny_detail_max_expansions']),
            'tiny_detail_context_radius_m': float(tiny_detail_policy['tiny_detail_context_radius_m']),
            'tiny_details_detected': 0,
            'tiny_details_preserved': 0,
            'tiny_details_expanded': 0,
            'tiny_details_skipped_by_limit': 0,
            'tiny_details_skipped_as_isolated': 0,
            'tiny_details_expansion_added_commands': 0,
        }
        if bool(tiny_detail_policy['preserve_tiny_details']):
            try:
                tiny_detail_result = expand_tiny_details_in_canonical_plan(
                    executable_plan,
                    preserve=True,
                    minimum_drawable_feature_m=float(tiny_detail_policy['minimum_drawable_feature_m']),
                    candidate_max_feature_m=float(
                        tiny_detail_policy['tiny_detail_candidate_max_feature_m']
                    ),
                    expand_mode=str(tiny_detail_policy['tiny_detail_expand_mode']),
                    max_expansions=int(tiny_detail_policy['tiny_detail_max_expansions']),
                    context_radius_m=float(tiny_detail_policy['tiny_detail_context_radius_m']),
                    bounds=writable_bounds,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            executable_plan = tiny_detail_result.plan
            tiny_detail_metrics = dict(tiny_detail_result.metrics)
        optimization_stats = dict(existing_optimizer_stats or {})
        optimization_ms = 0.0
        pre_optimization_sampled_paths = None
        optimizer_warnings: list[str] = []
        optimizer_available = True
        optimizer_used = 'none'
        if optimize_stroke_order and path_optimizer != 'none':
            pre_optimization_sampled_paths = _validated_runtime_sampled_paths(
                executable_plan,
                writable_bounds=writable_bounds,
                shared_config=shared,
                sampling_policy=runtime_sampling_policy,
            )
            optimization_started = time.perf_counter()
            if path_optimizer == 'vpype':
                vpype_plan, vpype_metadata = vpype_optimizer.optimize_with_vpype(executable_plan)
                optimizer_available = bool(vpype_metadata.get('available'))
                optimizer_warnings.extend(str(item) for item in vpype_metadata.get('warnings') or ())
                if vpype_plan is not None:
                    executable_plan = vpype_plan
                    optimization_stats = dict(vpype_metadata)
                    optimizer_used = 'vpype'
                else:
                    internal_result = optimize_canonical_plan(
                        executable_plan,
                        policy=_preview_optimization_policy(source_type),
                    )
                    executable_plan = internal_result.plan
                    optimization_stats = {
                        'fallback_from': 'vpype',
                        'vpype': dict(vpype_metadata),
                        'internal': internal_result.stats.to_dict(),
                    }
                    optimizer_used = 'internal'
            else:
                optimization_result = optimize_canonical_plan(
                    executable_plan,
                    policy=_preview_optimization_policy(source_type),
                )
                executable_plan = optimization_result.plan
                optimization_stats = optimization_result.stats.to_dict()
                optimizer_used = 'internal'
            optimization_ms = max(0.0, (time.perf_counter() - optimization_started) * 1000.0)

        sampled_paths = _validated_runtime_sampled_paths(
            executable_plan,
            writable_bounds=writable_bounds,
            shared_config=shared,
            sampling_policy=runtime_sampling_policy,
        )
        primitive_descriptor = canonical_plan_to_primitive_path_plan(executable_plan)
        primitive_plan = _primitive_path_plan_message_from_descriptor(primitive_descriptor)
        primitive_hash = _stable_hash(primitive_descriptor)
        execution_payload = _sampled_paths_stable_payload(sampled_paths)
        execution_hash = _stable_hash(execution_payload)
        cpp_available = getattr(_canonical_adapters, '_geometry_cpp', None) is not None
        draw_path_count = sum(1 for sampled in sampled_paths if sampled.draw)
        travel_path_count = sum(1 for sampled in sampled_paths if not sampled.draw)
        draw_sample_count = sum(len(sampled.points) for sampled in sampled_paths if sampled.draw)
        travel_sample_count = sum(len(sampled.points) for sampled in sampled_paths if not sampled.draw)
        canonical_geometry = _canonical_geometry_metrics(executable_plan)
        executable_geometry = _executable_geometry_metrics(sampled_paths)
        source_metadata = dict(getattr(canonical_plan, 'metadata', {}) or {})
        color_lineart_metrics = dict(source_metadata.get('color_lineart') or {})
        travel_before_m = None
        path_count_before = None
        if pre_optimization_sampled_paths is not None:
            travel_before_m = _sampled_paths_length(pre_optimization_sampled_paths, draw=False)
            path_count_before = sum(1 for sampled in pre_optimization_sampled_paths if sampled.draw)
        travel_after_m = _sampled_paths_length(sampled_paths, draw=False)
        path_count_after = int(draw_path_count)
        optimizer_metrics = {
            'name': optimizer_used,
            'requested': path_optimizer,
            'used': optimizer_used,
            'available': bool(optimizer_available),
            'warnings': tuple(optimizer_warnings),
            'travel_before_m': travel_before_m,
            'travel_after_m': travel_after_m,
            'path_count_before': path_count_before,
            'path_count_after': path_count_after,
        }
        return {
            'executable_canonical_plan': executable_plan,
            'executable_canonical_hash': canonical_plan_hash(executable_plan),
            'primitive_descriptor': primitive_descriptor,
            'primitive_plan': primitive_plan,
            'primitive_hash': primitive_hash,
            'execution_preview_svg': _execution_preview_svg_from_sampled_paths(
                sampled_paths,
                board_width_m=float(shared.board.width),
                board_height_m=float(shared.board.height),
            ),
            'execution_hash': execution_hash,
            'settings_hash': settings_hash(effective_settings_payload),
            'metrics': {
                'execution_preview_source': 'cpp_geometry_binding' if cpp_available else 'python_runtime_sampling',
                'cpp_exact_preview': bool(cpp_available),
                **tiny_detail_metrics,
                'optimized': bool(optimize_stroke_order),
                'optimization': optimization_stats,
                'optimization_ms': float(optimization_ms),
                'optimizer': optimizer_metrics,
                'canonical_command_count': int(len(canonical_plan.commands)),
                'executable_canonical_command_count': int(len(executable_plan.commands)),
                'canonical_geometry': canonical_geometry,
                'executable_geometry': executable_geometry,
                'primitive_count': int(len(primitive_descriptor.get('primitives') or ())),
                'draw_path_count': int(draw_path_count),
                'travel_path_count': int(travel_path_count),
                'draw_sample_count': int(draw_sample_count),
                'travel_sample_count': int(travel_sample_count),
                'draw_length_m': _sampled_paths_length(sampled_paths, draw=True),
                'travel_length_m': _sampled_paths_length(sampled_paths, draw=False),
                'bounds': _sampled_paths_bounds(sampled_paths),
                'runtime_sampling_policy': _stable_payload(runtime_sampling_policy),
                'color_lineart': color_lineart_metrics,
            },
        }

    def _preview_cache_expired(entry: PreviewCacheEntry, *, now: float) -> bool:
        return preview_cache.is_expired(entry, now=now)

    def _cleanup_preview_cache(*, now: float | None = None) -> None:
        preview_cache.prune(now=now)

    def _store_preview(
        *,
        preview_id: str | None = None,
        source_type: str,
        canonical_plan: CanonicalPathPlan,
        preview_payload: dict[str, Any],
        commit_request: dict[str, Any] | None,
        input_type: str,
        pipeline_mode: str,
        source_hash: str | None = None,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        warnings: tuple[str, ...] = (),
        source_filename: str = '',
        drawing_plan: DrawingPathPlan | None = None,
        command_metadata: tuple[dict[str, Any] | None, ...] | None = None,
        optimizer_stats: dict[str, Any] | None = None,
        route_metadata: dict[str, Any] | None = None,
        curve_fit_payload: dict[str, Any] | None = None,
        optimize_stroke_order: bool = False,
        path_optimizer: str = 'internal',
        writable_bounds: dict[str, float] | None = None,
    ) -> PreviewCacheEntry:
        _cleanup_preview_cache()
        normalized_id = uuid.uuid4().hex if preview_id is None else _validate_preview_id(preview_id)
        canonical_hash = canonical_plan_hash(canonical_plan)
        normalized_path_optimizer = _normalize_path_optimizer(path_optimizer)
        geometry_settings = {
            'source_type': str(source_type),
            'input_type': str(input_type),
            'pipeline_mode': str(pipeline_mode),
            'settings': dict(settings or {}),
            'optimize_stroke_order': bool(optimize_stroke_order),
            'path_optimizer': normalized_path_optimizer,
        }
        executable_payload = _build_executable_preview_payload(
            canonical_plan,
            source_type=str(source_type),
            settings_payload=geometry_settings,
            writable_bounds=writable_bounds or _preview_writable_bounds_for_source(str(source_type)),
            optimize_stroke_order=bool(optimize_stroke_order),
            path_optimizer=normalized_path_optimizer,
            existing_optimizer_stats=optimizer_stats,
        )
        enriched_preview = dict(preview_payload)
        enriched_preview['canonical_hash'] = canonical_hash
        enriched_preview['executable_canonical_hash'] = executable_payload['executable_canonical_hash']
        enriched_preview['primitive_hash'] = executable_payload['primitive_hash']
        enriched_preview['execution_hash'] = executable_payload['execution_hash']
        enriched_preview['settings_hash'] = executable_payload['settings_hash']
        enriched_preview['preview_id'] = normalized_id
        enriched_commit_request = dict(commit_request or {})
        enriched_commit_request['preview_id'] = normalized_id
        entry = PreviewCacheEntry(
            preview_id=normalized_id,
            source_type=str(source_type),
            canonical_plan=canonical_plan,
            canonical_hash=canonical_hash,
            executable_canonical_plan=executable_payload['executable_canonical_plan'],
            executable_canonical_hash=executable_payload['executable_canonical_hash'],
            primitive_descriptor=executable_payload['primitive_descriptor'],
            primitive_plan=executable_payload['primitive_plan'],
            primitive_hash=executable_payload['primitive_hash'],
            execution_preview_svg=executable_payload['execution_preview_svg'],
            execution_hash=executable_payload['execution_hash'],
            settings_hash=executable_payload['settings_hash'],
            metrics=dict(executable_payload['metrics']),
            preview_payload=enriched_preview,
            commit_request=enriched_commit_request,
            created_at_unix=time.time(),
            input_type=str(input_type),
            pipeline_mode=str(pipeline_mode),
            source_hash=source_hash,
            settings=dict(settings or {}),
            metadata=dict(metadata or {}),
            warnings=tuple(str(item) for item in warnings),
            source_filename=str(source_filename or ''),
            drawing_plan=drawing_plan,
            command_metadata=command_metadata,
            optimizer_stats=dict(optimizer_stats or {}),
            route_metadata=dict(route_metadata or {}),
            curve_fit_payload=dict(curve_fit_payload or {}),
        )
        preview_cache.store(normalized_id, entry)
        return entry

    def _load_preview(preview_id: Any) -> PreviewCacheEntry:
        normalized_id = _validate_preview_id(preview_id)
        entry = preview_cache.load(normalized_id)
        if entry is None:
            if normalized_id in preview_cache.entries():
                raise HTTPException(status_code=410, detail='preview_id has expired')
            raise HTTPException(status_code=404, detail='preview_id is unknown')
        return entry

    def _preview_contract_payload(entry: PreviewCacheEntry) -> dict[str, Any]:
        expires_at_unix = float(entry.created_at_unix) + float(_PREVIEW_CACHE_TTL_SECONDS)
        return {
            'preview_id': entry.preview_id,
            'canonical_hash': entry.canonical_hash,
            'executable_canonical_hash': entry.executable_canonical_hash,
            'primitive_hash': entry.primitive_hash,
            'execution_hash': entry.execution_hash,
            'settings_hash': entry.settings_hash,
            'execution_preview_svg': entry.execution_preview_svg,
            'metrics': dict(entry.metrics),
            'source_type': entry.source_type,
            'input_type': entry.input_type,
            'pipeline_mode': entry.pipeline_mode,
            'source_hash': entry.source_hash,
            'created_at_unix': float(entry.created_at_unix),
            'expires_at_unix': expires_at_unix,
            'ttl_seconds': max(0.0, expires_at_unix - time.time()),
        }

    def _attach_preview_contract(payload: dict[str, Any], entry: PreviewCacheEntry) -> dict[str, Any]:
        enriched = dict(payload)
        existing_metrics = dict(enriched.get('metrics') or {})
        contract = _preview_contract_payload(entry)
        contract_metrics = dict(contract.get('metrics') or {})
        existing_metrics.update(contract_metrics)
        contract['metrics'] = existing_metrics
        enriched.update(contract)
        enriched['preview'] = dict(enriched.get('preview') or {})
        enriched['preview']['canonical_hash'] = entry.canonical_hash
        enriched['preview']['executable_canonical_hash'] = entry.executable_canonical_hash
        enriched['preview']['primitive_hash'] = entry.primitive_hash
        enriched['preview']['execution_hash'] = entry.execution_hash
        enriched['preview']['settings_hash'] = entry.settings_hash
        enriched['preview']['preview_id'] = entry.preview_id
        enriched['preview']['execution_preview_svg'] = entry.execution_preview_svg
        enriched['commit_request'] = dict(enriched.get('commit_request') or entry.commit_request)
        enriched['commit_request']['preview_id'] = entry.preview_id
        enriched['draw_request'] = dict(enriched.get('draw_request') or enriched['commit_request'])
        enriched['draw_request']['preview_id'] = entry.preview_id
        return enriched

    def _sketch_preview_expired(entry: SketchPreviewCacheEntry, *, now: float) -> bool:
        return sketch_preview_cache.is_expired(entry, now=now)

    def _cleanup_sketch_preview_cache(*, now: float | None = None) -> None:
        sketch_preview_cache.prune(now=now)

    def _store_sketch_preview(
        *,
        preview_id: str | None = None,
        drawing_plan: DrawingPathPlan,
        canonical_plan: CanonicalPathPlan,
        preview_geometry_mode: str,
        metadata: dict[str, Any],
        warnings: tuple[str, ...],
        source_filename: str,
        parameters: dict[str, Any],
    ) -> str:
        preview_id = uuid.uuid4().hex if preview_id is None else str(preview_id)
        created_at = time.time()
        sketch_canonical_hash = canonical_plan_hash(canonical_plan)
        entry = SketchPreviewCacheEntry(
            preview_id=preview_id,
            drawing_plan=drawing_plan,
            canonical_plan=canonical_plan,
            canonical_hash=sketch_canonical_hash,
            preview_geometry_mode=preview_geometry_mode,
            metadata=dict(metadata),
            warnings=tuple(str(item) for item in warnings),
            created_at_unix=created_at,
            source_filename=str(source_filename or ''),
            parameters=dict(parameters),
            stroke_count=len(drawing_plan.strokes),
            point_count=sum(len(stroke.points) for stroke in drawing_plan.strokes),
            canonical_command_count=len(canonical_plan.commands),
        )
        sketch_preview_cache.store(preview_id, entry)
        return preview_id

    def _load_sketch_preview(preview_id: Any) -> SketchPreviewCacheEntry:
        if not isinstance(preview_id, str) or not preview_id.strip():
            raise HTTPException(status_code=422, detail='preview_id is required')
        normalized_id = preview_id.strip()
        entry = sketch_preview_cache.load(normalized_id)
        if entry is None:
            if normalized_id in sketch_preview_cache.entries():
                raise HTTPException(
                    status_code=410, detail='sketch preview_id has expired',
                )
            raise HTTPException(
                status_code=404, detail='sketch preview_id is unknown',
            )
        return entry

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

    @app.get('/api/debug/last-plan')
    async def last_plan_debug() -> JSONResponse:
        payload = runtime.last_plan_debug_snapshot()
        return JSONResponse(payload or {'available': False})

    @app.get('/api/debug/last-execution')
    async def last_execution_debug() -> JSONResponse:
        payload = runtime.last_execution_debug_snapshot() or {'available': False}
        payload = dict(payload)
        payload['executor'] = runtime.node.executor_diagnostics_snapshot()
        return JSONResponse(payload)

    @app.get('/api/debug/last-curve-fit')
    async def last_curve_fit_debug() -> JSONResponse:
        payload = runtime.last_curve_fit_debug_snapshot()
        return JSONResponse(payload or {'available': False})

    def _run_raster_vectorization_engine(engine_name: str, content: bytes) -> VectorizationEngineResult:
        if engine_name == 'autotrace_centerline':
            return vectorize_autotrace_centerline(content)
        if engine_name == 'potrace_bw':
            return vectorize_potrace_bw(content)
        if engine_name == 'vtracer_svg':
            return vectorize_vtracer_svg(content)
        return VectorizationEngineResult(
            engine_name=engine_name,
            available=engine_name == 'internal_centerline',
            warnings=(f'{engine_name} is handled by the internal preview path.' if engine_name == 'internal_centerline' else f'Unknown vectorization engine {engine_name}.',),
        )

    async def preview_sketch_centerline(
        file: UploadFile = File(...),
        margin_m: Optional[float] = Form(None),
        max_image_dim: Optional[int] = Form(None),
        min_component_area_px: Optional[int] = Form(None),
        min_stroke_length_px: Optional[float] = Form(None),
        simplify_epsilon_px: Optional[float] = Form(None),
        line_sensitivity: Optional[float] = Form(None),
        sketch_extraction_method: Optional[str] = Form(None),
        skeleton_prune_px: Optional[float] = Form(None),
        vectorization_engine: Optional[str] = Form(None),
        merge_gap_px: Optional[float] = Form(None),
        merge_max_angle_deg: Optional[float] = Form(None),
        optimization_preset: Optional[str] = Form(None),
        preview_geometry_mode: Optional[str] = Form(None),
        curve_tolerance_px: Optional[float] = Form(None),
        curve_tolerance_m: Optional[float] = Form(None),
        scale_percent: Optional[float] = Form(None),
        center_x_m: Optional[float] = Form(None),
        center_y_m: Optional[float] = Form(None),
        fit_to_safe_area: Optional[bool] = Form(None),
        optimize_stroke_order: Optional[bool] = Form(None),
        path_optimizer: Optional[str] = Form(None),
        preserve_tiny_details: Optional[bool] = Form(None),
        minimum_drawable_feature_m: Optional[float] = Form(None),
        tiny_detail_candidate_max_feature_m: Optional[float] = Form(None),
        tiny_detail_expand_mode: Optional[str] = Form(None),
        tiny_detail_max_expansions: Optional[int] = Form(None),
        requested_input_type: Optional[str] = None,
        color_lineart_method: Optional[str] = None,
        color_to_sketch_method: Optional[str] = None,
    ) -> JSONResponse:
        try:
            content = await file.read(_MAX_UPLOAD_BYTES + 1)
            _validate_sketch_upload(file, content)
            normalized_requested_input_type = str(requested_input_type or 'auto').strip().lower()
            if normalized_requested_input_type not in {'auto', 'sketch_image', 'colored_image', 'image'}:
                raise HTTPException(
                    status_code=422,
                    detail='input_type must be one of: auto, sketch_image, colored_image, image',
                )
            sketch_fit_to_safe_area = _coerce_bool(
                True if fit_to_safe_area is None else fit_to_safe_area,
                field_name='fit_to_safe_area',
                default=True,
            )
            sketch_safe_bounds = _carriage_safe_writable_bounds_for_sketch()
            sketch_board_bounds = _board_bounds_for_sketch()
            sketch_fit_bounds = sketch_safe_bounds if sketch_fit_to_safe_area else sketch_board_bounds
            sketch_margin_m = _coerce_float(
                0.05 if margin_m is None else margin_m,
                field_name='margin_m',
                minimum=0.0,
                maximum=min(float(sketch_fit_bounds['width']), float(sketch_fit_bounds['height'])) * 0.45,
            )
            sketch_max_image_dim = _coerce_int(
                1000 if max_image_dim is None else max_image_dim,
                field_name='max_image_dim',
                minimum=500,
                maximum=1600,
            )
            sketch_min_component_area_px = _coerce_int(
                2 if min_component_area_px is None else min_component_area_px,
                field_name='min_component_area_px',
                minimum=1,
                maximum=100000,
            )
            sketch_min_stroke_length_px = _coerce_float(
                1.0 if min_stroke_length_px is None else min_stroke_length_px,
                field_name='min_stroke_length_px',
                minimum=0.0,
                maximum=100000.0,
            )
            sketch_simplify_epsilon_px = _coerce_float(
                0.25 if simplify_epsilon_px is None else simplify_epsilon_px,
                field_name='simplify_epsilon_px',
                minimum=0.0,
                maximum=10000.0,
            )
            sketch_line_sensitivity = _coerce_float(
                0.35 if line_sensitivity is None else line_sensitivity,
                field_name='line_sensitivity',
                minimum=0.0,
                maximum=0.95,
            )
            sketch_skeleton_prune_px = _coerce_float(
                4.0 if skeleton_prune_px is None else skeleton_prune_px,
                field_name='skeleton_prune_px',
                minimum=0.0,
                maximum=100.0,
            )
            sketch_extraction = str(sketch_extraction_method or 'adaptive').strip().lower()
            if sketch_extraction not in {'hysteresis_ink', 'otsu', 'adaptive'}:
                raise HTTPException(
                    status_code=422,
                    detail='sketch_extraction_method must be one of: hysteresis_ink, otsu, adaptive',
                )
            sketch_vectorization_engine = str(vectorization_engine or 'internal_centerline').strip().lower()
            if sketch_vectorization_engine not in {
                'internal_centerline',
                'autotrace_centerline',
                'potrace_bw',
                'vtracer_svg',
                'direct_svg',
            }:
                raise HTTPException(
                    status_code=422,
                    detail='vectorization_engine must be one of: internal_centerline, autotrace_centerline, potrace_bw, vtracer_svg, direct_svg',
                )
            if sketch_vectorization_engine == 'direct_svg':
                raise HTTPException(
                    status_code=422,
                    detail='direct_svg is only valid for SVG input; raster uploads use internal_centerline or optional raster vectorizers',
                )
            sketch_color_method = str(color_lineart_method or color_to_sketch_method or 'auto_outline').strip().lower()
            if sketch_color_method in {'opencv_pencil', 'opencv_edge'}:
                sketch_color_method = 'opencv_edge_diagnostic'
            if sketch_color_method not in {'auto_outline', 'photo_diagram_edges', 'simple_cartoon', 'opencv_edge_diagnostic'}:
                raise HTTPException(
                    status_code=422,
                    detail='color_lineart_method must be one of: auto_outline, photo_diagram_edges, simple_cartoon, opencv_edge_diagnostic',
                )
            sketch_merge_gap_px = _coerce_float(
                0.0 if merge_gap_px is None else merge_gap_px,
                field_name='merge_gap_px',
                minimum=0.0,
                maximum=1000.0,
            )
            sketch_merge_max_angle_deg = _coerce_float(
                20.0 if merge_max_angle_deg is None else merge_max_angle_deg,
                field_name='merge_max_angle_deg',
                minimum=0.0,
                maximum=180.0,
            )
            sketch_optimization_preset = str(optimization_preset or 'detail').strip().lower()
            sketch_preview_geometry_mode = str(preview_geometry_mode or 'smooth_curves').strip().lower()
            if sketch_preview_geometry_mode not in {'smooth_curves', 'polyline'}:
                raise HTTPException(
                    status_code=422,
                    detail="preview_geometry_mode must be one of: smooth_curves, polyline",
                )
            sketch_optimize_stroke_order = _coerce_bool(
                False if optimize_stroke_order is None else optimize_stroke_order,
                field_name='optimize_stroke_order',
                default=False,
            )
            sketch_path_optimizer = _normalize_path_optimizer(path_optimizer)
            sketch_curve_tolerance_px = _coerce_float(
                1.0 if curve_tolerance_px is None else curve_tolerance_px,
                field_name='curve_tolerance_px',
                minimum=0.05,
                maximum=50.0,
            )
            sketch_curve_tolerance_m = (
                None if curve_tolerance_m is None
                else _coerce_float(
                    curve_tolerance_m,
                    field_name='curve_tolerance_m',
                    minimum=1.0e-6,
                    maximum=0.25,
                )
            )
            sketch_scale_percent = _coerce_float(
                100.0 if scale_percent is None else scale_percent,
                field_name='scale_percent',
                minimum=1.0,
                maximum=500.0,
            )
            sketch_center_x_m = (
                None if center_x_m is None
                else _coerce_float(
                    center_x_m,
                    field_name='center_x_m',
                    minimum=0.0,
                    maximum=float(shared.board.width),
                )
            )
            sketch_center_y_m = (
                None if center_y_m is None
                else _coerce_float(
                    center_y_m,
                    field_name='center_y_m',
                    minimum=0.0,
                    maximum=float(shared.board.height),
                )
            )
            sketch_parameters = {
                'margin_m': sketch_margin_m,
                'max_image_dim': sketch_max_image_dim,
                'min_component_area_px': sketch_min_component_area_px,
                'min_stroke_length_px': sketch_min_stroke_length_px,
                'simplify_epsilon_px': sketch_simplify_epsilon_px,
                'line_sensitivity': sketch_line_sensitivity,
                'skeleton_prune_px': sketch_skeleton_prune_px,
                'sketch_extraction_method': sketch_extraction,
                'vectorization_engine': sketch_vectorization_engine,
                'merge_gap_px': sketch_merge_gap_px,
                'merge_max_angle_deg': sketch_merge_max_angle_deg,
                'optimization_preset': sketch_optimization_preset,
                'preview_geometry_mode': sketch_preview_geometry_mode,
                'curve_tolerance_px': sketch_curve_tolerance_px,
                'curve_tolerance_m': sketch_curve_tolerance_m,
                'scale_percent': sketch_scale_percent,
                'center_x_m': sketch_center_x_m,
                'center_y_m': sketch_center_y_m,
                'fit_to_safe_area': sketch_fit_to_safe_area,
                'optimize_stroke_order': sketch_optimize_stroke_order,
                'path_optimizer': sketch_path_optimizer,
                'requested_input_type': normalized_requested_input_type,
                'color_lineart_method': sketch_color_method,
                'preserve_tiny_details': preserve_tiny_details,
                'minimum_drawable_feature_m': minimum_drawable_feature_m,
                'tiny_detail_candidate_max_feature_m': tiny_detail_candidate_max_feature_m,
                'tiny_detail_expand_mode': tiny_detail_expand_mode,
                'tiny_detail_max_expansions': tiny_detail_max_expansions,
            }
            try:
                detection = detect_raster_input_type(
                    content,
                    filename=str(file.filename or ''),
                    content_type=str(file.content_type or ''),
                    requested_input_type=normalized_requested_input_type,
                )
                effective_input_type = detection.input_type
                pipeline_mode = (
                    'local_outline_adaptive_centerline'
                    if effective_input_type == 'colored_image'
                    else 'sketch_centerline'
                )
                vectorization_content = content
                color_lineart_metadata: dict[str, Any] = {}
                color_lineart_preview: dict[str, Any] | None = None
                color_lineart_warnings: tuple[str, ...] = ()
                if effective_input_type == 'colored_image' and sketch_vectorization_engine == 'internal_centerline':
                    color_lineart_result = convert_color_image_to_lineart(
                        content,
                        method=sketch_color_method,
                        max_image_dim=sketch_max_image_dim,
                    )
                    vectorization_content = color_lineart_result.line_art_png
                    color_lineart_metadata = dict(color_lineart_result.metadata)
                    color_lineart_warnings = tuple(color_lineart_result.warnings)
                    encoded_preview = base64.b64encode(color_lineart_result.line_art_png).decode('ascii')
                    color_lineart_preview = {
                        'mime_type': 'image/png',
                        'data_url': f'data:image/png;base64,{encoded_preview}',
                        'quality': color_lineart_metadata.get('color_lineart_quality'),
                        'profile': color_lineart_metadata.get('color_lineart_profile'),
                        'foreground_ratio': color_lineart_metadata.get('foreground_ratio'),
                        'method': color_lineart_metadata.get('color_lineart_method'),
                        'metadata': color_lineart_metadata,
                        'warnings': list(color_lineart_warnings),
                    }
                sketch_parameters['input_type'] = effective_input_type
                sketch_parameters['pipeline_mode'] = pipeline_mode
                sketch_parameters['input_detection'] = detection.to_dict()
                if sketch_vectorization_engine != 'internal_centerline':
                    engine_result = _run_raster_vectorization_engine(sketch_vectorization_engine, content)
                    if not engine_result.available:
                        raise HTTPException(
                            status_code=422,
                            detail={
                                'engine_name': engine_result.engine_name,
                                'available': False,
                                'warnings': list(engine_result.warnings),
                            },
                        )
                    if engine_result.canonical_plan is None:
                        raise HTTPException(
                            status_code=422,
                            detail={
                                'engine_name': engine_result.engine_name,
                                'available': True,
                                'warnings': list(engine_result.warnings) or ['Vectorization engine did not return a CanonicalPathPlan.'],
                            },
                        )
                    raise HTTPException(
                        status_code=422,
                        detail={
                            'engine_name': engine_result.engine_name,
                            'available': True,
                            'warnings': list(engine_result.warnings) or [
                                'External raster vectorizer is available, but board placement is not active for this engine yet.'
                            ],
                        },
                    )
                preview_started = time.perf_counter()
                plan = vectorize_sketch_image_to_plan(
                    vectorization_content,
                    board_width_m=float(shared.board.width),
                    board_height_m=float(shared.board.height),
                    margin_m=sketch_margin_m,
                    max_image_dim=sketch_max_image_dim,
                    min_component_area_px=sketch_min_component_area_px,
                    min_stroke_length_px=sketch_min_stroke_length_px,
                    simplify_epsilon_px=sketch_simplify_epsilon_px,
                    line_sensitivity=sketch_line_sensitivity,
                    sketch_extraction_method=sketch_extraction,
                    skeleton_prune_px=sketch_skeleton_prune_px,
                    merge_gap_px=sketch_merge_gap_px,
                    merge_max_angle_deg=sketch_merge_max_angle_deg,
                    optimization_preset=sketch_optimization_preset,
                    scale_percent=sketch_scale_percent,
                    center_x_m=sketch_center_x_m,
                    center_y_m=sketch_center_y_m,
                    fit_bounds_m=sketch_fit_bounds,
                    validation_bounds_m=sketch_safe_bounds,
                )
                plan.metadata.update(
                    {
                        'requested_input_type': normalized_requested_input_type,
                        'detected_input_type': effective_input_type,
                        'input_detection_confidence': float(detection.confidence),
                        'input_detection_reason': str(detection.reason),
                        'input_detection': detection.to_dict(),
                        'pipeline_mode': pipeline_mode,
                    }
                )
                if color_lineart_metadata:
                    plan.metadata['color_lineart'] = color_lineart_metadata
                    plan.metadata.update(color_lineart_metadata)
                    if color_lineart_warnings:
                        existing_warnings = tuple(str(item) for item in plan.metadata.get('warnings') or ())
                        plan.metadata['warnings'] = existing_warnings + color_lineart_warnings
                curve_metadata: dict[str, Any] = {
                    'preview_geometry_mode': sketch_preview_geometry_mode,
                    'curve_tolerance_px': float(sketch_curve_tolerance_px),
                    'fit_to_safe_area': bool(sketch_fit_to_safe_area),
                    'safe_x_min': float(sketch_safe_bounds['x_min']),
                    'safe_x_max': float(sketch_safe_bounds['x_max']),
                    'safe_y_min': float(sketch_safe_bounds['y_min']),
                    'safe_y_max': float(sketch_safe_bounds['y_max']),
                    'safe_width': float(sketch_safe_bounds['width']),
                    'safe_height': float(sketch_safe_bounds['height']),
                    'safe_bounds_m': sketch_safe_bounds,
                    'fit_bounds_m': sketch_fit_bounds,
                    'validation_bounds_m': sketch_safe_bounds,
                }
                curve_fit_start = time.perf_counter()
                if sketch_preview_geometry_mode == 'smooth_curves':
                    scale_m_per_px = float(dict(plan.metadata).get('scale_m_per_px') or 0.0)
                    effective_curve_tolerance_m = (
                        float(sketch_curve_tolerance_m)
                        if sketch_curve_tolerance_m is not None
                        else max(1.0e-6, float(sketch_curve_tolerance_px) * scale_m_per_px)
                    )
                    smooth_result = drawing_path_plan_to_smooth_canonical(
                        plan,
                        curve_tolerance_m=effective_curve_tolerance_m,
                        max_curve_segment_points=32,
                        max_fit_time_ms=3000.0,
                    )
                    canonical_plan = smooth_result.plan
                    curve_metadata.update(dict(smooth_result.metadata))
                    curve_metadata['curve_tolerance_m'] = float(effective_curve_tolerance_m)
                else:
                    canonical_plan = drawing_path_plan_to_canonical(plan)
                    curve_metadata.update(
                        {
                            'curve_tolerance_m': None if sketch_curve_tolerance_m is None else float(sketch_curve_tolerance_m),
                            'line_primitive_count': sum(isinstance(command, LineSegment) for command in canonical_plan.commands),
                            'quadratic_primitive_count': 0,
                            'cubic_primitive_count': 0,
                            'curve_primitive_count': 0,
                        }
                    )
                curve_fit_time_ms = (time.perf_counter() - curve_fit_start) * 1000.0
                timing = dict(plan.metadata.get('timing') or {})
                timing['curve_fit_time_ms'] = float(curve_fit_time_ms)
                timing['preview_total_time_ms'] = (time.perf_counter() - preview_started) * 1000.0
                plan.metadata['timing'] = timing
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc))
            except ValueError as exc:
                raise _image_value_error_to_http(exc)

            preview_id = uuid.uuid4().hex
            response_payload = _sketch_preview_response(
                plan,
                preview_id=preview_id,
                canonical_plan=canonical_plan,
                board_width_m=float(shared.board.width),
                board_height_m=float(shared.board.height),
                preview_geometry_mode=sketch_preview_geometry_mode,
                use_smooth_svg=sketch_preview_geometry_mode == 'smooth_curves',
                curve_metadata=curve_metadata,
            )
            response_payload['detected_input_type'] = effective_input_type
            response_payload['input_detection'] = detection.to_dict()
            if color_lineart_preview:
                response_payload['converted_lineart_preview'] = color_lineart_preview
            generic_entry = _store_preview(
                preview_id=preview_id,
                source_type='sketch_centerline',
                canonical_plan=canonical_plan,
                preview_payload=dict(response_payload.get('preview') or {}),
                commit_request={'preview_id': preview_id},
                input_type=effective_input_type,
                pipeline_mode=pipeline_mode,
                source_hash=_content_hash(
                    {
                        'original': _content_hash(content),
                        'vectorized': _content_hash(vectorization_content),
                        'input_type': effective_input_type,
                    }
                ),
                settings=sketch_parameters,
                metadata=dict(response_payload.get('metadata') or {}),
                warnings=tuple(str(item) for item in response_payload.get('warnings') or ()),
                source_filename=str(file.filename or ''),
                drawing_plan=plan,
                route_metadata=dict(response_payload.get('metadata') or {}),
                optimize_stroke_order=sketch_optimize_stroke_order,
                path_optimizer=sketch_path_optimizer,
            )
            if color_lineart_metadata:
                generic_entry.metrics['color_lineart'] = dict(color_lineart_metadata)
            _store_sketch_preview(
                preview_id=preview_id,
                drawing_plan=plan,
                canonical_plan=canonical_plan,
                preview_geometry_mode=sketch_preview_geometry_mode,
                metadata=dict(response_payload.get('metadata') or {}),
                warnings=tuple(str(item) for item in response_payload.get('warnings') or ()),
                source_filename=str(file.filename or ''),
                parameters=sketch_parameters,
            )
            return JSONResponse(_attach_preview_contract(response_payload, generic_entry))
        finally:
            await file.close()

    async def draw_sketch_centerline(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='sketch draw request',
            max_bytes=4096,
        )
        _reject_extra_fields(raw, {'preview_id', 'optimize_stroke_order'}, 'sketch draw request')
        entry = _load_sketch_preview(raw.get('preview_id'))
        optimize_stroke_order = _coerce_bool(
            raw.get('optimize_stroke_order'),
            field_name='optimize_stroke_order',
            default=True,
        )
        publish_plan = entry.canonical_plan
        optimization_stats: dict[str, Any] = {}
        optimization_ms = 0.0
        if optimize_stroke_order:
            optimization_start = time.perf_counter()
            optimization_result = optimize_canonical_plan(
                entry.canonical_plan,
                policy=sketch_draw_optimization_policy,
            )
            optimization_ms = _elapsed_ms(optimization_start)
            publish_plan = optimization_result.plan
            optimization_stats = optimization_result.stats.to_dict()
        publish_canonical_hash = canonical_plan_hash(publish_plan)

        size_summary = _canonical_transport_size_summary(publish_plan)
        _enforce_sketch_draw_size_limits(size_summary)

        build_start = time.perf_counter()
        primitive_plan_msg = _build_execution_transport_message(
            publish_plan,
            writable_bounds=_carriage_safe_writable_bounds_for_sketch(),
            shared_config=shared,
            sampling_policy=runtime_sampling_policy,
        )
        transport_build_ms = _elapsed_ms(build_start)
        publish_start = time.perf_counter()
        transport = runtime.node.publish_execution_plan(
            primitive_plan_msg,
            allowed_modes=(MODE_DRAW,),
        )
        publish_ms = _elapsed_ms(publish_start)
        timings = {
            'optimization_ms': optimization_ms,
            'transport_build_ms': transport_build_ms,
            'publish_ms': publish_ms,
        }
        preview_payload = {
            'stroke_count': entry.stroke_count,
            'point_count': entry.point_count,
            'bounds': _drawing_plan_bounds(entry.drawing_plan),
            'diagnostics': canonical_plan_diagnostics(
                publish_plan,
                preview_sampling_policy=preview_sampling_policy,
                runtime_sampling_policy=runtime_sampling_policy,
            ),
        }
        _record_last_plan_debug(
            source_type='sketch_centerline',
            canonical_plan=publish_plan,
            preview_payload=preview_payload,
            timings=timings,
            optimizer_stats=optimization_stats,
            route_metadata={
                'preview_id': entry.preview_id,
                'canonical_hash': publish_canonical_hash,
                'cached_canonical_hash': entry.canonical_hash,
                'preview_geometry_mode': entry.preview_geometry_mode,
                'source_filename': entry.source_filename,
                'parameters': entry.parameters,
                'metadata': entry.metadata,
                'used_full_cached_plan': True,
                'optimized': bool(optimize_stroke_order),
            },
            transport=transport,
            committed=True,
        )
        _record_last_execution_debug(
            source_type='sketch_centerline',
            preview_payload=preview_payload,
            transport=transport,
            timings=timings,
        )
        return JSONResponse(
            {
                'ok': True,
                'published': True,
                'active_mode': MODE_DRAW,
                'source_type': 'sketch_centerline',
                'preview_id': entry.preview_id,
                'canonical_hash': publish_canonical_hash,
                'cached_canonical_hash': entry.canonical_hash,
                'preview_geometry_mode': entry.preview_geometry_mode,
                'stroke_count': entry.stroke_count,
                'point_count': entry.point_count,
                'canonical_command_count': len(publish_plan.commands),
                'cached_canonical_command_count': entry.canonical_command_count,
                'primitive_count': size_summary['primitive_count'],
                'primitive_descriptor_bytes': size_summary['primitive_descriptor_bytes'],
                'used_full_cached_plan': True,
                'optimized': bool(optimize_stroke_order),
                'optimization': optimization_stats,
                'transport': transport,
                'warnings': list(entry.warnings),
                'timings_ms': timings,
            }
        )

    def _cached_preview_allowed_modes(entry: PreviewCacheEntry) -> tuple[str, ...]:
        return _preview_allowed_modes(entry.source_type)

    def _cached_preview_writable_bounds(entry: PreviewCacheEntry) -> dict[str, float]:
        if entry.source_type == 'sketch_centerline':
            return _carriage_safe_writable_bounds_for_sketch()
        return runtime.node.carriage_safe_writable_bounds()

    def _draw_cached_preview_response(
        entry: PreviewCacheEntry,
    ) -> dict[str, Any]:
        if entry.source_type == 'sketch_centerline':
            size_summary = _canonical_transport_size_summary(entry.executable_canonical_plan)
            _enforce_sketch_draw_size_limits(size_summary)
        publish_start = time.perf_counter()
        transport = runtime.node.publish_execution_plan(
            entry.primitive_plan,
            allowed_modes=_cached_preview_allowed_modes(entry),
        )
        publish_ms = _elapsed_ms(publish_start)
        timings = {
            'optimization_ms': 0.0,
            'transport_build_ms': 0.0,
            'publish_ms': publish_ms,
        }
        preview_payload = dict(entry.preview_payload)
        preview_payload['preview_id'] = entry.preview_id
        preview_payload['canonical_hash'] = entry.canonical_hash
        preview_payload['executable_canonical_hash'] = entry.executable_canonical_hash
        preview_payload['primitive_hash'] = entry.primitive_hash
        preview_payload['execution_hash'] = entry.execution_hash
        preview_payload['settings_hash'] = entry.settings_hash
        preview_payload['execution_preview_svg'] = entry.execution_preview_svg
        preview_payload['diagnostics'] = canonical_plan_diagnostics(
            entry.executable_canonical_plan,
            preview_sampling_policy=preview_sampling_policy,
            runtime_sampling_policy=runtime_sampling_policy,
        )
        _record_last_plan_debug(
            source_type=entry.source_type,
            canonical_plan=entry.executable_canonical_plan,
            preview_payload=preview_payload,
            timings=timings,
            optimizer_stats=entry.metrics.get('optimization') or entry.optimizer_stats,
            route_metadata={
                **dict(entry.route_metadata or {}),
                'preview_id': entry.preview_id,
                'input_type': entry.input_type,
                'pipeline_mode': entry.pipeline_mode,
                'source_hash': entry.source_hash,
                'settings': entry.settings,
                'settings_hash': entry.settings_hash,
                'metadata': entry.metadata,
                'used_cached_executable_payload': True,
                'optimized': bool(entry.metrics.get('optimized')),
                'cached_canonical_hash': entry.canonical_hash,
                'published_canonical_hash': entry.executable_canonical_hash,
                'primitive_hash': entry.primitive_hash,
                'execution_hash': entry.execution_hash,
            },
            transport=transport,
            committed=True,
            command_metadata=entry.command_metadata,
        )
        _record_last_execution_debug(
            source_type=entry.source_type,
            preview_payload=preview_payload,
            transport=transport,
            timings=timings,
        )
        if entry.curve_fit_payload:
            _record_last_curve_fit_debug(entry.curve_fit_payload)
        elif entry.source_type == 'svg':
            _record_curve_fit_unavailable('svg')

        primitive_count = len(entry.primitive_descriptor.get('primitives') or ())
        primitive_descriptor_bytes = len(
            json.dumps(entry.primitive_descriptor, separators=(',', ':'), sort_keys=True).encode('utf-8')
        )
        return {
            'ok': True,
            'published': True,
            'active_mode': MODE_TEXT if entry.source_type == 'text' else MODE_DRAW,
            'source_type': entry.source_type,
            'input_type': entry.input_type,
            'pipeline_mode': entry.pipeline_mode,
            'preview_id': entry.preview_id,
            'canonical_hash': entry.canonical_hash,
            'cached_canonical_hash': entry.canonical_hash,
            'executable_canonical_hash': entry.executable_canonical_hash,
            'primitive_hash': entry.primitive_hash,
            'execution_hash': entry.execution_hash,
            'settings_hash': entry.settings_hash,
            'preview_draw_hash_match': entry.executable_canonical_hash == entry.executable_canonical_hash,
            'primitive_hash_match': True,
            'execution_hash_match': True,
            'used_cached_preview_plan': True,
            'used_cached_executable_payload': True,
            'optimized': bool(entry.metrics.get('optimized')),
            'optimization': dict(entry.metrics.get('optimization') or {}),
            'metrics': dict(entry.metrics),
            'canonical_command_count': len(entry.executable_canonical_plan.commands),
            'cached_canonical_command_count': len(entry.canonical_plan.commands),
            'primitive_count': int(primitive_count),
            'primitive_descriptor_bytes': int(primitive_descriptor_bytes),
            'transport': transport,
            'warnings': list(entry.warnings),
            'timings_ms': timings,
        }

    @app.post('/api/draw')
    @app.post('/api/preview/draw')
    async def draw_cached_preview(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='cached preview draw request',
            max_bytes=4096,
        )
        _reject_extra_fields(raw, {'preview_id'}, 'cached preview draw request')
        if raw.get('preview_id') is None:
            raise HTTPException(status_code=400, detail='preview_id is required')
        entry = _load_preview(raw.get('preview_id'))
        return JSONResponse(_draw_cached_preview_response(entry))

    @app.delete('/api/preview/{preview_id}')
    async def clear_cached_preview(preview_id: str) -> JSONResponse:
        normalized_id = _validate_preview_id(preview_id)
        removed = preview_cache.pop(normalized_id, None)
        sketch_preview_cache.pop(normalized_id, None)
        if removed is None:
            raise HTTPException(status_code=404, detail='preview_id is unknown')
        return JSONResponse({'ok': True, 'preview_id': normalized_id, 'cleared': True})

    def _preview_settings(raw_settings: Any, *, name: str) -> dict[str, Any]:
        if raw_settings is None:
            return {}
        if not isinstance(raw_settings, dict):
            raise HTTPException(status_code=422, detail=f'{name}.settings must be an object')
        return dict(raw_settings)

    def _preview_json_builder_raw(raw: dict[str, Any], *, required_key: str) -> dict[str, Any]:
        settings = _preview_settings(raw.get('settings'), name='preview request')
        payload = {
            str(key): value
            for key, value in settings.items()
            if str(key) not in {'path_optimizer', 'optimize_stroke_order'}
        }
        payload[required_key] = raw.get(required_key)
        if raw.get('placement') is not None and 'placement' not in payload:
            payload['placement'] = raw.get('placement')
        return payload

    def _svg_upload_requested(upload: Any, content: bytes, requested_input_type: str) -> bool:
        if requested_input_type == 'svg':
            return True
        if requested_input_type not in {'auto', ''}:
            return False
        suffix = Path(getattr(upload, 'filename', '') or '').suffix.lower()
        normalized_type = str(getattr(upload, 'content_type', '') or '').split(';', 1)[0].strip().lower()
        stripped = content.lstrip()[:256].lower()
        return (
            suffix == '.svg'
            or normalized_type in {'image/svg+xml', 'application/svg+xml', 'text/svg+xml'}
            or stripped.startswith(b'<svg')
            or b'<svg' in stripped
        )

    def _json_text_preview_response(raw: dict[str, Any]) -> JSONResponse:
        settings = _preview_settings(raw.get('settings'), name='preview request')
        path_optimizer = _normalize_path_optimizer(settings.get('path_optimizer'))
        optimize_stroke_order = _coerce_bool(
            settings.get('optimize_stroke_order'),
            field_name='preview request.settings.optimize_stroke_order',
            default=path_optimizer != 'none',
        )
        builder_raw = _preview_json_builder_raw(raw, required_key='text')
        _, placed_strokes, canonical_plan, placement_result, writable_bounds, commit_request, _, plan_preview, outside_safe_points, build_timings = _build_text_vector(
            builder_raw,
            request_name='preview text request',
        )
        preview_start = time.perf_counter()
        preview_payload = _preview_payload_from_strokes(
            placed_strokes,
            placement_result,
            outside_safe_points=outside_safe_points,
            normalized_plan=plan_preview,
            canonical_plan=canonical_plan,
            preview_sampling_policy=preview_sampling_policy,
            runtime_sampling_policy=runtime_sampling_policy,
        )
        build_timings['preview_sample_ms'] = _elapsed_ms(preview_start)
        build_timings['publish_ms'] = 0.0
        _record_last_plan_debug(
            source_type='text',
            canonical_plan=canonical_plan,
            preview_payload=preview_payload,
            timings=build_timings,
            committed=False,
        )
        preview_entry = _store_preview(
            source_type='text',
            canonical_plan=canonical_plan,
            preview_payload=preview_payload,
            commit_request=commit_request,
            input_type='text',
            pipeline_mode='text_vector',
            source_hash=_content_hash({'text': builder_raw.get('text'), 'settings': commit_request}),
            settings={**commit_request, 'path_optimizer': path_optimizer, 'optimize_stroke_order': optimize_stroke_order},
            optimize_stroke_order=optimize_stroke_order,
            path_optimizer=path_optimizer,
            writable_bounds=writable_bounds,
        )
        return JSONResponse(
            _attach_preview_contract(
                {
                    'ok': True,
                    'source_type': 'text',
                    'preview': preview_payload,
                    'preview_svg': preview_entry.execution_preview_svg,
                    'commit_request': commit_request,
                },
                preview_entry,
            )
        )

    def _json_svg_preview_response(raw: dict[str, Any], *, source_hash: str | None = None) -> JSONResponse:
        settings = _preview_settings(raw.get('settings'), name='preview request')
        path_optimizer = _normalize_path_optimizer(settings.get('path_optimizer'))
        optimize_stroke_order = _coerce_bool(
            settings.get('optimize_stroke_order'),
            field_name='preview request.settings.optimize_stroke_order',
            default=path_optimizer != 'none',
        )
        builder_raw = _preview_json_builder_raw(raw, required_key='svg')
        placed_strokes, canonical_plan, placement_result, writable_bounds, commit_request, _, plan_preview, outside_safe_points, build_timings = _build_svg_vector(
            builder_raw,
            request_name='preview svg request',
        )
        preview_start = time.perf_counter()
        preview_payload = _preview_payload_from_strokes(
            placed_strokes,
            placement_result,
            outside_safe_points=outside_safe_points,
            normalized_plan=plan_preview,
            canonical_plan=canonical_plan,
            preview_sampling_policy=preview_sampling_policy,
            runtime_sampling_policy=runtime_sampling_policy,
        )
        build_timings['preview_sample_ms'] = _elapsed_ms(preview_start)
        build_timings['publish_ms'] = 0.0
        _record_last_plan_debug(
            source_type='svg',
            canonical_plan=canonical_plan,
            preview_payload=preview_payload,
            timings=build_timings,
            committed=False,
        )
        _record_curve_fit_unavailable('svg')
        preview_entry = _store_preview(
            source_type='svg',
            canonical_plan=canonical_plan,
            preview_payload=preview_payload,
            commit_request=commit_request,
            input_type='svg',
            pipeline_mode='svg_vector',
            source_hash=source_hash or _content_hash(builder_raw.get('svg')),
            settings={
                key: value
                for key, value in commit_request.items()
                if key != 'svg'
            } | {'path_optimizer': path_optimizer, 'optimize_stroke_order': optimize_stroke_order},
            optimize_stroke_order=optimize_stroke_order,
            path_optimizer=path_optimizer,
            writable_bounds=writable_bounds,
        )
        return JSONResponse(
            _attach_preview_contract(
                {
                    'ok': True,
                    'source_type': 'svg',
                    'preview': preview_payload,
                    'preview_svg': preview_entry.execution_preview_svg,
                    'commit_request': commit_request,
                },
                preview_entry,
            )
        )

    @app.post('/api/preview')
    async def generate_preview(request: Request) -> JSONResponse:
        content_type = str(request.headers.get('content-type') or '').lower()
        if 'multipart/form-data' in content_type:
            form = await request.form()
            upload = form.get('file')
            if upload is None or not hasattr(upload, 'read'):
                raise HTTPException(status_code=422, detail='preview file upload requires a file field')
            requested_input_type = str(form.get('input_type') or 'auto').strip().lower()
            if requested_input_type not in {'auto', 'sketch_image', 'colored_image', 'image', 'svg'}:
                raise HTTPException(status_code=422, detail='input_type must be one of: auto, sketch_image, colored_image, image, svg')
            settings_json = form.get('settings_json')
            if settings_json in (None, ''):
                settings = {}
            else:
                try:
                    settings = json.loads(str(settings_json))
                except json.JSONDecodeError as exc:
                    raise HTTPException(status_code=422, detail=f'settings_json is invalid JSON: {exc}')
                settings = _preview_settings(settings, name='preview file request')
            form_settings = {
                str(key): value
                for key, value in form.items()
                if str(key) not in {'file', 'input_type', 'settings_json'}
            }
            settings = {**form_settings, **settings}
            content = await upload.read(_MAX_UPLOAD_BYTES + 1)
            if _svg_upload_requested(upload, content, requested_input_type):
                try:
                    svg_text = content.decode('utf-8')
                except UnicodeDecodeError as exc:
                    raise HTTPException(status_code=422, detail=f'preview SVG upload is not UTF-8: {exc}')
                return _json_svg_preview_response(
                    {
                        'input_type': 'svg',
                        'svg': svg_text,
                        'settings': settings,
                    },
                    source_hash=_content_hash(content),
                )
            if requested_input_type == 'svg':
                raise HTTPException(status_code=422, detail='selected SVG input is not an SVG upload')
            try:
                await upload.seek(0)
            except AttributeError:
                upload.file.seek(0)
            return await preview_sketch_centerline(
                file=upload,
                margin_m=settings.get('margin_m'),
                max_image_dim=settings.get('max_image_dim'),
                min_component_area_px=settings.get('min_component_area_px'),
                min_stroke_length_px=settings.get('min_stroke_length_px'),
                simplify_epsilon_px=settings.get('simplify_epsilon_px'),
                line_sensitivity=settings.get('line_sensitivity'),
                sketch_extraction_method=settings.get('sketch_extraction_method'),
                skeleton_prune_px=settings.get('skeleton_prune_px'),
                vectorization_engine=settings.get('vectorization_engine'),
                merge_gap_px=settings.get('merge_gap_px'),
                merge_max_angle_deg=settings.get('merge_max_angle_deg'),
                optimization_preset=settings.get('optimization_preset'),
                preview_geometry_mode=settings.get('preview_geometry_mode'),
                curve_tolerance_px=settings.get('curve_tolerance_px'),
                curve_tolerance_m=settings.get('curve_tolerance_m'),
                scale_percent=settings.get('scale_percent'),
                center_x_m=settings.get('center_x_m'),
                center_y_m=settings.get('center_y_m'),
                fit_to_safe_area=settings.get('fit_to_safe_area'),
                optimize_stroke_order=settings.get('optimize_stroke_order'),
                path_optimizer=settings.get('path_optimizer'),
                preserve_tiny_details=settings.get('preserve_tiny_details'),
                minimum_drawable_feature_m=settings.get('minimum_drawable_feature_m'),
                tiny_detail_candidate_max_feature_m=settings.get('tiny_detail_candidate_max_feature_m'),
                tiny_detail_expand_mode=settings.get('tiny_detail_expand_mode'),
                tiny_detail_max_expansions=settings.get('tiny_detail_max_expansions'),
                requested_input_type=requested_input_type,
                color_lineart_method=settings.get('color_lineart_method'),
                color_to_sketch_method=settings.get('color_to_sketch_method'),
            )

        raw = await _load_json_request(
            request,
            name='preview request',
            max_bytes=_MAX_VECTOR_REQUEST_BYTES,
        )
        _reject_extra_fields(raw, {'input_type', 'text', 'svg', 'settings', 'placement'}, 'preview request')
        input_type = str(raw.get('input_type') or 'auto').strip().lower()
        if input_type == 'text':
            return _json_text_preview_response(raw)
        if input_type == 'svg':
            return _json_svg_preview_response(raw)
        raise HTTPException(status_code=422, detail='JSON preview input_type must be text or svg')

    @app.post('/api/preview/compare')
    async def compare_preview_methods(request: Request) -> JSONResponse:
        content_type = str(request.headers.get('content-type') or '').lower()
        if 'multipart/form-data' not in content_type:
            raise HTTPException(status_code=415, detail='compare methods requires multipart/form-data')
        form = await request.form()
        upload = form.get('file')
        if upload is None or not hasattr(upload, 'read'):
            raise HTTPException(status_code=422, detail='compare methods requires a file field')
        settings_json = form.get('settings_json')
        settings: dict[str, Any] = {}
        if settings_json not in (None, ''):
            try:
                settings = json.loads(str(settings_json))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f'settings_json is invalid JSON: {exc}')
            settings = _preview_settings(settings, name='compare methods request')
        engines_raw = form.get('engines_json') or form.get('engines')
        if engines_raw in (None, ''):
            engines = ['internal_centerline', 'autotrace_centerline', 'potrace_bw', 'vtracer_svg']
        else:
            try:
                parsed_engines = json.loads(str(engines_raw))
            except json.JSONDecodeError:
                parsed_engines = [item.strip() for item in str(engines_raw).split(',') if item.strip()]
            if not isinstance(parsed_engines, list):
                raise HTTPException(status_code=422, detail='engines must be a JSON list or comma-separated string')
            engines = [str(item).strip().lower() for item in parsed_engines if str(item).strip()]
        allowed_engines = {'internal_centerline', 'autotrace_centerline', 'potrace_bw', 'vtracer_svg'}
        unknown = [engine for engine in engines if engine not in allowed_engines]
        if unknown:
            raise HTTPException(status_code=422, detail=f'unknown vectorization engines: {", ".join(unknown)}')
        content = await upload.read(_MAX_UPLOAD_BYTES + 1)
        _validate_sketch_upload(upload, content)
        results: list[dict[str, Any]] = []
        for engine in engines:
            if engine == 'internal_centerline':
                upload_copy = UploadFile(
                    filename=str(getattr(upload, 'filename', '') or 'compare.png'),
                    file=io.BytesIO(content),
                    content_type=str(getattr(upload, 'content_type', '') or 'image/png'),
                )
                try:
                    response = await preview_sketch_centerline(
                        file=upload_copy,
                        margin_m=settings.get('margin_m'),
                        max_image_dim=settings.get('max_image_dim'),
                        min_component_area_px=settings.get('min_component_area_px'),
                        min_stroke_length_px=settings.get('min_stroke_length_px'),
                        simplify_epsilon_px=settings.get('simplify_epsilon_px'),
                        line_sensitivity=settings.get('line_sensitivity'),
                        sketch_extraction_method=settings.get('sketch_extraction_method'),
                        skeleton_prune_px=settings.get('skeleton_prune_px'),
                        vectorization_engine='internal_centerline',
                        merge_gap_px=settings.get('merge_gap_px'),
                        merge_max_angle_deg=settings.get('merge_max_angle_deg'),
                        optimization_preset=settings.get('optimization_preset'),
                        preview_geometry_mode=settings.get('preview_geometry_mode'),
                        curve_tolerance_px=settings.get('curve_tolerance_px'),
                        curve_tolerance_m=settings.get('curve_tolerance_m'),
                        scale_percent=settings.get('scale_percent'),
                        center_x_m=settings.get('center_x_m'),
                        center_y_m=settings.get('center_y_m'),
                        fit_to_safe_area=settings.get('fit_to_safe_area'),
                        optimize_stroke_order=settings.get('optimize_stroke_order'),
                        path_optimizer=settings.get('path_optimizer'),
                        preserve_tiny_details=settings.get('preserve_tiny_details'),
                        minimum_drawable_feature_m=settings.get('minimum_drawable_feature_m'),
                        tiny_detail_candidate_max_feature_m=settings.get('tiny_detail_candidate_max_feature_m'),
                        tiny_detail_expand_mode=settings.get('tiny_detail_expand_mode'),
                        tiny_detail_max_expansions=settings.get('tiny_detail_max_expansions'),
                        requested_input_type=settings.get('input_type', 'auto'),
                        color_lineart_method=settings.get('color_lineart_method'),
                        color_to_sketch_method=settings.get('color_to_sketch_method'),
                    )
                except HTTPException as exc:
                    results.append(
                        {
                            'engine_name': engine,
                            'available': True,
                            'warnings': [str(exc.detail)],
                            'error': exc.detail,
                        }
                    )
                    continue
                payload = json.loads(response.body.decode('utf-8'))
                metrics = dict(payload.get('metrics') or {})
                results.append(
                    {
                        'engine_name': engine,
                        'available': True,
                        'warnings': list(payload.get('warnings') or ()),
                        'preview_id': payload.get('preview_id'),
                        'execution_preview_svg': payload.get('execution_preview_svg'),
                        'primitive_hash': payload.get('primitive_hash'),
                        'execution_hash': payload.get('execution_hash'),
                        'canonical_geometry': metrics.get('canonical_geometry') or {},
                        'executable_geometry': metrics.get('executable_geometry') or {},
                        'draw_length_m': metrics.get('draw_length_m'),
                        'travel_length_m': metrics.get('travel_length_m'),
                        'estimated_draw_time_ms': metrics.get('estimated_draw_time_ms'),
                        'metrics': metrics,
                        'payload': payload,
                    }
                )
                continue
            engine_result = _run_raster_vectorization_engine(engine, content)
            results.append(
                {
                    'engine_name': engine_result.engine_name,
                    'available': bool(engine_result.available),
                    'warnings': list(engine_result.warnings),
                    'metrics': dict(engine_result.metrics),
                    'has_canonical_plan': engine_result.canonical_plan is not None,
                    'has_svg_output': bool(engine_result.svg_output),
                }
            )
        return JSONResponse({'ok': True, 'results': results, 'active_preview_mutated': False})

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

    @app.post('/api/face/text')
    async def set_face_text(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='face text request',
            max_bytes=512,
        )
        _reject_extra_fields(raw, {'text'}, 'face text request')
        value = runtime.node.publish_face_text(str(raw.get('text', '')))
        return JSONResponse({'ok': True, 'text': value, 'topic': _FACE_TEXT_TOPIC})

    @app.post('/api/face/expression')
    async def set_face_expression(request: Request) -> JSONResponse:
        raw = await _load_json_request(
            request,
            name='face expression request',
            max_bytes=512,
        )
        _reject_extra_fields(raw, {'expression'}, 'face expression request')
        value = runtime.node.publish_face_expression(str(raw.get('expression', '')))
        return JSONResponse({'ok': True, 'expression': value, 'topic': _FACE_EXPRESSION_TOPIC})

    def _elapsed_ms(start_time: float) -> float:
        return max(0.0, (time.perf_counter() - start_time) * 1000.0)

    def _record_last_plan_debug(
        *,
        source_type: str,
        canonical_plan: CanonicalPathPlan,
        preview_payload: dict[str, Any],
        timings: dict[str, float],
        optimizer_stats: dict[str, Any] | None = None,
        route_metadata: dict[str, Any] | None = None,
        transport: dict[str, Any] | None = None,
        committed: bool,
        command_metadata: tuple[dict[str, Any] | None, ...] | None = None,
    ) -> None:
        diagnostics = preview_payload.get('diagnostics') or {}
        runtime.record_last_plan_debug(
            {
                'available': True,
                'source_type': source_type,
                'committed': bool(committed),
                'transport': transport,
                'plan': canonical_plan_debug_payload(
                    canonical_plan,
                    sampling_policy=runtime_sampling_policy,
                    command_metadata=command_metadata,
                ),
                'optimizer_stats': optimizer_stats or {},
                'route_metadata': route_metadata or {},
                'preview_sampling': diagnostics.get('preview_sampling'),
                'runtime_sampling': diagnostics.get('runtime_sampling'),
                'parity': diagnostics.get('parity'),
                'point_budget': diagnostics.get('point_budget'),
                'timings_ms': {key: float(value) for key, value in timings.items()},
            }
        )

    def _record_last_curve_fit_debug(payload: dict[str, Any] | None) -> None:
        runtime.record_last_curve_fit_debug(payload or {'available': False})

    def _record_last_execution_debug(
        *,
        source_type: str,
        preview_payload: dict[str, Any],
        transport: dict[str, Any],
        timings: dict[str, float],
    ) -> None:
        diagnostics = preview_payload.get('diagnostics') or {}
        runtime.record_last_execution_debug(
            {
                'available': True,
                'source_type': source_type,
                'chosen_transport': transport.get('preferred_transport'),
                'published_transports': transport.get('published'),
                'transport_topics': transport.get('topics'),
                'preview_runtime_sampling': {
                    'preview': diagnostics.get('preview_sampling'),
                    'runtime': diagnostics.get('runtime_sampling'),
                    'parity': diagnostics.get('parity'),
                    'point_budget': diagnostics.get('point_budget'),
                },
                'timings_ms': {key: float(value) for key, value in timings.items()},
            }
        )

    def _image_processing_defaults() -> dict[str, Any]:
        return dict(_DEFAULT_IMAGE_PREP_OPTIONS)

    def _default_uploaded_file_commit_request(
        source_type: str,
        *,
        upload_id: str,
        placement: VectorPlacement | None,
    ) -> dict[str, Any] | None:
        if placement is None:
            return None
        payload: dict[str, Any] = {
            'upload_id': upload_id,
            'placement': {
                'x': float(placement.x),
                'y': float(placement.y),
                'scale': float(placement.scale),
            },
        }
        if source_type == 'image':
            payload.update(_image_processing_defaults())
        return payload

    def _shrink_bounds_local(bounds: dict[str, float], margin_m: float) -> dict[str, float]:
        shrink = max(0.0, float(margin_m))
        shrunk = {
            'x_min': float(bounds['x_min']) + shrink,
            'x_max': float(bounds['x_max']) - shrink,
            'y_min': float(bounds['y_min']) + shrink,
            'y_max': float(bounds['y_max']) - shrink,
        }
        if shrunk['x_max'] <= shrunk['x_min'] or shrunk['y_max'] <= shrunk['y_min']:
            raise ValueError('Writable bounds are too small after applying draw fit margin.')
        return shrunk

    def _raster_overlay_payload(
        *,
        image_size: dict[str, Any] | None,
        writable_bounds: dict[str, float] | None,
        placement: VectorPlacement | None,
    ) -> dict[str, Any] | None:
        if not isinstance(image_size, dict) or writable_bounds is None or placement is None:
            return None
        width_px = int(image_size.get('width_px') or 0)
        height_px = int(image_size.get('height_px') or 0)
        if width_px <= 0 or height_px <= 0:
            return None
        fit_bounds = _shrink_bounds_local(
            writable_bounds,
            draw_execution_defaults.draw_scale_fit_margin_m,
        )
        fit_width = float(fit_bounds['x_max']) - float(fit_bounds['x_min'])
        fit_height = float(fit_bounds['y_max']) - float(fit_bounds['y_min'])
        fit_scale = min(fit_width / float(width_px), fit_height / float(height_px))
        final_scale = fit_scale * float(placement.scale)
        half_width = 0.5 * float(width_px) * final_scale
        half_height = 0.5 * float(height_px) * final_scale
        return {
            'kind': 'raster',
            'image_size': {
                'width_px': width_px,
                'height_px': height_px,
            },
            'bounds': {
                'x_min': float(placement.x) - half_width,
                'x_max': float(placement.x) + half_width,
                'y_min': float(placement.y) - half_height,
                'y_max': float(placement.y) + half_height,
                'width': half_width * 2.0,
                'height': half_height * 2.0,
            },
        }

    def _upload_file_status_payload(
        metadata: dict[str, Any],
        *,
        payload: bytes | None = None,
    ) -> dict[str, Any]:
        upload_id = _validate_upload_id(metadata.get('upload_id'))
        source_type = _uploaded_source_type(metadata)
        processing = runtime.upload_processing_snapshot(
            upload_id,
            metadata=metadata,
            payload=payload,
        )
        try:
            writable_bounds = runtime.node.carriage_safe_writable_bounds()
            safe_bounds = runtime.node.carriage_safe_safe_bounds()
            placement = default_image_placement(
                writable_bounds,
                safe_bounds=safe_bounds,
            )
        except HTTPException:
            writable_bounds = None
            placement = None
        except ValueError:
            writable_bounds = None
            placement = None

        commit_request = _default_uploaded_file_commit_request(
            source_type,
            upload_id=upload_id,
            placement=placement,
        )
        response_payload = {
            'ok': True,
            'stored_only': True,
            'source_type': source_type,
            'upload': metadata,
            'status': processing,
            'commit_request': commit_request,
        }
        if source_type == 'image':
            image_size = processing.get('image_size') if isinstance(processing, dict) else None
            response_payload['image_info'] = {
                'width_px': int(image_size.get('width_px') or 0) if isinstance(image_size, dict) else 0,
                'height_px': int(image_size.get('height_px') or 0) if isinstance(image_size, dict) else 0,
                'pipeline': {
                    'route': processing.get('route', {}).get('route') if isinstance(processing.get('route'), dict) else None,
                    'curve_fit_summary': processing.get('curve_fit_summary') or {},
                },
            }
            response_payload['raster_overlay'] = _raster_overlay_payload(
                image_size=image_size if isinstance(image_size, dict) else None,
                writable_bounds=writable_bounds,
                placement=placement,
            )
        return response_payload

    def _build_text_vector(
        raw: dict[str, Any],
        *,
        request_name: str,
    ) -> tuple[
        tuple[TextGlyphOutline, ...],
        tuple[tuple[tuple[float, float], ...], ...],
        CanonicalPathPlan,
        Any,
        dict[str, float],
        dict[str, Any],
        PrimitivePathPlan,
        dict[str, Any],
        int,
        dict[str, float],
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
        build_timings: dict[str, float] = {}
        try:
            ingest_start = time.perf_counter()
            grouped_source = vectorize_text_grouped(
                normalized_text,
                font_source=font_source,
                line_height=line_height,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
                max_line_width_units=max_line_width_units,
            )
            build_timings['ingest_ms'] = _elapsed_ms(ingest_start)
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
            place_start = time.perf_counter()
            placed_groups, placement_result = place_grouped_text_on_board(
                grouped_source,
                writable_bounds=writable_bounds,
                placement=placement,
                fit_padding=fit_padding,
                text_upward_bias_em=0.0,
            )
            build_timings['place_ms'] = _elapsed_ms(place_start)
            placed_strokes = tuple(
                stroke for glyph in placed_groups for stroke in glyph.strokes
            )
            canonical_plan = text_glyph_outlines_to_canonical_plan(
                placed_groups,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
            )
            build_timings['optimize_ms'] = 0.0
            runtime_export_start = time.perf_counter()
            primitive_plan_msg = _build_execution_transport_message(
                canonical_plan,
                writable_bounds=writable_bounds,
                shared_config=shared,
                sampling_policy=runtime_sampling_policy,
            )
            build_timings['runtime_export_ms'] = _elapsed_ms(runtime_export_start)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: {exc}')
        placed_strokes = canonical_plan_to_draw_strokes(
            canonical_plan,
            sampling_policy=runtime_sampling_policy,
        )
        outside_safe_points = _interpolated_outside_safe_workspace_count(
            placed_strokes,
            shared,
            step_m=_sampling_validation_step_m(runtime_sampling_policy),
        )
        plan_preview = canonical_plan_to_legacy_strokes(
            canonical_plan,
            sampling_policy=preview_sampling_policy,
        )
        commit_request = {
            'text': normalized_text,
            'placement': {'x': text_start.x, 'y': text_start.y, 'scale': text_start.scale},
            'font_source': font_source,
            'glyph_height_m': glyph_scale_m,
            'line_height': line_height,
            'curve_tolerance': curve_tolerance,
            'simplify_epsilon': simplify_epsilon,
            'fit_padding': fit_padding,
        }
        return (
            placed_groups,
            placed_strokes,
            canonical_plan,
            placement_result,
            writable_bounds,
            commit_request,
            primitive_plan_msg,
            plan_preview,
            outside_safe_points,
            build_timings,
        )

    def _build_svg_vector(
        raw: dict[str, Any],
        *,
        request_name: str,
    ) -> tuple[
        tuple[tuple[tuple[float, float], ...], ...],
        CanonicalPathPlan,
        Any,
        dict[str, float],
        dict[str, Any],
        PrimitivePathPlan,
        dict[str, Any],
        int,
        dict[str, float],
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
        build_timings: dict[str, float] = {}
        try:
            placement = normalize_placement(raw.get('placement'), writable_bounds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name}.placement invalid: {exc}')
        try:
            ingest_start = time.perf_counter()
            source_strokes = vectorize_svg(
                svg_payload,
                curve_tolerance=curve_tolerance,
                simplify_epsilon=simplify_epsilon,
            )
            source_plan = draw_strokes_to_canonical_plan(
                source_strokes,
                theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
            )
            build_timings['ingest_ms'] = _elapsed_ms(ingest_start)
            place_start = time.perf_counter()
            placed_plan, placement_result = place_canonical_plan_on_board(
                source_plan,
                writable_bounds=writable_bounds,
                placement=placement,
                fit_margin_m=draw_execution_defaults.draw_scale_fit_margin_m,
            )
            build_timings['place_ms'] = _elapsed_ms(place_start)
            optimize_start = time.perf_counter()
            canonical_plan = cleanup_canonical_plan(
                placed_plan,
                simplify_tolerance_m=draw_execution_defaults.draw_path_simplify_tolerance_m,
            )
            canonical_plan = optimize_canonical_plan(
                canonical_plan,
                policy=svg_optimization_policy,
            ).plan
            build_timings['optimize_ms'] = _elapsed_ms(optimize_start)
            runtime_export_start = time.perf_counter()
            primitive_plan_msg = _build_execution_transport_message(
                canonical_plan,
                writable_bounds=writable_bounds,
                shared_config=shared,
                sampling_policy=runtime_sampling_policy,
            )
            build_timings['runtime_export_ms'] = _elapsed_ms(runtime_export_start)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: {exc}')
        cleaned_strokes = canonical_plan_to_draw_strokes(
            canonical_plan,
            sampling_policy=runtime_sampling_policy,
        )
        outside_safe_points = _interpolated_outside_safe_workspace_count(
            cleaned_strokes,
            shared,
            step_m=_sampling_validation_step_m(runtime_sampling_policy),
        )
        plan_preview = canonical_plan_to_legacy_strokes(
            canonical_plan,
            sampling_policy=preview_sampling_policy,
        )
        commit_request = {
            'svg': svg_payload,
            'placement': {'x': placement.x, 'y': placement.y, 'scale': placement.scale},
            'curve_tolerance': curve_tolerance,
            'simplify_epsilon': simplify_epsilon,
        }
        return (
            cleaned_strokes,
            canonical_plan,
            placement_result,
            writable_bounds,
            commit_request,
            primitive_plan_msg,
            plan_preview,
            outside_safe_points,
            build_timings,
        )

    def _build_image_vector(
        content: bytes,
        raw: dict[str, Any],
        *,
        request_name: str,
        prepared_artifact: PreparedImageArtifact | None = None,
    ) -> tuple[
        tuple[tuple[tuple[float, float], ...], ...],
        CanonicalPathPlan,
        Any,
        dict[str, float],
        dict[str, Any],
        dict[str, Any],
        tuple[dict[str, Any] | None, ...],
        dict[str, Any],
        PrimitivePathPlan,
        dict[str, Any],
        int,
        dict[str, float],
    ]:
        allowed = {
            'placement',
            'min_perimeter_px',
            'contour_simplify_ratio',
            'max_strokes',
        }
        _reject_extra_fields(raw, allowed, request_name)
        default_image_options = (
            dict(prepared_artifact.defaults)
            if isinstance(prepared_artifact, PreparedImageArtifact)
            else _image_processing_defaults()
        )
        min_perimeter_px = _coerce_float(
            raw.get('min_perimeter_px', default_image_options['min_perimeter_px']),
            field_name=f'{request_name}.min_perimeter_px',
            minimum=1.0,
            maximum=1000.0,
        )
        contour_simplify_ratio = _coerce_float(
            raw.get('contour_simplify_ratio', default_image_options['contour_simplify_ratio']),
            field_name=f'{request_name}.contour_simplify_ratio',
            minimum=0.0001,
            maximum=0.2,
        )
        max_strokes = _coerce_int(
            raw.get('max_strokes', default_image_options['max_strokes']),
            field_name=f'{request_name}.max_strokes',
            minimum=1,
            maximum=16384,
        )
        if isinstance(prepared_artifact, PreparedImageArtifact):
            expected = prepared_artifact.defaults
            if (
                abs(float(min_perimeter_px) - float(expected['min_perimeter_px'])) > 1.0e-9
                or abs(float(contour_simplify_ratio) - float(expected['contour_simplify_ratio'])) > 1.0e-12
                or int(max_strokes) != int(expected['max_strokes'])
            ):
                raise HTTPException(
                    status_code=409,
                    detail='cached image preprocessing options are fixed after upload; re-upload to change them',
                )
        writable_bounds = runtime.node.carriage_safe_writable_bounds()
        build_timings: dict[str, float] = (
            dict(prepared_artifact.timings_ms)
            if isinstance(prepared_artifact, PreparedImageArtifact)
            else {}
        )
        if raw.get('placement') is None:
            try:
                placement = default_image_placement(
                    writable_bounds,
                    safe_bounds=runtime.node.carriage_safe_safe_bounds(),
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f'{request_name}.placement invalid: {exc}')
        else:
            try:
                placement = normalize_placement(raw.get('placement'), writable_bounds)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f'{request_name}.placement invalid: {exc}')
        try:
            if isinstance(prepared_artifact, PreparedImageArtifact):
                image_result = prepared_artifact.image_result
            else:
                ingest_start = time.perf_counter()
                image_result = vectorize_image_to_canonical_plan(
                    content,
                    theta_ref=draw_execution_defaults.fixed_draw_theta_rad,
                    min_perimeter_px=min_perimeter_px,
                    contour_simplify_ratio=contour_simplify_ratio,
                    max_strokes=max_strokes,
                )
                build_timings['ingest_ms'] = _elapsed_ms(ingest_start)
            place_start = time.perf_counter()
            placed_plan, placement_result = place_canonical_plan_on_board(
                image_result.plan,
                writable_bounds=writable_bounds,
                placement=placement,
                fit_margin_m=draw_execution_defaults.draw_scale_fit_margin_m,
            )
            build_timings['place_ms'] = _elapsed_ms(place_start)
            placed_command_metadata = tuple(
                dict(item) if item is not None else None
                for item in image_result.command_metadata
            )
            curve_fit_payload = curve_fit_debug_to_board(
                source_plan=image_result.plan,
                placed_plan=placed_plan,
                command_metadata=placed_command_metadata,
                raw_contours=image_result.raw_contours,
                curve_fit_debug={
                    **dict(image_result.curve_fit_debug or {}),
                    'route': image_result.route_decision.to_dict(),
                    'image_size': {
                        'width_px': int(image_result.image_size[0]),
                        'height_px': int(image_result.image_size[1]),
                    },
                },
                placement=placement,
                final_scale=placement_result.final_scale,
                source_type='image',
            )
            curve_fit_payload['route'] = image_result.route_decision.to_dict()
            curve_fit_payload['image_size'] = {
                'width_px': int(image_result.image_size[0]),
                'height_px': int(image_result.image_size[1]),
            }
            optimize_start = time.perf_counter()
            canonical_plan = placed_plan
            optimization_result = optimize_canonical_plan(
                canonical_plan,
                policy=image_optimization_policy,
            )
            canonical_plan = optimization_result.plan
            command_metadata = map_curve_fit_command_metadata(
                placed_plan,
                placed_command_metadata,
                canonical_plan,
            )
            build_timings['optimize_ms'] = _elapsed_ms(optimize_start)
            runtime_export_start = time.perf_counter()
            primitive_plan_msg = _build_execution_transport_message(
                canonical_plan,
                writable_bounds=writable_bounds,
                shared_config=shared,
                sampling_policy=runtime_sampling_policy,
            )
            build_timings['runtime_export_ms'] = _elapsed_ms(runtime_export_start)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: {exc}')
        cleaned_strokes = canonical_plan_to_draw_strokes(
            canonical_plan,
            sampling_policy=runtime_sampling_policy,
        )
        outside_safe_points = _interpolated_outside_safe_workspace_count(
            cleaned_strokes,
            shared,
            step_m=_sampling_validation_step_m(runtime_sampling_policy),
        )
        plan_preview = canonical_plan_to_legacy_strokes(
            canonical_plan,
            sampling_policy=preview_sampling_policy,
        )
        commit_tail = {
            'placement': {'x': placement.x, 'y': placement.y, 'scale': placement.scale},
            'min_perimeter_px': min_perimeter_px,
            'contour_simplify_ratio': contour_simplify_ratio,
            'max_strokes': max_strokes,
        }
        image_info = image_result.to_metadata()
        image_info['optimization'] = optimization_result.stats.to_dict()
        return (
            cleaned_strokes,
            canonical_plan,
            placement_result,
            writable_bounds,
            commit_tail,
            image_info,
            command_metadata,
            curve_fit_payload,
            primitive_plan_msg,
            plan_preview,
            outside_safe_points,
            build_timings,
        )

    def _uploaded_source_type(metadata: dict[str, Any]) -> str:
        return infer_uploaded_source_type(
            stored_filename=metadata.get('stored_filename'),
            original_filename=metadata.get('original_filename'),
            content_type=metadata.get('normalized_content_type') or metadata.get('content_type'),
            source_type=metadata.get('source_type'),
        )

    def _uploaded_svg_text(
        metadata: dict[str, Any],
        payload: bytes,
        *,
        request_name: str,
    ) -> str:
        try:
            upload_details = classify_uploaded_vector_file(
                metadata.get('original_filename'),
                metadata.get('normalized_content_type') or metadata.get('content_type'),
                payload,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: {exc}')
        if upload_details.source_type != 'svg' or upload_details.svg_text is None:
            raise HTTPException(status_code=422, detail=f'{request_name} failed: stored upload is not svg')
        return upload_details.svg_text

    def _record_curve_fit_unavailable(source_type: str) -> None:
        _record_last_curve_fit_debug({
            'available': False,
            'source_type': source_type,
        })

    def _uploaded_commit_request(upload_id: str, commit_request: dict[str, Any]) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in commit_request.items()
            if key != 'svg'
        }
        payload['upload_id'] = upload_id
        return payload

    @app.post('/api/draw/plan')
    async def submit_draw_plan(request: Request) -> JSONResponse:
        raise HTTPException(
            status_code=409,
            detail='raw /api/draw/plan has been removed; use /api/preview then /api/draw with preview_id',
        )

    return app


def _select_free_port(preferred: int, *, attempts: int = 32) -> int:
    """Return ``preferred`` if free, otherwise the next available TCP port.

    Some hosts (e.g. VS Code dev containers) keep ``preferred`` permanently
    bound for port-forwarding; in that case binding the FastAPI server to it
    fails with ``Address already in use`` even though no project process is
    holding the port. We probe a small window to find a usable alternative
    instead of crashing the launch.
    """
    base = max(1024, int(preferred))
    for offset in range(attempts):
        candidate = base + offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', candidate))
            return candidate
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError(
        f'Unable to find a free TCP port starting at {base} '
        f'(checked {attempts} ports).',
    )


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError('ROS 2 Python dependencies are required to run web_server.') from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node = WebBackendNode()
    runtime = BackendRuntime(node)
    app = create_app(runtime)
    selected_port = _select_free_port(node.port)
    if selected_port != node.port:
        node.get_logger().warn(
            f'Requested port {node.port} is busy; using {selected_port} instead. '
            f'Open http://localhost:{selected_port} in the browser.',
        )
    config = uvicorn.Config(app, host='0.0.0.0', port=selected_port, log_level='info')
    server = uvicorn.Server(config)
    runtime.attach_server(server)

    if node.open_browser:
        threading.Timer(0.75, lambda: webbrowser.open(f'http://localhost:{selected_port}')).start()

    runtime.start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()


if __name__ == '__main__':
    main()
