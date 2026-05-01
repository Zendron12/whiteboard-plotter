from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
from typing import Any

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, String
from wall_climber.runtime_topics import (
    CABLE_EXECUTOR_STATUS_TOPIC,
    CABLE_SUPERVISOR_STATUS_TOPIC,
    MANUAL_PEN_MODE_TOPIC,
    PEN_MODE_AUTO,
    PEN_MODE_DOWN,
    PEN_MODE_UP,
    PRIMITIVE_PATH_PLAN_TOPIC,
)
from wall_climber.x_plotter.demo_paths import VALID_DEMO_PATHS, build_demo_path
from wall_climber.x_plotter.frame_config import BoardFrameConfig
from wall_climber.x_plotter.primitive_sampler import SamplingPolicy, sample_primitive_path_plan
from wall_climber_interfaces.msg import PrimitivePathPlan

try:
    from rclpy.executors import ShutdownException
except ImportError:  # pragma: no cover - older rclpy versions do not expose this symbol
    class ShutdownException(Exception):
        pass

try:
    from webots_ros2_msgs.srv import SpawnNodeFromString
except ImportError:  # pragma: no cover - unavailable in non-Webots unit contexts
    SpawnNodeFromString = None


_TRANSIENT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass(frozen=True)
class _ExecutionSample:
    point: tuple[float, float]
    pen_down: bool
    primitive_index: int


def _float_property(properties: dict[str, Any], name: str, default: float) -> float:
    return float(properties.get(name, str(default)))


def _bool_property(properties: dict[str, Any], name: str, default: bool) -> bool:
    value = str(properties.get(name, str(default))).strip().lower()
    return value in {'1', 'true', 'yes', 'on'}


def _string_property(properties: dict[str, Any], name: str, default: str) -> str:
    return str(properties.get(name, default)).strip()


class XPlotterRobotPlugin:
    _BOARD_FRAME_ID = 'board'

    def init(self, webots_node, properties):
        self._robot = webots_node.robot
        self._lock = threading.Lock()
        self._spin_running = False

        self._frame = BoardFrameConfig(
            width=_float_property(properties, 'board_width', 6.3),
            height=_float_property(properties, 'board_height', 3.0),
            center_x=_float_property(properties, 'board_center_x', 0.0),
            center_z=_float_property(properties, 'board_center_z', 1.8),
            surface_y=_float_property(properties, 'board_surface_y', 2.410),
            carriage_plane_y=_float_property(properties, 'carriage_plane_y', 2.335),
            margin_left=_float_property(properties, 'margin_left', 0.20),
            margin_right=_float_property(properties, 'margin_right', 0.20),
            margin_top=_float_property(properties, 'margin_top', 0.20),
            margin_bottom=_float_property(properties, 'margin_bottom', 0.20),
            origin=_string_property(properties, 'board_origin', 'top_left'),
            x_direction=_string_property(properties, 'board_x_direction', 'right'),
            y_direction=_string_property(properties, 'board_y_direction', 'down'),
        )

        self._x_joint_name = _string_property(properties, 'x_joint_name', 'x_carriage_joint')
        self._y_joint_name = _string_property(properties, 'y_joint_name', 'y_carriage_joint')
        self._pen_joint_name = _string_property(properties, 'pen_joint_name', 'pen_slide_joint')
        self._optional_belt_device_names = tuple(
            item.strip()
            for item in _string_property(properties, 'optional_belt_devices', '').split(',')
            if item.strip()
        )

        self._carriage_speed = max(0.01, _float_property(properties, 'carriage_speed', 0.45))
        self._pen_slide_velocity = max(0.01, _float_property(properties, 'pen_slide_velocity', 0.35))
        self._pen_up = _float_property(properties, 'pen_up_position', 0.0)
        self._pen_down = _float_property(properties, 'pen_down_position', 0.025)
        self._initial_x = _float_property(properties, 'initial_x', self._frame.drawable_x_min)
        self._initial_y = _float_property(properties, 'initial_y', self._frame.drawable_y_min)
        self._current_point = self._frame.clamp_point(self._initial_x, self._initial_y)
        self._target_pen_down = False
        self._manual_pen_mode = PEN_MODE_AUTO
        self._queue: list[_ExecutionSample] = []
        self._status = 'starting'
        self._last_target_log: tuple[float, float, bool] | None = None
        self._published_compat_notice = False

        self._trail_enabled = _bool_property(properties, 'enable_webots_trail', False)
        self._trail_half_width = max(0.001, _float_property(properties, 'trail_half_width', 0.008))
        self._trail_min_spacing = max(0.001, _float_property(properties, 'trail_min_spacing', 0.010))
        self._trail_max_segments = max(1, int(_float_property(properties, 'trail_max', 1200)))
        self._trail_count = 0
        self._trail_last_world: tuple[float, float, float] | None = None
        self._trail_warned = False
        self._trail_client = None

        self._x_motor = self._required_device(self._x_joint_name, 'required X carriage joint')
        self._y_motor = self._required_device(self._y_joint_name, 'required Y carriage joint')
        self._pen_motor = self._required_device(self._pen_joint_name, 'required pen slide joint')
        self._configure_motor(self._x_motor, self._carriage_speed)
        self._configure_motor(self._y_motor, self._carriage_speed)
        self._configure_motor(self._pen_motor, self._pen_slide_velocity)
        self._set_motors(self._current_point, False)
        self._resolve_optional_belt_devices()

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node('x_plotter_robot_plugin')
        self._log = self._node.get_logger()
        self._robot_pose_pub = self._node.create_publisher(Pose2D, '/wall_climber/robot_pose_board', 1)
        self._pen_pose_pub = self._node.create_publisher(PointStamped, '/wall_climber/pen_pose_board', 1)
        self._board_info_pub = self._node.create_publisher(String, '/wall_climber/board_info', _TRANSIENT_QOS)
        self._pen_contact_pub = self._node.create_publisher(Bool, '/wall_climber/pen_contact', 1)
        self._pen_gap_pub = self._node.create_publisher(Float64, '/wall_climber/pen_gap', 1)
        self._x_status_pub = self._node.create_publisher(String, '/wall_climber/x_plotter_status', _TRANSIENT_QOS)
        self._compat_executor_status_pub = self._node.create_publisher(
            String,
            CABLE_EXECUTOR_STATUS_TOPIC,
            _TRANSIENT_QOS,
        )
        self._compat_supervisor_status_pub = self._node.create_publisher(
            String,
            CABLE_SUPERVISOR_STATUS_TOPIC,
            _TRANSIENT_QOS,
        )
        self._node.create_subscription(
            PrimitivePathPlan,
            PRIMITIVE_PATH_PLAN_TOPIC,
            self._primitive_plan_cb,
            10,
        )
        self._node.create_subscription(
            String,
            MANUAL_PEN_MODE_TOPIC,
            self._manual_pen_mode_cb,
            _TRANSIENT_QOS,
        )
        if self._trail_enabled and SpawnNodeFromString is not None:
            self._trail_client = self._node.create_client(SpawnNodeFromString, '/Ros2Supervisor/spawn_node_from_string')

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name='x_plotter_robot_plugin_spin',
            daemon=True,
        )
        self._spin_thread.start()

        self._board_info_json = json.dumps(self._frame.board_info_payload(), separators=(',', ':'))
        self._publish_board_info()
        self._publish_status('idle')
        self._log.info(
            'X plotter board frame ready: '
            f'origin={self._frame.origin}, +X={self._frame.x_direction}, +Y={self._frame.y_direction}, '
            f'width={self._frame.width:.3f}, height={self._frame.height:.3f}, '
            f'world_top_left=({self._frame.board_left:.3f},{self._frame.carriage_plane_y:.3f},{self._frame.board_top_z:.3f})'
        )

        demo_path = _string_property(properties, 'demo_path', 'line_square_triangle').lower()
        if demo_path not in VALID_DEMO_PATHS:
            self._log.warn(f'Unsupported demo_path {demo_path!r}; using "off".')
            demo_path = 'off'
        self._log.info(f'Active X plotter demo path: {demo_path}')
        if demo_path != 'off':
            try:
                demo_plan = build_demo_path(demo_path, frame=self._frame)
                sampled = sample_primitive_path_plan(demo_plan)
                self._queue_sampled_paths(sampled, source=f'demo:{demo_path}')
            except Exception as exc:
                self._log.error(f'Failed to build X plotter demo path {demo_path!r}: {exc}')
                self._publish_status('error')

    def _required_device(self, name: str, role: str):
        try:
            device = self._robot.getDevice(name)
        except Exception as exc:
            raise RuntimeError(f'X plotter {role} "{name}" is missing.') from exc
        if device is None:
            raise RuntimeError(f'X plotter {role} "{name}" is missing.')
        return device

    def _optional_device(self, name: str):
        try:
            return self._robot.getDevice(name)
        except Exception:
            return None

    def _resolve_optional_belt_devices(self) -> None:
        missing: list[str] = []
        for name in self._optional_belt_device_names:
            if self._optional_device(name) is None:
                missing.append(name)
        if missing:
            print(
                '[XPlotterRobotPlugin] WARNING: optional belt visual devices missing; '
                f'continuing with static crossed-belt visuals: {", ".join(missing)}'
            )

    def _configure_motor(self, motor, velocity: float) -> None:
        try:
            motor.setVelocity(float(velocity))
        except Exception:
            pass

    def _set_motors(self, point: tuple[float, float], pen_down: bool) -> None:
        x, y = point
        self._x_motor.setPosition(float(x))
        self._y_motor.setPosition(float(y))
        self._pen_motor.setPosition(self._pen_down if pen_down else self._pen_up)

    def _manual_pen_mode_cb(self, msg: String) -> None:
        mode = str(msg.data).strip().lower()
        if mode not in (PEN_MODE_AUTO, PEN_MODE_UP, PEN_MODE_DOWN):
            return
        with self._lock:
            self._manual_pen_mode = mode
        self._log.info(f'X plotter manual pen mode: {mode}')

    def _primitive_plan_cb(self, msg: PrimitivePathPlan) -> None:
        try:
            sampled = sample_primitive_path_plan(msg)
            self._queue_sampled_paths(sampled, source='primitive_path_plan')
            self._log.info(
                f'Received PrimitivePathPlan for X plotter: primitives={len(msg.primitives)}, '
                f'sampled_paths={len(sampled)}'
            )
        except Exception as exc:
            self._log.error(f'X plotter rejected primitive plan: {exc}')
            self._publish_status('error')

    def _queue_sampled_paths(self, sampled_paths, *, source: str) -> None:
        queue: list[_ExecutionSample] = []
        for sampled_path in sampled_paths:
            for point in sampled_path.points:
                if not self._frame.in_drawable_bounds(point[0], point[1]):
                    raise ValueError(
                        f'{source} point ({point[0]:.4f}, {point[1]:.4f}) is outside drawable bounds '
                        f'x=[{self._frame.drawable_x_min:.3f},{self._frame.drawable_x_max:.3f}], '
                        f'y=[{self._frame.drawable_y_min:.3f},{self._frame.drawable_y_max:.3f}]'
                    )
                queue.append(
                    _ExecutionSample(
                        point=(float(point[0]), float(point[1])),
                        pen_down=bool(sampled_path.draw),
                        primitive_index=int(sampled_path.primitive_index),
                    )
                )
        if not queue:
            raise ValueError(f'{source} produced no executable X plotter samples')
        final_point = queue[-1].point
        queue.append(_ExecutionSample(point=final_point, pen_down=False, primitive_index=queue[-1].primitive_index))
        with self._lock:
            self._queue = queue
            self._last_target_log = None
            self._trail_last_world = None
        self._publish_status('running')
        self._log.info(f'Queued X plotter execution from {source}: samples={len(queue)}')

    def _spin_loop(self):
        while self._spin_running:
            try:
                if not rclpy.ok():
                    break
                self._executor.spin_once(timeout_sec=0.01)
            except (ExternalShutdownException, ShutdownException):
                break
            except Exception:
                if not self._spin_running or not rclpy.ok():
                    break
                try:
                    self._node.get_logger().error('X plotter plugin spin loop stopped unexpectedly.')
                except Exception:
                    pass
                break

    def _publish_board_info(self) -> None:
        msg = String()
        msg.data = self._board_info_json
        self._board_info_pub.publish(msg)

    def _publish_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        msg = String()
        msg.data = status
        self._x_status_pub.publish(msg)
        # Compatibility only: the current web backend waits for these legacy cable status topics.
        self._compat_executor_status_pub.publish(msg)
        self._compat_supervisor_status_pub.publish(msg)
        if not self._published_compat_notice:
            self._published_compat_notice = True
            try:
                self._log.info(
                    'Publishing X-plotter compatibility status on legacy cable status topics; '
                    'no cable diagnostics are produced by the X plotter.'
                )
            except Exception:
                pass

    def _publish_pose_topics(self, point: tuple[float, float], pen_down: bool) -> None:
        pose = Pose2D()
        pose.x = float(point[0])
        pose.y = float(point[1])
        pose.theta = 0.0
        self._robot_pose_pub.publish(pose)

        pen = PointStamped()
        pen.header.stamp = self._node.get_clock().now().to_msg()
        pen.header.frame_id = self._BOARD_FRAME_ID
        pen.point.x = float(point[0])
        pen.point.y = float(point[1])
        pen.point.z = 0.0
        self._pen_pose_pub.publish(pen)

        contact = Bool()
        contact.data = bool(pen_down)
        self._pen_contact_pub.publish(contact)

        gap = Float64()
        gap.data = 0.0 if pen_down else abs(self._pen_down - self._pen_up)
        self._pen_gap_pub.publish(gap)

    def _effective_pen_down(self, requested: bool) -> bool:
        if self._manual_pen_mode == PEN_MODE_DOWN:
            return True
        if self._manual_pen_mode == PEN_MODE_UP:
            return False
        return bool(requested)

    def _step_toward(
        self,
        current: tuple[float, float],
        target: tuple[float, float],
        max_step: float,
    ) -> tuple[tuple[float, float], bool]:
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance = math.hypot(dx, dy)
        if distance <= max(1.0e-5, max_step):
            return target, True
        ratio = max_step / distance
        return (current[0] + dx * ratio, current[1] + dy * ratio), False

    def _spawn_trail_segment(self, start_world: tuple[float, float, float], end_world: tuple[float, float, float]) -> None:
        if not self._trail_enabled or self._trail_count >= self._trail_max_segments:
            return
        if self._trail_client is None:
            if not self._trail_warned:
                self._trail_warned = True
                self._log.warn('X plotter trail requested, but SpawnNodeFromString is unavailable.')
            return
        if not self._trail_client.service_is_ready():
            if not self._trail_warned:
                self._trail_warned = True
                self._log.warn('X plotter trail service is not ready; continuing without trail segments for now.')
            return

        dx = end_world[0] - start_world[0]
        dz = end_world[2] - start_world[2]
        length = math.hypot(dx, dz)
        if length < self._trail_min_spacing:
            return
        cx = (start_world[0] + end_world[0]) * 0.5
        cz = (start_world[2] + end_world[2]) * 0.5
        angle = -math.atan2(dz, dx)
        line_y = self._frame.surface_y - 0.006
        node = (
            f'DEF X_PLOTTER_TRAIL_{self._trail_count} Solid {{ '
            f'translation {cx:.5f} {line_y:.5f} {cz:.5f} '
            f'rotation 0 1 0 {angle:.6f} '
            'children [ Shape { castShadows FALSE isPickable FALSE '
            'appearance PBRAppearance { baseColor 0.01 0.01 0.01 roughness 0.9 metalness 0 } '
            f'geometry Box {{ size {length:.5f} 0.00400 {self._trail_half_width * 2.0:.5f} }} '
            '} ] '
            f'name "x_plotter_trail_{self._trail_count}" '
            '}'
        )
        request = SpawnNodeFromString.Request()
        request.data = node
        self._trail_client.call_async(request)
        self._trail_count += 1

    def _update_trail(self, point: tuple[float, float], pen_down: bool) -> None:
        if not self._trail_enabled:
            return
        current_world = self._frame.board_to_world(point[0], point[1], plane_y=self._frame.surface_y - 0.006)
        if not pen_down:
            self._trail_last_world = None
            return
        if self._trail_last_world is None:
            self._trail_last_world = current_world
            return
        if math.dist(self._trail_last_world, current_world) >= self._trail_min_spacing:
            self._spawn_trail_segment(self._trail_last_world, current_world)
            self._trail_last_world = current_world

    def _log_target_if_needed(self, sample: _ExecutionSample, effective_pen_down: bool) -> None:
        key = (round(sample.point[0], 4), round(sample.point[1], 4), effective_pen_down)
        if self._last_target_log == key:
            return
        self._last_target_log = key
        self._log.info(
            f'X plotter target: board=({sample.point[0]:.3f},{sample.point[1]:.3f}), '
            f'pen_down={effective_pen_down}, primitive_index={sample.primitive_index}'
        )

    def step(self):
        self._publish_board_info()
        timestep_sec = max(0.001, float(self._robot.getBasicTimeStep()) / 1000.0)
        with self._lock:
            queue = self._queue
            current_point = self._current_point

        if queue:
            sample = queue[0]
            effective_pen_down = self._effective_pen_down(sample.pen_down)
            self._log_target_if_needed(sample, effective_pen_down)
            next_point, reached = self._step_toward(
                current_point,
                sample.point,
                self._carriage_speed * timestep_sec,
            )
            with self._lock:
                self._current_point = next_point
                self._target_pen_down = effective_pen_down
                if reached and self._queue:
                    self._queue.pop(0)
                    if not self._queue:
                        self._publish_status('done')
                        self._log.info('X plotter execution finished.')
            self._set_motors(next_point, effective_pen_down)
            self._publish_pose_topics(next_point, effective_pen_down)
            self._update_trail(next_point, effective_pen_down)
        else:
            effective_pen_down = self._effective_pen_down(False)
            self._set_motors(current_point, effective_pen_down)
            self._publish_pose_topics(current_point, effective_pen_down)
            self._update_trail(current_point, effective_pen_down)

    def cleanup(self):
        self._spin_running = False
        spin_thread = getattr(self, '_spin_thread', None)
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=0.2)
        executor = getattr(self, '_executor', None)
        if executor is not None:
            try:
                executor.remove_node(self._node)
            except Exception:
                pass
            try:
                executor.shutdown(timeout_sec=0.1)
            except Exception:
                pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

