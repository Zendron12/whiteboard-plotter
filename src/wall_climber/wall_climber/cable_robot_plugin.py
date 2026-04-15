from __future__ import annotations

import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from wall_climber_interfaces.msg import CableSetpoint
from wall_climber.runtime_topics import (
    MANUAL_PEN_MODE_TOPIC,
    PEN_MODE_AUTO,
    PEN_MODE_DOWN,
    PEN_MODE_UP,
)


class CableRobotPlugin:
    def init(self, webots_node, properties):
        self._robot = webots_node.robot
        self._pen_joint_name = str(properties.get('pen_joint_name', 'pen_slide_joint'))
        self._pen_slide_velocity = max(0.01, float(properties.get('pen_slide_velocity', '0.30')))
        self._pen_up = float(properties.get('pen_up_position', '0.002'))
        self._pen_down = float(properties.get('pen_down_position', '-0.005'))
        self._spin_timeout_sec = max(0.001, float(properties.get('spin_timeout_sec', '0.01')))

        self._pen_motor = self._robot.getDevice(self._pen_joint_name)
        self._target_pen_position = self._pen_up
        self._manual_pen_mode = PEN_MODE_AUTO

        if self._pen_motor is not None:
            self._pen_motor.setPosition(self._pen_up)
            self._pen_motor.setVelocity(self._pen_slide_velocity)
        else:
            print(f'[CableRobotPlugin] WARNING: pen motor "{self._pen_joint_name}" not found')

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node('cable_robot_plugin')
        self._node.create_subscription(
            CableSetpoint,
            '/wall_climber/cable_setpoint',
            self._setpoint_cb,
            1,
        )
        self._node.create_subscription(
            String,
            MANUAL_PEN_MODE_TOPIC,
            self._manual_pen_mode_cb,
            1,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name='cable_robot_plugin_spin',
            daemon=True,
        )
        self._spin_thread.start()
        self._node.get_logger().info('Cable robot plugin ready.')

    def _setpoint_cb(self, msg: CableSetpoint):
        if self._manual_pen_mode != PEN_MODE_AUTO:
            return
        self._target_pen_position = self._pen_down if bool(msg.pen_down) else self._pen_up

    def _manual_pen_mode_cb(self, msg: String):
        mode = str(msg.data).strip().lower()
        if mode not in (PEN_MODE_AUTO, PEN_MODE_UP, PEN_MODE_DOWN):
            return
        self._manual_pen_mode = mode
        if mode == PEN_MODE_DOWN:
            self._target_pen_position = self._pen_down
        else:
            self._target_pen_position = self._pen_up

    def _spin_loop(self):
        while self._spin_running and rclpy.ok():
            self._executor.spin_once(timeout_sec=self._spin_timeout_sec)

    def step(self):
        if self._pen_motor is not None:
            self._pen_motor.setPosition(self._target_pen_position)

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
