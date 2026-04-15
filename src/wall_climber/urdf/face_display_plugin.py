from __future__ import annotations

import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


class FaceDisplayPlugin:
    def init(self, webots_node, properties):
        self._robot = webots_node.robot
        self._display_name = str(properties.get('display_name', 'face_display'))
        self._default_text = str(properties.get('default_text', 'READY'))
        self._text_topic = str(properties.get('text_topic', '/wall_climber/face/text'))
        self._expression_topic = str(
            properties.get('expression_topic', '/wall_climber/face/expression')
        )
        self._blink_period_sec = max(0.8, float(properties.get('blink_period_sec', '3.8')))
        self._blink_duration_sec = max(0.05, float(properties.get('blink_duration_sec', '0.14')))
        self._spin_timeout_sec = max(0.001, float(properties.get('spin_timeout_sec', '0.01')))

        self._display = self._robot.getDevice(self._display_name)
        self._width = 0
        self._height = 0
        self._text = self._default_text
        self._expression = 'happy'
        self._last_blink_started = time.monotonic()
        self._last_signature = None

        if self._display is None:
            print(f'[FaceDisplayPlugin] WARNING: display "{self._display_name}" not found')
        else:
            self._width = int(self._display.getWidth())
            self._height = int(self._display.getHeight())
            self._redraw(force=True)

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = rclpy.create_node('face_display_plugin')
        self._node.create_subscription(String, self._text_topic, self._text_cb, 1)
        self._node.create_subscription(String, self._expression_topic, self._expression_cb, 1)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop,
            name='face_display_plugin_spin',
            daemon=True,
        )
        self._spin_thread.start()
        self._node.get_logger().info(
            f'Face display plugin ready on device "{self._display_name}".'
        )

    def _text_cb(self, msg: String):
        value = str(msg.data).strip()
        self._text = value[:18] if value else self._default_text

    def _expression_cb(self, msg: String):
        value = str(msg.data).strip().lower()
        if value:
            self._expression = value

    def _spin_loop(self):
        while self._spin_running and rclpy.ok():
            self._executor.spin_once(timeout_sec=self._spin_timeout_sec)

    def _is_blinking(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_blink_started
        if elapsed >= self._blink_period_sec:
            self._last_blink_started = now
            return True
        return elapsed <= self._blink_duration_sec

    def _signature(self):
        return (self._expression, self._text, self._is_blinking())

    def _set_color(self, rgb_hex: int):
        self._display.setColor(int(rgb_hex))

    def _fill_rect(self, x: int, y: int, w: int, h: int):
        self._display.fillRectangle(int(x), int(y), int(w), int(h))

    def _fill_oval(self, x: int, y: int, w: int, h: int):
        # Webots Display supports oval primitives in Python controllers.
        self._display.fillOval(int(x), int(y), int(w), int(h))

    def _draw_text(self, text: str, x: int, y: int):
        self._display.drawText(str(text), int(x), int(y))

    def _redraw(self, force: bool = False):
        if self._display is None:
            return

        signature = self._signature()
        if not force and signature == self._last_signature:
            return
        self._last_signature = signature

        blinking = signature[2]
        width = self._width
        height = self._height

        bg = 0x071014
        bezel = 0x10242A
        fg = 0x63E5F0
        accent = 0xE96A19
        muted = 0x35626A

        self._set_color(bg)
        self._fill_rect(0, 0, width, height)

        self._set_color(bezel)
        self._fill_rect(0, 0, width, max(4, height // 18))
        self._fill_rect(0, height - max(4, height // 18), width, max(4, height // 18))
        self._fill_rect(0, 0, max(4, width // 40), height)
        self._fill_rect(width - max(4, width // 40), 0, max(4, width // 40), height)

        eye_w = max(18, int(width * 0.16))
        eye_h = max(12, int(height * 0.24))
        eye_y = int(height * 0.25)
        left_x = int(width * 0.25) - eye_w // 2
        right_x = int(width * 0.75) - eye_w // 2

        self._set_color(fg)
        if blinking:
            self._fill_rect(left_x, eye_y + eye_h // 2, eye_w, max(3, eye_h // 8))
            self._fill_rect(right_x, eye_y + eye_h // 2, eye_w, max(3, eye_h // 8))
        else:
            expr = self._expression
            if expr in ('sleep', 'sleepy', 'closed'):
                self._fill_rect(left_x, eye_y + eye_h // 2, eye_w, max(4, eye_h // 7))
                self._fill_rect(right_x, eye_y + eye_h // 2, eye_w, max(4, eye_h // 7))
            elif expr in ('angry', 'focus', 'focused'):
                self._fill_rect(left_x, eye_y + eye_h // 3, eye_w, max(8, eye_h // 2))
                self._fill_rect(right_x, eye_y + eye_h // 3, eye_w, max(8, eye_h // 2))
            else:
                self._fill_oval(left_x, eye_y, eye_w, eye_h)
                self._fill_oval(right_x, eye_y, eye_w, eye_h)

        mouth_y = int(height * 0.66)
        mouth_w = int(width * 0.30)
        mouth_x = (width - mouth_w) // 2
        self._set_color(accent)
        if self._expression in ('happy', 'smile', 'ready'):
            self._fill_rect(mouth_x, mouth_y, mouth_w, max(5, height // 16))
            self._fill_rect(mouth_x - max(4, width // 32), mouth_y - max(4, height // 20), max(4, width // 32), max(4, height // 14))
            self._fill_rect(mouth_x + mouth_w, mouth_y - max(4, height // 20), max(4, width // 32), max(4, height // 14))
        elif self._expression in ('sad', 'error'):
            self._fill_rect(mouth_x, mouth_y + max(4, height // 24), mouth_w, max(5, height // 16))
            self._fill_rect(mouth_x - max(4, width // 32), mouth_y + max(6, height // 16), max(4, width // 32), max(4, height // 14))
            self._fill_rect(mouth_x + mouth_w, mouth_y + max(6, height // 16), max(4, width // 32), max(4, height // 14))
        else:
            self._fill_rect(mouth_x, mouth_y, mouth_w, max(5, height // 16))

        if self._text:
            self._set_color(muted)
            self._draw_text(self._text, max(8, width // 10), height - max(18, height // 8))

    def step(self):
        self._redraw(force=False)

    def cleanup(self):
        self._spin_running = False
        spin_thread = getattr(self, '_spin_thread', None)
        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=0.2)
        executor = getattr(self, '_executor', None)
        node = getattr(self, '_node', None)
        if executor is not None and node is not None:
            try:
                executor.remove_node(node)
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
