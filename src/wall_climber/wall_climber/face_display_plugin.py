from __future__ import annotations

import math
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from std_msgs.msg import String

try:
    from rclpy.executors import ShutdownException
except ImportError:  # pragma: no cover - older rclpy versions do not expose this symbol
    class ShutdownException(Exception):
        pass


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
        while self._spin_running:
            try:
                if not rclpy.ok():
                    break
                self._executor.spin_once(timeout_sec=self._spin_timeout_sec)
            except (ExternalShutdownException, ShutdownException):
                break
            except Exception:
                if not self._spin_running or not rclpy.ok():
                    break
                try:
                    self._node.get_logger().error('Face display plugin spin loop stopped unexpectedly.')
                except Exception:
                    pass
                break

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

    def _draw_pixel(self, x: int, y: int) -> None:
        self._display.drawPixel(int(x), int(y))

    def _draw_line(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._display.drawLine(int(x1), int(y1), int(x2), int(y2))

    def _fill_arc_approximation(self, cx: int, cy: int, width: int, height: int,
                                 start_deg: float, sweep_deg: float,
                                 thickness: int = 2) -> None:
        """Draw a curved band as a series of short line segments.

        Webots' Display API does not expose a native arc primitive, so we
        approximate a thick arc by walking the ellipse and drawing short
        radial line stubs. This keeps curved mouth/eyebrow strokes smooth.
        """
        if width <= 0 or height <= 0:
            return
        segments = max(18, int(max(width, height) // 2))
        step = float(sweep_deg) / float(segments)
        rx_outer = width / 2.0
        ry_outer = height / 2.0
        rx_inner = max(0.5, rx_outer - thickness)
        ry_inner = max(0.5, ry_outer - thickness)
        for index in range(segments + 1):
            angle = math.radians(start_deg + step * index)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            x_outer = cx + rx_outer * cos_a
            y_outer = cy + ry_outer * sin_a
            x_inner = cx + rx_inner * cos_a
            y_inner = cy + ry_inner * sin_a
            self._draw_line(x_outer, y_outer, x_inner, y_inner)

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
        expr = self._expression
        width = self._width
        height = self._height

        # Palette: deep teal background with cyan eyes and warm accent.
        bg = 0x071014
        bezel = 0x10242A
        bezel_glow = 0x1A3942
        fg = 0x63E5F0
        fg_dim = 0x2D6672
        accent = 0xE96A19
        muted = 0x35626A
        white = 0xE8F6FA

        # Background.
        self._set_color(bg)
        self._fill_rect(0, 0, width, height)

        # Inner bezel frame with two-tone band so the screen looks like a real LCD.
        border = max(3, min(width, height) // 32)
        self._set_color(bezel)
        self._fill_rect(0, 0, width, border)
        self._fill_rect(0, height - border, width, border)
        self._fill_rect(0, 0, border, height)
        self._fill_rect(width - border, 0, border, height)
        self._set_color(bezel_glow)
        inner = max(1, border // 2)
        self._fill_rect(border, border, width - 2 * border, inner)
        self._fill_rect(border, height - border - inner, width - 2 * border, inner)

        # Eye geometry.
        eye_w = max(18, int(width * 0.18))
        eye_h = max(14, int(height * 0.30))
        eye_y_center = int(height * 0.42)
        eye_y = eye_y_center - eye_h // 2
        left_cx = int(width * 0.30)
        right_cx = int(width * 0.70)
        left_x = left_cx - eye_w // 2
        right_x = right_cx - eye_w // 2

        # Eyebrows above the eyes. Their tilt encodes emotion.
        brow_w = int(eye_w * 1.10)
        brow_h = max(3, height // 28)
        brow_y = max(border + 2, eye_y - int(eye_h * 0.45))
        self._set_color(fg_dim)
        self._draw_eyebrows(expr, left_cx, right_cx, brow_y, brow_w, brow_h)

        # Eyes.
        if blinking:
            self._set_color(fg)
            line_h = max(3, eye_h // 7)
            self._fill_rect(left_x, eye_y + eye_h // 2 - line_h // 2, eye_w, line_h)
            self._fill_rect(right_x, eye_y + eye_h // 2 - line_h // 2, eye_w, line_h)
        elif expr in ('sleep', 'sleepy', 'closed'):
            # Drooped closed eyes: a gentle curve instead of a flat line.
            self._set_color(fg)
            self._fill_arc_approximation(
                left_cx, eye_y + eye_h // 2,
                eye_w, eye_h, start_deg=200.0, sweep_deg=140.0,
                thickness=max(3, eye_h // 6),
            )
            self._fill_arc_approximation(
                right_cx, eye_y + eye_h // 2,
                eye_w, eye_h, start_deg=200.0, sweep_deg=140.0,
                thickness=max(3, eye_h // 6),
            )
        elif expr in ('angry', 'focus', 'focused'):
            # Narrow slit eyes centred vertically.
            self._set_color(fg)
            slit_h = max(8, eye_h // 2)
            self._fill_oval(left_x, eye_y + (eye_h - slit_h) // 2, eye_w, slit_h)
            self._fill_oval(right_x, eye_y + (eye_h - slit_h) // 2, eye_w, slit_h)
            # Pupil.
            pupil_w = max(4, eye_w // 3)
            pupil_h = max(4, slit_h // 2)
            self._set_color(bg)
            self._fill_oval(
                left_x + (eye_w - pupil_w) // 2,
                eye_y + (eye_h - pupil_h) // 2,
                pupil_w, pupil_h,
            )
            self._fill_oval(
                right_x + (eye_w - pupil_w) // 2,
                eye_y + (eye_h - pupil_h) // 2,
                pupil_w, pupil_h,
            )
        else:
            # Happy / neutral / ready: large oval eyes with pupil and sparkle.
            self._set_color(fg)
            self._fill_oval(left_x, eye_y, eye_w, eye_h)
            self._fill_oval(right_x, eye_y, eye_w, eye_h)
            pupil_w = max(5, eye_w // 3)
            pupil_h = max(5, eye_h // 3)
            pupil_off_y = eye_h // 4
            self._set_color(bg)
            self._fill_oval(
                left_x + (eye_w - pupil_w) // 2,
                eye_y + pupil_off_y,
                pupil_w, pupil_h,
            )
            self._fill_oval(
                right_x + (eye_w - pupil_w) // 2,
                eye_y + pupil_off_y,
                pupil_w, pupil_h,
            )
            # Sparkle: tiny white highlight on the upper-right of each eye.
            sparkle = max(2, min(eye_w, eye_h) // 8)
            self._set_color(white)
            self._fill_oval(
                left_x + eye_w - 2 * sparkle - sparkle // 2,
                eye_y + sparkle,
                sparkle, sparkle,
            )
            self._fill_oval(
                right_x + eye_w - 2 * sparkle - sparkle // 2,
                eye_y + sparkle,
                sparkle, sparkle,
            )

        # Mouth. Curved for happy/sad, flat bar for neutral.
        mouth_cy = int(height * 0.76)
        mouth_w = int(width * 0.34)
        mouth_h = max(6, height // 10)
        mouth_cx = width // 2
        mouth_thickness = max(3, height // 20)
        self._set_color(accent)
        if expr in ('happy', 'smile', 'ready'):
            self._fill_arc_approximation(
                mouth_cx, mouth_cy - mouth_h // 2,
                mouth_w, mouth_h,
                start_deg=20.0, sweep_deg=140.0,
                thickness=mouth_thickness,
            )
        elif expr in ('sad', 'error'):
            self._fill_arc_approximation(
                mouth_cx, mouth_cy + mouth_h // 2,
                mouth_w, mouth_h,
                start_deg=200.0, sweep_deg=140.0,
                thickness=mouth_thickness,
            )
        elif expr in ('angry',):
            # Slight frown.
            self._fill_arc_approximation(
                mouth_cx, mouth_cy + mouth_h // 3,
                int(mouth_w * 0.8), mouth_h,
                start_deg=200.0, sweep_deg=140.0,
                thickness=mouth_thickness,
            )
        else:
            # Neutral / focus: subtle horizontal bar with rounded ends.
            bar_h = max(4, mouth_thickness - 1)
            bar_w = int(mouth_w * 0.75)
            self._fill_rect(mouth_cx - bar_w // 2, mouth_cy - bar_h // 2, bar_w, bar_h)
            cap_r = max(2, bar_h // 2)
            self._fill_oval(mouth_cx - bar_w // 2 - cap_r, mouth_cy - bar_h // 2, cap_r * 2, bar_h)
            self._fill_oval(mouth_cx + bar_w // 2 - cap_r, mouth_cy - bar_h // 2, cap_r * 2, bar_h)

        # Status caption along the bottom bezel.
        if self._text:
            self._set_color(muted)
            caption_y = height - max(14, height // 10)
            caption_x = max(8, width // 16)
            self._draw_text(self._text, caption_x, caption_y)

    def _draw_eyebrows(self, expr: str, left_cx: int, right_cx: int,
                        brow_y: int, brow_w: int, brow_h: int) -> None:
        """Draw eyebrows as short thick bars with per-emotion tilt."""
        half_w = brow_w // 2
        slope_px = max(1, brow_h)
        if expr in ('angry', 'focus', 'focused'):
            # Inner ends drop: /  \
            self._draw_line(left_cx - half_w, brow_y, left_cx + half_w, brow_y + slope_px)
            self._draw_line(left_cx - half_w, brow_y + 1, left_cx + half_w, brow_y + slope_px + 1)
            self._draw_line(right_cx - half_w, brow_y + slope_px, right_cx + half_w, brow_y)
            self._draw_line(right_cx - half_w, brow_y + slope_px + 1, right_cx + half_w, brow_y + 1)
        elif expr in ('sad', 'error'):
            # Inner ends rise: \  /
            self._draw_line(left_cx - half_w, brow_y + slope_px, left_cx + half_w, brow_y)
            self._draw_line(left_cx - half_w, brow_y + slope_px + 1, left_cx + half_w, brow_y + 1)
            self._draw_line(right_cx - half_w, brow_y, right_cx + half_w, brow_y + slope_px)
            self._draw_line(right_cx - half_w, brow_y + 1, right_cx + half_w, brow_y + slope_px + 1)
        elif expr in ('sleep', 'sleepy', 'closed'):
            # Relaxed low bars.
            for index in range(brow_h):
                offset = index
                self._draw_line(left_cx - half_w + 2, brow_y + offset + 2, left_cx + half_w - 2, brow_y + offset + 2)
                self._draw_line(right_cx - half_w + 2, brow_y + offset + 2, right_cx + half_w - 2, brow_y + offset + 2)
        else:
            # Neutral bars.
            for index in range(brow_h):
                self._draw_line(left_cx - half_w, brow_y + index, left_cx + half_w, brow_y + index)
                self._draw_line(right_cx - half_w, brow_y + index, right_cx + half_w, brow_y + index)

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
