from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64, String
from wall_climber_interfaces.msg import CableSetpoint
from wall_climber.runtime_topics import (
    MANUAL_PEN_MODE_TOPIC,
    PEN_MODE_AUTO,
    PEN_MODE_DOWN,
    PEN_MODE_UP,
)


_TRANSIENT_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class CableSupervisorPlugin:
    _BOARD_FRAME_ID = 'board'
    _DEFAULT_ROTATION = [1.0, 0.0, 0.0, 1.5708]
    _SAFE_UNAVAILABLE_GAP = 1.0

    def init(self, webots_node, properties):
        self._supervisor = webots_node.robot
        self._target = None
        self._pen_node = None
        self._left_mount_node = None
        self._right_mount_node = None
        self._root_children = None
        self._step_count = 0
        self._last_status = None

        self._target_name = str(properties.get('target_robot', 'wall_climber'))
        self._board_center_x = float(properties.get('board_center_x', '0.0'))
        self._board_center_z = float(properties.get('board_center_z', '1.8'))
        self._board_width = float(properties.get('board_width', '6.3'))
        self._board_height = float(properties.get('board_height', '3.0'))
        self._board_surface_y = float(properties.get('board_surface_y', '2.410'))
        self._carriage_plane_y = float(properties.get('carriage_plane_y', '2.335'))
        self._margin_left = float(properties.get('margin_left', '0.10'))
        self._margin_right = float(properties.get('margin_right', '0.10'))
        self._margin_top = float(properties.get('margin_top', '0.10'))
        self._margin_bottom = float(properties.get('margin_bottom', '0.10'))
        self._line_height = float(properties.get('line_height', '0.14'))
        self._safe_x_min = float(properties.get('safe_x_min', '0.16'))
        self._safe_x_max = float(properties.get('safe_x_max', '6.14'))
        self._safe_y_min = float(properties.get('safe_y_min', '0.32'))
        self._safe_y_max = float(properties.get('safe_y_max', '2.82'))
        self._corner_keepout_radius = float(properties.get('corner_keepout_radius', '0.36'))

        self._anchor_left = (
            float(properties.get('anchor_left_x', '0.0')),
            float(properties.get('anchor_left_y', '0.0')),
        )
        self._anchor_right = (
            float(properties.get('anchor_right_x', f'{self._board_width}')),
            float(properties.get('anchor_right_y', '0.0')),
        )
        self._attach_left = (
            float(properties.get('carriage_attachment_left_x', '-0.104')),
            float(properties.get('carriage_attachment_left_y', '-0.075')),
        )
        self._attach_right = (
            float(properties.get('carriage_attachment_right_x', '0.104')),
            float(properties.get('carriage_attachment_right_y', '-0.075')),
        )
        self._carriage_width = float(properties.get('carriage_width', '0.29'))
        self._carriage_height = float(properties.get('carriage_height', '0.20'))
        self._mount_left_local = (
            float(properties.get('cable_mount_left_local_x', str(self._attach_left[0]))),
            float(properties.get('cable_mount_left_local_y', str(-self._attach_left[1]))),
            float(properties.get('cable_mount_local_z', '-0.033')),
        )
        self._mount_right_local = (
            float(properties.get('cable_mount_right_local_x', str(self._attach_right[0]))),
            float(properties.get('cable_mount_right_local_y', str(-self._attach_right[1]))),
            float(properties.get('cable_mount_local_z', '-0.033')),
        )
        self._pen_offset = (
            float(properties.get('pen_offset_x', '0.203')),
            float(properties.get('pen_offset_y', '0.020')),
        )
        self._pen_length = float(properties.get('pen_length', '0.082'))
        self._pen_radius = float(properties.get('pen_radius', '0.009'))
        self._pen_mass = float(properties.get('pen_mass', '0.025'))
        self._fallback_tip_local_center = (
            0.0,
            0.0,
            float(properties.get('pen_tip_local_z', '-0.011')),
        )
        self._fallback_tip_radius = float(properties.get('pen_tip_radius', '0.003'))
        self._initial_center = (
            float(properties.get('initial_center_x', f'{self._board_width * 0.5}')),
            float(properties.get('initial_center_y', '0.95')),
        )
        self._pen_contact_engage_gap = float(properties.get('pen_contact_engage_gap', '0.0012'))
        self._pen_contact_release_gap = float(properties.get('pen_contact_release_gap', '0.0022'))
        self._enable_webots_trail = False

        self._board_left = self._board_center_x - self._board_width * 0.5
        self._board_top_z = self._board_center_z + self._board_height * 0.5
        self._writable_x_min = self._margin_left
        self._writable_x_max = self._board_width - self._margin_right
        self._writable_y_min = self._margin_top
        self._writable_y_max = self._board_height - self._margin_bottom
        self._body_safe_x_min = max(self._writable_x_min, (self._carriage_width * 0.5) + self._pen_offset[0])
        self._body_safe_x_max = min(
            self._writable_x_max,
            self._board_width - (self._carriage_width * 0.5) + self._pen_offset[0],
        )
        self._body_safe_y_min = max(self._writable_y_min, (self._carriage_height * 0.5) + self._pen_offset[1])
        self._body_safe_y_max = min(
            self._writable_y_max,
            self._board_height - (self._carriage_height * 0.5) + self._pen_offset[1],
        )

        self._latest_setpoint = None
        self._pen_down_requested = False
        self._manual_pen_mode = PEN_MODE_AUTO
        self._pen_contact_latched = False
        self._current_center = self._initial_center
        self._current_pen_target = (
            self._current_center[0] + self._pen_offset[0],
            self._current_center[1] + self._pen_offset[1],
        )

        self._tip_sphere_local_center = None
        self._tip_sphere_radius = None
        self._tip_geometry_ready = False
        self._tip_geometry_warned = False

        self._trail_half_width = float(properties.get('trail_half_width', '0.010'))
        self._trail_round_segments = max(8, int(properties.get('trail_round_segments', '12')))
        self._trail_max = int(properties.get('trail_max', '8000'))
        self._trail_min_spacing = float(properties.get('trail_min_spacing', '0.0004'))
        self._trail_segment_count = 0
        self._trail_mesh_ready = False
        self._trail_point_field = None
        self._trail_index_field = None
        self._trail_last_dir = None
        self._trail_last_round_pos = None
        self._last_pos = None
        self._trail_disable_cleanup_done = False

        self._cable_points_field = None

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node('cable_supervisor')
        self._node.declare_parameter('enable_webots_trail', False)
        self._enable_webots_trail = bool(self._node.get_parameter('enable_webots_trail').value)
        self._log = self._node.get_logger()

        self._robot_board_pub = self._node.create_publisher(Pose2D, '/wall_climber/robot_pose_board', 1)
        self._pen_board_pub = self._node.create_publisher(PointStamped, '/wall_climber/pen_pose_board', 1)
        self._board_info_pub = self._node.create_publisher(String, '/wall_climber/board_info', _TRANSIENT_QOS)
        self._pen_contact_pub = self._node.create_publisher(Bool, '/wall_climber/pen_contact', 1)
        self._pen_gap_pub = self._node.create_publisher(Float64, '/wall_climber/pen_gap', 1)
        self._status_pub = self._node.create_publisher(String, '/wall_climber/cable_supervisor_status', _TRANSIENT_QOS)
        self._node.create_subscription(
            CableSetpoint,
            '/wall_climber/cable_setpoint',
            self._setpoint_cb,
            _TRANSIENT_QOS,
        )
        self._node.create_subscription(
            String,
            MANUAL_PEN_MODE_TOPIC,
            self._manual_pen_mode_cb,
            _TRANSIENT_QOS,
        )

        try:
            self._root_children = self._supervisor.getRoot().getField('children')
        except Exception:
            self._root_children = None

        self._board_info_json = json.dumps(
            {
                'frame_origin': 'top_left',
                'frame_x_axis': 'right',
                'frame_y_axis': 'down',
                'width': self._board_width,
                'height': self._board_height,
                'writable_x_min': self._writable_x_min,
                'writable_x_max': self._writable_x_max,
                'writable_y_min': self._writable_y_min,
                'writable_y_max': self._writable_y_max,
                'safe_x_min': self._safe_x_min,
                'safe_x_max': self._safe_x_max,
                'safe_y_min': self._safe_y_min,
                'safe_y_max': self._safe_y_max,
                'body_safe_x_min': self._body_safe_x_min,
                'body_safe_x_max': self._body_safe_x_max,
                'body_safe_y_min': self._body_safe_y_min,
                'body_safe_y_max': self._body_safe_y_max,
                'corner_keepout_radius': self._corner_keepout_radius,
                'line_height': self._line_height,
                'anchors': {
                    'left': {'x': self._anchor_left[0], 'y': self._anchor_left[1]},
                    'right': {'x': self._anchor_right[0], 'y': self._anchor_right[1]},
                },
            },
            separators=(',', ':'),
        )
        self._publish_board_info()
        self._set_status('starting')

    def _setpoint_cb(self, msg: CableSetpoint):
        self._latest_setpoint = msg
        if self._manual_pen_mode == PEN_MODE_AUTO:
            self._pen_down_requested = bool(msg.pen_down)

    def _manual_pen_mode_cb(self, msg: String):
        mode = str(msg.data).strip().lower()
        if mode not in (PEN_MODE_AUTO, PEN_MODE_UP, PEN_MODE_DOWN):
            return
        self._manual_pen_mode = mode
        if mode == PEN_MODE_AUTO:
            self._pen_down_requested = bool(self._latest_setpoint.pen_down) if self._latest_setpoint is not None else False
        else:
            self._pen_down_requested = (mode == PEN_MODE_DOWN)

    def _set_status(self, status: str):
        if self._last_status == status:
            return
        self._last_status = status
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _publish_board_info(self):
        msg = String()
        msg.data = self._board_info_json
        self._board_info_pub.publish(msg)

    def _world_to_board(self, world_x, world_z):
        board_x = world_x - self._board_left
        board_y = self._board_top_z - world_z
        return board_x, board_y

    def _board_to_world(self, board_x, board_y, world_y):
        return (self._board_left + board_x, world_y, self._board_top_z - board_y)

    def _publish_robot_pose(self, center_x, center_y):
        msg = Pose2D()
        msg.x = float(center_x)
        msg.y = float(center_y)
        msg.theta = 0.0
        self._robot_board_pub.publish(msg)

    def _publish_pen_pose(self, board_x, board_y):
        msg = PointStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._BOARD_FRAME_ID
        msg.point.x = float(board_x)
        msg.point.y = float(board_y)
        msg.point.z = 0.0
        self._pen_board_pub.publish(msg)

    def _publish_pen_contact(self, contact):
        msg = Bool()
        msg.data = bool(contact)
        self._pen_contact_pub.publish(msg)

    def _publish_pen_gap(self, gap):
        msg = Float64()
        msg.data = float(gap)
        self._pen_gap_pub.publish(msg)

    def _find_target(self):
        root = self._supervisor.getRoot()
        if root is None:
            return
        children = root.getField('children')
        if children is None:
            return
        for index in range(children.getCount()):
            child = children.getMFNode(index)
            if child is None:
                continue
            name_field = child.getField('name')
            if name_field is None:
                continue
            if name_field.getSFString() == self._target_name:
                self._target = child
                self._resolve_cable_mount_nodes()
                self._log.info(f'Found target robot "{self._target_name}".')
                return

    def _resolve_cable_mount_nodes(self):
        self._left_mount_node = None
        self._right_mount_node = None
        if self._target is None:
            return
        if getattr(self._target, 'isProto', lambda: False)():
            try:
                self._left_mount_node = self._target.getFromProtoDef('left_cable_mount')
            except Exception:
                self._left_mount_node = None
            try:
                self._right_mount_node = self._target.getFromProtoDef('right_cable_mount')
            except Exception:
                self._right_mount_node = None
        if self._left_mount_node is None:
            self._left_mount_node = self._find_named_descendant(self._target, 'left_cable_mount')
        if self._right_mount_node is None:
            self._right_mount_node = self._find_named_descendant(self._target, 'right_cable_mount')

    def _find_named_descendant(self, node, target_name, depth=0):
        if node is None or depth > 40:
            return None
        try:
            name_field = node.getField('name')
            if name_field is not None and name_field.getSFString() == target_name:
                return node
        except Exception:
            pass
        try:
            children_field = node.getField('children')
            if children_field is not None:
                for index in range(children_field.getCount()):
                    child = children_field.getMFNode(index)
                    result = self._find_named_descendant(child, target_name, depth + 1)
                    if result is not None:
                        return result
        except Exception:
            pass
        try:
            endpoint_field = node.getField('endPoint')
            if endpoint_field is not None:
                endpoint = endpoint_field.getSFNode()
                result = self._find_named_descendant(endpoint, target_name, depth + 1)
                if result is not None:
                    return result
        except Exception:
            pass
        return None

    def _world_position(self, node):
        if node is None:
            return None
        try:
            position = node.getPosition()
        except Exception:
            return None
        if position is None or len(position) != 3:
            return None
        return (float(position[0]), float(position[1]), float(position[2]))

    def _target_world_from_local(self, local_point):
        if self._target is None:
            return None
        try:
            position = self._target.getPosition()
            orientation = self._target.getOrientation()
        except Exception:
            return None
        if position is None or orientation is None:
            return None
        return self._world_from_local(position, orientation, local_point)

    def _find_joint_endpoint(self, node, motor_name, depth=0):
        if node is None or depth > 40:
            return None
        try:
            device_field = node.getField('device')
            if device_field is not None:
                for index in range(device_field.getCount()):
                    device = device_field.getMFNode(index)
                    if device is None:
                        continue
                    name_field = device.getField('name')
                    if name_field is not None and name_field.getSFString() == motor_name:
                        endpoint_field = node.getField('endPoint')
                        if endpoint_field is not None:
                            return endpoint_field.getSFNode()
        except Exception:
            pass
        try:
            children_field = node.getField('children')
            if children_field is not None:
                for index in range(children_field.getCount()):
                    child = children_field.getMFNode(index)
                    result = self._find_joint_endpoint(child, motor_name, depth + 1)
                    if result is not None:
                        return result
        except Exception:
            pass
        try:
            endpoint_field = node.getField('endPoint')
            if endpoint_field is not None:
                endpoint = endpoint_field.getSFNode()
                result = self._find_joint_endpoint(endpoint, motor_name, depth + 1)
                if result is not None:
                    return result
        except Exception:
            pass
        return None

    def _find_named_descendant(self, node, target_name, depth=0):
        if node is None or depth > 50:
            return None
        try:
            name_field = node.getField('name')
            if name_field is not None and name_field.getSFString() == target_name:
                return node
        except Exception:
            pass
        try:
            children_field = node.getField('children')
            if children_field is not None:
                for index in range(children_field.getCount()):
                    child = children_field.getMFNode(index)
                    result = self._find_named_descendant(child, target_name, depth + 1)
                    if result is not None:
                        return result
        except Exception:
            pass
        try:
            endpoint_field = node.getField('endPoint')
            if endpoint_field is not None:
                endpoint = endpoint_field.getSFNode()
                result = self._find_named_descendant(endpoint, target_name, depth + 1)
                if result is not None:
                    return result
        except Exception:
            pass
        return None

    def _find_named_descendant_contains(self, node, name_fragment, depth=0):
        if node is None or depth > 50:
            return None
        lowered_fragment = str(name_fragment).lower()
        try:
            name_field = node.getField('name')
            if name_field is not None:
                name_value = str(name_field.getSFString())
                if lowered_fragment in name_value.lower():
                    return node
        except Exception:
            pass
        try:
            children_field = node.getField('children')
            if children_field is not None:
                for index in range(children_field.getCount()):
                    child = children_field.getMFNode(index)
                    result = self._find_named_descendant_contains(child, lowered_fragment, depth + 1)
                    if result is not None:
                        return result
        except Exception:
            pass
        try:
            endpoint_field = node.getField('endPoint')
            if endpoint_field is not None:
                endpoint = endpoint_field.getSFNode()
                result = self._find_named_descendant_contains(endpoint, lowered_fragment, depth + 1)
                if result is not None:
                    return result
        except Exception:
            pass
        return None

    def _find_pen_tip_node(self):
        endpoint = self._find_joint_endpoint(self._target, 'pen_slide_joint')
        if endpoint is None:
            return None
        # Prefer explicit tip links when they are present in the converted Webots tree.
        for candidate_name in ('pen_tip_stage', 'pen_tip', 'tip_stage', 'tip_link'):
            tip_node = self._find_named_descendant(endpoint, candidate_name)
            if tip_node is not None:
                return tip_node
        for candidate_fragment in ('pen_tip', 'tip_stage', 'tip'):
            tip_node = self._find_named_descendant_contains(endpoint, candidate_fragment)
            if tip_node is not None:
                self._log.info(
                    f'Using pen-tip node discovered by partial name match "{candidate_fragment}".'
                )
                return tip_node
        self._log.warn(
            'Could not find a named pen-tip node; falling back to pen-slide endpoint geometry search.'
        )
        return endpoint

    def _field(self, node, name):
        try:
            return node.getField(name)
        except Exception:
            return None

    def _field_sfnode(self, node, name):
        field = self._field(node, name)
        if field is None:
            return None
        try:
            return field.getSFNode()
        except Exception:
            return None

    def _field_sfvec3f(self, node, name):
        field = self._field(node, name)
        if field is None:
            return None
        try:
            value = field.getSFVec3f()
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return None

    def _field_sffloat(self, node, name):
        field = self._field(node, name)
        if field is None:
            return None
        try:
            return float(field.getSFFloat())
        except Exception:
            return None

    def _vec3_add(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def _is_sphere_node(self, node):
        radius = self._field_sffloat(node, 'radius')
        if radius is None:
            return False
        return self._field(node, 'height') is None and self._field(node, 'size') is None

    def _collect_tip_sphere_geometries(self, node, accumulated_translation=(0.0, 0.0, 0.0), out=None):
        if out is None:
            out = []
        if node is None:
            return out
        local_translation = accumulated_translation
        translation = self._field_sfvec3f(node, 'translation')
        if translation is not None:
            local_translation = self._vec3_add(local_translation, translation)

        if self._is_sphere_node(node):
            radius = self._field_sffloat(node, 'radius')
            if radius is not None:
                out.append((local_translation, radius))

        geometry_node = self._field_sfnode(node, 'geometry')
        if geometry_node is not None:
            self._collect_tip_sphere_geometries(geometry_node, local_translation, out)

        child_node = self._field_sfnode(node, 'child')
        if child_node is not None:
            self._collect_tip_sphere_geometries(child_node, local_translation, out)

        endpoint_node = self._field_sfnode(node, 'endPoint')
        if endpoint_node is not None:
            self._collect_tip_sphere_geometries(endpoint_node, local_translation, out)

        bounding_object = self._field_sfnode(node, 'boundingObject')
        if bounding_object is not None:
            self._collect_tip_sphere_geometries(bounding_object, local_translation, out)

        children_field = self._field(node, 'children')
        if children_field is not None:
            try:
                for index in range(children_field.getCount()):
                    child = children_field.getMFNode(index)
                    self._collect_tip_sphere_geometries(child, local_translation, out)
            except Exception:
                pass
        return out

    def _select_tip_sphere_geometry(self, candidates):
        if not candidates:
            return None
        # In the pen local frame, more negative Z is closer to the board.
        # Tie-break on smaller radius so a support caster sphere is deprioritized.
        return min(candidates, key=lambda item: (item[0][2], item[1]))

    def _resolve_tip_geometry(self):
        if self._tip_geometry_ready:
            return True
        if self._pen_node is None:
            return False
        bounding_object = self._field_sfnode(self._pen_node, 'boundingObject')
        candidates = self._collect_tip_sphere_geometries(bounding_object) if bounding_object is not None else []
        if not candidates:
            candidates = self._collect_tip_sphere_geometries(self._pen_node)
        resolved = self._select_tip_sphere_geometry(candidates)
        if resolved is not None:
            self._tip_sphere_local_center, self._tip_sphere_radius = resolved
            self._tip_geometry_ready = True
            self._tip_geometry_warned = False
            self._log.info(
                f'Resolved pen tip sphere: center={self._tip_sphere_local_center} '
                f'radius={self._tip_sphere_radius:.5f} '
                f'(candidates={len(candidates)})'
            )
            return True
        self._tip_sphere_local_center = self._fallback_tip_local_center
        self._tip_sphere_radius = self._fallback_tip_radius
        self._tip_geometry_ready = True
        if not self._tip_geometry_warned:
            self._log.warn(
                'Pen tip collision sphere is unavailable; using configured pen-tip fallback geometry.'
            )
            self._tip_geometry_warned = True
        return True

    def _world_from_local(self, position, orientation, local_point):
        if orientation is None or len(orientation) < 9:
            return None
        return (
            float(position[0])
            + float(orientation[0]) * local_point[0]
            + float(orientation[1]) * local_point[1]
            + float(orientation[2]) * local_point[2],
            float(position[1])
            + float(orientation[3]) * local_point[0]
            + float(orientation[4]) * local_point[1]
            + float(orientation[5]) * local_point[2],
            float(position[2])
            + float(orientation[6]) * local_point[0]
            + float(orientation[7]) * local_point[1]
            + float(orientation[8]) * local_point[2],
        )

    def _solve_center_from_lengths(self, left_length, right_length):
        center_left = (
            self._anchor_left[0] - self._attach_left[0],
            self._anchor_left[1] - self._attach_left[1],
        )
        center_right = (
            self._anchor_right[0] - self._attach_right[0],
            self._anchor_right[1] - self._attach_right[1],
        )
        dx = center_right[0] - center_left[0]
        dy = center_right[1] - center_left[1]
        distance = math.hypot(dx, dy)
        if distance <= 1.0e-9:
            return None
        if left_length + right_length < distance:
            return None
        if left_length + distance < right_length or right_length + distance < left_length:
            return None
        a = (left_length * left_length - right_length * right_length + distance * distance) / (2.0 * distance)
        h_sq = max(0.0, left_length * left_length - a * a)
        h = math.sqrt(h_sq)
        px = center_left[0] + a * dx / distance
        py = center_left[1] + a * dy / distance
        rx = -dy * h / distance
        ry = dx * h / distance
        solution_a = (px + rx, py + ry)
        solution_b = (px - rx, py - ry)
        return solution_a if solution_a[1] >= solution_b[1] else solution_b

    def _apply_target_pose(self, center_x, center_y):
        if self._target is None:
            return
        translation = self._board_to_world(center_x, center_y, self._carriage_plane_y)
        translation_field = self._target.getField('translation')
        rotation_field = self._target.getField('rotation')
        if translation_field is not None:
            translation_field.setSFVec3f(list(translation))
        if rotation_field is not None:
            rotation_field.setSFRotation(self._DEFAULT_ROTATION)
        try:
            self._target.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        except Exception:
            pass
        self._current_center = (center_x, center_y)
        self._current_pen_target = (
            center_x + self._pen_offset[0],
            center_y + self._pen_offset[1],
        )

    def _find_root_child_index(self, target_node):
        if self._root_children is None or target_node is None:
            return None
        try:
            target_id = target_node.getId()
        except Exception:
            return None
        try:
            count = self._root_children.getCount()
        except Exception:
            return None
        for index in range(count):
            try:
                child = self._root_children.getMFNode(index)
            except Exception:
                continue
            if child is None:
                continue
            try:
                if child.getId() == target_id:
                    return index
            except Exception:
                continue
        return None

    def _drop_existing_node(self, def_name):
        node = self._supervisor.getFromDef(def_name)
        if node is None:
            return
        index = self._find_root_child_index(node)
        if index is None:
            return
        try:
            self._root_children.removeMF(index)
        except Exception:
            pass

    def _bind_cable_fields(self):
        cable_node = self._supervisor.getFromDef('CABLE_LINES')
        if cable_node is None:
            return False
        try:
            children = cable_node.getField('children')
            shape = children.getMFNode(0)
            geometry = shape.getField('geometry').getSFNode()
            coord = geometry.getField('coord').getSFNode()
            self._cable_points_field = coord.getField('point')
        except Exception:
            self._cable_points_field = None
        return self._cable_points_field is not None

    def _ensure_cable_visuals(self):
        if self._cable_points_field is not None:
            return True
        if self._root_children is None:
            return False
        self._drop_existing_node('CABLE_LINES')
        node_str = (
            'DEF CABLE_LINES Transform { '
            'children [ Shape { '
            'appearance Appearance { '
            'material Material { diffuseColor 0.96 0.38 0.10 emissiveColor 0.22 0.07 0.02 } '
            '} '
            'geometry IndexedLineSet { '
            'coord Coordinate { point [ 0 0 0, 0 0 0, 0 0 0, 0 0 0, 0 0 0, 0 0 0, 0 0 0, 0 0 0 ] } '
            'coordIndex [ 0 1 -1 2 3 -1 4 5 -1 6 7 -1 ] '
            '} '
            '} ] '
            '}'
        )
        try:
            self._root_children.importMFNodeFromString(-1, node_str)
        except Exception as exc:
            self._log.warn(f'Failed to create cable visual lines: {exc}')
            return False
        return self._bind_cable_fields()

    def _expanded_cable_points(self, anchor_world, attach_world, half_width=0.0045):
        dx = float(attach_world[0]) - float(anchor_world[0])
        dz = float(attach_world[2]) - float(anchor_world[2])
        norm = math.hypot(dx, dz)
        if norm <= 1.0e-9:
            nx, nz = 0.0, 1.0
        else:
            nx = -dz / norm
            nz = dx / norm
        ox = nx * half_width
        oz = nz * half_width
        return (
            (float(anchor_world[0]) + ox, float(anchor_world[1]), float(anchor_world[2]) + oz),
            (float(attach_world[0]) + ox, float(attach_world[1]), float(attach_world[2]) + oz),
            (float(anchor_world[0]) - ox, float(anchor_world[1]), float(anchor_world[2]) - oz),
            (float(attach_world[0]) - ox, float(attach_world[1]), float(attach_world[2]) - oz),
        )

    def _update_cable_visuals(self, center_x, center_y):
        if not self._ensure_cable_visuals():
            return
        anchor_y = self._board_surface_y - 0.002
        left_anchor_world = self._board_to_world(self._anchor_left[0], self._anchor_left[1], anchor_y)
        right_anchor_world = self._board_to_world(self._anchor_right[0], self._anchor_right[1], anchor_y)
        left_attach_world = self._world_position(self._left_mount_node)
        if left_attach_world is None:
            left_attach_world = self._target_world_from_local(self._mount_left_local)
        if left_attach_world is None:
            left_attach_world = self._board_to_world(center_x + self._attach_left[0], center_y + self._attach_left[1], self._carriage_plane_y)
        right_attach_world = self._world_position(self._right_mount_node)
        if right_attach_world is None:
            right_attach_world = self._target_world_from_local(self._mount_right_local)
        if right_attach_world is None:
            right_attach_world = self._board_to_world(center_x + self._attach_right[0], center_y + self._attach_right[1], self._carriage_plane_y)
        points = [
            *self._expanded_cable_points(left_anchor_world, left_attach_world),
            *self._expanded_cable_points(right_anchor_world, right_attach_world),
        ]
        for index, point in enumerate(points):
            self._cable_points_field.setMFVec3f(index, [float(point[0]), float(point[1]), float(point[2])])

    def _reset_trail_runtime_state(self):
        self._last_pos = None
        self._trail_segment_count = 0
        self._trail_mesh_ready = False
        self._trail_point_field = None
        self._trail_index_field = None
        self._trail_last_dir = None
        self._trail_last_round_pos = None

    def _cleanup_trail_if_disabled(self):
        if self._enable_webots_trail or self._trail_disable_cleanup_done:
            return
        self._drop_existing_node('TRAIL')
        self._reset_trail_runtime_state()
        self._trail_disable_cleanup_done = True

    def _bind_trail_mesh_fields(self, trail_node):
        if trail_node is None:
            return False
        try:
            children = trail_node.getField('children')
            shape = children.getMFNode(0)
            geometry = shape.getField('geometry').getSFNode()
            coord = geometry.getField('coord').getSFNode()
            self._trail_point_field = coord.getField('point')
            self._trail_index_field = geometry.getField('coordIndex')
        except Exception:
            self._trail_point_field = None
            self._trail_index_field = None
            return False
        self._trail_mesh_ready = self._trail_point_field is not None and self._trail_index_field is not None
        return self._trail_mesh_ready

    def _init_trail_mesh(self):
        if not self._enable_webots_trail or self._root_children is None:
            return False
        trail_node = self._supervisor.getFromDef('TRAIL')
        if trail_node is not None and self._bind_trail_mesh_fields(trail_node):
            return True
        self._drop_existing_node('TRAIL')
        line_y = self._board_surface_y - 0.006
        node_str = (
            f'DEF TRAIL Transform {{ translation 0 {line_y:.5f} 0 '
            'children [ Shape { castShadows FALSE isPickable FALSE '
            'appearance Appearance { material Material { diffuseColor 0 0 0 emissiveColor 0.05 0.05 0.05 } } '
            'geometry IndexedFaceSet { solid FALSE '
            'coord Coordinate { point [ -100 0 -100, -99.999 0 -100, -99.999 0 -99.999, -100 0 -99.999 ] } '
            'coordIndex [ 0 1 2 3 -1 ] } } ] }'
        )
        try:
            self._root_children.importMFNodeFromString(-1, node_str)
        except Exception as exc:
            self._log.warn(f'Failed to create trail mesh: {exc}')
            return False
        trail_node = self._supervisor.getFromDef('TRAIL')
        return self._bind_trail_mesh_fields(trail_node)

    def _add_trail_quad(self, x0, z0, x1, z1):
        if not self._enable_webots_trail or self._trail_segment_count >= self._trail_max:
            return
        dx = x1 - x0
        dz = z1 - z0
        length = math.sqrt(dx * dx + dz * dz)
        if length < 1.0e-6:
            return
        dx /= length
        dz /= length
        hw = self._trail_half_width
        px, pz = -dz * hw, dx * hw
        n = self._trail_point_field.getCount()
        self._trail_point_field.insertMFVec3f(-1, [x0 + px, 0, z0 + pz])
        self._trail_point_field.insertMFVec3f(-1, [x0 - px, 0, z0 - pz])
        self._trail_point_field.insertMFVec3f(-1, [x1 - px, 0, z1 - pz])
        self._trail_point_field.insertMFVec3f(-1, [x1 + px, 0, z1 + pz])
        self._trail_index_field.insertMFInt32(-1, n)
        self._trail_index_field.insertMFInt32(-1, n + 1)
        self._trail_index_field.insertMFInt32(-1, n + 2)
        self._trail_index_field.insertMFInt32(-1, n + 3)
        self._trail_index_field.insertMFInt32(-1, -1)
        self._trail_segment_count += 1

    def _trail_round_guard_dist(self):
        return max(self._trail_min_spacing * 0.5, self._trail_half_width * 0.35)

    def _should_skip_round_feature(self, x, z):
        if self._trail_last_round_pos is None:
            return False
        lx, lz = self._trail_last_round_pos
        return math.hypot(x - lx, z - lz) < self._trail_round_guard_dist()

    def _add_trail_round_cap(self, x, z):
        if not self._enable_webots_trail or self._trail_segment_count >= self._trail_max:
            return
        if self._should_skip_round_feature(x, z):
            return
        hw = self._trail_half_width
        segments = self._trail_round_segments
        n = self._trail_point_field.getCount()
        for index in range(segments):
            angle = (2.0 * math.pi * index) / segments
            self._trail_point_field.insertMFVec3f(
                -1,
                [x + math.cos(angle) * hw, 0, z + math.sin(angle) * hw],
            )
            self._trail_index_field.insertMFInt32(-1, n + index)
        self._trail_index_field.insertMFInt32(-1, -1)
        self._trail_segment_count += 1
        self._trail_last_round_pos = (x, z)

    def _draw_line_to(self, x, z):
        if not self._enable_webots_trail:
            return
        if not self._trail_mesh_ready and not self._init_trail_mesh():
            return
        if self._last_pos is None:
            self._last_pos = (x, z)
            self._trail_last_dir = None
            return
        lx, lz = self._last_pos
        dx = x - lx
        dz = z - lz
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < self._trail_min_spacing:
            return
        dir_x = dx / dist
        dir_z = dz / dist
        if self._trail_last_dir is not None:
            prev_x, prev_z = self._trail_last_dir
            dot = max(-1.0, min(1.0, prev_x * dir_x + prev_z * dir_z))
            if dot < 0.995:
                self._add_trail_round_cap(lx, lz)
        self._add_trail_quad(lx, lz, x, z)
        self._last_pos = (x, z)
        self._trail_last_dir = (dir_x, dir_z)

    def _update_pen_state(self):
        if self._pen_node is None:
            self._pen_node = self._find_pen_tip_node()
            if self._pen_node is not None:
                self._tip_geometry_ready = False
                self._tip_geometry_warned = False
                self._resolve_tip_geometry()
        if self._pen_node is None:
            pen_x, pen_y = self._current_pen_target
            self._publish_pen_pose(pen_x, pen_y)
            self._publish_pen_gap(self._SAFE_UNAVAILABLE_GAP)
            self._publish_pen_contact(False)
            self._pen_contact_latched = False
            return

        try:
            pen_pos = self._pen_node.getPosition()
            pen_orientation = self._pen_node.getOrientation()
        except Exception:
            self._pen_node = None
            self._tip_geometry_ready = False
            self._tip_geometry_warned = False
            return

        tip_geometry_ready = self._resolve_tip_geometry()
        tip_center_world = None
        gap = self._SAFE_UNAVAILABLE_GAP
        if tip_geometry_ready and pen_orientation is not None:
            tip_center_world = self._world_from_local(
                pen_pos,
                pen_orientation,
                self._tip_sphere_local_center,
            )
        if tip_center_world is not None:
            tip_surface_y = tip_center_world[1] + self._tip_sphere_radius
            gap = self._board_surface_y - tip_surface_y

        if self._pen_down_requested:
            if self._pen_contact_latched:
                contact = gap <= self._pen_contact_release_gap
            else:
                contact = gap <= self._pen_contact_engage_gap
        else:
            contact = False

        effective_pen_world = tip_center_world if tip_center_world is not None else pen_pos
        pen_board_x, pen_board_y = self._world_to_board(effective_pen_world[0], effective_pen_world[2])
        self._publish_pen_pose(pen_board_x, pen_board_y)
        self._publish_pen_gap(gap)
        self._publish_pen_contact(contact)

        if contact:
            if self._enable_webots_trail:
                self._draw_line_to(effective_pen_world[0], effective_pen_world[2])
        else:
            if self._enable_webots_trail and self._last_pos is not None:
                self._add_trail_round_cap(self._last_pos[0], self._last_pos[1])
            self._last_pos = None
            self._trail_last_dir = None
            self._trail_last_round_pos = None

        self._pen_contact_latched = contact

    def step(self):
        rclpy.spin_once(self._node, timeout_sec=0)
        self._step_count += 1
        self._publish_board_info()

        if self._target is None:
            self._find_target()
            if self._target is None:
                self._set_status('waiting_for_robot')
                return
            self._apply_target_pose(self._current_center[0], self._current_center[1])

        if self._latest_setpoint is None:
            self._set_status('waiting_for_setpoint')
            self._apply_target_pose(self._current_center[0], self._current_center[1])
        else:
            solved_center = self._solve_center_from_lengths(
                float(self._latest_setpoint.left_cable_length),
                float(self._latest_setpoint.right_cable_length),
            )
            if solved_center is None:
                self._set_status('error')
            else:
                self._apply_target_pose(solved_center[0], solved_center[1])
                self._set_status('tracking')

        self._publish_robot_pose(self._current_center[0], self._current_center[1])
        self._update_cable_visuals(self._current_center[0], self._current_center[1])
        self._update_pen_state()
        self._cleanup_trail_if_disabled()
