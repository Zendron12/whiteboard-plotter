"""Helper node that requests robot spawning from Ros2Supervisor.

The normal path imports a URDF robot. For the wall climber we convert the URDF
to a Webots node locally so we can inject a real Display node for the face
screen; the Webots URDF importer does not create Display devices from ROS 2
driver ``<device>`` tags.
"""

import os
import re
import sys

import rclpy
from rclpy.node import Node
from webots_ros2_msgs.srv import SpawnNodeFromString, SpawnUrdfRobot
from webots_ros2_msgs.msg import UrdfRobot


_FACE_DISPLAY_NODE = '''    Display {
      translation -0.006 0 0.052
      children [
        Shape {
          appearance PBRAppearance {
            baseColorMap ImageTexture {
            }
            roughness 1
            metalness 0
          }
          geometry IndexedFaceSet {
            coord Coordinate {
              point [
                -0.117 -0.072 0
                0.117 -0.072 0
                0.117 0.072 0
                -0.117 0.072 0
              ]
            }
            texCoord TextureCoordinate {
              point [
                0 0
                1 0
                1 1
                0 1
              ]
            }
            coordIndex [
              0, 1, 2, 3, -1
            ]
            texCoordIndex [
              0, 1, 2, 3, -1
            ]
          }
        }
      ]
      name "face_display"
      width 256
      height 128
    }
'''


def _load_urdf2webots_converter():
    import webots_ros2_importer

    sys.path.insert(
        1,
        os.path.join(os.path.dirname(webots_ros2_importer.__file__), 'urdf2webots'),
    )
    from urdf2webots.importer import convertUrdfContent

    return convertUrdfContent


def _inject_face_display(robot_node: str, robot_name: str) -> str:
    """Insert a Webots Display as a real Robot child and keep robot name first."""
    robot_node = re.sub(
        rf'\n\s*name "{re.escape(robot_name)}"\n',
        '\n',
        robot_node,
        count=1,
    )
    robot_node = robot_node.replace(
        'Robot {\n',
        f'Robot {{\n  name "{robot_name}"\n',
        1,
    )
    return robot_node.replace('  children [\n', f'  children [\n{_FACE_DISPLAY_NODE}', 1)


class URDFSpawnerNode(Node):
    def __init__(self):
        super().__init__('urdf_spawner')

        # Service path provided by Ros2Supervisor in ROS 2 Humble.
        self.urdf_client = self.create_client(SpawnUrdfRobot, '/Ros2Supervisor/spawn_urdf_robot')
        self.node_client = self.create_client(SpawnNodeFromString, '/Ros2Supervisor/spawn_node_from_string')

        self.declare_parameter('robot_description', '')
        self.declare_parameter('robot_name', 'wall_climber')
        self.declare_parameter('spawn_translation', '0 0 0')
        self.declare_parameter('spawn_rotation', '0 0 1 0')

        max_wait_seconds = 60
        elapsed_seconds = 0
        while not self.urdf_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /Ros2Supervisor/spawn_urdf_robot service...')
            elapsed_seconds += 1
            if elapsed_seconds >= max_wait_seconds:
                raise RuntimeError(
                    'Timed out waiting for /Ros2Supervisor/spawn_urdf_robot service.'
                )

        urdf_content = self.get_parameter('robot_description').value
        robot_name = self.get_parameter('robot_name').value
        spawn_translation = self.get_parameter('spawn_translation').value
        spawn_rotation = self.get_parameter('spawn_rotation').value

        if not urdf_content:
            raise ValueError('Parameter robot_description is empty.')

        if robot_name == 'wall_climber':
            while not self.node_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for /Ros2Supervisor/spawn_node_from_string service...')
                elapsed_seconds += 1
                if elapsed_seconds >= max_wait_seconds:
                    raise RuntimeError(
                        'Timed out waiting for /Ros2Supervisor/spawn_node_from_string service.'
                    )

            convert_urdf_content = _load_urdf2webots_converter()
            robot_node = convert_urdf_content(
                input=urdf_content,
                robotName=robot_name,
                normal=True,
                initTranslation=spawn_translation,
                initRotation=spawn_rotation,
            )
            robot_node = _inject_face_display(robot_node, robot_name)

            request = SpawnNodeFromString.Request()
            request.data = robot_node
            self.get_logger().info(
                f'Sending Webots node spawn request for "{robot_name}" at {spawn_translation}'
                f' with rotation {spawn_rotation}.'
            )
            self.future = self.node_client.call_async(request)
            return

        request = SpawnUrdfRobot.Request()
        robot = UrdfRobot()
        robot.name = robot_name
        robot.robot_description = urdf_content
        robot.translation = spawn_translation
        robot.rotation = spawn_rotation
        # Explicitly request a normal Robot import; relying on message defaults
        # can result in a non-controller robot on some Webots versions.
        robot.normal = True
        request.robot = robot

        self.get_logger().info(
            f'Sending spawn request for "{robot_name}" at {spawn_translation}'
            f' with rotation {spawn_rotation}.'
        )
        self.future = self.urdf_client.call_async(request)


def main():
    rclpy.init()
    node = None
    try:
        node = URDFSpawnerNode()
        rclpy.spin_until_future_complete(node, node.future)

        response = node.future.result()
        if response and response.success:
            node.get_logger().info('Spawn request completed successfully.')
        else:
            node.get_logger().error('Spawn request failed according to Ros2Supervisor response.')
    except Exception as error:
        if node is not None:
            node.get_logger().error(f'URDF spawner failed: {error}')
        else:
            print(f'URDF spawner failed before node creation: {error}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
