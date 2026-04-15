import os
import socket
import shutil
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher

from wall_climber.shared_config import load_shared_config


def _require_supervisor_action(webots_launcher: WebotsLauncher):
    supervisor_action = getattr(webots_launcher, '_supervisor', None)
    if supervisor_action is None:
        raise RuntimeError(
            'WebotsLauncher no longer exposes the internal "_supervisor" action. '
            'Update my_robot.launch.py for the current webots_ros2_driver API.'
        )
    return supervisor_action


def _port_is_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _select_webots_port(requested_port: int, *, attempts: int = 32) -> int:
    base = max(1024, int(requested_port))
    for offset in range(attempts):
        candidate = base + offset
        if _port_is_available(candidate):
            return candidate
    raise RuntimeError(
        f'Unable to find a free Webots port starting at {base} (checked {attempts} ports).'
    )


def generate_launch_description():
    package_name = 'wall_climber'
    pkg_dir = get_package_share_directory(package_name)
    shared = load_shared_config()
    requested_webots_port = os.environ.get('WEBOTS_PORT', '1234')
    try:
        selected_webots_port = _select_webots_port(int(requested_webots_port))
    except ValueError as exc:
        raise RuntimeError(
            f'WEBOTS_PORT must be an integer, got {requested_webots_port!r}.'
        ) from exc
    webots_port = str(selected_webots_port)
    if webots_port != requested_webots_port:
        print(
            f'[wall_climber.launch] Requested Webots port {requested_webots_port} is busy; '
            f'using {webots_port} instead.'
        )
    webots_runtime_root = Path('/tmp') / 'webots' / os.environ.get('USER', 'user')
    try:
        webots_runtime_root.mkdir(parents=True, exist_ok=True)
        stale_port_dir = webots_runtime_root / webots_port
        if stale_port_dir.exists():
            shutil.rmtree(stale_port_dir)
    except OSError:
        pass

    webots_prefix = LaunchConfiguration('webots_prefix')
    enable_webots_trail = LaunchConfiguration('enable_webots_trail')
    writer_mode = LaunchConfiguration('writer_mode')

    world_path = os.path.join(pkg_dir, 'worlds', 'wall_world.wbt')
    climber_xacro_path = os.path.join(pkg_dir, 'urdf', 'my_robot.urdf.xacro')
    supervisor_xacro_path = os.path.join(pkg_dir, 'urdf', 'cable_supervisor.urdf.xacro')

    climber_description = ParameterValue(Command(['xacro ', climber_xacro_path]), value_type=str)

    webots = WebotsLauncher(
        world=world_path,
        mode='realtime',
        ros2_supervisor=True,
        port=webots_port,
        prefix=webots_prefix,
    )
    supervisor_action = _require_supervisor_action(webots)

    climber_spawner = Node(
        package=package_name,
        executable='urdf_spawner',
        name='climber_spawner',
        output='screen',
        parameters=[
            {'robot_description': climber_description},
            {'robot_name': 'wall_climber'},
            {'spawn_translation': shared.initial_spawn_translation_str()},
            {'spawn_rotation': '1 0 0 1.5708'},
        ],
    )
    wall_climber_driver = WebotsController(
        robot_name='wall_climber',
        port=webots_port,
        parameters=[
            {'robot_description': climber_xacro_path},
            {'use_sim_time': True},
        ],
        respawn=True,
    )

    cable_supervisor_driver = WebotsController(
        robot_name='cable_supervisor',
        port=webots_port,
        parameters=[
            {'robot_description': supervisor_xacro_path},
            {'use_sim_time': True},
            {'enable_webots_trail': enable_webots_trail},
        ],
        respawn=True,
    )

    rosbridge = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        output='screen',
    )
    web_server = Node(
        package='wall_climber',
        executable='web_server',
        name='web_ui_server',
        output='screen',
        parameters=[
            {'port': 8080},
            {'initial_mode': ParameterValue(writer_mode, value_type=str)},
            {'enable_webots_trail': ParameterValue(enable_webots_trail, value_type=bool)},
            {'open_browser': False},
        ],
    )
    cable_draw_executor = Node(
        package='wall_climber_draw_body',
        executable='cable_draw_executor',
        name='cable_draw_executor',
        output='screen',
        parameters=[shared.cable_executor_params()],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'webots_prefix',
            default_value='',
            description='Optional command prefix for the Webots process itself.',
        ),
        DeclareLaunchArgument(
            'enable_webots_trail',
            default_value='false',
            description='Enable the optional visual-only Webots trail mesh.',
        ),
        DeclareLaunchArgument(
            'writer_mode',
            default_value='off',
            description='Initial UI mode: off | text | draw',
        ),
        SetEnvironmentVariable('ALSOFT_DRIVERS', 'null'),
        SetEnvironmentVariable('WEBOTS_TMPDIR', '/tmp'),
        SetEnvironmentVariable('TMPDIR', '/tmp'),
        webots,
        supervisor_action,
        climber_spawner,
        RegisterEventHandler(
            OnProcessExit(
                target_action=webots,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        ),
        wall_climber_driver,
        cable_supervisor_driver,
        rosbridge,
        web_server,
        cable_draw_executor,
    ])
