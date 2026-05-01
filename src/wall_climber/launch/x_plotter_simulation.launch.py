import os
import re
import socket
import shutil
import subprocess
from pathlib import Path

import yaml
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


def _require_supervisor_action(webots_launcher: WebotsLauncher):
    supervisor_action = getattr(webots_launcher, '_supervisor', None)
    if supervisor_action is None:
        raise RuntimeError(
            'WebotsLauncher no longer exposes the internal "_supervisor" action. '
            'Update x_plotter_simulation.launch.py for the current webots_ros2_driver API.'
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


def _active_x_displays() -> list[str]:
    try:
        result = subprocess.run(
            ['ps', '-eo', 'args='],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    displays: list[str] = []
    for line in result.stdout.splitlines():
        if not any(server in line for server in ('Xorg', 'Xwayland', 'Xvfb')):
            continue
        for match in re.finditer(r'(?<!\d)(:\d+)\b', line):
            display = match.group(1)
            if display not in displays:
                displays.append(display)
    return displays


def _candidate_xauthority_paths() -> list[str]:
    candidates: list[str] = []

    def add_candidate(path: str | None) -> None:
        if not path:
            return
        candidate = Path(path)
        if candidate.is_file():
            resolved = str(candidate)
            if resolved not in candidates:
                candidates.append(resolved)

    add_candidate(os.environ.get('XAUTHORITY'))

    runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
    if runtime_dir:
        for path in sorted(Path(runtime_dir).glob('xauth_*')):
            add_candidate(str(path))

    run_user_root = Path('/run/user')
    if run_user_root.is_dir():
        for path in sorted(run_user_root.glob('*/xauth_*')):
            add_candidate(str(path))

    return candidates


def _resolve_webots_display_environment() -> dict[str, str]:
    active_displays = _active_x_displays()
    if not active_displays:
        return {}

    current_display = os.environ.get('DISPLAY')
    if current_display in active_displays:
        selected_display = current_display
    elif ':0' in active_displays:
        selected_display = ':0'
    else:
        selected_display = active_displays[0]

    environment = {'DISPLAY': selected_display}
    for path in _candidate_xauthority_paths():
        environment['XAUTHORITY'] = path
        break

    return environment


def _load_x_plotter_config(pkg_dir: str) -> dict:
    config_path = Path(pkg_dir) / 'config' / 'x_plotter.yaml'
    with config_path.open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f'{config_path} must contain a YAML object.')
    return data


def _spawn_translation_from_config(config: dict) -> str:
    board = config['board']
    carriage = config['carriage']
    board_left = float(board['center_x']) - (float(board['width']) * 0.5)
    board_top_z = float(board['center_z']) + (float(board['height']) * 0.5)
    return f'{board_left:.4f} {float(carriage["plane_y"]):.4f} {board_top_z:.4f}'


def generate_launch_description():
    package_name = 'wall_climber'
    pkg_dir = get_package_share_directory(package_name)
    config = _load_x_plotter_config(pkg_dir)

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
            f'[x_plotter.launch] Requested Webots port {requested_webots_port} is busy; '
            f'using {webots_port} instead.'
        )

    display_environment = _resolve_webots_display_environment()
    selected_display = display_environment.get('DISPLAY')
    current_display = os.environ.get('DISPLAY')
    if selected_display and selected_display != current_display:
        print(
            f'[x_plotter.launch] DISPLAY {current_display!r} is not active; '
            f'using {selected_display!r} for Webots.'
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
    demo_path = LaunchConfiguration('demo_path')

    world_path = os.path.join(pkg_dir, 'worlds', 'x_plotter_world.wbt')
    xacro_path = os.path.join(pkg_dir, 'urdf', 'x_plotter_robot.urdf.xacro')
    x_plotter_description = ParameterValue(
        Command([
            'xacro ',
            xacro_path,
            ' demo_path:=',
            demo_path,
            ' enable_webots_trail:=',
            enable_webots_trail,
        ]),
        value_type=str,
    )

    webots = WebotsLauncher(
        world=world_path,
        mode='realtime',
        ros2_supervisor=True,
        port=webots_port,
        prefix=webots_prefix,
    )
    supervisor_action = _require_supervisor_action(webots)

    x_plotter_spawner = Node(
        package=package_name,
        executable='urdf_spawner',
        name='x_plotter_spawner',
        output='screen',
        parameters=[
            {'robot_description': x_plotter_description},
            {'robot_name': 'x_plotter'},
            {'spawn_translation': _spawn_translation_from_config(config)},
            {'spawn_rotation': '1 0 0 1.5708'},
        ],
    )

    x_plotter_driver = WebotsController(
        robot_name='x_plotter',
        port=webots_port,
        parameters=[
            {'robot_description': x_plotter_description},
            {'use_sim_time': True},
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

    return LaunchDescription([
        DeclareLaunchArgument(
            'webots_prefix',
            default_value='',
            description='Optional command prefix for the Webots process itself.',
        ),
        DeclareLaunchArgument(
            'demo_path',
            default_value='line_square_triangle',
            description='Built-in X plotter demo: off | line | square | triangle | line_square_triangle',
        ),
        DeclareLaunchArgument(
            'enable_webots_trail',
            default_value='true',
            description='Enable optional service-spawned drawing trail segments for the X plotter.',
        ),
        DeclareLaunchArgument(
            'writer_mode',
            default_value='draw',
            description='Initial UI mode: off | text | draw',
        ),
        SetEnvironmentVariable('ALSOFT_DRIVERS', 'null'),
        SetEnvironmentVariable('WEBOTS_TMPDIR', '/tmp'),
        SetEnvironmentVariable('TMPDIR', '/tmp'),
        *[
            SetEnvironmentVariable(name, value)
            for name, value in display_environment.items()
        ],
        webots,
        supervisor_action,
        x_plotter_spawner,
        RegisterEventHandler(
            OnProcessExit(
                target_action=webots,
                on_exit=[EmitEvent(event=Shutdown())],
            )
        ),
        x_plotter_driver,
        rosbridge,
        web_server,
    ])

