import os
import re
import socket
import shutil
import subprocess
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


def _cleanup_stale_launch_processes() -> None:
    """Kill any leftover ROS / Webots / web_server processes from a previous
    launch that did not exit cleanly.

    A crashed web_server or a tab that kept its WebSocket alive past the
    launcher's grace period can leave the listening socket in TIME_WAIT,
    which forces the next launch to fall back to port 8081 / 8082 and
    breaks the VS Code Ports panel auto-forward (we only forward 8080 and
    9090). Clearing those stragglers up front makes "ros2 launch" feel
    deterministic again.

    This is a best-effort cleanup; failures are silently ignored so a
    fresh first launch still works. We deliberately do NOT match the
    word "ros2 launch wall_climber" itself because that would kill the
    invocation that just started.
    """
    patterns = (
        # Long-lived sub-processes started by my_robot.launch.py
        'wall_climber/web_server',
        'rosbridge_websocket',
        'webots-controller',
        'cable_draw_executor',
        'ros2_supervisor',
        # Webots renderer + binary (only if a previous launch left them)
        '/.ros/webotsR2025a/webots/bin/webots',
    )
    for pattern in patterns:
        try:
            subprocess.run(
                ['pkill', '-9', '-f', pattern],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # pkill missing in some minimal images; nothing to do.
            pass


def generate_launch_description():
    _cleanup_stale_launch_processes()
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
    display_environment = _resolve_webots_display_environment()
    webots_port = str(selected_webots_port)
    if webots_port != requested_webots_port:
        print(
            f'[wall_climber.launch] Requested Webots port {requested_webots_port} is busy; '
            f'using {webots_port} instead.'
        )
    selected_display = display_environment.get('DISPLAY')
    current_display = os.environ.get('DISPLAY')
    if selected_display and selected_display != current_display:
        print(
            f'[wall_climber.launch] DISPLAY {current_display!r} is not active; '
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

    world_path = os.path.join(pkg_dir, 'worlds', 'wall_world_basic.wbt')
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
        # Explicitly set the parameters that rosbridge warns about under
        # Humble: it tells us "the defaults will change in Jazzy". Setting
        # them explicitly here picks the future-default behaviour now and
        # silences the warnings.
        parameters=[{
            'default_call_service_timeout': 5.0,
            'call_services_in_new_thread': True,
            'send_action_goals_in_new_thread': True,
        }],
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
        # Tell FastDDS to skip the shared-memory transport. /dev/shm is
        # restricted in the dev container, so SHM allocation always fails
        # and floods the logs with "Failed to create segment" errors
        # before silently falling back to UDP. Pointing at our XML config
        # makes UDP the only transport and removes the noise.
        SetEnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE',
            os.path.join(pkg_dir, 'config', 'fastdds_no_shm.xml'),
        ),
        *[
            SetEnvironmentVariable(name, value)
            for name, value in display_environment.items()
        ],
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
