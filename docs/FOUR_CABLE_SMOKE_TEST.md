# Four-Cable Webots Smoke Test

Date: 2026-05-01

Branch tested: `four-cable-kinematic-plugin`

Commit tested: `915c870`

## Launch Command

The existing launch file was used directly:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch wall_climber my_robot.launch.py enable_webots_trail:=false writer_mode:=off 2>&1 | tee /tmp/wall_climber_smoke/webots_launch_retry.log
```

The launch used the existing `wall_world.wbt` through `my_robot.launch.py`. No new Webots world, launch file, plugin, robot, ROS message, or runtime path was created.

## Startup Result

Startup succeeded.

Observed log evidence:

- `Cable draw executor ready.`
- `Controller successfully connected to robot in Webots simulation.`
- `Cable robot plugin ready.`
- `Found target robot "wall_climber".`
- `Resolved pen tip sphere...`

Required topics were present:

- `/wall_climber/board_info`
- `/wall_climber/cable_supervisor_status`
- `/wall_climber/robot_pose_board`
- `/wall_climber/pen_pose_board`

The backend health endpoint returned ready with both executor and supervisor statuses observed.

## Diagnostics Before Movement

Command:

```bash
curl -s http://127.0.0.1:8080/api/runtime > /tmp/wall_climber_smoke/runtime_retry_before.json
```

Observed:

- `ready: true`
- `active_mode: off`
- `cable_executor_status: idle`
- `cable_supervisor_status: waiting_for_setpoint`
- legacy `board_info.anchors.left/right` present
- `board_info.four_cable_kinematics` present
- four anchors present: `top_left`, `top_right`, `bottom_left`, `bottom_right`
- four attachments present: `top_left`, `top_right`, `bottom_left`, `bottom_right`
- initial four lengths present and positive:
  - `top_left`: `3.169186173136567`
  - `top_right`: `3.169186173136567`
  - `bottom_left`: `3.6302535724106106`
  - `bottom_right`: `3.6302535724106106`

## Movement Command

The first `/api/text` request was rejected while the runtime mode was `off`, which is the existing backend behavior.

Mode was switched using the existing backend endpoint:

```bash
curl -s -X POST http://127.0.0.1:8080/api/mode \
  -H 'Content-Type: application/json' \
  -d '{"mode":"text"}'
```

Then a small safe text command was sent through the existing runtime flow:

```bash
curl -s -X POST http://127.0.0.1:8080/api/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"HI","font_source":"relief_singleline"}'
```

Observed response:

- `ok: true`
- `published: true`
- transport: `primitive_path_plan`
- preview `stroke_count: 5`
- preview `point_count: 60`
- preview `can_commit: true`

## Movement Result

After the tolerance fix described below, the final runtime state was:

- `cable_executor_status: done`
- `cable_supervisor_status: tracking`
- final setpoint:
  - `carriage_pose.x: 0.14499999999999996`
  - `carriage_pose.y: 2.8000000000000003`
  - `pen_down: false`
  - `progress: 1.0`
- final robot pose matched the final carriage pose
- final pen contact was `false`
- final four lengths were present and positive:
  - `top_left`: `2.7253084229129003`
  - `top_right`: `6.636281036845863`
  - `bottom_left`: `0.13155227098001723`
  - `bottom_right`: `6.052290971194297`

## Tiny Fix Applied

The first movement attempt exposed a boundary-roundoff bug in `CableSupervisorPlugin._pose_within_safe_workspace`.

The executor produced a valid final park setpoint at the configured lower-left safety boundary:

```text
carriage_pose.x: 0.14499999999999996
carriage_pose.y: 2.8000000000000003
pen_down: false
progress: 1.0
```

This was rejected because the supervisor compared exact floating-point values against the carriage center minimum of `0.145` and safe pen maximum of `2.82`. A minimal `1.0e-9` comparison tolerance was added to the supervisor safety check. No ROS messages, C++ executor code, image processing code, worlds, launch files, duplicate plugins, or duplicate runtimes were changed.

## Visual Evidence

The existing Webots launch starts Webots in batch mode:

```text
/usr/local/webots/webots --port=1234 ... --batch --mode=realtime
```

Because of that, live interactive visual inspection through X11 was not available in this environment. X11 screenshots taken from displays `:0` and `:4` were black.

An existing Webots animation recording service was used to capture a reviewable artifact:

```bash
ros2 service call /Ros2Supervisor/animation_start_recording \
  webots_ros2_msgs/srv/SetString \
  "{value: '/tmp/wall_climber_smoke/four_cable_smoke.html'}"

curl -s -X POST http://127.0.0.1:8080/api/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"HI","font_source":"relief_singleline"}'

ros2 service call /Ros2Supervisor/animation_stop_recording \
  webots_ros2_msgs/srv/GetBool \
  "{ask: false}"
```

Generated files:

- `/tmp/wall_climber_smoke/four_cable_smoke.html`
- `/tmp/wall_climber_smoke/four_cable_smoke.w3d`
- `/tmp/wall_climber_smoke/four_cable_smoke.json`
- `/tmp/wall_climber_smoke/four_cable_smoke.css`

The recorded `.w3d` contains the runtime `CABLE_LINES` `IndexedLineSet` with 16 coordinates and 8 line segments, representing four cables as two visual line segments per cable. The exported cable endpoints include top anchor Z coordinates near `3.30` and bottom anchor Z coordinates near `0.30`, matching the configured top and bottom board anchors.

## Risks Remaining

- A human should review `/tmp/wall_climber_smoke/four_cable_smoke.html` in a browser with Webots web-scene support to confirm the cable appearance frame by frame.
- The smoke test verified runtime movement, diagnostics, setpoints, pen state, and exported cable geometry, but the batch-mode launch prevented direct live visual inspection from this shell.
- The Webots launch still depends on the existing `webots_ros2_driver` behavior that starts Webots in batch mode.
