# X/CoreXY Plotter Foundation

## Purpose

The existing Y/cable robot remains unchanged and usable as the legacy implementation. Its main files are:

- `src/wall_climber/launch/my_robot.launch.py`
- `src/wall_climber/config/cable_robot.yaml`
- `src/wall_climber/urdf/my_robot.urdf.xacro`
- `src/wall_climber/urdf/cable_supervisor.urdf.xacro`
- `src/wall_climber/wall_climber/cable_robot_plugin.py`
- `src/wall_climber/wall_climber/cable_supervisor_plugin.py`
- `src/wall_climber_draw_body/src/cable_draw_executor.cpp`

The new X/CoreXY path is added beside it as the main graduation-demo direction. It starts with a clean rectangular frame, crossed-belt visual mechanism, moving carriage, and pen up/down control without replacing the legacy cable path.

The new X/CoreXY files are:

- `src/wall_climber/launch/x_plotter_simulation.launch.py`
- `src/wall_climber/config/x_plotter.yaml`
- `src/wall_climber/urdf/x_plotter_robot.urdf.xacro`
- `src/wall_climber/worlds/x_plotter_world.wbt`
- `src/wall_climber/wall_climber/x_plotter/`

## Docs Policy

Installable ROS package docs live in `src/wall_climber/docs` and are installed to `share/wall_climber/docs` by `colcon build`. The root `docs/README.md` file is repository navigation only and links back to these package docs.

## Architecture

The X plotter foundation uses temporary primitive-direct execution:

- subscribes to `/wall_climber/primitive_path_plan`
- samples `PrimitivePathPlan` in `wall_climber/x_plotter/primitive_sampler.py`
- drives the X/Y carriage directly in the Webots Python plugin
- publishes the same board/pose/pen topics used by the current web UI

This keeps the existing drawing transport working while avoiding changes to `vector_pipeline.py`, the hybrid image pipeline, or the C++ cable executor.

## Coordinate Frame

Board coordinates are explicit in `src/wall_climber/config/x_plotter.yaml`:

- origin: top-left of the whiteboard
- +X: right
- +Y: down
- width: `6.3 m`
- height: `3.0 m`

The Webots mapping is:

```text
world_x = board_left + board_x
world_y = carriage_plane_y
world_z = board_top_z - board_y
```

The CoreXY helper is available for future executor work:

```text
a = x + y
b = x - y
x = (a + b) / 2
y = (a - b) / 2
```

## Topics

The X plotter subscribes to:

- `/wall_climber/primitive_path_plan`
- `/wall_climber/internal/manual_pen_mode`

It publishes:

- `/wall_climber/board_info`
- `/wall_climber/robot_pose_board`
- `/wall_climber/pen_pose_board`
- `/wall_climber/pen_contact`
- `/wall_climber/pen_gap`
- `/wall_climber/x_plotter_status`

For current web backend compatibility only, it also publishes simple status strings on:

- `/wall_climber/cable_executor_status`
- `/wall_climber/cable_supervisor_status`

Those compatibility topics are not cable diagnostics and do not contain cable-specific data.

## Run

Build and source the workspace:

```bash
colcon build --packages-select wall_climber_interfaces wall_climber_draw_body wall_climber
source install/setup.bash
```

Run the X/CoreXY simulation:

```bash
ros2 launch wall_climber x_plotter_simulation.launch.py
```

Disable the built-in demo:

```bash
ros2 launch wall_climber x_plotter_simulation.launch.py demo_path:=off
```

Available demo paths:

- `line`
- `square`
- `triangle`
- `line_square_triangle`
- `off`

The legacy cable/Y robot still runs through:

```bash
ros2 launch wall_climber my_robot.launch.py
```

## Known Limitations

- The crossed belts are visual-only and are not a physical belt simulation.
- The X plotter executes sampled X/Y points directly in Python for this foundation.
- There is no dedicated CoreXY motor executor yet.
- The current `PrimitivePathPlan` transport is preserved temporarily.
- A later task should introduce a unified `DrawingPathPlan`, refactor `vector_pipeline.py`, and then connect the hybrid image pipeline, voice commands, image library, and any dedicated CoreXY executor.
