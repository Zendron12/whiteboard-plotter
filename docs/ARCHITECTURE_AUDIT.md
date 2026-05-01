# Architecture Audit

## Current Project Structure

This repository is a ROS 2 workspace with three active packages:

- `src/wall_climber`: Python application package for Webots launch, Webots plugins, shared configuration, FastAPI backend, browser UI assets, text/SVG/image ingestion, canonical path planning, and Webots URDF/xacro assets.
- `src/wall_climber_interfaces`: ROS 2 message package for `BoardPoint`, `PathPrimitive`, `PrimitivePathPlan`, and `CableSetpoint`.
- `src/wall_climber_draw_body`: C++ package for geometry sampling/evaluation, ROS transport conversion, Python geometry bindings, and the `cable_draw_executor` runtime node.

Existing docs describe the canonical path and primitive transport migration:

- `docs/canonical-first-ingestion.md`
- `docs/legacy-compatibility.md`
- `docs/primitive-transport.md`

## Webots World, Plugin, And Controller Flow

The simulation is a Webots cable-driven wall robot. It is not a CNC rail system.

Primary launch entrypoint:

- `src/wall_climber/launch/my_robot.launch.py`

Launch starts:

- Webots with `worlds/wall_world.wbt`.
- The Webots ROS supervisor.
- `urdf_spawner`, which spawns `wall_climber` from `urdf/my_robot.urdf.xacro`.
- A Webots controller for `wall_climber`.
- A Webots controller for `cable_supervisor`.
- `rosbridge_server`.
- `wall_climber/web_server`.
- `wall_climber_draw_body/cable_draw_executor`.

World files:

- `src/wall_climber/worlds/wall_world.wbt`: break-room world that preserves the original whiteboard coordinates and includes the whiteboard, existing top anchor mounts, and an external `cable_supervisor` robot.
- `src/wall_climber/worlds/wall_world_basic.wbt`: simpler lab world with the same board concept.

Robot xacro files:

- `urdf/my_robot.urdf.xacro`: declares the `wall_climber` robot and attaches `CableRobotPlugin`.
- `urdf/cable_supervisor.urdf.xacro`: declares the supervisor plugin properties, board dimensions, anchors, carriage geometry, pen properties, safety bounds, cable visuals, and optional Webots trail settings.
- `urdf/base_body.xacro`, `urdf/simple_pen_mount.xacro`, and `urdf/robot_face_screen.xacro`: visual and mechanical structure for the robot body, pen mount, and display.

## How The Cable Robot Moves In Webots

The runtime path is:

1. Browser/API input creates text, SVG, or image drawing requests.
2. Python builds a `CanonicalPathPlan`.
3. The backend exports a `PrimitivePathPlan` and publishes `/wall_climber/primitive_path_plan`.
4. `wall_climber_draw_body/cable_draw_executor` samples the primitives, validates board and cable safety bounds, converts pen points to carriage center and legacy top-cable lengths, then publishes `/wall_climber/cable_setpoint`.
5. `CableSupervisorPlugin` receives cable setpoints, uses `CableSetpoint.carriage_pose.x/y` as the board-space carriage center, computes four kinematic cable lengths, and applies the resulting pose to the Webots robot.
6. `CableRobotPlugin` receives the same setpoint and controls the pen slide up/down state.

The supervisor plugin intentionally drives the simulation through a plugin-level cable model and pose application. This avoids implementing full cable physics or dynamics in this task.

## What The Webots Plugins Do

`src/wall_climber/wall_climber/cable_supervisor_plugin.py`:

- Reads board, anchor, carriage, pen, workspace, and visual settings from xacro properties generated from `config/cable_robot.yaml`.
- Publishes board metadata, robot pose, pen pose, pen contact, pen gap, and supervisor status.
- Subscribes to `/wall_climber/cable_setpoint` and manual pen mode.
- Computes four kinematic cable lengths from `CableSetpoint.carriage_pose`, four board anchors, and four carriage attachment offsets.
- Applies the target robot translation and rotation in Webots.
- Creates visual cable lines and optional trail mesh.
- Tracks pen contact using the pen tip geometry or fallback geometry.

`src/wall_climber/wall_climber/cable_robot_plugin.py`:

- Subscribes to `/wall_climber/cable_setpoint` and manual pen mode.
- Moves the `pen_slide_joint` motor to configured up/down positions.
- Leaves carriage motion to the supervisor plugin.

## Existing Robot Motion And Execution Flow

`src/wall_climber_draw_body/src/cable_draw_executor.cpp`:

- Subscribes to `/wall_climber/primitive_path_plan`.
- Converts ROS primitives to C++ geometry commands.
- Samples lines, arcs, quadratic Beziers, and cubic Beziers.
- Builds travel and draw paths.
- Validates writable board bounds, carriage-safe bounds, safe cable workspace, and anchor keepout.
- Computes legacy left and right top-cable lengths from carriage attachment points while also publishing `carriage_pose` as the carriage center.
- Publishes `/wall_climber/cable_setpoint`.
- Publishes executor status and JSON diagnostics.

`src/wall_climber_draw_body/src/geometry_sampling.cpp` and `geometry_eval.cpp`:

- Provide deterministic geometry evaluation, curve sampling, and polyline cleanup.

`src/wall_climber_draw_body/src/transport_conversions.cpp`:

- Converts `PrimitivePathPlan` ROS messages to internal C++ `PathPlan` geometry.

## Existing Image, Vector, And Path-Generation Files

Core canonical model:

- `wall_climber/canonical_path.py`: Python `CanonicalPathPlan`, point aliases, and primitive command dataclasses.
- `wall_climber/canonical_builders.py`: converts draw strokes and grouped text glyphs into canonical plans.
- `wall_climber/canonical_adapters.py`: sampling, diagnostics, preview/export conversion, and `PrimitivePathPlan` descriptors.
- `wall_climber/canonical_optimizer.py`: travel reduction, tiny primitive pruning, duplicate removal, curve merging, arc fitting, and hatch ordering.
- `wall_climber/canonical_ops.py`: compatibility exports for placement and cleanup helpers.

Ingestion and vectorization:

- `wall_climber/vector_pipeline.py`: compatibility facade and much of the existing text/SVG/image vector logic.
- `wall_climber/ingestion/text.py`: text vectorization facade.
- `wall_climber/ingestion/svg.py`: SVG vectorization facade.
- `wall_climber/ingestion/image.py`: image vectorization facade.
- `wall_climber/ingestion/image_curve_fitting.py`: image contour tracing, centerline behavior, curve fitting, route debug output, and metadata mapping.
- `wall_climber/ingestion/upload_routing.py`: upload classification for SVG versus raster image input.
- `wall_climber/image_routing.py`: image route compatibility exports.

The project already has a canonical path model. The new `DrawingPathPlan` added in this task is only a future-facing image-pipeline interface and must not replace or wire into the current runtime yet.

## Existing Web UI And Backend

Backend:

- `wall_climber/web_server.py`: FastAPI server and ROS node. It serves the web UI, exposes health/runtime/debug endpoints, handles text/SVG/image preview and commit endpoints, stores uploads under `~/.ros/wall_climber/uploads`, and publishes `PrimitivePathPlan` for execution.

Frontend:

- `web/index.html`: browser UI for board visualization, mode switching, manual pen controls, text, SVG, upload preview/commit, runtime diagnostics, curve-fit overlay, rosbridge telemetry, robot pose, pen pose, pen contact, and trail visualization.
- `web/vendor/roslib.min.js`: browser ROS bridge client library.

## Existing Python Responsibilities

- Webots plugins and URDF spawning.
- Shared YAML configuration loading and derived board/safety bounds.
- FastAPI backend and browser API contracts.
- Text, SVG, and image ingestion.
- Canonical plan construction, placement, preview, diagnostics, and optimization.
- Upload routing and background image preparation.
- Runtime topic constants.

## Existing C++ Responsibilities

- Geometry evaluation and sampling.
- C++ geometry Python bindings for faster sampling when available.
- Primitive transport conversion.
- Execution scheduling, workspace validation, cable-length computation, status publication, and setpoint publication.

## Files That Should Not Be Modified Casually

- `src/wall_climber/worlds/*.wbt`: board, anchor, and supervisor setup must remain compatible with the launch and plugin assumptions.
- `src/wall_climber/urdf/*.xacro`: plugin wiring, board dimensions, anchors, carriage geometry, and pen geometry are coupled to the Webots plugins.
- `src/wall_climber/config/cable_robot.yaml`: single shared source for board, anchor, carriage, pen, execution, and visual defaults.
- `src/wall_climber/launch/my_robot.launch.py`: orchestrates Webots, controllers, rosbridge, web backend, spawner, and executor.
- `src/wall_climber/wall_climber/cable_supervisor_plugin.py`: central plugin-based cable behavior.
- `src/wall_climber/wall_climber/cable_robot_plugin.py`: pen plugin behavior.
- `src/wall_climber_interfaces/msg/*.msg`: runtime transport contracts.
- `src/wall_climber_draw_body/src/cable_draw_executor.cpp`: production execution and cable-length output.
- `src/wall_climber_draw_body/src/geometry_*` and `transport_conversions.cpp`: shared geometry and primitive transport behavior.
- `src/wall_climber/wall_climber/web_server.py`: live API and ROS publishing flow.

## Risks In The Current Architecture

- `my_robot.launch.py` depends on a private `WebotsLauncher._supervisor` action; a future `webots_ros2_driver` API change could break launch.
- Geometry configuration is shared across YAML, xacro, Python, and C++; changes require careful test coverage.
- The supervisor plugin directly applies Webots pose from cable setpoints. This is intentional, but it means simulation behavior depends heavily on plugin correctness.
- Python package tests are not discovered by `colcon test --packages-select wall_climber`; direct pytest is required.
- Running `colcon build` without sourcing ROS fails because `ament_cmake` is not on `CMAKE_PREFIX_PATH`.
- `wall_climber/setup.py` emits a setuptools warning for `tests_require`.
- The current image/vector code is powerful but broad; large refactors could break preview/runtime parity.

## Build And Test Commands Discovered

Useful commands:

- `colcon list`
- `colcon graph`
- `PYTHONPATH=src/wall_climber python3 -m pytest -q src/wall_climber/test`
- `source /opt/ros/humble/setup.bash && colcon build --packages-select wall_climber_interfaces wall_climber_draw_body wall_climber --cmake-args -DBUILD_TESTING=ON`
- `source /opt/ros/humble/setup.bash && colcon test --packages-select wall_climber_interfaces wall_climber_draw_body --event-handlers console_direct+`

Observed results during this audit:

- `PYTHONPATH=src/wall_climber python3 -m pytest -q src/wall_climber/test`: 55 passed.
- `colcon build --packages-select wall_climber_interfaces wall_climber_draw_body --cmake-args -DBUILD_TESTING=ON`: failed until ROS Humble was sourced; failure was missing `ament_cmake`.
- `source /opt/ros/humble/setup.bash && colcon build --packages-select wall_climber_interfaces wall_climber_draw_body --cmake-args -DBUILD_TESTING=ON`: passed.
- `source /opt/ros/humble/setup.bash && colcon test --packages-select wall_climber_interfaces wall_climber_draw_body --event-handlers console_direct+`: passed, including `test_geometry_kernel` and `test_transport_conversions`.
- `source /opt/ros/humble/setup.bash && colcon build --packages-select wall_climber --cmake-args -DBUILD_TESTING=ON`: passed with a `tests_require` warning.
- `source /opt/ros/humble/setup.bash && colcon test --packages-select wall_climber --event-handlers console_direct+`: ran 0 tests.

## Recommended Next Steps

1. Keep the Webots plugin-based four-cable kinematic simulation covered by tests before changing launch, xacro, or plugin behavior again.
2. Add a future adapter from the new `DrawingPathPlan` DTO to the existing `CanonicalPathPlan`, but do not wire it into production until tests cover parity.
3. Add focused tests for any new image pipeline module before replacing existing vectorization behavior.
4. Add voice command parsing as a separate Python layer that produces the same neutral drawing plan format.
5. Add a small numbered image library service around the manifest after the manifest contract is reviewed.
6. Consider fixing Python test discovery in colcon and removing the `tests_require` warning.
7. Treat any move beyond kinematic four-cable behavior, such as tension or slack modeling, as a separate future design task with its own tests and simulator plan.
