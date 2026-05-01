# Agent Guidance

## Project Goal

This repository is a ROS 2 / Webots project for a cable-driven whiteboard drawing robot. The current target is a Webots-simulated Y-shape / two-cable wall robot for large boards, not a rail-based CNC machine or Cartesian XY plotter.

## Hard Constraints

- Do not redesign the robot as CNC rails or a Cartesian plotter.
- Preserve the existing Webots plugin-based simulation approach.
- Do not replace `wall_climber.cable_supervisor_plugin.CableSupervisorPlugin` or `wall_climber.cable_robot_plugin.CableRobotPlugin` with custom cable physics unless explicitly requested.
- Do not implement cable dynamics or four-cable kinematics unless explicitly requested.
- Treat a four-cable wall robot with top-left, top-right, bottom-left, and bottom-right anchors as future work only.
- Avoid large rewrites without tests and without preserving existing behavior.

## Preferred Responsibilities

- Python: image processing, vectorization, SVG parsing, path planning, voice command parsing, web UI/backend, metrics, and graduation-project orchestration.
- C++: performance-critical geometry, transport conversion, execution loops, and Webots/ROS integration where the project already uses it.
- Do not move image processing to C++ without measured performance evidence.

## Current Execution Shape

- The web backend emits `PrimitivePathPlan` messages on `/wall_climber/primitive_path_plan`.
- `wall_climber_draw_body/cable_draw_executor` samples and validates geometry, computes cable lengths, and publishes `/wall_climber/cable_setpoint`.
- The Webots supervisor plugin consumes cable setpoints and applies the simulated carriage pose.
- The robot plugin controls the pen slide state.

## Build And Test Commands

- Python tests:
  - `PYTHONPATH=src/wall_climber python3 -m pytest -q src/wall_climber/test`
- ROS build:
  - `source /opt/ros/humble/setup.bash && colcon build --packages-select wall_climber_interfaces wall_climber_draw_body wall_climber --cmake-args -DBUILD_TESTING=ON`
- C++/ROS tests:
  - `source /opt/ros/humble/setup.bash && colcon test --packages-select wall_climber_interfaces wall_climber_draw_body --event-handlers console_direct+`
- Package discovery:
  - `colcon list`

## Notes

- In this shell, `pytest` may not be available as a direct executable; use `python3 -m pytest`.
- `colcon build` requires the ROS Humble setup script to be sourced first.
- `colcon test --packages-select wall_climber` currently reports zero Python tests, so run the direct pytest command for Python coverage.
