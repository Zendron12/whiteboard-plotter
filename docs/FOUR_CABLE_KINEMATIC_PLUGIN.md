# Four-Cable Kinematic Plugin

## Current Flow

The existing Webots supervisor flow remains integrated into the same project structure:

- `cable_draw_executor.cpp` publishes `CableSetpoint` messages.
- `CableSetpoint.carriage_pose.x/y` are the carriage center in board coordinates. This was verified in `make_setpoint()`: the executor computes `carriage_center = pen_point - pen_offset`, assigns it to `carriage_pose.x/y`, and separately fills legacy `left_cable_length` and `right_cable_length`.
- `CableSupervisorPlugin` receives the setpoint and applies the Webots carriage pose.
- `cable_robot.yaml` remains the shared source for board, anchor, carriage, pen, execution, and visual defaults.
- `cable_supervisor.urdf.xacro` passes the shared config into the existing supervisor plugin.

## Model

This is a four-cable kinematic model, not a dynamic cable-physics simulation.

The model uses:

- board anchors: `top_left`, `top_right`, `bottom_left`, `bottom_right`
- carriage attachment offsets: `top_left`, `top_right`, `bottom_left`, `bottom_right`
- carriage center from `CableSetpoint.carriage_pose`

For each cable:

```text
attachment_board = carriage_center + attachment_offset
length = hypot(anchor.x - attachment_board.x, anchor.y - attachment_board.y)
```

The plugin validates that computed lengths are finite and positive, applies the carriage pose through the existing Webots pose path, and updates four visual cable lines from the four anchors to the four kinematic attachment points.

## Diagnostics

Existing board-info fields are preserved for UI/backend compatibility, including the legacy `anchors.left` and `anchors.right` fields.

Four-cable diagnostics are added under `four_cable_kinematics`:

- model name
- pose source
- four anchors
- four attachment offsets
- latest computed cable lengths when available

## Configuration Compatibility

The new four-cable fields are present in `cable_robot.yaml`.

Legacy aliases are intentionally retained:

- `anchors.left_x`, `anchors.left_y`
- `anchors.right_x`, `anchors.right_y`
- `carriage.attachment_left_x`, `carriage.attachment_left_y`
- `carriage.attachment_right_x`, `carriage.attachment_right_y`

Those aliases remain compatible with the current C++ executor and map to the top-left/top-right cable pair.

## Limits

This model does not implement:

- cable tension control
- cable elasticity
- slack dynamics
- rope collision
- physical rope segments

That is intentional. The supervisor computes real four-cable kinematic lengths from the carriage pose and displays the four cables, while avoiding expensive full cable dynamics. This is appropriate for a graduation-project Webots simulator focused on drawing flow, pose execution, and architecture clarity.
