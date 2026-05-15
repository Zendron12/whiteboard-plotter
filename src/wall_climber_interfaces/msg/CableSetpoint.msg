# Carriage pose in board coordinates. theta is currently the reference tool angle.
geometry_msgs/Pose2D carriage_pose

# Commanded cable lengths in meters, ordered as
# [top_left, top_right, bottom_left, bottom_right]. See
# wall_climber.four_cable_kinematics.FOUR_CABLE_NAMES for the canonical order.
float64[4] cable_lengths

# True while the pen should touch/draw on the board.
bool pen_down

# Index of the active executable segment, or -1 when idle/parking.
int32 active_segment_index

# Normalized progress through the active execution plan, in [0, 1] when available.
float64 progress
