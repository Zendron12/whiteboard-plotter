# Carriage pose in board coordinates. theta is currently the reference tool angle.
geometry_msgs/Pose2D carriage_pose

# Commanded left and right cable lengths in meters.
float64 left_cable_length
float64 right_cable_length

# True while the pen should touch/draw on the board.
bool pen_down

# Index of the active executable segment, or -1 when idle/parking.
int32 active_segment_index

# Normalized progress through the active execution plan, in [0, 1] when available.
float64 progress
