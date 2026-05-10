# Coordinate frame for all primitives. The current executor expects "board".
string frame

# Reference tool angle used by downstream geometry/executor code.
float64 theta_ref

# Ordered primitive stream to execute. Draw endpoints publish the exact cached
# primitive payload generated during preview.
wall_climber_interfaces/PathPrimitive[] primitives
