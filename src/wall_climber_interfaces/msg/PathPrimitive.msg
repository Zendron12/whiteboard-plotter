# Primitive type constants. The executor interprets these as pen state, travel,
# and drawable geometry commands in board coordinates.
uint8 PEN_UP=0
uint8 PEN_DOWN=1
uint8 TRAVEL_MOVE=2
uint8 LINE_SEGMENT=3
uint8 ARC_SEGMENT=4
uint8 QUADRATIC_BEZIER=5
uint8 CUBIC_BEZIER=6

# One of the constants above.
uint8 type

# Common start/end points for travel, line, arc, and curve primitives.
wall_climber_interfaces/BoardPoint start
wall_climber_interfaces/BoardPoint end

# Bezier control points. control1 is used by quadratic and cubic curves;
# control2 is used by cubic curves.
wall_climber_interfaces/BoardPoint control1
wall_climber_interfaces/BoardPoint control2

# Arc center point and radius in meters.
wall_climber_interfaces/BoardPoint center
float64 radius

# Arc start angle and sweep angle in radians.
float64 start_angle_rad
float64 sweep_angle_rad

# True for clockwise arc execution.
bool clockwise

# True when this primitive should be drawn with the pen down.
bool pen_down
