ACTIVE_MODE_TOPIC = '/wall_climber/internal/active_mode'
MANUAL_PEN_MODE_TOPIC = '/wall_climber/internal/manual_pen_mode'
MANUAL_GRIP_MODE_TOPIC = '/wall_climber/internal/manual_grip_mode'
DRAW_PLAN_TOPIC = '/wall_climber/draw_plan'

CABLE_SETPOINT_TOPIC = '/wall_climber/cable_setpoint'
CABLE_EXECUTOR_STATUS_TOPIC = '/wall_climber/cable_executor_status'
CABLE_SUPERVISOR_STATUS_TOPIC = '/wall_climber/cable_supervisor_status'
PEN_ATTACHED_TOPIC = '/wall_climber/pen_attached'

MODE_OFF = 'off'
MODE_TEXT = 'text'
MODE_DRAW = 'draw'
VALID_MODES = (MODE_OFF, MODE_TEXT, MODE_DRAW)

PEN_MODE_AUTO = 'auto'
PEN_MODE_UP = 'up'
PEN_MODE_DOWN = 'down'
VALID_MANUAL_PEN_MODES = (PEN_MODE_AUTO, PEN_MODE_UP, PEN_MODE_DOWN)

GRIP_MODE_AUTO = 'auto'
GRIP_MODE_CLOSED = 'closed'
GRIP_MODE_OPEN = 'open'
VALID_MANUAL_GRIP_MODES = (GRIP_MODE_AUTO, GRIP_MODE_CLOSED, GRIP_MODE_OPEN)
