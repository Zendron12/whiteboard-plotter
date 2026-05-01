import pytest

from wall_climber.shared_config import load_shared_config


def test_executor_completion_park_is_left_middle_of_board() -> None:
    config = load_shared_config()
    params = config.cable_executor_params()
    safe = config.carriage_safe_workspace_bounds()

    assert params['completion_park_x'] == pytest.approx(safe['x_min'])
    assert params['completion_park_y'] == pytest.approx(config.board.height * 0.5)
    assert safe['y_min'] <= params['completion_park_y'] <= safe['y_max']

