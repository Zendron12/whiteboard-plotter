from __future__ import annotations

import builtins
import importlib
import math

import pytest

from wall_climber.four_cable_kinematics import FOUR_CABLE_NAMES, compute_four_cable_lengths
from wall_climber.shared_config import load_shared_config


def _config():
    load_shared_config.cache_clear()
    return load_shared_config()


def test_four_cable_config_fields_load() -> None:
    config = _config()

    assert config.four_cable_anchors() == {
        'top_left': (0.0, 0.0),
        'top_right': (config.board.width, 0.0),
        'bottom_left': (0.0, config.board.height),
        'bottom_right': (config.board.width, config.board.height),
    }
    assert config.four_cable_attachments() == {
        'top_left': (-0.104, -0.075),
        'top_right': (0.104, -0.075),
        'bottom_left': (-0.104, 0.075),
        'bottom_right': (0.104, 0.075),
    }


def test_legacy_two_cable_alias_fields_still_load() -> None:
    config = _config()

    assert config.anchors.left_x == config.anchors.top_left_x
    assert config.anchors.left_y == config.anchors.top_left_y
    assert config.anchors.right_x == config.anchors.top_right_x
    assert config.anchors.right_y == config.anchors.top_right_y
    assert config.carriage.attachment_left_x == config.carriage.attachment_top_left_x
    assert config.carriage.attachment_left_y == config.carriage.attachment_top_left_y
    assert config.carriage.attachment_right_x == config.carriage.attachment_top_right_x
    assert config.carriage.attachment_right_y == config.carriage.attachment_top_right_y


def test_four_cable_lengths_are_positive_and_finite() -> None:
    config = _config()

    lengths = compute_four_cable_lengths(
        (config.board.width * 0.5, config.board.height * 0.5),
        config.four_cable_anchors(),
        config.four_cable_attachments(),
    )

    assert tuple(lengths.keys()) == FOUR_CABLE_NAMES
    assert all(math.isfinite(length) and length > 0.0 for length in lengths.values())


def test_symmetric_center_pose_produces_symmetric_lengths() -> None:
    config = _config()

    lengths = compute_four_cable_lengths(
        (config.board.width * 0.5, config.board.height * 0.5),
        config.four_cable_anchors(),
        config.four_cable_attachments(),
    )

    assert lengths['top_left'] == pytest.approx(lengths['top_right'])
    assert lengths['top_left'] == pytest.approx(lengths['bottom_left'])
    assert lengths['top_left'] == pytest.approx(lengths['bottom_right'])


def test_nan_pose_is_rejected() -> None:
    config = _config()

    with pytest.raises(ValueError, match='carriage center'):
        compute_four_cable_lengths(
            (math.nan, 1.0),
            config.four_cable_anchors(),
            config.four_cable_attachments(),
        )


def test_degenerate_zero_length_cable_is_rejected() -> None:
    config = _config()
    anchors = dict(config.four_cable_anchors())
    attachments = dict(config.four_cable_attachments())
    anchors['top_left'] = (1.0, 1.0)
    attachments['top_left'] = (0.0, 0.0)

    with pytest.raises(ValueError, match='top_left'):
        compute_four_cable_lengths((1.0, 1.0), anchors, attachments)


def test_four_cable_helper_import_does_not_require_ros_or_webots_runtime(monkeypatch) -> None:
    real_import = builtins.__import__
    forbidden_roots = {
        'controller',
        'rclpy',
        'webots_ros2_driver',
        'webots_ros2_msgs',
    }

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split('.')[0] in forbidden_roots:
            raise AssertionError(f'four-cable helper imported runtime dependency {name!r}')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    module = importlib.reload(importlib.import_module('wall_climber.four_cable_kinematics'))

    assert module.compute_four_cable_lengths is not None
