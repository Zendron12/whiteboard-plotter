"""Webots scene-tree navigation helpers used by the supervisor plugin.

These functions walk the supervisor's node tree looking for joint endpoints,
named descendants or simple field values. They do not know anything about the
project; only that they are given a Webots Node-like object with the usual
``getField`` accessors.

Extracted from ``cable_supervisor_plugin.py`` so the plugin can stay focused
on robot behaviour. Every function returns ``None`` instead of raising when a
field or child is missing; this preserves the original tolerant behaviour of
the inline helpers.
"""

from __future__ import annotations

from typing import Any

_MAX_JOINT_DEPTH = 40
_MAX_NAME_DEPTH = 50


def find_joint_endpoint(node: Any, motor_name: str, depth: int = 0) -> Any:
    """Return the endPoint sub-node of a joint whose device has ``motor_name``.

    Walks children/endPoint recursively up to :data:`_MAX_JOINT_DEPTH`.
    """
    if node is None or depth > _MAX_JOINT_DEPTH:
        return None
    try:
        device_field = node.getField('device')
        if device_field is not None:
            for index in range(device_field.getCount()):
                device = device_field.getMFNode(index)
                if device is None:
                    continue
                name_field = device.getField('name')
                if (
                    name_field is not None
                    and name_field.getSFString() == motor_name
                ):
                    endpoint_field = node.getField('endPoint')
                    if endpoint_field is not None:
                        return endpoint_field.getSFNode()
    except Exception:
        pass
    try:
        children_field = node.getField('children')
        if children_field is not None:
            for index in range(children_field.getCount()):
                child = children_field.getMFNode(index)
                result = find_joint_endpoint(child, motor_name, depth + 1)
                if result is not None:
                    return result
    except Exception:
        pass
    try:
        endpoint_field = node.getField('endPoint')
        if endpoint_field is not None:
            endpoint = endpoint_field.getSFNode()
            result = find_joint_endpoint(endpoint, motor_name, depth + 1)
            if result is not None:
                return result
    except Exception:
        pass
    return None


def find_named_descendant(node: Any, target_name: str, depth: int = 0) -> Any:
    """Return the descendant whose ``name`` field exactly matches ``target_name``."""
    if node is None or depth > _MAX_NAME_DEPTH:
        return None
    try:
        name_field = node.getField('name')
        if name_field is not None and name_field.getSFString() == target_name:
            return node
    except Exception:
        pass
    try:
        children_field = node.getField('children')
        if children_field is not None:
            for index in range(children_field.getCount()):
                child = children_field.getMFNode(index)
                result = find_named_descendant(child, target_name, depth + 1)
                if result is not None:
                    return result
    except Exception:
        pass
    try:
        endpoint_field = node.getField('endPoint')
        if endpoint_field is not None:
            endpoint = endpoint_field.getSFNode()
            result = find_named_descendant(endpoint, target_name, depth + 1)
            if result is not None:
                return result
    except Exception:
        pass
    return None


def find_named_descendant_contains(
    node: Any, name_fragment: str, depth: int = 0,
) -> Any:
    """Return the first descendant whose ``name`` contains ``name_fragment`` (case-insensitive)."""
    if node is None or depth > _MAX_NAME_DEPTH:
        return None
    lowered_fragment = str(name_fragment).lower()
    try:
        name_field = node.getField('name')
        if name_field is not None:
            name_value = str(name_field.getSFString())
            if lowered_fragment in name_value.lower():
                return node
    except Exception:
        pass
    try:
        children_field = node.getField('children')
        if children_field is not None:
            for index in range(children_field.getCount()):
                child = children_field.getMFNode(index)
                result = find_named_descendant_contains(
                    child, lowered_fragment, depth + 1,
                )
                if result is not None:
                    return result
    except Exception:
        pass
    try:
        endpoint_field = node.getField('endPoint')
        if endpoint_field is not None:
            endpoint = endpoint_field.getSFNode()
            result = find_named_descendant_contains(
                endpoint, lowered_fragment, depth + 1,
            )
            if result is not None:
                return result
    except Exception:
        pass
    return None


def get_field(node: Any, name: str) -> Any:
    """Safe wrapper for ``node.getField(name)``; returns ``None`` on any error."""
    try:
        return node.getField(name)
    except Exception:
        return None


def get_field_sfnode(node: Any, name: str) -> Any:
    field = get_field(node, name)
    if field is None:
        return None
    try:
        return field.getSFNode()
    except Exception:
        return None


def get_field_sfvec3f(node: Any, name: str) -> tuple[float, float, float] | None:
    field = get_field(node, name)
    if field is None:
        return None
    try:
        value = field.getSFVec3f()
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return None


def get_field_sffloat(node: Any, name: str) -> float | None:
    field = get_field(node, name)
    if field is None:
        return None
    try:
        return float(field.getSFFloat())
    except Exception:
        return None


__all__ = [
    'find_joint_endpoint',
    'find_named_descendant',
    'find_named_descendant_contains',
    'get_field',
    'get_field_sfnode',
    'get_field_sfvec3f',
    'get_field_sffloat',
]
