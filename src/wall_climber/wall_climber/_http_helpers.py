"""Pure HTTP validation and stable-hash helpers used by the FastAPI backend.

These helpers do not touch ROS, the UploadStore, or any module-level state.
They are safe to import from anywhere. Behaviour is identical to the inline
helpers that previously lived at the top of ``wall_climber.web_server``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy
from fastapi import HTTPException, Request


def require_json_object(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=422, detail=f'{name} body must be a JSON object',
        )
    return raw


async def load_json_request(
    request: Request,
    *,
    name: str,
    max_bytes: int,
) -> dict[str, Any]:
    raw_body = await request.body()
    if len(raw_body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f'{name} exceeds the maximum allowed payload size',
        )
    try:
        raw_json = json.loads(raw_body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f'invalid {name} JSON: {exc}')
    return require_json_object(raw_json, name)


def reject_extra_fields(
    payload: dict[str, Any], allowed: set[str], name: str,
) -> None:
    extras = sorted(set(payload.keys()) - allowed)
    if extras:
        raise HTTPException(
            status_code=422,
            detail=f'{name} contains unsupported fields: {extras}',
        )


def validate_text_value(
    value: Any,
    field_name: str,
    *,
    max_chars: int,
    max_bytes: int,
) -> str:
    if not isinstance(value, str):
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be a string',
        )
    if not value.strip():
        raise HTTPException(
            status_code=422, detail=f'{field_name} must not be empty',
        )
    if len(value) > max_chars or len(value.encode('utf-8')) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f'{field_name} exceeds the maximum allowed size',
        )
    return value


def coerce_float(
    value: Any,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be numeric',
        )
    if not numpy.isfinite(numeric):
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be finite',
        )
    if minimum is not None and numeric < minimum:
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be >= {minimum}',
        )
    if maximum is not None and numeric > maximum:
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be <= {maximum}',
        )
    return numeric


def coerce_int(
    value: Any,
    *,
    field_name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be an integer',
        )
    if minimum is not None and numeric < minimum:
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be >= {minimum}',
        )
    if maximum is not None and numeric > maximum:
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be <= {maximum}',
        )
    return numeric


def coerce_bool(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise HTTPException(
                status_code=422, detail=f'{field_name} is required',
            )
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    raise HTTPException(status_code=422, detail=f'{field_name} must be boolean')


def _validate_hex32(raw: Any, *, field_name: str) -> str:
    if not isinstance(raw, str):
        raise HTTPException(
            status_code=422, detail=f'{field_name} must be a string',
        )
    value = raw.strip().lower()
    if len(value) != 32 or any(ch not in '0123456789abcdef' for ch in value):
        raise HTTPException(
            status_code=422,
            detail=f'{field_name} must be a 32-char lowercase hex string',
        )
    return value


def validate_upload_id(raw_upload_id: Any) -> str:
    return _validate_hex32(raw_upload_id, field_name='upload_id')


def validate_preview_id(raw_preview_id: Any) -> str:
    return _validate_hex32(raw_preview_id, field_name='preview_id')


def stable_float(value: float, *, precision: int = 7) -> float:
    rounded = round(float(value), precision)
    return 0.0 if abs(rounded) < (10.0 ** -precision) else rounded


def stable_point_payload(point: tuple[float, float]) -> list[float]:
    return [stable_float(point[0]), stable_float(point[1])]


def stable_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return stable_float(value)
    if isinstance(value, numpy.generic):
        return stable_payload(value.item())
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): stable_payload(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    if hasattr(value, '__dict__'):
        return stable_payload(vars(value))
    return str(value)


def stable_hash(value: Any) -> str:
    payload = stable_payload(value)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def settings_hash(settings: dict[str, Any]) -> str:
    excluded = {
        'preview_id',
        'created_at',
        'created_at_unix',
        'expires_at_unix',
        'ttl_seconds',
    }
    filtered = {
        key: value
        for key, value in settings.items()
        if key not in excluded
    }
    return stable_hash(filtered)


__all__ = [
    'require_json_object',
    'load_json_request',
    'reject_extra_fields',
    'validate_text_value',
    'coerce_float',
    'coerce_int',
    'coerce_bool',
    'validate_upload_id',
    'validate_preview_id',
    'stable_float',
    'stable_point_payload',
    'stable_payload',
    'stable_hash',
    'settings_hash',
]
