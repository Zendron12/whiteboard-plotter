"""Thread-safe store for the most recent plan/execution/curve-fit snapshots.

Used by the ``/api/debug/last-*`` endpoints to expose the last payloads seen
by the backend. Each slot is independent; reads and writes take a single
shared lock to keep the slots consistent.
"""

from __future__ import annotations

import threading
from typing import Any


class DebugSnapshotStore:
    """Three-slot last-value store for plan, execution and curve-fit payloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_plan: dict[str, Any] | None = None
        self._last_execution: dict[str, Any] | None = None
        self._last_curve_fit: dict[str, Any] | None = None

    # -- writers ------------------------------------------------------

    def record_plan(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._last_plan = dict(payload)

    def record_execution(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._last_execution = dict(payload)

    def record_curve_fit(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._last_curve_fit = dict(payload)

    # -- readers ------------------------------------------------------

    def plan_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._last_plan) if self._last_plan is not None else None

    def execution_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return (
                dict(self._last_execution)
                if self._last_execution is not None
                else None
            )

    def curve_fit_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return (
                dict(self._last_curve_fit)
                if self._last_curve_fit is not None
                else None
            )


__all__ = ['DebugSnapshotStore']
