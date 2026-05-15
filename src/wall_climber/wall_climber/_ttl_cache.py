"""Tiny TTL + LRU bounded cache used by the web backend preview stores.

The implementation is intentionally small: it wraps an ``OrderedDict`` with
size and time-to-live bounds. It does not know or care about the shape of the
entries it holds, only that each entry exposes a ``created_at_unix`` float.

The web backend keeps two such caches (one for generic previews, one for
sketch previews). Extracting this class removes ~40 lines of duplicated
house-keeping from ``web_server.py`` while keeping its behaviour identical.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Generic, Protocol, TypeVar


class _HasCreatedAt(Protocol):
    created_at_unix: float


E = TypeVar('E', bound=_HasCreatedAt)


class TTLCache(Generic[E]):
    """Bounded OrderedDict with ``(max_entries, ttl_seconds)`` eviction policy.

    Entries are evicted eagerly on :meth:`store` and :meth:`load`; callers do
    not need to schedule separate cleanup work.
    """

    def __init__(self, *, max_entries: int, ttl_seconds: float) -> None:
        if max_entries <= 0:
            raise ValueError('max_entries must be positive')
        if ttl_seconds <= 0:
            raise ValueError('ttl_seconds must be positive')
        self._max_entries = int(max_entries)
        self._ttl_seconds = float(ttl_seconds)
        self._entries: OrderedDict[str, E] = OrderedDict()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def entries(self) -> 'OrderedDict[str, E]':
        """Return a live view of the underlying ordered dictionary.

        Mainly intended for callers that need to iterate (e.g. diagnostics);
        mutating the returned dict bypasses the cache invariants.
        """
        return self._entries

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def is_expired(self, entry: E, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        return (current - float(entry.created_at_unix)) > self._ttl_seconds

    def prune(self, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        expired_ids = [
            key
            for key, entry in self._entries.items()
            if self.is_expired(entry, now=current)
        ]
        for key in expired_ids:
            self._entries.pop(key, None)
        self._enforce_capacity()

    def store(self, key: str, entry: E) -> None:
        self.prune()
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._enforce_capacity()

    def load(self, key: str) -> E | None:
        """Return the stored entry, or ``None`` if it is missing or expired.

        Expired entries are removed as a side effect.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        now = time.time()
        if self.is_expired(entry, now=now):
            self._entries.pop(key, None)
            return None
        self.prune(now=now)
        self._entries.move_to_end(key)
        return entry

    def pop(self, key: str, default: Any = None) -> Any:
        return self._entries.pop(key, default)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enforce_capacity(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


__all__ = ['TTLCache']
