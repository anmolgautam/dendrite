"""Deprecated — use runtime.persistence.PersistenceRecorder instead.

This module re-exports PersistenceRecorder as PersistenceObserver for
backward compatibility. Will be removed in a future version.
"""

from __future__ import annotations

import warnings as _warnings

from dendrux.runtime.persistence import (
    PersistenceRecorder as _PersistenceRecorder,
)
from dendrux.runtime.persistence import (
    _identity,
    _redact_dict,
    _redact_value,
    _serialize_message,
)

__all__ = [
    "_identity",
    "_redact_dict",
    "_redact_value",
    "_serialize_message",
]


class _DeprecatedAlias:
    """Descriptor that emits a deprecation warning on access."""

    def __init__(self, target: type, old_name: str) -> None:
        self._target = target
        self._old_name = old_name

    def __set_name__(self, owner: type, name: str) -> None:
        self._attr_name = name

    def __get__(self, obj: object, objtype: type | None = None) -> type:
        _warnings.warn(
            f"{self._old_name} is deprecated, use PersistenceRecorder from "
            "dendrux.runtime.persistence instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._target


class _Module:
    """Module-level namespace that warns on PersistenceObserver access."""

    PersistenceObserver = _DeprecatedAlias(_PersistenceRecorder, "PersistenceObserver")


# For `from dendrux.runtime.observer import PersistenceObserver` —
# this triggers the deprecation warning at import time.
def __getattr__(name: str) -> object:
    if name == "PersistenceObserver":
        _warnings.warn(
            "PersistenceObserver is deprecated, use PersistenceRecorder from "
            "dendrux.runtime.persistence instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _PersistenceRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
