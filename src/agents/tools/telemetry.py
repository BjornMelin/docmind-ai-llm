"""Internal telemetry helpers for agent tools.

Currently minimal; future sprints may extend with metrics backends.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from loguru import logger

from src.utils.log_safety import build_pii_log_entry

_SAFE_KEYS = {"event_type", "status", "route"}


def log_event(event: str, **kwargs: Any) -> None:
    """Log a structured telemetry event (non-failing)."""
    with suppress(Exception):
        safe: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and key not in _SAFE_KEYS:
                safe[key] = build_pii_log_entry(
                    value, key_id=f"tool_telemetry:{event}:{key}"
                ).redacted
            else:
                safe[key] = value
        logger.bind(event=event, **safe).info("telemetry: {}", event)
