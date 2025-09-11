"""Internal telemetry helpers for agent tools.

Currently minimal; future sprints may extend with metrics backends.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from loguru import logger


def log_event(event: str, **kwargs: Any) -> None:
    """Log a structured telemetry event (non-failing)."""
    with suppress(Exception):
        logger.bind(event=event).info("telemetry: {event}", **kwargs)
