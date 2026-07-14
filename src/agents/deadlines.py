"""Canonical absolute-deadline helpers for agent tools."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any


def remaining_deadline_seconds(
    state: Mapping[str, Any],
    *,
    operation: str,
) -> float:
    """Return the finite positive budget remaining for an agent operation."""
    try:
        deadline_ts = float(state["deadline_ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{operation} deadline is unavailable") from exc
    remaining = deadline_ts - time.monotonic()
    if not math.isfinite(remaining):
        raise ValueError(f"{operation} deadline is invalid")
    if remaining <= 0:
        raise TimeoutError(f"{operation} deadline exceeded")
    return remaining


__all__ = ["remaining_deadline_seconds"]
