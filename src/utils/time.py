"""Time utilities.

Keep small and dependency-free.
"""

from __future__ import annotations

import time


def now_ms() -> int:
    """Return current wall-clock time in milliseconds."""
    return time.time_ns() // 1_000_000
