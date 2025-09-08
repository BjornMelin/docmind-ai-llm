"""Lightweight JSONL telemetry emitter.

Writes events to logs/telemetry.jsonl to keep observability local-first.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TELEM_PATH = Path("./logs/telemetry.jsonl")


def _ensure_dir(path: Path) -> None:
    with contextlib.suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)


def _maybe_rotate(path: Path) -> None:
    """Rotate file if it exceeds DOCMIND_TELEMETRY_ROTATE_BYTES (size-based)."""
    try:
        limit = int(os.getenv("DOCMIND_TELEMETRY_ROTATE_BYTES", "0"))
    except (ValueError, TypeError):
        limit = 0
    if limit <= 0:
        return
    try:
        if path.exists() and path.stat().st_size >= limit:
            rotated = path.with_suffix(path.suffix + ".1")
            # Remove previous rotated if present
            with contextlib.suppress(Exception):
                rotated.unlink()
            path.rename(rotated)
    except OSError as exc:
        # Never fail the app due to rotation; log at debug level
        logging.debug("telemetry rotation skipped: %s", exc)


def log_jsonl(event: dict[str, Any]) -> None:
    """Append a JSON event with ISO timestamp to the local JSONL file.

    Args:
        event: Flat key-value telemetry dictionary.
    """
    # avoid network/file egress if disabled
    if os.getenv("DOCMIND_TELEMETRY_DISABLED", "false").lower() in {"1", "true", "yes"}:
        return
    # sampling
    try:
        rate = float(os.getenv("DOCMIND_TELEMETRY_SAMPLE", "1.0"))
    except (ValueError, TypeError):
        rate = 1.0
    if rate < 1.0 and random.random() > max(0.0, min(1.0, rate)):  # noqa: S311
        return

    rec = {
        "ts": datetime.now(UTC).isoformat(),
        **event,
    }
    _ensure_dir(_TELEM_PATH)
    _maybe_rotate(_TELEM_PATH)
    with _TELEM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


__all__ = ["log_jsonl"]
