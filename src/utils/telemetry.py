"""Lightweight JSONL telemetry emitter.

Writes events to logs/telemetry.jsonl to keep observability local-first.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TELEM_PATH = Path("./logs/telemetry.jsonl")

# Context-managed request id (optional). When set, it will be added to events.
_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    """Set the request ID for subsequent telemetry events in this context.

    Args:
        request_id: The request identifier to attach; None to clear.
    """
    _REQUEST_ID.set(request_id)


def get_request_id() -> str | None:
    """Return the current context request ID, if any."""
    return _REQUEST_ID.get()


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
    # Clamp sampling rate into [0,1] once, then sample.
    # rate=1.0 logs all events; rate=0.0 logs none.
    rate = max(0.0, min(1.0, rate))
    if rate < 1.0 and random.random() >= rate:  # noqa: S311
        return

    rec = {
        "ts": datetime.now(UTC).isoformat(),
        **event,
    }
    # Include request_id when present
    try:
        _rid = _REQUEST_ID.get()
        if _rid:
            rec.setdefault("request_id", _rid)
    except LookupError:  # pragma: no cover - contextvar edge
        pass
    _ensure_dir(_TELEM_PATH)
    _maybe_rotate(_TELEM_PATH)
    with _TELEM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


__all__ = ["log_jsonl"]
