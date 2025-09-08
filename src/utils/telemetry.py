"""Lightweight JSONL telemetry emitter.

Writes events to logs/telemetry.jsonl to keep observability local-first.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TELEM_PATH = Path("./logs/telemetry.jsonl")


def _ensure_dir(path: Path) -> None:
    with contextlib.suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)


def log_jsonl(event: dict[str, Any]) -> None:
    """Append a JSON event with ISO timestamp to the local JSONL file.

    Args:
        event: Flat key-value telemetry dictionary.
    """
    # avoid network/file egress if disabled
    if os.getenv("DOCMIND_TELEMETRY_DISABLED", "false").lower() in {"1", "true", "yes"}:
        return
    rec = {
        "ts": datetime.now(UTC).isoformat(),
        **event,
    }
    _ensure_dir(_TELEM_PATH)
    with _TELEM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


__all__ = ["log_jsonl"]
