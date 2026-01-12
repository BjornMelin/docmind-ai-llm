"""Lightweight JSONL telemetry emitter.

Writes events to logs/telemetry.jsonl to keep observability local-first.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import random
from collections import Counter
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

TELEMETRY_JSONL_PATH = Path("./logs/telemetry.jsonl")
# Public constant for consumers that need the canonical telemetry path.
_TELEM_PATH = TELEMETRY_JSONL_PATH

# Public constant for consumers that need the canonical local analytics DB path.
ANALYTICS_DUCKDB_PATH = Path("data/analytics/analytics.duckdb")

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
        logger.debug(f"telemetry rotation skipped: {exc}")


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
    with contextlib.suppress(LookupError):
        if rid := _REQUEST_ID.get():
            rec.setdefault("request_id", rid)
    _ensure_dir(_TELEM_PATH)
    _maybe_rotate(_TELEM_PATH)
    with _TELEM_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_telemetry_jsonl_path() -> Path:
    """Return the canonical local telemetry JSONL path."""
    return TELEMETRY_JSONL_PATH


def get_analytics_duckdb_path(
    override: Path | None = None,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Return the local analytics DuckDB path (optionally overridden)."""
    resolved_base = (base_dir or Path("data")).resolve()
    default_path = resolved_base / "analytics" / "analytics.duckdb"
    if override is None:
        return default_path

    candidate = Path(override)
    try:
        resolved = candidate.resolve()
    except OSError:
        return default_path

    is_under_base = resolved_base in resolved.parents
    has_valid_suffix = resolved.suffix in {".duckdb", ".db"}
    if is_under_base and has_valid_suffix:
        return resolved

    reason = "outside data/" if not is_under_base else "invalid extension"
    logger.warning(
        f"analytics db path override ignored ({reason}): "
        f"override={candidate} base={resolved_base}"
    )
    return default_path


@dataclasses.dataclass(slots=True)
class TelemetryEventCounts:
    """Aggregated telemetry counters for safe display in the Analytics page."""

    router_selected_by_route: dict[str, int]
    snapshot_stale_detected: int
    export_performed: int
    lines_read: int
    bytes_read: int
    invalid_lines: int
    truncated: bool


def parse_telemetry_jsonl_counts(
    path: Path | None = None,
    *,
    max_lines: int = 50_000,
    max_bytes: int = 25 * 1024 * 1024,
) -> TelemetryEventCounts:
    """Parse local telemetry JSONL and return bounded aggregate counts.

    This is intentionally streaming/bounded to avoid loading large telemetry logs
    into memory. Invalid JSON lines are ignored.
    """
    if max_lines <= 0 or max_bytes <= 0:
        return TelemetryEventCounts(
            router_selected_by_route={},
            snapshot_stale_detected=0,
            export_performed=0,
            lines_read=0,
            bytes_read=0,
            invalid_lines=0,
            truncated=False,
        )

    p = get_telemetry_jsonl_path() if path is None else Path(path)
    if not p.exists():
        return TelemetryEventCounts(
            router_selected_by_route={},
            snapshot_stale_detected=0,
            export_performed=0,
            lines_read=0,
            bytes_read=0,
            invalid_lines=0,
            truncated=False,
        )

    router_counts: Counter[str] = Counter()
    stale = 0
    exports = 0
    bytes_read = 0
    lines_read = 0
    invalid_lines = 0
    truncated = False

    try:
        with p.open("rb") as f:
            for raw_line in f:
                next_lines_read = lines_read + 1
                next_bytes_read = bytes_read + len(raw_line)
                if next_lines_read > max_lines or next_bytes_read > max_bytes:
                    # Truncation stops before counting/processing the next line.
                    truncated = True
                    break
                lines_read = next_lines_read
                bytes_read = next_bytes_read
                try:
                    line = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    invalid_lines += 1
                    continue
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                if not isinstance(evt, dict):
                    invalid_lines += 1
                    continue

                if evt.get("router_selected"):
                    route = str(evt.get("route") or "unknown")
                    router_counts[route] += 1
                if evt.get("snapshot_stale_detected"):
                    stale += 1
                if evt.get("export_performed"):
                    exports += 1
    except OSError:
        return TelemetryEventCounts(
            router_selected_by_route={},
            snapshot_stale_detected=0,
            export_performed=0,
            lines_read=0,
            bytes_read=0,
            invalid_lines=0,
            truncated=False,
        )

    if truncated:
        logger.info(
            "telemetry parse cap hit: "
            f"path={p} lines={lines_read} bytes={bytes_read} "
            f"(max_lines={max_lines} max_bytes={max_bytes})"
        )

    return TelemetryEventCounts(
        router_selected_by_route=dict(router_counts),
        snapshot_stale_detected=stale,
        export_performed=exports,
        lines_read=lines_read,
        bytes_read=bytes_read,
        invalid_lines=invalid_lines,
        truncated=truncated,
    )


__all__ = [
    "ANALYTICS_DUCKDB_PATH",
    "TELEMETRY_JSONL_PATH",
    "TelemetryEventCounts",
    "get_analytics_duckdb_path",
    "get_telemetry_jsonl_path",
    "log_jsonl",
    "parse_telemetry_jsonl_counts",
]
