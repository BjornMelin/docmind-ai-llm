"""Unit tests for bounded telemetry parsing used by the Analytics page."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from src.utils import telemetry as telemetry_module
from src.utils.telemetry import (
    ANALYTICS_DUCKDB_PATH,
    get_analytics_duckdb_path,
    parse_telemetry_jsonl_counts,
)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_telemetry_jsonl_counts_ignores_invalid_lines(tmp_path: Path) -> None:
    p = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        p,
        [
            '{"router_selected": true, "route": "semantic_search"}',
            "not-json",
            '{"snapshot_stale_detected": true}',
            '{"export_performed": true}',
            '["not-a-dict"]',
        ],
    )

    counts = parse_telemetry_jsonl_counts(p, max_lines=100, max_bytes=1024 * 1024)

    assert counts.router_selected_by_route == {"semantic_search": 1}
    assert counts.snapshot_stale_detected == 1
    assert counts.export_performed == 1
    assert counts.invalid_lines >= 2
    assert counts.truncated is False


def test_parse_telemetry_jsonl_counts_enforces_max_lines(tmp_path: Path) -> None:
    p = tmp_path / "telemetry.jsonl"
    _write_jsonl(
        p,
        ['{"router_selected": true, "route": "hybrid_search"}'] * 10,
    )

    counts = parse_telemetry_jsonl_counts(p, max_lines=3, max_bytes=1024 * 1024)

    assert counts.router_selected_by_route == {"hybrid_search": 3}
    assert counts.truncated is True


def test_parse_telemetry_jsonl_counts_enforces_max_bytes(tmp_path: Path) -> None:
    p = tmp_path / "telemetry.jsonl"
    line = '{"router_selected": true, "route": "knowledge_graph"}'
    raw = (line + "\n").encode("utf-8")
    # Ensure we hit the cap deterministically in bytes.
    p.write_bytes(raw * 10)

    counts = parse_telemetry_jsonl_counts(p, max_lines=100, max_bytes=len(raw) * 2)

    assert counts.router_selected_by_route == {"knowledge_graph": 2}
    assert counts.truncated is True


def test_parse_telemetry_jsonl_counts_uses_canonical_path_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = tmp_path / "telemetry.jsonl"
    _write_jsonl(p, ['{"router_selected": true, "route": "semantic_search"}'])
    monkeypatch.setattr(telemetry_module, "TELEMETRY_JSONL_PATH", p, raising=False)

    counts = parse_telemetry_jsonl_counts(None, max_lines=10, max_bytes=1024)

    assert counts.router_selected_by_route == {"semantic_search": 1}


def test_get_analytics_duckdb_path_ignores_outside_data_dir(tmp_path: Path) -> None:
    override = tmp_path / "outside.duckdb"
    assert get_analytics_duckdb_path(override) == ANALYTICS_DUCKDB_PATH


def test_load_query_metrics_closes_connection_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    analytics = importlib.import_module("src.pages.03_analytics")

    class _FakeResult:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def df(self) -> pd.DataFrame:
            return self._df

    class _FakeConn:
        def __init__(self) -> None:
            self.closed = False
            self.calls = 0

        def execute(self, _sql: str) -> _FakeResult:
            self.calls += 1
            return _FakeResult(pd.DataFrame({"n": [self.calls]}))

        def close(self) -> None:
            self.closed = True

    conn = _FakeConn()
    monkeypatch.setattr(analytics.duckdb, "connect", lambda _p: conn)

    analytics._load_query_metrics(tmp_path / "analytics.duckdb")

    assert conn.closed is True


def test_load_query_metrics_closes_connection_on_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    analytics = importlib.import_module("src.pages.03_analytics")

    class _FakeResult:
        def df(self) -> pd.DataFrame:
            return pd.DataFrame({"n": [1]})

    class _FakeConn:
        def __init__(self) -> None:
            self.closed = False
            self.calls = 0

        def execute(self, _sql: str) -> _FakeResult:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return _FakeResult()

        def close(self) -> None:
            self.closed = True

    conn = _FakeConn()
    monkeypatch.setattr(analytics.duckdb, "connect", lambda _p: conn)

    with pytest.raises(RuntimeError, match="boom"):
        analytics._load_query_metrics(tmp_path / "analytics.duckdb")

    assert conn.closed is True
