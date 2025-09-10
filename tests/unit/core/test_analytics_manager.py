"""Unit tests for AnalyticsManager (ADR-032).

These tests validate insert and prune behavior using a temporary DuckDB path.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.core.analytics import AnalyticsConfig, AnalyticsManager


def test_log_and_prune(tmp_path: Path) -> None:
    """Verify that logging inserts a row and pruning removes old rows.

    Args:
        tmp_path: Temporary directory provided by pytest.
    """
    db = tmp_path / "analytics.duckdb"
    am = AnalyticsManager.instance(
        AnalyticsConfig(enabled=True, db_path=db, retention_days=0)
    )
    am.log_query(
        query_type="chat",
        latency_ms=12.3,
        result_count=3,
        retrieval_strategy="hybrid",
        success=True,
    )
    con = duckdb.connect(str(db))
    n_before = con.execute("SELECT COUNT(*) FROM query_metrics").fetchone()[0]
    assert n_before == 1
    am.prune_old_records()
    n_after = con.execute("SELECT COUNT(*) FROM query_metrics").fetchone()[0]
    assert n_after == 0
