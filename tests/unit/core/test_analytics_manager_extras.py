"""Extra coverage tests for AnalyticsManager synchronous writes and prune.

These tests exercise the embedding and reranking logging paths and validate
retention pruning across multiple tables using a temporary DuckDB file.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from src.core.analytics import AnalyticsConfig, AnalyticsManager


def test_log_embedding_and_reranking_and_prune(tmp_path: Path) -> None:
    """Embedding+reranking logging writes rows and prune clears them.

    Uses retention_days=0 so prune removes all rows.
    """
    db = tmp_path / "analytics.duckdb"
    cfg = AnalyticsConfig(enabled=True, db_path=db, retention_days=0)
    am = AnalyticsManager.instance(cfg)

    # Log a couple of records synchronously
    am.log_embedding(model="bge-m3", items=5, latency_ms=12.5)
    am.log_reranking(model="bge-reranker-v2", items=3, latency_ms=7.2)

    con = duckdb.connect(str(db))
    emb_n = con.execute("SELECT COUNT(*) FROM embedding_metrics").fetchone()[0]
    rerank_n = con.execute("SELECT COUNT(*) FROM reranking_metrics").fetchone()[0]
    assert emb_n == 1
    assert rerank_n == 1

    # Prune should remove all since retention_days=0
    am.prune_old_records()
    emb_n_after = con.execute("SELECT COUNT(*) FROM embedding_metrics").fetchone()[0]
    rerank_n_after = con.execute("SELECT COUNT(*) FROM reranking_metrics").fetchone()[0]
    assert emb_n_after == 0
    assert rerank_n_after == 0
