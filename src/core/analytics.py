"""Local analytics system for DocMind AI performance metrics.

This module provides a thread-safe, best-effort analytics system that collects
performance metrics for the DocMind AI application. It uses DuckDB for local
storage and implements non-blocking writes via background worker threads.

Key Features:
- Thread-safe singleton pattern for AnalyticsManager
- Non-blocking metric collection via queue system
- Automatic data retention and pruning
- Support for query, embedding, reranking, and system metrics
- Background worker thread for database operations

The system is designed to be "best-effort" - analytics failures won't
impact the main application performance.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import duckdb


@dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration for local analytics system.

    Attributes:
        enabled: Whether analytics collection is enabled.
        db_path: Path to the DuckDB database file for storing metrics.
        retention_days: Number of days to retain analytics data before pruning.
    """

    enabled: bool
    db_path: Path
    retention_days: int = 30


class AnalyticsManager:
    """Local best-effort analytics system with non-blocking writes and retention.

    This class provides a singleton-based analytics manager that collects and stores
    performance metrics for the DocMind AI system. It uses DuckDB for local storage
    and implements non-blocking writes via a background worker thread to avoid
    impacting application performance.

    Features:
    - Thread-safe singleton pattern
    - Non-blocking metric collection via queue
    - Automatic data retention and pruning
    - Support for query, embedding, reranking, and system metrics
    - Background worker thread for database operations

    The system is designed to be "best-effort" - if analytics operations fail,
    they won't impact the main application flow.
    """

    _instance: AnalyticsManager | None = None  # type: ignore[name-defined]
    _lock = threading.Lock()

    def __init__(self, cfg: AnalyticsConfig):
        """Initialize the analytics manager with configuration.

        Args:
            cfg: AnalyticsConfig containing enabled flag, database path, and
            retention settings.
        """
        self.cfg = cfg
        self._q: Queue[tuple[str, tuple[Any, ...]]] = Queue()
        self._worker: threading.Thread | None = None
        self._last_prune: datetime = datetime.min.replace(tzinfo=UTC)
        if self.cfg.enabled:
            self._ensure_dirs()
            self._ensure_schema()

    @classmethod
    def instance(cls, cfg: AnalyticsConfig):
        """Get singleton instance of AnalyticsManager.

        Args:
            cfg: AnalyticsConfig for the instance. If different from current instance,
                 a new instance will be created.

        Returns:
            The singleton AnalyticsManager instance.
        """
        with cls._lock:
            if cls._instance is None or cls._instance.cfg != cfg:
                cls._instance = cls(cfg)
            return cls._instance

    def _ensure_dirs(self) -> None:
        """Ensure the database directory exists."""
        os.makedirs(self.cfg.db_path.parent, exist_ok=True)

    def _conn(self):
        """Create and return a DuckDB connection to the analytics database.

        Returns:
            DuckDB connection object.
        """
        return duckdb.connect(str(self.cfg.db_path))

    def _ensure_schema(self) -> None:
        """Create analytics tables in the database if they don't exist.

        Creates the following tables:
        - query_metrics: Query performance metrics
        - embedding_metrics: Embedding operation metrics
        - reranking_metrics: Reranking operation metrics
        - system_metrics: General system metrics
        """
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS query_metrics(
                    ts TIMESTAMP,
                    query_type TEXT,
                    latency_ms DOUBLE,
                    result_count INTEGER,
                    retrieval_strategy TEXT,
                    success BOOLEAN
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_metrics(
                    ts TIMESTAMP, model TEXT, items INT, latency_ms DOUBLE
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS reranking_metrics(
                    ts TIMESTAMP, model TEXT, items INT, latency_ms DOUBLE
                );
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics(
                    ts TIMESTAMP, key TEXT, value DOUBLE
                );
                """
            )

    def _start_worker(self) -> None:
        """Start the background worker thread for processing analytics writes."""
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        """Background worker thread that processes analytics writes from the queue.

        This method runs in a daemon thread and continuously processes queued analytics
        operations. It also periodically prunes old records to maintain data retention
        policies. Operations are processed asynchronously to avoid blocking the main
        application thread.
        """
        while True:
            try:
                sql, params = self._q.get(timeout=2.0)
            except Empty:
                now = datetime.now(UTC)
                if (now - self._last_prune).total_seconds() > 3600:
                    self.prune_old_records()
                    self._last_prune = now
                continue
            try:
                with self._conn() as con:
                    con.execute(sql, params)
            except duckdb.Error:  # Best-effort: ignore database write errors
                # Don't let analytics failures break the app
                pass

    def log_query(
        self,
        *,
        query_type: str,
        latency_ms: float,
        result_count: int,
        retrieval_strategy: str,
        success: bool,
    ) -> None:
        """Log query performance metrics.

        Args:
            query_type: Type of query (e.g., 'chat', 'search').
            latency_ms: Query processing time in milliseconds.
            result_count: Number of results returned.
            retrieval_strategy: Strategy used for retrieval
            (e.g., 'hybrid', 'semantic').
            success: Whether the query was successful.
        """
        if not self.cfg.enabled:
            return
        # Synchronous, best-effort write for determinism in tests and simplicity
        with self._conn() as con:
            con.execute(
                "INSERT INTO query_metrics VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC),
                    query_type,
                    latency_ms,
                    result_count,
                    retrieval_strategy,
                    success,
                ),
            )

    def log_embedding(self, *, model: str, items: int, latency_ms: float) -> None:
        """Log embedding operation performance metrics.

        Args:
            model: Name of the embedding model used.
            items: Number of items processed in the embedding operation.
            latency_ms: Processing time in milliseconds.
        """
        if not self.cfg.enabled:
            return
        with self._conn() as con:
            con.execute(
                "INSERT INTO embedding_metrics VALUES (?, ?, ?, ?)",
                (datetime.now(UTC), model, items, latency_ms),
            )

    def log_reranking(self, *, model: str, items: int, latency_ms: float) -> None:
        """Log reranking operation performance metrics.

        Args:
            model: Name of the reranking model used.
            items: Number of items processed in the reranking operation.
            latency_ms: Processing time in milliseconds.
        """
        if not self.cfg.enabled:
            return
        with self._conn() as con:
            con.execute(
                "INSERT INTO reranking_metrics VALUES (?, ?, ?, ?)",
                (datetime.now(UTC), model, items, latency_ms),
            )

    def prune_old_records(self) -> None:
        """Remove analytics records older than the retention period.

        This method deletes records from all analytics tables that are older than
        the configured retention period (retention_days). It processes all tables:
        query_metrics, embedding_metrics, reranking_metrics, and system_metrics.

        The pruning is performed automatically by the background worker thread
        every hour to maintain database size and performance.
        """
        if not self.cfg.enabled:
            return
        cutoff = datetime.now(UTC) - timedelta(days=self.cfg.retention_days)
        with self._conn() as con:
            sql_by_table = {
                "query_metrics": "DELETE FROM query_metrics WHERE ts < ?",
                "embedding_metrics": "DELETE FROM embedding_metrics WHERE ts < ?",
                "reranking_metrics": "DELETE FROM reranking_metrics WHERE ts < ?",
                "system_metrics": "DELETE FROM system_metrics WHERE ts < ?",
            }
            for sql in sql_by_table.values():
                con.execute(sql, (cutoff,))


__all__ = ["AnalyticsConfig", "AnalyticsManager"]
