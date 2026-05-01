"""Streamlit Analytics page.

Reads metrics from the local DuckDB analytics database and renders a few
high-level charts for quick visibility into usage and performance.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import duckdb
import plotly.express as px
import pyarrow as pa
import streamlit as st
from loguru import logger

from src.config.settings import settings
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import (
    get_analytics_duckdb_path,
    parse_telemetry_jsonl_counts,
)


def _load_query_metrics(
    db_path: Path,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Load and aggregate query performance metrics from DuckDB.

    Args:
        db_path: Path to the DuckDB database file.

    Returns:
        tuple[pa.Table, pa.Table, pa.Table]: Arrow tables containing
            strategy counts, daily average latency, and success/failure counts.
    """
    con = duckdb.connect(str(db_path))
    try:
        strategy_table = con.execute(
            """
            SELECT retrieval_strategy, COUNT(*) AS n
            FROM query_metrics
            GROUP BY retrieval_strategy
            ORDER BY n DESC
            """
        ).fetch_arrow_table()
        latency_table = con.execute(
            """
            SELECT date_trunc('day', ts) AS day, AVG(latency_ms) AS avg_ms
            FROM query_metrics
            GROUP BY 1
            ORDER BY 1
            """
        ).fetch_arrow_table()
        success_table = con.execute(
            """
            SELECT success, COUNT(*) AS n
            FROM query_metrics
            GROUP BY success
            """
        ).fetch_arrow_table()
        return strategy_table, latency_table, success_table
    finally:
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            con.close()


def main() -> None:  # pragma: no cover - Streamlit page
    """Render the Analytics page and show charts from DuckDB metrics."""
    st.title("Analytics")

    if not getattr(settings, "analytics_enabled", False):
        st.info("Analytics disabled. Enable DOCMIND_ANALYTICS_ENABLED=true and retry.")
        st.stop()

    db_path = get_analytics_duckdb_path(
        settings.analytics_db_path,
        base_dir=settings.data_dir,
    )
    if not db_path.exists():
        st.warning("No analytics DB yet.")
        st.stop()

    try:
        strategy_table, latency_table, success_table = _load_query_metrics(db_path)
    except Exception as exc:  # pragma: no cover - UX best effort
        redaction = build_pii_log_entry(str(exc), key_id="analytics.load_db")
        logger.warning(
            "Failed to load analytics DB (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        st.error(f"Failed to load analytics DB ({type(exc).__name__}).")
        st.caption(f"Error reference: {redaction.redacted}")
        st.stop()

    st.subheader("Query volumes by strategy")
    st.plotly_chart(
        px.bar(strategy_table, x="retrieval_strategy", y="n"),
        use_container_width=True,
    )

    st.subheader("Latency over time (avg ms)")
    st.plotly_chart(
        px.line(latency_table, x="day", y="avg_ms"),
        use_container_width=True,
    )

    st.subheader("Success rate")
    st.plotly_chart(
        px.bar(success_table, x="success", y="n"),
        use_container_width=True,
    )

    # Telemetry JSONL (optional)
    counts = parse_telemetry_jsonl_counts()
    if counts.lines_read > 0:
        st.subheader("Telemetry — Router Selection (JSONL)")
        if counts.router_selected_by_route:
            routes_table = pa.table(
                {
                    "route": list(counts.router_selected_by_route.keys()),
                    "n": list(counts.router_selected_by_route.values()),
                }
            )
            st.plotly_chart(
                px.bar(routes_table, x="route", y="n"), use_container_width=True
            )
        else:
            st.caption("No router selection events found.")

        st.subheader("Telemetry — Stale Snapshots & Exports (JSONL)")
        st.write(f"Stale detections: {counts.snapshot_stale_detected}")
        st.write(f"Exports performed: {counts.export_performed}")
        if counts.truncated:
            st.caption(
                "Telemetry view truncated due to size caps "
                "(showing aggregates from a bounded window)."
            )


if __name__ == "__main__":  # pragma: no cover
    main()
