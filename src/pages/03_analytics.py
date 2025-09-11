# pylint: disable=invalid-name,too-many-statements
"""Streamlit Analytics page.

Reads metrics from the local DuckDB analytics database and renders a few
high-level charts for quick visibility into usage and performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import plotly.express as px
import streamlit as st

from src.config.settings import settings


def main() -> None:  # pragma: no cover - Streamlit page
    """Render the Analytics page and show charts from DuckDB metrics."""
    st.title("Analytics")

    if not getattr(settings, "analytics_enabled", False):
        st.info("Analytics disabled. Enable DOCMIND_ANALYTICS__ENABLED=true and retry.")
        st.stop()

    db_path = settings.analytics_db_path or (
        settings.data_dir / "analytics" / "analytics.duckdb"
    )
    db_path = Path(db_path)
    if not db_path.exists():
        st.warning("No analytics DB yet.")
        st.stop()

    con = duckdb.connect(str(db_path))

    st.subheader("Query volumes by strategy")
    df_strategy = con.execute(
        """
        SELECT retrieval_strategy, COUNT(*) AS n
        FROM query_metrics
        GROUP BY retrieval_strategy
        ORDER BY n DESC
        """
    ).df()
    st.plotly_chart(
        px.bar(df_strategy, x="retrieval_strategy", y="n"), use_container_width=True
    )

    st.subheader("Latency over time (avg ms)")
    df_latency = con.execute(
        """
        SELECT date_trunc('day', ts) AS day, AVG(latency_ms) AS avg_ms
        FROM query_metrics
        GROUP BY 1
        ORDER BY 1
        """
    ).df()
    st.plotly_chart(px.line(df_latency, x="day", y="avg_ms"), use_container_width=True)

    st.subheader("Success rate")
    df_success = con.execute(
        """
        SELECT success, COUNT(*) AS n
        FROM query_metrics
        GROUP BY success
        """
    ).df()
    st.plotly_chart(px.bar(df_success, x="success", y="n"), use_container_width=True)

    # Telemetry JSONL (optional)
    telem_path = Path("./logs/telemetry.jsonl")
    if telem_path.exists():
        st.subheader("Telemetry — Router Selection (JSONL)")
        counts: dict[str, int] = {}
        try:
            for line in telem_path.read_text(encoding="utf-8").splitlines():
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("router_selected"):
                    route = str(evt.get("route") or "unknown")
                    counts[route] = counts.get(route, 0) + 1
        except OSError:
            counts = {}
        if counts:
            df_routes = duckdb.from_df(
                __import__("pandas").DataFrame(
                    {"route": list(counts.keys()), "n": list(counts.values())}
                )
            ).df()
            st.plotly_chart(
                px.bar(df_routes, x="route", y="n"), use_container_width=True
            )

        st.subheader("Telemetry — Stale Snapshots & Exports (JSONL)")
        stale = 0
        exports = 0
        try:
            for line in telem_path.read_text(encoding="utf-8").splitlines():
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("snapshot_stale_detected"):
                    stale += 1
                if evt.get("export_performed"):
                    exports += 1
        except OSError:
            pass
        st.write(f"Stale detections: {stale}")
        st.write(f"Exports performed: {exports}")


if __name__ == "__main__":  # pragma: no cover
    main()
