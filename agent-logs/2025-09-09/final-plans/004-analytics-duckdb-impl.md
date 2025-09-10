# ADR-032 — Local Analytics (DuckDB) Implementation

Date: 2025-09-09

## Purpose

Implement an optional, local-only analytics database using DuckDB, separate from caches, with best-effort non-blocking writes and simple retention. Charts are rendered in the Analytics page (SPEC-008).

## Prerequisites

- DuckDB Python package installed
- `settings.analytics_enabled`, `settings.analytics_retention_days`, `settings.analytics_db_path` configured via env or defaults

## Files to Create/Update (Checklist)

- [x] Create: `src/core/analytics.py` (AnalyticsManager)
- [x] Update: `src/agents/coordinator.py` (log queries)
- [ ] Update: `src/processing/document_processor.py` (log ingestion)
- [x] Update: `src/pages/03_analytics.py` (charts; see SPEC-008 doc)

Code references: final-plans/011-code-snippets.md (Section 4 for manager; Section 5 for charts)

## Imports and Libraries

- Libraries: `duckdb`, `datetime`, `threading`, `queue`, `pathlib`
- App modules: `src.config.settings.settings`, `src.core.analytics.AnalyticsManager, AnalyticsConfig`
- UI charts: `plotly.express as px`, `pandas as pd`

Example imports:

```python
import duckdb
from datetime import datetime, timedelta, timezone
from queue import Queue
from pathlib import Path
from src.core.analytics import AnalyticsManager, AnalyticsConfig
```

## Schema and Config

- DB path: `settings.analytics_db_path` or `settings.data_dir / "analytics" / "analytics.duckdb"`
- Tables:
  - `query_metrics(ts TIMESTAMP, query_type TEXT, latency_ms DOUBLE, result_count INT, retrieval_strategy TEXT, success BOOLEAN)`
  - `embedding_metrics(ts TIMESTAMP, model TEXT, items INT, latency_ms DOUBLE)`
  - `reranking_metrics(ts TIMESTAMP, model TEXT, items INT, latency_ms DOUBLE)`
  - `system_metrics(ts TIMESTAMP, key TEXT, value DOUBLE)`
- Retention: hourly prune of records older than `settings.analytics_retention_days`

## Step-by-Step Implementation

1) Create `AnalyticsManager` in `src/core/analytics.py`

- Singleton per `AnalyticsConfig`.
- Queue-based writer with a background daemon thread.
- Short-lived `duckdb.connect` per batch write to avoid cross-thread issues.
- Methods: `.log_query(...)`, `.log_embedding(...)`, `.log_reranking(...)`, `.prune_old_records()`.
- Enclose all disk operations with try/except to keep best-effort semantics.

2) Wire Coordinator Query Logging

- In `src/agents/coordinator.py`:
  - On each processed chat request, compute `latency_ms` and record via `.log_query(query_type="chat", latency_ms=..., result_count=<k>, retrieval_strategy="hybrid", success=True/False)`.
  - Initialize instance with `AnalyticsConfig(enabled=settings.analytics_enabled, db_path=..., retention_days=settings.analytics_retention_days)`.

3) Wire Document Processor Ingestion Logging

- In `src/processing/document_processor.py`:
  - After successful processing, compute `ingest_ms` and `items=len(processed_docs)` and call `.log_embedding(model="unstructured+index", items=items, latency_ms=ingest_ms)`.

4) Analytics Charts

- In `src/pages/03_analytics.py` queries (see SPEC-008 doc):
  - Strategy counts (bar), daily avg latency (line), success counts (bar)
  - Use Plotly Express for rendering; gate page by `settings.analytics_enabled`.

## Acceptance Criteria

- Enabling analytics creates the DuckDB file and schema on first write.
- Chat queries and document ingestions append records without blocking UI.
- Old metrics are pruned based on retention.
- Analytics page displays charts or meaningful disabled/empty states.

Gherkin:

```gherkin
Feature: Local analytics
  Scenario: Query metrics insertion
    Given analytics is enabled
    When a chat query completes
    Then a row is appended to query_metrics

  Scenario: Ingestion metrics insertion
    Given analytics is enabled
    When I ingest documents
    Then rows are appended to embedding_metrics

  Scenario: Retention pruning
    Given analytics is enabled with retention_days=0
    When prune_old_records runs
    Then all previous rows are deleted
```

## Testing and Notes

- Unit test: create temp DB path, call `.log_query` and `.prune_old_records`, then assert counts.
- Performance: target <5ms per insert (expected due to local file DB and short SQL).
- Separation: analytics DB is separate from ingestion cache per ADR-032.

## Environment Variables

- Enable/disable: `DOCMIND_ANALYTICS__ENABLED=true|false`
- Retention days: `DOCMIND_ANALYTICS__RETENTION_DAYS=60`
- DB path override: `DOCMIND_ANALYTICS__DB_PATH=./data/analytics/analytics.duckdb`

## Cross-Links

- UI Analytics page: 003-ui-multipage-impl.md
- Code snippets for manager and charts: 011-code-snippets.md (Sections 4–5)

## No Backwards Compatibility

- Do not couple analytics to the ingestion DuckDBKV store. Remove any previous attempts to re-use the cache database for analytics; use the dedicated analytics DB only.
