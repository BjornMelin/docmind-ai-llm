# ADR-032: Local Analytics & Metrics (DuckDB)

## Metadata

**Status:** Proposed  
**Version/Date:** v1.0 / 2025-09-02

## Title

Optional Local Analytics Database for Performance and Quality Metrics

## Description

Define an optional, local-only analytics database using DuckDB to record lightweight performance and quality metrics. The analytics database is decoupled from the document-processing cache (ADR-030) and the persistence architecture (ADR-031). It is disabled by default and can be enabled by advanced users who want local insights without relying on external services.

## Context

- Archived ADR-007 included a full analytics subsystem in DuckDB with multiple metric tables.  
- ADR-030/ADR-031 simplified cache/persistence and intentionally avoided coupling analytics to the cache DB.  
- We want to retain the value of local analytics while keeping it optional, minimal, and separate from production responsibilities.

## Decision Drivers

- Local-first and offline-friendly; zero external services.
- Minimal footprint and clear separation of concerns from cache (ADR-030).
- Optional, opt-in usage to avoid unnecessary complexity for default users.
- Simple schema with standard tables and straightforward retention.

## Alternatives

- External observability services (rejected: violates local-first by default).
- Coupling analytics with cache DB (rejected: breaks separation of responsibilities established in ADR-031).
- No analytics at all (accepted for default; this ADR enables optional opt-in).

## Decision

Provide a separate DuckDB analytics database with a minimal standard schema. Off by default. When enabled, the app records metrics by writing to these tables via narrow ingestion helpers with best-effort error handling.

## Design

### Database Location

- `settings.data_dir / "analytics" / "analytics.duckdb"`

### Minimal Schema

- `query_metrics(timestamp, query_type, latency_ms, result_count, retrieval_strategy, success)`
- `embedding_metrics(timestamp, model_name, embedding_time_ms, text_length, dimension)`
- `reranking_metrics(timestamp, query_type, document_count, rerank_time_ms, quality_score)`
- `system_metrics(timestamp, metric_name, metric_value, metric_unit)`

### Retention

- Prune records older than a configurable window (e.g., 30/60/90 days).  
- Pruning runs only when analytics is enabled.

### Ingestion

- Narrow ingestion helpers collect and write metrics.  
- Best-effort writes; analysis should tolerate missing data.

## Configuration

- `analytics.enabled`: bool (default: False)  
- `analytics.retention_days`: int (default: 60)  
- `analytics.db_path`: Optional path override (default path above)

## Testing

- **Unit**: Ensure schema creation succeeds; inserts work for each table; prune operation deletes old rows.  
- **Integration**: When enabled, write and read a small set of metrics; confirm no interference with cache.

## Dependencies

- `duckdb`

## Related Decisions

- **ADR-010** (Performance Optimization): Can surface FP8 KV cache and agent timing metrics locally.
- **ADR-012** (Evaluation Strategy): Complements evaluation/quality metrics for local analysis.  
- **ADR-013** (UI Architecture): Analytics page may read this DB when enabled.
- **ADR-031** (Persistence Architecture): Keeps analytics separate from cache and ops data.

## Consequences

- **Positive**: Local insight into latency and quality without external services; entirely optional.  
- **Trade-offs**: Additional disk usage; extra code paths (opt-in only).

## Changelog

- **v1.0 (2025-09-02)**: Initial proposal for optional DuckDB analytics DB and schema.
