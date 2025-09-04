---
ADR: 030
Title: Cache Unification via IngestionCache (DuckDBKVStore)
Status: Accepted
Version: 1.1
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 031, 026
Tags: cache, duckdb, ingestion
References:
- LlamaIndex IngestionCache, DuckDBKVStore
---

## Description

Unify document‑processing cache on LlamaIndex IngestionCache with DuckDBKVStore (single‑file DB). Remove custom caches and JSON persistence.

## Context

Custom wrappers duplicated maintained library features; lacked durability.

## Decision Drivers

- KISS; local‑first; durability

## Alternatives

- JSON store — fragile
- Custom cache — more code

### Decision Framework

| Option             | Leverage (35%) | Maint (30%) | Perf (25%) | Simp (10%) | Total | Decision |
| ------------------ | -------------- | ----------- | ---------- | ---------- | ----- | -------- |
| Ingest+DuckDB (Sel)| 0.95           | 0.9         | 0.9        | 0.9        | 0.92  | ✅ Sel.  |

## Decision

Use IngestionCache + DuckDBKVStore at `settings.cache_dir/docmind.duckdb` as the single processing cache.

## High-Level Architecture

Processor → IngestionPipeline → IngestionCache → DuckDB file

## Design

### Implementation Details

```python
def get_cache_path():
    return settings.cache_dir / "docmind.duckdb"
```

## Testing

- Verify cache hits/misses; performance under local IO

## Consequences

### Positive Outcomes

- Durable, simple cache; less code

### Dependencies

- Python: `llama-index>=0.10`, `duckdb`

## Changelog

- 1.1 (2025‑09‑03): Accepted unified cache
