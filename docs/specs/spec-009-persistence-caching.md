---
spec: SPEC-009
title: Persistence and Caching: DuckDBKV Ingestion Cache + SQLite WAL Ops + Versioned Qdrant
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-PERS-001: Ingestion cache SHALL use DuckDBKV via LlamaIndex IngestionCache.
  - FR-PERS-002: Metadata SHALL use SQLite in WAL mode.
  - FR-PERS-003: Qdrant collections SHALL be versioned and idempotent.
related_adrs: ["ADR-010","ADR-031","ADR-030"]
---


## Objective

Adopt a simple offline-first persistence split: Qdrant for vectors, DuckDBKV for ingestion cache, SQLite WAL for ops metadata.

## Libraries and Imports

```python
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
import sqlite3
```

## File Operations

### UPDATE

- `src/core/processing.py`: `build_cache(settings)` returning IngestionCache(DuckDBKVStore).
- `src/models/storage.py`: SQLite helpers enabling WAL.

## Acceptance Criteria

```gherkin
Feature: Cache reuse
  Scenario: Re-ingest same file
    Given I ingest a file twice
    Then the second run SHALL hit the ingestion cache
```

## References

- LlamaIndex IngestionCache docs; SQLite WAL docs.
