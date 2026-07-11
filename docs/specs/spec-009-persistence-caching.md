---
spec: SPEC-009
title: Persistence and Caching: DuckDBKV Ingestion Cache and Qdrant
version: 1.1.0
date: 2026-07-11
owners: ["ai-arch"]
status: Revised
related_requirements:
  - FR-PERS-001: Ingestion cache SHALL use DuckDBKV via LlamaIndex IngestionCache.
  - FR-PERS-003: Qdrant collections SHALL use the canonical named-vector schema and explicit compatibility checks.
related_adrs: ["ADR-010","ADR-031","ADR-030"]
---

## Objective

Define the implemented ingestion persistence split:

- Qdrant stores dense and sparse vectors.
- LlamaIndex `IngestionCache` backed by `DuckDBKVStore` caches ingestion work.

The proposed SQLite WAL operational-metadata store is future design, not part
of this release contract. See SPEC-039 and ADR-055. Existing chat and memory
features own their separate SQLite stores.

## Libraries and Imports

```python
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
```

## Implementation ownership

- `src/processing/ingestion_pipeline.py` owns `_ensure_cache_path()` and
  `build_ingestion_pipeline()`.
- `build_ingestion_pipeline()` constructs `DuckDBKVStore`, wraps it in
  `IngestionCache`, and passes the cache to LlamaIndex.
- `src/utils/storage.py` owns Qdrant collection creation and compatibility
  checks. Normal startup does not mutate an incompatible collection.

Cache location is controlled by `DOCMIND_CACHE__DIR` and
`DOCMIND_CACHE__FILENAME`.

## Acceptance Criteria

```gherkin
Feature: Cache reuse
  Scenario: Re-ingest same file
    Given I ingest a file twice
    Then the second run SHALL hit the ingestion cache
```

Verification lives in `tests/unit/processing/test_ingestion_pipeline.py` and
the Qdrant schema tests under `tests/unit/utils/storage/`.

## References

- LlamaIndex `IngestionCache` and `DuckDBKVStore` documentation
- `docs/specs/spec-039-operational-metadata-sqlite-wal.md`
- `docs/developers/adrs/ADR-055-operational-metadata-sqlite-wal.md`
