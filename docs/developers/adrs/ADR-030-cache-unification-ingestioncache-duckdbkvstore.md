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
- [LlamaIndex — Ingestion Cache](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents/ingestion_cache/)
- [LlamaIndex — DuckDBKVStore](https://docs.llamaindex.ai/en/stable/module_guides/storing/kv_stores/#duckdbkvstore)
---

## Description

Unify document‑processing cache on LlamaIndex IngestionCache with DuckDBKVStore (single‑file DB). Remove custom caches and JSON persistence.

## Context

- Custom cache wrappers increased maintenance and re-implemented functionality already provided by LlamaIndex.
- JSON-based persistence lacks robustness/concurrency characteristics for larger caches.
- We require a single, durable, local-first cache with minimal code surface.

## Decision Drivers

- KISS; local‑first; durability

## Alternatives

- A: IngestionCache + JSON (SimpleKVStore) — Pros: simplest; Cons: limited durability/concurrency.
- B: IngestionCache + DuckDBKVStore — Pros: robust, single-file DB, local-first; Cons: integration dependency.
- C: Custom SimpleCache wrapper — Pros: known behavior; Cons: re-invents wheel; more code; harder to maintain.

### Decision Framework

| Model / Option         | Solution Leverage (35%) | Maintenance (30%) | Performance (25%) | Simplicity (10%) | Total Score | Decision      |
| ---------------------- | ----------------------- | ----------------- | ----------------- | ---------------- | ----------- | ------------- |
| **B: Ingest+DuckDBKV** | 0.95                    | 0.9               | 0.9               | 0.9              | **0.92**    | ✅ **Selected** |
| A: Ingest+JSON         | 0.8                     | 0.85              | 0.6               | 1.0              | 0.79        | Rejected      |
| C: Custom Wrapper      | 0.4                     | 0.3               | 0.6               | 0.5              | 0.43        | Rejected      |

## Decision

Use IngestionCache + DuckDBKVStore at `settings.cache_dir/docmind.duckdb` as the single processing cache.

## High-Level Architecture

Processor → IngestionPipeline → IngestionCache → DuckDB file

## Design

### Implementation Details

```python
def get_cache_path():
    return settings.cache_dir / "docmind.duckdb"

from llama_index.core import IngestionPipeline
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.kvstore import KVDocumentStore
from llama_index.core.storage.kvstore.duckdb_kvstore import DuckDBKVStore
from llama_index.core.indices.ingestion.cache import IngestionCache

def make_ingestion_cache(db_path) -> IngestionCache:
    kv = DuckDBKVStore(db_path=str(db_path))
    docstore = KVDocumentStore(kvstore=kv)
    return IngestionCache(docstore=docstore)
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
