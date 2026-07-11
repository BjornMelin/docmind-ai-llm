# DocMind AI Cache Implementation

> Implementation Guide — Source of truth: ADR-030/ADR-031. This page documents how to wire, configure, operate, and troubleshoot the cache. Architectural rationale lives in the ADRs.

## Overview

DocMind uses a single document-processing cache based on LlamaIndex IngestionCache with DuckDBKVStore. The cache is a local single-file database and requires no external services.

## Design Goals

- Library-first: use LlamaIndex cache APIs directly
- Local-first: operate fully offline
- Minimal code: no custom cache wrappers
- Clear boundaries: cache is separate from vectors (Qdrant) and operational data (SQLite if used)

## Architecture

```mermaid
graph TD
    A[DocumentProcessor] --> B[IngestionPipeline]
    B --> C[IngestionCache]
    C --> D[DuckDBKVStore (docmind.duckdb)]
```

- Cache file: `settings.cache.ingestion_db_path`
- Collection/namespace: `docmind_processing`
- Policy: do not add a custom cache wrapper or interface; use library types directly.

## Implementation

### Wiring in code

```python
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.kvstore.duckdb import DuckDBKVStore

cache_db = settings.cache.ingestion_db_path
cache_db.parent.mkdir(parents=True, exist_ok=True)

kv = DuckDBKVStore(db_path=str(cache_db))
ingestion_cache = IngestionCache(cache=kv, collection="docmind_processing")
```

### Clearing the cache

```python
cache_db = settings.cache.ingestion_db_path
if cache_db.exists():
    cache_db.unlink()
```

### Minimal statistics

Expose only basic information unless the KV API provides counts.

```python
def get_cache_stats() -> dict[str, object]:
    return {
        "cache_type": "duckdb_kvstore",
        "db_path": str(cache_db),
        "total_documents": -1,  # use -1 if counting keys is not available
    }
```

## Behavior

- Single local file for durability
- No external dependencies
- Cache keys and invalidation are handled by LlamaIndex’s ingestion layer

## Operations

- Inspect location: confirm `settings.cache.ingestion_db_path` exists.
- Backup/restore: copy or move the single file while the app is stopped.
- Relocate: change `DOCMIND_CACHE__DIR`; the file will be created on demand.
- Clear: delete `docmind.duckdb`; it will be recreated on next use.

## Failure Modes and Fixes

- ImportError: `llama_index.storage.kvstore.duckdb` not found
  - Ensure LlamaIndex version exposes `DuckDBKVStore` and required integration is installed.
- PermissionError on cache path
  - Use a writable `DOCMIND_CACHE__DIR`; create parent directories up front.
- File lock or DB busy
  - Stop concurrent writers; DuckDB is single-writer. Retry after closing processes.
- Corrupted DB file
  - Stop the app, delete `docmind.duckdb`, restart (cache will rebuild as needed).

## Configuration

- `DOCMIND_CACHE__DIR` → `settings.cache.dir`
- `DOCMIND_CACHE__FILENAME` → `settings.cache.filename`

## Testing Guidance

- Use a temporary directory for `settings.cache.dir` in tests to avoid collisions.
- For unit tests that do not need DuckDB behavior, use a small stub/mocked cache local to `tests/` instead of adding production wrappers.
- Clean up the temp cache file between tests.

## Related Components

- Vectors: Qdrant (local) per ADR-031
- Operational metadata: SQLite if needed; not coupled to the cache DB

## Migration Notes

- Custom `SimpleCache` and SimpleKVStore usage are removed.
- No backward compatibility for prior cache state.
- To clear existing state, delete `cache/ingestion/docmind.duckdb`.

## References

- ADR-030: Cache Unification via IngestionCache (DuckDBKVStore)
- ADR-031: Local-First Persistence Architecture
