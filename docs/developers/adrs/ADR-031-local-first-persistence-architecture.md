# ADR-031: Local-First Persistence Architecture (Vectors, Cache, Operational Data)

## Metadata

**Status:** Accepted  
**Version/Date:** v1.0 / 2025-09-02

## Title

Local-First Persistence Architecture for DocMind AI

## Description

Define a clean, non-legacy persistence architecture:

- **Vectors**: Qdrant (local) for embeddings and retrieval
- **Cache**: LlamaIndex IngestionCache with DuckDBKVStore (single-file cache DB)
- **Operational data**: SQLite allowed for settings/session metadata where applicable
- No external services required; fully offline after installation

## Context

Previous ADRs proposed multiple storage backends and optional analytics. To simplify and avoid over-engineering, we separate concerns:

- Vector retrieval persists in Qdrant
- Document-processing cache persists in a single local DuckDB file via LlamaIndex IngestionCache
- Operational metadata may use SQLite (via app configuration), but must not introduce test seams or legacy wrappers

## Decision Drivers

- KISS (minimal components and responsibilities)
- Local-first, offline operation
- Library-first (use LlamaIndex and Qdrant clients directly)
- Maintainability and clarity (no custom cache implementations)

## Alternatives

- **A**: Redis or external services — rejected (violates offline/local-first)
- **B**: Custom caching layers (JSON/diskcache) — rejected (reinvents LlamaIndex features)
- **C**: Single-store for everything — rejected (forces mismatched requirements)

### Decision Framework

| Option | Simplicity (30%) | Library Fit (30%) | Performance (25%) | Maintainability (15%) | Total |
|-------|-------------------|-------------------|-------------------|-----------------------|-------|
| **Selected (Qdrant + IngestCache/DuckDBKV)** | 0.9 | 0.95 | 0.9 | 0.95 | 0.92 |
| Redis-based | 0.5 | 0.6 | 0.85 | 0.7 | 0.64 |
| Custom cache | 0.4 | 0.5 | 0.7 | 0.4 | 0.51 |

## Decision

Adopt the following local-first architecture:

- **Vectors**: Qdrant (local) via llama-index-vector-stores-qdrant
- **Cache**: LlamaIndex IngestionCache with DuckDBKVStore (single file at `settings.cache_dir/docmind.duckdb`)
- **Operational metadata**: SQLite permitted (e.g., ChatMemoryStore), but no test-only hooks in src

## High-Level Architecture

```mermaid
graph TD
    A[App] --> B[DocumentProcessor]
    B --> C[IngestionPipeline]
    C --> D[IngestionCache (DuckDBKVStore)]
    B --> E[Vector Store (Qdrant)]
    A --> F[Operational (SQLite, if used)]
```

## Related Requirements

### Functional

- **FR-1**: Persist cache entries and reuse them across runs
- **FR-2**: Store & retrieve vectors for similarity search

### Non-Functional

- **NFR-1**: Fully offline/local
- **NFR-2**: Minimal code and clean boundaries (no test seams)
- **NFR-3**: Durable single-file cache

### Integration

- **IR-1**: LlamaIndex integrations for cache and vector stores

## Related Decisions

- **ADR-030**: Cache Unification (IngestionCache/DuckDBKVStore)
- **ADR-026**: Test-production separation
- **ADR-033**: Local Backup & Retention (optional manual backups/rotation)

## Design

### Implementation Details

- Cache wiring in DocumentProcessor:

  ```python
  from pathlib import Path
  from llama_index.core.ingestion import IngestionCache
  from llama_index.storage.kvstore.duckdb import DuckDBKVStore

  cache_db = Path(settings.cache_dir) / "docmind.duckdb"
  kv = DuckDBKVStore(db_path=str(cache_db))
  self.cache = IngestionCache(cache=kv, collection="docmind_processing")
  ```

- Vector store uses Qdrant (local client) via `llama-index-vector-stores-qdrant`.

### Configuration

- Cache file path: `DOCMIND_CACHE_DIR` → `settings.cache_dir`
- No service dependencies beyond local DB files

## Testing

- **Unit**: cache store/get/clear; minimal stats
- **Integration**: pipeline re-run behaves identically but faster on second run

## Consequences

### Positive

- Clean, minimal architecture; easy maintenance
- Fully offline
- Clear separation of responsibilities

### Trade-offs

- Analytics not co-located with cache DB; add a separate DB if needed later

### Maintenance

- Track LlamaIndex and DuckDB integration versions

## Dependencies

- llama-index==0.13.x
- llama-index-storage-kvstore-duckdb
- llama-index-vector-stores-qdrant
- duckdb

## Changelog

- **1.0 (2025-09-02)**: Initial accepted version.
