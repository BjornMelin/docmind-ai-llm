---
ADR: 031
Title: Local-First Persistence Architecture (Vectors, Cache, Operational Data)
Status: Accepted (Amended)
Version: 1.5
Date: 2025-09-09
Supersedes:
Superseded-by:
Related: 026, 030, 033, 035, 038
Tags: persistence, storage, qdrant, duckdb, sqlite, cache
References:
- [Qdrant — Documentation](https://qdrant.tech/documentation/)
- [LlamaIndex — Ingestion Pipeline & Cache](https://docs.llamaindex.ai/)
- [DuckDB — Documentation](https://duckdb.org/docs/)
---

## Description

Adopt a simple, offline-first persistence split: Qdrant for vectors, LlamaIndex IngestionCache backed by DuckDBKVStore for document-processing cache, and optional SQLite for operational metadata. No external services required; minimal code and clear responsibilities.

## Context

Earlier designs mixed concerns and introduced multiple storage backends. To reduce complexity and maintenance, we separate responsibilities: vector search in Qdrant; processing cache in a single DuckDB file via LlamaIndex; operational metadata optionally in SQLite. Analytics remains separate (ADR‑032/033), avoiding coupling to cache or vectors.

## Decision Drivers

- KISS: minimal components and responsibilities
- Local-first, fully offline operation
- Library-first: direct LlamaIndex and Qdrant usage
- Maintainability: remove custom caches and legacy wrappers

## Alternatives

- A: Redis/external services — Pros: familiar; Cons: breaks offline/local-first
- B: Custom caching (JSON/diskcache) — Pros: control; Cons: reinvents LlamaIndex features
- C: Single-store for all data — Pros: one DB; Cons: mismatched requirements and tuning

### Decision Framework

| Model / Option                             | Simplicity (30%) | Library Fit (30%) | Performance (25%) | Maintenance (15%) | Total Score | Decision      |
| ------------------------------------------ | ---------------- | ----------------- | ----------------- | ----------------- | ----------- | ------------- |
| Qdrant + IngestionCache/DuckDBKV (Selected)| 9                | 10                | 9                 | 9                 | **9.2**     | ✅ Selected    |
| Redis-based                                | 5                | 6                 | 8.5               | 7                 | 6.4         | Rejected      |
| Custom cache                               | 4                | 5                 | 7                 | 4                 | 5.1         | Rejected      |

## Decision

Adopt Qdrant for vectors, LlamaIndex IngestionCache with DuckDBKVStore for processing cache (single file at `settings.cache_dir/docmind.duckdb`), and optionally SQLite for operational metadata. No test-only hooks in src; rely on library clients.

Hybrid Retrieval Schema (Qdrant Collections):

- Qdrant collections SHALL define named vectors to support server-side hybrid queries via the Query API:
  - `text-dense`: `VectorParams(distance=COSINE, size=<embed_dim>)`
  - `text-sparse`: `SparseVectorParams(modifier=IDF)`
- Collection ensure is idempotent at startup; schema updates warn when dimensions differ.
- See SPEC‑004 for hybrid query and fusion details (Prefetch + FusionQuery; RRF default; DBSF optional via env).

### SnapshotManager (Amendment — GraphRAG)

For GraphRAG and indices requiring consistent reloads, adopt a SnapshotManager:

- Write under `storage/_tmp-<uuid>`; `fsync` and atomically rename to `storage/<timestamp>`; readers resolve the `CURRENT` pointer first before falling back to lexicographic ordering.
- Persist vector index via `StorageContext.persist`; persist property graph via `SimpleGraphStore.persist` and package graph exports under `graph/graph_export-YYYYMMDDTHHMMSSZ.*` (JSONL required, Parquet optional) with telemetry metadata (`seed_count`, `size_bytes`, `duration_ms`, checksum).
- Emit tri-file manifests (`manifest.jsonl`, `manifest.meta.json`, `manifest.checksum`) plus optional `errors.jsonl`; `manifest.meta.json` keeps `complete=false` until promotion succeeds, then flips to `true`. Legacy `manifest.json` SHALL NOT be emitted.
- Use a bounded `SnapshotLock` (`.lock` + JSON metadata) implemented via `portalocker` with heartbeats and stale eviction; metadata captures `owner_id`, `ttl_seconds`, `takeover_count`, and timestamps. Rotate stale lock files to `.stale-*` suffixes; fall back to `os.O_EXCL` when `portalocker` is unavailable. Telemetry captures export operations, lock takeovers, and stale-snapshot detection for observability.
- Run retention under the same lock, pruning `_tmp-*` workspaces older than `gc_grace_seconds` while never deleting the directory referenced by `CURRENT`.
- Load latest snapshot in Chat (ADR‑038; SPEC‑014) using the pointer and staleness digests for badge display.

## High-Level Architecture

```mermaid
graph TD
    A["App"] --> B["DocumentProcessor"]
    B --> C["IngestionPipeline"]
    C --> D["IngestionCache (DuckDBKVStore)"]
    B --> E["Vector Store (Qdrant)"]
    A --> F["Operational (SQLite, optional)"]
```

## Related Requirements

### Functional Requirements

- FR‑1: Persist and reuse processing cache entries across runs
- FR‑2: Store and retrieve vectors for similarity search

### Non-Functional Requirements

- NFR‑1: Fully offline/local
- NFR‑2: Minimal code and clean boundaries (no test seams)
- NFR‑3: Durable single-file cache

### Performance Requirements

- PR‑1: Cache lookup adds <5ms per document on local hardware
- PR‑2: Vector inserts throughput meets ingestion targets for batch indexing

### Integration Requirements

- IR‑1: Use LlamaIndex integrations for cache and vector stores
- IR‑2: Keep analytics/backup separate (ADR‑032/033)

## Design

### Architecture Overview

- Qdrant handles vectors; DuckDBKVStore handles cache via LlamaIndex IngestionCache
- Optional SQLite for ops data (e.g., session metadata) via app settings

### Implementation Details

In `src/core/processing.py` (illustrative wiring):

```python
from pathlib import Path
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.kvstore.duckdb import DuckDBKVStore

def build_cache(settings):
    cache_db = Path(settings.cache_dir) / "docmind.duckdb"
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    return IngestionCache(cache=DuckDBKVStore(db_path=str(cache_db)), collection="docmind_processing")
```

### Configuration

```env
DOCMIND_CACHE__DIR=./cache
DOCMIND_CACHE__FILENAME=docmind.duckdb
```

## Testing

```python
def test_cache_roundtrip(cache):
    key, value = ("k", {"v": 1})
    cache.put(key, value)
    assert cache.get(key) == value
```

### Observability

- configure_observability() (SPEC-012) provisions OpenTelemetry tracer/meter providers with OTLP (or console) exporters and registers LlamaIndex instrumentation.
- Snapshot, ingestion, and router workflows emit spans and metrics annotated with snapshot IDs, hashes, and GraphRAG metadata; console fallback keeps local-first deployments viable.
- JSONL telemetry events (`router_selected`, `export_performed`, `lock_takeover`, `snapshot_stale_detected`) serve as the structured logging contract for offline audits.

## Consequences

### Positive Outcomes

- Clean, minimal architecture; easy maintenance
- Fully offline; straightforward local deployment
- Clear separation of responsibilities

### Negative Consequences / Trade-offs

- Analytics not co-located with cache DB; separate DB if needed (ADR‑032)

### Ongoing Maintenance & Considerations

- Track LlamaIndex and DuckDB integration versions
- Monitor cache file growth; pair with retention/backup policies (ADR‑033)

### Dependencies

- Python: `llama-index>=0.13`, `llama-index-storage-kvstore-duckdb`, `llama-index-vector-stores-qdrant`, `duckdb`

## Changelog

- 1.5 (2025-09-16): Clarified SnapshotLock heartbeat/takeover semantics, retention discipline, and OpenTelemetry logging requirements.
- 1.4 (2025-09-16): Documented portalocker-based locking with TTL metadata, fallback locking, graph export telemetry, and removal of legacy `manifest.json`.
- 1.3 (2025-09-16): Documented tri-file manifest layout, `complete` flag semantics, CURRENT pointer resolution, timestamped graph exports, and telemetry expectations.
- 1.2 (2025-09-09): Added SnapshotManager and manifest hashing for GraphRAG; linked ADR‑038/SPEC‑014
- **1.1 (2025-09-03)**: DOCS - Added Related Decisions note referencing ADR-035 (application-level semantic cache)
- **1.0 (2025-09-02)**: Initial accepted version.
