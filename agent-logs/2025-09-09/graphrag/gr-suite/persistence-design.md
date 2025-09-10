Persistence Design — SnapshotManager

Goals

- Atomic, consistent snapshots of vector index + property graph store
- Staleness detection via manifest
- Single-writer safety via lock

Layout

- storage/
  - _tmp-<uuid>/
    - vector/ (index files)
    - graph/ (SimpleGraphStore JSON)
    - manifest.json
  - <timestamp>/ (finalized)

Manifest Fields

- index_id, graph_store_type, vector_store_type
- corpus_hash, config_hash
- created_at (UTC ISO), lib_versions (llama_index, embeddings lib, app version)

Algorithm

1) begin_snapshot(): acquire file lock; create_tmp dir
2) persist_vector_index(): index.storage_context.persist(tmp/vector)
3) persist_graph_store(): store.persist(tmp/graph/graph_store.json)
4) write_manifest(): write hashes, versions
5) finalize_snapshot(): fsync and atomic rename_tmp → <timestamp>
6) cleanup_tmp(): on error, remove_tmp

Staleness Detection

- Compute corpus_hash from sorted (path, size, mtime)
- Compute config_hash from retrieval/chunking/embedding settings
- Compare to manifest in latest snapshot; if mismatch → staleness badge + rebuild option in Documents page (“Rebuild GraphRAG Snapshot”).

Concurrency

- filelock on storage/.lock with timeout (e.g., 120s)
- UI disables ingest controls while lock held

Notes

- Parquet export is separate and optional (not part of snapshot)
- For remote stores (Neo4j, Qdrant), persistence defers to backend; manifest still written
