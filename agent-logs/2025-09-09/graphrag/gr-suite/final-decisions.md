# Final Decisions (D1–D6)

## D1 — Router Toolset

- **Decision**: Use RouterQueryEngine with tools [vector_query_engine, graph_query_engine] when a graph is present; otherwise vector-only.
- **Rationale**: Better coverage of relational queries; safe fallback.
- **Notes**: Graph engine built from PropertyGraphIndex.as_query_engine(include_text, similarity_top_k, path_depth=1).

## D2 — Persistence

- **Decision**: Implement SnapshotManager for atomic, versioned snapshots; include manifest.json with corpus_hash and config_hash; lock during writes.
- **Rationale**: Consistency, debuggability, staleness detection.
- **Notes**: StorageContext.persist for indices; SimpleGraphStore.persist for graph; temp dir + atomic rename.

## D3 — Sub-Retrievers

- **Decision**: Keep LLMSynonymRetriever + VectorContextRetriever behind feature flag (off by default).
- **Rationale**: Reduce complexity/flakiness; add later behind configuration.

## D4 — UI Defaults

- **Decision**: "Build GraphRAG (beta)" toggle off by default (configurable). Default chat strategy to router when pg_index present; allow switching.
- **Rationale**: Sensible defaults + user control; avoids surprise.

## D5 — Exports

- **Decision**: JSONL export required; Parquet export optional when pyarrow is installed.
- **Rationale**: Portable baseline; optional rich format without hard dep.

## D6 — Tests

- **Decision**: Unit (router override, graph helpers, snapshot manager), integration (exports, ingest→router), E2E smoke for chat with router.
- **Rationale**: Confidence with minimal footprint; avoid flaky nets/LLM calls.
