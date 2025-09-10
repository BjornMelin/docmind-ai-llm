# 004 — SPEC/ADR Update Notes

Update SPEC‑014 (Index Persistence Snapshots):
- Clarify graph store persistence to persist_dir; loading via from_persist_dir.
- Add schema_version and persist_format_version to manifest; include versions: app, llama_index, qdrant_client, embed_model.
- Document corpus_hash relpath normalization (POSIX); add base_dir guidance.

Update SPEC‑006 (GraphRAG):
- Default graph traversal depth=1 via RetrieverQueryEngine wrapper; configurable via settings.
- Sub‑retrievers optional flag (graphrag.subretrievers) documented; default off.
- Export schema: subject, relation (preserve label when available), object, depth, path_id, source_ids.

Update ADR‑038:
- Chat autoload policy: load latest non‑stale snapshot by default; pinning and ignore modes available.
- Router unification: deprecate adaptive engine; use router_factory for composition. Provide migration notes.

