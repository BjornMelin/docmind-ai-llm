# Implementation Plan

## New Files

- `src/persistence/snapshot.py`
  - `SnapshotManager`: `begin_snapshot`, `persist_vector_index`, `persist_graph_store`, `write_manifest`, `finalize_snapshot`, `cleanup_tmp`
  - Hash helpers: `corpus_hash(paths)`, `config_hash(settings)`
- `src/retrieval/router_factory.py`
  - `build_router_engine(vector_index, pg_index, settings)` → `RouterQueryEngine`
  - Tool builders, selector chooser, health checks

## Updated Files

- `src/pages/02_documents.py`
  - Add GraphRAG toggle; optional `PropertyGraphIndex` build; call `SnapshotManager`; export buttons (seed cap=32); snapshot utilities (Rebuild button); place objects in `st.session_state`
- `src/pages/01_chat.py`
  - `settings_override` forwards `router_engine` + indices; default router when graph present; staleness badge with explicit rebuild copy
- `src/retrieval/graph_config.py`
  - Replace internals with library-first helpers: `traverse_relations`, `export_rel_map_jsonl/parquet`, `create_graph_rag_components`
- `src/retrieval/__init__.py`
  - Export only valid helpers + new factory
- `CHANGELOG.md`
  - Add Phase 2 bullets

## Tests

- `tests/unit/agents/test_settings_override_router.py`
- `tests/unit/retrieval/test_graph_helpers.py`
- `tests/unit/persistence/test_snapshot_manager.py`
- `tests/integration/test_graphrag_exports.py`
- `tests/integration/test_ingest_router_flow.py`
- `tests/e2e/test_chat_graphrag_smoke.py`

## Config/Flags

- `graphrag.enabled` (default false)
- `graphrag.subretrievers` (default false)
- `graphrag.default_path_depth` (default 1)

## Quality Gates

- `ruff/pylint`: clean
- All new tests pass; avoid modifying unrelated tests
