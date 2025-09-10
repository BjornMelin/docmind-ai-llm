# File Impact Map

## Create

- `src/persistence/snapshot.py`
- `src/retrieval/router_factory.py`

## Update

- `src/pages/02_documents.py`
- `src/pages/01_chat.py`
- `src/retrieval/graph_config.py`
- `src/retrieval/__init__.py`
- `CHANGELOG.md`
- `agent-logs/2025-09-09/graphrag/gr-suite/*` (refined UI wiring, persistence notes, seed cap, rebuild button)

## Tests

- `tests/unit/agents/test_settings_override_router.py`
- `tests/unit/retrieval/test_graph_helpers.py`
- `tests/unit/persistence/test_snapshot_manager.py`
- `tests/integration/test_graphrag_exports.py`
- `tests/integration/test_ingest_router_flow.py`
- `tests/e2e/test_chat_graphrag_smoke.py`

## Docs/Specs

- `specs/adr/ADR-019-graphrag-persistence-router.md`
- `specs/requirements/FR-009-GraphRAG-updates.md`
- `agent-logs/2025-09-09/graphrag/gr-suite/*` (this suite)

## Delete (if present and legacy)

- Any index monkey-patching helpers attached directly to PropertyGraphIndex
- Deprecated internal helpers relying on get_nodes/get_edges
