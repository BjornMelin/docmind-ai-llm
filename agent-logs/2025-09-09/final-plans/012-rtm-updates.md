# RTM Updates — Final Mapping

Date: 2025-09-09

## Corrections

- [x] FR-010 Streamlit multipage → Implemented
- [x] FR-009 GraphRAG PropertyGraphIndex → Implemented (exports/toggles/tests)
- Add rows for evaluation harness (BEIR/RAGAS) and model pull CLI

## New/Updated Rows (examples)

- FR-EVAL-001 — Offline BEIR metrics
  - ADR/SPEC: SPEC-010, ADR-039
  - Code: `tools/eval/run_beir.py`
  - Tests: CLI smoke with mocks
  - Status: Implemented (post-merge)

- FR-EVAL-002 — Offline RAGAS metrics
  - ADR/SPEC: SPEC-010, ADR-039
  - Code: `tools/eval/run_ragas.py`
  - Tests: CLI smoke with mocks
  - Status: Implemented (post-merge)

- FR-PKG-001 — Model pre-download CLI
  - ADR/SPEC: SPEC-013, ADR-040
  - Code: `tools/models/pull.py`
  - Tests: unit mocking `hf_hub_download`
  - Status: Implemented (post-merge)

- FR-UI-001 — Programmatic multipage navigation
  - ADR/SPEC: SPEC-008, ADR-012/016/013
  - Code: `src/app.py`, `src/pages/*`
  - Tests: integration UI smoke
  - Status: Implemented (post-merge)

- FR-RET-002 — Router engine override in Chat
  - ADR/SPEC: SPEC-008, ADR-003
  - Code: `src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/retrieval/router_factory.py`
  - Tests: unit `tests/unit/agents/test_settings_override_router.py`
  - Status: Implemented

- FR-009 — GraphRAG PropertyGraphIndex
  - ADR/SPEC: ADR‑019, ADR‑038; SPEC‑006, SPEC‑014
  - Code: `src/retrieval/graph_config.py`; `src/retrieval/router_factory.py`; `src/persistence/snapshot.py`; `src/pages/01_chat.py`; `src/pages/02_documents.py`
  - Tests: `tests/unit/agents/test_settings_override_router.py`; `tests/unit/retrieval/test_graph_helpers.py`; `tests/unit/persistence/test_snapshot_roundtrip.py`; `tests/unit/persistence/test_corpus_hash_relpaths.py`; `tests/integration/test_graphrag_exports.py`; `tests/integration/test_ingest_router_flow.py`; `tests/e2e/test_chat_graphrag_smoke.py`
  - Status: Implemented

## Documentation Touchpoints

- Update `docs/specs/traceability.md` accordingly (FR‑009 row updated)
- Cross-link to SPEC/ADR documents and code modules
