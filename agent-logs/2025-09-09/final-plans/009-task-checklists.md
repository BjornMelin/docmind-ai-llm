# Implementation Task Checklists

Date: 2025-09-09

## SPEC-008 — Multipage UI

- [x] Update `src/app.py` to programmatic `st.Page` + `st.navigation`
- [x] Create `src/pages/01_chat.py` (chat UI + streaming fallback)
- [x] Create `src/pages/02_documents.py` (form + status + toast + GraphRAG checkbox)
- [x] Create `src/pages/03_analytics.py` (charts via DuckDB)
- [x] Create `src/ui/ingest_adapter.py` (ingest files, optional GraphRAG)
- [ ] Verify `src/pages/04_settings.py` continues to work
- [x] Remove monolithic logic from `src/app.py` (No Backwards Compatibility): chat/ingestion/analytics; keep only nav
- [x] Move any provider badge or read-only rerank info panels into Chat page sidebar as needed
- [x] Index newly ingested documents into Qdrant (adapter → VectorStoreIndex)
- [x] Build RouterQueryEngine after ingestion and store in session (`st.session_state.router_engine`)
- [x] Use router engine in Chat via `settings_override` when present

## ADR-032 — Analytics (DuckDB)

- [x] Create `src/core/analytics.py` (AnalyticsManager)
- [x] Wire query logging in `src/agents/coordinator.py`
- [ ] Wire ingestion logging in `src/processing/document_processor.py` (implemented in adapter for now)
- [x] Ensure `src/pages/03_analytics.py` charts load when enabled

## SPEC-010 + ADR-039 — Evaluation Harness

- [x] Create `tools/eval/run_beir.py`
- [x] Create `tools/eval/run_ragas.py`
- [x] Create `data/eval/README.md`
- [x] Add tests to assert `leaderboard.csv` row creation (mock heavy deps)

## SPEC-013 + ADR-040 — Model CLI

- [x] Create `tools/models/pull.py` minimal CLI
- [x] Add unit test mocking `hf_hub_download`
- [x] Delete `scripts/model_prep/predownload_models.py`
- [x] Remove any redundant model download scripts after CLI adoption

## SPEC-006 — GraphRAG

- [x] Add Parquet/JSONL export helpers to `src/retrieval/graph_config.py`
- [x] Add ingestion toggle path to build graph and export (adapter)
- [x] Add a tiny integration test that validates export files exist
- [ ] Create ADR‑038 (router + SnapshotManager) and SPEC‑014 (snapshots); add cross‑links in SPEC‑006/requirements/RTM

## SPEC-012 — Observability + Security

- [ ] Ensure retrieval telemetry fields are complete
- [ ] Add reranking path and timeout details to telemetry
- [ ] Audit allowlist/egress and redaction; add tests

## Documentation & RTM

- [ ] Update `docs/specs/traceability.md` statuses
- [ ] Update ADR statuses for 032/039/040
- [ ] Document prompts, flags, and acceptance notes in SPEC docs
