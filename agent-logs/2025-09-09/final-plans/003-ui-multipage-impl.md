# SPEC-008 — Streamlit Multipage UI Implementation

Date: 2025-09-09

## Purpose

Implement a clean, programmatic Streamlit UI using `st.Page` + `st.navigation` with four pages: Chat, Documents, Analytics, Settings. Use native chat components and a streaming fallback, a form-based ingestion flow with `st.status` and `st.toast`, and link the Analytics page to the local DuckDB metrics (ADR-032). Ensure ingestion indexes vectors into Qdrant and, after ingestion, builds a router engine that Chat uses via `settings_override`.

## Prerequisites

- Streamlit `>=1.48.0`
- Existing Settings (`src/config/settings.py`) and provider badge (`src/ui/components/provider_badge.py`)
- JSONL telemetry in `src/utils/telemetry.py`
- ADR-032 analytics manager (see 004-analytics-duckdb-impl.md)

## Files to Create/Update (Checklist)

- [x] Update: `src/app.py` (programmatic pages and navigation)
- [x] Create: `src/pages/01_chat.py` (chat UI)
- [x] Create: `src/pages/02_documents.py` (ingestion UI)
- [x] Create: `src/pages/03_analytics.py` (analytics UI)
- [x] Create: `src/ui/ingest_adapter.py` (thin ingestion adapter)

Code references: see final-plans/011-code-snippets.md (Sections 1–5)

## Imports and Libraries

- Libraries: `streamlit`, `time` (for stream fallback), optionally `asyncio`
- App: `src.agents.coordinator.MultiAgentCoordinator`, `src.config.settings.settings`, `src.ui.components.provider_badge.provider_badge`
- Ingestion adapter: `src.ui.ingest_adapter.ingest_files`
- Router: `src.retrieval.router_factory.build_router_engine`

Example imports for pages:

```python
import streamlit as st
import time
from src.config.settings import settings
from src.ui.components.provider_badge import provider_badge
```

## Step-by-Step Implementation

1) Entry point `src/app.py`

- Set page config and define 4 pages:
  - Chat → `src/pages/01_chat.py`
  - Documents → `src/pages/02_documents.py`
  - Analytics → `src/pages/03_analytics.py`
  - Settings → `src/pages/04_settings.py` (already present)
- Run navigation: `st.navigation([chat, docs, analytics, settings]).run()`

Refactor/Deletion in `src/app.py` (No Backwards Compatibility):
- Remove all monolithic UI logic (chat, ingestion, analytics) from this file; those now live in `src/pages/*`.
- Remove hardware detection banners and startup summaries here; if desired, move a minimal provider badge to Chat page.
- Remove calls to `build_reranker_controls` here; if kept, render it in the Chat page sidebar as read-only context.
- Remove any direct Qdrant or ingestion pipeline orchestration in this file to avoid heavy imports at app import time.
- Keep only the minimal page definitions and navigation run line.

2) Chat Page `src/pages/01_chat.py`

- Initialize `st.session_state["messages"]` as list of `{"role","content"}`.
- Render message history with `st.chat_message`.
- Display provider badge at top for clarity: `provider_badge(settings)`.
- Accept input via `st.chat_input`.
- On submit:
  - Append user message to session state and render.
  - Build or fetch the agent coordinator (`MultiAgentCoordinator`).
  - Compute `settings_override` from session if present: `{ "router_engine": st.session_state.router_engine }`.
  - Invoke `process_query` with `settings_override` to enable adaptive routing.
  - Streaming fallback:
    - If no native stream API, define a generator that yields ~40 char chunks (optional tiny sleeps) and pass it to `st.write_stream`.
    - Otherwise write the full answer.
  - Append assistant message to session state.

See streaming adapter variants in final-plans/011-code-snippets.md (Section 10).

3) Documents Page `src/pages/02_documents.py`

- Use `st.form` to batch the uploader and options.
- Provide a checkbox “Enable GraphRAG”.
- On submit:
  - If no files, show `st.warning`.
  - Else, show `st.status("Ingesting…")`, call `ingest_files(files, enable_graphrag)`, then mark complete and `st.toast`.
  - After success, create a `VectorStoreIndex` from Qdrant, initialize a server‑hybrid retriever via `ServerHybridRetriever`, then call `build_router_engine(vector_index, graph_index, settings)` and store the returned router in `st.session_state.router_engine` for Chat.
  - On exception: mark error and `st.error`.

4) Analytics Page `src/pages/03_analytics.py`

- Gate on `settings.analytics_enabled`; if disabled, show info and stop.
- Resolve `db_path` to `settings.analytics_db_path` or `data/analytics/analytics.duckdb`.
- If DB absent, `st.warning` and stop.
- Open DuckDB connection and query:
  - Strategy counts (bar chart)
  - Daily avg latency (line chart)
  - Success distribution (bar chart)

See chart queries in final-plans/011-code-snippets.md (Section 5).

5) Ingestion Adapter `src/ui/ingest_adapter.py`

- Implement `ingest_files(files, enable_graphrag) -> int`:
  - Save uploaded files to `settings.data_dir / "uploads"`.
  - Construct `DocumentProcessor` and call `process_document_async` per file (use `asyncio.run` helper). Count successes.
  - Convert processed elements to LlamaIndex `Document` and index into Qdrant via `VectorStoreIndex.from_documents(...)` using a `create_vector_store(...)` (hybrid enabled).
  - If `enable_graphrag` is true, build a `PropertyGraphIndex` from new docs and export Parquet + JSONL triplets under `data/graph/`.

See ingestion adapter snippet in final-plans/011-code-snippets.md (Section 12).

## Acceptance Criteria

- Navigation shows and switches across Chat, Documents, Analytics, and Settings.
- Chat displays history and renders assistant responses; streaming fallback behaves as progressive output.
- Documents page batches ingestion and displays status/toast; errors are surfaced clearly.
- Analytics page shows disabled/empty states or plots when data exists.

Gherkin:

```gherkin
Feature: Programmatic multipage UI
  Scenario: Navigate between pages
    Given the app is running
    When I click on Chat, Documents, Analytics, Settings
    Then each page renders without error

  Scenario: Chat streaming fallback
    Given the Chat page
    When I submit a prompt
    Then the assistant response is displayed progressively or fully
    And the message history updates

  Scenario: Document ingestion
    Given the Documents page
    When I upload files and submit
    Then a status panel displays progress
    And a success toast appears on completion

  Scenario: Analytics
    Given analytics is enabled and metrics exist
    When I open Analytics
    Then charts are displayed for strategy counts, latency, and success
```

## Testing and Notes

- Integration tests:
  - Confirm pages load and Chat form works with a mocked coordinator.
  - Confirm status/toast usage; you can assert presence via Streamlit testing harness or snapshot.
- Performance:
  - Aim for first-response under ~100ms with stream fallback (best-effort).
- Security:
  - Ensure no external egress from UI by default; any outbound endpoints must remain on allowlist.

## Cross-Links

- Analytics DB implementation: 004-analytics-duckdb-impl.md
- GraphRAG flow and exports: 007-graphrag-impl.md
- Code snippets for pages and streaming: 011-code-snippets.md (Sections 1–3, 10, 12)

## No Backwards Compatibility

- Replace the monolithic `src/app.py` UI with programmatic pages; delete any legacy UI functions or blocks in `app.py` that duplicate page logic.
- Update all imports and references to new page modules; remove any dead imports.
