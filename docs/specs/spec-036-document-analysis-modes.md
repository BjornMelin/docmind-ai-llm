---
spec: SPEC-036
title: Document Analysis Modes (Separate / Combined / Auto)
version: 1.0.1
date: 2026-01-10
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-028: Users can run document analysis in Separate/Combined/Auto modes.
  - NFR-PERF-001: Chat p50 latency budgets remain within targets.
  - NFR-MAINT-003: No placeholder APIs; docs/specs/RTM match code.
related_adrs: ["ADR-023","ADR-013","ADR-016","ADR-011","ADR-052"]
---

## Goals

1. Provide a user-visible **Analysis Mode** selector with `auto | separate | combined`.
2. Implement a small **domain service** that routes analysis through existing retrieval + synthesis stacks.
3. Keep Streamlit pages thin: no business logic embedded in pages.
4. Provide deterministic, offline tests (MockLLM, no network).

## Non-goals

- Replacing the main Chat interaction model (this feature augments it).
- Adding new retrieval strategies beyond the existing router/hybrid/graph tools.
- Introducing async event-loop complexity in Streamlit.

## User Stories

1. As a user, I can select **Combined** mode to get a single holistic answer across my corpus.
2. As a user, I can select **Separate** mode to get one answer per document (shown in tabs), plus an optional compare/synthesis summary.
3. The system should support **Auto** mode that selects the best analysis mode based on document count and token budget.
4. Users need the ability to cancel long analyses (best-effort) without corrupting state or publishing partial artifacts.

## Technical Design

### Surface Area

- New domain module: `src/analysis/service.py`
- UI wiring:
  - Chat page (`src/pages/01_chat.py`): mode selector + results rendering
  - Optional Documents page (`src/pages/02_documents.py`): analysis “run on selection” entry point (future; only if needed)

### Mode Selection

- Inputs:
  - explicit UI selection (highest priority)
  - default from `settings.analysis.mode`
  - auto selection uses:
    - number of selected documents
    - approximate token budget (use existing token estimation helpers)
- Configuration:
  - `DOCMIND_ANALYSIS__MODE=auto|separate|combined`
  - `DOCMIND_ANALYSIS__MAX_WORKERS=<int>`

### Separate Mode (Map → Optional Reduce)

- Map step:
  - run analysis per document concurrently (ThreadPool)
  - each per-doc run uses existing retrieval and synthesis but applies a **doc filter**:
    - prefer server-side filtering (Qdrant payload filter on `doc_id`)
    - fall back to client-side grouping when doc filters are unavailable
- Reduce step (optional):
  - take per-doc outputs (bounded truncation) and produce a short comparison summary
  - reduce must not re-run full retrieval unless explicitly requested

### Combined Mode

- Single run using existing retrieval and synthesis across the selected corpus.

### Data Structures

Add minimal, typed result structures in `src/analysis/models.py`:

- `PerDocResult`: `{doc_id, doc_name, answer, citations, duration_ms}`
- `AnalysisResult`: `{mode, per_doc: list[PerDocResult], combined: str|None, reduce: str|None, warnings: list[str]}`

### Observability

Emit local JSONL telemetry events (and OTel spans when enabled):

- `analysis_mode_selected` with `{mode, auto_decision_reason, doc_count}`
- `analysis_completed` with `{mode, duration_ms, per_doc_count, success}`
- `analysis_cancelled` with `{mode, duration_ms}`

Never log raw prompts or document text.

### Security

- No unsafe HTML rendering for analysis output.
- Do not persist analysis outputs by default unless explicitly exported (ADR-022 / export system).
- Respect offline-first policy (no new network calls).

## Testing Strategy

### Unit

- `auto_select_mode()` chooses expected mode given thresholds.
- per-doc grouping/dedup logic works with synthetic node metadata.

### Integration (Streamlit AppTest)

- Keep AppTest integration as UI wiring-only: stub the analysis entry point
  (e.g. `src.analysis.service.run_analysis`) to return a deterministic
  `AnalysisResult` when validating rendering/selection logic.
- Chat page renders:
  - Combined mode produces a single output container.
  - Separate mode produces one tab per document and shows per-doc results.
- Cancellation triggers:
  - sets job state to cancelled
  - no partial results are published as “complete”
- Use `tests/helpers/apptest_utils.py` (`apptest_timeout_sec()`) for
  `default_timeout=`; override with `TEST_TIMEOUT=<seconds>` for slow runners.

### System (optional)

- Playwright MCP smoke flow for “Select mode → Run → See results”.

## Rollout / Migration

- Default mode remains `auto`.
- No data migrations.

## Performance Guardrails

- No Streamlit calls from worker threads.
- Bound parallelism by `settings.analysis.max_workers`.
- Reduce step must cap input size (token limit).

## RTM Updates

`FR-028` is tracked in `docs/specs/traceability.md` with code + test coverage.
