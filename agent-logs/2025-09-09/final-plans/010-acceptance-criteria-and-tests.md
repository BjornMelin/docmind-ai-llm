# Acceptance Criteria and Test Plans

Date: 2025-09-09

## SPEC-008 — Multipage UI

### Acceptance

- Programmatic pages present and navigable
- Chat shows history and progressive output or full answer
- Chat passes `settings_override` with `router_engine` when present in session
- Documents page processes uploads with status + toast
- Analytics page shows disabled/empty or charts with data

Tests

- Integration tests for page presence and flows; mock coordinator and ingestion
- Unit test validates Chat router override mapping (`tests/unit/test_chat_router_override.py`)
- Run: `pytest -k test_ui` or system/integration suites as configured

## ADR-032 — Analytics

Acceptance

- Analytics DB created when enabled, writes succeed, retention prunes

Tests

- Unit test logs and prunes in temp DB
- Run: `pytest -k test_analytics`

## SPEC-010 — Evaluation Harness (BEIR + RAGAS)

Acceptance

- BEIR CLI computes NDCG@10, Recall@10, MRR and writes leaderboard
- RAGAS CLI computes selected metrics and writes leaderboard

Tests

- Smoke tests with mocks to verify leaderboard row creation
- Run: `pytest -k test_eval_cli_parsing`

## SPEC-013 — Model CLI

Acceptance

- CLI downloads default or added files and prints paths

Tests

- Unit test mocking `hf_hub_download`
- Run: `pytest -k test_models_pull_cli`

## SPEC-006 — GraphRAG

Acceptance

- Enabling GraphRAG yields non-empty Parquet exports (and JSONL fallback) under `data/graph/`

Tests

- Integration test verifies JSONL exists and is non-empty; Parquet is created when `pyarrow` is available
- Run: `pytest -k test_graph_exports`

## SPEC-012 — Observability + Security

Acceptance

- Telemetry includes required fields without sensitive data
- Egress disabled blocks non-allowlisted endpoints

Tests

- Verify JSONL structure and block behavior under disabled egress
- Run: `pytest -k telemetry or -k security`

## Gherkin Summary

See detailed scenarios in:

- 003-ui-multipage-impl.md
- 004-analytics-duckdb-impl.md
- 005-evaluation-harness-impl.md
- 006-model-cli-impl.md
- 007-graphrag-impl.md
- 008-observability-security-impl.md

## Quality Gates (uv)

Run all checks via uv (examples):

```bash
uv run pytest -q
uv run ruff format --check . && uv run ruff check .
uv run pylint src tools tests
uv run python scripts/run_quality_gates.py --ci
```
# Acceptance Criteria and Tests — GraphRAG Phase‑2 Additions

Date: 2025-09-09

## AC‑FR‑009 — GraphRAG Router, Persistence, and Exports

```gherkin
Scenario: Router composition and fallback
  Given GraphRAG is enabled and a graph exists
  When I query
  Then RouterQueryEngine SHALL include vector and graph tools
  And route to vector only if the graph is missing or unhealthy

Scenario: Snapshot manifest and staleness
  Given SnapshotManager created storage/<timestamp> with manifest.json
  And current corpus/config hashes differ
  When I open Chat
  Then a staleness badge SHALL be visible

Scenario: Exports
  Given a graph store and seeds
  When I export
  Then JSONL SHALL be written (one relation per line)
  And Parquet SHALL be written when pyarrow is available
```

## Test Inventory Updates

- Unit
  - tests/unit/agents/test_settings_override_router.py — router_engine present and used
  - tests/unit/retrieval/test_graph_helpers.py — traversal via get_rel_map; export JSONL correctness
  - tests/unit/persistence/test_snapshot_manager.py — atomic rename; manifest fields; lock behavior

- Integration
  - tests/integration/test_graphrag_exports.py — JSONL/Parquet conditional exports
  - tests/integration/test_ingest_router_flow.py — ingest → router tools composed

- E2E
  - tests/e2e/test_chat_graphrag_smoke.py — tiny corpus; router response with sources
