# Task Checklist — Hybrid Retrieval (Qdrant)

This checklist aggregates ALL tasks and subtasks from: 005–010, 012–014.
Update this file as tasks are completed during implementation.

## Schema & Storage (`src/utils/storage.py`)

- [x] Idempotent ensure of collection with named vectors
  - [x] Add/verify `text-dense` (COSINE, correct dim)
  - [x] Add/verify `text-sparse` (SparseVectorParams with IDF modifier)
  - [x] Startup log: created vs already exists; warn if IDF unsupported

## Retriever (`src/retrieval/hybrid.py`)

- [x] Implement ServerHybridRetriever (single path)
  - [x] Build Prefetch: sparse using `text-sparse` with configured limit
  - [x] Build Prefetch: dense using `text-dense` with configured limit
  - [x] Fusion selection: RRF default; DBSF if `RETRIEVAL_DBSF_ENABLED=true`
  - [x] Call `query_points` and capture results with payloads
  - [x] Dedup by `page_id` after fusion, before truncation; keep highest fused score
  - [x] Emit telemetry: `retrieval.fusion_mode`, `retrieval.prefetch_dense_limit`, `retrieval.prefetch_sparse_limit`, `retrieval.fused_limit`, `retrieval.return_count`, `retrieval.latency_ms`, `retrieval.sparse_fallback`, `dedup.*`

## Config (src/config/settings.py)

- [x] Add envs: `PREFETCH_DENSE_LIMIT`, `PREFETCH_SPARSE_LIMIT`, `RETRIEVAL_DBSF_ENABLED`, `TELEMETRY_ENABLED`
- [x] Validate/parse envs; log active fusion mode on startup

## Observability (src/observability/telemetry.py or equivalent)

- [x] Implement structured telemetry emission (PII-safe, bounded size)
- [x] Support sampling for production

## Removals

- [x] Remove client-side fusion codepaths
- [x] Remove UI fusion toggles
- [x] Delete or rewrite legacy tests relying on client-side fusion

## Tests (tests/retrieval/)

- [x] RRF suite: `test_qdrant_prefetch_rrf.py`
- [x] DBSF suite: `test_qdrant_prefetch_dbsf.py`
- [x] Dedup suite: `test_qdrant_dedup_before_limit.py`
- [x] Telemetry suite: `test_qdrant_telemetry.py`
- [x] Deterministic fixtures for vectors and payloads

## Docs & Governance

- [x] SPEC‑004 (`docs/specs/spec-004-hybrid-retrieval.md`) updates applied (see `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/005-spec-004-updates.md`)
- [x] Requirements & RTM updated (see `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/006-reqs-rtm-updates.md`)
  - [x] `docs/specs/requirements.md` updates
  - [x] `docs/specs/traceability.md` updated
- [x] ADR‑024/031 updated (see `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/007-adr-updates.md`)
  - [x] `docs/developers/adrs/ADR-024-configuration-architecture.md` updated
  - [x] `docs/developers/adrs/ADR-031-local-first-persistence-architecture.md` updated
- [x] Docs merge plan executed (see `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/013-docs-merge-and-integration-plan.md`)
- [x] `merge-matrix.csv` updated
- [x] Ensure user documentation mentions activation via `settings.retrieval.enable_server_hybrid`

## Change Log

- [x] Add Unreleased notes (see `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/012-change-log-notes.md`)

## Quality Gates

- [x] ruff format .
- [x] ruff check . --fix
- [x] pylint --fail-under=9.5 on `src/` and `tests/`
- [x] pytest -q (or `uv run python scripts/run_tests.py`)

## Cross-References

- Index: `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/000-index.md`
- Final Plan: `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/012-final-implementation-plan.md`
- Implementation Prompt: `agent-logs/2025-09-09/research/hybrid-retrieval-qdrant/014-IMPLEMENTATION-PROMPT.md`
