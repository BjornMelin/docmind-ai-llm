---
spec: SPEC-021
title: Release Readiness v1 — Ship Plan, Work Packages, and Quality Gates
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-SEC-001: Default egress disabled; only local endpoints allowed unless explicitly configured.
  - NFR-MAINT-002: Ruff/pyright pass (ruff enforces pylint-equivalent rules).
  - NFR-MAINT-003: No placeholder APIs; docs/specs/RTM must match code.
  - NFR-PORT-003: Docker/compose artifacts run and are reproducible from uv.lock.
related_adrs:
  - "ADR-023"
  - "ADR-033"
  - "ADR-035"
  - "ADR-041"
  - "ADR-042"
  - "ADR-058"
  - "ADR-044"
  - "ADR-045"
  - "ADR-046"
  - "ADR-047"
  - "ADR-048"
  - "ADR-049"
  - "ADR-050"
  - "ADR-051"
  - "ADR-052"
  - "ADR-053"
  - "ADR-055"
  - "ADR-056"
---

## Objective

Define the **complete set of work packages** required to ship the first finished DocMind AI release with:

- runnable local Streamlit app
- runnable container artifacts
- offline-first posture preserved
- documentation + RTM consistency restored
- no TODO/NotImplemented placeholders left in production modules
- advanced, opt-in capabilities shipped (semantic cache, analysis modes, backups) without violating local-first posture
- correctness hardening shipped (ops metadata store, cooperative deadline propagation) without adding new remote surfaces

This SPEC is the umbrella “release plan” and links to the individual ADR/SPEC/prompts for each package.

## Non-goals

- Introducing new RAG strategies beyond what is already implemented/spec’d
- Changing the core model stack (Torch/vLLM/Transformers pins)
- Adding cloud dependencies or enabling remote endpoints by default

## Work packages

Each work package MUST ship with:

- ADR (decision + alternatives + ≥9.0 decision table)
- SPEC (goals, design, security, tests, RTM updates)
- Implementation Prompt (atomic executor prompt)

| WP | Title | ADR | SPEC | Prompt |
| ---: | --- | --- | --- | --- |
| 01 | Settings UI hardening + safe provider badge | ADR-041 | SPEC-022 | `docs/developers/prompts/prompt-022-settings-ui-hardening.md` |
| 02 | Containerization hardening (Dockerfile + compose) | ADR-042 | SPEC-023 | `docs/developers/prompts/prompt-023-containerization-hardening.md` |
| 03 | Chat persistence + hybrid agentic memory (LangGraph SQLite) | ADR-058 | SPEC-041 | `docs/developers/prompts/implemented/prompt-041-chat-persistence-langgraph-sqlite-hybrid-memory.md` |
| 04 | Keyword tool: sparse-only Qdrant retriever | ADR-044 | SPEC-025 | `docs/developers/prompts/prompt-025-keyword-tool-sparse-only.md` |
| 05 | Ingestion API + legacy facade cleanup (`src.utils.document`) | ADR-045 | SPEC-026 | `docs/developers/prompts/prompt-026-ingestion-api-facade.md` |
| 06 | Remove legacy `src/main.py` entrypoint | ADR-046 | SPEC-027 | `docs/developers/prompts/prompt-027-remove-legacy-main-entrypoint.md` |
| 07 | Safe logging policy (remove PII redactor stub) | ADR-047 | SPEC-028 | `docs/developers/prompts/prompt-028-safe-logging-no-pii-redactor.md` |
| 08 | Docs consistency pass (spec drift + ADR number backfill) | ADR-048 | SPEC-029 | `docs/developers/prompts/prompt-029-docs-consistency-pass.md` |
| 09 | Multimodal helper cleanup (remove TODO; clarify test-only helper) | ADR-049 | SPEC-030 | `docs/developers/prompts/prompt-030-multimodal-helper-cleanup.md` |
| 10 | Config discipline: remove `os.getenv` sprawl; fix ADR-XXX marker; formalize hashing secret | ADR-050 | SPEC-031 | `docs/developers/prompts/prompt-031-config-discipline-env-bridges.md` |
| 11 | Documents snapshot service boundary (extract rebuild/export) | ADR-051 | SPEC-032 | `docs/developers/prompts/prompt-032-documents-snapshot-service-boundary.md` |
| 12 | Background ingestion & snapshot jobs (progress + cancel) | ADR-052 | SPEC-033 | `docs/developers/prompts/prompt-033-background-ingestion-jobs.md` |
| 13 | Analytics page hardening (DuckDB + telemetry parsing) | ADR-053 | SPEC-034 | `docs/developers/prompts/prompt-034-analytics-page-hardening.md` |
| 14 | Document analysis modes (auto/separate/combined) | ADR-023 | SPEC-036 | `docs/developers/prompts/prompt-036-document-analysis-modes.md` |
| 15 | Local backup & retention (snapshots + cache + Qdrant) | ADR-033 | SPEC-037 | `docs/developers/prompts/prompt-037-local-backup-and-retention.md` |
| 16 | Semantic response cache (Qdrant-backed, guardrailed) | ADR-035 | SPEC-038 | `docs/developers/prompts/prompt-038-semantic-cache-qdrant.md` |
| 17 | Operational metadata store (SQLite WAL) | ADR-055 | SPEC-039 | `docs/developers/prompts/prompt-039-operational-metadata-sqlite-wal.md` |
| 18 | Agent deadline propagation + router injection | ADR-056 | SPEC-040 | `docs/developers/prompts/prompt-040-agent-deadline-propagation-and-router-injection.md` |

All ADR/SPEC/prompt files referenced above are stored under `docs/` in this repo and are part of this release plan.

## Execution order and dependencies

Recommended dependency order (minimizes churn):

1. WP01 → fixes UI security sink and prevents invalid persisted config.
2. WP10 → stabilizes config discipline (reduce `os.getenv` sprawl; unify settings).
3. WP05/WP04 → stabilizes ingestion + retrieval tool surfaces used by agents/docs.
4. WP18 → restores retrieval tool contract and aligns per-call timeouts to the supervisor decision budget.
5. WP11 → moves snapshot rebuild/export into a testable service boundary (enables WP12).
6. WP17 → adds transactional ops metadata needed for background job reliability and restart recovery.
7. WP12 → adds background ingestion UX on top of the stable service boundary (writes to ops DB).
8. WP03 → ships chat persistence + time travel + hybrid memory UX (depends on stable settings/data_dir and stable supervisor wiring).
9. WP14 → adds analysis modes on top of stable retrieval + chat UX.
10. WP16 → adds semantic cache (optional; default-off; strict invalidation).
11. WP15 → adds local backup/retention once data layout is stable (uses snapshots/cache dirs).
12. WP06/WP07/WP13 → eliminates legacy entrypoints, strengthens logging safety, and hardens Analytics.
13. WP09/WP08 → removes remaining TODOs and aligns docs/RTM.
14. WP02 (Docker/Compose) → can be done anytime, but should be validated before release.

## Release quality gates (commands)

All packages MUST pass these commands before being marked complete:

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
uv run python scripts/run_quality_gates.py --ci --report
```

Containerization packages MUST also pass:

```bash
docker build -t docmind:dev .
docker compose config
docker compose up --build -d
```

## RTM updates

Work packages that change behavior MUST update:

- `docs/specs/traceability.md` with new/updated rows (Status = Planned → Implemented when done)
- any referenced SPEC/ADR cross-links that currently drift (see WP08)
- Consult the RTM format/schema in `docs/specs/traceability.md` (see “RTM Entry Format”).

## Rollback plan

Release rollback is config-first:

- Settings UI changes: revert `.env` changes; relaunch Streamlit.
- Chat persistence: delete `data/chat/` artifacts (local only).
- Keyword tool: disable via env flag or remove tool registration.
- Docker: revert to prior Dockerfile/compose; local run remains unaffected.
