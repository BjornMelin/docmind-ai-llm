---
spec: SPEC-021
title: Release Readiness v1 — Ship Plan, Work Packages, and Quality Gates
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-SEC-001: Default egress disabled; only local endpoints allowed unless explicitly configured.
  - NFR-MAINT-002: Pylint score ≥9.5; Ruff passes.
  - NFR-MAINT-003: No placeholder APIs; docs/specs/RTM must match code.
  - NFR-PORT-003: Docker/compose artifacts run and are reproducible from uv.lock.
related_adrs:
  - "ADR-041"
  - "ADR-042"
  - "ADR-043"
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
  - "ADR-054"
---

## Objective

Define the **minimum complete set of work packages** required to ship the first finished DocMind AI release with:

- runnable local Streamlit app
- runnable container artifacts
- offline-first posture preserved
- documentation + RTM consistency restored
- no TODO/NotImplemented placeholders left in production modules

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
| 03 | Chat persistence (SimpleChatStore JSON) | ADR-043 | SPEC-024 | `docs/developers/prompts/prompt-024-chat-persistence-simplechatstore.md` |
| 04 | Keyword tool: sparse-only Qdrant retriever | ADR-044 | SPEC-025 | `docs/developers/prompts/prompt-025-keyword-tool-sparse-only.md` |
| 05 | Ingestion API + legacy facade cleanup (`src.utils.document`) | ADR-045 | SPEC-026 | `docs/developers/prompts/prompt-026-ingestion-api-facade.md` |
| 06 | Remove legacy `src/main.py` entrypoint | ADR-046 | SPEC-027 | `docs/developers/prompts/prompt-027-remove-legacy-main-entrypoint.md` |
| 07 | Safe logging policy (remove PII redactor stub) | ADR-047 | SPEC-028 | `docs/developers/prompts/prompt-028-safe-logging-no-pii-redactor.md` |
| 08 | Docs consistency pass (spec drift + ADR number backfill) | ADR-048 | SPEC-029 | `docs/developers/prompts/prompt-029-docs-consistency-pass.md` |
| 09 | Multimodal helper cleanup (remove TODO; clarify test-only helper) | ADR-049 | SPEC-030 | `docs/developers/prompts/prompt-030-multimodal-helper-cleanup.md` |
| 10 | Config discipline: remove `os.getenv` sprawl; fix ADR-XXX marker | ADR-050 | SPEC-031 | `docs/developers/prompts/prompt-031-config-discipline-env-bridges.md` |
| 11 | Documents snapshot service boundary (extract rebuild/export) | ADR-051 | SPEC-032 | `docs/developers/prompts/prompt-032-documents-snapshot-service-boundary.md` |
| 12 | Background ingestion & snapshot jobs (progress + cancel) | ADR-052 | SPEC-033 | `docs/developers/prompts/prompt-033-background-ingestion-jobs.md` |
| 13 | Analytics page hardening (DuckDB + telemetry parsing) | ADR-053 | SPEC-034 | `docs/developers/prompts/prompt-034-analytics-page-hardening.md` |
| 14 | Config surface pruning (remove unused/no-op knobs) | ADR-054 | SPEC-035 | `docs/developers/prompts/prompt-035-config-surface-pruning-unused-knobs.md` |

## Execution order and dependencies

Recommended dependency order (minimizes churn):

1. WP01 → fixes UI security sink and prevents invalid persisted config.
2. WP05/WP04 → stabilizes ingestion + retrieval tool surfaces used by agents/docs.
3. WP11 → moves snapshot rebuild/export into a testable service boundary (enables WP12).
4. WP12 → adds background ingestion UX on top of the stable service boundary.
5. WP03 → adds persistence UX (depends on stable settings/data_dir).
6. WP06/WP07/WP10/WP14 → eliminates legacy debt, strengthens config discipline, and removes unused config knobs.
7. WP13 → hardens Analytics and telemetry parsing (best after WP10 if telemetry path moves into settings).
8. WP09/WP08 → removes remaining TODOs and aligns docs/RTM.
9. WP02 (Docker/Compose) → can be done anytime, but should be validated before release.

## Release quality gates (commands)

All packages MUST pass these commands before being marked complete:

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run pylint --fail-under=9.5 src/ tests/ scripts/
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

## Rollback plan

Release rollback is config-first:

- Settings UI changes: revert `.env` changes; relaunch Streamlit.
- Chat persistence: delete `data/chat/` artifacts (local only).
- Keyword tool: disable via env flag or remove tool registration.
- Docker: revert to prior Dockerfile/compose; local run remains unaffected.
