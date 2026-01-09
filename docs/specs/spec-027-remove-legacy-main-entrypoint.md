---
spec: SPEC-027
title: Remove Legacy Entrypoint (`src/main.py`)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-PORT-001: Single definitive architecture; no prod/local forks.
  - NFR-MAINT-003: No placeholder/dead entrypoints.
related_adrs: ["ADR-046", "ADR-013", "ADR-024"]
---

## Objective

Remove the dead/legacy Python entrypoint (`src/main.py`) so that the only supported runtime entrypoint is the Streamlit multipage UI (`src/app.py`) and documented scripts.

## Non-goals

- Creating a new CLI for DocMind v1 (evaluation/model-pull CLIs already exist under `tools/` and `src/eval/`)
- Changing Streamlit navigation or UI routing

## Technical design

1. Delete `src/main.py`.

2. Remove `src/main.py` from any config references:

   - coverage omit list (`pyproject.toml`)
   - docs that mention running `python src/main.py`

3. Ensure README and scripts continue to use:

```bash
streamlit run src/app.py
```

## Security

- Removing `src/main.py` eliminates an import-time `.env` load path and any risk of logging raw model outputs from that script.

## Testing strategy

Verification steps:

- `rg "src/main.py" -n` returns no results (except in archived docs, if any remain intentionally).
- `uv run python -m compileall src` succeeds.
- standard quality gates still pass.

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- NFR-MAINT-003: “Remove legacy entrypoints”
  - Code: `src/main.py` (deleted), `pyproject.toml` (updated)
  - Tests: N/A
  - Verification: inspection + quality gates
  - Status: Planned → Implemented
