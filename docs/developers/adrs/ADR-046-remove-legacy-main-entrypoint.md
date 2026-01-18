---
ADR: 046
Title: Remove Legacy `src/main.py` Entrypoint
Status: Implemented
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: ADR-012, ADR-024, ADR-015
Tags: cleanup, entrypoint, streamlit, legacy
References:
  - https://docs.streamlit.io/develop/api-reference/cli/run
---

## Description

Delete the legacy `src/main.py` entrypoint and standardize on the supported Streamlit entrypoint (`uv run streamlit run app.py`) plus repo scripts.

## Context

`src/main.py`:

- is not referenced by the Streamlit UI
- performs import-time `.env` loading (`load_dotenv()`), violating config discipline
- contains placeholder/basic RAG logic and “phase 2” comments that do not reflect the actual architecture

Keeping it increases confusion and creates a latent “wrong entrypoint” trap, especially for Docker/compose and new contributors.

## Alternatives

- A: Keep `src/main.py` as-is (status quo)
- B: Move `src/main.py` into an archived folder (keeps dead code)
- C: Delete `src/main.py` and remove all references (Selected)

### Decision Framework (≥9.0)

| Option        | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| ------------- | ---------------: | ---------: | --------------: | ------: |
| **C: Delete** |               10 |         10 |               9 | **9.7** |
| B: Archive    |                7 |         10 |               8 |     8.2 |
| A: Keep       |                3 |          8 |               3 |     4.5 |

Scoring rubric:

- Complexity (40%): higher is simpler. Delete is minimal code; archive adds process overhead.
- Performance (30%): no runtime perf change expected; scores reflect negligible impact (low discriminative power when parity across options).
- Alignment (30%): alignment to Streamlit-only entrypoint and config discipline; delete is highest.

## Decision

We will:

1. Delete `src/main.py`.

2. Remove any references in configuration and documentation (e.g., coverage omit list, Docker entrypoints).

3. Ensure the canonical run path remains:

```bash
uv run streamlit run app.py
```

## Consequences

### Positive Outcomes

- Eliminates a misleading entrypoint and dead code path.
- Reduces security risk from accidental import-time `.env` loading and logging of outputs.
- Clarifies the supported execution model (Streamlit multipage).

### Trade-offs

- None for supported app usage; any custom users relying on `python src/main.py` must migrate (explicitly unsupported for v1).
- Migration guidance must be documented in v1 release notes: “`python src/main.py` removed; use `uv run streamlit run app.py` instead.”
- Downstream projects should search for hardcoded `src/main.py` references and update CI/deployment scripts.
