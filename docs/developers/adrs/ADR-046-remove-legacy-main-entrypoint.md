---
ADR: 046
Title: Remove Legacy `src/main.py` Entrypoint
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 012, 024, 015
Tags: cleanup, entrypoint, streamlit, legacy
References:
  - https://docs.streamlit.io/develop/api-reference/cli/run
---

## Description

Delete the legacy `src/main.py` entrypoint and standardize on the supported Streamlit entrypoint (`streamlit run src/app.py`) plus repo scripts.

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

## Decision

We will:

1. Delete `src/main.py`.

2. Remove any references in configuration and documentation (e.g., coverage omit list, Docker entrypoints).

3. Ensure the canonical run path remains:

```bash
streamlit run src/app.py
```

## Consequences

### Positive Outcomes

- Eliminates a misleading entrypoint and dead code path.
- Reduces security risk from accidental import-time `.env` loading and logging of outputs.
- Clarifies the supported execution model (Streamlit multipage).

### Trade-offs

- None for supported app usage; any custom users relying on `python src/main.py` must migrate (explicitly unsupported for v1).
