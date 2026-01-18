---
prompt: PROMPT-045
title: NLP Enrichment (spaCy 3.8.11) — Runtime Selection + Ingestion Transform + UI Preview
status: Completed
date: 2026-01-16
version: 1.0
related_adrs: ["ADR-061"]
related_specs: ["SPEC-015"]
---

**Purpose:** Ship a centralized spaCy subsystem for optional ingestion-time enrichment (sentences + entities) with cross-platform device selection (`cpu|cuda|apple|auto`) and Streamlit-safe caching.

**Source of truth:** `docs/developers/adrs/ADR-061-spacy-nlp-subsystem.md` + `docs/specs/spec-015-nlp-enrichment-spacy.md`.

## Scope

- Add typed spaCy settings and a single runtime service that owns:
  - device selection (before model load)
  - `spacy.load()` vs blank fallback
  - schema-first outputs (sentences + entities)
- Wire enrichment into ingestion as a transform that writes `node.metadata["docmind_nlp"]`.
- Surface a lightweight preview in the Documents UI with `st.cache_resource`.
- Add CPU-path unit tests and optional GPU tests guarded by markers/env.

## Key files

- Settings: `src/nlp/settings.py`
- Runtime: `src/nlp/spacy_service.py`
- Transform: `src/processing/nlp_enrichment.py`
- Wiring: `src/processing/ingestion_pipeline.py`
- UI preview: `src/pages/02_documents.py`
- Tests: `tests/unit/nlp/test_spacy_service.py`

## Installation (cross-platform)

CPU (default):

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

NVIDIA CUDA (Linux/Windows):

```bash
uv sync --extra gpu
uv run python -m spacy download en_core_web_sm
export SPACY_DEVICE=auto  # or cuda
export SPACY_GPU_ID=0
```

Apple Silicon (macOS arm64):

```bash
uv sync --extra apple
uv run python -m spacy download en_core_web_sm
export SPACY_DEVICE=auto  # or apple
```

## Verification

```bash
uv run python -c "import spacy; print(spacy.__version__)"
uv run python -m spacy info
uv run ruff format .
uv run ruff check .
uv run pyright
uv run python scripts/run_tests.py --fast
```
