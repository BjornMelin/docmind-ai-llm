---
spec: SPEC-015
title: NLP Enrichment (spaCy 3.8.11) — Runtime Selection + Ingestion Wiring
version: 1.0.0
date: 2026-01-16
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-ING-002: Canonical nodes with safe metadata.
  - NFR-MAINT-001: Library-first; avoid bespoke layers.
  - NFR-SEC-002: No raw content in logs/telemetry.
related_adrs: ["ADR-024","ADR-013","ADR-030","ADR-058","ADR-061"]
---

## Objective

Add a **first-class NLP enrichment stage** to ingestion using **spaCy 3.8.11** with:

- CPU default (offline-first)
- NVIDIA CUDA acceleration on Linux/Windows when available
- Apple Silicon acceleration on macOS (arm64)
- Centralized runtime selection, caching, and typed outputs

## Architecture (repo truth)

### Central subsystem (single ownership boundary)

- Settings model: `src/nlp/settings.py::SpacyNlpSettings`
- Runtime + extraction: `src/nlp/spacy_service.py::SpacyNlpService`

### Ingestion wiring

- Transform component: `src/processing/nlp_enrichment.py::SpacyNlpEnrichmentTransform`
- Pipeline insertion: `src/processing/ingestion_pipeline.py::build_ingestion_pipeline`
  - runs **after chunking** and **before embeddings**
  - fail-open on missing models (blank fallback), but **fail-fast** when the
    operator explicitly requests `SPACY_DEVICE=cuda|apple` and the runtime
    cannot activate it

### UI surfacing

- Streamlit preview: `src/pages/02_documents.py` expander shows sample entities/sentences
- Streamlit caching: `src/pages/02_documents.py::_get_spacy_service` uses `st.cache_resource`

## Configuration (canonical env vars)

DocMind remains nested-settings first, but supports the required flat knobs.

### Preferred (DocMind-native)

```bash
DOCMIND_SPACY__ENABLED=true
DOCMIND_SPACY__MODEL=en_core_web_sm
DOCMIND_SPACY__DEVICE=auto   # cpu|cuda|apple|auto
DOCMIND_SPACY__GPU_ID=0
DOCMIND_SPACY__DISABLE_PIPES='["parser"]'  # JSON array
DOCMIND_SPACY__BATCH_SIZE=32
DOCMIND_SPACY__N_PROCESS=1
```

### Supported (operator-friendly)

These are bridged into `DOCMIND_SPACY__*` at startup in `src/config/settings.py`:

```bash
SPACY_ENABLED=true
SPACY_MODEL=en_core_web_sm
SPACY_DEVICE=auto
SPACY_GPU_ID=0
SPACY_DISABLE_PIPES=parser,lemmatizer   # CSV also supported
SPACY_BATCH_SIZE=32
SPACY_N_PROCESS=1
```

## Output contract

### Node metadata (durable)

Each emitted node may include:

- `node.metadata["docmind_nlp"]` (dict) with:
  - `provider`: `"spacy"`
  - `model`: configured model name
  - `sentences`: list of `{start_char, end_char, text}`
  - `entities`: list of `{label, text, start_char, end_char, kb_id?, ent_id?}`

### Ingestion result metadata (PII-safe counters)

`src/processing/ingestion_pipeline.py` adds:

- `nlp.enabled` (bool)
- `nlp.enriched_nodes` (int)
- `nlp.entity_count` (int)

## Installation (cross-platform)

### CPU (Linux/Windows/macOS)

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

### NVIDIA CUDA (Linux/Windows)

Recommended (repo-supported):

```bash
uv sync --extra gpu
uv run python -m spacy download en_core_web_sm
```

Runtime:

```bash
export SPACY_DEVICE=cuda
export SPACY_GPU_ID=0
```

### Apple Silicon (macOS arm64)

```bash
uv sync --extra apple
uv run python -m spacy download en_core_web_sm
```

Runtime:

```bash
export SPACY_DEVICE=apple   # or auto
```

## Performance notes

- Enrichment uses `nlp.pipe()` with `batch_size` and optional `n_process`.
- `max_characters` prevents pathological large-node processing.
- Model caching:
  - Streamlit caches the service with `st.cache_resource`
  - the underlying pipeline load uses a process-local `lru_cache`.
