---
ADR: ADR-061
Title: Centralized spaCy NLP Subsystem (Runtime Selection + Ingestion Enrichment)
Status: Accepted
Version: 1.0
Date: 2026-01-16
Supersedes:
Superseded-by:
Related: ADR-024, ADR-030, ADR-058, SPEC-015
Tags: nlp, ingestion, spacy, gpu, streamlit, settings
References:
  - https://github.com/explosion/spaCy/releases
  - https://pypi.org/project/spacy/
  - https://spacy.io/usage
  - https://spacy.io/api/top-level
  - https://pypi.org/project/cupy-cuda12x/
---

## Description

We centralize all spaCy interactions into a single typed NLP subsystem with explicit
CPU/CUDA/Apple runtime selection, ingestion-time enrichment, and Streamlit-safe caching.

## Context

DocMind AI needs optional NLP enrichment (sentence segmentation + entities) that:

- is **offline-first** (no runtime downloads),
- runs on **CPU by default**, with optional acceleration on **NVIDIA CUDA** and **Apple Silicon**,
- avoids **fragile import-time side effects** (device selection must happen before model loads),
- is safe to integrate into Streamlit (reruns + caching),
- remains maintainable (one ownership boundary; schema-first outputs).

The prior codebase had fragmented spaCy usage via a bespoke manager module. This created
multiple issues:

- unclear ownership (spaCy imports/config scattered),
- caching behavior tied to module globals rather than UI/runtime boundaries,
- hard-to-test device selection behavior.

## Decision Drivers

- **Solution leverage (35%)**: Use spaCy’s supported top-level APIs (`prefer_gpu`, `require_gpu`,
  `require_cpu`) and LlamaIndex transforms rather than custom glue.
- **Application value (30%)**: Add useful, typed enrichment outputs and surface them in the UI.
- **Maintenance (25%)**: One module owns spaCy import/load/device selection; strict typing.
- **Adaptability (10%)**: Cross-platform behavior; settings-controlled toggles.
- **Offline-first / deterministic installs**: `uv.lock` is the single reproducibility artifact.

## Alternatives

- A: Ad-hoc `spacy.load()` at call sites — simple but creates duplicated config, repeated loads,
  and import-time ordering hazards.
- B: Import-time global singleton + eager model load — avoids repeated loads but breaks Streamlit
  rerun semantics, makes tests brittle, and risks implicit device selection.
- C: Centralized service with explicit settings + caching + typed outputs (Selected).

### Decision Framework

| Option | Solution leverage (35%) | Application value (30%) | Maintenance (25%) | Adaptability (10%) | Total | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| **C: Centralized service** | 9.5 | 9.2 | 9.4 | 9.0 | **9.33** | ✅ Selected |
| A: Ad-hoc call sites | 6.0 | 8.0 | 5.5 | 7.0 | 6.57 | Rejected |
| B: Import-time singleton | 6.5 | 8.3 | 6.0 | 6.0 | 6.79 | Rejected |

## Decision

We adopt a centralized spaCy subsystem:

- **Typed settings**: `src/nlp/settings.py::SpacyNlpSettings`
- **Runtime service**: `src/nlp/spacy_service.py::SpacyNlpService`
- **Ingestion integration**: `src/processing/nlp_enrichment.py::SpacyNlpEnrichmentTransform`
  wired into `src/processing/ingestion_pipeline.py::build_ingestion_pipeline`
- **Streamlit caching boundary**: `src/pages/02_documents.py::_get_spacy_service`

Device selection is applied **before** loading any pipeline and is controlled by:

- `SPACY_DEVICE=cpu|cuda|apple|auto`
- `SPACY_GPU_ID=<int>`

Missing spaCy models fail open by falling back to `spacy.blank("en")`. Explicit GPU requests
fail fast with a clear error.

## High-Level Architecture

```mermaid
flowchart LR
  A["Settings: DocMindSettings.spacy"] --> B["SpacyNlpService"]
  B --> C["spacy.require/prefer_* (device)"]
  C --> D["spacy.load(model) or spacy.blank('en')"]
  D --> E["SpacyNlpEnrichmentTransform"]
  E --> F["LlamaIndex Nodes metadata: docmind_nlp"]
  F --> G["Streamlit Documents page preview"]
```

## Dependency & Packaging Notes (CUDA vs Apple)

We keep `spacy==3.8.11` pinned in base dependencies for reproducibility.

spaCy 3.8.8+ switches its CLI dependency to `typer-slim`. In practice, some optional
GPU/server dependencies install `typer` (which shares the same `typer/` import path).
To avoid environment churn where uninstalling `typer` can remove files needed by
`typer-slim`, the project pins `typer==0.21.1` as a direct dependency.

For acceleration:

- **Apple Silicon**: `uv sync --extra apple` installs `spacy[apple]` on macOS arm64.
- **NVIDIA CUDA (12.x)**: `uv sync --extra gpu` installs `cupy-cuda12x>=13` on non-darwin.

Why not use `spacy[cuda12x]==3.8.11`?

- spaCy’s `cuda12x` extra pins `cupy-cuda12x<13`.
- This repo’s lock resolves `numpy==2.x` (required by the Apple stack and widely used across
  the project), but `cupy-cuda12x<13` is constrained to `numpy<1.29`, which is incompatible.
- `uv.lock` is a single-version lock per package, so the CUDA and Apple paths must agree on a
  compatible NumPy line.

We therefore rely on `cupy-cuda12x>=13`, which supports NumPy 2.x, while still enabling spaCy
GPU execution via Thinc.

## Testing

- Unit tests validate schema stability and the blank-model fallback:
  `tests/unit/nlp/test_spacy_service.py`.
- GPU tests are optional and guarded by `@pytest.mark.requires_gpu`.

## Consequences

### Positive Outcomes

- One ownership boundary for spaCy imports/config; fewer foot-guns.
- Streamlit-safe caching: no rerun-driven model reloads.
- Cross-platform and operator-friendly runtime selection (`cpu|cuda|apple|auto`).
- Typed, stable enrichment payload (`docmind_nlp`) usable in UI and downstream features.

### Negative Consequences / Trade-offs

- Ingestion performs extra work when NLP enrichment is enabled.
- GPU installs depend on local CUDA/driver correctness; explicit `SPACY_DEVICE=cuda` fails fast.
- CUDA extra deviates from spaCy’s published `cuda12x` constraint to remain compatible with the
  project’s NumPy 2 lock.

### Dependencies

- **Python**: `spacy==3.8.11` (base)
- **Apple**: `spacy[apple]` (macOS arm64 extra)
- **CUDA**: `cupy-cuda12x>=13` (non-darwin extra)
- **Removed**: `src/core/spacy_manager.py` (replaced by `src/nlp/*`)

## Changelog

- **1.0 (2026-01-16)**: Initial accepted version.
