---
spec: SPEC-013
title: Packaging: Model Pre-download with huggingface_hub and Integrity Checks
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-PKG-001: Provide CLI to pre-download models to cache.
  - NFR-PORT-002: Cross-platform paths and cache env overrides.
related_adrs: ["ADR-010"]
---


## Objective

Provide a CLI tool to pre-download text and image embedding models and verify file hashes using `huggingface_hub`.

## Libraries and Imports

```python
from huggingface_hub import hf_hub_download
```

## File Operations

### CREATE

- `tools/models/pull.py`: CLI accepting model ids and target cache dir.

### Model IDs (default set)

- Text embeddings: `BAAI/bge-m3`
- Text reranker: `BAAI/bge-reranker-v2-m3`
- Image/text (multimodal): `google/siglip-base-patch16-224`
- Sparse (BM42): `Qdrant/bm42-all-minilm-l6-v2-attentions`

Offline flags to set before runtime: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Acceptance Criteria

```gherkin
Feature: Pre-download
  Scenario: Download bge-m3
    When I run the CLI with model BAAI/bge-m3
    Then weights SHALL exist under the HF cache directory
```

## References

- huggingface_hub file_download docs.

## Offline Mode & Extras

- The CLI MUST support offline operation via `HF_HUB_OFFLINE=1` and deterministic tests with strict mocks.
- Optional extras MAY provide Parquet support (e.g., `[parquet]`); when Parquet is requested without `pyarrow`, the CLI MUST issue a clear warning and fall back to JSONL.
