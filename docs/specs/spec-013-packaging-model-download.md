---
spec: SPEC-013
title: Packaging: Model Pre-download with huggingface_hub and Integrity Checks
version: 1.1.0
date: 2026-07-11
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-PKG-001: Provide CLI to pre-download models to cache.
  - NFR-PORT-002: Cross-platform paths and cache env overrides.
related_adrs: ["ADR-040"]
---


## Objective

Provide a CLI tool to pre-download complete text and image embedding snapshots
at repository-owned revisions through `huggingface_hub`. Parser artifacts use
separate app-owned manifests with exact size and SHA-256 verification.

## Libraries and Imports

```python
from huggingface_hub import hf_hub_download
```

## File Operations

### CREATE

- `tools/models/pull.py`: CLI accepting model ids and target cache dir.

### Model IDs (default set)

- Text embeddings: `BAAI/bge-m3`
- Image/text (multimodal): `google/siglip-base-patch16-224`

`--all` downloads complete pinned snapshots for those two models. Parser
artifacts use `--parser-defaults` and their app-owned manifests. Text reranking
and sparse encoding are optional, fail-open stages; the CLI does not pretend
that one detached weight file is a runnable local model.

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
