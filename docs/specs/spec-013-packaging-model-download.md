---
spec: SPEC-013
title: Packaging: Model Pre-download with huggingface_hub and Integrity Checks
version: 1.4.0
date: 2026-07-16
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-PKG-001: Provide CLI to pre-download models to cache.
  - NFR-PORT-002: Cross-platform paths and cache env overrides.
related_adrs: ["ADR-040"]
---


## Objective

Provide a CLI tool to pre-download complete text and image embedding snapshots
and the default text reranker at repository-owned revisions through
`huggingface_hub`. The Docling layout bundle uses one app-owned manifest with
exact size and SHA-256 verification. RapidOCR's locked wheel owns its model
files and checksums.

## Libraries and Imports

```python
from huggingface_hub import hf_hub_download
```

## File Operations

### CREATE

- `tools/models/pull.py`: CLI accepting model ids and target cache dir.

### Model IDs (default set)

- Text embeddings: `BAAI/bge-m3`
- Sparse text: `Qdrant/bm42-all-minilm-l6-v2-attentions` from
  `Qdrant/all_miniLM_L6_v2_with_attentions`
- Text reranking: `BAAI/bge-reranker-v2-m3`
- Image/text (multimodal): `google/siglip-base-patch16-224`

`--all` downloads complete pinned snapshots for those four models. The Docling
layout bundle uses `--parser-defaults` and its app-owned manifest. RapidOCR is
not a downloader target because its locked wheel supplies the runnable models.
The BGE reranker manifest contains every Transformers file required by the
runtime CrossEncoder. The BM42 manifest contains FastEmbed's ONNX model,
tokenizer, and stopword files. The model-free BM25 fallback needs no snapshot.

When `--cache_dir` or `--parser-cache-dir` is omitted for requested work, the
CLI MUST run the canonical settings bootstrap once, including `.env`, and use
`embedding.cache_folder` or `parsing.model_cache_dir`, respectively. Explicit
CLI destinations remain authoritative and MUST skip settings bootstrap when
both requested destinations are supplied.

Offline flags to set before runtime: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Acceptance Criteria

```gherkin
Feature: Pre-download
  Scenario: Download bge-m3
    When I run the CLI with model BAAI/bge-m3
    Then weights SHALL exist under the HF cache directory

  Scenario: Download runtime defaults
    When I run the CLI with --all
    Then the pinned BGE-M3, BM42, BGE reranker, and SigLIP snapshots SHALL exist under one HF cache directory

  Scenario: Use configured cache destinations
    Given model and parser cache directories are configured in .env
    When I omit both cache destination flags
    Then the CLI SHALL bootstrap settings once and use both configured directories

  Scenario: Override cache destinations explicitly
    When I supply both cache destination flags
    Then those paths SHALL be authoritative
    And settings bootstrap SHALL NOT run solely to resolve cache paths
```

## References

- huggingface_hub file_download docs.

## Offline Mode & Extras

- The CLI MUST support offline operation via `HF_HUB_OFFLINE=1` and deterministic tests with strict mocks.
- Optional extras MAY provide Parquet support (e.g., `[parquet]`); when Parquet is requested without `pyarrow`, the CLI MUST issue a clear warning and fall back to JSONL.
