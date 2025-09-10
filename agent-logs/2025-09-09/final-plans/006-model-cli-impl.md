# SPEC-013 + ADR-040 — Model Pre-download CLI Implementation

Date: 2025-09-09

## Purpose

Provide a simple CLI to pre-download required models to the local Hugging Face cache to enable fully offline operation.

## Prerequisites

- `huggingface_hub` installed
- Adequate disk space in cache directory

## Files to Create (Checklist)

- [x] `tools/models/pull.py` — minimal CLI using `hf_hub_download`
- [ ] (Optional later) `tools/models/verify.py`, `tools/models/list.py` and manifest JSON

Code reference: final-plans/011-code-snippets.md (Section 9)

## Imports and Libraries

- `from huggingface_hub import hf_hub_download`
- Standard: `argparse`, `pathlib.Path`, `typing.Iterable`

Example imports:

```python
import argparse
from pathlib import Path
from typing import Iterable
from huggingface_hub import hf_hub_download
```

## Default Models

- Text embeddings: `BAAI/bge-m3`
- Text reranker: `BAAI/bge-reranker-v2-m3`
- Image/text (visual): `google/siglip-base-patch16-224`
- Sparse (BM42): `Qdrant/bm42-all-minilm-l6-v2-attentions`

## CLI Behavior

- Flags:
  - `--all` to download the default set
  - `--add REPO_ID FILENAME` repeatable for custom files
  - `--cache_dir` to override cache path (respects HF envs as well)
- Output: prints resolved local paths and a reminder for `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` runtime flags
- Errors: fail clearly if nothing to download

## Acceptance Criteria

- Running `python tools/models/pull.py --all` downloads all default files into the chosen cache and prints paths.
- `--add` allows appending arbitrary repo file pairs.
- Offline runtime works after downloads with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Gherkin:

```gherkin
Feature: Model pre-download CLI
  Scenario: Download default set
    When I run the CLI with --all
    Then default model files are present in the cache directory
    And the CLI prints resolved paths

  Scenario: Add specific file
    When I run the CLI with --add REPO FILE
    Then the specified file is present in the cache directory
```

## Testing and Notes

- Unit test mocks `hf_hub_download` and asserts invocations and outputs.
- Optional enhancement: `--manifest` to dump a manifest JSON and/or `snapshot_download` for full repo mirror per 005.

## Environment Variables

- Force offline runtime after downloads:
  - `export HF_HUB_OFFLINE=1`
  - `export TRANSFORMERS_OFFLINE=1`

## Refactors/Deletions

- Delete `scripts/model_prep/predownload_models.py` after adding `tools/models/pull.py` and updating any references in docs/tests to the new CLI.
- Remove any duplicated model download code paths elsewhere after the new CLI is adopted.

## Cross-Links

- Code snippets for CLI and tests: 011-code-snippets.md (Section 9 and Section 8)
