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

## Acceptance Criteria

```gherkin
Feature: Pre-download
  Scenario: Download bge-m3
    When I run the CLI with model BAAI/bge-m3
    Then weights SHALL exist under the HF cache directory
```

## References

- huggingface_hub file_download docs.
