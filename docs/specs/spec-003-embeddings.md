---
spec: SPEC-003
title: Unified Embeddings: BGE-M3 Text + Tiered Image (OpenCLIP/SigLIP) + Optional Visualized‑BGE
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-EMB-001: Text embeddings SHALL use BAAI/bge-m3 with dense+sparse.
  - FR-EMB-002: Image embeddings SHALL support OpenCLIP ViT-L/H and SigLIP base.
  - FR-EMB-003: Batch size and precision SHALL adapt to hardware.
  - NFR-PERF-002: Embed throughput ≥ 200 doc/sec on CPU for short chunks (64 tokens).
related_adrs: ["ADR-002","ADR-004","ADR-007"]
---


## Objective

Provide unified text and image embeddings. Text uses **BGE‑M3** via FlagEmbedding or HuggingFace. Image uses **OpenCLIP** (ViT-L/H/14) for CPU/low‑VRAM and **SigLIP base** for mid‑GPU; allow optional **BGE‑Visualized** for unified multimodal on capable GPUs.

## Libraries and Imports

```python
from FlagEmbedding import BGEM3FlagModel  # text
import torch
# Image encoders
import open_clip  # OpenCLIP
from transformers import AutoModel, AutoProcessor  # SigLIP or Visualized-BGE
```

## File Operations

### UPDATE

- `src/models/embeddings.py`: implement `TextEmbedder`, `ImageEmbedder`, `UnifiedEmbedder` classes with `encode_text`, `encode_image`, `encode_pair` methods. Expose normalize, device, batch size.
- `src/retrieval/embeddings.py`: route embedding calls to above and support sparse vector extraction from BGE‑M3 when hybrid is enabled.

### CREATE

- `src/utils/multimodal.py`: image preprocessing helpers for CLIP/SigLIP.

## Acceptance Criteria

```gherkin
Feature: Embedding stack
  Scenario: Text dense+sparse
    Given BGEM3 is configured
    When I encode a list of texts
    Then I obtain 1024-d dense vectors and a sparse representation

  Scenario: Image encode
    When I encode a PNG
    Then I obtain a 768/1024-d vector depending on the backbone
```

## Git Plan

- Branch: `feat/embeddings`
- Commits:
  1. `feat(embeddings): add BGE-M3 text embedder with sparse output`
  2. `feat(embeddings): add OpenCLIP and SigLIP image embedders`
  3. `refactor(retrieval): route to unified embedder`

## References

- BGE‑M3, OpenCLIP, SigLIP, Visualized‑BGE.
