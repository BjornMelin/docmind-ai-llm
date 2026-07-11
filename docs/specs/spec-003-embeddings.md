---
spec: SPEC-003
title: Unified embeddings with BGE-M3 text and SigLIP images
version: 1.4.0
date: 2026-07-11
owners: ["ai-arch"]
status: Completed
related_requirements:
  - FR-EMB-001: Text embeddings SHALL use BAAI/bge-m3 with dense+sparse.
  - FR-EMB-002: Image embeddings SHALL use SigLIP.
  - FR-EMB-003: Batch size and precision SHALL adapt to hardware.
  - NFR-PERF-002: Embedding benchmarks SHALL identify their hardware and inputs.
related_adrs: ["ADR-002","ADR-004","ADR-007","ADR-024"]
---


## Objective

Define the one supported embedding stack for text and images:

- Text: BGE-M3 dense embeddings through LlamaIndex and direct FastEmbed BM42/BM25 sparse vectors for Qdrant hybrid retrieval.
- Images: Transformers SigLIP through the shared `src/utils/vision_siglip.py` loader.
- No image-backend selector, OpenCLIP path, BGE image path, custom sparse scoring, or compatibility layer.

## Architecture and libraries

- Text:
  - BGE-M3 produces normalized 1024-dimensional dense vectors.
  - The default model revision is pinned in
    `src/config/embedding_defaults.py`.
  - LlamaIndex receives native `max_length`, `normalize`, and
    `embed_batch_size` options plus an offline cache or explicit local snapshot.
  - Direct FastEmbed produces BM42 sparse vectors and falls back to BM25 when BM42 is unavailable.
  - Qdrant performs server-side fusion through the Query API.

- Images:
  - `SiglipModel` and `SiglipProcessor` provide image and text features.
  - The default model uses the source-controlled revision in `src/utils/vision_siglip.py`.
  - `tools/models/pull.py --all` downloads the pinned Transformers snapshot.
  - Configuration exposes the SigLIP model ID, optional custom revision, normalization, and batch size. It exposes no backend selector.

- Hosting note: Hugging Face Text Embeddings Inference (TEI) is dense‑only for BGE‑M3. Do not use TEI when sparse is required.

## Current implementation

1. Text retrieval:

   - Use LlamaIndex `Settings` for query-time dense embeddings.
   - Use direct FastEmbed for sparse queries.
   - Use Qdrant for server-side fusion.

2. Image embeddings:

   - Use only SigLIP.
   - Load the model and processor through `src/utils/vision_siglip.py`.
   - Derive nonempty output dimensions from the model.

3. Configuration:

   - Keep text model ID, revision, optional local path, cache folder,
     dimensions, sparse enablement, normalization, and batch sizes under
     `settings.embedding`.
   - Keep SigLIP model ID, optional custom revision, normalization, and image batch size under the same group.

4. Tests:

   - Cover BGE-M3 dimensions, direct FastEmbed sparse output, SigLIP loading, image encoding, and Qdrant prefetch typing.

## Acceptance criteria

```gherkin
Feature: Embedding stack
  Scenario: Text dense+sparse
    Given direct FastEmbed and LlamaIndex Settings are configured
    When I retrieve over a list of texts
    Then I obtain hybrid retrieval using dense 1024‑d vectors and sparse lexical weights

  Scenario: Image encode
    When I encode a PNG
    Then I obtain a normalized SigLIP vector with the model's output dimension

  Scenario: Offline operation
    Given HF offline flags are set and models are predownloaded
    When I request embeddings
    Then model loading uses the local cache or fails without downloading

  Scenario: Eliminate duplicate/legacy wrappers
    Given the codebase
    Then no duplicate BGE-M3 or image-backend compatibility wrapper remains
```

## Implementation status

- BGE-M3 dense and direct FastEmbed sparse paths are implemented.
- BGE-M3 is pinned to an exact revision and its configured dimension is fixed
  at 1024. A failed load leaves LlamaIndex's backing slot empty.
- SigLIP is the only image backend.
- The default SigLIP revision is pinned in source and prefetched by `tools/models/pull.py --all`.
- Qdrant validates named-vector dimensions before indexing.

## References

- FastEmbed BM42/BM25 sparse embedding documentation
- LlamaIndex Settings for dense query alignment; Qdrant Query API hybrid
- Transformers SigLIP documentation
- TEI (HF) limitations for BGE‑M3 sparse (dense-only)

## Routing and dimensions

- Text embeddings SHOULD use BGE‑M3 (dimension 1024). Implement checks to assert dimension alignment across indexing/retrieval.
- Image and text-to-image similarity MUST use SigLIP.
- Embedding routers MUST report model ID or dimension mismatches before indexing.
