---
spec: SPEC-003
title: Unified Embeddings: BGE‑M3 Text (dense+sparse+ColBERT) + Multimodal Images (OpenCLIP/SigLIP)
version: 1.1.0
date: 2025-09-07
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-EMB-001: Text embeddings SHALL use BAAI/bge-m3 with dense+sparse (and optional ColBERT).
  - FR-EMB-002: Image embeddings SHALL support OpenCLIP ViT‑L/H and SigLIP base.
  - FR-EMB-003: Batch size and precision SHALL adapt to hardware.
  - NFR-PERF-002: Embed throughput ≥ 200 doc/sec on CPU for short chunks (64 tokens).
related_adrs: ["ADR-002","ADR-004","ADR-007","ADR-024"]
---


## Objective

Deliver a minimal, library‑first embedding stack for text and images, removing duplicate/legacy code while preserving SPEC‑003 capabilities:

- Text: BGE‑M3 dense + sparse + optional ColBERT using FlagEmbedding and LlamaIndex’s native BGE‑M3 index/retriever.
- Images: OpenCLIP (ViT‑L/14, ViT‑H/14) and SigLIP via official libraries; prefer LlamaIndex ClipEmbedding where applicable.
- No custom sparse scoring, no bespoke CLIP wrappers, and no backward‑compat layers.

## Architecture and Libraries

- Text (primary):
  - FlagEmbedding BGEM3FlagModel/M3Embedder for unified outputs: `dense_vecs`, `lexical_weights`, `colbert_vecs`.
  - LlamaIndex BGEM3Index + BGEM3Retriever for tri‑mode retrieval (configurable weights). No custom sparse/ColBERT glue.

- Images:
  - OpenCLIP via `open_clip.create_model_and_transforms('ViT-L-14'|'ViT-H-14', pretrained=...)`.
  - Transformers SigLIP via `SiglipModel` + `SiglipProcessor`, using `get_image_features()`.
  - In LlamaIndex contexts, prefer `ClipEmbedding` for CLIP‑based features.

- Hosting note: Hugging Face Text Embeddings Inference (TEI) is dense‑only for BGE‑M3. Do not use TEI when sparse/ColBERT are required.

## Migration and Cleanup Plan (no back‑compat)

1) Text pipeline standardization

- Adopt LlamaIndex BGEM3Index + BGEM3Retriever as the primary retrieval path for BGE‑M3 tri‑mode.
- Remove custom sparse/ColBERT glue and any BGEM3‑specific wrappers beyond a thin adapter (if needed by non‑LI callsites).

2) Image pipeline consolidation

- Use LlamaIndex ClipEmbedding for CLIP where the pipeline already uses LI; keep SigLIP for non‑LI contexts.
- Remove ad‑hoc CLIP heuristics and redundant wrappers; centralize any required device/batch settings in config.
- Derive embedding dimensions from model outputs at runtime (no hard‑coded OpenCLIP/SigLIP dims).

3) Code deletions and refactors

- Remove `BGEM3Embedding` (LI adapter) after migrating tests and callsites to BGEM3Index/BGEM3Retriever.
- Remove CLIP VRAM heuristic helpers and any nonessential image preprocess shims.
- Keep a single, minimal UnifiedEmbedder only if it provides value beyond LI composition; otherwise remove it and use LI objects directly.

4) Tests

- Update tests to target BGEM3Index/BGEM3Retriever outputs and ClipEmbedding (with stubbed backends). Remove tests that assert previous wrapper‑specific shapes/keys.

5) Configuration

- Keep SPEC‑003 knobs in `settings.embedding` but narrow to: model id, enable_sparse, device, batch sizes, image backbone.
- Document that `enable_sparse=True` implies FlagEmbedding + LI retriever path; TEI is unsupported for sparse.

6) Quality gates

- All modules must pass `ruff` (format+lint) and `pylint --fail-under=9.5`.
- No legacy/duplicate code remains; no deprecation notes or back‑compat toggles.

## Acceptance Criteria (updated)

```gherkin
Feature: Embedding stack
  Scenario: Text dense+sparse
    Given BGEM3Index is configured with FlagEmbedding
    When I retrieve over a list of texts
    Then I obtain hybrid retrieval using dense 1024‑d vectors and sparse lexical weights

  Scenario: Image encode
    When I encode a PNG
    Then I obtain a normalized vector with the expected dimension for the selected backbone

  Scenario: Eliminate duplicate/legacy wrappers
    Given the codebase
    Then no BGEM3Embedding or ad‑hoc CLIP wrapper remains and tests use LI classes
```

## Implementation Phases (high level)

1. Introduce LI BGEM3Index/BGEM3Retriever wiring and migrate retrieval paths.
2. Replace CLIP usage with LI ClipEmbedding where applicable; keep SigLIP path for non‑LI use.
3. Remove duplicate wrappers (BGEM3Embedding, ad‑hoc CLIP helpers), adjust imports/usages.
4. Rewrite/realign tests to new interfaces; maintain fast offline stubs.
5. Run quality gates; address lint and style; finalize docs.

### Status Tracking (2025‑09‑07)

- Phase 1: Completed
- Phase 2: Completed — callers migrated; imports updated; tests no longer import legacy symbols.
- Phase 3: Completed — ad‑hoc CLIP helpers removed; LI ClipEmbedding used; image dims derived at runtime.
- Phase 4: Completed — removed `src/retrieval/embeddings.py` and all references.
- Phase 5: Completed — docs updated; ruff/pylint gates green.

## References

- FlagEmbedding BGEM3 (dense/sparse/ColBERT) — official tutorials and APIs.
- LlamaIndex BGEM3Index/BGEM3Retriever; ClipEmbedding; vector store integrations.
- OpenCLIP model zoo and transforms; Transformers SigLIP docs.
- TEI (HF) limitations for BGE‑M3 sparse/ColBERT (dense‑only).
