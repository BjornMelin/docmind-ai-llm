---
spec: SPEC-003
title: Unified Embeddings: BGE‑M3 Text (dense+sparse) + Multimodal Images (SigLIP default)
version: 1.2.0
date: 2025-09-07
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-EMB-001: Text embeddings SHALL use BAAI/bge-m3 with dense+sparse.
  - FR-EMB-002: Image embeddings SHOULD default to SigLIP; OpenCLIP MAY be used when explicitly selected.
  - FR-EMB-003: Batch size and precision SHALL adapt to hardware.
  - NFR-PERF-002: Embed throughput ≥ 200 doc/sec on CPU for short chunks (64 tokens).
related_adrs: ["ADR-002","ADR-004","ADR-007","ADR-024"]
---


## Objective

Deliver a minimal, library‑first embedding stack for text and images, removing duplicate/legacy code while preserving SPEC‑003 capabilities:

- Text: BGE‑M3 dense + sparse using FlagEmbedding and LlamaIndex `Settings` for query‑time dense alignment; FastEmbed BM42/BM25 for sparse alignment with Qdrant hybrid.
- Images: OpenCLIP (ViT‑L/14, ViT‑H/14) and SigLIP via official libraries; prefer LlamaIndex ClipEmbedding where applicable.
- No custom sparse scoring, no bespoke CLIP wrappers, and no backward‑compat layers.

## Architecture and Libraries

- Text (primary):
  - BGE‑M3 dense (LlamaIndex embed model) + FastEmbed sparse (BM42/BM25) for hybrid alignment with Qdrant Query API.

- Images:
  - Transformers SigLIP (default) via `SiglipModel` + `SiglipProcessor`, using `get_image_features()` / `get_text_features()`.
  - OpenCLIP optionally via `open_clip.create_model_and_transforms('ViT-L-14'|'ViT-H-14', pretrained=...)` when explicitly required.
  - In LlamaIndex contexts, use a SigLIP adapter or `TextImageReranker` equivalents; avoid legacy CLIP-first wrappers.
  - Metadata: set `image_backbone="siglip"` for image/page nodes; default DPI≈200 for page image renders.

- Hosting note: Hugging Face Text Embeddings Inference (TEI) is dense‑only for BGE‑M3. Do not use TEI when sparse is required.

## Migration and Cleanup Plan (no back‑compat)

1) Text pipeline standardization

   - Adopt LI Settings for query‑time dense; FastEmbed for sparse queries; Qdrant Query API for server‑side fusion.
   - Remove custom sparse glue or tri‑mode layerings; keep library‑first wiring only.

2) Image pipeline consolidation

   - Prefer SigLIP as the primary image encoder; use OpenCLIP only when explicitly selected.
   - Remove ad‑hoc CLIP heuristics and redundant wrappers; centralize device/batch settings.
   - Derive embedding dimensions from model outputs at runtime (no hard‑coded dims).

3) Code deletions and refactors

   - Remove BGEM3 tri‑mode specific wrappers; avoid duplicative adapters.
   - Remove CLIP VRAM heuristic helpers and any nonessential image preprocess shims.
   - Keep a single, minimal SigLIP adapter for visual similarities only where needed; otherwise use Transformers directly.

4) Tests

   - Update tests to target LlamaIndex dense query alignment and FastEmbed sparse alignment; verify Qdrant Query API Prefetch typing.

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
    Given FastEmbed (BM42/BM25) and LlamaIndex Settings are configured
    When I retrieve over a list of texts
    Then I obtain hybrid retrieval using dense 1024‑d vectors and sparse lexical weights

  Scenario: Image encode
    When I encode a PNG
    Then I obtain a normalized vector with the expected dimension for the selected backbone

  Scenario: Offline operation
    Given HF offline flags are set and models are predownloaded
    When I request embeddings
    Then no network egress SHALL occur and the local cache SHALL be used

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
- Phase 2: Completed — callers migrated; imports updated; tests no longer import legacy symbols; SigLIP default enforced.
- Phase 3: Completed — ad‑hoc CLIP helpers removed; LI ClipEmbedding used; image dims derived at runtime.
- Phase 4: Completed — removed `src/retrieval/embeddings.py` and all references.
- Phase 5: Completed — docs updated; ruff/pylint gates green.

## References

- FlagEmbedding BGEM3 (dense/sparse/ColBERT) — official tutorials and APIs.
- LlamaIndex BGEM3Index/BGEM3Retriever; ClipEmbedding; vector store integrations.
- Transformers SigLIP docs; OpenCLIP model zoo and transforms.
- TEI (HF) limitations for BGE‑M3 sparse/ColBERT (dense‑only).
