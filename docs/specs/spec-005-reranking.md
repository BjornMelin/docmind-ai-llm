---
spec: SPEC-005
title: Reranking: BGE Cross-Encoder for Text + SigLIP Visual Re‑score (ColPali Optional)
version: 1.1.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-RER-001: Text rerank SHALL use BAAI/bge-reranker-v2-m3.
  - FR-RER-002: Visual rerank SHOULD default to SigLIP text–image similarity; ColPali MAY be enabled optionally on capable GPUs.
  - NFR-QUAL-001: reranking SHALL be always‑on with internal caps/timeouts and fail‑open behavior (no UI toggle).
related_adrs: ["ADR-037","ADR-024","ADR-036"]
---


## Objective

Improve retrieval quality by applying **BGE Cross-Encoder** for text nodes and **SigLIP** text–image similarity for page‑image nodes by default, with **ColPali** as an optional “pro” reranker on capable GPUs. Modality gating selects the appropriate path.

## Implementation Guidance

- Pipeline
  - Stage 1: Qdrant server-side hybrid returns fused candidates (text+image nodes).
  - Stage 2: Apply text rerank (`BAAI/bge-reranker-v2-m3`) to text nodes (top_n ≤ 40, timeout 250 ms).
  - Stage 3: Apply visual rerank: SigLIP text–image cosine to image/page nodes (top_n ≤ 10, timeout 150 ms).
  - Optional Stage 3b: If enabled and thresholds met, apply ColPali visual rerank (top_n ≤ 10, timeout 400 ms).
  - Stage 4: Rank-level RRF merge of modality-specific results; fail-open to fused order on timeouts.

- Activation policy (visual rerank)
  - Default SigLIP; auto-enable ColPali when: visual-heavy corpus OR high `visual_fraction`, and rerank `top_n ≤ 10–16`, and GPU VRAM ≥ 8–12 GB, and budget ≥ ~30 ms extra.
  - Cascade option: SigLIP prune to m (e.g., 64) → ColPali final on m' (e.g., 16).

- Telemetry
  - Log per-stage latency, device, top_n, and activation decisions; count timeouts and fail-open events.

## Development Notes

- Keep UI free of reranker toggles; provide ops env overrides only.
- Early-exit: if there are no image/page nodes, skip visual rerank stage entirely.

## Libraries and Imports

```python
from llama_index.core.postprocessor import SentenceTransformerRerank  # BGE text cross-encoder
# Optional visual reranker (plugin):
from llama_index.postprocessor.colpali_rerank import ColPaliRerank  # if installed
# SigLIP similarity implemented in-project (transformers AutoModel/AutoProcessor)
```

## File Operations

### UPDATE

- `src/retrieval/reranking.py`: implement `build_rerankers()` producing:
  - Text reranker: `SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3")` with `top_n<=40`, 250 ms timeout.
  - Visual rerank default: SigLIP text–image cosine re‑score with `top_n<=10`, 150 ms timeout.
  - Optional: `ColPaliRerank(model="vidore/colpali-v1.2")` with `top_n<=10`, 400 ms timeout.
  - Merge modality reranks by rank‑level RRF; fail‑open to fused order on timeouts.
- Remove UI toggles; expose only internal caps/timeouts and ops env overrides.

## Acceptance Criteria

```gherkin
Feature: Reranking modes
  Scenario: Text rerank
    Given a mixed candidate set
    Then `BAAI/bge-reranker-v2-m3` SHALL be applied to text nodes with top_n<=40 and 250 ms timeout

  Scenario: Visual rerank (default)
    Given a mixed candidate set
    Then SigLIP text–image cosine SHALL re‑score image/page nodes with top_n<=10 and 150 ms timeout

  Scenario: Visual rerank (optional ColPali)
    Given ColPali plugin installed and a suitable GPU
    Then ColPali MAY re‑score image/page nodes with top_n<=10 and 400 ms timeout

  Scenario: Fail-open on timeouts
    Given reranking timeouts occur in any stage
    Then the final ranked list SHALL fall back to the fused retrieval order and record a timeout event
```

## References

- BGE reranker; LlamaIndex node postprocessors; ColPali rerank example.
