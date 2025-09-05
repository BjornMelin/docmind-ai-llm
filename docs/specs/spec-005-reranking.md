---
spec: SPEC-005
title: Reranking: BGE Cross-Encoder for Text + ColPali for Page-Images
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-RER-001: Text rerank SHALL use BAAI/bge-reranker-v2-m3.
  - FR-RER-002: Visual rerank SHALL use ColPali on pdf_page_image nodes.
  - NFR-QUAL-001: reranking SHALL be toggleable and configurable in UI.
related_adrs: ["ADR-007","ADR-036"]
---


## Objective

Improve retrieval quality by applying **BGE Cross-Encoder** for text nodes and **ColPali** for page-image nodes with modality gating.

## Libraries and Imports

```python
from llama_index.core.postprocessor import FlagEmbeddingReranker
from llama_index.postprocessor.colbert_rerank import ColbertRerank as ColPaliRerank  # name may vary
```

## File Operations

### UPDATE

- `src/retrieval/reranking.py`: implement `build_rerankers(mode: Literal["auto","text","multimodal"])` returning a list of postprocessors.
- `src/pages/settings.py`: radio for reranker mode and sliders for `top_n` and `normalize_scores`.

## Acceptance Criteria

```gherkin
Feature: Reranking modes
  Scenario: Text-only rerank
    Given reranker mode = text
    Then `FlagEmbeddingReranker` SHALL be used

  Scenario: Multimodal rerank
    Given reranker mode = multimodal
    Then ColPali SHALL re-score pdf_page_image nodes
```

## References

- BGE reranker; LlamaIndex node postprocessors; ColPali rerank example.
