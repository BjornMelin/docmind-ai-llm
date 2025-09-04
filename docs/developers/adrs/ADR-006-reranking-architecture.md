---
ADR: 006
Title: Text-Only Reranking with BGE v2‑m3 (Superseded)
Status: Superseded
Version: 3.5
Date: 2025-09-04
Supersedes:
Superseded-by: 037
Related: 002, 003, 010, 036, 037
Tags: retrieval, reranking, text, sentence-transformers
References:
- [BAAI/bge-reranker-v2-m3 — Hugging Face](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [sentence-transformers — CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder.html)
- [LlamaIndex — Node Postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)
---

## Description

Historical record of the text‑only reranking baseline using `BAAI/bge-reranker-v2-m3` via sentence‑transformers CrossEncoder. This ADR is superseded by ADR‑037 (multimodal reranking) and retained to document rationale and prior configuration.

## Context

Vector similarity alone produced weaker ordering for text passages. A lightweight, library‑first CrossEncoder improved relevance within latency budgets on local hardware. Later, multimodal corpora (PDF page images, figures) required visual reranking, prompting ADR‑037.

## Decision Drivers

- Improve text relevance without custom implementations
- Keep latency low on consumer GPU
- Align with adaptive retrieval (ADR‑003) and UI controls (ADR‑036)

## Alternatives

- A: No reranking — Pros: simplest; Cons: lower relevance
- B: Multi‑model ensemble — Pros: quality; Cons: complexity, resources
- C: Single CrossEncoder (chosen historically) — Pros: balanced; Cons: text‑only

### Decision Framework

| Model / Option                 | Quality (40%) | Simplicity (30%) | Performance (20%) | Maintainability (10%) | Total Score | Decision       |
| ------------------------------ | ------------- | ---------------- | ----------------- | --------------------- | ----------- | -------------- |
| **CrossEncoder BGE v2‑m3**     | 8.5           | 9.5              | 8.5               | 9.0                   | **8.9**     | ✅ Selected     |
| No reranking                   | 5.0           | 10.0             | 9.0               | 9.5                   | 7.3         | Rejected       |
| Multi‑model ensemble           | 9.0           | 4.5              | 6.5               | 5.5                   | 6.9         | Rejected       |

## Decision

Adopt a single CrossEncoder reranker (`BAAI/bge-reranker-v2-m3`) for text nodes. This was the project baseline until multimodal reranking (ADR‑037) superseded it.

## High-Level Architecture

```mermaid
graph LR
  R[Retriever Results (text nodes)] --> CE[CrossEncoder BGE v2‑m3]
  CE --> T[Top‑N Sorted Nodes]
```

## Related Requirements

### Functional Requirements

- FR‑1: Rerank text results from retrieval
- FR‑2: Expose top‑N and normalization via settings/UI (ADR‑036)

### Non-Functional Requirements

- NFR‑1: ≤100ms for 20 docs on RTX 4090 Laptop
- NFR‑2: Fully offline; GPU‑accelerated when available

### Performance Requirements

- PR‑1: Achieve ≥10% NDCG@5 improvement vs similarity only

### Integration Requirements

- IR‑1: Integrate as LlamaIndex postprocessor or direct ST call
- IR‑2: Controlled via unified settings (ADR‑024) and UI (ADR‑036)

## Design

### Architecture Overview

- Single CrossEncoder reranks text nodes before synthesis
- Router (ADR‑003) feeds text nodes into reranker when mode=`text|auto`

### Implementation Details (Historical)

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

def rerank(query: str, documents: list[str], top_k: int = 10) -> list[tuple[str, float]]:
    pairs = [(query, d) for d in documents]
    scores = model.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

### Configuration

```env
DOCMIND_RETRIEVAL__RERANK_TOP_K=10
DOCMIND_RETRIEVAL__RERANK_NORMALIZE_SCORES=true
```

## Testing

```python
def test_rerank_orders_higher_scores_first():
    docs = ["a", "b", "c"]
    ranked = rerank("q", docs, top_k=2)
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]
```

## Consequences

### Positive Outcomes

- Better ordering for text passages with minimal code
- Easy to maintain; leverages sentence‑transformers

### Negative Consequences / Trade-offs

- Text‑only; does not handle images/diagrams
- Adds small latency (tens of ms) per query

### Ongoing Maintenance & Considerations

- Track sentence‑transformers releases
- Validate latency on upgrades

### Dependencies

- Python: `sentence-transformers`, `torch`

## Changelog

- 3.5 (2025-09-04): Standardized to ADR template; condensed historical content; noted superseded by ADR‑037.
- 3.2 (2025-09-03): Added cross‑reference to ADR‑036 (UI controls)
- 3.1 (2025-08-21): Implementation complete with BGE CrossEncoder
- 3.0 (2025-08-18): Updated performance targets for RTX 4090 Laptop
- 2.0 (2025-08-17): Simplified to direct CrossEncoder usage
- 1.0 (2025-01-16): Initial design with enhanced pipeline
