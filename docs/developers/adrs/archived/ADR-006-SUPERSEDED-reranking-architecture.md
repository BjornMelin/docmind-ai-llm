---
ADR: 006
Title: Text‑Only CrossEncoder Reranking (Superseded)
Status: Superseded
Version: 3.4
Date: 2025-09-04
Supersedes:
Superseded-by: 037
Related: 003, 024, 037
Tags: retrieval, reranking, text, crossencoder
References:
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [sentence-transformers CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder.html)
---

## Description

Historical ADR documenting the text‑only reranking decision: adopt a single‑model CrossEncoder (`BAAI/bge-reranker-v2-m3`) to rerank vector candidates before LLM synthesis. This improved precision with minimal code and local performance. Superseded by ADR‑037, which adds multimodal reranking.

## Context

Flat vector similarity alone returned off‑topic results for nuanced queries. We needed a simple, local, fast reranker that improved top‑K precision without large engineering overhead, and that fit the 128K context budgeting (ADR‑004). A text‑only CrossEncoder offered strong relevance gains with a tiny code footprint.

## Decision Drivers

- Local‑first performance on consumer GPUs
- Higher precision@K than vector similarity alone
- Simplicity and maintainability (library‑first)
- Predictable latency and memory footprint

## Alternatives

- A: No reranking (vector‑only) — Pros: simplest; Cons: lower precision on hard queries. Rejected.
- B: Multi‑model ensemble — Pros: highest quality; Cons: complex, heavy. Rejected for local.
- C: Single‑model CrossEncoder (Selected) — Pros: high leverage, simple, fast.

### Decision Framework

| Option                         | Leverage (35%) | Quality (30%) | Latency (20%) | Maintain (15%) | Total | Decision |
| ------------------------------ | -------------- | ------------- | ------------- | -------------- | ----- | -------- |
| CrossEncoder (Selected)        | 5              | 4             | 4             | 5              | 4.55  | ✅ Selected |
| Vector‑only                    | 2              | 2             | 5             | 5              | 3.10  | Rejected |
| Multi‑model ensemble           | 4              | 5             | 2             | 2              | 3.55  | Rejected |

## Decision

Adopt a text‑only CrossEncoder reranker using `sentence-transformers` with `BAAI/bge-reranker-v2-m3`. Use it as a post‑processor on retrieved candidates to produce a tighter top‑K for the LLM. Integrate via simple, library‑first code. Note: This decision is superseded by ADR‑037 for multimodal data.

## High-Level Architecture

```mermaid
flowchart LR
  Q["Query"] --> VS["Vector Store<br/>Top N candidates"]
  VS --> CE["CrossEncoder<br/>(bge-reranker-v2-m3)"]
  CE --> TK["Top K reranked"]
  TK --> LLM["LLM @128K"]
```

## Related Requirements

### Functional Requirements

- **FR-1:** Rerank vector candidates to improve precision@K.
- **FR-2:** Support batching for throughput.
- **FR-3:** Integrate as a retrieval post‑processor.
- **FR-4:** Expose configurable `top_k`.

### Non-Functional Requirements

- **NFR-1 (Performance):** <100ms for 20 candidates on target GPU.
- **NFR-2 (Quality):** ≥10% NDCG@5 improvement vs vector‑only.
- **NFR-3 (Memory):** <1GB additional footprint.
- **NFR-4 (Local‑First):** No external dependencies.

### Performance Requirements

- **PR-1:** Reranking step P95 latency <100ms for K≤20.
- **PR-2:** Throughput ≥200 pairs/sec on RTX 4090 Laptop.

### Integration Requirements

- **IR-1:** Implemented as a LlamaIndex post‑processor or adapter callable from retrieval pipeline (ADR‑003).
- **IR-2:** Configured via `DocMindSettings.retrieval` (ADR‑024).
- **IR-3:** Plays nicely with modality‑aware reranking (ADR‑037) when enabled.

## Design

### Architecture Overview

- Retrieve top‑N via vector store (Qdrant).
- Rerank N pairs `(query, doc_text)` with CrossEncoder.
- Keep top‑K for LLM context assembly, honoring the 128K cap.

### Implementation Details

```python
# src/retrieval/reranking.py (skeleton)
from sentence_transformers import CrossEncoder
from typing import Sequence


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", top_k: int = 10):
        self.model = CrossEncoder(model)
        self.top_k = top_k

    def rerank(self, query: str, texts: Sequence[str]) -> list[tuple[str, float]]:
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
        return ranked[: self.top_k]
```

### Configuration

```env
DOCMIND_RETRIEVAL__USE_RERANKING=true
DOCMIND_RETRIEVAL__RERANKING_TOP_K=5
DOCMIND_RETRIEVAL__RERANKER_MODE=text   # auto|text|multimodal (ADR‑037)
```

## Testing

```python
# tests/unit/test_text_reranker.py (skeleton)
import pytest
from src.retrieval.reranking import CrossEncoderReranker


@pytest.mark.unit
def test_top_k_applied():
    r = CrossEncoderReranker(top_k=3)
    texts = [f"doc {i}" for i in range(10)]
    ranked = r.rerank("query", texts)
    assert len(ranked) == 3


@pytest.mark.unit
def test_rerank_orders_by_score(monkeypatch):
    r = CrossEncoderReranker(top_k=5)
    monkeypatch.setattr(r, "model", type("M", (), {"predict": lambda _self, pairs: list(range(len(pairs)))[::-1]})())
    texts = [f"d{i}" for i in range(5)]
    ranked = r.rerank("q", texts)
    assert [t for t, _ in ranked] == texts  # reverse scores -> original order is highest first
```

## Consequences

### Positive Outcomes

- Significant precision gains vs vector‑only with minimal code.
- Fast, local, and easy to maintain.
- Clean fit into retrieval pipeline before LLM.

### Negative Consequences / Trade-offs

- Text‑only: cannot assess image/PDF visual relevance (reason for ADR‑037).
- Adds up to ~100ms latency on large batches.
- Requires GPU for best throughput.

### Ongoing Maintenance & Considerations

- Pin `sentence-transformers` and model versions; review quarterly.
- Track `torch` CUDA compatibility for target machines.
- Validate that `top_k` and reranker cutoff align with 128K budgeting.

### Dependencies

- Python: `sentence-transformers>=2.2.0`, `torch>=2.0.0`.
- Models: `BAAI/bge-reranker-v2-m3`.
- Removed: Custom reranking pipelines; rely on library‑first CrossEncoder.

## Changelog

- 3.4 (2025-09-04): Standardized to ADR template; added decision matrix, architecture, config/testing skeletons; clarified superseded status and integration requirements.
- 3.3 (2025-09-03): Marked as Superseded by ADR‑037; noted multimodal path.
- 3.1 (2025-08-21): Implementation complete with `bge-reranker-v2-m3`.
- 3.0 (2025-08-18): Updated performance targets for RTX 4090 Laptop.
- 2.0 (2025-08-17): Simplified to single‑model `sentence-transformers` path.
- 1.0 (2025-01-16): Initial adaptive reranking concept.
