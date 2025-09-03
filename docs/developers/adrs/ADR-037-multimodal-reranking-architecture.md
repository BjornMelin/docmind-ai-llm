---
ADR: 037
Title: Multimodal Reranking with ColPali (visual) and BGE v2‑m3 (text)
Status: Accepted
Version: 1.0
Date: 2025-09-03
Supersedes: 006
Superseded-by:
Related: 003, 004, 024, 036
Tags: retrieval, reranking, multimodal, images, pdf, llamaindex
References:
- [LlamaIndex ColPali Rerank API](https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/colpali_rerank/)
- [LlamaIndex Node Postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [ColBERT overview](https://colbert.ai/)
---

## Description

Adopt a dual-path reranking architecture: ColPali for image/PDF-page nodes and BGE v2‑m3 CrossEncoder for text nodes. Selection is automatic per-node based on `metadata.modality`, and results are fused in the adaptive retrieval pipeline.

## Context

Prior reranking (ADR‑006) used a single text CrossEncoder (`bge‑reranker‑v2‑m3`), which underperforms on image-first nodes (page renders, charts). We require full multimodality without adding services and while staying library-first.

## Decision Drivers

- Multimodal relevance across text and visual nodes
- Library-first using LlamaIndex postprocessors
- No new infra; keep local-first and GPU-friendly
- Alignment with adaptive retrieval and UI controls

## Alternatives

- A: ColBERT only — Pros: fast text LI; Cons: text-only, no images
- B: BGE v2‑m3 only — Pros: strong text X‑encoder; Cons: no visual semantics
- C: ColPali + BGE v2‑m3 (Selected) — Pros: covers visual + text; Cons: two models

### Decision Framework

| Model / Option           | Coverage (35%) | Quality (30%) | Perf (20%) | Ops Simplicity (15%) | Total | Decision      |
| ------------------------ | -------------- | ------------- | ---------- | -------------------- | ----- | ------------- |
| ColPali + BGE v2‑m3      | 10             | 9             | 8          | 8                    | **9.0** | ✅ Selected |
| ColBERT only            | 5              | 8             | 9          | 9                    | 7.3   | Rejected      |
| BGE v2‑m3 only          | 6              | 9             | 9          | 9                    | 7.8   | Rejected      |

## Decision

Use ColPali for nodes with `metadata.modality in {"image","pdf_page_image"}` and BGE v2‑m3 CrossEncoder otherwise. Keep top‑N and normalization controlled by settings/UI. Supersedes ADR‑006.

## High-Level Architecture

Retriever → Mixed nodes (text + images) → Gate per-node modality → Rerank (ColPali or BGE) → Merge/fuse → Synthesis

## Related Requirements

### Functional Requirements

- **FR‑1:** Rerank visual nodes using a VLM reranker
- **FR‑2:** Preserve text reranking quality for pure text corpora
- **FR‑3:** Auto‑select reranker per node; no manual switching needed

### Non-Functional Requirements

- **NFR‑1:** Library-first (LlamaIndex)
- **NFR‑2:** GPU‑accelerated inference where available
- **NFR‑3:** No external services required

### Performance Requirements

- **PR‑1:** ≤150ms P95 reranking for 20 nodes on 4090 Laptop
- **PR‑2:** ≤1GB additional VRAM total for rerankers

### Integration Requirements

- **IR‑1:** Integrate as LlamaIndex `node_postprocessors`
- **IR‑2:** Configurable via `RetrievalConfig` and Streamlit controls

## Design

### Architecture Overview

- Reranking layer chooses ColPali or BGE v2‑m3 based on `node.metadata.modality`
- Normalizes and truncates to `top_n`, optional de-dup & fusion

### Implementation Details

**In `src/retrieval/reranking.py`:**

```python
from typing import List
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank  # text
from llama_index.postprocessor.colpali_rerank import ColPaliRerank   # visual
from src.config import settings

def _make_text_reranker(top_n: int):
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=top_n,
        use_fp16=True,
        normalize=settings.retrieval.reranker_normalize_scores,
    )

def _make_visual_reranker(top_n: int):
    return ColPaliRerank(
        model="vidore/colpali-v1.2",
        top_n=top_n,
    )

def multimodal_rerank(nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
    text_nodes = [n for n in nodes if n.node.metadata.get("modality", "text") == "text"]
    img_nodes = [n for n in nodes if n.node.metadata.get("modality") in {"image", "pdf_page_image"}]

    out: list[NodeWithScore] = []
    if text_nodes:
        out += _make_text_reranker(settings.retrieval.reranking_top_k).postprocess_nodes(text_nodes, query_str=query)
    if img_nodes:
        out += _make_visual_reranker(settings.retrieval.reranking_top_k).postprocess_nodes(img_nodes, query_str=query)

    # fuse and de-dup by node id, keep highest score
    best: dict[str, NodeWithScore] = {}
    for n in out:
        nid = n.node.node_id
        best[nid] = max(best.get(nid, n), n, key=lambda x: x.score or 0.0)

    return sorted(best.values(), key=lambda x: x.score or 0.0, reverse=True)[: settings.retrieval.reranking_top_k]
```

### Configuration

- `RetrievalConfig.reranker_mode`: `auto|text|multimodal` (default: `auto`)
- `RetrievalConfig.reranker_normalize_scores`: bool (default: True)
- `RetrievalConfig.reranking_top_k`: int (default: 10)

## Testing

**In `tests/test_multimodal_rerank.py`:**

```python
import pytest
from src.retrieval.reranking import multimodal_rerank

@pytest.mark.asyncio
async def test_auto_gating_text_and_image_nodes(sample_nodes):
    ranked = multimodal_rerank(sample_nodes, "What does the chart show?")
    assert ranked
```

## Consequences

### Positive Outcomes

- True multimodal relevance; better answers on diagrams/figures
- Minimal code surface; library-first

### Negative Consequences / Trade-offs

- Two models to ship; small VRAM overhead

### Ongoing Maintenance & Considerations

- Track LlamaIndex ColPali integration updates
- Periodically evaluate reranker latency

### Dependencies

- Python: `llama-index`, `llama-index-postprocessor-colpali-rerank`, `sentence-transformers`, `torch`

## Changelog

- **1.0 (2025-09-03):** Initial accepted version; supersedes ADR‑006
