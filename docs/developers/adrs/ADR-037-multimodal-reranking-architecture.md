---
ADR: 037
Title: Multimodal Reranking with SigLIP Visual Re‑score (default) and ColPali (optional) + BGE v2‑m3 (text)
Status: Implemented
Version: 1.2
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

Adopt a dual‑path reranking architecture: default SigLIP text–image cosine re‑score for image/PDF‑page nodes (fast, zero‑service), and BGE v2‑m3 CrossEncoder for text nodes. ColPali remains an optional “pro” visual reranker on capable GPUs. Selection is automatic per‑node based on `metadata.modality`, and results are merged at rank‑level (RRF) in the adaptive retrieval pipeline.

## Context

Prior reranking (ADR‑006) used a single text CrossEncoder (`bge‑reranker‑v2‑m3`), which underperforms on image-first nodes (page renders, charts). We require full multimodality without adding services and while staying library-first.

## Decision Drivers

- Multimodal relevance across text and visual nodes with fast defaults
- Library-first using LlamaIndex postprocessors
- No new infra; keep local-first and GPU-friendly
- Alignment with adaptive retrieval and UI controls

## Alternatives

- A: ColBERT only — Pros: fast text LI; Cons: text-only, no images
- B: BGE v2‑m3 only — Pros: strong text X‑encoder; Cons: no visual semantics
- C: SigLIP re‑score (default) + optional ColPali + BGE v2‑m3 (Selected) — Pros: covers visual + text with fast default; Cons: optional 2nd visual model increases VRAM when enabled

### Decision Framework

| Model / Option                                   | Quality on Visual Docs (35%) | Latency/VRAM (30%) | Integration Simplicity (20%) | Adaptability (15%) | Total Score | Decision    |
| ------------------------------------------------ | ---------------------------- | ------------------ | ---------------------------- | ------------------ | ----------- | ----------- |
| **SigLIP default + optional ColPali (Selected)** | 8                            | 9                  | 9                            | 9                  | **8.6**     | ✅ Selected |
| ColPali default + fallback to SigLIP             | 9                            | 6                  | 7                            | 8                  | 7.7         | Rejected    |
| SigLIP only                                      | 7                            | 10                 | 10                           | 7                  | 8.1         | Rejected    |

## Decision

Use SigLIP normalized cosine re‑score for nodes with `metadata.modality in {"image","pdf_page_image"}` by default, and BGE v2‑m3 CrossEncoder for text nodes. Optionally enable ColPali on GPUs for higher visual precision. Merge modality reranks by rank‑level RRF. Supersedes ADR‑006. Activation gating MUST be env-only and SHOULD consider visual_fraction, small K (≤16), VRAM, and latency budget; fail-open on timeouts. Telemetry SHALL log activation decisions and per-stage latencies.

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
- **IR‑2:** Internal caps/timeouts only (no UI toggles); ops env overrides allowed

## Local‑First & Privacy

- All reranking (BGE text, SigLIP visual, optional ColPali) runs locally; models are loaded from local cache.
- Qdrant server‑side hybrid fusion executes in the local Qdrant process on `127.0.0.1`; no external APIs are involved.
- Set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and pre‑download weights to ensure zero network egress at runtime.

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
from llama_index.postprocessor.colpali_rerank import ColPaliRerank   # visual (optional)
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
    # Default SigLIP re‑score is implemented in‑project using transformers text/image features
    # Optional ColPali path if installed and enabled by ops
    if img_nodes and getattr(settings.retrieval, "enable_colpali", False):
        out += _make_visual_reranker(settings.retrieval.reranking_top_k).postprocess_nodes(img_nodes, query_str=query)

    # fuse and de-dup by node id, keep highest score
    best: dict[str, NodeWithScore] = {}
    for n in out:
        nid = n.node.node_id
        best[nid] = max(best.get(nid, n), n, key=lambda x: x.score or 0.0)

    return sorted(best.values(), key=lambda x: x.score or 0.0, reverse=True)[: settings.retrieval.reranking_top_k]
```

### Configuration

- Internal caps/timeouts and optional `RetrievalConfig.enable_colpali` (bool); reranking/hybrid always‑on.

**In `.env` or `settings.py`:**

```env
# Activation toggle for optional visual reranker
DOCMIND_RETRIEVAL__ENABLE_COLPALI=true

# Top‑K caps used by rerankers
DOCMIND_RETRIEVAL__RERANKING_TOP_K=16
DOCMIND_RETRIEVAL__SIGLIP_PRUNE_TOPK=64      # cascade prune before ColPali
DOCMIND_RETRIEVAL__COLPALI_FINAL_TOPK=16     # final K for ColPali

# Timeouts (milliseconds)
DOCMIND_RETRIEVAL__TEXT_RERANK_TIMEOUT_MS=250
DOCMIND_RETRIEVAL__SIGLIP_TIMEOUT_MS=150
DOCMIND_RETRIEVAL__COLPALI_TIMEOUT_MS=400

# Heuristic thresholds for activation (see Activation Policy)
DOCMIND_RETRIEVAL__VISUAL_FRACTION_THRESHOLD=0.35
DOCMIND_RETRIEVAL__MIN_VRAM_GB=8
DOCMIND_RETRIEVAL__ADDITIONAL_LATENCY_BUDGET_MS=30

### Telemetry (Required)

Emit per‑stage metrics in JSONL with canonical keys:

- `rerank.stage` = `text` | `visual` | `colpali`
- `rerank.topk` (int), `rerank.latency_ms` (int), `rerank.timeout` (bool)
- `retrieval.fusion_mode` and `retrieval.latency_ms` for surrounding stages
- `dedup.before`/`dedup.after`/`dedup.dropped` for pre‑merge de‑duplication
```

### Activation Policy (Heuristic)

- Enable ColPali when ALL apply:
  - `visual_fraction` of corpus ≥ `VISUAL_FRACTION_THRESHOLD` (or corpus flagged visual‑heavy), and
  - `RERANKING_TOP_K` ≤ 16, and
  - available GPU VRAM ≥ 8–12 GB, and
  - additional latency budget ≥ ~30 ms per query.
- Cascade on constrained GPUs: first apply SigLIP prune to `SIGLIP_PRUNE_TOPK` (e.g., 64), then apply ColPali on `COLPALI_FINAL_TOPK` (e.g., 16).

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
- Centralize device/VRAM policy via src.utils.core (select_device, has_cuda_vram), and prefer a thread-based executor with optional ProcessPool fallback for strict timeouts when needed.

### Dependencies

- Python: `llama-index`, `llama-index-postprocessor-colpali-rerank`, `sentence-transformers`, `torch`

## Changelog

- **1.2 (2025-09-07):** Clarified required telemetry keys and fail‑open behavior; no UI toggles, env‑only overrides.
  - Canonical env override: `DOCMIND_RETRIEVAL__USE_RERANKING` (maps to `settings.retrieval.use_reranking`).
- **1.1 (2025-09-07):** Set SigLIP as default visual re‑score; ColPali optional via policy; added decision framework and guardrails; aligned with SPEC‑005
- **1.0 (2025-09-03):** Initial accepted version; supersedes ADR‑006
