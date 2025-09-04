---
ADR: 002
Title: Unified Embedding Strategy with BGE‑M3
Status: Accepted
Version: 4.3
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 003, 006, 009, 031, 034, 001
Tags: embeddings, retrieval, hybrid, multimodal, local-first
References:
- [BAAI/bge-m3 — Hugging Face](https://huggingface.co/BAAI/bge-m3)
- [OpenAI CLIP ViT‑B/32 — Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
- [LlamaIndex — Embeddings](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/)
- [Qdrant — Documentation](https://qdrant.tech/documentation/)
---

## Description

Replace the three‑model setup (BGE‑large + SPLADE + CLIP) with a two‑model approach: **BGE‑M3** for unified dense+sparse text and **CLIP ViT‑B/32** for images. Reduces complexity and memory while improving retrieval quality.

## Context

Separate dense and sparse models increase coordination cost and memory. BGE‑M3 unifies dense/sparse (and supports multi‑granularity) with strong quality and 8K context, enabling simpler hybrid search with fewer moving parts.

## Decision Drivers

- Reduce model count and memory footprint
- Maintain or improve retrieval quality
- 100% local operation on consumer hardware
- Fit adaptive retrieval (ADR‑003) and storage (ADR‑031)

## Alternatives

- A: Current three‑model (BGE‑large + SPLADE + CLIP) — High overhead
- B: BGE‑M3 + CLIP (Selected) — Unified dense/sparse + multimodal
- C: Nomic‑v2 + CLIP — MoE, good multilingual; alternative local setup

### Decision Framework

| Model / Option         | Quality (40%) | Simplicity (30%) | Perf (30%) | Total Score | Decision      |
| ---------------------- | ------------- | ---------------- | ---------- | ----------- | ------------- |
| BGE‑M3 + CLIP          | 9             | 9                | 9          | **9.0**     | ✅ Selected    |
| Nomic‑v2 + CLIP        | 8             | 8                | 8          | 8.0         | Rejected      |
| 3‑model (baseline)     | 7             | 2                | 3          | 4.0         | Rejected      |

## Decision

We adopt BGE‑M3 (1024‑dim) for unified dense+sparse text embeddings and CLIP ViT‑B/32 (512‑dim) for images. Vectors are stored in Qdrant (ADR‑031) and consumed by the adaptive retrieval pipeline (ADR‑003). This replaces the previous three‑model setup to reduce complexity and memory while maintaining quality.

## High-Level Architecture

```mermaid
graph LR
  C[Chunking (ADR‑009)] --> T[BGE‑M3 Text Vectors]
  C --> I[CLIP Image Vectors]
  T --> Q[Qdrant]
  I --> Q
  Q --> H[Hybrid Retrieval (ADR‑003)]
```

## Related Requirements

### Functional Requirements

- FR‑1: Generate dense embeddings for semantic similarity
- FR‑2: Generate sparse embeddings for keyword retrieval
- FR‑3: Support multimodal search via image vectors

### Non-Functional Requirements

- NFR‑1: >30% embedding memory reduction vs baseline
- NFR‑2: Maintain or improve retrieval accuracy
- NFR‑3: Fully offline on consumer hardware

### Integration Requirements

- IR‑1: Use LlamaIndex embedding interfaces
- IR‑2: Persist in Qdrant with hybrid search enabled

### Performance Requirements

- PR‑1: <50ms per‑chunk embedding on RTX 4090 Laptop
- PR‑2: Efficient hybrid query latency via unified vectors

## Design

### Architecture Overview

- Unified embedding path for text; separate path for images
- Single ingestion→embedding→storage flow; hybrid retrieval consumes both

### Implementation Details

Expose a small `get_bgem3_embedding()` factory and set `Settings.embed_model` to the instance. Keep implementations thin and library‑first; avoid custom optimizers or caches in this ADR.

### Configuration

- Model names and batch sizes controlled via settings
- No bespoke env vars beyond standard model configuration

```env
DOCMIND_EMBEDDING__MODEL_NAME=BAAI/bge-m3
DOCMIND_EMBEDDING__BATCH_SIZE_GPU=64
DOCMIND_EMBEDDING__BATCH_SIZE_CPU=8
```

## Testing

- Unit: encode a few texts, assert shapes/dtypes and latency budget
- Integration: end‑to‑end retrieval improves or matches baseline quality

```python
def test_bgem3_shape(embed_model):
    vecs = embed_model.get_text_embedding("hello")
    assert len(vecs) == 1024
```

## Consequences

### Positive Outcomes

- Fewer models; lower memory; simpler maintenance
- Improved coordination between dense and sparse signals

### Negative Consequences / Trade-offs

- Requires one‑time re‑indexing of existing documents
- Increased reliance on BGE‑M3 implementation

### Dependencies

- Python: `FlagEmbedding>=1.2.0`, `torch>=2.0.0`, `llama-index>=0.10`
- Models: `BAAI/bge-m3`, `openai/clip-vit-base-patch32`

### Ongoing Maintenance & Considerations

- Track FlagEmbedding and LlamaIndex releases for embedding API changes
- Re‑evaluate batch sizes when hardware or drivers change
- Validate hybrid performance quarterly with a small benchmark set

## Changelog

- 4.2 (2025‑09‑04): Standardized to template; added PR/IR, config/tests; no behavior change
- 4.1 (2025‑09‑02): Replace ADR‑007 refs with ADR‑031; add ADR‑034; formatting
- 4.1 (2025‑08‑26): Implementation complete; integrated with ADR‑009
- 4.0 (2025‑08‑18): Update perf targets for RTX 4090 Laptop
- 3.0 (2025‑08‑17): Remove API‑only Voyage‑3; set BGE‑M3 as primary local
- 1.0 (2025‑01‑16): Initial design
