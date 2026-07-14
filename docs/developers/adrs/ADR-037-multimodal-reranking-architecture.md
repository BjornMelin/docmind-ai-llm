---
ADR: 037
Title: Multimodal reranking with SigLIP and BGE
Status: Implemented
Version: 1.6
Date: 2026-07-14
Supersedes: 006
Superseded-by:
Related: 003, 004, 024, 036, 058
Tags: retrieval, reranking, multimodal, images, pdf, llamaindex
References:
- [LlamaIndex node postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
---

## Decision

Use BGE v2-m3 for text nodes and SigLIP normalized cosine scoring for image and PDF page-image nodes. Merge modality-specific rankings with reciprocal rank fusion (RRF).

SigLIP is the single visual reranker. It already owns image embeddings, supports the canonical content-addressed artifact contract, and reads encrypted artifacts through DocMind's shared image loader.

## Context

The former text-only reranker could not score charts, page renders, or scanned documents. DocMind needs local visual relevance without another service or a second visual model contract.

The persistence boundary stores `ArtifactRef` identifiers and suffixes instead of raw filesystem paths. Any visual reranker must consume that canonical shape without creating a parallel path-based contract.

## Decision drivers

- One canonical visual model and loader
- Encrypted artifact support without temporary compatibility adapters
- Local inference with no required external service
- Bounded work and explicit router-owned lifecycle
- Native LlamaIndex postprocessor attachment

## Alternatives

| Option | Solution leverage | Application value | Maintenance | Adaptability | Weighted score | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BGE text + SigLIP visual | 9.2 | 9.3 | 9.5 | 8.6 | **9.3** | Selected |
| BGE text only | 8.8 | 6.0 | 9.8 | 6.5 | 7.9 | Rejected |
| Add a second visual model and loader | 6.0 | 8.0 | 4.0 | 6.0 | 6.1 | Rejected |

Scores use the repository decision weights: solution leverage 35%, application value 30%, maintenance 25%, and adaptability 10%.

## Architecture

```text
retrieved nodes
  -> split by metadata.modality
  -> BGE text rerank + SigLIP visual rerank
  -> RRF merge and deduplication
  -> synthesis
```

`MultimodalReranker` attaches to semantic, hybrid, multimodal, and graph query engines through LlamaIndex `node_postprocessors`. Its owned queue-free worker executes blocking inference outside asyncio's global executor.

## Failure and lifecycle contract

Reranking fails open to the exact original candidate order when a modality stage errors or exceeds its budget. A timeout does not release worker capacity until the native call exits.

The router owns each postprocessor. Async close drains admitted work. Sync close rejects new work without blocking on native inference.

## Configuration

Operations may set these values:

```env
DOCMIND_RETRIEVAL__USE_RERANKING=true
DOCMIND_RETRIEVAL__RERANKING_TOP_K=16
DOCMIND_RETRIEVAL__SIGLIP_PRUNE_M=64
DOCMIND_RETRIEVAL__TEXT_RERANK_TIMEOUT_MS=250
DOCMIND_RETRIEVAL__SIGLIP_TIMEOUT_MS=150
DOCMIND_RETRIEVAL__TOTAL_RERANK_BUDGET_MS=400
```

The UI may edit RRF and timeout values. It does not own the reranking enablement policy.

## Consequences

The selected design supports visual relevance with fewer dependencies, one encrypted-aware loader, and one durable metadata contract. It does not provide late-interaction visual reranking.

Future visual models must accept in-memory images or DocMind's artifact references directly. Do not add raw path metadata, temporary path adapters, alternate artifact shapes, or compatibility aliases.

## Verification

```bash
uv run pytest -q tests/unit/retrieval/reranking tests/unit/retrieval/test_router_async_tools.py tests/unit/retrieval/test_async_work.py
```
