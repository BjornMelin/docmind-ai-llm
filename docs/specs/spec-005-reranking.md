---
spec: SPEC-005
title: Reranking with BGE for text and SigLIP for visual nodes
version: 1.5.0
date: 2026-07-14
owners: ["ai-arch"]
status: Completed
related_requirements:
  - FR-RER-001: Text rerank SHALL use BAAI/bge-reranker-v2-m3.
  - FR-RER-002: Visual rerank SHALL use SigLIP text-image similarity.
  - NFR-QUAL-001: Reranking SHALL be enabled by default with internal caps, timeouts, and fail-open behavior.
related_adrs: ["ADR-037", "ADR-024", "ADR-036"]
---

## Objective

Improve retrieval quality with one reranker per modality. BGE scores text nodes, SigLIP scores image and PDF page-image nodes, and reciprocal rank fusion (RRF) merges both lists.

## Pipeline contract

The reranking pipeline has four stages:

1. Qdrant returns the fused candidate set.
2. `BAAI/bge-reranker-v2-m3` reranks text nodes within a 250 ms stage budget.
3. SigLIP reranks visual nodes within a 150 ms stage budget.
4. RRF merges and deduplicates both modality lists within a 400 ms total budget.

If a stage times out or its model fails, reranking returns the original retrieval order. It never publishes a partial modality result as a complete answer.

## Execution and lifecycle

Each `MultimodalReranker` owns one queue-free `AsyncWorkExecutor` for blocking BGE and SigLIP inference. Async postprocessing overrides LlamaIndex's default `asyncio.to_thread` path.

A caller timeout returns immediately but keeps executor capacity occupied until the native call exits. Later queries cannot queue unbounded work. The router owns every postprocessor it creates: asynchronous shutdown drains active workers, while synchronous shutdown rejects new work without waiting for native inference.

## Model and artifact ownership

The text reranker resolves the pinned `BAAI/bge-reranker-v2-m3` snapshot from `settings.embedding.cache_folder` without a runtime download. SigLIP model and processor loading is centralized in `src/utils/vision_siglip.py`.

Visual nodes persist only content-addressed `ArtifactRef` metadata. SigLIP resolves those references through `ArtifactStore` and uses the encrypted-aware image loader. Durable stores never contain raw host paths.

## Telemetry

Every stage records these fields:

- `rerank.stage`: `text`, `visual`, `final`, or `total`
- `rerank.topk`
- `rerank.latency_ms`
- `rerank.timeout`

The final stage may also record `rerank.delta_changed_count`, `rerank.path`, and `rerank.total_timeout_budget_ms`. Telemetry must not include document text, paths, or secrets.

## Configuration

The canonical operational settings are:

```env
DOCMIND_RETRIEVAL__USE_RERANKING=true
DOCMIND_RETRIEVAL__RERANKING_TOP_K=16
DOCMIND_RETRIEVAL__SIGLIP_PRUNE_M=64
DOCMIND_RETRIEVAL__TEXT_RERANK_TIMEOUT_MS=250
DOCMIND_RETRIEVAL__SIGLIP_TIMEOUT_MS=150
DOCMIND_RETRIEVAL__TOTAL_RERANK_BUDGET_MS=400
```

The Settings page exposes RRF and timeout values. It does not expose a reranking on/off control.

## Attachment points

Semantic, hybrid, multimodal, and graph query engines receive `MultimodalReranker` through LlamaIndex's native `node_postprocessors` mechanism. Text-only graph results select the BGE path. Mixed vector results select both modality paths.

`DOCMIND_RETRIEVAL__USE_RERANKING=false` is the only supported opt-out. When disabled, `router_factory` omits node postprocessors.

## Acceptance criteria

```gherkin
Feature: Modality-aware reranking
  Scenario: Text rerank
    Given a candidate set with text nodes
    Then BGE SHALL rerank those nodes within the text stage budget

  Scenario: Visual rerank
    Given a candidate set with image or PDF page-image nodes
    Then SigLIP SHALL rerank those nodes within the visual stage budget

  Scenario: Mixed-modality stage failure
    Given either modality stage fails or times out
    Then reranking SHALL return the exact original candidate order

  Scenario: Router cleanup
    Given a router owns active reranking work
    Then asynchronous router shutdown SHALL drain the owned worker
```

## Verification

Run the focused contract suite:

```bash
uv run pytest -q tests/unit/retrieval/reranking tests/unit/retrieval/test_router_async_tools.py tests/unit/retrieval/test_async_work.py
```
