---
prompt: PROMPT-043
title: Hybrid Retrieval Logic (Text + Image Fusion)
status: Completed
date: 2026-01-13
version: 1.0
related_adrs: ["ADR-058"]
related_specs: ["SPEC-042"]
---

## Implementation Prompt — 0043: Hybrid Retrieval Logic (Text + Image Fusion)

**Purpose:** Final-release multimodal retrieval: text hybrid + SigLIP visual retrieval + RRF fusion.  
**Source of truth:** ADR-058 + SPEC-042.

## Scope

Implement and/or verify:

- **Server-side Hybrid Prerequisites**:
  - Environment variable `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=true` is set.
  - Qdrant exposes named vectors `text-dense` and `text-sparse` (with an IDF modifier applied) for `ServerHybridRetriever` to work.
  - Implementation uses the Query API with `Prefetch`/`FusionQuery` for fusion.
- `multimodal_search` router/tool is enabled by default so RRF fusion, dedup, deterministic caps, sanitized sources, and contextual recall operate correctly.
- Retrieval fuses:
  - text hybrid retrieval (`ServerHybridRetriever`)
  - image retrieval (SigLIP text→image)
- RRF fusion + dedup are deterministic and capped.
- Returned sources are sanitized (no runtime-only paths) but keep artifact refs.
- Contextual recall works when retrieval returns empty.

## Notes on contextual recall

- Trigger: a “contextual” query (e.g. “that chart”) plus empty retrieval.
- Implementation: `src/agents/tools/retrieval.py` uses `_looks_contextual(...)` to
  detect these queries and `_recall_recent_sources(state)` to reuse the most recent
  persisted sources from `state["synthesis_result"]["documents"]` (preferred) or
  prior `state["retrieval_results"][*]["documents"]`.

## Key modules

- Fusion retriever: `src/retrieval/multimodal_fusion.py`
- Router tool assembly: `src/retrieval/router_factory.py`
- Agent retrieval tool: `src/agents/tools/retrieval.py`
- Reranking policy (optional ColPali): `src/retrieval/reranking.py`

## Step-by-step

1. Verify Qdrant image collection exists in config and is reachable.
2. Ensure `ImageSiglipRetriever` requests thin payload fields only.
3. Ensure `MultimodalFusionRetriever`:
   - fuses via RRF (rank-based)
   - dedups by `settings.retrieval.dedup_key` (default `page_id`)
   - caps outputs by `settings.retrieval.fused_top_k`
4. Ensure agent-visible source docs drop runtime-only paths:
   - `image_path`, `thumbnail_path`, `source_path`
5. Implement/verify contextual recall fallback when user refers to prior context.
6. Verify tests:
   - `uv run python scripts/run_tests.py --fast`

## Acceptance criteria

- “What does that chart show?” queries return image sources when images are indexed.
- Sources contain `image_artifact_id/suffix` and/or thumbnail refs (no base64, no absolute paths).
