# Implementation Prompt — 0043: Hybrid Retrieval Logic (Text + Image Fusion)

**Purpose:** Final-release multimodal retrieval: text hybrid + SigLIP visual retrieval + RRF fusion.  
**Source of truth:** ADR-058 + SPEC-042.

## Scope

Implement and/or verify:

- `multimodal_search` router tool exists and is enabled by default.
- Retrieval fuses:
  - text hybrid retrieval (`ServerHybridRetriever`)
  - image retrieval (SigLIP text→image)
- RRF fusion + dedup are deterministic and capped.
- Returned sources are sanitized (no runtime-only paths) but keep artifact refs.
- Contextual recall works when retrieval returns empty.

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

