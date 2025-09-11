# Hybrid Retrieval Tuning (Quick Guide)

A quick guide to tuning hybrid retrieval for optimal performance.

## Recommended starting points (local-first, BGE-M3 + BM42)

The recommended starting points for hybrid retrieval.

- dense prefetch: 200
- sparse prefetch: 400
- fused_top_k: 60
  
Note: Fusion is performed server‑side via Qdrant Query API (Prefetch + FusionQuery). There are no client‑side fusion knobs (no alpha/rrf_k). DBSF may be enabled via environment when supported.

## Tune by Observing

- `retrieval.latency_ms` — keep within SLO (e.g., ≤ 200 ms)
- `dedup.before/after/dropped` — high `dropped` with low recall may indicate lowering fused_top_k or rebalancing prefetch
- Rerank: gate at `reranking_top_k` 5–16; TEXT timeout 250 ms (CPU), use fp16 on GPU, SigLIP 150 ms budget

## Notes

- Use `DOCMIND_RETRIEVAL__FUSION_MODE=rrf|dbsf` to evaluate DBSF; prefer RRF as default
- BM42 requires sparse `modifier=IDF`; verify collection schema
- Consider enabling server group_by (future) for page/doc dominance; current build dedups by `retrieval.dedup_key`

## When to prefer DBSF (experimental)

- Extremely long-tail keyword distributions with heavy sparse dominance and dense noise.
- Domains with many near-duplicate dense vectors where sparse cues disambiguate best.
- You monitor recall@K on a labeled set and observe consistent improvements at the same latency budget.

## Caveats

- DBSF may over-weight rare token matches on short queries; validate per domain.
- Keep fused_top_k high enough (≥60) to allow DBSF to work; reranker_top_k should remain conservative (5–16).
