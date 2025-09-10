# Title: Final Decisions, ADR/SPEC Updates, and RTM Corrections

**Date:** 2025-09-09

## Hybrid Retrieval (SPEC‑004; ADR‑005/006/024) — Checklist

- [x] Server‑side only with Qdrant Query API fusion (RRF default; DBSF optional).
- [x] Named vectors `text-dense` (BGE‑M3 1024 COSINE) and `text-sparse` (BM42/BM25) are required.
- [x] Internal sparse fallback: FastEmbed BM42→BM25→None; emit telemetry flag `retrieval.sparse_fallback` when sparse unavailable.
- [x] De‑dup by `page_id` before final fused cut; configurable `dedup_key`.
- [x] No LanceDB fallback.

## UI (SPEC‑008; ADR‑012/016/013)

- Programmatic pages via `st.Page` + `st.navigation`.
- Chat: `st.chat_message`/`st.chat_input` + `st.write_stream` (with best‑effort chunking fallback). When a router engine is available in session, pass it via `settings_override` to the coordinator.
- Documents: form‑based ingestion + `st.status` + `st.toast`.
- Analytics: charts from local DuckDB when enabled.

## Analytics (ADR‑032)

- Separate DuckDB file at `settings.analytics_db_path or data/analytics/analytics.duckdb`.
- Non‑blocking writes via background worker; hourly retention pruning.
- Tables: `query_metrics`, `embedding_metrics`, `reranking_metrics`, `system_metrics`.

## Evaluation Harness (SPEC‑010; ADR‑039 complementing ADR‑012)

- BEIR: IR metrics (`NDCG@10`, `Recall@k`, `MRR`) on tiny datasets; TREC runfile + JSON metrics + CSV leaderboard.
- RAGAS: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`; local evaluator wrapper if needed; CSV leaderboard.
- Keep DeepEval in CI for quick guardrails.

## Model Management (SPEC‑013; ADR‑040)

- Minimal CLI using `hf_hub_download` for default models (BGE‑M3, BGE reranker, SigLIP, BM42) and `--add` pairs.
- Optional enhancement: manifest generation and `snapshot_download` full repo mirroring.

## GraphRAG (SPEC‑006; ADR‑019) — Checklist

- [x] Update SPEC‑006 to library‑first traversal via `get_rel_map`, add router composition (vector+graph with fallback), and SnapshotManager integration (SPEC‑014). Add staleness badge AC.
- [x] Provide exports: JSONL baseline (1 relation per line) with optional Parquet (PyArrow).
- [x] Add Documents page toggle to enable GraphRAG on demand; exports are available via UI.

## New ADR/SPEC — Checklist

- [x] Create ADR‑038 (GraphRAG router + SnapshotManager) and SPEC‑014 (Index Persistence Snapshots), cross‑link in SPEC‑006, requirements, and RTM.

## Observability & Security (SPEC‑012; ADR‑024, ADR‑038/039 from 005)

- JSONL telemetry with required fields; enrich reranking events (time budgets, chosen path).
- Endpoint allowlist; default egress OFF; redacted logs; AES‑GCM for image payloads; restrict HTTP clients where used.

## RTM Corrections (docs/specs/traceability.md) — Checklist

- [x] FR‑010 Streamlit multipage → Implemented.
- [x] FR‑009 GraphRAG PropertyGraphIndex → Implemented (exports/toggles/tests).
- [x] Add rows for BEIR/RAGAS harness and model pull CLI; set to Implemented post‑merge.

## ADR Status Changes — Checklist

- [x] ADR‑032 → Implemented with DB path/retention details.
- [x] ADR‑039 → Implemented (Offline Evaluation Harness) alongside ADR‑012.
- [x] ADR‑040 → Implemented (Model Pre‑download CLI).

## No Backwards Compatibility (Global)

- Do not keep legacy code paths or toggles. Replace and remove obsolete code rather than deprecating silently.
- Update all imports and references across the codebase when relocating functionality (e.g., UI from `src/app.py` to `src/pages/*`).
- Remove any redundant helpers or dead code discovered during refactors.
