Title: Plan 004 — Consolidated Reranking/Multimodal Refactor (Authoritative)

Objective
- Centralize device/VRAM policy via utils.core; unify SigLIP adapter to shared loader; enforce minimal reranking telemetry; align ADRs/SPECs/traceability; pass lint/tests.

Phases & Checkboxes

Phase A — Centralize Device/VRAM Policy [core][emb][mm][qa]
- [x] Implement utils.core.has_cuda_vram and utils.core.select_device (optional torch-safe)
- [x] Update embeddings.ImageEmbedder._choose_auto_backbone to use has_cuda_vram thresholds
- [x] Guard utils.multimodal.validate_vram_usage with has_cuda_vram
- [x] Unit tests for select_device and has_cuda_vram

Phase B — Unify SigLIP Adapter [mm][core][qa]
- [x] Refactor utils.siglip_adapter.SiglipEmbedding to reuse utils.vision_siglip.load_siglip
- [x] Delegate device choice to core.select_device with offline-safe fallback
- [x] Validate device/caching behavior via existing tests; add unit tests if needed

Phase C — Documentation Alignment [docs]
- [x] ADR-037: add note on centralized device/VRAM policy and optional ProcessPool fallback
- [x] Traceability: add src/utils/core.py under FR-004 mapping
- [x] Configuration reference: document DEVICE_POLICY_CORE, SIGLIP_ADAPTER_UNIFIED, RERANK_EXECUTOR flags
- [x] Integration README: update reranking examples to use build_text_reranker (SentenceTransformerRerank) and remove CrossEncoder mentions
- [x] CHANGELOG: summarize consolidation and device policy centralization

Phase D — Telemetry Conformance Sweep [qa][core][mm]
- [x] Verify minimal reranking telemetry schema (stage, topk, latency_ms, timeout; final may include delta_changed_count, path, total_timeout_budget_ms)
- [x] Ensure no rerank batch-size fields are logged

Phase E — Quality Gates [qa]
- [x] Ruff format and lint
- [x] Pylint on touched modules (minor long-line warnings remain acceptable)
- [x] Tests via scripts/run_tests.py (unit + integration)

Notes
- Deterministic ordering retained in reranking (score desc, id asc) and RRF tie-break.
- Encrypted image helper already in use across reranking and pdf page I/O.
- Plan 004 supersedes 003; remaining 002 items merged here.

