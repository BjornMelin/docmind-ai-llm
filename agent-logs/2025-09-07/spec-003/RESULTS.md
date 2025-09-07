# SPEC-003 Results Summary

Date: 2025-09-07
Branch: feat/embeddings

## Outcomes
- Text tri-mode via LlamaIndex BGEM3Index/BGEM3Retriever (weights [0.4,0.2,0.4]).
- Images via LlamaIndex ClipEmbedding (OpenCLIP); SigLIP only outside LI.
- Removed all legacy wrappers; no back-compat stubs remain.
- Image dims derived at runtime; no hard-coded dims in active code.

## Tests & Quality
- Tests: 1218 passed, 1 skipped, 5 xfailed, 4 xpassed.
- Coverage: 68.23% (≥65%).
- Lint: ruff clean; pylint 9.64/10.

## Artifacts
- Factories: `src/retrieval/bge_m3_index.py` (index/retriever builders).
- Integrations: `src/config/integrations.py` uses LI ClipEmbedding.
- Fallbacks for offline determinism: fastembed optional (dense-only fallback), managed-index fallback to vector index.

## Traceability
- FR-004: Completed — see `docs/specs/traceability.md` row; tests listed.
- SPEC-003: Status Tracking shows Phases 1–5 Completed in `docs/specs/spec-003-embeddings.md`.

## Suggested follow-ups
- Optional: expose tri-mode weights in UI for advanced users.
- Optional: enable Visualized-BGE behind a capability flag only for supported GPUs.

