# ADR-058: Final Multimodal Pipeline + Cognitive Persistence (End-to-End)

**Status:** Accepted (implemented)  
**Date:** 2026-01-12  
**Supersedes (as integrated source of truth):** ADR-037 (multimodal reranking), ADR-057 (chat persistence + memory)  
**Related:** SPEC-041, SPEC-042, ADR-009, ADR-011, ADR-031, ADR-047, ADR-055

## Dependencies & Security Constraints

- `langgraph-checkpoint-sqlite==3.0.1` (pinned; versions `<3.0.1` have known CVEs affecting checkpoint integrity)
- `sqlite-vec` (runtime dependency used by LangGraph checkpointing; required for vector-backed SQLite)
- Optional multimodal deps (fail-open): `transformers` + `torch` (SigLIP model execution/embeddings; multimodal search is disabled if unavailable)

Verification (upstream snapshot reference): `opensrc/langgraph-checkpoint-sqlite@3.0.1/`.

## Context

DocMind must ship a **final-release** multimodal experience that works end-to-end:

1. **Ingestion** renders PDF page images, produces stable artifacts, and indexes images for retrieval.
2. **Retrieval** can return visually relevant pages for text queries (“what does that chart show?”), and can fuse visual results with text hybrid retrieval.
3. **UI** renders image sources without freezing and without unsafe HTML, and supports query-by-image.
4. **Cognitive persistence** stores conversation state durably (sessions + time travel) and supports long-term memory, **without persisting blobs** (no base64 images in DB/Qdrant).

We previously had multimodal primitives (SigLIP, optional ColPali rerank) but the pipeline was disconnected: no image indexing, no fused retrieval, no UI rendering, and no durable persistence.

## Decision

We implement a **local-first, thin-payload, end-to-end multimodal pipeline** with two key invariants:

1. **Durable stores never persist raw filesystem paths or image blobs.**  
   Durable references are content-addressed `ArtifactRef(sha256, suffix)`.
   - Forbidden: absolute/host-specific paths in Qdrant payloads, LangGraph checkpoints, DocMind store tables, or telemetry JSONL.
   - Allowed: SnapshotManager-controlled snapshot-internal member paths in `manifest.jsonl` required to locate files within a snapshot directory (not used as stable identifiers outside the snapshot boundary).
2. **All multimodal capabilities fail open.**  
   If optional dependencies (SigLIP model, sqlite-vec, ColPali) are unavailable, the app continues with reduced capability.

### Architecture (final)

#### Artifacts

- `ArtifactStore` stores binary artifacts under `settings.data_dir/artifacts` (override supported).
- `ArtifactRef` is the canonical durable pointer; all DB/Qdrant payloads store only `(sha256, suffix)` pairs.

#### Ingestion → Image indexing (SigLIP)

- During ingestion, rendered PDF page images are converted to `ArtifactRef`s.
- Images are indexed into a dedicated Qdrant collection (`settings.database.qdrant_image_collection`) using SigLIP text/image embeddings.
- Qdrant payload is **thin** (ids + artifact refs + page metadata) and explicitly excludes base64 and raw paths.

#### Retrieval → Multimodal fusion

- Text retrieval remains server-side hybrid (Qdrant Query API).
- Image retrieval is SigLIP text→image against the image collection.
- App-level fusion uses rank-based RRF and then existing postprocessors/reranking.
- Retrieval tools sanitize runtime-only paths before returning sources to agent-visible JSON.

#### UI

- Chat page renders “Sources” including page thumbnails via `ArtifactRef` resolution.
- Encrypted images (`*.enc`) are decrypted at render time using a safe context manager (no unsafe HTML).
- Chat provides a sidebar **query-by-image** “Visual search” (SigLIP image→image) for rapid inspection.
- Documents page previews image exports and exposes snapshot rebuild utilities regardless of export presence.

#### Cognitive persistence (ADR-057 integrated)

- LangGraph `SqliteSaver` persists thread state (`messages` + agent outputs).
- A DocMind session registry table provides user-friendly session lists, rename/delete, and hard purge.
- A DocMind `BaseStore` implementation provides long-term memory with optional semantic search.
- “Time travel” is implemented by resuming from a prior checkpoint id (forking a new head).

## Decision framework score (must be ≥ 9.0/10.0)

Weighted decision criteria:

- **Solution leverage (35%)**: 9.2/10  
  Uses LangGraph `SqliteSaver`, LangGraph `BaseStore`, Qdrant Query API, and SigLIP; minimal bespoke code.
- **Application value (30%)**: 9.1/10  
  Delivers true multimodal UX (PDF images discoverable + renderable) and durable chat sessions/time travel.
- **Maintenance & cognitive load (25%)**: 9.0/10  
  Clear separation: artifacts/indexing/retrieval/UI/persistence; fail-open behavior; strong typing + tests.
- **Architectural adaptability (10%)**: 9.1/10  
  ArtifactRef abstraction allows swapping storage backends; retrieval can evolve to server-side late-interaction later.

**Weighted total:** **9.10 / 10.0** (finalized).

## Consequences

### Positive

- Multimodal is now a first-class capability across ingest/retrieve/render/persist.
- No base64 blobs in Qdrant or SQLite; fewer latency and storage risks.
- Persistence/time travel enables “what did that chart show?” across restarts.

### Tradeoffs & Limitations

- **Performance:** RRF (rank-based reciprocal rank fusion) fusion adds latency compared to single-retrieval approaches. Future optimization: implement late-interaction fusion at the Qdrant level for reduced app-side processing.
- **Storage:** Duplicate artifacts (original images + thumbnails) increase storage requirements. Mitigation: aggressive artifact pruning policies and optional S3 offload.
- **Complexity:** Content-addressed artifact storage adds cognitive overhead for developers working with images/artifacts. Mitigation: stable `ArtifactRef` API hides complexity; comprehensive examples in docs.
- **Migration:** Existing deployments with pre-ArtifactRef data require explicit migration path. Approach: implement `migrate_legacy_artifact_paths()` in snapshot upgrade procedures.

### Future Extensions (not required for baseline correctness)

- Server-side late-interaction in Qdrant (ColPali multivectors + MaxSim) can be added as an optional optimization path; current design keeps late-interaction local via reranking.

## Implementation (repo truth)

Key files:

- Artifacts: `src/persistence/artifacts.py`
- Ingestion image indexing: `src/processing/ingestion_pipeline.py`, `src/retrieval/image_index.py`
- Multimodal fusion: `src/retrieval/multimodal_fusion.py`
- UI rendering + visual search: `src/pages/01_chat.py`, `src/pages/02_documents.py`
- Persistence/memory: `src/persistence/chat_db.py`, `src/persistence/memory_store.py`, `src/ui/chat_sessions.py`

## Verification

### Success Criteria

- All linting, formatting, and type-checking passes (ruff, pyright)
- Fast-tier unit tests pass (coverage ≥ 80% for multimodal/persistence code)
- Integration tests validate multimodal ingestion, retrieval, and persistence flows
- No type errors or unsafe type ignores in snapshot/artifact/persistence modules

### Verification Tiers

#### Fast tier (local, <60s)

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run python scripts/run_tests.py --fast
```

**Success criteria:** All checks pass, unit tests complete without errors.

#### Integration tier (local or CI, ~5min)

```bash
uv run python scripts/run_tests.py --integration
```

**Validates:**

- `tests/integration/test_multimodal_ingestion.py` — PDF image rendering, artifact creation, SigLIP indexing
- `tests/integration/test_retrieval_multimodal_fusion.py` — text/image retrieval fusion, RRF scoring
- `tests/integration/test_persistence_chat_db.py` — LangGraph SqliteSaver, session persistence
- `tests/integration/test_memory_store.py` — long-term memory, semantic search with artifacts

#### System tier (GPU, slow, optional)

Full end-to-end validation with real models, GPU-accelerated ingestion, and large-scale persistence tests.

**Validates:** All multimodal paths under realistic load, artifact cleanup, memory leaks.
