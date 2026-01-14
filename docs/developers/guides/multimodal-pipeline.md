# Multimodal Pipeline (Final Release): PDF Images → Retrieval → UI → Persistence

**Source of truth:** `docs/developers/adrs/ADR-058-final-multimodal-pipeline-and-persistence.md` + `docs/specs/spec-042-final-multimodal-pipeline-and-persistence.md`.

This guide explains how DocMind’s **end-to-end multimodal** path works in production (and how to operate it):

- Render PDF page images (optional encryption)
- Store images as **content-addressed artifacts** (no durable host paths)
- Index page images into Qdrant (`siglip` vectors)
- Fuse text+image results (`multimodal_search`)
- Render image sources in Streamlit
- Persist chat sessions + time travel + memory via LangGraph SQLite

## What “multimodal” means in DocMind (today)

- **Implemented:** text → relevant PDF page images (SigLIP) + text hybrid fusion (RRF) + UI rendering + durable chat persistence.
- **Optional:** ColPali late-interaction rerank (local postprocessor) when installed/enabled.
- **Not implemented (by design):** server-side late-interaction (ColPali multivectors + MaxSim) inside Qdrant.

## Final-release invariants (non-negotiable)

1. **No base64 blobs in durable stores** (Qdrant payloads, LangGraph SQLite checkpoints/store, telemetry JSONL).
2. **No raw filesystem paths in durable stores.** Durable references are `ArtifactRef(sha256, suffix)`.
3. **Fail open**: missing optional deps (SigLIP, sqlite-vec, ColPali) must not break the app.

## Key configuration

### Qdrant collections (text + images)

```bash
DOCMIND_DATABASE__QDRANT_URL=http://localhost:6333
DOCMIND_DATABASE__QDRANT_COLLECTION=docmind_docs
DOCMIND_DATABASE__QDRANT_IMAGE_COLLECTION=docmind_images
DOCMIND_DATABASE__QDRANT_TIMEOUT=60
```

DocMind will create the image collection automatically (if missing) when indexing page images.

### Artifact store (page images + thumbnails)

```bash
# Optional override; default is ${DOCMIND_DATA_DIR}/artifacts
# DOCMIND_ARTIFACTS__DIR=./data/artifacts
DOCMIND_ARTIFACTS__MAX_TOTAL_MB=4096
DOCMIND_ARTIFACTS__GC_MIN_AGE_SECONDS=3600
```

### PDF page images (optional encryption)

```bash
DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=false

# Optional AES-GCM key for encrypting page images (.enc); base64-encoded 32B key
DOCMIND_IMG_AES_KEY_BASE64=
DOCMIND_IMG_KID=local-key-1
DOCMIND_IMG_DELETE_PLAINTEXT=false
```

## UX: where to use multimodal features

- **Chat page:** “Sources” renders image thumbnails via artifact refs (safe + cached via fragments).
- **Chat sidebar:** “Visual search” supports query-by-image (SigLIP image→image) for quick inspection.
- **Documents page:**
  - “Reindex page images” maintenance (recreates SigLIP vectors for all PDFs).
  - “Delete uploaded document” can optionally purge local artifacts (may break old chats).

## Operations / troubleshooting

### Reindex page images

Use the Documents page “Reindex page images” tool. This is safe and idempotent:

- Existing image points for each `doc_id` are purged before upserting to avoid stale points.
- If SigLIP/Qdrant is unavailable, ingestion remains functional (images will be missing until reindexed).

### Recover missing thumbnails/images in chat

If the UI shows “Image artifact unavailable”:

1. Confirm the artifact store directory exists under `DOCMIND_DATA_DIR`.
2. Re-run “Reindex page images” (Documents page).
3. If artifacts were purged, re-ingest the PDF (or reindex after restoring artifacts).

### Artifact purging (GC)

Artifacts are locally GC’d by disk budget and age. If you enable manual artifact purging on delete:

- It only checks refcounts in the **Qdrant image collection** (not LangGraph SQLite).
- Old chats can lose images. Keep the “purge artifacts” checkbox off unless you want aggressive cleanup.

## Code map

- Artifact storage: `src/persistence/artifacts.py`
- PDF page render/export: `src/processing/pdf_pages.py`, `src/utils/images.py`, `src/utils/security.py`
- Ingestion wiring: `src/processing/ingestion_pipeline.py`
- Qdrant image indexing: `src/retrieval/image_index.py`
- Multimodal fusion retriever: `src/retrieval/multimodal_fusion.py`
- Tool sanitization (no paths/base64): `src/agents/tools/retrieval.py`, `src/utils/telemetry.py`
- Streamlit UI: `src/pages/01_chat.py`, `src/pages/02_documents.py`

## Verification

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run python scripts/run_tests.py
```
