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

## Final-release invariants (non-negotiable)

1. **No base64 blobs in durable stores** (Qdrant payloads, LangGraph SQLite checkpoints/store, telemetry JSONL).
2. **No raw filesystem paths in durable stores.** Durable references are `ArtifactRef(sha256, suffix)`.
3. **Fail closed during generation**: a new generation activates only after image indexing and exact point-count verification succeed.
4. **Fail open during queries**: visual retrieval falls back to text results, and reranking falls back to the original candidate order.

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
  - “Rebuild search index” reparses the corpus and rebuilds text, image, and optional graph indexes as one generation.
  - “Delete uploaded document” retains content-addressed artifacts so historical chat evidence remains renderable.

## Operations / troubleshooting

### Rebuild image and text indexes

Use **Documents > Maintenance > Rebuild search index**. The rebuild has these guarantees:

- It reparses every retained upload and builds isolated text, image, and optional graph indexes.
- It requires exact text and image point counts before activation.
- If SigLIP or Qdrant fails, the rebuild fails and the active generation remains unchanged.

### Recover missing thumbnails/images in chat

If the UI shows “Image artifact unavailable”:

1. Confirm the artifact store directory exists under `DOCMIND_DATA_DIR`.
2. Confirm the original upload still exists under `DOCMIND_DATA_DIR/uploads`.
3. Run **Documents > Maintenance > Rebuild search index** to render and index the corpus again.

### Artifact retention

Document deletion retains content-addressed artifacts for historical chat evidence. The current application has no automatic garbage collection policy or artifact-purge UI action.

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
uv run pytest tests/unit tests/integration -q --no-cov
```
