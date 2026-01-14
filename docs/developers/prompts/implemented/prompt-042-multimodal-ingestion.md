---
prompt: PROMPT-042
title: Multimodal Ingestion (PDF page images → ArtifactRef → Qdrant)
status: Completed
date: 2026-01-13
version: 1.0
related_adrs: ["ADR-058"]
related_specs: ["SPEC-042"]
---

## Implementation Prompt — 0042: Multimodal Ingestion (PDF page images → ArtifactRef → Qdrant)

**Purpose:** Final-release ingestion wiring for multimodal PDFs (page images).  
**Source of truth:** ADR-058 + SPEC-042.  
**Key constraints:** no base64 blobs in durable stores; no raw paths persisted; fail-open.

## Scope

Implement and/or verify:

- PDF page images are rendered during ingestion.
- Images + thumbnails are stored as `ArtifactRef` via `ArtifactStore`.
- Qdrant image collection is created/updated and points are upserted with SigLIP vectors.
- Payload is thin and includes artifact refs, not raw paths.
- Artifacts GC runs best-effort (disk budget) after indexing.

## Repo entry points

- Ingestion: `src/processing/ingestion_pipeline.py`
- Page image rendering: `src/processing/pdf_pages.py`, `src/utils/images.py`
- Artifacts: `src/persistence/artifacts.py`
- Qdrant image indexing: `src/retrieval/image_index.py`

## Step-by-step

1. **Confirm settings wiring**
   - Image collection name: `settings.database.qdrant_image_collection`
   - Artifacts root: `settings.data_dir/artifacts` (or override)
2. **Ensure exports include modality + page info**
   - `modality == "pdf_page_image"`
   - `doc_id`, `page_no`, `phash`, `bbox`, and best-effort `page_text`
3. **Convert exports to ArtifactRefs**
   - Call `ArtifactStore.put_file` for image + thumbnail.
   - Write back `image_artifact_id/suffix` and `thumbnail_artifact_id/suffix` into export metadata.
4. **Index into Qdrant**
   - Create/ensure image collection (SigLIP vector config).
   - Upsert with thin payload (no base64, no paths).
5. **Run artifact GC**
   - `ArtifactStore.prune` using `settings.artifacts.*` disk budget config.
6. **Verify**
   - `uv run python scripts/run_tests.py --fast`

## Acceptance criteria

- Ingestion returns successfully even when SigLIP/Qdrant is unavailable (fail-open).
- Qdrant payload contains artifact ids/suffixes and page metadata only.
- UI can render thumbnails by resolving `ArtifactRef`.
