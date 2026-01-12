---
spec: SPEC-002
title: Document Ingestion (LlamaIndex IngestionPipeline + PDF Page Images)
version: 1.1.0
date: 2026-01-12
owners: ["ai-arch"]
status: Revised
related_requirements:
  - FR-ING-001: The system SHALL parse PDFs and common office/text formats using library-first loaders.
  - FR-ING-002: The system SHALL emit canonical nodes with stable identifiers and safe metadata.
  - FR-ING-003: The system SHALL export PDF page images for multimodal retrieval and UI rendering.
  - NFR-MAINT-001: Use library caching for node+transform hashing.
related_adrs: ["ADR-002","ADR-003","ADR-030","ADR-031","ADR-058"]
---

## Objective

Provide a **final-release, library-first** ingestion pipeline that:

1. Loads documents via **LlamaIndex** primitives (UnstructuredReader when installed; safe plaintext fallback).
2. Chunks text via `TokenTextSplitter` and optionally enriches nodes via `TitleExtractor`.
3. Uses `IngestionCache` (DuckDB KV) for deterministic reuse.
4. Exports **PDF page images** (WebP/JPEG, optional `*.enc` encryption).
5. Wires multimodal exports into the final-release multimodal stack:
   - store images as content-addressed artifacts (`ArtifactRef(sha256, suffix)`),
   - best-effort index images into Qdrant for SigLIP retrieval,
   - never persist base64 blobs or host paths in durable stores.

This spec describes ingestion and page-image export/index wiring. For full end-to-end
multimodal behavior and persistence invariants, see:

- `docs/developers/adrs/ADR-058-final-multimodal-pipeline-and-persistence.md`
- `docs/specs/spec-042-final-multimodal-pipeline-and-persistence.md`

## Final-release invariants (must hold)

1. **No base64 blobs in durable stores** (Qdrant payloads, LangGraph SQLite checkpoints/store, telemetry JSONL).
2. **No raw filesystem paths in durable stores.** Durable references use `ArtifactRef(sha256, suffix)`.
3. **Fail open**: optional dependencies (UnstructuredReader, SigLIP, Qdrant) must not break ingestion.

## Architecture (repo truth)

### Entry points

- Ingestion pipeline assembly and execution: `src/processing/ingestion_pipeline.py`
  - `build_ingestion_pipeline(cfg, embedding=...)`
  - `_load_documents(cfg, inputs)` (UnstructuredReader + fallback)
  - `_page_image_exports(...)` (PDF page images via PyMuPDF)
  - `_index_page_images(...)` (ArtifactStore + SigLIP + Qdrant)

### Document loading (library-first)

- Preferred: `llama_index.readers.file.UnstructuredReader` (when installed)
- Fallback: plaintext `Path.read_text(...)` for UTF-8-ish inputs
- Metadata hygiene:
  - drop path-like metadata keys (`source_path`, `file_path`, `path`)
  - normalize `metadata["source"]` to a basename when it is path-like

### Chunking and enrichment

Pipeline transforms (in order):

1. `TokenTextSplitter(chunk_size, chunk_overlap, separator="\n")`
2. Optional: `TitleExtractor(show_progress=False)` (best-effort; skip on failure)
3. Optional: embedding transform (resolved from settings when not injected)

### Caching / docstore

- Cache: `IngestionCache` backed by `DuckDBKVStore` at `${cache_dir}/${cache_filename}`
  - defaults to `./cache/docmind.duckdb` (see `DOCMIND_CACHE__DIR`, `DOCMIND_CACHE__FILENAME`)
- Docstore: `SimpleDocumentStore` with optional persistence when configured by `IngestionConfig`

### PDF page images (export)

- Rendering: `src/processing/pdf_pages.py::save_pdf_page_images(...)`
  - Stable naming: `<stem>__page-<n>.(webp|jpg)[.enc]`
  - Optional encryption: `DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=true` (writes `*.enc`)
  - Best-effort `page_text`, `bbox`, `phash` in returned metadata
- Export packaging: `src/models/processing.py::ExportArtifact`
  - `ExportArtifact.path` is runtime-only (excluded from serialization) to prevent accidental
    persistence of host paths.

### Artifacts (durable image references)

- Content-addressed artifact store: `src/persistence/artifacts.py`
  - Stores page images and thumbnails under `data/artifacts` by default.
  - Durable reference: `ArtifactRef(sha256, suffix)`

### Image indexing (best-effort; part of ingestion)

Ingestion performs a best-effort post-step for `image/*` exports:

1. Convert image export files → `ArtifactRef` via `ArtifactStore.put_file(...)`
2. Generate thumbnail (best-effort) → `ArtifactRef`
3. Index into Qdrant image collection (`settings.database.qdrant_image_collection`)
   using SigLIP embeddings:
   - helper: `src/retrieval/image_index.py`
   - payload is **thin**: ids + artifact refs + page metadata
   - no base64 and no raw filesystem paths

## Configuration (canonical env vars)

```bash
# Cache
DOCMIND_CACHE__DIR=./cache
DOCMIND_CACHE__FILENAME=docmind.duckdb

# PDF page images (optional encryption)
DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=false
DOCMIND_IMG_AES_KEY_BASE64=
DOCMIND_IMG_KID=local-key-1
DOCMIND_IMG_DELETE_PLAINTEXT=false

# Artifact store (content-addressed)
# DOCMIND_ARTIFACTS__DIR=./data/artifacts
DOCMIND_ARTIFACTS__MAX_TOTAL_MB=4096
DOCMIND_ARTIFACTS__GC_MIN_AGE_SECONDS=3600

# Qdrant collections
DOCMIND_DATABASE__QDRANT_URL=http://localhost:6333
DOCMIND_DATABASE__QDRANT_COLLECTION=docmind_docs
DOCMIND_DATABASE__QDRANT_IMAGE_COLLECTION=docmind_images
DOCMIND_DATABASE__QDRANT_TIMEOUT=60
```

## Acceptance criteria

```gherkin
Feature: Ingestion pipeline

  Scenario: PDF ingestion exports page images
    Given a multi-page PDF
    When I ingest the file
    Then the ingestion result includes image exports for each page
    And exported images use deterministic names (<stem>__page-<n>.*)

  Scenario: Metadata hygiene
    Given an input with a local file path
    When I ingest the file
    Then no durable output contains raw filesystem paths
    And any path-like metadata["source"] is normalized to a safe basename

  Scenario: Fail-open image indexing
    Given a PDF and Qdrant is unavailable
    When I ingest the file
    Then ingestion succeeds and returns text outputs
    And image indexing reports enabled but zero indexed (best-effort)
```

## Touched files (canonical)

- `src/processing/ingestion_pipeline.py`
- `src/processing/pdf_pages.py`
- `src/persistence/artifacts.py`
- `src/retrieval/image_index.py`
- `src/utils/images.py`
- `src/utils/security.py`
- `src/utils/document.py`
- `src/models/processing.py`

## Tests (repo truth)

- Integration:
  - `tests/integration/test_ingestion_pipeline_pdf_images.py`
- Unit:
  - `tests/unit/processing/test_ingestion_pipeline.py`
  - `tests/unit/processing/test_reindex_purges_stale_image_points.py`
  - `tests/unit/persistence/test_artifact_store.py`
  - `tests/unit/utils/test_images_thumbnail_and_encrypted_open.py`

## Validation & Quality

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run python scripts/run_tests.py --fast
```

