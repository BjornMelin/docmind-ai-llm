---
spec: SPEC-002
title: Ingest documents with the local parser and LlamaIndex
version: 1.7.0
date: 2026-07-11
owners: ["ai-arch"]
status: Revised
related_requirements:
  - FR-ING-001: The system SHALL parse PDFs and common office/text formats using library-first loaders.
  - FR-ING-002: The system SHALL emit canonical nodes with stable identifiers and safe metadata.
  - FR-ING-003: The system SHALL export PDF page images for multimodal retrieval and UI rendering.
  - NFR-MAINT-001: Use library caching for node+transform hashing.
related_adrs: ["ADR-002","ADR-003","ADR-030","ADR-031","ADR-034","ADR-058"]
---

## Objective

Define the production ingestion pipeline:

1. Loads documents via the **DocMind parser service**:
   Docling conversion, pypdfium2 PDF inspection/rasterization, and RapidOCR
   CPU OCR, with direct UTF-8 loading restricted to canonical text formats.
2. Chunks text via `TokenTextSplitter` and optionally enriches nodes with spaCy.
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

## Ingestion invariants

1. **No base64 blobs in durable stores** (Qdrant payloads, LangGraph SQLite checkpoints/store, telemetry JSONL).
2. **No raw filesystem paths in durable stores.** Durable references use `ArtifactRef(sha256, suffix)`.
3. **Fail closed at the parsing boundary**: PDF and other binary parser failures
   raise `DocumentParseError`; source bytes MUST NOT be decoded as plaintext or
   published as partial documents.
4. **Fail open after successful parsing**: optional post-parse enrichment,
   searchable-PDF export, visual indexing, SigLIP, and recoverable vector or
   graph indexing failures may be skipped without invalidating already parsed
   text. Contract and integrity failures, such as an incompatible Qdrant schema
   or failed stale-point cleanup, still propagate.
5. **Bound parser resources**: validate source bytes, page count, render pixels,
   and total extracted text. Async ingestion runs parsing in a killable worker
   process and terminates it when `parse_timeout_seconds` expires.
6. **Use prefetched parser models**: PDF parsing validates each cached file
   against the canonical model manifests. Health output reports relative paths
   and integrity reasons without exposing cache roots.
7. **Preserve canonical document ownership**: every text node carries the base
   document ID under `docmind_document_id`. LlamaIndex owns and may rewrite its
   generic `document_id`, `doc_id`, and `ref_doc_id` payload fields.
8. **Make re-ingestion idempotent**: text points use deterministic UUIDs. After
   a successful Qdrant upsert, remove only prior point IDs for the same canonical
   documents that are absent from the new node set.
9. **Reject ambiguous batch ownership before I/O**: every `document_id` in one
   ingestion call is unique, and duplicates fail before embedding resolution,
   pipeline construction, parsing, or filesystem access.
10. **Use complete, content-sensitive identities**: file-derived document IDs
    use `doc-<full lowercase SHA-256>`. The corpus identity includes the file
    corpus hash plus inline payload entries ordered by `document_id`, with each
    entry represented by `(document_id, sha256(payload_text))`.

## Architecture

### Entry points

- Ingestion pipeline assembly and execution: `src/processing/ingestion_pipeline.py`
  - `build_ingestion_pipeline(cfg, embedding=...)`
  - `_load_documents(cfg, inputs)` (loads via `src/processing/ingestion_api.py`)
  - `_page_image_exports(...)` (PDF page images via pypdfium2)
  - `_index_page_images(...)` (ArtifactStore + SigLIP + Qdrant)

### Implementation owners

The current implementation lives in these modules:

- `src/processing/ingestion_pipeline.py`
- `src/ui/_ingest_adapter_impl.py`
- `src/processing/pdf_pages.py`
- `src/persistence/artifacts.py`
- `src/retrieval/image_index.py`

### Document loading (local parser first)

- Preferred: `src/processing/parsing/service.py::parse_document(...)`
  - Plain text/Markdown: direct text read.
  - PDFs: pypdfium2 inspection plus Docling conversion; low-text pages are
    OCRed with the canonical CPU-safe RapidOCR path when native text is sparse.
  - Office/HTML/multiformat documents: Docling conversion.
- Direct text is read only by the parser service and only for `.txt`, `.md`,
  `.markdown`, and `.rst` inputs. The ingestion facade has no second text
  fallback path.
- In-memory ingestion accepts canonical `payload_text: str` only. Binary inputs
  require a source path and format-aware parsing.
- Parser-boundary failure: inspection, dependency or model readiness, conversion,
  required OCR and page-fidelity work, and required parser post-processing
  propagate as `DocumentParseError` before LlamaIndex documents or artifacts are
  published.
- Optional post-parse failure: searchable-PDF export and the best-effort
  enrichment and indexing stages follow the fail-open contract above; they are
  not reclassified as `DocumentParseError` after canonical text parsing succeeds.
- Model readiness: `scripts/parser_health.py --check` verifies dependency imports
  and every file in the source-controlled Docling and RapidOCR manifests.
- Metadata hygiene:
  - drop path-like metadata keys (`source_path`, `file_path`, `path`)
  - normalize `metadata["source"]` to a basename when it is path-like.
  - parser provenance is excluded from embedding/LLM metadata strings so
    chunking remains stable.
  - direct-text provenance does not claim PDF or OCR backends that did not run.
  - `docmind_document_id` is parser-owned, excluded from embedding text, and
    retained in Qdrant payloads for document-scoped replacement and deletion.

### Chunking and enrichment

Pipeline transforms (in order):

1. `TokenTextSplitter(chunk_size, chunk_overlap, separator="\n")`
2. Optional: spaCy NLP enrichment (sentences + entities) via `SpacyNlpEnrichmentTransform` (see `docs/specs/spec-015-nlp-enrichment-spacy.md`)
3. Optional: embedding transform (resolved from settings when not injected)

### Caching and replacement ownership

- Cache: `IngestionCache` backed by `DuckDBKVStore` at `${cache_dir}/ingestion/${cache_filename}`
  - defaults to `./cache/ingestion/docmind.duckdb` (see `DOCMIND_CACHE__DIR`, `DOCMIND_CACHE__FILENAME`)
- The transformation pipeline does not attach a LlamaIndex docstore. Docstore
  duplicate filtering omits unchanged nodes from replacement batches, whereas
  the cache replays a repeated batch's full transformed node set without
  recomputing it.
- Text indexing: `src/ui/_ingest_adapter_impl.py` assigns deterministic UUIDv5
  point IDs from canonical document ID, page ID, and chunk position. It captures
  existing point IDs only for successfully loaded canonical documents, inserts
  through `VectorStoreIndex`, and deletes the stale subset only after insertion
  succeeds. This scoped lifecycle is safe for partial upload batches;
  whole-docstore `UPSERTS_AND_DELETE` is not.

### Breaking full-SHA identity migration

The full-SHA document identity and `docmind_document_id` ownership key are a
forward-only break from builds that used `doc-<sha256[:16]>` and generic
LlamaIndex ownership fields. There is no compatibility lookup for legacy IDs:
new ingestion and deletion cannot identify old Qdrant points as the same
documents.

Before upgrading an existing corpus:

1. Stop every ingestion and indexing writer.
2. Back up any state that must be retained, then select fresh Qdrant text and
   image collection names or explicitly remove the legacy collections.
3. Remove the legacy ingestion DuckDB cache and persisted LlamaIndex docstore;
   cached nodes may contain truncated IDs or omit `docmind_document_id`.
4. Retire snapshots built from the legacy collections, re-ingest every source,
   and rebuild the active snapshot before serving queries.

Old snapshots and chats may retain legacy node or artifact references. Preserve
their backing data when historical rendering matters; they are not rewritten by
the migration. The operator-facing checklist is also published in the
[README migration section](../../README.md#full-sha-ingestion-identity-migration).

### PDF page image export

- Rendering: `src/processing/pdf_pages.py::save_pdf_page_images(...)`
  using pypdfium2.
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

Searchable-PDF export is separate from parsing. It is an optional fail-open
OCRmyPDF and Tesseract step. It requires POSIX process groups, so Windows users
must run it under WSL2. Cancellation, timeout, and failure paths kill and reap
the full subprocess group.

## Canonical environment variables

```bash
# Cache
DOCMIND_CACHE__DIR=./cache
DOCMIND_CACHE__FILENAME=docmind.duckdb

# PDF page images (optional encryption)
DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=false
DOCMIND_IMG_AES_KEY_BASE64=
DOCMIND_IMG_KID=local-key-1
DOCMIND_IMG_DELETE_PLAINTEXT=false

# Fixed validation literals (not operator-selectable)
DOCMIND_PARSING__FRAMEWORK=docling
DOCMIND_PARSING__PROFILE=cpu_safe
# Parser/OCR resource controls
DOCMIND_PARSING__MAX_PAGES=500
DOCMIND_PARSING__MAX_RENDER_PIXELS=40000000
DOCMIND_PARSING__MAX_TOTAL_TEXT_CHARS=10000000
DOCMIND_PARSING__PARSE_TIMEOUT_SECONDS=300
DOCMIND_PARSING__OCRMYPDF_TIMEOUT_SECONDS=300
DOCMIND_PDF_BACKEND__RENDER_DPI=200
DOCMIND_PDF_BACKEND__MIN_TEXT_CHARS_PER_PAGE=24
DOCMIND_OCR__ENGINE=rapidocr
DOCMIND_OCR__MODEL_CACHE_DIR=cache/models
DOCMIND_OCR__OCRMYPDF_JOBS=1
DOCMIND_OCR__SEARCHABLE_PDF_ENABLED=false

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

`DOCMIND_PARSING__FRAMEWORK`, `DOCMIND_PARSING__PROFILE`, and
`DOCMIND_OCR__ENGINE` state immutable validation literals, not backend selectors.
Only `docling`, `cpu_safe`, and `rapidocr`, respectively, are accepted.

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

  Scenario: Fail-closed malformed PDF
    Given a malformed or unreadable PDF
    When I ingest the file
    Then ingestion raises DocumentParseError
    And no Document, node, image artifact, or snapshot is published

  Scenario: Reject duplicate document ownership before ingestion work
    Given two inputs with the same document ID
    When I ingest the batch
    Then ingestion raises ValueError before embedding resolution or source I/O

  Scenario: Inline payload content participates in corpus identity
    Given two inline text payloads with unique document IDs
    When their input order changes
    Then the corpus hash is unchanged
    When either document's text changes
    Then the corpus hash changes

  Scenario: Re-ingest a changed multi-page document
    Given Qdrant contains text points for a canonical document ID
    When I ingest a changed version with the same document ID
    Then current deterministic point IDs are upserted
    And stale points for that document are deleted after the upsert
    And points owned by other documents are unchanged
```

## Canonical files

- `src/processing/ingestion_pipeline.py`
- `src/processing/ingestion_api.py`
- `src/ui/_ingest_adapter_impl.py`
- `src/processing/pdf_pages.py`
- `src/persistence/artifacts.py`
- `src/retrieval/image_index.py`
- `src/utils/images.py`
- `src/utils/security.py`
- `src/models/processing.py`
- `src/utils/hashing.py`

## Tests

- Integration:
  - `tests/integration/test_ingestion_pipeline_pdf_images.py`
- Unit:
  - `tests/unit/processing/test_ingestion_pipeline.py`
  - `tests/unit/ui/test_ingest_adapter_return_shape.py`
  - `tests/unit/processing/test_reindex_purges_stale_image_points.py`
  - `tests/unit/persistence/test_artifact_store.py`
  - `tests/unit/utils/test_images_thumbnail_and_encrypted_open.py`

## Validation

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run python scripts/run_tests.py --fast
```
