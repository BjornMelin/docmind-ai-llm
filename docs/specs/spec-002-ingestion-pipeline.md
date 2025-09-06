---
spec: SPEC-002
title: Document Ingestion with Unstructured + LlamaIndex IngestionPipeline and Page Images
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-ING-001: The system SHALL parse PDFs, Office docs, images using Unstructured.
  - FR-ING-002: The system SHALL emit canonical nodes with deterministic IDs and lineage.
  - FR-ING-003: The system SHALL emit `pdf_page_image` artifacts for visual reranking.
  - NFR-MAINT-001: Use library caching for node+transform hashing.
related_adrs: ["ADR-002","ADR-003","ADR-010"]
---


## Objective

Standardize ingestion using **Unstructured** (`partition(auto)`) with strategy mapping (`hi_res` / `fast` / `ocr_only`), plus OCR fallback. Integrate with **LlamaIndex IngestionPipeline** using a custom `UnstructuredTransformation` and a DuckDBKV‑backed `IngestionCache`. Emit `pdf_page_image` nodes and maintain deterministic IDs for all emitted nodes.

## Architecture

- Entry point: `src/processing/document_processor.py`.
- Transformation: `UnstructuredTransformation` (a LlamaIndex `TransformComponent`) performs:
  - `unstructured.partition.auto.partition(path, **config)` to extract elements
  - Heuristic chunking:
    - Prefer `unstructured.chunking.title.chunk_by_title` when title density ≥ 5% or ≥ 3 titles
    - Else fallback to `unstructured.chunking.basic.chunk_elements`
  - Preserve Unstructured metadata (page_number, coordinates, HTML, image path)
  - Convert chunks to `llama_index.core.Document` nodes with deterministic `doc_id`
- Caching: `IngestionCache` with `DuckDBKVStore` at `./cache/docmind.duckdb`
- Page images: `src/processing/pdf_pages.py::save_pdf_page_images()` renders stable `__page-<n>.png` files (idempotent) and returns `page_no`, `image_path`, `bbox`; `DocumentProcessor` appends `pdf_page_image` metadata elements to the result
- Deterministic IDs: `src/processing/utils.py::sha256_id()` normalizes and hashes `(source_path, page_no, text_or_image_hash)`

### Rationale & Alternatives (Library‑First)

- Why not use `partition(..., chunking_strategy=...)` end‑to‑end?
  - Unstructured exposes `chunking_strategy` on some partitioners, but does not surface all tuning knobs (e.g., `combine_text_under_n_chars`, `multipage_sections`) across file types via the `partition` facade. Selecting chunker explicitly (`chunk_by_title` vs `basic`) keeps the heuristic and parameterization in our control while still leveraging the built‑in chunkers.
  - We considered `llama_index.readers.file.unstructured` readers; however, integrating Unstructured directly as a `TransformComponent` lets us (a) preserve full Unstructured metadata, (b) control chunking heuristics, and (c) plug into `IngestionCache` seamlessly. This reduces custom code elsewhere and centralizes ingestion logic.
  - LlamaIndex splitters (e.g., `SentenceSplitter`, `TokenTextSplitter`) are powerful, but our target is layout‑aware chunking that Unstructured already provides (by‑title/table‑aware). We use LlamaIndex only for orchestration/caching.

- Why DuckDBKVStore for caching?
  - Official LlamaIndex supports KV stores (Redis/DuckDB). DuckDBKVStore is local‑first, requires zero services, and works well for CI and offline testing. It also keeps the cache as a single file (`./cache/docmind.duckdb`), simplifying lifecycle and stats.

## Key Libraries

- `unstructured.partition.auto.partition`
- `unstructured.chunking.title.chunk_by_title`, `unstructured.chunking.basic.chunk_elements`
- `llama_index.core.ingestion.IngestionPipeline`, `IngestionCache`
- `llama_index.storage.kvstore.duckdb.DuckDBKVStore`
- `fitz` (PyMuPDF) for page rendering

See also: LlamaIndex TransformComponent examples and KV cache usage in the official docs; Unstructured chunker signatures (`chunk_by_title`, `chunk_elements`) for supported parameters.

## Touched Files

- `src/processing/document_processor.py`: UnstructuredTransformation, strategy mapping, caching, page-image emission, deterministic IDs
- `src/processing/pdf_pages.py`: `save_pdf_page_images(pdf_path, out_dir, dpi=180)` with stable naming and bbox
- `src/processing/utils.py`: `sha256_id()` normalization + hashing; `is_unstructured_like()` guard for test doubles
- `src/models/schemas.py`: `PdfPageImageNode` model (page_no, bbox, modality, source_path, hash); `ErrorResponse` enriched (traceback, suggestion)

## Performance Budgets

- Partition + page-image extraction: target ≤ 1.5 s per 10 pages (mid hardware)
- Re-ingestion cache hit ratio: target ≥ 0.7

## Acceptance Criteria

```gherkin
Feature: Ingestion pipeline
  Scenario: PDF with tables
    Given a multi-page PDF with tables
    When I ingest with strategy auto
    Then nodes SHALL include text chunks and pdf_page_image entries
    And subsequent runs SHALL reuse cache entries

  Scenario: OCR fallback
    Given a scanned PDF
    Then partition SHALL use OCR agent and produce text chunks
```

## Detailed Checklist

- [x] Use `strategy="auto"` policy implemented via strategy mapping and configuration per file type; pages with tables/images handled with high‑resolution path; scanned/no text handled via OCR path; text‑dominant via fast path.
- [x] OCR path available for scanned/image‑only inputs; text nodes are produced under OCR.
- [x] Persist DuckDBKV cache at `./cache/docmind.duckdb` through `IngestionCache`.
- [x] Emit deterministic IDs using SHA‑256 of normalized content + source/page lineage; include page‑image nodes with bbox.

## RTM (Requirements Traceability)

- FR-ING-001 (parse via Unstructured):
  - Code: `src/processing/document_processor.py` (partition, strategy mapping)
  - Tests: `tests/integration/test_chunking_integration.py`, `tests/unit/processing/*`
- FR-ING-002 (deterministic IDs & lineage):
  - Code: `src/processing/utils.py::sha256_id`, `src/processing/document_processor.py` (node_id, parent_id)
  - Tests: `tests/unit/processing/test_deterministic_ids_unit.py` (added), chunking unit/integration assertions
- FR-ING-003 (page images):
  - Code: `src/processing/pdf_pages.py`, `src/processing/document_processor.py` emission with modality=pdf_page_image
  - Tests: `tests/integration/test_ingestion_pipeline_pdf_images.py`, `tests/unit/processing/test_pdf_pages_unit.py`

## Validation & Quality

- Fast tests: `uv run python scripts/run_tests.py --fast`
- Full coverage: `uv run python scripts/run_tests.py --coverage`
- Lint/format: `uv run ruff format . && uv run ruff check . --fix`
- Pylint (target ≥ 9.5): `uv run pylint --fail-under=9.5 src tests scripts`
- CI quality gates: `uv run python scripts/run_quality_gates.py --ci --report`
  - Note: current project-wide coverage baseline is ~70.5%; CI coverage gate is configured at 80% and may fail until broader areas gain tests

## References

- Unstructured partition strategies (auto/hi_res/fast) and OCR fallback
- LlamaIndex IngestionPipeline + IngestionCache + DuckDBKVStore
- PyMuPDF page rendering best practices (idempotent writes, bbox capture)
