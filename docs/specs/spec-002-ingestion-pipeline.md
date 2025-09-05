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

Standardize ingestion using **Unstructured** with `strategy=auto` and `hi_res`/`fast` policy, plus OCR fallback. Use **LlamaIndex IngestionPipeline** with DuckDBKV cache. Emit `pdf_page_image` nodes and maintain deterministic IDs.

## Architecture Notes

- Module: `src/processing/document_processor.py` orchestrates partition → chunk → embed queue.
- Use Unstructured to partition; create page-level image assets for each PDF page where available.
- Use `IngestionPipeline` to apply `SemanticChunker` + `HierarchicalNodeParser` as transformations.
- Deduplicate by SHA256 of normalized text; store lineage and bbox for page-image nodes.

## Libraries and Imports

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticChunker, HierarchicalNodeParser
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from llama_index.core.ingestion import IngestionCache
from PIL import Image
import fitz  # PyMuPDF for page images
```

## File Operations

### UPDATE

- `src/processing/document_processor.py`: add functions `partition_file`, `emit_page_images`, `build_ingestion_pipeline`.
- `src/processing/pdf_pages.py`: implement `save_pdf_page_images(pdf_path, out_dir)` using PyMuPDF.
- `src/models/schemas.py`: extend Node schema with fields: `node_id`, `parent_id`, `page_no`, `bbox`, `modality`.

### CREATE

- `src/models/schemas.py` additions for `PdfPageImageNode` dataclass or Pydantic model.
- `src/utils/storage.py`: helpers to write images and compute SHA256.

## Performance Budgets

- Partition + page-image extraction ≤ 1.5 s per 10 pages on mid hardware.
- Cache hit ratio ≥ 0.7 on re-ingestion.

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

- [ ] Use `strategy="auto"`; when tables/images detected prefer `hi_res` else `fast`.
- [ ] Set OCR agent via Unstructured when needed.
- [ ] Persist DuckDBKV cache at `./cache/docmind.duckdb` with `IngestionCache`.
- [ ] Emit deterministic IDs via `sha256(normalized_text)`.

## Git Plan

- Branch: `feat/ingestion-pipeline`
- Commits:
  1. `feat(ingestion): add Unstructured auto partition with OCR fallback`
  2. `feat(ingestion): page-image extraction and PdfPageImageNode schema`
  3. `feat(ingestion): LlamaIndex IngestionPipeline with DuckDBKV cache`
  4. `refactor(ingestion): deterministic node ids and lineage`

## References

- Unstructured partition strategies; OCR agent setup.
- LlamaIndex IngestionPipeline caching.
