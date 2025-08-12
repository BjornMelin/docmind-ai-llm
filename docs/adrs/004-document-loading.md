# ADR-004: Document Loading

## Title

Offline Document Loading and Parsing Strategy

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Offline parsing for PDFs (text/tables/images) without APIs. Unstructured hi_res strategy extracts elements (YOLOX/Tesseract OCR), queryable/scalable for local use.

## Related Requirements

- Offline (local, no servers).
- Multimodal (text+images+tables).
- Integrate with IngestionPipeline for chunking/metadata.

## Alternatives

- PyMuPDF: Text/images only, no tables (leaky temps).
- pdfplumber: Tables only, no images.
- Tika: Local server (heavier setup).

## Decision

Use UnstructuredReader (local, hi_res strategy) in LlamaIndex. Strategy toggle via AppSettings.parse_strategy or "hi_res".

## Related Decisions

- ADR-016 (Multimodal—feeds Jina v4).
- ADR-005 (Chunking post-parsing).

## Design

- **Loading**: In utils.py load_documents_llama: from llama_index.readers.unstructured import UnstructuredReader; reader = UnstructuredReader(); elements = reader.load_data(file_path, strategy=AppSettings.parse_strategy or "hi_res"); docs = [Document.from_element(e) for e in elements].
- **Integration**: Feed docs to IngestionPipeline for chunking/metadata → indexes. For multimodal: elements with type=="image" to MultiModalVectorStoreIndex.
- **Docker Setup**: In Dockerfile: apt-get update && apt-get install -y tesseract-ocr poppler-utils libmagic-dev—ensure deps installed.
- **Implementation Notes**: Handle errors (e.g., try/except logger.error(e); fallback to text-only if OCR fails). Cleanup: No temps needed with Unstructured.
- **Testing**: In tests/test_utils.py: def test_unstructured_parse(): elements = reader.load_data(pdf_path); assert len(elements) > 0; assert any(e.type == "table" for e in elements); assert any(e.type == "image" for e in elements); def test_parse_strategy_toggle(): AppSettings.parse_strategy = "fast"; docs = load_documents_llama([pdf]); assert no images in docs.

## Consequences

- Full offline parsing (text/tables/images/OCR, scalable for local).
- Modular (strategy toggle).
- Better than custom (unified, no leaks).

- Deps: unstructured[all-docs]==0.15.13 (add to pyproject.toml; Docker ~200MB increase for deps).
- Future: Add OCR toggle via AppSettings.ocr_enabled (strategy="ocr_only" if True).

**Changelog:**  

- 2.0 (July 25, 2025): Switched to Unstructured for offline/full parsing (added Docker deps/strategy toggle/integration with pipeline/multimodal; Enhanced testing for dev.
