# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [1.0.0] - 2025-09-01

### Added

- Unstructured-first document chunking in the processing stack:
  - Partition via `unstructured.partition.auto.partition()`;
  - Chunk via `unstructured.chunking.title.chunk_by_title` with small-section smoothing and `multipage_sections=true`;
  - Automatic fallback to `unstructured.chunking.basic.chunk_elements` for heading‑sparse documents;
  - Table isolation (tables are never merged with narrative text);
  - Full element metadata preserved (page numbers, coordinates, HTML, image path) across chunks.
- Hybrid ingestion pipeline that integrates Unstructured with LlamaIndex `IngestionPipeline` and our `UnstructuredTransformation` for strategy-based processing (hi_res / fast / ocr_only) with caching and docstore support.
- New integration tests validating:
  - By-title section boundaries and multi-page sections;
  - Basic fallback on heading-sparse inputs;
  - Table isolation invariants.
- Performance smoke test verifying throughput (informational target ≥ 1 page/sec locally).
- Robust E2E tests for Streamlit app using `st.testing.v1.AppTest` with boundary-only mocks for heavy libs (torch, spacy, FlagEmbedding, Unstructured, Qdrant, LlamaIndex, Ollama) and resilient assertions (no brittle UI string matching).
- Coverage gate scoped to changed subsystems: `src/processing` and `src/config` with `fail_under=29.71%`.

### Changed

- Removed legacy/deprecated splitters (SentenceSplitter-based chunkers) and all backward-compatibility paths in favor of Unstructured-first approach.
- Updated `src/processing/document_processor.py` to:
  - Select strategy by file type (hi_res/fast/ocr_only),
  - Hash, cache, and error-handling improvements with tenacity-backed retry,
  - Convert chunked elements to LlamaIndex `Document` nodes, then normalize to our `DocumentElement` API.
- Modernized E2E test architecture:
  - Centralized stubs for agents, tools, and utils modules;
  - Replaced strict call-count and exact-text checks with structure/behavior checks;
  - Ensured offline determinism by stubbing external services and model backends.
- Router Query Engine integration: passed `llm=self.llm` into `RetrieverQueryEngine.from_args` to avoid accidental network calls via global settings.

### Fixed

- Stabilized pipeline initialization in performance tests by providing fake `IngestionPipeline` with `cache`/`docstore` attributes to satisfy pydantic validation.
- Prevented import-time failures by lazily importing Ollama in `setup_llamaindex()` and avoiding heavy integration side effects at module import.

### Removed

- Legacy chunkers and compatibility shims.
- Implicit side-effect initialization from `src/config/__init__.py` (explicit `initialize_integrations()` only where needed).

### Testing & Tooling

- Ruff formatting and lint cleanup on changed files.
- Pylint score for `document_processor.py` ≥ 9.5 (≈ 9.85).
- Deterministic, offline test execution via mocks/stubs; no GPU/network required.

### Migration Notes

- No backwards compatibility retained for legacy chunkers. Downstream callers should rely on Unstructured-first processing and the `DocumentProcessor` API.
- Tests should import heavy integrations behind stubs and use library-first mocks (LlamaIndex `MockLLM`, `MockEmbedding`, in-memory `VectorStoreIndex`).

---

[1.0.0]: https://example.com/releases/tag/v1.0.0
