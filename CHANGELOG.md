# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [1.0.0] - 2025-09-02

### Added

- Unstructured-first document chunking in the processing stack:
  - Partition via `unstructured.partition.auto.partition()`;
  - Chunk via `unstructured.chunking.title.chunk_by_title` with small-section smoothing and `multipage_sections=true`;
  - Automatic fallback to `unstructured.chunking.basic.chunk_elements` for heading‑sparse documents;
  - Table isolation (tables are never merged with narrative text);
  - Full element metadata preserved (page numbers, coordinates, HTML, image path) across chunks.
- Hybrid ingestion pipeline that integrates Unstructured with LlamaIndex `IngestionPipeline` and our `UnstructuredTransformation` for strategy-based processing (hi_res / fast / ocr_only) with caching and docstore support.
- Processing utilities: `src/processing/utils.py` with `is_unstructured_like()` to centralize element safety checks for Unstructured.
- Reranker configuration: surfaced `retrieval.reranker_normalize_scores` in settings and .env; factory respects settings with explicit override precedence.
- Performance: lightweight informational smoke test for processing throughput (kept out of CI critical path).
- Integration tests: additional chunking edge-cases (clustered titles, frequent small headings) and deterministic patches for Unstructured element detection.
- Sprint documentation and navigation:
  - `agent-logs/2025-09-01/sprint-unstructured-first-chunking/README.md` (index) and `011_current_status_and_remaining_work.md` (status)
  - `agent-logs/2025-09-02/processing/001_semantic_cache_and_reranker_ui_research_plan.md` (next sprint research plan)
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
- DocumentProcessor: removed test-only code paths (e.g., partition kwargs/type fallbacks, patched-chunker detection) to keep src production-pure; tests now patch compatibility in tests/.
- Reranker: `create_bge_cross_encoder_reranker()` now resolves settings first, then explicit args; enforces device-aware batch sizing; returns fresh `NodeWithScore` instances instead of mutating inputs.
- Cache stats: `SimpleCache` now derives `total_documents` from underlying KVStore instead of internal counters.
- .env.example: added missing mappings for processing (new_after_n_chars, combine_text_under_n_chars, multipage_sections, debug_chunk_flow), retrieval (reranker_normalize_scores), cache (ttl_seconds, max_size_mb), DB (qdrant_timeout), and vLLM base URLs.
- Router Query Engine integration: passed `llm=self.llm` into `RetrieverQueryEngine.from_args` to avoid accidental network calls via global settings.

### Fixed

- Stabilized pipeline initialization in performance tests by providing fake `IngestionPipeline` with `cache`/`docstore` attributes to satisfy pydantic validation.
- Prevented import-time failures by lazily importing Ollama in `setup_llamaindex()` and avoiding heavy integration side effects at module import.
- E2E and integration tests: ensured async mocks are awaited; stabilized partition side-effects to accept kwargs; improved deterministic patches for Unstructured element detection.
- Unit tests: added reranker factory properties test; tightened exception expectations and removed unused mocks where applicable.

### Removed

- Legacy chunkers and compatibility shims.
- Implicit side-effect initialization from `src/config/__init__.py` (explicit `initialize_integrations()` only where needed).
- Test-compatibility logic from production code (src/). Tests now own the compatibility shims and patches.

### Testing & Tooling

- Ruff formatting and lint cleanup on changed files.
- Pylint score for `document_processor.py` ≥ 9.5 (≈ 9.85).
- Deterministic, offline test execution via mocks/stubs; no GPU/network required.
- Ruff: formatted and lint-clean on changed files.
- Pylint: changed modules meet quality gate; unrelated large modules earmarked for a future maintenance pass.

### Migration Notes

- No backwards compatibility retained for legacy chunkers. Downstream callers should rely on Unstructured-first processing and the `DocumentProcessor` API.
- Tests should import heavy integrations behind stubs and use library-first mocks (LlamaIndex `MockLLM`, `MockEmbedding`, in-memory `VectorStoreIndex`).

---

[1.0.0]: https://example.com/releases/tag/v1.0.0
