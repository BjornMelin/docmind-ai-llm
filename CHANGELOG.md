# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- SPEC‑008: Programmatic Streamlit UI with `st.Page` + `st.navigation`.
  - New pages: `src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/pages/03_analytics.py`.
  - New adapter: `src/ui/ingest_adapter.py` for form-based ingestion.
- ADR‑032: Local analytics manager (`src/core/analytics.py`) with DuckDB and best‑effort background writes; coordinator logs query metrics.
- SPEC‑013 (ADR‑040): Model pre‑download CLI `tools/models/pull.py` using `huggingface_hub`.
- SPEC‑010 (ADR‑039): Offline evaluation CLIs:
  - `tools/eval/run_beir.py` (NDCG@10, Recall@10, MRR@10)
  - `tools/eval/run_ragas.py` (faithfulness, answer_relevancy, context_recall, context_precision)
  - `data/eval/README.md` with usage instructions.
  
- Post‑ingest Qdrant indexing (hybrid) wired into ingestion adapter; Documents page builds a router engine for Chat.
- SPEC‑006: GraphRAG exports (Parquet + JSONL) triggered by Documents page checkbox.
- GraphRAG (Phase 1): Library‑first refactor of `src/retrieval/graph_config.py` to use only documented LlamaIndex APIs (`as_retriever`, `as_query_engine`, `get`, `get_rel_map`); removed legacy/dead code and index mutation; added portable JSONL/Parquet exports via `get_rel_map`.
- GraphRAG (Phase 2): Added `create_graph_rag_components()` factory to return (`graph_store`, `query_engine`, `retriever`) from a `PropertyGraphIndex`.
- UI wiring: Documents page stores `vector_index`, `hybrid_retriever`, and optional `graphrag_index` in `st.session_state`; Chat page forwards these in `settings_override` (`vector`, `retriever`, `kg`) for coordinator/tools.
 - GraphRAG (Phase 2): SnapshotManager for atomic snapshots with `manifest.json` (corpus/config hashes, versions, lock) under `src/persistence/snapshot.py`.
 - Router factory for GraphRAG: `src/retrieval/router_factory.py` builds a `RouterQueryEngine` with vector+graph tools and safe fallback to vector-only.
 - Export helpers: JSONL baseline (subject, relation, object, depth, path_id, source_ids) and Parquet (optional, guarded when `pyarrow` missing).
 - UI: Documents page toggle "Build GraphRAG (beta)", snapshot creation notice, and export buttons; Chat staleness badge when manifest hashes mismatch.
 
 - Server-side hybrid retriever module: `src/retrieval/hybrid.py` exposing `ServerHybridRetriever` and `_HybridParams` (Qdrant Query API Prefetch + FusionQuery; RRF default, DBSF optional). Deterministic de‑duplication and dense‑only fallback when sparse unavailable.
 - Router composition unification via `src/retrieval/router_factory.py`:
   - Tools: `semantic_search` (vector.as_query_engine), `hybrid_search` (RetrieverQueryEngine wrapping `ServerHybridRetriever`), `knowledge_graph` (PropertyGraphIndex retriever/query engine with `path_depth=1`).
   - Selector preference: `PydanticSingleSelector` when available, else `LLMSingleSelector`; fail‑open to vector‑only when graph is absent/unhealthy.
 - GraphRAG exports now preserve relation labels when provided by `get_rel_map`; fallback label is `related`. JSONL baseline retained; Parquet optional (guarded when PyArrow missing).
 - Retriever‑first seed policy helper `get_export_seed_ids()` in `src/retrieval/graph_config.py` used by `src/ui/ingest_adapter.py` for deterministic export seeding (graph → vector → stable fallback; dedup + stable tie‑break).
 - Snapshot manifest enrichment in `src/persistence/snapshot.py`:
   - Added `schema_version`, `persist_format_version`, `complete`, and enriched `versions` (`app`, `llama_index`, `qdrant_client`, `embed_model`).
   - Normalized `compute_corpus_hash(paths, base_dir=uploads/)` to use POSIX relpaths for cross‑platform stability.
 - Chat autoload/staleness UX: Chat/Documents pages compute corpus hash with relpaths and surface staleness badge based on manifest.
  
### Tests
- Added unit tests for analytics manager insert/prune.
- Added CLI smoke tests for model pull and RAGAS/BEIR harnesses.
- Added page import smoke tests for new Streamlit pages.
- Added unit test for Chat router override mapping.
- Added unit tests: GraphRAG factory (`tests/unit/retrieval/test_graph_rag_factory.py`), graph helpers (`tests/unit/retrieval/test_graph_config_utils.py`), and portable exports (`tests/integration/test_graphrag_exports.py`).
 - Added unit tests for SnapshotManager (`tests/unit/persistence/test_snapshot_manager.py`) and router factory (`tests/unit/retrieval/test_router_factory.py`).
 - Added integration tests for router composition (`tests/integration/test_ingest_router_flow.py`) and exports (`tests/integration/test_graphrag_exports.py`).
- Added E2E smoke test for Chat via router override (`tests/e2e/test_chat_graphrag_smoke.py`).
- Updated Chat router override test to allow additional forwarded components when present.
 
 - New hybrid/router/graph tests:
   - `tests/unit/retrieval/test_hybrid_retriever_basic.py` (dedup determinism; sparse‑unavailable dense fallback)
   - `tests/unit/retrieval/test_router_factory_hybrid.py` (vector + hybrid + knowledge_graph tools)
   - `tests/unit/retrieval/test_seed_policy.py` (retriever‑first seed policy and fallbacks)
   - `tests/unit/retrieval/test_graph_helpers.py` (label preservation + `related` fallback)
   - `tests/unit/persistence/test_corpus_hash_relpaths.py` (relpath hashing determinism)
   - Updated/removed legacy integration tests; examples now use `router_factory`

### Changed

- UI refactor: `src/app.py` now only defines pages and runs navigation; all monolithic UI logic moved to `src/pages/*`.
- Coordinator: best‑effort analytics logging added after processing each query.
 - Router toolset unified: `router_factory.build_router_engine(...)` composes `semantic_search`, `hybrid_search`, and `knowledge_graph` tools; selector policy prefers `PydanticSingleSelector` then falls back to `LLMSingleSelector`.
 - GraphRAG helpers (`graph_config.py`) now emit label‑preserving exports and provide `get_export_seed_ids()` for deterministic seeding.
- Snapshot manifest enriched and corpus hashing normalized to relpaths; Chat autoload/staleness detection wired to these fields.
 - Docs: aligned developer and API docs to new router/hybrid/GraphRAG/snapshots; added `docs/api/api.md` and `docs/developers/ci-cd-pipeline.md`.

### Removed

- Deleted legacy model predownload script: `scripts/model_prep/predownload_models.py`.
- Removed monolithic UI blocks from `src/app.py` (chat/ingestion/analytics).
 - Removed legacy/custom router code and tests; all retrieval routes via `router_factory`. No backwards compatibility retained.

## [1.3.0] - 2025-09-08

### Breaking

- Removed legacy prompts module (`src/prompts.py`) and all usages/tests. Migrate to the new file‑based prompt template system (SPEC‑020) via `src.prompting` APIs (`list_templates`, `render_prompt`, `format_messages`, `list_presets`).
- Removed deprecated retrieval modules no longer used after the factory refactor:
  - `src/retrieval/bge_m3_index.py`
  - `src/retrieval/optimization.py`

### Added

- SPEC‑020: Prompt Template System (RichPromptTemplate, file‑based)
  - New `src/prompting/` package: models, loader, registry, renderer, validators
  - Templates under `templates/prompts/*.prompt.md` (YAML front matter + Jinja body)
  - Presets under `templates/presets/{tones,roles,lengths}.yaml`
  - Public API: `list_templates`, `get_template`, `render_prompt`, `format_messages`, `list_presets`
  - Streamlit UI integration (replaces PREDEFINED_PROMPTS)
  - Developer Guide: `docs/developers/guides/adding-prompt-template.md`
- Prompt telemetry logging in app: logs `prompt.template_id`, `prompt.version`, `prompt.name` to local JSONL after render (async + sync paths); sampling/rotation controlled via existing telemetry envs.

### Changed

- Standardized SPEC‑020 document to match repository SPEC format (YAML header, related requirements/ADRs, file operations, checklist).
- ADR‑018 (DSPy Prompt Optimization): marked Implemented; added note on compatibility with SPEC‑020 (rewriter can run before template rendering or on free‑form input).
- RTM updated: FR‑020 marked Completed with code/test references; README “Choosing Prompts” updated to document templates/presets and API usage.

### Removed

- Legacy prompt constants and associated tests; replaced by file‑based templates with RichPromptTemplate.
- Deprecated retrieval modules left from pre‑factory architecture (`bge_m3_index.py`, `optimization.py`).

### Tests

- New unit/integration/E2E smokes for prompting:
  - Unit: loader, registry/renderer, validators
  - Integration: registry list + render smoke
  - E2E: prompt catalog presence
- Updated existing tests to use `src.prompting` instead of legacy prompts.

## [1.2.0] - 2025-09-08

### Breaking

- Removed all legacy client-side fusion knobs (rrf_alpha, rrf_k_constant, fusion weights) and UI reranker toggles. Server-side Qdrant Query API fusion is authoritative; RRF default, DBSF env-gated via `DOCMIND_RETRIEVAL__FUSION_MODE`. Also removed the deprecated `retrieval.reranker_mode` setting — reranker implementation is now auto‑detected (FlagEmbedding preferred when available, else LI).

### Added

- Retrieval telemetry (JSONL) with canonical keys: retrieval.*and dedup.*.
- Rerank telemetry with rerank.*: stage, latency_ms, timeout, delta_changed_count.
- Enforced BM42 sparse with IDF modifier in Qdrant schema; migration helper for existing collections.
- SIGLIP model env `DOCMIND_EMBEDDING__SIGLIP_MODEL_ID` and predownload script `scripts/predownload_models.py`.
- Imaging: DPI≈200 via PyMuPDF, EXIF-free WebP/JPEG, optional AES‑GCM at-rest encryption.
- Tests: unit tests for Query API fusion (RRF/DBSF), rerank timeout fail-open, SigLIP rescore mock, encrypted imaging round-trip, retrieval env mapping.

### Changed

- Reranking is always-on (BGE v2‑m3 text + SigLIP visual) with policy-gated ColPali; UI no longer exposes reranker knobs. Implementation selection is automatic (no env/config toggles).

### Fixed

- Read-only settings panel simplified; no longer references removed `reranker_mode`.
- README updated with offline predownload steps and new envs.

## [1.1.0] - 2025-09-07

### Added

- Hybrid retrieval (SPEC‑004): Qdrant server‑side fusion via Query API
  - Named vectors `text-dense` (BGE‑M3 1024D) and `text-sparse` (sparse index)
  - Prefetch dense≈200 / sparse≈400; default fusion RRF; DBSF experimental via env `DOCMIND_RETRIEVAL__FUSION_MODE`
  - De‑dup by `page_id` before fused cut; fused_top_k≈60; telemetry

- Reranking (SPEC‑005; ADR‑037):
  - Default visual re‑score with SigLIP text–image cosine (timeout 150ms)
  - Text BGE v2‑m3 CrossEncoder (timeout 250ms)
  - Optional ColPali policy (VRAM ≥8–12GB, small‑K ≤16, visual‑heavy); cascade SigLIP prune → ColPali final; fail‑open
  - Rank‑level RRF merge across modalities; always‑on (no UI toggles; ops env only)

- PDF page images: WebP default (q≈70, method=6), JPEG fallback; DPI≈200; long‑edge≈2000px; simple pHash for dedup hints.

### Changed

- ADR/specs alignment: ADR‑036 marked Superseded by ADR‑024 v2.7 and SPEC‑005; ADR‑002 reflects SigLIP default; SPEC‑004/005 cross‑refs updated.

## [1.0.0] - 2025-09-02

### Added

- SPEC‑003: Unified embeddings
  - BGE‑M3 text embeddings with dense+sparse outputs via LI factories
  - Tiered image embeddings (OpenCLIP ViT‑L/H, SigLIP base) with LI ClipEmbedding
  - Runtime dimension derivation; deterministic offline tests
  - Legacy wrappers removed; LI‑first wiring throughout

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

- LLM runtime (SPEC‑001: Multi‑provider runtime & UI):
  - Unified LLM factory `src/config/llm_factory.py` using LlamaIndex adapters:
    - vLLM/LM Studio/llama.cpp server via `OpenAILike` (OpenAI‑compatible)
    - Ollama via `llama_index.llms.ollama.Ollama`
    - Local llama.cpp via `LlamaCPP(model_path=…, model_kwargs={"n_gpu_layers": -1|0})`
    - Respects `model`, `context_window`, and `llm_request_timeout_seconds`
  - New Settings page `src/pages/04_settings.py` with provider dropdown, URL fields, model/path input, context window, timeout, GPU toggle
    - “Apply runtime” rebinds `Settings.llm` (force)
    - “Save” persists to `.env` via a minimal updater
  - Provider badge `src/ui/components/provider_badge.py` shows provider/model/base_url on Chat and Settings
  - Security validation: `allow_remote_endpoints` default False with `endpoint_allowlist`; LM Studio URL must end with `/v1`
  - Observability: logs provider/model/base_url on apply, simple counters (`provider_used`, `streaming_enabled`)
  - Chat wiring: `src/app.py` uses `Settings.llm` for agent system after `initialize_integrations()`
  - Docs/specs: SPEC‑001 checklist checked; RTM updated for FR‑010/FR‑012
  - Tests:
    - Factory type assertions: `tests/unit/test_llm_factory.py`
    - Extended factory behavior (overrides, /v1, llama.cpp local): `tests/unit/test_llm_factory_extended.py`
    - Runtime rebind: `tests/unit/test_integrations_runtime.py`
    - Settings Apply/Save roundtrip: `tests/integration/test_settings_page.py`
    - Provider toggle + apply (ollama/vllm/lmstudio/llamacpp server+local): `tests/integration/test_settings_page.py`

- Multimodal reranking (ADR‑037): ColPali (visual) + BGE v2‑m3 (text), auto‑gated per node modality with fusion and top‑K truncation; builders for text/visual rerankers; integration test added.
- UI controls (ADR‑036): Sidebar Reranker Mode (`auto|text|multimodal`), Normalize scores, Top N.
- Configuration (ADR‑024): global `llm_context_window_max=131072`.
- Ingestion (ADR‑009): PDF page image emission via `pdf_pages_to_image_documents()` tagging `metadata.modality="pdf_page_image"`.

- Integration test for reranker toggle parity (Quick vs Agentic): `tests/integration/test_reranker_parity.py`.
- New utils unit tests to raise coverage: document helpers, core async contexts, monitoring timers.

### Changed

- Agents tools refactor: split `src/agents/tools.py` into cohesive modules under `src/agents/tools/` (`router_tool.py`, `planning.py`, `retrieval.py`, `synthesis.py`, `validation.py`, `telemetry.py`) with `src.agents.tools` as an aggregator. Public API preserved via re-exports; targeted `cyclic-import` disables added where necessary.
- Linting: re-enabled complexity rules (`too-many-statements`, `too-many-branches`, `too-many-nested-blocks`) and fixed violations by extracting helpers. Imports organized per Ruff; helper signatures annotated to satisfy `ANN001`.

- Tools now import patch points (ToolFactory, logger, ChatMemoryBuffer, time) via aggregator for resilient tests.
- Retrieval tool hardened: explicit strategy path by default; conditional aggregator fast‑path for resilience scenarios; DSPy optional with short‑query fallback.
- Planning tweaks for list/categorize decomposition; timing via aggregator time.
- Validation thresholds tuned for source overlap (inclusive) via constants.
- Test runner `scripts/run_tests.py` updated to ASCII‑only output and corrected import validation list.

- UI/runtime (SPEC‑001): removed legacy in‑app backend selection and ad‑hoc LLM construction; centralized provider selection and LLM creation via Settings page + unified factory with strict endpoint validation.

### Fixed

- Stabilized supervisor shims in integration tests (compile/stream signatures) and ensured InjectedState overrides visibility.
- Addressed flaky/residual lint warnings across test suites; ensured ruff clean and pylint tests ≥ 9.8.

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
- Retrieval pipeline (ADR‑003): inject reranker by mode (auto/multimodal → `MultimodalReranker`, text → text‑only builder). Updated `src/retrieval/__init__.py` docstring to ADR‑037.

### Fixed

- Stabilized pipeline initialization in performance tests by providing fake `IngestionPipeline` with `cache`/`docstore` attributes to satisfy pydantic validation.
- Prevented import-time failures by lazily importing Ollama in `setup_llamaindex()` and avoiding heavy integration side effects at module import.
- E2E and integration tests: ensured async mocks are awaited; stabilized partition side-effects to accept kwargs; improved deterministic patches for Unstructured element detection.
- Unit tests: added reranker factory properties test; tightened exception expectations and removed unused mocks where applicable.
- Tests: replaced brittle full‑app UI import with deterministic unit/integration coverage to avoid side effects.

### Removed

- Legacy chunkers and compatibility shims.
- Implicit side-effect initialization from `src/config/__init__.py` (explicit `initialize_integrations()` only where needed).
- Test-compatibility logic from production code (src/). Tests now own the compatibility shims and patches.
- ColBERT reranker integration and legacy BGECrossEncoder path; all related tests deleted. No backward compatibility retained.

### Testing & Tooling

- Ruff formatting and lint cleanup on changed files.
- Pylint score for `document_processor.py` ≥ 9.5 (≈ 9.85).
- Deterministic, offline test execution via mocks/stubs; no GPU/network required.
- Ruff: formatted and lint-clean on changed files.
- Pylint: changed modules meet quality gate; unrelated large modules earmarked for a future maintenance pass.
- Scoped pylint to `src/` for refactor; dependencies updated: replace `llama-index-postprocessor-colbert-rerank` with `llama-index-postprocessor-colpali-rerank`.

### Migration Notes

- No backwards compatibility retained for legacy chunkers. Downstream callers should rely on Unstructured-first processing and the `DocumentProcessor` API.
- Tests should import heavy integrations behind stubs and use library-first mocks (LlamaIndex `MockLLM`, `MockEmbedding`, in-memory `VectorStoreIndex`).

---

[1.0.0]: https://example.com/releases/tag/v1.0.0

### Added — Router, GraphRAG, Multimodal, Settings

- LangGraph Supervisor integration with `router_tool` wired into coordinator; InjectedState carries router engine and runtime toggles.
- Sub‑question decomposition strategy via `SubQuestionQueryEngine` exposed as `sub_question_search` in router toolset.
- GraphRAG default ON per ADR‑019; knowledge_graph tool registered when KG index present; one‑line rollback flag.
- Multimodal ingestion and `MultiModalVectorStoreIndex` built only when image/page‑image nodes exist; `multimodal_search` tool conditional.
- Settings convenience aliases and helpers:
  - Aliases: `context_window_size`, `chunk_size`, `chunk_overlap`, `enable_multi_agent` (validated and mapped to nested fields).
  - Helpers: `get_model_config()`, `get_embedding_config()`, `get_vllm_env_vars()`.
- Optional BM25 keyword tool behind `retrieval.enable_keyword_tool` (disabled by default).
- Telemetry logging tests for router creation, strategy selection, and fallback.
- New router tool unit tests (success, error, no metadata) and supervisor integration path.
- Sub‑question fallback unit test (tree_summarize) when SQE creation fails.

### Changed

- Agents tools refactor: split `src/agents/tools.py` into cohesive modules under `src/agents/tools/` (`router_tool.py`, `planning.py`, `retrieval.py`, `synthesis.py`, `validation.py`, `telemetry.py`) with `src.agents.tools` as an aggregator. Public API preserved via re-exports; targeted `cyclic-import` disables added where necessary.
- Linting: re-enabled complexity rules (`too-many-statements`, `too-many-branches`, `too-many-nested-blocks`) and fixed violations by extracting helpers. Imports organized per Ruff; helper signatures annotated to satisfy `ANN001`.

- Replaced all legacy `multi_query_search` references with `sub_question_search`; updated selector descriptions and constants.
- System and integration tests migrated to nested settings and final defaults (embedding, database.sqlite path, monitoring limits, GraphRAG default ON); system workflows updated for async behavior.
- Documentation: added developer guide `docs/developers/configuration-usage-guide.md`; linked from developers README.

### (Test-related items grouped under Added/Changed)

### Removed

- Legacy `multi_query_search` references and assumptions in code and tests.

- LlamaIndex integration packages added as first‑class dependencies:
  - `llama-index-embeddings-clip` for CLIP multimodal embeddings
  - `llama-index-postprocessor-colbert-rerank` for ColBERT reranking
  - `llama-index-llms-ollama` for local Ollama backend
- New cache unit test: DuckDBKV lifecycle (create/stats/clear) under `tests/unit/cache/test_ingestion_cache.py`.
- Shared backlog tracker: `agent-logs/2025-09-02/next-tasks/001-next-tasks-and-research-backlog.md`.
- Caching architecture: replaced custom `SimpleCache` with LlamaIndex `IngestionCache` backed by `DuckDBKVStore` (single local file at `cache/docmind.duckdb`). No back‑compat.
- DocumentProcessor: wired DuckDBKV‐backed `IngestionCache`; added safe JSON coercion for Unstructured element metadata to support KV persistence.
- Utils: cache clear/stats now operate on the DuckDB cache file (delete/inspect path).
- Tests: modernized app and perf test patches to target `src.app` functions; ensured deterministic performance tests and removed brittle timing checks.
- DI: container creation logs a standard message for test capture (`Created ApplicationContainer`).
- Resolved import errors by adding LlamaIndex split packages and using correct namespaced imports (CLIP, ColBERT, Ollama).
- Prevented JSON serialization errors when persisting pipeline outputs by coercing metadata to JSON‑safe types.
- Stabilized performance tests by patching the correct targets and relaxing environment‑sensitive assertions.
- `src/cache/simple_cache.py` and all references.
- `tests/unit/cache/test_simple_cache.py` and docs/spec references to SimpleCache; specs now reflect ADR‑030.
- Agents tools refactor: split `src/agents/tools.py` into cohesive modules under `src/agents/tools/` (`router_tool.py`, `planning.py`, `retrieval.py`, `synthesis.py`, `validation.py`, `telemetry.py`) with `src.agents.tools` as an aggregator. Public API preserved via re-exports; targeted `cyclic-import` disables added where necessary.
- Linting: re-enabled complexity rules (`too-many-statements`, `too-many-branches`, `too-many-nested-blocks`) and fixed violations by extracting helpers. Imports organized per Ruff; helper signatures annotated to satisfy `ANN001`.
- Ingestion (SPEC‑002): Completed Unstructured‑first ingestion with LlamaIndex IngestionPipeline
  - Strategy mapping (hi_res / fast / ocr_only) with OCR fallback
  - Deterministic IDs for text and page‑image nodes via `sha256_id`
  - PDF page‑image emission with stable filenames (`__page-<n>.png`) and bbox
  - DuckDBKV‑backed `IngestionCache` at `./cache/docmind.duckdb`; surfaced cache stats
  - Tests: unit and integration for page images, deterministic IDs, chunking heuristics

- Config: `llm_backend` is a strict literal ("ollama"|"llamacpp"|"vllm"|"lmstudio"); tests updated accordingly
- Models: `ErrorResponse` enriched with optional `traceback` and `suggestion`; `PdfPageImageNode` no longer carries error fields
- Tests: PDF rendering patched/stubbed in unit/integration paths to avoid heavy I/O while preserving behavior under test
- Docs: SPEC‑002 updated to reflect actual implementation (Unstructured chunking, IngestionPipeline transform, caching, page‑image emission, validation commands)
