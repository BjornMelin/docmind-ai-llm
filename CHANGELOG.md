# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Optional llama-index adapter with lazy availability checks and install guidance (`src/retrieval/llama_index_adapter.py`).
- Coverage for OCR policy, canonicalization, image I/O, router fallbacks, and UI ingestion flows (new unit tests under `tests/unit`).
- Canonical ingestion models and hashing helpers powering the library-first ingestion pipeline (`src/models/processing.py`, `src/persistence/hashing.py`).
- LlamaIndex-based ingestion pipeline, DuckDB-backed cache/docstore wiring, AES-GCM page-image exports, and OpenTelemetry spans (`src/processing/ingestion_pipeline.py`).
- Snapshot lock and writer modules with heartbeat/takeover metadata, atomic promotion, tri-file manifest, and timestamped graph export metadata (`src/persistence/lockfile.py`, `src/persistence/snapshot_writer.py`).
- Snapshot lock heartbeat refresher prevents TTL expiry during long-running persists (`src/persistence/lockfile.py`).
- Manifest metadata now records the active embedding model and spec-compliant `versions["llama_index"]` entry (`src/pages/02_documents.py`).
- PDF page-image exports accept explicit encryption flags to avoid global state mutation (`src/processing/pdf_pages.py`, `src/processing/ingestion_pipeline.py`).
- GraphRAG router/query helpers using native LlamaIndex retrievers and telemetry instrumentation for export counters and spans (`src/retrieval/graph_config.py`, `src/retrieval/router_factory.py`, `src/agents/tools/router_tool.py`).
- Streamlit UI integration that surfaces manifest metadata, staleness badges, GraphRAG export tooling, and ingestion orchestration (`src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/ui/ingest_adapter.py`, `src/agents/coordinator.py`).
- Observability configuration and helpers for OTLP/console exporters and optional LlamaIndex instrumentation (`ObservabilityConfig`, `configure_observability`, updated `scripts/demo_metrics_console.py`).
- Quick-start demos for the overhaul: `scripts/run_ingestion_demo.py` (pipeline smoke test) and refreshed console metrics demo.
- OpenAIConfig (openai.*) with idempotent /v1 base_url normalization and api_key.
- SecurityConfig (security.*) centralizing allow_remote_endpoints, endpoint_allowlist, trust_remote_code.
- HybridConfig (hybrid.*) declarative policy (enabled/server_side/method/rrf_k/dbsf_alpha).
- Clear caches feature: Settings page button and `src/ui/cache.py` helper (bumps `settings.cache_version` and clears Streamlit caches).
- Pure prompting helper: `src/prompting/helpers.py` with `build_prompt_context` (pure; no UI/telemetry) and unit tests.
- SPEC-008: Programmatic Streamlit UI with `st.Page` + `st.navigation`.
  - New pages: `src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/pages/03_analytics.py`.
  - New adapter: `src/ui/ingest_adapter.py` for form-based ingestion.
- ADR-032: Local analytics manager (`src/core/analytics.py`) with DuckDB and best-effort background writes; coordinator logs query metrics.
- SPEC-013 (ADR-040): Model pre-download CLI `tools/models/pull.py` using `huggingface_hub`.
- SPEC-010 (ADR-039): Offline evaluation CLIs:
  - `tools/eval/run_beir.py` (NDCG@10, Recall@10, MRR@10)
  - `tools/eval/run_ragas.py` (faithfulness, answer_relevancy, context_recall, context_precision)
  - `data/eval/README.md` with usage instructions.
- Evaluation harness hardening (schema + determinism):
  - Dynamic `@{k}` metric headers for BEIR (`ndcg@{k}`, `recall@{k}`, `mrr@{k}`) and explicit `k` field.
  - Leaderboard rows now include `schema_version` and `sample_count` for reproducibility.
  - JSON Schemas: `schemas/leaderboard_beir.schema.json`, `schemas/leaderboard_ragas.schema.json`, `schemas/doc_mapping.schema.json`.
  - Validator script: `scripts/validate_schemas.py` (enforces header↔k consistency; validates required fields/types).
  - Determinism utilities under `src/eval/common/`: seeds + thread caps.
  - Doc id mapping persisted to `doc_mapping.json` per run.

### Changed
- Streamlit ingestion adapter now re-checks embeddings after ingestion and logs when vector index is skipped or built.
- Ingestion pipeline elevates embedding auto-setup failures to warnings and logs plaintext fallbacks for Unstructured reader errors.
- CI now runs base and llama profiles with optional `llama` extra to enforce optional dependency coverage.
- Snapshot manifest/schema now records `complete`, `schema_version`, `persist_format_version`, graph export metadata, and enforces `_tmp-` workspace plus `CURRENT` pointer discipline.
- Guard snapshot workspace initialization to release file locks if creation fails (`src/persistence/snapshot.py`).
- Router, UI, and telemetry layers consistently emit OpenTelemetry spans/metrics for ingestion, snapshot promotion, GraphRAG selection, and export flows.
- Packaging and CI rely on `uv` with an `observability` extra (OTLP exporters, portalocker, LlamaIndex OTEL) and run `ruff`, `pylint`, and `uv run scripts/run_tests.py --coverage` under locked environments.
- Shared fixtures/tests cover ingestion pipeline builders, snapshot locks, Streamlit AppTest interactions, and console exporter stubs.
- Enforced backend-aware OpenAI-like `/v1` normalization in LLM factory for LM Studio, vLLM (OpenAI-compatible), and llama.cpp server.
- Moved all import-time I/O from settings into explicit `startup_init(settings)` in integrations.
- Unified server-side hybrid gating to `retrieval.enable_server_hybrid` + `fusion_mode`; removed legacy flags.
- Settings UI now shows read-only policy state (server-side hybrid, fusion mode, remote endpoint allowance, allowlist size) and resolved normalized backend base URL.
- `.env.example` rewritten to use `DOCMIND_OPENAI__*`, `DOCMIND_SECURITY__*`, and `DOCMIND_VLLM__*`; removed raw `VLLM_*` keys.
- BEIR and RAGAS CLIs now call determinism setup first; BEIR CLI respects `--k` for metric computation and emits dynamic headers matching `k`.
- CI workflow: added schema validation step after tests to catch leaderboard schema drift.
- Post-ingest Qdrant indexing (hybrid) wired into ingestion adapter; Documents page builds a router engine for Chat.
- SPEC-006: GraphRAG exports (Parquet + JSONL) triggered by Documents page checkbox.
- GraphRAG (Phase 1): Library-first refactor of `src/retrieval/graph_config.py` to use only documented LlamaIndex APIs (`as_retriever`, `as_query_engine`, `get`, `get_rel_map`); removed legacy/dead code and index mutation; added portable JSONL/Parquet exports via `get_rel_map`.
- GraphRAG (Phase 2): Added `create_graph_rag_components()` factory to return (`graph_store`, `query_engine`, `retriever`) from a `PropertyGraphIndex`.
- UI wiring: Documents page stores `vector_index`, `hybrid_retriever`, and optional `graphrag_index` in `st.session_state`.
- Coordinator: best-effort analytics logging added after processing each query.
- Router toolset unified: `router_factory.build_router_engine(...)` composes `semantic_search`, `hybrid_search`, and `knowledge_graph` tools; selector policy prefers `PydanticSingleSelector` then falls back to `LLMSingleSelector`.
- GraphRAG helpers (`graph_config.py`) now emit label-preserving exports and provide `get_export_seed_ids()` for deterministic seeding.
- Snapshot manifest enriched and corpus hashing normalized to relpaths; Chat autoload/staleness detection wired to these fields.
- Docs updated: README and system-architecture examples use `MultiAgentCoordinator` directly instead of removed helpers.
- UI refactor: `src/app.py` now only defines pages and runs navigation; all monolithic UI logic moved to `src/pages/*`.
- Tests now patch real library seams (LlamaIndex, utils) instead of `src.app` re-exports.
- Removed short-lived re-exports from `src/app.py` (e.g., LlamaIndex classes, loader helpers) to maintain strict production/test separation.

### Removed
- Legacy ingestion/analytics modules (document processor, cache manager, legacy telemetry instrumentation) and associated compatibility shims/tests.
- Legacy `openai_like_*` fields from settings and corresponding env keys from `.env.example`.
- Legacy `retrieval.hybrid_enabled` and `retrieval.dbsf_enabled`; tests updated accordingly.
- Duplicate and conflicting env keys in `.env.example`.
- Compatibility shims for `DOCMIND_ALLOW_REMOTE_ENDPOINTS`; remote access policy now lives solely under `security.*`.
- Removed legacy helpers from `src/app.py`; app remains a thin multipage shell.

### Tests
- Rebuilt unit/integration suites for ingestion, snapshot locking, GraphRAG seed policy, Streamlit pages, and observability helpers (`tests/unit/processing/test_ingestion_pipeline.py`, `tests/unit/persistence/test_snapshot_*`, `tests/unit/retrieval/test_graph_seed_policy.py`, `tests/unit/ui/test_documents_snapshot_utils.py`, `tests/unit/observability/test_config.py`, integration UI tests).
- Coverage workflow consolidated under `scripts/run_tests.py --coverage` with HTML/JSON/XML artifacts.
- Updated unit and integration tests for new openai.*, security.*, and unified hybrid policy.
- Adjusted factory tests to expect /v1-normalized api_base for OpenAI-like servers.
- Removed legacy env toggle tests and added/updated allowlist and normalization tests.
- New unit tests: allowlist validation (`tests/unit/config/test_settings_allowlist.py`), prompt helper (`tests/unit/prompting/test_helpers.py`), clear caches helper (`tests/unit/ui/test_cache_clear.py`).
- Removed dependency on deleted app helpers: `tests/unit/app/test_app.py` removed.
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
  - `tests/unit/retrieval/test_hybrid_retriever_basic.py` (dedup determinism; sparse-unavailable dense fallback)
  - `tests/unit/retrieval/test_router_factory_hybrid.py` (vector + hybrid + knowledge_graph tools)
  - `tests/unit/retrieval/test_seed_policy.py` (retriever-first seed policy and fallbacks)
  - `tests/unit/retrieval/test_graph_helpers.py` (label preservation + `related` fallback)
  - `tests/unit/persistence/test_corpus_hash_relpaths.py` (relpath hashing determinism)
  - Updated/removed legacy integration tests; examples now use `router_factory`

### Docs
- SPEC-014, SPEC-006, SPEC-012, requirements, traceability matrix, README, overview, PRD, and ADR-031/033/019 updated to describe the new ingestion pipeline, snapshot/lock semantics, GraphRAG workflow, and observability configuration.
- ADR-024 amended with OpenAI-compatible servers and openai.* group; documented idempotent `/v1` base URL policy and linked to the canonical configuration guide.
- Configuration Reference updated with a canonical “OpenAI-Compatible Local Servers” section and a Local vs Cloud configuration matrix.
- README updated with DOCMIND_OPENAI__* examples (LM Studio, vLLM, llama.cpp) and a link to the canonical configuration section.
- SPEC-001 (LLM Runtime) updated to reflect OpenAILike usage for OpenAI-compatible backends and corrected Settings page path.
- Traceability (FR-SEC-NET-001) updated for OpenAI-like `/v1` normalization and local-first default posture.
- Requirements specification aligned with nested `openai.*`, `security.*`, and `retrieval.hybrid.*` groups, reiterated the local-first security policy, and linked to the canonical configuration guide.
- Docs updated: README and system-architecture examples use `MultiAgentCoordinator` directly instead of removed helpers.

### Tooling
- CI workflow pins `uv sync --extra observability --extra test --frozen`, runs Ruff/pylint/test gates, and enforces updated formatting/lint rules.
- Developer documentation references the new extras, commands, and smoke scripts for ingestion and telemetry verification.

### Reranking/Multimodal Consolidation
- Centralized device and VRAM policy via `src/utils/core` (`select_device`, `has_cuda_vram`) and delegated usage in embeddings and multimodal helpers.
- Unified SigLIP loader (`src/utils/vision_siglip.py`) reused by adapter for consistent caching and device placement.
- Enforced minimal reranking telemetry schema (stage, topk, latency_ms, timeout) with deterministic sorting and RRF tie-breakers.

### Fixed
- Read-only settings panel simplified; no longer references removed `reranker_mode`.
- README updated with offline predownload steps and new envs.
- `validate_startup_configuration` handles Qdrant `ResponseHandlingException`/`UnexpectedResponse` as connectivity failures with structured results (production-safe, test-friendly behavior).

### Security
- Hardened endpoint allowlist validation in `src/config/settings.py` to validate parsed hostnames/IPs and block spoofed `localhost` and malformed URLs.
- Router telemetry test stability: patch `log_jsonl` on the module object via `importlib` to account for LangChain `StructuredTool` wrapper.
- Security: `validate_export_path` error messages aligned with tests and documentation (egress/traversal vs. outside project root).
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
- Canonical env override: `DOCMIND_RETRIEVAL__USE_RERANKING=true|false` (no UI toggle).
- Deprecated: `DOCMIND_DISABLE_RERANKING` (use `DOCMIND_RETRIEVAL__USE_RERANKING`).

- Test stability and design-for-testability:
  - Removed the last test-only seam from production code: integrations no longer expose a `ClipEmbedding` alias or accept test-only injection. Embedding setup always uses `HuggingFaceEmbedding`; tests patch the constructor via `monkeypatch` when needed.
  - Reverted/avoided test shims in `src/app.py`, `src/retrieval/router_factory.py`, and page modules; imports are production-only.
  - Stabilized import-order–sensitive UI and persistence tests by patching the consumer seams directly and, where necessary, clearing only the specific modules in test-local `conftest.py` (no global module cache hacks). AppTest-based UI tests patch the page module’s attributes (`build_router_engine`, export helpers) instead of relying on import order.
  - Snapshot roundtrip tests stub `llama_index` loaders deterministically by overriding `sys.modules` for the exact import points used by the helpers.

### Fixed

- Read-only settings panel simplified; no longer references removed `reranker_mode`.
- README updated with offline predownload steps and new envs.
- `validate_startup_configuration` handles Qdrant `ResponseHandlingException`/`UnexpectedResponse` as connectivity failures with structured results (production-safe, test-friendly behavior).

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

- Retrieval & Reranking:
  - Router parity: RouterQueryEngine now passes reranking `node_postprocessors` for vector/hybrid/KG tools when `DOCMIND_RETRIEVAL__USE_RERANKING=true` (mirrors ToolFactory). Safe fallbacks keep older signatures working.
  - Tests: Added router_factory injection toggle test and KG fallback tests; added hybrid injection test behind explicit `enable_hybrid=True` with stubs.

### Docs/Specs/RTM

- Specs updated:
  - spec‑014: Added UI staleness badge exact tooltip, single‑writer lock semantics with timeout, manifest fields, atomic rename guidance, and acceptance/UX mapping.
  - spec‑004: Clarified server‑side‑only hybrid via Qdrant Query API (Prefetch + FusionQuery), named vectors + IDF, dedup, and telemetry; prohibited client‑side fusion and UI fusion toggles.
  - spec‑006: Defined GraphRAG exports and seeds policy (JSONL required; Parquet optional; deterministic deduped seeds capped at 32).
  - spec‑002/spec‑003/spec‑005: Clarified ingestion OCR fallback; embedding routing/dimensions (BGE‑M3 1024; SigLIP); always‑on reranking and ColPali gating.
  - spec‑010: Documented offline evaluation CLIs and CSV schema expectations; strict mocks/no heavy downloads in CI.
  - spec‑012: Added canonical telemetry events (router_selected, snapshot_stale_detected, export_performed, traversal_depth) and DuckDB analytics guidance.
  - spec‑013: Documented offline mode (HF_HUB_OFFLINE) and Parquet extras; JSONL fallback when pyarrow missing.
  - spec‑001/spec‑011: Settings scope & validation; offline‑first allowlist; LM Studio /v1 rule; selector policy; secrets redaction and non‑egress export requirements.
  - ADR‑011: Supervisor output_mode (last_message/full_history), add_handoff_messages rename, streaming fallback, and best‑effort analytics guidance.
  - ADR‑024: Offline defaults and endpoint allowlist policy.

- Requirements/RTM:
  - requirements.md: Added FR‑009.1–009.6 (staleness badge; SnapshotManager lock/rename; exports JSONL/Parquet; deterministic seed policy; export path security; telemetry events) and FR‑SEC‑NET‑001 (offline‑first allowlist; LM Studio /v1).
  - traceability.md: Mapped new FRs to code/tests and marked them Implemented.

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
- Server-side hybrid toggle and UI
  - Added `retrieval.enable_server_hybrid` in `src/config/settings.py` (default off) and wired precedence in `src/retrieval/router_factory.py` (explicit param > settings > default).
  - Added Settings UI toggle in `src/pages/04_settings.py` with `.env` persistence via `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID`.
- Ingestion analytics
  - Added best-effort analytics logging inside `src/processing/document_processor.py` using `AnalyticsManager` (DuckDB prepared statements). Logs ingest latency and element counts behind `analytics_enabled`.
- Telemetry/security completeness
  - Enriched reranker telemetry with `rerank.path` and `rerank.total_timeout_budget_ms` in `src/retrieval/reranking.py`.
  - Added endpoint allowlist tests under `tests/unit/config/test_endpoint_allowlist.py`.
- Hook robustness
  - Hardened LangGraph pre/post hooks in `src/agents/coordinator.py` to annotate state on failures (`hook_error`, `hook_name`) without crashing.
- Micro-tests for stability/coverage
  - Added tests: router settings fallback, hooks resilience, security utils, sparse query encoding, and settings round‑trip.

### Changed

- Minor router_factory docs/comments; no behavior change beyond settings fallback tests.

### Tests

- `tests/unit/retrieval/test_router_factory_settings_fallback.py` to validate settings‑driven hybrid tool registration.
- `tests/unit/agents/test_hooks_resilience.py` to ensure hook exceptions are non‑fatal.
- `tests/unit/utils/test_security.py` for PII redaction/egress checks.
- `tests/unit/retrieval/test_sparse_query_encode.py` for sparse query encoding path.
- `tests/unit/config/test_settings_roundtrip.py` for retrieval flag presence.
- BREAKING: Removed `src/agents` public re-exports. All imports must use explicit module paths, e.g. `from src.agents.coordinator import MultiAgentCoordinator` and `from src.agents.tool_factory import ToolFactory`.
- Docs updated to reflect explicit import guidance and to avoid aggregator/app re-exports in tests.
- Deprecated/legacy aggregator import examples were removed from documentation.
