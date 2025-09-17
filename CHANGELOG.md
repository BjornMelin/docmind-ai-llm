# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
* Canonical ingestion models and hashing helpers powering the library-first ingestion pipeline (`src/models/processing.py`, `src/persistence/hashing.py`).
* LlamaIndex-based ingestion pipeline, DuckDB-backed cache/docstore wiring, AES-GCM page-image exports, and OpenTelemetry spans (`src/processing/ingestion_pipeline.py`).
* Snapshot lock + writer modules with heartbeat/takeover metadata, atomic promotion, tri-file manifest, and timestamped graph export metadata (`src/persistence/lockfile.py`, `src/persistence/snapshot_writer.py`).
* GraphRAG router/query helpers using native LlamaIndex retrievers and telemetry instrumentation for export counters and spans (`src/retrieval/graph_config.py`, `src/retrieval/router_factory.py`, `src/agents/tools/router_tool.py`).
* Streamlit UI integration that surfaces manifest metadata, staleness badges, GraphRAG export tooling, and ingestion orchestration (`src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/ui/ingest_adapter.py`, `src/agents/coordinator.py`).
* Observability configuration + helpers for OTLP/console exporters and optional LlamaIndex instrumentation (`ObservabilityConfig`, `configure_observability`, updated `scripts/demo_metrics_console.py`).
* Quick-start demos for the overhaul: `scripts/run_ingestion_demo.py` (pipeline smoke test) and refreshed console metrics demo.

### Changed
* Snapshot manifest/schema now records `complete`, `schema_version`, `persist_format_version`, graph export metadata, and enforces `_tmp-` workspace + `CURRENT` pointer discipline.
* Router, UI, and telemetry layers consistently emit OpenTelemetry spans/metrics for ingestion, snapshot promotion, GraphRAG selection, and export flows.
* Packaging + CI rely on `uv` with an `observability` extra (OTLP exporters, portalocker, LlamaIndex OTEL) and run `ruff`, `pylint`, and `uv run scripts/run_tests.py --coverage` under locked environments.
* Shared fixtures/tests cover ingestion pipeline builders, snapshot locks, Streamlit AppTest interactions, and console exporter stubs.

### Removed
* Legacy ingestion/analytics modules (document processor, cache manager, legacy telemetry instrumentation) and associated compatibility shims/tests.

### Tests
* Rebuilt unit/integration suites for ingestion, snapshot locking, GraphRAG seed policy, Streamlit pages, and observability helpers (`tests/unit/processing/test_ingestion_pipeline.py`, `tests/unit/persistence/test_snapshot_*`, `tests/unit/retrieval/test_graph_seed_policy.py`, `tests/unit/ui/test_documents_snapshot_utils.py`, `tests/unit/observability/test_config.py`, integration UI tests).
* Coverage workflow consolidated under `scripts/run_tests.py --coverage` with html/json/xml artifacts.

### Docs
* SPEC-014, SPEC-006, SPEC-012, requirements, traceability matrix, README, overview, PRD, and ADR-031/033/019 updated to describe the new ingestion pipeline, snapshot/lock semantics, GraphRAG workflow, and observability configuration.

### Tooling
* CI workflow pins `uv sync --extra observability --extra test --frozen`, runs Ruff/pylint/test gates, and enforces updated formatting/lint rules.
* Developer documentation references the new extras, commands, and smoke scripts for ingestion + telemetry verification.

----


All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Streamlit ingestion adapter backed by the LlamaIndex `IngestionPipeline`, including
  deterministic manifest summaries, optional GraphRAG index creation, and
  OpenTelemetry spans for rebuild/export operations.
- UI telemetry scaffolding: `configure_observability` helper and OTEL span/counter
  instrumentation for the multi-agent coordinator and router tool.
- OpenAIConfig (openai.*) with idempotent /v1 base_url normalization and api_key.
- SecurityConfig (security.*) centralizing allow_remote_endpoints, endpoint_allowlist, trust_remote_code.
- HybridConfig (hybrid.*) declarative policy (enabled/server_side/method/rrf_k/dbsf_alpha).

### Changed
- Observability settings now live under `observability.*` with explicit endpoint, protocol, sampling, metrics interval, and LlamaIndex instrumentation toggles; `configure_observability` wires OTLP span/metric exporters and optionally registers `LlamaIndexOpenTelemetry`.
- Streamlit ingestion adapter consumes the new observability config and continues emitting graph export metrics via OpenTelemetry helpers.
- Documents and Chat pages now hydrate router/session state via the new ingestion
  adapter, surface manifest metadata from `manifest.meta.json`, and sanitize
  manual GraphRAG export paths.
- GraphRAG helpers rely solely on documented LlamaIndex APIs; router factory and
  tests updated accordingly.
- Enforced backend-aware OpenAI-like /v1 normalization in LLM factory for LM Studio, vLLM (OpenAI-compatible), and llama.cpp server.
- Moved all import-time I/O from settings into explicit startup_init(settings) in integrations.
- Unified server-side hybrid gating to retrieval.enable_server_hybrid + fusion_mode; removed legacy flags.
- Settings UI now shows read-only policy state (server-side hybrid, fusion mode, remote endpoint allowance, allowlist size) and resolved normalized backend base URL.
- .env.example rewritten to use DOCMIND_OPENAI__*, DOCMIND_SECURITY__*, and DOCMIND_VLLM__*; removed raw VLLM_* keys.

### Removed
- Legacy openai_like_* fields from settings and corresponding env keys from .env.example.
- Legacy retrieval.hybrid_enabled and retrieval.dbsf_enabled; tests updated accordingly.
- Duplicate and conflicting env keys in .env.example.
- Compatibility shims for `DOCMIND_ALLOW_REMOTE_ENDPOINTS`; remote access policy now lives solely under `security.*`.

### Tests
- Added OpenTelemetry instrumentation unit test coverage and refreshed ingestion adapter tests for the new settings schema.
- Updated unit and integration tests for new openai.*, security.*, and unified hybrid policy.
- Adjusted factory tests to expect /v1-normalized api_base for OpenAI-like servers.
- Removed legacy env toggle tests and added/updated allowlist and normalization tests.

### Docs
- Observability guide updated with `observability.*` configuration terminology.
- ADR‑024 amended with OpenAI‑compatible servers and openai.* group; documented idempotent `/v1` base URL policy and linked to the canonical configuration guide.
- Configuration Reference updated with a canonical “OpenAI‑Compatible Local Servers” section and a Local vs Cloud configuration matrix.
- README updated with DOCMIND_OPENAI__* examples (LM Studio, vLLM, llama.cpp) and a link to the canonical configuration section.
- SPEC‑001 (LLM Runtime) updated to reflect OpenAILike usage for OpenAI‑compatible backends and corrected Settings page path.
- Traceability (FR‑SEC‑NET‑001) updated for OpenAI‑like `/v1` normalization and local‑first default posture.
- Requirements specification aligned with nested `openai.*`, `security.*`, and `retrieval.hybrid.*` groups, reiterated the local-first security policy, and linked to the canonical configuration guide.

### Added

- Clear caches feature: Settings page button and `src/ui/cache.py` helper (bumps `settings.cache_version` and clears Streamlit caches).
- Pure prompting helper: `src/prompting/helpers.py` with `build_prompt_context` (pure; no UI/telemetry) and unit tests.
- SPEC‑008: Programmatic Streamlit UI with `st.Page` + `st.navigation`.
  - New pages: `src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/pages/03_analytics.py`.
  - New adapter: `src/ui/ingest_adapter.py` for form-based ingestion.
- ADR‑032: Local analytics manager (`src/core/analytics.py`) with DuckDB and best‑effort background writes; coordinator logs query metrics.
- SPEC‑013 (ADR‑040): Model pre‑download CLI `tools/models/pull.py` using `huggingface_hub`.
- SPEC‑010 (ADR‑039): Offline evaluation CLIs:
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

- BEIR and RAGAS CLIs now call determinism setup first; BEIR CLI respects `--k` for metric computation and emits dynamic headers matching `k`.
- CI workflow: added schema validation step after tests to catch leaderboard schema drift.
  
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
