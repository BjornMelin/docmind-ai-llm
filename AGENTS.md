# DocMind AI - Agent Instructions

## Purpose

This file is the operating guide for contributors and automation. Keep it in sync
with the current codebase, pyproject.toml, scripts, and docs/specs/ADRs.

## Repository Layout

- `src/app.py`: Streamlit entrypoint
- `src/pages/`: UI pages (chat, documents, analytics, settings)
- `src/config/`: Pydantic Settings + LLM integration wiring
- `src/processing/`: Ingestion pipeline, OCR, PDF page exports
- `src/retrieval/`: Router, hybrid retrieval, reranking, GraphRAG helpers
- `src/agents/`: LangGraph coordinator, tools (external package name `langgraph-supervisor` remains unchanged)
- `src/persistence/`: Snapshot writer, hashing, locking
- `src/telemetry/` and `src/utils/telemetry.py`: OpenTelemetry hooks + JSONL events
- `templates/`: Prompt templates and presets
- `docs/specs/` and `docs/developers/adrs/`: specs and ADRs
- `scripts/`, `tools/`: run and utility scripts (tests, performance, model pull)

```text
/
|-- src/
|   |-- agents/
|   |-- config/
|   |-- core/
|   |-- pages/
|   |-- persistence/
|   |-- processing/
|   |-- retrieval/
|   |-- telemetry/
|   |-- utils/
|   `-- models/
|-- templates/
|-- tests/
|-- scripts/
|-- tools/
|-- docs/
|-- pyproject.toml
|-- README.md
`-- .env.example
```

## Quick Commands (uv only)

- Setup: `uv sync && cp .env.example .env`
- Run app: `streamlit run src/app.py`
- Run app with port env: `./scripts/run_app.sh`
- Lint/format: `uv run ruff format . && uv run ruff check . --fix && uv run pyright`
- Tests (fast tier): `uv run python scripts/run_tests.py --fast`
- Tests (all): `uv run python scripts/run_tests.py`
- Coverage: `uv run python scripts/run_tests.py --coverage`
- Quality gates: `uv run python scripts/run_quality_gates.py --ci --report`
- Coverage report: `uv run python scripts/check_coverage.py --collect --report --html`
- Performance check: `uv run python scripts/performance_monitor.py --run-tests --check-regressions --report`
- GPU check: `uv run python scripts/test_gpu.py --quick`
- Prefetch models: `uv run python tools/models/pull.py --all --cache_dir ./models_cache`
- spaCy model (optional): `uv run python -m spacy download en_core_web_sm`

Optional extras:

- `uv sync --extra gpu` (vLLM, FlashInfer, fastembed-gpu)
- `uv sync --extra graph` (GraphRAG adapters)
- `uv sync --extra multimodal` (ColPali reranker)
- `uv sync --extra observability` (OTLP exporters + portalocker)
- `uv sync --extra eval` (ragas, beir)

## Dependency Constraints and Compatibility (do not violate)

### Runtime and Core Libraries

- Python: `>=3.11,<3.12` (strict)
- Pydantic: `pydantic==2.11.7`, `pydantic-settings==2.10.1`
- Streamlit: `>=1.52.2,<2.0.0`
- Unstructured: `unstructured[all-docs]>=0.18.26,<0.19.0`, `unstructured-ingest>=1.2.32,<2.0.0`
- PyMuPDF: `pymupdf==1.26.4`
- python-docx: `python-docx==1.2.0`
- Pillow: `pillow>=11.0.0,<12.0.0`
- PyArrow: `pyarrow>=21.0.0,<22.0.0`
- spaCy: `spacy>=3.8.7,<4.0.0` (models installed separately)
- Torch: `torch==2.7.1`
- Transformers: `>=4.55.0,<4.58.0` (aligned with vLLM 0.10.x)
- tiktoken: `tiktoken==0.11.0`
- openai-whisper: `openai-whisper==20250625`
- Qdrant client: `>=1.15.1,<2.0.0`
- DuckDB: `>=1.3.2,<1.4.0` (LlamaIndex DuckDB integrations cap <1.4.0)
- OpenAI client: `>=1.109.1,<2.0.0`
- pandas: `>=2.2,<3.0`
- plotly: `>=5.22,<6.0`
- python-dotenv: `python-dotenv==1.1.1`

### LlamaIndex Bundle (keep in lockstep)

- `llama-index>=0.14.12,<0.15.0`
- `llama-index-vector-stores-qdrant>=0.9.0,<0.10.0`
- `llama-index-vector-stores-duckdb>=0.5.1,<0.6.0`
- `llama-index-storage-kvstore-duckdb>=0.2.1,<0.3.0`
- `llama-index-llms-vllm>=0.6.1,<0.7.0`
- `llama-index-llms-openai>=0.6.12,<0.7.0`
- `llama-index-llms-ollama>=0.9.1,<1.0.0`
- `llama-index-llms-llama-cpp>=0.5.1,<0.6.0`
- `llama-index-llms-openai-like>=0.5.1,<0.6.0`
- `llama-index-embeddings-huggingface>=0.6.1,<0.7.0`
- `llama-index-embeddings-clip>=0.5.1,<0.6.0`
- `llama-index-embeddings-fastembed>=0.5.0,<0.6.0`
- `llama-index-embeddings-openai>=0.5.1,<0.6.0`
- `llama-index-readers-file>=0.5.6,<0.6.0`

### Agent Orchestration

- `langgraph>=1.0.5,<2.0.0`
- `langgraph-supervisor>=0.0.31,<0.1.0`
- `langchain-core>=1.2.6,<2.0.0`
- `langchain-openai>=1.1.6,<2.0.0`

### Retrieval/Reranking

- `FlagEmbedding>=1.3.5,<2.0.0` (BGE-M3)
- `sentence-transformers>=5.2.0,<6.0.0` (cross-encoder)

### Observability

- `opentelemetry-sdk>=1.39.1,<2.0.0`
- `opentelemetry-exporter-otlp-proto-http>=1.39.1,<2.0.0`
- `opentelemetry-exporter-otlp-proto-grpc>=1.39.1,<2.0.0`
- `portalocker>=3.2.0,<4.0.0`

### Optional Extras

- GPU: `vllm>=0.10.1,<0.11.0`, `flashinfer-python>=0.5.3,<0.6.0`, `fastembed-gpu>=0.7.4,<0.8.0`
- GraphRAG: `llama-index-graph-stores-kuzu>=0.9.1,<1.0.0`, `kuzu>=0.11.3,<1.0.0`
- Multimodal: `llama-index-postprocessor-colpali-rerank==0.3.1`, `colpali-engine>=0.3.13,<0.4.0`

### Constraints to Avoid Conflicts

- vLLM 0.10.x requires Torch 2.7.x; do not upgrade Torch without upgrading vLLM
  and Transformers together.
- DuckDB must stay <1.4.0 while LlamaIndex DuckDB integrations cap it.
- Keep all LlamaIndex packages within the same minor range (<0.15.0).
- Streamlit must remain <2.0.0.
- `rapidfuzz>=3.14.1,<4.0.0` is enforced in `[tool.uv]` to avoid a yanked build.

## Configuration and Environment

- Source of truth: `src/config/settings.py` (Pydantic BaseSettings).
- Env prefix: `DOCMIND_`; nested fields use `__`.
- Use `from src.config import settings` and read from settings.
- Do not use `os.getenv` directly in core code.
- Avoid import-time IO; use `startup_init()` and `initialize_integrations()` from
  `src/config/integrations.py` to set up directories, LlamaIndex Settings, and OTel.
- Agent tooling: `DOCMIND_AGENTS__ENABLE_PARALLEL_TOOL_EXECUTION=true|false` controls parallel tool execution (see ADR-010).

LLM backends:

- `ollama`, `vllm`, `lmstudio`, `llamacpp` (set via `DOCMIND_LLM_BACKEND`).
- OpenAI-compatible servers must use base URLs normalized to include `/v1`.
- Base URL normalization and allowlist enforcement are centralized in settings.
- For OpenAI-compatible servers, prefer `DOCMIND_OPENAI__BASE_URL` and
  `DOCMIND_OPENAI__API_KEY` (local servers accept placeholder keys), or use
  backend-specific base URLs (`DOCMIND_LMSTUDIO_BASE_URL`, `DOCMIND_VLLM__VLLM_BASE_URL`,
  `DOCMIND_LLAMACPP_BASE_URL`).
- `DOCMIND_MODEL` (top-level) overrides `DOCMIND_VLLM__MODEL` at runtime.

Security policy:

- Remote endpoints are blocked unless `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true`
  or the host is in `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST`.
- Optional Analytics page is gated by `DOCMIND_ANALYTICS_ENABLED=true` and reads
  from `data/analytics/analytics.duckdb`.

## Ingestion and Processing

- Use LlamaIndex `IngestionPipeline` with `TokenTextSplitter` and optional
  `TitleExtractor` (library-first).
- Prefer `UnstructuredReader` when installed; fall back to plain-text read.
- Ingestion cache is DuckDB KV store (`DOCMIND_CACHE__DIR`, `DOCMIND_CACHE__FILENAME`).
- PDF page images are rendered via PyMuPDF; optional AES-GCM encryption with
  `DOCMIND_IMG_AES_KEY_BASE64` and `DOCMIND_IMG_DELETE_PLAINTEXT`.

## Embeddings and Retrieval

- Text embeddings: BGE-M3 (1024D) via LlamaIndex Settings.
- Sparse queries: FastEmbed BM42 preferred, BM25 fallback when unavailable.
- Hybrid retrieval is server-side via Qdrant Query API:
  - Use `Prefetch` for dense + sparse and `FusionQuery` with RRF (default) or DBSF.
  - Named vectors must exist: `text-dense`, `text-sparse` with IDF modifier.
  - Deduplicate by `page_id` (default) or `doc_id` before final cut.
- Enable server-side hybrid with `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=true`.

## Router and Tooling

- RouterQueryEngine composes tools: `semantic_search`, optional `hybrid_search`,
  optional `knowledge_graph`.
- Selector preference: `PydanticSingleSelector` when supported; fallback to
  `LLMSingleSelector` (matches LlamaIndex router docs).
- Build routers via `src/retrieval/router_factory.py` to keep postprocessors
  and tool metadata consistent.

## Reranking

- Default rerank: BGE cross-encoder (`BAAI/bge-reranker-v2-m3`).
- Visual rerank: SigLIP when image nodes exist; ColPali optional with
  `--extra multimodal`.
- Reranking must fail open on timeouts and keep latency budgets
  (`settings.retrieval.*_timeout_ms`).

DSPy (optional):

- DSPy query optimization is gated by `DOCMIND_ENABLE_DSPY_OPTIMIZATION=true`.
- Guard imports and fail open when DSPy is unavailable.

## GraphRAG

- Enable only when both are true:
  - `DOCMIND_ENABLE_GRAPHRAG=true`
  - `DOCMIND_GRAPHRAG_CFG__ENABLED=true`
- Requires `--extra graph` adapters (kuzu + LI graph stores).
- Exports: JSONL baseline; Parquet optional (requires PyArrow).
- Seed policy: graph retriever -> vector retriever -> deterministic fallback.
- Relation labels: preserve `get_rel_map` labels; fallback to `related`.

## Snapshots and Persistence

- Snapshots live under `data/storage/` and use a temp workspace pattern.
- Manifest triad: `manifest.jsonl`, `manifest.meta.json`, `manifest.checksum`.
- `manifest.meta.json` must include `index_id`, `graph_store_type`,
  `vector_store_type`, `corpus_hash`, `config_hash`, `versions`, and
  optional `graph_exports`.
- Snapshot locking uses `portalocker` when available (fallback O_EXCL).
- Always persist vector and graph indexes via `SnapshotManager` APIs.

## Observability and Telemetry

- OpenTelemetry is optional and disabled by default.
  - Enable with `DOCMIND_OBSERVABILITY__ENABLED=true` and `--extra observability`.
- LlamaIndex instrumentation is optional; guarded by extra package.
- JSONL telemetry is local-first (`logs/telemetry.jsonl`) and includes:
  `router_selected`, `export_performed`, `snapshot_stale_detected`,
  plus rerank/postprocessor fallback events.
- Telemetry sampling/disable controls (env):
  - `DOCMIND_TELEMETRY_DISABLED`
  - `DOCMIND_TELEMETRY_SAMPLE=0.0..1.0`
  - `DOCMIND_TELEMETRY_ROTATE_BYTES=<int>`

## Offline-First Behavior

- Use `tools/models/pull.py` to predownload models.
- Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline runs.
- Avoid network calls in tests and runtime unless explicitly enabled.

## Coding Standards and Patterns

- Library-first, KISS, DRY, YAGNI.
- Use `pathlib.Path`, f-strings, and guard clauses.
- Prefer lazy imports for heavy ML dependencies (see `src/config/integrations.py`).
- Log with `loguru.logger`; avoid `print` in production code.
- All new functions should be typed; avoid `Any` unless necessary.

## Testing Principles

- Boundary-first testing (see `docs/testing/testing-guide.md`).
- Use `tests/integration/conftest.py` MockLLM fixture to keep tests offline.
- Mark tests with `unit`, `integration`, `system`, plus resource markers
  (`requires_gpu`, `requires_network`, `requires_ollama`).
- Keep unit tests fast (<5s) and integration tests <30s where possible.

## Docs and Architecture Sources

- Specs: `docs/specs/_index.md` (LLM runtime, ingestion, embeddings, hybrid,
  reranking, GraphRAG, persistence, observability, packaging).
- ADRs: `docs/developers/adrs/` (configuration, persistence, testing, GraphRAG).
- When behavior changes, update README and relevant spec/ADR.

<!-- opensrc:start -->
## opensrc Reference Library

`opensrc/` contains **dependency source snapshots** for deeper internals/edge cases (see `opensrc/sources.json`).

Guidelines:

- Treat `opensrc/` as **read-only** local references; it is excluded from version control (see `.gitignore`) and Ruff linting to preserve snapshots.
- Prefer repo-truth (local code + official docs) first; use `opensrc/` when documentation is ambiguous or behavior is subtle.
- Always cite exact `opensrc/…` paths + versions when relying on internals in ADRs/SPECs or incident writeups.
- Fetch additional sources only when necessary to understand implementation details (not just public APIs).
- Prefer **non-interactive** runs when possible:
  - `npx opensrc pypi:<package>@<version> --modify=false`

Fetch additional sources as needed:

```bash
npx opensrc <package>           # npm package (e.g., npx opensrc zod)
npx opensrc pypi:<package>      # Python package (e.g., npx opensrc pypi:requests)
npx opensrc crates:<package>    # Rust crate (e.g., npx opensrc crates:serde)
npx opensrc <owner>/<repo>      # GitHub repo (e.g., npx opensrc vercel/ai)

# Manage sources
npx opensrc list
npx opensrc remove <name>
```

Refresh sources after dependency upgrades or when investigating a bug fixed upstream (remove then re-fetch).

<!-- opensrc:end -->
