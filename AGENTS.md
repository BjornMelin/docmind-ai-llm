# DocMind AI - Agent Instructions

## Purpose

Repo guardrails for contributors/automation. Keep aligned with code, `pyproject.toml`,
and docs under `docs/specs/` + `docs/developers/adrs/`.

## Layout

- `src/app.py`: Streamlit entrypoint
- `src/pages/`: UI pages (chat/documents/analytics/settings)
- `src/config/`: settings + integration wiring
- `src/processing/`: ingestion, OCR, PDF page exports
- `src/retrieval/`: router, hybrid retrieval, reranking, GraphRAG helpers
- `src/agents/`: LangGraph coordinator (package name `langgraph-supervisor` stays)
- `src/persistence/`: snapshots, hashing, locking, chat DB
- `src/telemetry/` + `src/utils/telemetry.py`: OTEL + JSONL events
- `templates/`: prompt templates/presets
- `docs/specs/` + `docs/developers/adrs/`: specs/ADRs (source-of-truth docs)
- `scripts/` + `tools/`: dev utilities (tests, perf, model pull)

## Quick Commands (uv)

- Setup: `uv sync && cp .env.example .env`
- Run: `streamlit run src/app.py` (or `./scripts/run_app.sh`)
- Env: prefer `uv run ...` (uses the project env, typically `.venv`).
- Verify (batch): after a batch of edits, run lint/type on touched paths + focused tests.
  - Lint (all): `uv run ruff format . && uv run ruff check . --fix`
  - Type (paths): `uv run pyright --threads 4 <paths>`
  - Tools-only (when `tools/` changed): `uv run pyright --threads 4 tools`
  - Tests (focused): `uv run pytest <tests/...> -vv` (or `-k <expr>` for a narrow slice)
- Verify (final): before finishing the task/prompt, run full lint/type then full tests: `uv run ruff format . && uv run ruff check . --fix && uv run pyright --threads 4 && uv run python scripts/run_tests.py`
- Tests (fast): `uv run python scripts/run_tests.py --fast`
- Coverage: `uv run python scripts/run_tests.py --coverage`
- Coverage report: `uv run python scripts/check_coverage.py --collect --report --html`
- Quality gates (CI): `uv run python scripts/run_quality_gates.py --ci --report`
- Perf check: `uv run python scripts/performance_monitor.py --run-tests --check-regressions --report`
- GPU check: `uv run python scripts/test_gpu.py --quick`
- Prefetch models: `uv run python tools/models/pull.py --all --cache_dir ./models_cache`
- spaCy model (opt): `uv run python -m spacy download en_core_web_sm`
- Review triage: `python3 scripts/analyze_github_reviews.py --json-file <path>` (or set `DOCMIND_REVIEW_JSON`)

## Non-negotiables (CI + Security)

- No `TODO|FIXME|XXX` under `src tests docs scripts tools`.
- CI expects `ruff format --check` and clean `ruff check` (CI runs `ruff check --fix --exit-non-zero-on-fix`).
- Offline-first default: no implicit egress; remote endpoints stay gated by allowlist policy.
- Streamlit: no `unsafe_allow_html=True` for untrusted content.
- Logging/telemetry: metadata-only; never log secrets or raw prompt/doc/model output (use `src/utils/log_safety.py`).

## Optional extras

- `uv sync --extra gpu` (vLLM, FlashInfer, fastembed-gpu)
- `uv sync --extra graph` (GraphRAG adapters)
- `uv sync --extra multimodal` (ColPali reranker)
- `uv sync --extra observability` (OTLP exporters + portalocker)
- `uv sync --extra eval` (ragas, beir)

## Dependency constraints (don’t drift)

Source of truth for exact pins: `pyproject.toml` + `uv.lock`.

- Python: `>=3.11,<3.12`
- Keep these coupled:
  - Torch 2.7.x ↔ vLLM 0.10.x ↔ Transformers `<4.58`
  - DuckDB `<1.4.0` (LlamaIndex integrations cap it)
  - LlamaIndex packages stay `<0.15.0`
  - Streamlit `<2.0.0`
  - `[tool.uv]` enforces `rapidfuzz>=3.14.1,<4.0.0`

## Configuration

- Source of truth: `src/config/settings.py` (Pydantic Settings v2).
- Env: prefix `DOCMIND_`, nested `__`.
- Prefer `from src.config import settings` (exports the settings object).
- No `os.getenv`/`os.environ` outside `src/config/*` (env bridges live there).
- Avoid import-time IO; use `startup_init()` + `initialize_integrations()` (`src/config/integrations.py`).
- Agent tool parallelism: `DOCMIND_AGENTS__ENABLE_PARALLEL_TOOL_EXECUTION=true|false` (ADR-010).

## LLM backends

- Backends: `ollama|vllm|lmstudio|llamacpp` via `DOCMIND_LLM_BACKEND`.
- OpenAI-compatible base URLs are normalized to a single `/v1` segment.
- Prefer `DOCMIND_OPENAI__BASE_URL` + `DOCMIND_OPENAI__API_KEY` for OpenAI-like servers (local servers can use placeholder keys).
- Backend-specific base URLs: `DOCMIND_LMSTUDIO_BASE_URL`, `DOCMIND_VLLM__VLLM_BASE_URL`, `DOCMIND_LLAMACPP_BASE_URL`.
- `DOCMIND_MODEL` overrides `DOCMIND_VLLM__MODEL`.

## Security policy

- Remote endpoints blocked unless `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true` or host is in `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST`.
- Analytics page is gated by `DOCMIND_ANALYTICS_ENABLED=true` and reads `data/analytics/analytics.duckdb`.

## Containerization (CI-enforced)

- `Dockerfile`: Python 3.11 base; final `USER` non-root; `CMD`/`ENTRYPOINT` exec-form (no `sh -c`); no `:latest`.
- `.dockerignore`: ignore `.env` (`.env`, `.env.*`, or `.env*`); don’t bake `.env` into images.
- Compose: use canonical `DOCMIND_*` env vars (no legacy `OLLAMA_BASE_URL`/`VLLM_BASE_URL`/`LMSTUDIO_BASE_URL`); prod override sets `read_only: true` and `tmpfs: /tmp`.

## Streamlit UI

- Keep import-time work minimal (pages rerun).
- Use `st.cache_resource` for long-lived objects (DB connections, checkpointers, clients).
- AppTest: prefer deterministic stubs + small render-time budgets over increasing `timeout=...`.

## Ingestion / processing

- Use LlamaIndex `IngestionPipeline` with `TokenTextSplitter` (and optional `TitleExtractor`).
- Prefer `UnstructuredReader` when installed; fall back to plain-text read.
- Cache: DuckDB KV store (`DOCMIND_CACHE__DIR`, `DOCMIND_CACHE__FILENAME`).
- PDF page images: PyMuPDF rendering; optional AES-GCM (`DOCMIND_IMG_AES_KEY_BASE64`, `DOCMIND_IMG_DELETE_PLAINTEXT`).
- Multimodal artifacts: store page images/thumbnails as content-addressed `ArtifactRef` in the local ArtifactStore (`DOCMIND_ARTIFACTS__*`).

## Retrieval

- Text embeddings: BGE-M3 (1024D). Sparse: FastEmbed BM42 preferred, BM25 fallback.
- Hybrid retrieval: Qdrant server-side via `Prefetch` + `FusionQuery` (RRF default; DBSF env-gated).
- Named vectors: `text-dense`, `text-sparse` (+ IDF modifier). Dedup by `page_id` (default) or `doc_id`.
- Enable server-side hybrid: `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=true`.
- Multimodal search: SigLIP text→image fused by RRF; image collection uses `settings.database.qdrant_image_collection`.

## Router

- Router composes tools: `semantic_search`, optional `hybrid_search`, optional `knowledge_graph`.
- Selector: prefer `PydanticSingleSelector`; fall back to `LLMSingleSelector`.
- Build routers via `src/retrieval/router_factory.py` to keep tool metadata/postprocessors consistent.

## Reranking

- Text rerank: BGE cross-encoder (`BAAI/bge-reranker-v2-m3`).
- Visual rerank: SigLIP when image nodes exist; ColPali is optional (`--extra multimodal`).
- Fail open on timeouts; respect `settings.retrieval.*_timeout_ms`.

## DSPy (opt)

- Gated by `DOCMIND_ENABLE_DSPY_OPTIMIZATION=true`; guard imports and fail open if unavailable.

## GraphRAG

- Enable only when both are true: `DOCMIND_ENABLE_GRAPHRAG=true` and `DOCMIND_GRAPHRAG_CFG__ENABLED=true`.
- Requires `--extra graph` adapters (kuzu + LlamaIndex).
- Exports: JSONL baseline; Parquet optional (PyArrow). Preserve `get_rel_map` labels (fallback `related`).
- Seed policy: graph retriever → vector retriever → deterministic fallback.

## Snapshots

- Location: `data/storage/`; write via temp workspace + atomic finalize.
- Manifest triad: `manifest.jsonl`, `manifest.meta.json`, `manifest.checksum`.
- `manifest.meta.json` includes `index_id`, `graph_store_type`, `vector_store_type`, `corpus_hash`, `config_hash`, `versions`, optional `graph_exports`.
- Don’t persist base64 blobs or raw filesystem paths; persist `ArtifactRef` and sanitize path-like metadata to basenames.
- Locking: `portalocker` when available (fallback `O_EXCL`). Persist via `SnapshotManager` APIs.

## SQLite / WAL

- Don’t share a sqlite connection across threads; prefer one connection per operation.
- Persist metadata only (no raw prompt/doc text).

## Observability / telemetry

- OTEL is optional: `DOCMIND_OBSERVABILITY__ENABLED=true` + `--extra observability`.
- JSONL telemetry is local-first (`logs/telemetry.jsonl`): `router_selected`, `export_performed`, `snapshot_stale_detected`, rerank fallback events.
- Log safety:
  - No secrets/API keys; no raw prompt/doc/model output.
  - URLs: log origin only; sanitize exception strings before logging.
- Controls: `DOCMIND_TELEMETRY_DISABLED`, `DOCMIND_TELEMETRY_SAMPLE`, `DOCMIND_TELEMETRY_ROTATE_BYTES`.

## Agents (budgets)

- When deadline propagation is enabled, cap per-call timeouts to the supervisor budget (avoid a single call exceeding `settings.agents.decision_timeout`).

## Offline-first

- Predownload models: `tools/models/pull.py` (avoid runtime downloads).
- Use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline runs; tests must remain deterministic/offline unless explicitly marked.

## Coding standards

- Prefer library primitives; keep changes small and typed.
- Prefer lazy imports for heavy ML deps (see `src/config/integrations.py`).
- Use `loguru.logger`; avoid `print` in production code.
- Use `pathlib.Path` and guard clauses; avoid `Any` unless isolated behind a narrow boundary.
- Device/VRAM policy: use `src/utils/core.py` (`resolve_device`, `has_cuda_vram`) for business logic.

## Testing

- Boundary-first; keep unit tests <5s and integration <30s when possible.
- Use markers: `unit|integration|system` + `requires_gpu|requires_network|requires_ollama`.
- AppTest: render-time budgets (looser in CI) + deterministic stubs.
- Prefer patching real consumer seams, not `src.app` (see `docs/developers/testing-notes.md`).

## Docs

- Specs: `docs/specs/_index.md`
- ADRs: `docs/developers/adrs/` (do not implement anything under `docs/developers/adrs/superseded/`)
- When behavior changes, update README and the owning spec/ADR.
- Keep active docs pointing at real `src/...` paths (avoid doc drift).

## Browser Automation

Use `agent-browser` for web automation. Run `agent-browser --help` for all commands.

Core workflow:

1. `agent-browser open <url>` - Navigate to page
2. `agent-browser snapshot -i` - Get elements with refs (@e1, @e2)
3. `agent-browser click @e1` / `fill @e2 "text"` - Interact using refs
4. Re-snapshot after page changes

## Source Code Reference

Source code for deps is available in `opensrc/` for deeper understanding of implementation details.

See `opensrc/sources.json` for the list of available packages and their versions.

Use this source code when you need to understand how a package works internally, not just its types/interface.

### Fetching Additional Source Code

To fetch source code for a package or repository you need to understand, run:

```bash
# NOTE: append the --modify=false to all commands
npx opensrc <package>           # npm package (e.g., npx opensrc zod)
npx opensrc pypi:<package>      # Python package (e.g., npx opensrc pypi:requests)
npx opensrc crates:<package>    # Rust crate (e.g., npx opensrc crates:serde)
npx opensrc <owner>/<repo>      # GitHub repo (e.g., npx opensrc vercel/ai)
```
