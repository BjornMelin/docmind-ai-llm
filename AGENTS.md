# DocMind AI: Agent instructions

## Purpose

Repo guardrails for contributors/automation. Keep aligned with code, `pyproject.toml`,
and docs under `docs/specs/` + `docs/developers/adrs/`.

## Layout

- `app.py`: Streamlit entrypoint
- `src/app.py`: Streamlit app module (imported by `app.py`)
- `src/pages/`: UI pages (chat/documents/analytics/settings)
- `src/config/`: settings + integration wiring
- `src/processing/`: ingestion, OCR, PDF page exports
- `src/retrieval/`: router, hybrid retrieval, reranking, GraphRAG helpers
- `src/agents/`: LangGraph coordinator (graph-native supervisor via `StateGraph`)
- `src/persistence/`: snapshots, hashing, locking, chat DB
- `src/telemetry/` + `src/utils/telemetry.py`: OTEL + JSONL events
- `src/prompting/templates/`: bundled prompt templates and presets
- `docs/specs/` + `docs/developers/adrs/`: specs/ADRs (source-of-truth docs)
- `scripts/` + `tools/`: benchmarks, health checks, documentation gates, and model pull

## Quick commands with uv

- Setup: `uv sync && cp .env.example .env`
- Run: `uv run streamlit run app.py` (or `./scripts/run_app.sh`)
- Env: prefer `uv run ...` (uses the project env, typically `.venv`).
- Verify (batch): after a batch of edits, run lint/type on touched paths + focused tests.
  - Format (all): `uv run ruff format .`
  - Lint (all): `uv run ruff check .`
  - Type (paths): `uv run pyright --threads 4 <paths>`
  - Tools-only (when `tools/` changed): `uv run pyright --threads 4 tools`
  - Tests (focused): `uv run pytest <tests/...> -vv --no-cov` (or `-k <expr>` for a narrow slice)
- Verify (final): before finishing the task/prompt, run non-mutating full lint/type
  checks, then full tests: `uv run ruff format --check . && uv run ruff check . &&
  uv run pyright --threads 4 && uv run pytest tests/unit tests/integration -q --no-cov`
- Tests (unit): `uv run pytest tests/unit -q --no-cov`
- Tests (integration): `uv run pytest tests/integration -q --no-cov`
- Tests (fast/full): `uv run pytest tests/unit tests/integration -q --no-cov`
- Tests (GPU): `uv run pytest -m requires_gpu --no-cov`
- Coverage: `uv run pytest tests/unit tests/integration -q --cov=src --cov-branch --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-report=json:coverage.json --cov-fail-under=80 --junitxml=junit.xml`
- Coverage report: `uv run python scripts/check_coverage.py --collect --report --html`
- Parser benchmark: `uv run python scripts/benchmark_parsing.py --generate-minimal-fixtures --output cache/benchmarks/parsing/results.json`
- GPU check: `uv run python scripts/test_gpu.py --quick`
- Prefetch default retrieval and Docling layout models: `uv run python tools/models/pull.py --all --cache_dir ./models_cache --parser-defaults --parser-cache-dir ./cache/models`; RapidOCR models come from its locked wheel.
- spaCy model (opt): `uv run python -m spacy download en_core_web_sm`
- Review triage: `uv run python scripts/analyze_github_reviews.py --json-file <path>` (or set `DOCMIND_REVIEW_JSON`)

## Non-negotiables for CI and security

- No `TODO|FIXME|XXX` under `src tests docs scripts tools`.
- CI expects `ruff format --check` and a clean, non-mutating `ruff check`.
- Offline-first default: no implicit egress. Base URL validation is strict by default and includes DNS resolution of allowlisted non-loopback hosts as SSRF/DNS-rebinding hardening.
- Streamlit: no `unsafe_allow_html=True` for untrusted content.
- Logging/telemetry: metadata-only; never log secrets or raw prompt/doc/model output (use `src/utils/log_safety.py`).

## Compaction + continuity worklogs

This repo uses a lightweight, compaction-resilient log so we can resume work without re-researching or re-deciding.

Worklogs are **local-only** by default (gitignored) and should not be committed to the repo.

After any material research, decisions, or implementation:

1. Update `docs/developers/worklogs/CONTEXT.md` with:
   - current status + next steps
   - research notes (primary links)
   - key decisions + rationale
   - important quirks/constraints (no secrets)
2. If you call any `mcp__zen__*` tool that returns a `continuation_id`, record it in:
   - `docs/developers/worklogs/continuations.json`
3. For normative changes (architecture/policy): add or update an ADR under `docs/developers/adrs/`.
4. For behavior changes: update the owning spec under `docs/specs/` and keep `docs/specs/traceability.md` aligned.

Timing rule: write ADRs/spec updates **immediately when finalized** (do not batch them at the end of a long session) to avoid losing decision context during auto-compaction.

When resuming after compaction:

1. Read `docs/developers/worklogs/CONTEXT.md` first.
2. Read `docs/developers/worklogs/continuations.json` and reuse any stored `continuation_id` values when continuing `mcp__zen__*` threads.

## Optional extras

- `uv sync --frozen --no-group cpu --extra gpu` (PyTorch CUDA and CuPy;
  sparse FastEmbed remains CPU-based)
- `uv sync --frozen --extra observability` (LlamaIndex OpenTelemetry instrumentation)
- `uv sync --frozen --extra eval` (beir)

## Dependency constraints (don’t drift)

Source of truth for exact pins: `pyproject.toml` + `uv.lock`.

- Python: `>=3.12,<3.14` (primary dev/runtime: Python 3.12.13)
- Keep these coupled:
  - Torch 2.11.x ↔ Transformers `>=5.0,<6.0` (vLLM is external-only via OpenAI-compatible HTTP)
  - DuckDB `<1.4.0` (LlamaIndex integrations cap it)
  - `llama-index-core>=0.14.21,<0.15.0` plus selected integration packages
  - Direct `fastembed>=0.5.1`; do not restore the removed LlamaIndex FastEmbed adapter
  - Ollama Python client `0.6.2`
  - Streamlit `<2.0.0`
  - `[tool.uv]` enforces `rapidfuzz>=3.14.1,<4.0.0`

## Configuration

- Source of truth: `src/config/settings.py` (Pydantic Settings v2).
- Settings helpers (URL parsing/normalization + endpoint allowlist policy) live in `src/config/settings_utils.py` to keep `settings.py` focused on typed models and validators.
- Env: prefix `DOCMIND_`, nested `__`.
- Prefer `from src.config import settings` (exports the settings object).
- No `os.getenv`/`os.environ` outside `src/config/*` (env bridges live there).
- Avoid import-time IO; use `startup_init()` + `initialize_integrations()` (`src/config/integrations.py`).

## LLM backends

- Backends: `ollama|vllm|lmstudio|llamacpp|openai_compatible` via `DOCMIND_LLM_BACKEND`.
- OpenAI-compatible base URLs are normalized to a single `/v1` segment.
- Prefer `DOCMIND_OPENAI__BASE_URL` + `DOCMIND_OPENAI__API_KEY` for OpenAI-like servers (local servers can use placeholder keys).
- Backend-specific base URLs: `DOCMIND_LMSTUDIO_BASE_URL`, `DOCMIND_VLLM_BASE_URL`, `DOCMIND_LLAMACPP_BASE_URL`.
- Provider-neutral request controls use `DOCMIND_LLM_REQUEST__MODEL`, `DOCMIND_LLM_REQUEST__CONTEXT_WINDOW`, `DOCMIND_LLM_REQUEST__MAX_OUTPUT_TOKENS`, and `DOCMIND_LLM_REQUEST__TEMPERATURE`.

## Security policy

- When `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false` (default):
  - Loopback hosts are always allowed.
  - Non-loopback hosts must be in `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST` **and** must DNS-resolve to public IPs (private/link-local/reserved ranges are rejected; fail-closed when DNS resolution fails).
  - Use `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true` for private/internal endpoints (e.g., Docker service hostnames), or use a loopback-only architecture (`http://localhost:...`).
- Analytics page is gated by `DOCMIND_ANALYTICS_ENABLED=true` and reads `data/analytics/analytics.duckdb`.

## Containerization (CI-enforced)

- `Dockerfile`: Python 3.12.13 base; final `USER` non-root; `CMD`/`ENTRYPOINT` exec-form (no `sh -c`); no `:latest`.
- `scripts/container_health.py` owns container health; Dockerfile and Compose call it directly.
- `.dockerignore`: ignore `.env` (`.env`, `.env.*`, or `.env*`); don’t bake `.env` into images.
- Compose: use canonical `DOCMIND_*` env vars (no legacy `OLLAMA_BASE_URL`/`VLLM_BASE_URL`/`LMSTUDIO_BASE_URL`); prod override sets `read_only: true` and `tmpfs: /tmp`.

## Streamlit UI

- Keep page imports light. `src.app`, Chat, and Documents must pass
  `scripts/check_ui_import_boundary.py` without loading `torch`, `transformers`,
  `llama_index`, or `qdrant_client`. Import those implementations only inside
  their recovery, status, cached-resource, or user-action seam; use
  `TYPE_CHECKING` for
  annotations and patch canonical owners in tests instead of page aliases.
- Compute Chat snapshot status once on steady reruns. Clear/hydrate mutations
  re-read once inside `admission_quiescence()` before changing runtime owners;
  do not add a freshness TTL.
- Keep Chat model readiness local-only and typed. Missing cache, incomplete
  local overrides, and initialization failures have distinct sanitized states;
  never turn model readiness into an implicit download or relabel unexpected
  Chat DB/persistence failures.
- Treat GraphRAG package metadata as `installed`, not `ready`. Validate
  `PropertyGraphIndex` only at the Settings/action seam before claiming runtime
  capability.
- Use `st.cache_resource` for long-lived objects (DB connections, checkpointers, clients).
- Give every closeable `st.cache_resource` a typed `on_release` callback. Acquire
  its current value inside `JobManager.foreground_runtime_activity()` and retain
  that counted lease through the resource's last use; nested foreground callbacks
  are supported.
- Keep the ADR-052 background `JobManager` process-owned behind its module lock;
  do not put it in `st.cache_resource`, because cache clearing must not replace
  active job or maintenance-lease ownership.
- Keep mutable job records and progress queues manager-private. `get()` publishes
  an immutable shallow view captured under the manager lock. Publish terminal
  payload/error before terminal status, and only after the worker future is done
  during normal completion. Owner consumption and TTL expiry must clear payload
  references before removing registry and future ownership.
- Acquire `JobManager.foreground_runtime_activity()` before obtaining any live
  Chat coordinator/router handle or vector-backed export seed. Acquire
  `admission_quiescence()` before every runtime ownership mutation, recheck state
  inside the lease, and never nest a maintenance lease with a foreground lease.
- A Chat analysis fragment persists result, one terminal notice, and its completed
  marker before consumption, then requests `st.rerun(scope="app")`. Only the full
  app renders and pops that notice, after the job panel transition. Deferred
  consumption retries without republishing; successful or missing-state cleanup
  also requests a full-app rerun so controls re-enable.
- Manual GraphRAG export must revalidate the current session vector resource as
  its first action inside the foreground lease, before borrowing graph or vector
  indices, and retain that lease through the final file write.
- Treat Settings Apply as one transaction: catch any ordinary `Exception`,
  restore the exact settings and LlamaIndex globals, and let process-control
  `BaseException` subclasses propagate.

## Ingestion / processing

- Use LlamaIndex `IngestionPipeline` with `TokenTextSplitter` and optional
  spaCy enrichment. Do not add a second title-extraction path.
- Use `src/processing/parsing/` as the parser boundary. The parser always uses
  local Docling, pypdfium2, and RapidOCR artifacts. Plain text and Markdown use
  the direct text loader.
- Prefetch the app-owned Docling layout bundle before PDF parsing. Health checks
  hash its manifest files and report relative-path integrity issues. RapidOCR's
  locked wheel owns its model files and checksums; the image/test gate proves
  offline fixture inference.
- Searchable-PDF export is an optional, fail-open OCRmyPDF utility. It requires
  POSIX process groups; use WSL2 on Windows. Cancellation and timeout paths must
  kill and reap the whole subprocess group.
- Fail closed with `DocumentParseError` for every parser error. The parser service
  alone owns direct UTF-8 loading; do not add an ingestion-facade text fallback.
- Accept in-memory ingestion as `payload_text: str`; route every binary input
  through a source path and the parser boundary.
- Snapshot finalization returns `FinalizedSnapshot(path, manifest)`, whose manifest
  was verified and captured inside the writer-lock/retention boundary. Before an
  ingestion worker returns, derive its bounded object-free presentation DTO from
  that captured manifest. Do not reopen the worker's result snapshot. Terminal
  handoff must persist its notice before owner-authorized consumption; retain
  tracking when consumption races worker completion.
- Reject duplicate document IDs before ingestion I/O. Derive file document IDs
  as `doc-<full lowercase SHA-256>` through `src/utils/hashing.py`.
- Cache: DuckDB KV store (`DOCMIND_CACHE__DIR`, `DOCMIND_CACHE__FILENAME`).
- PDF page images: pypdfium2 rendering; optional AES-GCM (`DOCMIND_IMG_AES_KEY_BASE64`, `DOCMIND_IMG_DELETE_PLAINTEXT`).
- Multimodal artifacts: store page images/thumbnails as content-addressed
  `ArtifactRef` in the local ArtifactStore (`DOCMIND_ARTIFACTS__*`). Historical
  chats and snapshots can retain those references; do not add automatic
  artifact garbage collection.

## Retrieval

- Text embeddings: BGE-M3 (1024D). Sparse: FastEmbed BM42 preferred, BM25 fallback.
- Hybrid retrieval: Qdrant server-side via `Prefetch` plus `RrfQuery` (configured k) or DBSF `FusionQuery`.
- Named vectors: `text-dense`, `text-sparse` (+ IDF modifier). Dedup by `page_id` (default) or `doc_id`.
- Runtime creates a missing collection but never mutates an incompatible one.
  Use `scripts/qdrant_schema.py rebuild-empty` only after stopping writers; it
  requires an exact zero count and refuses nonempty or error states.
- Enable server-side hybrid: `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=true`.
- Multimodal search: SigLIP text→image fused by RRF; image collection uses `settings.database.qdrant_image_collection`.

## Router

- Router composes `semantic_search` plus configured `hybrid_search`,
  `keyword_search`, `multimodal_search`, and `knowledge_graph` tools.
- LlamaIndex's native `RouterQueryEngine` owns tool selection.
- Build routers via `src/retrieval/router_factory.py` to keep tool metadata/postprocessors consistent.
- Close the session router before invalidating its Chat coordinator; async
  Qdrant clients are bound to the graph loop that first uses them.
- Run retrieval and reranking CPU work through
  `src/retrieval/async_work.py`; do not use `asyncio.to_thread` for FastEmbed,
  BGE, or SigLIP inference.

## Reranking

- Text rerank: BGE cross-encoder (`BAAI/bge-reranker-v2-m3`).
- Visual rerank: SigLIP when image nodes exist.
- SigLIP Hugging Face model/processor loading is centralized in
  `src/utils/vision_siglip.py`; do not add alternate `from_pretrained` loaders,
  fallback canary flags, or duplicate revision pins.
- Fail open on timeouts; respect `settings.retrieval.*_timeout_ms`.
- Rerankers own one queue-free worker. A timed-out call retains that worker's
  capacity until native inference exits. Async router close drains it; a
  sync-only router close rejects new work without blocking on native exit.

## GraphRAG

- Use `DOCMIND_GRAPHRAG_CFG__ENABLED` as the sole ingestion default; router inclusion is owned by the presence of a healthy property graph index.
- Uses LlamaIndex core's `PropertyGraphIndex` and `SimplePropertyGraphStore`; there is no graph extra.
- Exports: JSONL baseline; Parquet optional (PyArrow). Preserve `get_rel_map` labels (fallback `related`).
- Seed policy: graph retriever → vector retriever → deterministic fallback.

## Snapshots

- Location: `data/storage/`; write via temp workspace + atomic finalize.
- `portalocker` is required. The permanent `.lock` sentinel's OS lock is the
  sole writer authority; heartbeat metadata is diagnostic only.
- Manifest triad: `manifest.jsonl`, `manifest.meta.json`, `manifest.checksum`.
- App snapshots own the activation manifest plus optional property-graph artifacts;
  Qdrant backups alone own point-in-time vector data.
- `manifest.meta.json` includes physical `collections.text` and
  `collections.image` identities, optional collection metadata, hashes,
  versions, graph type, and optional `graph_exports`.
- The canonical manifest validator owns presentation-safe metadata: every version
  key is a string and every value is a string, finite number, boolean, or null;
  every graph export has a unique basename of at most 200 characters, a nonempty
  format of at most 32 characters, a nonnegative integer size, and a lowercase
  SHA-256. Graph type `none` requires no exports. Reject invalid metadata before
  moving `CURRENT`; canonical hash preflight rejects non-finite numbers across
  the complete manifest before any artifact is written.
- `CURRENT` is the sole activation boundary. Never infer an active snapshot from
  directory ordering or promote an unreferenced snapshot during recovery.
- Result presentation must use the manifest captured by finalization. Canonical
  readiness and recovery checks may resolve and verify `CURRENT`; that validation
  is not a result-snapshot presentation reload.
- `data/.deployment-id` is the stable ownership boundary for DocMind Qdrant
  collections. Bootstrap it atomically only before durable snapshots exist. A
  missing, invalid, or replaced identity with retained snapshot state fails closed.
- Source promotion and quarantine journals recover under the snapshot writer lock
  against exact physical collection identities. The activation journal removes an
  uncommitted promoted snapshot or retires itself after `CURRENT` commits; recovery
  never chooses a snapshot by directory ordering.
- Never delete physical Qdrant generations during online activation. After stopping
  every app reader and writer, use `scripts/cleanup_collections.py` (dry-run by
  default) and review its deployment-scoped candidates before adding `--delete`.
- Don’t persist base64 blobs or raw filesystem paths; persist `ArtifactRef` and sanitize path-like metadata to basenames.
- Persist via `SnapshotManager` APIs; do not add lease-file lock fallbacks.

## SQLite and WAL

- Use native `SqliteStore` and `AsyncSqliteSaver` for LangGraph persistence; do not add parallel store or checkpoint schemas.
- Let each long-lived LangGraph primitive own its connection. Do not share raw connections between components.
- Keep the Ops DB metadata-only. Treat Chat DB checkpoints and memories as local user data.

## Backups

- Quiesce DocMind writers before backup/restore; use SQLite online backup and
  DuckDB `COPY FROM DATABASE`, never raw live database-file copies.
- A complete recovery point includes chat, ingestion cache, the verified
  `CURRENT` snapshot, authoritative uploads, `data/.deployment-id`, the existing
  `ArtifactStore`, and exact verified single-node Qdrant snapshots. Distributed
  Qdrant requires its per-node procedure. Maintenance warnings do not invalidate
  an otherwise complete recovery point.
- Retention may delete only `backup_*` directories with `manifest.complete=true`.
  Recoverability warnings produce `incomplete-backup_*`, which never evicts a
  recovery point. `maintenance_warnings` do not invalidate verified recovery data.
- Restore Qdrant into a fresh, writer-quiesced target using the exact physical
  names in `activation.collections`. Verify checksum and point count, and preserve
  the prior Qdrant instance as the rollback owner until acceptance.

## Observability / telemetry

- OTEL is optional: `DOCMIND_OBSERVABILITY__ENABLED=true` + `--extra observability`.
- Router construction emits an OpenTelemetry `router_selected` event with tool
  count and names. Local JSONL telemetry records retrieval backend/outcome,
  `export_performed`, `snapshot_stale_detected`, and rerank fallback events; it
  does not claim per-query route or graph traversal events.
- Log safety:
  - No secrets/API keys; no raw prompt/doc/model output.
  - URLs: log origin only; sanitize exception strings before logging.
- Controls: `DOCMIND_TELEMETRY_DISABLED`, `DOCMIND_TELEMETRY_SAMPLE`, `DOCMIND_TELEMETRY_ROTATE_BYTES`.

## Agent budgets

- Cap every provider request timeout to the supervisor budget. Every graph run carries an absolute deadline, and no call may exceed `settings.agents.decision_timeout`.

## Offline-first

- Predownload BGE-M3, BM42, the BGE reranker, SigLIP, and Docling layout defaults: `uv run python tools/models/pull.py --all --cache_dir ./models_cache --parser-defaults --parser-cache-dir ./cache/models`; never copy or separately prefetch RapidOCR's packaged models.
- Check parser readiness: `uv run python scripts/parser_health.py --check`.
- Use `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for offline runs; tests must remain deterministic/offline unless explicitly marked.

## Coding standards

- Prefer library primitives; keep changes small and typed.
- Prefer lazy imports for heavy ML deps (see `src/config/integrations.py`).
- Use `loguru.logger`; avoid `print` in production code.
- Use `pathlib.Path` and guard clauses; avoid `Any` unless isolated behind a narrow boundary.
- Device/VRAM policy: use `src/utils/core.py` (`resolve_device`, `has_cuda_vram`) for business logic.
- Python style (apply to all code; when editing a file, align the whole file):
  4-space indent, 88-char lines; public docstrings use `"""` + 1-line summary
  <=80 chars and include `Args:`, `Returns:`/`Yields:`, `Raises:` when needed
  with 4-space hanging indents; absolute imports only, one per line, top of
  file; None checks use `is (not) None`, booleans use `if not x` (add
  `and x is not None` only to distinguish False vs None); always use context
  managers (`with open(...)` or `contextlib.closing(...)`); prefer generator
  expressions over materialized lists when possible.

## Testing

- Boundary-first; keep unit tests <5s and integration <30s when possible.
- Use markers: `unit|integration|system|e2e|retrieval` plus
  `requires_gpu|requires_network|requires_llama`.
- AppTest: temp cwd; reuse fixtures; deterministic stubs; `default_timeout` (CI).
- Prefer patching real consumer seams, not `src.app` (see `docs/developers/testing-notes.md`).

## Docs

- Specs: `docs/specs/_index.md`
- ADRs: `docs/developers/adrs/` (do not implement anything under `docs/developers/adrs/superseded/`)
- When behavior changes, update README and the owning spec/ADR.
- Keep active docs pointing at real `src/...` paths (avoid doc drift).

## Browser automation

Use `agent-browser` for web automation. Run `agent-browser --help` for all commands.

Core workflow:

1. `agent-browser open <url>` - Navigate to page
2. `agent-browser snapshot -i` - Get elements with refs (@e1, @e2)
3. `agent-browser click @e1` / `fill @e2 "text"` - Interact using refs
4. Re-snapshot after page changes

## Source code reference

Source code for deps is available in `opensrc/` for deeper understanding of implementation details.

See `opensrc/sources.json` for the list of available packages and their versions.

Use this source code when you need to understand how a package works internally, not just its types/interface.

### Fetching additional source code

To fetch source code for a package or repository you need to understand, run:

```bash
opensrc fetch --cwd . <package>       # npm package (for example, zod)
opensrc fetch --cwd . pypi:<package>  # Python package (for example, requests)
opensrc fetch crates:<package>        # Rust crate (for example, serde)
opensrc fetch <owner>/<repo>          # GitHub repository (for example, vercel/ai)
```
