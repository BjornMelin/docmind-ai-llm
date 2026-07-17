# Plan 003: Remove duplicate and eager page-startup work

> **Executor instructions**: Execute after Plans 001 and 002. Keep every safety and
> retrieval contract intact. Update `plans/README.md` when done.
>
> **Drift check**:
> `git diff --stat 9accab1..HEAD -- src/app.py src/pages/01_chat.py src/pages/02_documents.py src/ui src/retrieval scripts/check_ui_import_boundary.py tests/unit/ui tests/integration/ui`
> Plan 001 is expected to change `get_job_manager` call signatures in Chat/
> Documents and job tests. Plan 002 is expected to replace `_load_chat_messages`
> return handling and remove `_chunked_stream` plus its direct tests. Reconcile
> only those named changes. Any other drift in startup/import/snapshot symbols
> is a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: Plans 001 and 002
- **Category**: perf
- **Planned at**: commit `9accab1`, 2026-07-16

## Why this matters

Fresh-process audit measurements were 3.83 seconds/446 MiB to import Chat and
2.55 seconds/434 MiB to import Documents before rendering. Chat also discovers
and hashes snapshot state twice per rerun. The accepted ADR target is under two
seconds and favors native, minimal page shells. Removing duplicate work and
lazy-loading action-only ML dependencies improves every interaction while
lowering redesign risk.

## Current state

- `src/pages/01_chat.py:602-671` loads snapshot state in
  `_ensure_router_engine` and independently recomputes staleness in
  `_render_staleness_badge`.
- `src/pages/01_chat.py:21-87` eagerly imports coordinator, LangGraph,
  retrieval, memory, analysis, and artifact modules.
- `src/pages/02_documents.py:16-64` eagerly imports snapshot, GraphRAG,
  telemetry, router, and ingestion infrastructure.
- `src/app.py:16-24` imports snapshot recovery while defining the supported app
  shell, so `import src.app` currently loads LlamaIndex before navigation.
- `src/ui/ingest_adapter.py:17+` imports LlamaIndex infrastructure at module
  load.
- `docs/developers/adrs/ADR-013-user-interface-architecture.md:95-104`
  specifies page and interaction budgets.

## Commands

| Purpose | Command | Expected |
| --- | --- | --- |
| Import proof | `uv run python scripts/check_ui_import_boundary.py` | exit 0; no prohibited roots loaded |
| Focused | `uv run pytest tests/unit/ui tests/unit/pages tests/integration/ui -q --no-cov` | all pass |
| Quality | `uv run ruff format --check . && uv run ruff check . && uv run pyright --threads 4` | exit 0 |
| Full | `uv run pytest tests/unit tests/integration -q --no-cov` | all pass |

Create `scripts/check_ui_import_boundary.py`. In a fresh subprocess per target,
import `src.app`, `src.pages.01_chat`, and `src.pages.02_documents`, then fail if
any module root in the fixed set `torch`, `transformers`, `llama_index`, or
`qdrant_client` is loaded. The script must print the target and sorted violating
roots, and exit 1 on violation. Do not weaken the prohibited set after
implementation starts; an unavoidable chain is a STOP condition with
`python -X importtime` evidence.

## Scope

**In scope**: `src.app` recovery import placement, Chat/Documents page imports
and snapshot-status computation, `scripts/check_ui_import_boundary.py`, small
typed helpers under `src/ui`, and direct tests.

**Out of scope**: model replacement, retrieval algorithm changes, snapshot
format changes, global caches with arbitrary TTLs, settings-page redesign, or
new dependencies.

## Git workflow

Use `feat/ui-foundation`. Commit convention: `perf(ui): defer heavy page
imports`. Do not push or open a PR until parent review.

## Steps

### 1. Compute snapshot status once per rerun

Create one typed per-rerun snapshot-status value containing the exact snapshot
identity, manifest presence, staleness, and sanitized error state needed by
autoload and badge rendering. Compute it once in Chat `main` and pass it to
both consumers. Do not time-cache filesystem state; changes must be visible on
the next rerun.

**Verify**: a unit spy proves corpus/config staleness computation occurs once
per main rerun and stale/up-to-date/missing/error UI behavior is unchanged.

### 2. Keep app recovery fail-closed behind a lightweight import boundary

Move the `recover_snapshot_transactions` import inside
`_recover_persistence_once` (or an equally narrow call boundary) so importing
`src.app` does not initialize snapshot/LlamaIndex modules. Preserve the exact
try/error/`st.stop()` behavior in `main`; recovery still runs before navigation.

**Verify**: `src.app` passes the import-boundary script, and AppTest proves a
recovery exception still renders the sanitized error and stops navigation.

### 3. Move heavy imports behind action/resource boundaries

Use `TYPE_CHECKING` for types and import coordinator/router/ingestion/model
implementations inside cached constructors or the action functions that need
them. Keep lightweight dataclasses and protocols at module scope. Update tests
that monkeypatch page-module symbols to patch the canonical owner or an
explicit injection seam; do not add compatibility aliases solely for tests.

**Verify**: fresh-process import tracing no longer loads action-only Torch,
Transformers, and LlamaIndex paths before the user requests them, subject to
the documented boundary above.

### 4. Render an honest Chat state without embedding artifacts

`main` must not unconditionally construct `_get_coordinator()` and
`_get_memory_store()` before it can render useful first-run guidance. Keep
lightweight title, provider status, navigation/session DB, and local snapshot
status renderable first. Move embedding/coordinator construction to the first
capability that requires it and represent expected model-artifact unavailability
with a typed/sanitized result in `src/ui/chat_runtime.py`. Render an explicit
native unavailable/degraded state with the documented model setup action and
disable only coordinator/memory/visual capabilities that cannot work. Do not
catch corrupt chat DB, deployment identity, or other fail-closed persistence
errors as “model unavailable.” Do not download models implicitly.

**Verify**: with an isolated empty data/cache directory, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`, a real Chat page render completes without an uncaught
exception, displays the unavailable guidance, makes no external request, and
does not construct embedding-dependent resources. AppTest separately proves an
unexpected/corrupt persistence failure still fails closed.

### 5. Record reproducible before/after evidence

Use these exact evidence commands before and after:

```bash
/usr/bin/time -f 'elapsed=%e maxrss_kb=%M' uv run python -c "import src.app"
/usr/bin/time -f 'elapsed=%e maxrss_kb=%M' uv run python -c "import importlib; importlib.import_module('src.pages.01_chat')"
/usr/bin/time -f 'elapsed=%e maxrss_kb=%M' uv run python -c "import importlib; importlib.import_module('src.pages.02_documents')"
uv run python -X importtime -c "import src.app" 2>/tmp/docmind-app-importtime.log
uv run python -X importtime -c "import importlib; importlib.import_module('src.pages.01_chat')" 2>/tmp/docmind-chat-importtime.log
uv run python -X importtime -c "import importlib; importlib.import_module('src.pages.02_documents')" 2>/tmp/docmind-documents-importtime.log
```

The machine-checkable gate is `scripts/check_ui_import_boundary.py`; timing and
RSS are comparative evidence because shared-runner timing is not stable. Keep
the logs under `/tmp` and record only totals and environment in the PR.

## Test plan

Cover snapshot missing, current, stale, malformed/error, autoload policy, and
module import boundaries. Preserve all AppTest flows and full tests.

## Done criteria

- [ ] Snapshot freshness is computed once per Chat rerun.
- [ ] Action-only heavy ML/retrieval modules are not imported by page startup.
- [ ] Empty offline caches render a sanitized degraded Chat state without an
  implicit model download or weakened persistence failure.
- [ ] No TTL can hide local file/config changes across reruns.
- [ ] `scripts/check_ui_import_boundary.py` passes with the fixed prohibited
  roots, and before/after evidence is recorded with environment details.
- [ ] Focused, Ruff, Pyright, and full tests pass.

## STOP conditions

Stop if lazy imports create circular dependencies, change cache ownership, or
require compatibility shims. Stop if a proposed cache can conceal local file
changes or if import reduction requires changing model/retrieval behavior.

## Maintenance notes

Keep UI modules import-light. Future page features should import heavyweight
backends at the resource/action seam, not at page module scope.
