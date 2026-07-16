# Plan 001: Preserve job ownership and serialize corpus mutations

> **Executor instructions**: Follow this plan step by step. Run every
> verification command before continuing. If a STOP condition occurs, stop
> and report instead of improvising. Update this plan's row in
> `plans/README.md` when complete.
>
> **Drift check (run first)**:
> `git diff --stat 9accab1..HEAD -- src/ui/background_jobs.py src/pages/01_chat.py src/pages/02_documents.py src/pages/04_settings.py tests/unit/ui tests/unit/pages tests/integration/ui`
> If an in-scope symbol below changed, reconcile it before editing.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `9accab1`, 2026-07-16

## Why this matters

Documents tracks one `ingest_job_id`, but allows a second ingest, rebuild, or
delete while the first task is queued or running. The second task overwrites
the only observable job ID. Separately, the job manager is cached by runtime
`cache_version`, so Apply/Clear Caches can replace the manager and make running
ingestion or analysis unreachable. These are correctness failures around
destructive, long-running work and must precede visual redesign.

## Current state

- `src/ui/background_jobs.py:330-343` caches `JobManager` by
  `cache_version` and registers only `atexit` shutdown.
- `src/pages/02_documents.py:390-425` starts a job unconditionally, then
  overwrites `st.session_state["ingest_job_id"]`.
- `src/pages/02_documents.py:124-137` renders ingestion and maintenance
  controls without an active-job state.
- `src/pages/02_documents.py:1271-1289` uses a constant confirmation key while
  the selected deletion target can change.
- `src/pages/04_settings.py:193-200,1292-1304` changes cache generation or
  clears resources while background jobs may exist.
- `docs/specs/spec-014-index-persistence-snapshots.md:280-286` requires
  Documents controls to reflect the writer lock and prevent concurrent work.

Use `src/ui/vector_session.py` and its tests as the local pattern for explicit
resource ownership and generation-aware state. Do not weaken snapshot locks;
the UI guard is an additional invariant, not a replacement.

## Commands

| Purpose | Command | Expected |
| --- | --- | --- |
| Focused tests | `uv run pytest tests/unit/ui/test_background_jobs.py tests/unit/pages/test_documents_page_helpers.py tests/integration/ui/test_documents_ingestion_job.py -q --no-cov` | all pass |
| UI integration | `uv run pytest tests/integration/ui -q --no-cov` | all pass |
| Typecheck | `uv run pyright --threads 4` | exit 0 |
| Lint | `uv run ruff format --check . && uv run ruff check .` | exit 0 |
| Full tests | `uv run pytest tests/unit tests/integration -q --no-cov` | all pass |

## Scope

**In scope**:

- `src/ui/background_jobs.py`
- `src/pages/01_chat.py`
- `src/pages/02_documents.py`
- `src/pages/04_settings.py` only if an active-job warning/disable is needed
- corresponding unit and AppTest files under `tests/unit` and
  `tests/integration/ui`
- governing UI/spec docs only when behavior wording changes

**Out of scope**:

- snapshot writer, activation, deployment-identity, parser, or Qdrant contracts
- replacing the thread pool with a queue service
- changing cache invalidation for model/router/vector resources
- security or deployment configuration

## Git workflow

Use the existing `feat/ui-foundation` branch/worktree. Commit convention is
Conventional Commits, for example `fix(ui): serialize corpus mutations`.
Do not push or open a PR until the parent orchestrator reviews the diff.

## Steps

### 1. Make the job manager process-stable

Remove `cache_version` from `get_job_manager` and every caller. Prefer a
single process-lifetime standard-library cache (for example a one-entry
`functools.cache`) plus the existing idempotent `shutdown` at process exit, so
Streamlit cache clearing cannot replace the manager. Add a deterministic test
that repeated retrieval returns the same manager across settings generation
changes. Provide a test-only reset without exposing a production reset button.

**Verify**: focused background-job tests pass.

### 2. Enforce one process-wide corpus mutation slot

Extend `JobManager.start_job` with one optional typed exclusivity key. Under the
manager lock, reject a new job when any queued/running job already owns the same
key, regardless of owner/session; terminal jobs do not retain the slot. Use a
dedicated exception such as `JobConflictError` so the UI can show a concise
"another corpus change is already running" state. Ingestion, rebuild, and
delete all submit with the exact key `corpus-mutation`; Chat analysis does not.
Do not serialize unrelated jobs.

Add one helper that resolves the tracked ingestion job for the current owner
and classifies only `queued`/`running` as active. Stale/missing/terminal IDs
must be cleared or rendered terminally; they must never disable controls
forever. Compute the state before rendering the ingest form and maintenance
controls, and pass it explicitly rather than rereading scattered session keys.

Disable Ingest, Rebuild, and Delete while active and show concise progress-
aware copy. Recheck inside `_start_ingestion_job` immediately before
submission so direct calls and same-rerun double actions cannot bypass the UI.
Do not block the independent read-only export controls.

**Verify**: add manager tests proving two different owners cannot hold the same
`corpus-mutation` slot while queued/running and can submit after the first job is
terminal. Add an AppTest that attempts a second mutation and proves the second
worker never starts.

### 3. Bind destructive confirmation to the selected file

Use native Streamlit state so confirmation for file A cannot authorize file B.
A selection-specific widget key is acceptable; alternatively clear the
confirmation in the selectbox `on_change` callback. Put the exact selected
filename in the confirmation sentence and destructive action label. Preserve
the existing generation-safe quarantine/rebuild implementation.

**Verify**: an AppTest confirms A, switches to B, and verifies B remains
disabled until separately confirmed.

### 4. Preserve ownership through Settings actions

With the manager independent from model cache versions, Apply and Clear Caches
must not drop job observation. If Clear Caches still tears down the manager in
the live implementation, disable that action while any current-session job is
active and show why. Do not silently cancel work.

**Verify**: start a fake ingestion/analysis job, bump cache version/clear the
model caches, and prove the same job remains queryable and cancelable.

## Test plan

- Reuse `tests/unit/pages/test_documents_page_helpers.py:195-235` for start-job
  fixtures and `tests/integration/ui/test_documents_ingestion_job.py` for
  AppTest behavior.
- Cover same-owner and different-owner conflicts, active queued/running,
  terminal release, missing/stale IDs, second submit, rebuild/delete
  disablement, selection-change confirmation, and runtime cache changes.
- Preserve the full existing unit/integration suite.

## Done criteria

- [ ] No production caller passes `settings.cache_version` to
  `get_job_manager`.
- [ ] One process manager retains a job across runtime cache version changes.
- [ ] No owner/session can start a corpus mutation while another process job
  owns the `corpus-mutation` slot.
- [ ] Confirmation for one filename never enables deletion of another.
- [ ] Focused, UI integration, Ruff, Pyright, and full tests are green.
- [ ] No out-of-scope persistence or parser files changed.

## STOP conditions

Stop if process-stable ownership requires disabling snapshot locks, sharing a
job across different owner IDs, persisting executable callables, or changing
the worker transaction contract. Stop if Streamlit cache clearing cannot be
separated without a broader lifecycle design; report the exact API behavior.

## Maintenance notes

Reviewers should scrutinize job-state races and terminal cleanup. Future
background task types must declare whether they are mutually exclusive with a
corpus mutation; do not convert the global executor into a one-job-only queue.
