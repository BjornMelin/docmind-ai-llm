---
spec: SPEC-033
title: Background Ingestion & Snapshot Jobs (Progress + Cancellation)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-025: Background ingestion jobs (progress + cancel)
  - FR-010: Streamlit multipage UI
related_adrs: ["ADR-052", "ADR-051", "ADR-013", "ADR-016"]
---

## Goals

1. Make Documents ingestion non-blocking: the UI remains responsive during ingestion.
2. Provide progress reporting (phase + percent + recent message).
3. Provide best-effort cancellation without publishing partial snapshots.
4. Keep the implementation dependency-free (stdlib only) and Streamlit-aligned.

## Non-goals

- Distributed workers (Celery/RQ) or external services.
- Hard-kill cancellation (SIGKILL/terminate) that risks corrupt outputs.
- Streaming token output from the ingestion pipeline.

## User Stories

1. As a user, when I click **Ingest**, I see progress updates and can continue navigating the app.
2. As a user, I can cancel an ingestion job and the app does not leave partial snapshots.
3. When the job completes, the user sees the new snapshot ID and the Chat page can use it.

## Technical Design

### Job Manager (`src/ui/background_jobs.py`)

Implement:

- `JobManager` (plain class; no Streamlit APIs)
- `get_job_manager()` (Streamlit wrapper using `@st.cache_resource` to keep a single manager per process)
- `JobManager.shutdown()` to gracefully stop the executor (best-effort), registered via `atexit` in the Streamlit wrapper so dev restarts don't leak threads
  - **Shutdown behavior**: in-flight jobs are allowed to complete their current phase (graceful drain, ~5s timeout), then marked `canceled` if still running; no partial outputs are published
- `JobState` tracking:
  - `job_id`, `owner_id`, `created_at`, `last_seen_at`
  - `status`: queued/running/succeeded/failed/canceled
  - `cancel_event: threading.Event`
  - `progress_queue: queue.Queue[ProgressEvent]` (bounded)
  - `result` / `error`

### Progress events

Use a small typed structure:

- percent: `0..100`
- phase: `"save" | "ingest" | "index" | "snapshot" | "done"`
- message: short (no secrets, no raw doc text)
- timestamp (UTC)

### Documents page integration (`src/pages/02_documents.py`)

1. On submit:

   - validate input
   - save uploaded files to disk (or do this as the first worker phase)
   - start job and store `job_id` in `st.session_state`

2. Render progress:

   - `@st.fragment(run_every="1s")` poller drains queue for the active job and updates a progress bar/status block
   - **Polling interval rationale**: 1s balances UX responsiveness (users see updates quickly) with minimal overhead (one rerun/sec). Configurable via `DOCMIND_UI__PROGRESS_POLL_INTERVAL_SEC` if needed.

3. Completion:
   - on success, update `st.session_state` with indices (routing configuration / router engine) and snapshot metadata (see ADR-052)
   - on failure, render error details (non-sensitive)

### Cancellation

- Cancel button calls `job_manager.cancel(job_id)`.
- Worker checks `cancel_event.is_set()` at safe boundaries and aborts with cleanup.

## Observability

Emit local JSONL events (best effort):

- `ingest_job_started`: { job_id, enable_graphrag, encrypt_images }
- `ingest_job_progress`: { job_id, phase, pct } (optional sampling)
- `ingest_job_completed`: { job_id, snapshot_id, duration_ms }
- `ingest_job_canceled`: { job_id, phase }
- `ingest_job_failed`: { job_id, phase, error_code }

## Security

- Worker threads must never call Streamlit APIs.
- No new network calls.
- Never log secrets; do not include document contents in progress/events.

## Testing Strategy

### Unit

- JobManager start/complete/cancel with a stub worker that emits deterministic progress.
- Queue boundedness (no unbounded memory growth).
- TTL cleanup for orphaned jobs.
- Shutdown behavior: `JobManager.shutdown()` stops the executor cleanly (best-effort).

### Integration (AppTest)

- Start a job (stubbed to complete quickly) and assert progress UI renders.
- Cancel a job and assert UI state reflects cancellation.

## Rollout / Migration

- Feature is UI-only and local-first; no data migrations.
- Rollback by disabling job manager usage in Documents page (revert files).

## RTM Updates

Add a planned row:

- FR-025: Background ingestion jobs (progress + cancel)
  - Code: `src/ui/background_jobs.py`, `src/pages/02_documents.py`
  - Tests: `tests/unit/ui/test_background_jobs.py`, `tests/integration/ui/*`
