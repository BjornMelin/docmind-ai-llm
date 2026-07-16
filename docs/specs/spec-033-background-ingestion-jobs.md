---
spec: SPEC-033
title: Background Ingestion & Snapshot Jobs (Progress + Cancellation)
version: 1.6.0
date: 2026-07-16
owners: ["ai-arch"]
status: Implemented
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

- Constants:
  - `MAX_PROGRESS_QUEUE_SIZE = 100` (default bound for progress queue)
  - `SHUTDOWN_GRACE_PERIOD_SEC = 5` (default graceful shutdown timeout)

- `JobManager` (plain class; no Streamlit APIs)
- `get_job_manager()` uses a module lock to create one manager per Python process.
  Streamlit cache clearing cannot replace the manager while work is active.
- `JobManager.activity_snapshot()` returns queued/running jobs, counted
  foreground runtime activity, and maintenance state from one manager-lock
  acquisition. UI copy MUST distinguish those states.
- `JobManager.exclusivity_activity_snapshot()` returns one exclusivity key's
  occupancy with the process activity snapshot atomically. Documents stores the
  `(mutation_active, maintenance_active)` tuple and requests one full rerun on
  either edge.
- `JobManager.foreground_runtime_activity()` owns every synchronous operation
  that uses a retireable live coordinator, router, vector client, or cached
  closeable resource. The lease begins before resource acquisition and remains
  held through its last use. It rejects while maintenance is active and always
  releases its counted, nesting-safe lease on failure.
- `JobManager.admission_quiescence()` rejects destructive runtime maintenance
  while any job is queued/running or foreground runtime operation is active.
  Once admitted, it blocks new job and foreground-runtime admission until the
  maintenance section exits.
- `JobManager.shutdown()` requests cooperative cancellation and waits up to
  `SHUTDOWN_GRACE_PERIOD_SEC` before marking remaining jobs canceled.
  Worker completion after closure cannot overwrite canceled status or publish a
  late result/error, including workers that ignore cooperative cancellation.
  - Calling `shutdown()` during normal application operation provides the bounded
    wait. CPython joins `ThreadPoolExecutor` workers before running normal user
    `atexit` callbacks, so the registered exit hook cannot guarantee that same
    bound during interpreter termination. The implementation does not use private
    executor hooks or replace the executor with a custom queue service.
- Test reset joins workers and then waits for any outstanding maintenance lease
  and foreground runtime lease before publishing a replacement process manager.
- `JobManager.consume_terminal(job_id, owner_id=...)` atomically removes only an
  owner-authorized terminal state whose worker future is done. It clears payload
  and error references and removes both registry and future ownership; queued,
  running, foreign, and not-yet-done jobs remain available for a later retry.
- Mutable job records and progress queues are manager-private. `get()` returns an
  immutable shallow `JobStateView` captured under the same lock as its status,
  result, and error.
- Normal worker completion first finishes its `Future`, then publishes result or
  error before terminal status as the release marker. Shutdown may publish a
  monotonic canceled status while a future is pending; consumption then returns
  false until that future finishes.
- TTL expiry uses the same release path as explicit consumption and clears
  manager-held result/error references before removing registry and future owners.
- Internal job tracking:
  - `job_id`, `owner_id`, `created_at`, `last_seen_at`
  - `status`: queued/running/succeeded/failed/canceled
  - `cancel_event: threading.Event`
- `progress_queue: queue.Queue[ProgressEvent]` (bounded; create with `maxsize=MAX_PROGRESS_QUEUE_SIZE`)
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
   - save uploaded files to disk before job submission (UI responsibility)
   - start job and store `job_id` in `st.session_state`

2. Render progress:

   - `@st.fragment(run_every="1s")` poller drains queue for the active job and updates a progress bar/status block
   - **Polling interval rationale**: 1s balances UX responsiveness (users see updates quickly) with minimal overhead (one rerun/sec). Configurable via `DOCMIND_UI__PROGRESS_POLL_INTERVAL_SEC` if needed.

3. Completion:
   - acquire admission quiescence before preparing or consuming a successful
     terminal result; maintenance or unrelated active jobs defer the handoff
     without consuming its durable result
   - snapshot finalization verifies and captures the committed manifest inside
     its writer-lock and retention boundary, returning the typed
     `FinalizedSnapshot(path, manifest)` result
   - before returning, the worker derives its bounded object-free presentation
     DTO directly from that captured manifest
   - terminal preparation validates the captured DTO without rereading the
     finalized snapshot, including after superseded snapshot cleanup
   - compare the job-submission runtime generation and finalized snapshot with
     the live generation and canonical `CURRENT` snapshot
   - build the replacement router while the prior runtime remains intact
   - publish the vector resource, router, generation, graph, collection, and
     snapshot identities as one rollback-capable session-state handoff
   - close the old router and vector client only after every assignment succeeds
   - store a bounded, object-free manifest and result presentation for the
     one-time full-app rerun; the rerun does not read the manifest again
   - after the notice is durable, owner-authorized consume the terminal manager
     state and completed future; a consume race retains job tracking and retries
   - preserve a stale or superseded snapshot as a truthful success while closing
     its unaccepted vector resource and leaving the current runtime unchanged
   - recompute readiness when rendering from the live generation, canonical
     `CURRENT`, snapshot identity, vector owner, router owner, and graph owner;
     stale graph exports and ready copy MUST NOT render
   - on failure, render error details (non-sensitive)

4. Chat analysis completion:
   - the polling fragment publishes the durable `AnalysisResult`, one terminal
     notice, and the completed-job marker, in that order, before owner-authorized
     manager consumption
   - after first terminal capture, successful deferred consumption, or
     missing-state cleanup, request `st.rerun(scope="app")`; rerun interrupts the
     fragment and allows sidebar controls to recompute
   - if the completed future is not yet consumable, retain job tracking and retry
     consumption without publishing the result or notice again
   - `main()` runs the job panel before the terminal-notice and result renderers;
     it is the only notice renderer and pops each outcome exactly once
   - missing manager state clears polling keys but preserves any durable result,
     notice, and completed marker

### Runtime maintenance

- Settings **Apply runtime** and **Clear caches** acquire admission quiescence
  before changing settings, runtime generations, session owners, coordinators, or
  Streamlit caches. Every closeable `st.cache_resource` registers an
  `on_release` callback, so cache maintenance closes resources only after all
  foreground readers have drained.
- The UI disables both destructive actions while any background job is queued or
  running or live foreground operation is active. **Save** remains available
  because it does not mutate the live runtime.
- Action handlers recheck the same lock-backed boundary. UI disabled state is
  guidance, not the concurrency authority.
- Router registration, vector registration, combined publication, and global
  retirement share one reentrant lifecycle lock. Canonical runtime clearing
  removes vector, router, graph, collection, and snapshot identity together.
- Settings Apply advances the generation and clears session runtime inside the
  same rollback boundary as provider binding. Any ordinary exception restores
  the previous settings, generation, exact LlamaIndex LLM, embedding model,
  context window, output count, session state, and owners; process-control
  `BaseException` subclasses propagate.
- Chat snapshot clearing/hydration and Documents stale-runtime cleanup perform a
  cheap current-state probe, then recheck and mutate only under admission
  quiescence. Lease conflicts preserve the previous runtime and render sanitized
  deferral copy.
- Chat acquires the foreground lease before obtaining the current coordinator,
  router override, session DB connection, memory store, or SigLIP image
  retriever. Session sidebar/time-travel operations, memory review operations,
  visual retrieval, history, checkpoints, query processing, post-query session
  touch, forks, and hard purges keep their current resource within the lease;
  nested callbacks use counted foreground leases. Documents manual export keeps
  its lease from live index and seed acquisition through graph serialization and
  the final JSONL or Parquet file write. Its first in-lease action revalidates the
  registered session vector resource; stale or closed owners never reach either
  index, seed lookup, or export.
- Chat analysis and all Documents mutation controls are disabled during
  maintenance, and their handlers report maintenance-admission races separately.

### Cancellation

- Cancel button calls `job_manager.cancel(job_id)`.
- Worker checks `cancel_event.is_set()` at safe boundaries and aborts with cleanup.

### Configuration

- `DOCMIND_UI__PROGRESS_POLL_INTERVAL_SEC` (default: `1`) — UI polling interval for progress updates.
- `max_progress_queue_size` (default: `MAX_PROGRESS_QUEUE_SIZE`) — `JobManager` init parameter to bound progress queue.
- `shutdown_grace_period_sec` (default: `SHUTDOWN_GRACE_PERIOD_SEC`) — `JobManager` init parameter for graceful shutdown timeout.

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

- JobManager start/complete/cancel with a stub worker fixture that emits deterministic progress.
- Cancellation edge cases:
  - cancel before job starts
  - cancel during shutdown
  - cancel after completion (no-op, status stays succeeded)
- Queue boundedness (overflow/dropping/backpressure) using a bounded queue; assert size never exceeds the cap.
- TTL cleanup for orphaned jobs clears payload references before removing job and
  future ownership.
- Terminal publication: prove immutable observer views, payload/error completeness
  at every visible terminal status, future-done handoff, and pending-future retry.
- Shutdown behavior: `JobManager.shutdown()` stops the executor cleanly (best-effort); use thread mocking to simulate racey cancellations.
- Admission fencing: any-active queries include queued and running work, active
  or foreground work rejects maintenance, and job/foreground submissions reject
  while maintenance holds the fence.
- Shutdown races: queued and running workers that ignore cancellation cannot
  replace the monotonic canceled terminal state with a late success or failure.
- Foreground races: prove both maintenance/foreground winner orders, coordinator
  reacquisition inside the lease, snapshot mutation deferral, export-seed
  exclusion, and reader-safe release of cached DB, memory, and image resources.
- Runtime handoff: inject a failure at every session assignment and prove the old
  resource and router remain live while the rejected replacements close once.
- Lifecycle serialization: use barriers to prove publication and global
  retirement cannot overlap in either winner order and each owner closes once.
- Terminal handoff: capture one valid manifest DTO in the worker, reject missing
  or malformed DTOs before ownership transfer, never reread a superseded or
  deleted snapshot, and render a bounded deterministic image preview fairly.
- Terminal races: prove both maintenance/terminal winner orders, deferral behind
  unrelated jobs, stale generation and `CURRENT` handling, and render-time
  readiness invalidation without terminal-state loss. Prove terminal consumption
  for every outcome, reference release, owner checks, and consume-race retry.
- Corpus mutation barriers: block bounded result construction after activation and
  prove a second mutation cannot enter until the first worker returns.
- Analysis ownership: prove durable result/notice publication precedes consumption,
  pending consumption retries without duplicate UI, and registry/future owners are
  released for successful, failed, and canceled outcomes.
- Fragment transitions: model app reruns as control-flow interruption and prove
  success, failure, cancellation, deferred consumption, and missing-state cleanup
  each re-enable controls without duplicate notice or result publication.

### Integration (AppTest)

- Start a job (stubbed to complete quickly) and assert progress UI renders.
- Cancel a job and assert UI state reflects cancellation.
- Mock ingestion pipeline and heavy IO using `unittest.mock.patch` so AppTest runs in seconds, not minutes.
- Use Streamlit AppTest for UI rendering; if needed, add a minimal UI smoke test via Playwright or a simple DOM snapshot.
- Use `tests/helpers/apptest_utils.py` (`apptest_timeout_sec()`) for AppTest
  timeouts, and `TEST_TIMEOUT=<seconds>` to reproduce CI slowness locally.

## Rollout / Migration

- Feature is UI-only and local-first; no data migrations.
- Rollback by disabling job manager usage in Documents page (revert files).

## Requirements Traceability

The implemented FR-025 code and test ownership map is maintained in the
canonical [requirements traceability matrix](traceability.md).

## Changelog

- 1.6.0 (2026-07-16): Moved analysis terminal display to full-app reruns with
  interrupt-accurate transition tests, preserved durable state across deferred or
  missing manager records, and rejected stale manual-export resources before index
  borrowing.
- 1.5.0 (2026-07-16): Made terminal status a payload-complete release marker,
  exposed immutable job views, centralized owner/TTL release, carried typed
  finalized manifests through bounded result construction, extended manual-export
  leases through file writes, and added exactly-once Chat analysis consumption.
- 1.4.0 (2026-07-16): Extended foreground lifecycle ownership to cached Chat
  session DB, memory-store, and SigLIP resources; added worker-captured manifest
  handoff, terminal job consumption, and complete ordinary-exception Settings
  rollback, with lifecycle race and reference-release proofs.
- 1.3.0 (2026-07-16): Added counted foreground runtime leases, coordinator
  reacquisition, lease-bound Chat and Documents runtime access, monotonic
  shutdown cancellation, atomic Documents activity-edge rerenders, and complete
  LlamaIndex Settings Apply rollback.
- 1.2.0 (2026-07-16): Added maintenance-visible activity snapshots, lease-bound
  terminal publication, generation and `CURRENT` freshness checks, truthful stale
  success, render-time readiness, one runtime lifecycle lock, canonical full
  clears, and transactional Settings Apply rollback.
- 1.1.0 (2026-07-16): Added process-owned job admission fencing, atomic runtime
  publication, pre-transfer manifest validation, bounded terminal DTOs, and
  accurate CPython exit semantics.
- 1.0.1 (2026-01-09): Implemented background ingestion and snapshot jobs.
