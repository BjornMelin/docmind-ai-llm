---
ADR: 052
Title: Background Ingestion and Snapshot Jobs in Streamlit (Threads + Fragments)
Status: Implemented
Version: 1.6
Date: 2026-07-16
Supersedes:
Superseded-by:
Related: 013, 016, 031, 051
Tags: streamlit, ingestion, concurrency, snapshots, ux
References:
  - https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment
  - https://docs.streamlit.io/develop/concepts/app-design/multithreading
---

## Description

Implement non-blocking document ingestion + snapshot rebuild with progress reporting and best-effort cancellation in the Streamlit Documents page.

## Context

Ingestion and snapshot rebuild can take significant time (PDF parsing/OCR, embedding, graph building, snapshot persistence). Today, `src/pages/02_documents.py` runs the work synchronously inside the Streamlit script:

- the UI blocks during ingestion
- users cannot cancel safely
- reruns can re-trigger work unless carefully guarded

Streamlit’s execution model requires that UI commands be executed only on the main script thread; background work must not call Streamlit APIs directly.

## Decision Drivers

- Improve UX for large corpora (responsive UI, visible progress)
- Preserve offline-first posture (no external workers, no network dependencies)
- Use only stdlib concurrency (no new dependencies)
- Ensure snapshot outputs are atomic and never partially published
- Make behavior testable (unit tests + AppTest)

## Alternatives

- A: Thread-based job manager + fragment polling + cooperative cancel (Selected)
- B: Process-based jobs + filesystem polling + hard termination (high complexity, platform issues)
- C: Keep synchronous ingestion with `st.status` only (no background/cancel)

### Architecture Tier‑2 Decision (≥9.0 rule)

Weights: Complexity 40% · Performance 30% · Alignment 30% (10 = best)

| Option | Complexity (40%) | Perf (30%) | Alignment (30%) | Total | Decision |
| --- | --- | --- | --- | --- | --- |
| A: Threads + `st.fragment(run_every=...)` polling | 9.0 | 9.0 | 10.0 | **9.3** | Selected |
| B: Processes + terminate | 5.5 | 8.5 | 6.5 | 6.7 | Rejected |
| C: synchronous | 10.0 | 3.0 | 7.0 | 6.7 | Rejected |

## Decision

We will implement Option A:

1. Add a small process-owned job manager using stdlib `ThreadPoolExecutor` and a
   module lock. The manager owns:

   - private job registry: `{job_id -> _JobState}` with immutable `JobStateView`
   - `threading.Event` cancellation token per job
   - bounded `queue.Queue(maxsize=N)` for progress messages per job
   - job TTL and orphan cleanup
   - private mutable records with immutable observer views

   Normal completion publishes result/error only after the worker future is done,
   then sets terminal status as the release marker. Explicit consumption and TTL
   expiry share one payload-clearing registry/future release path.

2. Update the Documents page to:

   - start ingestion/snapshot jobs on submit
   - render progress and status via an `@st.fragment(run_every="1s")` poller
   - allow cancel with cooperative semantics (`cancel_event.set()`)

3. Enforce atomic publishing:

   - snapshot writes are staged under `_tmp-*` workspace and only published via atomic rename (`SnapshotManager.finalize_snapshot`)
   - snapshot finalization verifies and captures the manifest under its writer
     lock, returns `FinalizedSnapshot(path, manifest)`, and the worker builds its
     bounded object-free DTO from that result without reopening the snapshot
   - the replacement router is built before any session owner changes
   - vector, router, generation, graph, collection, and snapshot state publish as
     one rollback-capable handoff; old owners close only after publication succeeds
   - the terminal rerun receives bounded object-free presentation and manifest
     data and does not read the manifest again, even if superseded retention has
     already removed the finalized snapshot
   - after its terminal notice is durable, owner-authorized consumption clears
     result/error references and removes the completed state and future; a race
     with worker completion retains UI tracking and retries
   - successful terminal preparation owns the maintenance lease through runtime
     publication and terminal-state consumption
   - publication requires the captured runtime generation and finalized snapshot
     to remain current; superseded work is still reported as durable success but
     never replaces the live runtime
   - terminal rendering recomputes readiness from live ownership and suppresses
     stale ready copy and graph exports

4. Fence live runtime maintenance:

   - Settings Apply and cache clearing reject while any process job is queued or
     running or any counted foreground runtime operation is active
   - once admitted, maintenance pauses all job submissions until its destructive
     section exits
   - foreground admission rejects during maintenance; Chat acquires this lease
     before the current coordinator, router override, cached session DB, memory
     store, or image retriever, and Documents uses it around live export-seed
     acquisition
   - foreground leases are counted and nesting-safe, and remain held from current
     resource acquisition through last use, including nested Chat callbacks
   - every cached closeable resource registers an `on_release` callback; cache
     maintenance closes it only after foreground readers drain
   - UI controls reflect the same state, but lock-backed action checks remain the
     concurrency authority
   - activity snapshots expose jobs and maintenance separately so Chat,
     Documents, and Settings can disable destructive actions with accurate copy
   - runtime registration, combined publication, and retirement share one
     reentrant lifecycle lock; one canonical clear removes all runtime identity
   - Settings Apply includes its generation advance and runtime clear in the
     provider-binding rollback transaction; every ordinary exception restores
     all captured settings and LlamaIndex globals
   - every Chat snapshot mutation and Documents stale-runtime clear performs a
     cheap probe, then rechecks and mutates under admission quiescence
   - shutdown cancellation is monotonic: late worker completion cannot publish a
     result or error after manager closure
   - manual GraphRAG export retains its foreground lease through serialization and
     the final JSONL or Parquet file write, and revalidates the registered session
     vector resource before borrowing either live index
   - Chat analysis stores its durable result and one-time outcome notice before
     its completed marker and manager consumption; first capture, successful retry,
     and missing-state cleanup interrupt the fragment with a full-app rerun
   - the full app evaluates the analysis panel before rendering its terminal notice
     and results; the fragment never renders or pops terminal UI

## High-Level Architecture

```mermaid
flowchart TD
  UI[Documents Page] --> JM[Process JobManager]
  CHAT[Chat live coordinator access] --> FG[Foreground runtime lease]
  FG --> JM
  SET[Settings Apply / Clear caches] --> FENCE[Admission quiescence]
  FENCE --> JM
  FENCE --> RUNTIME[Live runtime mutation]
  JM -->|submit| TH[Worker thread]
  TH -->|progress| Q[Queue]
  UI -->|poll| FRAG[@st.fragment run_every]
  FRAG -->|drain| Q
  TH --> PIPE[Ingestion pipeline]
  TH --> SNAP[snapshot_service]
  SNAP --> FS[(data/storage)]
```

## Cancellation Semantics

- Cancellation is **best-effort**:
  - checked at safe boundaries (after file save, before/after ingestion pipeline run, before snapshot finalize)
  - cannot reliably interrupt a running LlamaIndex pipeline mid-call without new dependencies or unsafe termination
- On cancel:
  - job marks state as canceled
  - snapshot workspace is cleaned up
  - UI does not publish partial results
- A direct production call to `JobManager.shutdown()` waits for the configured
  grace period. At interpreter exit, CPython joins `ThreadPoolExecutor` workers
  before normal user `atexit` callbacks. The registered hook therefore requests
  cleanup but cannot guarantee the configured bound for a worker that ignores
  cooperative cancellation. We do not depend on private executor hooks.

## Security & Privacy

- Workers never call Streamlit APIs.
- No new network surface; all work is local.
- Progress messages and logs must not include secrets or raw document contents.

## Testing

- Unit tests for the job manager and cooperative cancellation behavior (no Streamlit needed).
- AppTest integration:
  - start job
  - observe progress UI rendering (using deterministic stub worker)
  - cancel job and confirm cleanup state

## Consequences

### Positive Outcomes

- Responsive UX for long-running ingestion/snapshot operations.
- Safer operation: cancel without corrupting snapshots.
- Foundation for future background tasks (exports, backups) without changing architecture.

### Trade-offs

- More moving parts (job registry, polling fragment).
- Cancellation is cooperative, not immediate hard-kill.
- Interpreter exit cannot impose the application shutdown grace period on an
  executor worker that does not return.

## Changelog

- 1.6 (2026-07-16): Made Chat terminal completion a full-app lifecycle transition
  with one notice owner, added stale-resource rejection before manual export index
  borrowing, and proved deferred/missing-state control re-enablement.
- 1.5 (2026-07-16): Made terminal status payload-complete and future-done,
  exposed immutable observer views, centralized manager release, carried typed
  finalized manifests without result reloads, extended export leases through file
  writes, and made Chat analysis handoff exactly once across consume retries.
- 1.4 (2026-07-16): Extended counted foreground leases and deterministic cache
  release to Chat session DB, memory-store, and SigLIP reader lifecycles; added
  worker-captured manifests, owner-authorized terminal consumption, and complete
  ordinary-exception Settings rollback with race and reference-release proofs.
- 1.3 (2026-07-16): Added counted foreground leases, current-coordinator
  reacquisition, lease-bound runtime readers/writers, monotonic shutdown
  cancellation, atomic Documents activity edges, and complete Settings rollback.
- 1.2 (2026-07-16): Added lease-bound terminal publication, stale-success and
  render-readiness semantics, maintenance-visible activity, a shared runtime
  lifecycle lock, canonical full clears, and transactional Settings Apply.
- 1.1 (2026-07-16): Added process-owned admission fencing, atomic runtime
  publication, manifest-first terminal handoff, and accurate CPython executor
  exit semantics.
- 1.0 (2026-01-09): Proposed thread and fragment job architecture for v1 UX.
