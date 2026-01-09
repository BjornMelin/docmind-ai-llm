---
ADR: 052
Title: Background Ingestion and Snapshot Jobs in Streamlit (Threads + Fragments)
Status: Proposed
Version: 1.0
Date: 2026-01-09
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

| Option | Complexity (40%) | Perf (30%) | Alignment (30%) | Total |
|---|---:|---:|---:|---:|
| **A: Threads + `st.fragment(run_every=...)` polling** | 9.0 | 9.0 | 10.0 | **9.3** |
| B: Processes + terminate | 5.5 | 8.5 | 6.5 | 6.7 |
| C: synchronous | 10.0 | 3.0 | 7.0 | 6.7 |

## Decision

We will implement Option A:

1) Add a small job manager using stdlib `ThreadPoolExecutor`, cached via `st.cache_resource`, that owns:

- job registry: `{job_id -> JobState}`
- `threading.Event` cancellation token per job
- bounded `queue.Queue(maxsize=N)` for progress messages per job
- job TTL and orphan cleanup

2) Update the Documents page to:

- start ingestion/snapshot jobs on submit
- render progress and status via an `@st.fragment(run_every="1s")` poller
- allow cancel with cooperative semantics (`cancel_event.set()`)

3) Enforce atomic publishing:

- snapshot writes are staged under `_tmp-*` workspace and only published via atomic rename (`SnapshotManager.finalize_snapshot`)
- job completion only updates UI/session_state after snapshot is finalized

## High-Level Architecture

```mermaid
flowchart TD
  UI[Documents Page] --> JM[JobManager (cached resource)]
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

## Changelog

- 1.0 (2026-01-09): Proposed thread+fragment job architecture for v1 UX.

