# Implementation Prompt — Background Ingestion & Snapshot Jobs

Implements `ADR-052` + `SPEC-033`.

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### YOU ARE

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes
- tests
- documentation updates (ADR/SPEC/RTM)
- deletion of dead code and removal of legacy/backcompat shims within scope

You must keep changes minimal, library-first, and maintainable.

---

### FEATURE CONTEXT (FILLED)

**Primary Task:** Implement background document ingestion + snapshot rebuild with progress reporting and best-effort cancellation in the Streamlit Documents page using stdlib threads and `st.fragment(run_every=...)` polling.

**Why now:** Synchronous ingestion blocks the UI and provides no safe cancellation. For large corpora, this feels broken and increases the risk of partial/abandoned snapshot state.

**Definition of Done (DoD):**

- A job manager exists (`src/ui/background_jobs.py`) using stdlib concurrency only (ThreadPoolExecutor + queue + Event).
- JobManager has a best-effort `shutdown()` and is registered via `atexit` from the Streamlit wrapper so dev restarts don’t leak threads.
- Documents page starts ingestion as a background job and shows progress updates via a fragment poller.
- Cancel button exists; cancellation is cooperative and cleans up any temp workspaces; no partial snapshots are published.
- Unit tests cover JobManager lifecycle, bounded queues, and cancellation.
- AppTest integration covers basic UI wiring (job start → progress UI → completion/cancel using stubs).
- RTM updated: FR-025 planned → implemented.

**In-scope modules/files (initial):**

- `src/ui/background_jobs.py` (new)
- `src/pages/02_documents.py`
- `tests/unit/ui/test_background_jobs.py` (new)
- `tests/integration/ui/test_documents_ingestion_job.py` (new or extend existing)
- `docs/developers/adrs/ADR-052-background-ingestion-jobs.md`
- `docs/specs/spec-033-background-ingestion-jobs.md`
- `docs/specs/traceability.md` (+ requirements if needed)

**Out-of-scope (explicit):**

- Multiprocessing/process termination.
- External task queues/services.
- Canceling mid-LlamaIndex pipeline call (unsafe).

---

### HARD RULES (EXECUTION)

1. Worker threads must not call Streamlit APIs.
2. Job progress channel must be bounded (no unbounded queues).
3. Snapshot outputs must only become visible after atomic finalize.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Implement `src/ui/background_jobs.py`:
   - JobManager + JobState + ProgressEvent types
   - st.cache_resource wrapper to construct singleton executor/manager
   - best-effort `shutdown()` + `atexit` registration
   - bounded progress queue and cooperative cancellation
   - TTL cleanup for orphan jobs
2. [ ] Update `src/pages/02_documents.py`:
   - start background job on ingest submit
   - `@st.fragment(run_every=\"1s\")` poller drains progress and renders status
   - cancel button calls JobManager cancel
3. [ ] Add unit tests for JobManager (no Streamlit).
4. [ ] Add/adjust AppTest integration using stubs (no heavy ingestion).
5. [ ] Update RTM and run quality gates.

Commands:

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run pylint --fail-under=9.5 src/ tests/ scripts/
uv run python scripts/run_tests.py --fast
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Calling `st.*` from worker threads.
2. Unbounded progress queues or accumulating UI elements on fragment reruns.
3. Publishing partial snapshots before finalize.
4. Hard-kill cancellation (terminate/kill) for filesystem work.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes                                  |
| ----------- | ------ | ---------------------------------------------- |
| Formatting  |        | `ruff format`                                  |
| Lint        |        | `ruff check` clean                             |
| Types       |        | `pyright` clean                                |
| Pylint      |        | meets threshold                                |
| Tests       |        | JobManager + UI wiring green                   |
| Docs        |        | ADR/SPEC/RTM updated                           |
| Security    |        | no Streamlit calls in threads; atomic finalize |

**EXECUTE UNTIL COMPLETE.**
