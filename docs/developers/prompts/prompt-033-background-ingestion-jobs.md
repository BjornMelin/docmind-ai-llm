# Implementation Prompt — Background Ingestion & Snapshot Jobs

Implements `ADR-052` + `SPEC-033`.

## Tooling & Skill Strategy (fresh Codex sessions)

**Use skill:** `$streamlit-master-architect`

Mandatory Streamlit evergreen steps:

```bash
uv sync
uv run python -c "import streamlit as st; print(st.__version__)"
uv run python /home/bjorn/.codex/skills/streamlit-master-architect/scripts/audit_streamlit_project.py --root . --format md
uv run python /home/bjorn/.codex/skills/streamlit-master-architect/scripts/sync_streamlit_docs.py --out /tmp/streamlit-docs
```

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

Skill references to consult (as needed):
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/caching_and_fragments.md` (fragments + reruns)
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/security.md` (threading + unsafe patterns)
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/testing_apptest.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/e2e_playwright_mcp.md` (optional E2E smoke)

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Identify current ingestion workflow and any import-time heavy work:
  - `rg -n \"ingest|rebuild_snapshot|IngestionPipeline|Snapshot\" -S src/pages/02_documents.py src/processing src/persistence`
  - `rg -n \"st\\.fragment\\(|ThreadPoolExecutor|threading\\.\" -S src`
- Read in parallel:
  - `src/pages/02_documents.py`
  - `src/processing/ingestion_pipeline.py`
  - `src/persistence/snapshot_service.py` (if WP07 already landed) or the current snapshot module used by Documents

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Streamlit fragments/threading guidance; prefer local resources before web search.

**API verification (Context7):**

- `functions.mcp__context7__resolve-library-id` → `streamlit`
- `functions.mcp__context7__query-docs` → confirm `st.fragment` and any caveats for reruns and thread safety on Streamlit `1.52.2`.

**Long-running verification (use native capabilities):**

- If you run `streamlit run src/app.py`, keep it alive and use `functions.write_stdin` to fetch logs and avoid rerunning startup.
- Attach screenshots of progress/cancel states with `functions.view_image` if verification is visual.
- Optional E2E smoke: use the skill’s Playwright script referenced above.

**Security gate (required):**

- `functions.mcp__zen__secaudit` must confirm:
  - no `st.*` calls happen inside worker threads
  - progress events do not include raw document content or secrets
  - temp workspaces cannot escape allowed directories (no traversal)

**Review gate (recommended):**

- `functions.mcp__zen__codereview` after tests pass (threading + lifecycle correctness).

**MCP tool sequence (use when it adds signal):**

1. `functions.mcp__zen__planner` → concurrency design + test plan.
2. Context7:
   - resolve `streamlit` and query docs for `st.fragment`, reruns, and threading guidance.
3. `functions.mcp__gh_grep__searchGitHub` → search for `st.fragment(run_every=` patterns in real repos.
4. `functions.mcp__zen__secaudit` → verify no unsafe thread/UI interactions; no path/secret leaks in progress events.

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
