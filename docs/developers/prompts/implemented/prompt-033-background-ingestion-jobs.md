---
prompt: PROMPT-033
title: Background Ingestion & Snapshot Jobs
status: Completed
date: 2026-01-17
version: 1.0
related_adrs: ["ADR-052"]
related_specs: ["SPEC-033"]
---

Implements `ADR-052` + `SPEC-033`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-052-background-ingestion-jobs.md`
- SPEC: `docs/specs/spec-033-background-ingestion-jobs.md`
- Requirements: `docs/specs/requirements.md` (FR-025)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment> — `st.fragment` polling and rerun semantics.
- <https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource> — Resource caching for singletons (JobManager).
- <https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state> — Session state discipline for job ids/progress.
- <https://docs.python.org/3/library/concurrent.futures.html> — `ThreadPoolExecutor` semantics and lifecycle.
- <https://docs.python.org/3/library/threading.html#threading.Event> — Cooperative cancellation primitives.

## Tooling & Skill Strategy (fresh Codex sessions)

**Use skill:** `$streamlit-master-architect`

**Note:** Paths use `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}`. Set this env var or adjust to your local skill install location before running.

Mandatory Streamlit evergreen steps:

```bash
uv sync
uv run python -c "import streamlit as st; print(st.__version__)"
uv run python ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/scripts/audit_streamlit_project.py --root . --format md
uv run python ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/scripts/sync_streamlit_docs.py --out /tmp/streamlit-docs
```

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

Skill references to consult (as needed):

- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/caching_and_fragments.md` (fragments + reruns)
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/security.md` (threading + unsafe patterns)
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/testing_apptest.md`
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/e2e_playwright_mcp.md` (optional E2E smoke)

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Identify current ingestion workflow and any import-time heavy work:
  - `rg -n \"ingest|rebuild_snapshot|IngestionPipeline|Snapshot\" -S src/pages/02_documents.py src/processing src/persistence`
  - `rg -n \"st\\.fragment\\(|ThreadPoolExecutor|threading\\.\" -S src`
- Read in parallel:
  - `src/pages/02_documents.py`
  - `src/processing/ingestion_pipeline.py`
  - `src/persistence/snapshot.py` (SnapshotManager) and related snapshot modules used by Documents

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Streamlit fragments/threading guidance; prefer local resources before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc` (if you need to confirm ingestion pipeline behavior for cancellation boundaries)
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (rare for this package)

**API verification (Context7):**

- `functions.mcp__context7__resolve-library-id` → `streamlit`
- `functions.mcp__context7__query-docs` → confirm `st.fragment` and any caveats for reruns and thread safety on Streamlit `1.52.2`.

**Long-running verification (use native capabilities):**

- If you run `uv run streamlit run app.py`, keep it alive and use `functions.write_stdin` to fetch logs and avoid rerunning startup.
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

## Implementation Executor Template (DocMind / Python)

### YOU ARE

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes
- tests
- documentation updates (ADR/SPEC/RTM)
- deletion of dead code and removal of legacy/backcompat shims within scope

You must keep changes minimal, library-first, and maintainable.

---

### Feature Context (Filled)

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

### Hard Rules (Execution)

1. Worker threads must not call Streamlit APIs.
2. Job progress channel must be bounded (no unbounded queues).
3. Snapshot outputs must only become visible after atomic finalize.

---

### Step-by-Step Execution Plan (Filled)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

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
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
```

---

### Anti-Pattern Kill List (Immediate Deletion/Rewrite)

1. Calling `st.*` from worker threads.
2. Unbounded progress queues or accumulating UI elements on fragment reruns.
3. Publishing partial snapshots before finalize.
4. Hard-kill cancellation (terminate/kill) for filesystem work.

---

### MCP Tool Strategy (For Implementation Run)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (concurrency + tests).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → Streamlit fragment/threading docs if time-sensitive behavior is unclear.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API details (`streamlit` `st.fragment` caveats).
4. `functions.mcp__gh_grep__searchGitHub` → real-world `st.fragment(run_every=...)` patterns.
5. `functions.mcp__zen__analyze` → use if job manager interacts with multiple domain boundaries.
6. `functions.mcp__zen__codereview` → post-implementation review (thread safety + lifecycle correctness).
7. `functions.mcp__zen__secaudit` → required security audit (paths, secrets, no `st.*` in threads).

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### Final Verification Checklist (Must Complete)

| Requirement     | Status | Proof / Notes                                                                                                    |
| --------------- | ------ | ---------------------------------------------------------------------------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                                                                                  |
| **Formatting**  |        | `uv run ruff format .`                                                                                           |
| **Lint**        |        | `uv run ruff check .` clean                                                                                      |
| **Types**       |        | `uv run pyright` clean                                                                                           |
| **Tests**       |        | JobManager + UI wiring green; `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**        |        | ADR/SPEC/RTM updated                                                                                             |
| **Security**    |        | no Streamlit calls in threads; atomic finalize; no secret/content leaks                                          |
| **Tech Debt**   |        | zero work-marker placeholders introduced                                                                         |
| **Performance** |        | bounded queues; fragments don’t leak UI elements on reruns                                                       |

**EXECUTE UNTIL COMPLETE.**
