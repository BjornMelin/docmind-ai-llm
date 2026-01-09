# Implementation Prompt — Document Analysis Modes (Separate / Combined / Auto)

Implements `ADR-023` + `SPEC-036`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-023-analysis-mode-strategy.md`
- SPEC: `docs/specs/spec-036-document-analysis-modes.md`
- RTM: `docs/specs/traceability.md`
- Requirements: `docs/specs/requirements.md`

## Official docs (research during implementation)

- <https://docs.streamlit.io/develop/api-reference/layout/st.tabs> — Per-document result presentation in Separate mode.
- <https://docs.streamlit.io/develop/api-reference/status/st.status> — Status container for long-running analysis.
- <https://docs.streamlit.io/develop/concepts/design/multithreading> — Streamlit threading guidance (no Streamlit calls from worker threads).
- <https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest> — AppTest for deterministic UI integration tests.
- <https://langchain-ai.github.io/langgraph/concepts/> — LangGraph concepts (when wiring analysis through the coordinator).
- <https://docs.llamaindex.ai/en/stable/module_guides/querying/query_pipeline/> — QueryPipeline patterns (only if needed).

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Use skill:** `$streamlit-master-architect`

Load and follow its workflows for:

- rerun discipline + `st.session_state` correctness
- long-running work patterns (containers + status; no Streamlit calls in threads)
- AppTest patterns
- security-by-default

Skill references to consult (as needed):

- `/home/bjorn/.codex/skills/streamlit-master-architect/references/architecture_state.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/caching_and_fragments.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/testing_apptest.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/widget_keys_and_reruns.md`

**Streamlit preflight (version + docs + audit):**

```bash
uv sync
uv run python -c "import streamlit as st; print(st.__version__)"
uv run python /home/bjorn/.codex/skills/streamlit-master-architect/scripts/audit_streamlit_project.py --root . --format md
uv run python /home/bjorn/.codex/skills/streamlit-master-architect/scripts/sync_streamlit_docs.py --out /tmp/streamlit-docs
```

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel` for independent work):**

- Locate current query/answer flow:
  - `rg -n "MultiAgentCoordinator|run\\(|answer\\(|router_engine" src/pages/01_chat.py src/agents src/retrieval`
- Locate document identity metadata fields used by retrieval:
  - `rg -n "\\bdoc_id\\b|\\bpage_id\\b|metadata\\[\\\"doc_id\\\"\\]" src/retrieval src/processing src/agents`
- Confirm Streamlit threading constraints (no `st.*` in threads):
  - `rg -n "ThreadPoolExecutor|threading\\.|concurrent\\.futures" src/pages src/ui src/agents`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Streamlit/LlamaIndex/LangGraph resources.
- `functions.read_mcp_resource` → read relevant local docs/indexes before web search.

**API verification (Context7, only when uncertain):**

- `functions.mcp__context7__resolve-library-id` → `streamlit`, `langgraph`, `llama-index`
- `functions.mcp__context7__query-docs` → verify signatures/behavior for:
  - `st.tabs`, `st.status`, AppTest usage
  - any LangGraph/LlamaIndex APIs you plan to call directly

**Time-sensitive facts (use web tools):**

- Prefer `functions.mcp__exa__web_search_exa` for discovery; use `web.run` if you need citations or dates.

---

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

### FEATURE CONTEXT
**Primary Task:** Implement Document Analysis Modes (`auto | separate | combined`) with a domain service and Streamlit UI wiring, enabling per-document analysis in parallel with a safe UI presentation (tabs) and optional reduce/synthesis.

**Why now:** Analysis-mode configuration exists but has no concrete implementation. This is a core “document analysis” capability expected in a finished release and must be testable and offline-first.

**Definition of Done (DoD):**
- `src/analysis/*` provides a typed, testable service for mode routing and execution.
- Chat UI exposes analysis mode selection and renders:
  - Combined: single result
  - Separate: one tab per document + optional reduce summary
- Separate mode runs bounded parallelism (`settings.analysis.max_workers`) and never calls Streamlit APIs from worker threads.
- Telemetry events defined in `SPEC-036` are emitted (JSONL; OTel when enabled).
- Tests added (unit + AppTest integration) and all quality gates pass.
- `docs/specs/traceability.md` updated with `FR-028` mapping to code/tests.

**In-scope modules/files (initial):**
- `src/analysis/service.py` (new)
- `src/analysis/models.py` (new)
- `src/pages/01_chat.py`
- `src/config/settings.py` (only if needed to tighten `AnalysisConfig` semantics)
- `tests/unit/analysis/test_analysis_service.py` (new)
- `tests/integration/ui/test_analysis_modes.py` (new)
- `docs/specs/spec-036-document-analysis-modes.md`
- `docs/specs/requirements.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**
- Introducing a new retrieval backend or index type.
- Calling Streamlit APIs from background threads via `add_script_run_ctx()` (unsupported).

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging
- Python version must remain **3.11.x** (respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`

#### 2) Style, Types, and Lint
Your code must pass:
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`
- `uv run pylint --fail-under=9.5 src/ tests/ scripts/`

Rules:
- Prefer typed dataclasses / Pydantic models over untyped dicts.
- Avoid `Any` unless you can justify it (and isolate it behind a narrow boundary).
- No silent exception swallowing. Catch specific exceptions and log meaningfully.

#### 3) Streamlit UI Discipline
- `src/app.py` stays a thin shell (no business logic).
- Pages in `src/pages/*` should:
  - keep UI concerns local
  - call domain-layer services/helpers (do not rebuild pipelines in UI)
  - use `st.session_state` intentionally and avoid hidden global state
- Avoid expensive work at import time; Streamlit reruns frequently.
- Never call `st.*` from worker threads; collect results and update UI in the script thread.

#### 4) Config Discipline (Pydantic Settings v2)
- Configuration source of truth is `src/config/settings.py`.
- Do not scatter `os.getenv` in domain code.

#### 5) LlamaIndex + LangGraph Alignment
- Prefer existing coordinator/retrieval wiring; do not re-implement agent logic.
- Preserve offline-first operation:
  - do not add implicit network calls
  - gate network/exporters behind config flags

---

### STEP-BY-STEP EXECUTION PLAN
You MUST produce a plan and keep exactly one step “in_progress” at a time.

1. [ ] Read ADR/SPEC/RTM/Requirements and map current chat flow + doc metadata.
   - Commands:
     - `rg -n "MultiAgentCoordinator|router_engine" src/pages/01_chat.py src/agents -S`
2. [ ] Implement `src/analysis/models.py` and `src/analysis/service.py` with mode routing + bounded parallel execution.
   - Commands:
     - `uv run ruff format src/analysis/`
     - `uv run ruff check src/analysis/`
3. [ ] Wire Chat UI: mode selector + rendering (tabs/status) + cancellation UX.
   - Commands:
     - `uv run ruff check src/pages/01_chat.py`
4. [ ] Add telemetry events per SPEC (JSONL + optional OTel spans).
   - Commands:
     - `uv run ruff check src/utils src/telemetry src/pages/01_chat.py`
5. [ ] Add tests (unit + AppTest integration).
   - Commands:
     - `uv run python -m pytest tests/unit/analysis/test_analysis_service.py -q`
     - `uv run python -m pytest tests/integration/ui/test_analysis_modes.py -q`
6. [ ] Update docs + RTM (add FR-028).
   - Commands:
     - `rg -n "FR-028" docs/specs -S`
7. [ ] Run full quality gates.
   - Commands:
     - `uv run ruff format .`
     - `uv run ruff check .`
     - `uv run pyright`
     - `uv run pylint --fail-under=9.5 src/ tests/ scripts/`
     - `uv run python scripts/run_tests.py --fast`

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

Scan the feature scope and delete or refactor immediately if found:

1. **God Modules:** single file > 400 LOC without clear layering.
2. **Import-time side effects:** heavy IO/model loads at import (breaks Streamlit reruns/tests).
3. **Config sprawl:** repeated `os.getenv` usage outside settings module.
4. **Swallowed errors:** broad `except Exception` without re-raise or explicit handling.
5. **Async misuse:** blocking calls inside async paths; mixed event loops without control.
6. **Unbounded caches:** files growing without rotation/limits; missing TTL/invalidation.
7. **Security footguns:** path traversal, unsafe temp files, remote endpoints ungated.
8. **Dead code:** unused exports, unreferenced entrypoints, obsolete compatibility layers.
9. **Undocumented behavior:** feature ships without SPEC/ADR/RTM updates.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes |
|---|---|---|
| **Packaging** |  | `uv sync` clean |
| **Formatting** |  | `uv run ruff format .` |
| **Lint** |  | `uv run ruff check .` |
| **Types** |  | `uv run pyright` |
| **Pylint** |  | meets threshold |
| **Tests** |  | `uv run python scripts/run_tests.py --fast` |
| **Docs** |  | ADR/SPEC/RTM updated |
| **Security** |  | no unsafe HTML; no new egress |
| **Tech Debt** |  | zero TODO/FIXME introduced |
| **Performance** |  | no new import-time heavy work; parallelism bounded |

**EXECUTE UNTIL COMPLETE.**

