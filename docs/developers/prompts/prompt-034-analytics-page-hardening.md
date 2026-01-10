# Implementation Prompt — Analytics Page Hardening

Implements `ADR-053` + `SPEC-034`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-053-analytics-page-hardening.md`
- SPEC: `docs/specs/spec-034-analytics-page-hardening.md`
- Requirements: `docs/specs/requirements.md` (FR-010, NFR-OBS-001)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://duckdb.org/docs/api/python/overview.html> — DuckDB Python API overview (connections, results, `.df()` patterns).
- <https://duckdb.org/docs/guides/python/import_pandas.html> — DuckDB ↔ pandas integration patterns.
- <https://docs.streamlit.io/develop/api-reference/charts> — Streamlit charting APIs (rendering dataframes/charts).
- <https://docs.python.org/3/library/json.html> — JSON parsing primitives (streaming JSONL parsing strategy).

## Tooling & Skill Strategy (fresh Codex sessions)

**Use skill:** `$streamlit-master-architect` (page import discipline, AppTest patterns)

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Note:** Paths use `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}`. Adjust to your local skill install if needed.

Skill references to consult (as needed):

- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/testing_apptest.md`
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/architecture_state.md`

**Preflight:**

```bash
uv sync
uv run python -c "import streamlit as st; print(st.__version__)"
rg -n "duckdb\\.connect|__import__\\(" src/pages/03_analytics.py
```

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Locate DB usage + telemetry parsing:
- `rg -n 'duckdb\.connect|\.execute\(|\.df\(' -S src/pages/03_analytics.py src`
- `rg -n 'telemetry\.jsonl|log_jsonl|rotate' -S src/utils/telemetry.py src/telemetry`
- Read in parallel:
  - `src/pages/03_analytics.py`
  - `src/utils/telemetry.py`
  - any analytics storage modules under `src/persistence/` or `src/telemetry/`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local DuckDB/telemetry resources before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc` (rare for this package)
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (rare for this package)
- OpenAI API docs: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (rare for this package)

**API verification (Context7 + web):**

- If available, use `functions.mcp__context7__resolve-library-id` → `duckdb` and `functions.mcp__context7__query-docs` for connection lifecycle patterns.
- Otherwise, prefer Exa/web for DuckDB Python docs (time-sensitive and version-specific).

**Long-running UI verification (use native capabilities):**

- If you run `streamlit run src/app.py`, keep it alive and use `functions.write_stdin` for logs.
- Attach screenshots of analytics charts/tables with `functions.view_image` if needed.

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` focused on:
  - bounded telemetry parsing (no full-file load)
  - privacy defaults (no raw payload display)

**MCP tool sequence (use when it adds signal):**

1. `functions.mcp__zen__planner` → plan: DB lifecycle + telemetry parser + tests.
2. Context7:
   - resolve `duckdb` and query docs for connection lifecycle and `.df()` usage.
3. `functions.mcp__zen__secaudit` → ensure telemetry parsing is bounded and no secrets are logged.

**opensrc (optional):**

```bash
cat opensrc/sources.json | rg -n "duckdb" || true
# Fetch only if missing and behavior is surprising; treat opensrc/ as read-only.
npx opensrc pypi:duckdb
```

## Implementation Executor Template (DocMind / Python)

### You Are

As an autonomous implementation agent for the **DocMind AI LLM** repository, you will:

Implement the feature described below end-to-end, including:

- code changes
- tests
- documentation updates (ADR/SPEC/RTM)

You must keep changes minimal, library-first, and maintainable.

---

### Feature Context (Filled)

**Primary Task:** Refactor the Streamlit Analytics page to close DuckDB connections deterministically, parse local telemetry JSONL efficiently, and enforce an explicit runtime gate plus canonical DuckDB path.

**Why now:** Current Analytics page risks resource leaks and reads telemetry via hardcoded paths with full-file loads. This is avoidable and hurts reliability for long-running sessions.

**Definition of Done (DoD):**

- Analytics page is gated by `DOCMIND_ANALYTICS_ENABLED=true` at runtime.
- DuckDB file path is canonical: `data/analytics/analytics.duckdb`.
- `src/pages/03_analytics.py` closes DuckDB connections (context manager or `try/finally`).
- No dynamic `__import__` remains in Analytics page.
- Telemetry parsing is streaming/bounded and uses canonical telemetry path.
- Canonical analytics DB path constant + getter live in `src/utils/telemetry.py` and are enforced in `src/pages/03_analytics.py`.
- Unit tests cover telemetry parsing caps and invalid lines.
- Page remains importable (existing smoke test passes).

**In-scope modules/files (initial):**

- `src/pages/03_analytics.py`
- `src/utils/telemetry.py` (add a public telemetry path getter/constant if needed)
- `tests/unit/pages/test_analytics_telemetry_parsing.py` (new)
- `docs/developers/adrs/ADR-053-analytics-page-hardening.md`
- `docs/specs/spec-034-analytics-page-hardening.md`
- `docs/specs/traceability.md`

---

### Step-by-Step Execution Plan (Filled)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Implement telemetry parsing helper with caps (stream lines, ignore invalid JSON).
2. [ ] Refactor `src/pages/03_analytics.py`:
   - close DuckDB connections deterministically
   - remove dynamic imports
   - use canonical telemetry path
3. [ ] Add unit tests for parsing helper.
4. [ ] Update RTM and run quality gates.

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

1. `duckdb.connect(...)` without close.
2. `Path(...).read_text().splitlines()` on potentially large telemetry files.
3. Hardcoded duplicate telemetry paths.
4. Showing raw telemetry payloads by default (privacy risk).

---

### MCP Tool Strategy (For Implementation Run)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (DB lifecycle + parser + tests).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → DuckDB Python docs if time-sensitive behavior is unclear.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API details (`duckdb`) when available.
4. `functions.mcp__gh_grep__searchGitHub` → optional patterns for bounded JSONL parsing and DuckDB usage.
5. `functions.mcp__zen__analyze` → optional if analytics code touches multiple layers (telemetry + persistence).
6. `functions.mcp__zen__codereview` → recommended post-implementation review.
7. `functions.mcp__zen__secaudit` → required security audit (privacy defaults + bounded parsing).

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### Final Verification Checklist (Must Complete)

| Requirement     | Status | Proof / Notes                                                                                           |
| --------------- | ------ | ------------------------------------------------------------------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                                                                         |
| **Formatting**  |        | `uv run ruff format .`                                                                                  |
| **Lint**        |        | `uv run ruff check .` clean                                                                             |
| **Types**       |        | `uv run pyright` clean                                                                                  |
| **Tests**       |        | parsing tests green; `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**        |        | ADR/SPEC/RTM updated                                                                                    |
| **Security**    |        | bounded parsing; privacy-safe defaults; no secret logs                                                  |
| **Tech Debt**   |        | zero TODO/FIXME introduced                                                                              |
| **Performance** |        | streaming JSONL parsing; DuckDB connections closed deterministically                                    |

**EXECUTE UNTIL COMPLETE.**
