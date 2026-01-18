---
prompt: PROMPT-022
title: Settings UI Hardening + Safe Provider Badge
status: Completed
date: 2026-01-11
version: 1.0
related_adrs: ["ADR-041"]
related_specs: ["SPEC-022"]
---

Implements `ADR-041` + `SPEC-022`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-041-settings-ui-hardening-and-safe-badges.md`
- SPEC: `docs/specs/spec-022-settings-ui-hardening.md`
- Requirements: `docs/specs/requirements.md` (FR-021, FR-012, NFR-SEC-004)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.streamlit.io/develop/api-reference/text/st.badge> — Replace HTML badges with Streamlit-native badges.
- <https://docs.streamlit.io/develop/api-reference/execution-flow/st.form> — Forms, submit behavior, and rerun semantics.
- <https://docs.streamlit.io/develop/api-reference/execution-flow/st.form_submit_button> — Form submit button behavior (disabled gating, enter-to-submit).
- <https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state> — Session state discipline (Streamlit reruns).
- <https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest> — AppTest reference for integration tests.
- <https://docs.pydantic.dev/latest/concepts/models/> — `model_validate`, validation errors, and patterns.
- <https://docs.pydantic.dev/latest/concepts/pydantic_settings/> — Pydantic Settings v2 env mapping patterns.
- <https://python-dotenv.readthedocs.io/en/v1.0.0/> — `set_key`/`unset_key` reference.
- <https://github.com/theskumar/python-dotenv> — python-dotenv canonical project + usage examples.

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Use skill:** `$streamlit-master-architect`

Load and follow its workflows for:

- rerun discipline + `st.session_state` correctness
- AppTest patterns
- security-by-default (no unsafe HTML)

Skill references to consult (as needed):

- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/security.md`
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/testing_apptest.md`
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/widget_keys_and_reruns.md`

**Streamlit preflight (version + docs + audit):**

```bash
uv sync
uv run python -c "import streamlit as st; print(st.__version__)"
uv run python ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/scripts/audit_streamlit_project.py --root . --format md
uv run python ${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/scripts/sync_streamlit_docs.py --out /tmp/streamlit-docs
```

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel` for independent work):**

- Repo truth scan:
  - `rg -n "unsafe_allow_html=True" -S src/ui src/pages`
  - `rg -n "(_persist_env|\\.env|set_key\\(|unset_key\\()" -S src/pages/04_settings.py src/config src/utils`
- Read key files (in parallel): `src/pages/04_settings.py`, `src/ui/components/provider_badge.py`, `src/config/settings.py`, `src/config/integrations.py`.

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Streamlit/Pydantic/python-dotenv references.
- `functions.read_mcp_resource` → read any relevant local docs/indexes before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc`
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain`
- OpenAI API docs: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (only if this work package touches OpenAI API semantics)

**API verification (Context7, only when uncertain):**

- `functions.mcp__context7__resolve-library-id` → `streamlit`, `pydantic`, `python-dotenv`
- `functions.mcp__context7__query-docs` → verify exact signatures/behavior for:
  - `st.badge`, forms (`st.form`, `st.form_submit_button`), disabled buttons
  - Pydantic v2: `model_validate`, `ValidationError` handling
  - python-dotenv: `set_key`, `unset_key` quoting behavior

**Time-sensitive facts (use web tools):**

- Prefer `functions.mcp__exa__web_search_exa` for discovery. Use `web.run` when you need citations or dates. Streamlit form changes and dotenv quote options are the typical cases.

**Long-running UI validation (use native capabilities):**

- If you start `uv run streamlit run app.py`, keep it running and use `functions.write_stdin` to fetch logs instead of restarting.
- If you capture UI screenshots during verification, attach them with `functions.view_image`.
- For user-critical E2E smoke, use the skill’s Playwright flow:
  - `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/scripts/mcp/run_playwright_mcp_e2e.py`

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` after implementation to confirm:
  - no `unsafe_allow_html=True` sinks remain
  - `.env` persistence rejects newline/control characters and does not log secrets

**Final review gate (recommended):**

- Run `functions.mcp__zen__codereview` after tests pass to catch layering regressions (UI vs domain).

### MCP tool sequence (use when it adds signal)

1. `functions.mcp__zen__planner` → plan the UI + tests steps.
2. Context7 API verification:
   - `functions.mcp__context7__resolve-library-id` → `streamlit`, `pydantic`, `python-dotenv`
   - `functions.mcp__context7__query-docs` → `st.badge`, `st.form_submit_button`, `st.session_state`, dotenv `set_key/unset_key`
3. Exa (official docs only) for Streamlit forms + best practices if uncertain.
4. `functions.mcp__zen__secaudit` → validate no new injection/logging sinks.

**opensrc (only if subtle behavior):**

```bash
cat opensrc/sources.json | rg -n "python-dotenv|streamlit|pydantic" || true
# Fetch only if missing and behavior is surprising; treat opensrc/ as read-only.
npx opensrc pypi:python-dotenv
```

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

**Primary Task:** Harden the Streamlit Settings page with pre-validation and safe `.env` persistence, and replace the unsafe provider badge HTML with Streamlit-native UI.

**Why now:** Current Settings can persist invalid `.env` values and includes an avoidable XSS-class sink via `unsafe_allow_html=True`. This is a ship blocker for v1.

**Definition of Done (DoD):**

- `src/ui/components/provider_badge.py` contains **no** `unsafe_allow_html=True` usage.
- Settings Save/Apply validate a candidate settings payload before mutating global `settings` or writing `.env`.
- Apply runtime updates the process-global settings singleton in-place (do not rebind `src.config.settings.settings`) so imports using `from src.config import settings` remain consistent.
- `.env` persistence uses `python-dotenv` (`set_key`/`unset_key`) and is covered by tests.
- AppTest/pytest coverage verifies invalid settings disable actions and valid settings apply/persist successfully.
- `docs/specs/traceability.md` updated (planned → implemented row for FR-021).

**In-scope modules/files (initial):**

- `src/pages/04_settings.py`
- `src/ui/components/provider_badge.py`
- `tests/integration/test_settings_page.py`
- `docs/developers/adrs/ADR-041-settings-ui-hardening-and-safe-badges.md`
- `docs/specs/spec-022-settings-ui-hardening.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Adding new providers/backends.
- Enabling remote endpoints by default.

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python baseline is **3.13.11** (Python 3.13-only; respect `pyproject.toml`).
- Use **uv only**.

#### 2) Style, Types, and Lint

Must pass:

- `uv run ruff format .`
- `uv run ruff check . --fix`
- `uv run pyright`

#### 3) Streamlit UI Discipline

- No expensive work at import time.
- Use Streamlit-native components; avoid unsafe HTML.

#### 4) Config Discipline

- Validate using `DocMindSettings` before persistence.
- Do not scatter `os.getenv` in UI code.
- Do not rebind `src.config.settings.settings` (mutate/apply in-place).
- Remember Pydantic Settings precedence: env vars override `.env`. Saving `.env` does not affect values already exported into the process environment.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Baseline scan: confirm only `src/ui/components/provider_badge.py` uses `unsafe_allow_html=True`.

   - Run: `rg -n \"unsafe_allow_html\\s*=\\s*True\" -S src`

2. [ ] Replace provider badge with Streamlit-native UI (`st.badge` + `st.caption`).

   - Files: `src/ui/components/provider_badge.py`
   - Run: `uv run ruff format src/ui/components/provider_badge.py`

3. [ ] Implement Settings candidate validation (pre-validate before apply/save).

   - Files: `src/pages/04_settings.py`
   - Use `DocMindSettings.model_validate(...)` to validate a candidate payload.

4. [ ] Replace custom `.env` writer with python-dotenv `set_key`/`unset_key`.

   - Files: `src/pages/04_settings.py`

5. [ ] Update/add tests (AppTest + unit) for settings validation and persistence.

   - Files: `tests/integration/test_settings_page.py` (+ new files if needed)

6. [ ] Update RTM and verify quality gates.

   - Files: `docs/specs/traceability.md`
   - Run:

     ```bash
     uv run ruff format .
     uv run ruff check . --fix
     uv run pyright
     uv run python scripts/run_tests.py --fast
     uv run python scripts/run_tests.py
     ```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. `unsafe_allow_html=True` rendering of dynamic content.
2. Persisting `.env` without Pydantic validation first.
3. Silent exception swallowing around Apply/Save.
4. Import-time IO in Streamlit pages/components.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (UI + tests).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → official docs/changelogs if Streamlit/python-dotenv behavior is time-sensitive.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API details (`streamlit`, `pydantic`, `python-dotenv`).
4. `functions.mcp__gh_grep__searchGitHub` → real-world patterns for forms/validation gating (only if needed).
5. `functions.mcp__zen__analyze` → only if you find unexpected UI↔config coupling.
6. `functions.mcp__zen__codereview` → post-implementation review.
7. `functions.mcp__zen__secaudit` → required security audit (no unsafe HTML; safe `.env` persistence).

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes                                                                                            |
| ----------- | ------ | -------------------------------------------------------------------------------------------------------- |
| Packaging   |        | `uv sync` clean                                                                                          |
| Formatting  |        | `ruff format`                                                                                            |
| Lint        |        | `ruff check` clean                                                                                       |
| Types       |        | `pyright` clean                                                                                          |
| Tests       |        | settings tests green; `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| Docs        |        | ADR/SPEC/RTM updated                                                                                     |
| Security    |        | no unsafe HTML; allowlist enforced                                                                       |
| Tech Debt   |        | no work-marker placeholders introduced                                                                   |
| Performance |        | no new import-time heavy work                                                                            |

**EXECUTE UNTIL COMPLETE.**
