---
prompt: PROMPT-027
title: Remove Legacy `src/main.py` Entrypoint
status: Completed
date: 2026-01-11
version: 1.0
related_adrs: ["ADR-046"]
related_specs: ["SPEC-027"]
---

Implements `ADR-046` + `SPEC-027`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-046-remove-legacy-main-entrypoint.md`
- SPEC: `docs/specs/spec-027-remove-legacy-main-entrypoint.md`
- Requirements: `docs/specs/requirements.md` (NFR-PORT-001, NFR-MAINT-003)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.streamlit.io/develop/get-started/run-your-app> — Streamlit run command expectations (supported entrypoint patterns).
- <https://docs.streamlit.io/develop/concepts/multipage-apps/overview> — Multipage app entrypoint patterns.

## Tooling & Skill Strategy (fresh Codex sessions)

This work is primarily repo hygiene; prefer local repo truth over web research.

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` to find imports/references and doc mentions.
- Use `multi_tool_use.parallel` for independent searches (imports + docs + configs) to minimize back-and-forth.
- `functions.mcp__zen__analyze` if unexpected coupling is found (imports from `src.main`).
- `functions.mcp__zen__codereview` after deletion to ensure no dangling references remain.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Find code references:
  - `rg -n \"\\bsrc/main\\.py\\b|\\bsrc\\.main\\b\" -S src tests`
- Find docs/scripts references:
  - `rg -n \"\\bsrc/main\\.py\\b|\\bsrc\\.main\\b\" -S docs scripts pyproject.toml README.md`
- Read file before deletion:
  - `cat src/main.py` (confirm no hidden exports used elsewhere; file is typically <250 lines)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local “entrypoints/runbook” resources (if present).

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc`
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain`
- OpenAI API docs: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (rare for this package)

**Editing discipline:**

- Use `functions.apply_patch` for small documentation updates and `*** Delete File: src/main.py` for the deletion.

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

**Primary Task:** Delete `src/main.py` and remove all references so the supported entrypoint is only `uv run streamlit run app.py`.

**Why now:** `src/main.py` is dead code, contains misleading “phase 2” placeholders, and does import-time `.env` loading. It is a ship blocker for clarity and container correctness.

**Definition of Done (DoD):**

- `src/main.py` removed.
- No docs/config refer to `src/main.py` as a run path.
- `pyproject.toml` coverage omit list updated accordingly.
- Quality gates pass.

**In-scope modules/files (initial):**

- `src/main.py` (delete)
- `pyproject.toml`
- `docs/` (any references)
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Adding a replacement CLI entrypoint.

---

### Hard Rules (Execution)

#### 1) Python + Packaging

- Python **>=3.13,<3.14** (Python 3.13-only; respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`

#### 2) Style, Types, and Lint

Must pass:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`

#### 3) Deletion safety

- Delete only `src/main.py` and its direct references; do not remove unrelated files.
- Avoid destructive git/history actions (`git reset --hard`, etc.).

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Confirm nothing imports `src.main` (`rg "from src\\.main|import src\\.main"`).
2. [ ] Delete `src/main.py`.
3. [ ] Update `pyproject.toml` (coverage omit list) to remove `src/main.py`.
4. [ ] Search docs for `src/main.py` references and replace with `uv run streamlit run app.py`.
5. [ ] Update RTM row (NFR-MAINT-003 planned → implemented).
6. [ ] Run quality gates.

Commands:

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run pyright
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
```

---

### Anti-Pattern Kill List (Immediate Deletion/Rewrite)

1. Leaving dual entrypoints (docs or scripts still referencing `src/main.py`).
2. Import-time `.env` loads or behavior that bypasses `src/config/settings.py`.
3. Deleting “unused” code without proving it’s unused (must `rg` imports/references first).

---

### MCP Tool Strategy (For Implementation Run)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → plan deletion + reference cleanup + verification.
2. `functions.mcp__exa__web_search_exa` / `web.run` → only if you need to confirm a packaging/entrypoint best practice (rare).
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → generally unnecessary for this prompt.
4. `functions.mcp__gh_grep__searchGitHub` → unnecessary (repo hygiene task).
5. `functions.mcp__zen__analyze` → use if you find unexpected runtime coupling to `src.main`.
6. `functions.mcp__zen__codereview` → recommended post-implementation review.
7. `functions.mcp__zen__secaudit` → not required unless you discover new security-sensitive behavior.

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### Final Verification Checklist (Must Complete)

| Requirement     | Status | Proof / Notes                                                                      |
| --------------- | ------ | ---------------------------------------------------------------------------------- |
| **Packaging**   | ✅     | `uv sync` clean                                                                    |
| **Formatting**  | ✅     | `uv run ruff format .`                                                             |
| **Lint**        | ✅     | `uv run ruff check .` clean                                                        |
| **Types**       | ✅     | `uv run pyright` clean                                                             |
| **Tests**       | ✅     | `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**        | ✅     | docs/scripts reference only `uv run streamlit run app.py`                          |
| **Security**    | ✅     | no new config bypasses introduced                                                  |
| **Tech Debt**   | ✅     | zero work-marker placeholders introduced                                           |
| **Performance** | ✅     | no new import-time heavy work                                                      |

**EXECUTE UNTIL COMPLETE.**
