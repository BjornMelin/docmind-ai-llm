# Implementation Prompt — Remove Legacy `src/main.py` Entrypoint

Implements `ADR-046` + `SPEC-027`.

## Tooling & Skill Strategy (fresh Codex sessions)

This work is primarily repo hygiene; prefer local repo truth over web research.

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

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
  - `sed -n '1,220p' src/main.py` (confirm no hidden exports used elsewhere)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local “entrypoints/runbook” resources (if present).

**Editing discipline:**

- Use `functions.apply_patch` for small documentation updates and `*** Delete File: src/main.py` for the deletion.

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### FEATURE CONTEXT (FILLED)

**Primary Task:** Delete `src/main.py` and remove all references so the supported entrypoint is only `streamlit run src/app.py`.

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

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Confirm nothing imports `src.main` (`rg "from src\\.main|import src\\.main"`).
2. [ ] Delete `src/main.py`.
3. [ ] Update `pyproject.toml` (coverage omit list) to remove `src/main.py`.
4. [ ] Search docs for `src/main.py` references and replace with `streamlit run src/app.py`.
5. [ ] Update RTM row (NFR-MAINT-003 planned → implemented).
6. [ ] Run quality gates.

Commands:

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pylint --fail-under=9.5 src/ tests/ scripts/
uv run python scripts/run_tests.py --fast
```

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement   | Status | Proof / Notes                                    |
| ------------- | ------ | ------------------------------------------------ |
| Entrypoints   |        | `streamlit run src/app.py` documented            |
| References    |        | `rg "src/main.py" docs src pyproject.toml` clean |
| Quality gates |        | commands green                                   |

**EXECUTE UNTIL COMPLETE.**
