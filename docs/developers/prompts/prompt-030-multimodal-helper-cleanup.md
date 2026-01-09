# Implementation Prompt — Multimodal Helper Cleanup

Implements `ADR-049` + `SPEC-030`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-049-multimodal-helper-cleanup.md`
- SPEC: `docs/specs/spec-030-multimodal-helper-cleanup.md`
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.python.org/3/library/importlib.html> — Import mechanics (useful if you find dynamic/indirect imports before deletion).
- <https://docs.python.org/3/library/pathlib.html> — Filesystem/path handling patterns (if refactors touch file IO).

## Tooling & Skill Strategy (fresh Codex sessions)

This is a deletion/cleanup task. Prefer local repo truth.

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` for finding all references (production and tests).
- Use `multi_tool_use.parallel` to scan `src/`, `tests/`, and `docs/` concurrently.
- `functions.mcp__zen__analyze` if references look indirect or dynamic.
- `functions.mcp__zen__codereview` after deletion to ensure no dangling imports.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Confirm real usage:
  - `rg -n \"\\bsrc\\.utils\\.multimodal\\b|\\bmultimodal\\.py\\b\" -S src tests docs`
  - `rg -n \"TODO\\(multimodal-phase-2\\)\" -S src/utils/multimodal.py || true`
- Read before delete:
  - `sed -n '1,120p' src/utils/multimodal.py` (ensure no production exports are expected)

**Architecture gate (optional but cheap):**

- If you see dynamic imports or indirect references, run `functions.mcp__zen__analyze` to avoid deleting something required at runtime.

**Editing discipline:**

- Use `functions.apply_patch` with `*** Delete File: src/utils/multimodal.py` and delete/update test files in the same patch to keep diffs reviewable.

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

**Primary Task:** Remove `src/utils/multimodal.py` and its tests if it is not used by production code, eliminating TODO placeholders and dead code.

**Why now:** The module is a test-only “toy pipeline” with TODOs, which is not acceptable for v1 release readiness.

**Definition of Done (DoD):**

- `src/utils/multimodal.py` removed (or moved out of production package) and no production imports remain.
- Tests referencing it are removed/updated.
- Docs/architecture maps updated to remove references (or ticketed under WP08 if explicitly deferred).
- Quality gates pass.

**In-scope modules/files (initial):**

- `src/utils/multimodal.py` (delete)
- `tests/unit/utils/multimodal/` (delete/update)
- `docs/specs/traceability.md`
- `docs/developers/system-architecture.md` (if referenced)

**Out-of-scope (explicit):**

- Building a real multimodal pipeline.

---

### HARD RULES (EXECUTION)

#### 1) Deletion safety

- Prove unused before deleting: `rg` all references in `src/`, `tests/`, `docs/`, `scripts/`.
- If anything imports the module at runtime, stop and reassess scope (do not delete blindly).

#### 2) Quality gates

- Run standard repo gates after deletion:
  - `uv run ruff format .`
  - `uv run ruff check . --fix`
  - `uv run pyright`
  - `uv run pylint --fail-under=9.5 src/ tests/ scripts/`
  - `uv run python scripts/run_tests.py --fast` (then `uv run python scripts/run_tests.py` before marking complete)

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Confirm unused in production: `rg "src\\.utils\\.multimodal" src/`.
2. [ ] Delete `src/utils/multimodal.py`.
3. [ ] Delete/update `tests/unit/utils/multimodal/*`.
4. [ ] Update docs references if present (or coordinate with WP08).
5. [ ] Update RTM row and run quality gates.

Commands:

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run pylint --fail-under=9.5 src/ tests/ scripts/
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Moving dead code to another production module/package instead of deleting it.
2. Keeping “phase 2” TODO placeholders in `src/` (violates release readiness).
3. Removing tests without replacing coverage where the behavior is still required elsewhere.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → plan deletion + reference cleanup.
2. `functions.mcp__exa__web_search_exa` / `web.run` → unnecessary (repo hygiene task).
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → unnecessary.
4. `functions.mcp__gh_grep__searchGitHub` → unnecessary.
5. `functions.mcp__zen__analyze` → use if you find dynamic/indirect references.
6. `functions.mcp__zen__codereview` → recommended post-implementation review.
7. `functions.mcp__zen__secaudit` → optional; use if deletion affects security-sensitive surfaces.

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement     | Status | Proof / Notes                                                                      |
| --------------- | ------ | ---------------------------------------------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                                                    |
| **Formatting**  |        | `uv run ruff format .`                                                             |
| **Lint**        |        | `uv run ruff check .` clean                                                        |
| **Types**       |        | `uv run pyright` clean                                                             |
| **Pylint**      |        | meets threshold                                                                    |
| **Tests**       |        | `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**        |        | references removed or updated                                                      |
| **Security**    |        | no security surfaces regressed by deletion                                         |
| **Tech Debt**   |        | zero TODO/FIXME introduced                                                         |
| **Performance** |        | less import surface; no new heavy imports                                          |

**EXECUTE UNTIL COMPLETE.**
