# Implementation Prompt — Multimodal Helper Cleanup

Implements `ADR-049` + `SPEC-030`.

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

### STEP-BY-STEP EXECUTION PLAN (FILLED)

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
```

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes                       |
| ----------- | ------ | ----------------------------------- |
| Dead code   |        | no unused multimodal helper in prod |
| Tests       |        | suite green                         |
| Docs        |        | references removed or updated       |

**EXECUTE UNTIL COMPLETE.**
