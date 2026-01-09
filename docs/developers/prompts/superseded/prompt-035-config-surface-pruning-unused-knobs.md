# Implementation Prompt — Config Surface Pruning (Unused Knobs)

Implements `ADR-054` + `SPEC-035`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/superseded/ADR-054-config-surface-pruning-unused-knobs.md`
- SPEC: `docs/specs/superseded/spec-035-config-surface-pruning-unused-knobs.md`
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.pydantic.dev/latest/concepts/pydantic_settings/> — Pydantic Settings v2 behavior (unknown env vars, nested keys).
- <https://docs.pydantic.dev/latest/concepts/config/> — Model config patterns (`extra`, strictness, etc.).

## Tooling & Skill Strategy (fresh Codex sessions)

This is config + docs truth work; keep it minimal and test-backed.

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` to confirm each knob is truly unused (no runtime imports or env mappings).
- Context7 for Pydantic Settings v2 behaviors (unknown env vars, nested keys) if needed.
- `functions.mcp__zen__codereview` after pruning to ensure no accidental breakage.

**Security note:**

Removing knobs is a behavior change. If a knob relates to backup/security/egress, run `functions.mcp__zen__secaudit` after implementation.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Inventory candidate knobs and their usage:
  - `rg -n \"class (Backup|SemanticCache|Analysis|Agent)Config|backup_|semantic_cache|analysis_mode\" -S src/config/settings.py`
  - `rg -n \"DOCMIND_(BACKUP|SEMANTIC_CACHE|ANALYSIS|AGENT)\" -S src docs .env.example`
  - `rg -n \"backup_|semantic_cache|analysis_mode|enable_deadline_propagation|enable_router_injection\" -S src tests docs`
- Read in parallel:
  - `src/config/settings.py`
  - `docs/developers/configuration-reference.md`
  - ADRs referenced by this prompt (so removals are documented correctly)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for config/Settings resources; prefer local resources before web search.

**API verification (Context7):**

- `functions.mcp__context7__resolve-library-id` → `pydantic`, `pydantic-settings`
- `functions.mcp__context7__query-docs` → confirm how “unknown env vars” are handled and how to keep startup tolerant.

**Review gate (required):**

- Run `functions.mcp__zen__codereview` after tests pass to confirm no knob removal breaks imports or docs guarantees.

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### YOU ARE

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes
- tests
- documentation updates (ADR/SPEC/RTM)

You must keep changes minimal, library-first, and maintainable.

---

### FEATURE CONTEXT (FILLED)

**Primary Task:** Remove unused/no-op configuration knobs from `src/config/settings.py` and update docs so v1 does not advertise unsupported features.

**Why now:** The repo currently ships settings fields that are never used. This is misleading and increases support burden. v1 should have a truthful configuration surface.

**Definition of Done (DoD):**

- Unused settings fields removed (backup, semantic_cache, analysis, unused agent flags).
- Docs updated to remove references to removed knobs.
- Backward compatibility preserved: unknown env vars do not break startup (extra=ignore) and optional warning behavior is documented.
- Unit tests cover settings load with deprecated env vars present.
- RTM updated under NFR-MAINT-003.

**In-scope modules/files (initial):**

- `src/config/settings.py`
- `docs/developers/configuration-reference.md`
- `docs/developers/adrs/ADR-035-semantic-cache-qdrant.md` (note: superseded plan; see prompt-038)
- `docs/developers/adrs/ADR-033-local-backup-and-retention.md` (note: superseded plan; see prompt-037)
- `docs/developers/adrs/ADR-023-analysis-mode-strategy.md` (note: superseded plan; see prompt-036)
- `tests/unit/config/test_settings_pruned_fields.py` (new)
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Implementing semantic cache / analysis mode / backup features for v1.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Identify unused settings fields (confirm no references in `src/`).
2. [ ] Remove unused fields/models from `src/config/settings.py` and update exports.
3. [ ] Update docs to remove references and/or mark ADRs as out-of-scope for v1.
4. [ ] Add unit tests validating settings loading when deprecated env vars are set.
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

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Shipping no-op config flags without docs warnings.
2. Adding new dependency groups for deferred features.
3. Breaking settings load due to stricter `extra` handling.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (prune + docs + tests).
2. `functions.mcp__exa__web_search_exa` / `web.run` → generally unnecessary; prefer repo truth.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative Pydantic Settings v2 behavior (unknown env vars).
4. `functions.mcp__gh_grep__searchGitHub` → optional patterns for “deprecated env vars tolerated” behaviors.
5. `functions.mcp__zen__analyze` → optional if pruning affects multiple subsystems.
6. `functions.mcp__zen__codereview` → required post-implementation review (avoid accidental breakage).
7. `functions.mcp__zen__secaudit` → required only if pruning touches security/egress-related knobs.

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement     | Status | Proof / Notes                                                                                                  |
| --------------- | ------ | -------------------------------------------------------------------------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                                                                                |
| **Formatting**  |        | `uv run ruff format .`                                                                                         |
| **Lint**        |        | `uv run ruff check .` clean                                                                                    |
| **Types**       |        | `uv run pyright` clean                                                                                         |
| **Pylint**      |        | meets threshold                                                                                                |
| **Tests**       |        | settings-prune tests green; `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**        |        | config reference + ADR links updated; RTM NFR-MAINT-003 updated                                                |
| **Security**    |        | no security/egress knobs removed without docs/tests; startup remains tolerant to unknown env vars              |
| **Tech Debt**   |        | zero TODO/FIXME introduced                                                                                     |
| **Performance** |        | settings load remains fast (no heavy validators)                                                               |

**EXECUTE UNTIL COMPLETE.**
