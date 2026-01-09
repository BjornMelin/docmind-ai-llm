# Implementation Prompt — Config Surface Pruning (Unused Knobs)

Implements `ADR-054` + `SPEC-035`.

## Tooling & Skill Strategy (fresh Codex sessions)

This is config + docs truth work; keep it minimal and test-backed.

**Primary tools to leverage:**

- `rg` to confirm each knob is truly unused (no runtime imports or env mappings).
- Context7 for Pydantic Settings v2 behaviors (unknown env vars, nested keys) if needed.
- `functions.mcp__zen__codereview` after pruning to ensure no accidental breakage.

**Security note:**

Removing knobs is a behavior change. If a knob relates to backup/security/egress, run `functions.mcp__zen__secaudit` after implementation.

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
- `docs/developers/adrs/ADR-035-semantic-cache-gptcache-sqlite-faiss.md` (update status or cross-link as out-of-scope for v1)
- `docs/developers/adrs/ADR-033-local-backup-and-retention.md` (update status or cross-link as out-of-scope for v1)
- `docs/developers/adrs/ADR-023-analysis-mode-strategy.md` (update status or cross-link as out-of-scope for v1)
- `tests/unit/config/test_settings_pruned_fields.py` (new)
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Implementing semantic cache / analysis mode / backup features for v1.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

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
uv run pylint --fail-under=9.5 src/ tests/ scripts/
uv run python scripts/run_tests.py --fast
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Shipping no-op config flags without docs warnings.
2. Adding new dependency groups for deferred features.
3. Breaking settings load due to stricter `extra` handling.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes                        |
| ----------- | ------ | ------------------------------------ |
| Formatting  |        | `ruff format`                        |
| Lint        |        | `ruff check` clean                   |
| Types       |        | `pyright` clean                      |
| Pylint      |        | meets threshold                      |
| Tests       |        | settings-prune tests green           |
| Docs        |        | config reference + ADR links updated |
| RTM         |        | NFR-MAINT-003 updated                |

**EXECUTE UNTIL COMPLETE.**
