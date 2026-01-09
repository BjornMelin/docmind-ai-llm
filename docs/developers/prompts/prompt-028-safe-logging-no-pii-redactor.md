# Implementation Prompt — Safe Logging Policy (No PII Redactor Stub)

Implements `ADR-047` + `SPEC-028`.

## Tooling & Skill Strategy (fresh Codex sessions)

This is security-sensitive. Use structured security review tools.

**Primary tools to leverage:**

- `rg` to inventory logging call sites (`logger.*`, `loguru`, `logging`, `print`).
- Context7/Exa only if you need authoritative guidance for log safety patterns (otherwise stay repo-local).
- `functions.mcp__zen__secaudit` (mandatory) to review new helpers and any touched log statements.
- `functions.mcp__zen__codereview` to ensure no accidental raw-content logging is introduced.

**opensrc (optional):**

Use only if you must confirm behavior of a logging dependency; otherwise avoid.

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### FEATURE CONTEXT (FILLED)

**Primary Task:** Remove `src.utils.security.redact_pii` no-op stub and enforce metadata-only logging patterns by adding small helpers.

**Why now:** A no-op PII redactor creates false confidence and increases the chance that sensitive content is logged. DocMind must be safe-by-default.

**Definition of Done (DoD):**

- `redact_pii` removed from `src/utils/security.py` and exports.
- Tests updated (no assertions on no-op redaction).
- `src/utils/log_safety.py` exists with text fingerprinting + safe URL logging helpers.
- RTM updated: NFR-SEC-002 planned → implemented.
- No logging statements include raw user prompts, documents, or model outputs within the touched scope.

**In-scope modules/files (initial):**

- `src/utils/security.py`
- `src/utils/log_safety.py` (new)
- `tests/unit/utils/security/`
- `docs/specs/spec-028-safe-logging-no-pii-redactor.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Implementing regex-based PII redaction.
- Adding external scrubbing dependencies.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Identify all `redact_pii` references (`rg "redact_pii"`).
2. [ ] Remove `redact_pii` from `src/utils/security.py` and update `__all__`.
3. [ ] Update/delete the tests asserting no-op behavior.
4. [ ] Add `src/utils/log_safety.py` helpers (typed, minimal).
5. [ ] Replace any raw-content logging within scope with metadata-only logging.
6. [ ] Update RTM row and run quality gates.

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

| Requirement    | Status | Proof / Notes                    |
| -------------- | ------ | -------------------------------- |
| Sensitive logs |        | no raw prompt/doc content logged |
| Tests          |        | security tests updated           |
| Docs           |        | SPEC/RTM updated                 |

**EXECUTE UNTIL COMPLETE.**
