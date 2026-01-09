# Implementation Prompt — Safe Logging Policy (No PII Redactor Stub)

Implements `ADR-047` + `SPEC-028`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-047-safe-logging-policy-no-pii-redactor.md`
- SPEC: `docs/specs/spec-028-safe-logging-no-pii-redactor.md`
- Requirements: `docs/specs/requirements.md` (NFR-SEC-002, NFR-MAINT-003)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://loguru.readthedocs.io/> — Loguru logging docs (repo standard logger).
- <https://opentelemetry.io/docs/languages/python/> — OpenTelemetry Python docs (optional exporters; ensure default-off).
- <https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html> — OWASP logging guidance (avoid sensitive data in logs).

## Tooling & Skill Strategy (fresh Codex sessions)

This is security-sensitive. Use structured security review tools.

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` to inventory logging call sites (`logger.*`, `loguru`, `logging`, `print`).
- Context7/Exa only if you need authoritative guidance for log safety patterns (otherwise stay repo-local).
- `functions.mcp__zen__secaudit` (mandatory) to review new helpers and any touched log statements.
- `functions.mcp__zen__codereview` to ensure no accidental raw-content logging is introduced.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Inventory logging sinks and current redaction usage:
  - `rg -n \"\\bredact_pii\\b\" -S src tests`
  - `rg -n \"\\blogger\\.|\\bloguru\\b|\\bprint\\(\" -S src`
  - `rg -n \"telemetry\\.jsonl|log_jsonl\" -S src`
- Read in parallel:
  - `src/utils/security.py`
  - `src/utils/telemetry.py` and/or `src/utils/telemetry.py` (where JSONL logging is emitted)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local “security/logging/PII” resources before web search.

**API verification (Context7, only when needed):**

- `functions.mcp__context7__resolve-library-id` → `loguru` (and optionally `opentelemetry-sdk` if touched)
- `functions.mcp__context7__query-docs` → confirm any logging API details if you introduce wrappers.

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` with focus:
  - threat_level: high (because logs can exfiltrate PII)
  - ensure helpers never accept raw content without hashing/fingerprinting
  - ensure URLs are sanitized (no embedding API keys)

**Review gate (recommended):**

- Run `functions.mcp__zen__codereview` to ensure no accidental raw-content logging slipped in.

**opensrc (optional):**

Use only if you must confirm behavior of a logging dependency; otherwise avoid.

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

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python version must remain **3.11.x** (respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`
- Do not add dependencies for redaction/scrubbing for this prompt.

#### 2) Logging/PII discipline

- Never log raw prompts, documents, chat messages, or model outputs.
- If you need correlation, log only metadata (hashes/fingerprints, lengths, counts, status codes).
- Never log secrets or full URLs containing keys/tokens.

#### 3) Style, Types, and Lint

Must pass:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

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
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. “Security theater” stubs (like a no-op redactor) that imply safety without enforcement.
2. Broad `except Exception` around logging that hides failures silently.
3. Logging full URLs, request headers, or environment variables (potential secrets).
4. Logging raw content and calling it “sanitized” without proof/tests.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (refactor + tests).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → only if you need authoritative security/logging references.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → logging APIs (e.g., `loguru`) if you add small wrappers.
4. `functions.mcp__gh_grep__searchGitHub` → optional: see common “fingerprint logging” patterns.
5. `functions.mcp__zen__analyze` → only if refactor touches multiple layers (telemetry + UI + agents).
6. `functions.mcp__zen__codereview` → post-implementation review (required for security-sensitive prompts).
7. `functions.mcp__zen__secaudit` → mandatory security audit (PII/secrets in logs).

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
| **Docs**        |        | SPEC/RTM updated                                                                   |
| **Security**    |        | no raw prompt/doc/chat/model-output logs; no secret logs                           |
| **Tech Debt**   |        | zero TODO/FIXME introduced                                                         |
| **Performance** |        | logging remains constant-time/metadata-only                                        |

**EXECUTE UNTIL COMPLETE.**
