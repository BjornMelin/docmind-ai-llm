# Implementation Prompt — Config Discipline (Remove `os.getenv` Sprawl)

Implements `ADR-050` + `SPEC-031`.

## Tooling & Skill Strategy (fresh Codex sessions)

This is security-sensitive config work. Prefer repo truth and run structured audits.

**Primary tools to leverage:**

- `rg` to inventory every `os.getenv` call and ensure all are removed from core modules.
- Context7 for authoritative Pydantic Settings v2 patterns and typing (nested env parsing).
- `opensrc/` for Pydantic internals only when behavior is surprising (prefer docs first).
- `functions.mcp__zen__secaudit` (mandatory) after changes: confirm no secret logging and no new egress surfaces.
- `functions.mcp__zen__codereview` for final correctness gate.

**MCP tool sequence (use when it adds signal):**

1. `functions.mcp__zen__planner` → plan settings schema + refactors + tests.
2. Context7:
   - resolve `pydantic` (and `pydantic-settings`) and query docs for env mapping and nested keys.
3. Exa search (official pydantic docs) if a validator/SettingsConfigDict behavior is unclear.

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### FEATURE CONTEXT (FILLED)

**Primary Task:** Centralize JSONL telemetry + image encryption env config in `DocMindSettings` and remove `os.getenv` sprawl from core modules.

**Why now:** Scattered env reads undermine settings discipline and hide security-sensitive toggles. There is also an `ADR-XXX` marker and unused hashing config that should not ship.

**Definition of Done (DoD):**

- `TelemetryConfig` and `ImageEncryptionConfig` exist in `src/config/settings.py`.
- `DOCMIND_TELEMETRY_*`, `DOCMIND_IMG_*`, and `DOCMIND_ENVIRONMENT` map through settings.
- `src/utils/telemetry.py`, `src/utils/security.py`, `src/processing/pdf_pages.py`, and `src/telemetry/opentelemetry.py` do not call `os.getenv`.
- Unused hashing config + ADR-XXX marker removed (or properly wired if truly used).
- Unit tests validate env→settings mapping and telemetry behavior.
- Quality gates pass.

**In-scope modules/files (initial):**

- `src/config/settings.py`
- `src/utils/telemetry.py`
- `src/utils/security.py`
- `src/processing/pdf_pages.py`
- `src/telemetry/opentelemetry.py`
- `tests/unit/config/` (new tests)
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Renaming env vars.
- Introducing a new config layer outside settings.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Inventory env reads: `rg "os\\.getenv\\(" src/`.
2. [ ] Add new settings groups and mappings in `src/config/settings.py`.
3. [ ] Refactor `src/utils/telemetry.py` to read from `settings` (no env reads).
4. [ ] Refactor image encryption env reads to settings and update callers.
5. [ ] Refactor OTEL resource env read (`DOCMIND_ENVIRONMENT`) to settings.
6. [ ] Remove unused hashing config + ADR-XXX marker (or document and wire if truly used).
7. [ ] Add/update unit tests for settings mappings and telemetry.
8. [ ] Update RTM row and run quality gates.

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

| Requirement | Status | Proof / Notes                                              |
| ----------- | ------ | ---------------------------------------------------------- |
| Env reads   |        | `rg "os\\.getenv\\(" src/` only in settings layer (if any) |
| Tests       |        | mapping tests green                                        |
| Docs        |        | RTM updated                                                |

**EXECUTE UNTIL COMPLETE.**
