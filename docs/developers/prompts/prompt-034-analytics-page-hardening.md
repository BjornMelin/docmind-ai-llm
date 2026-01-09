# Implementation Prompt — Analytics Page Hardening

Implements `ADR-053` + `SPEC-034`.

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

**Primary Task:** Refactor the Streamlit Analytics page to close DuckDB connections deterministically and parse local telemetry JSONL efficiently using canonical paths.

**Why now:** Current Analytics page risks resource leaks and reads telemetry via hardcoded paths with full-file loads. This is avoidable and hurts reliability for long-running sessions.

**Definition of Done (DoD):**

- `src/pages/03_analytics.py` closes DuckDB connections (context manager or `try/finally`).
- No dynamic `__import__` remains in Analytics page.
- Telemetry parsing is streaming/bounded and uses canonical telemetry path.
- Unit tests cover telemetry parsing caps and invalid lines.
- Page remains importable (existing smoke test passes).

**In-scope modules/files (initial):**

- `src/pages/03_analytics.py`
- `src/utils/telemetry.py` (add a public telemetry path getter/constant if needed)
- `tests/unit/pages/test_analytics_telemetry_parsing.py` (new)
- `docs/developers/adrs/ADR-053-analytics-page-hardening.md`
- `docs/specs/spec-034-analytics-page-hardening.md`
- `docs/specs/traceability.md`

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Implement telemetry parsing helper with caps (stream lines, ignore invalid JSON).
2. [ ] Refactor `src/pages/03_analytics.py`:
   - close DuckDB connections deterministically
   - remove dynamic imports
   - use canonical telemetry path
3. [ ] Add unit tests for parsing helper.
4. [ ] Update RTM and run quality gates.

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

1. `duckdb.connect(...)` without close.
2. `Path(...).read_text().splitlines()` on potentially large telemetry files.
3. Hardcoded duplicate telemetry paths.
4. Showing raw telemetry payloads by default (privacy risk).

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes        |
| ----------- | ------ | -------------------- |
| Formatting  |        | `ruff format`        |
| Lint        |        | `ruff check` clean   |
| Types       |        | `pyright` clean      |
| Pylint      |        | meets threshold      |
| Tests       |        | parsing tests green  |
| Docs        |        | ADR/SPEC/RTM updated |

**EXECUTE UNTIL COMPLETE.**
