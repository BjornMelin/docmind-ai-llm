# Implementation Prompt — Docs Consistency Pass (Specs/Handbook/RTM)

Implements `ADR-048` + `SPEC-029`.

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### FEATURE CONTEXT (FILLED)

**Primary Task:** Fix documentation drift for v1 and add a lightweight automated drift check so CI catches future mismatches.

**Why now:** Current docs reference removed files/APIs (notably ingestion) and weaken trust. Shipping v1 requires docs to be correct.

**Definition of Done (DoD):**

- `docs/specs/spec-002-ingestion-pipeline.md` references real ingestion modules/APIs.
- `docs/specs/spec-012-observability.md` matches the current `ObservabilityConfig` and `src/telemetry/opentelemetry.py` behavior.
- `docs/developers/developer-handbook.md` no longer references placeholder ingestion functions.
- `docs/developers/system-architecture.md` reflects actual modules.
- A drift checker exists and runs in quality gates.
- RTM updated (NFR-MAINT-003 planned → implemented).

**In-scope modules/files (initial):**

- `docs/specs/spec-002-ingestion-pipeline.md`
- `docs/specs/spec-012-observability.md`
- `docs/developers/developer-handbook.md`
- `docs/developers/system-architecture.md`
- `docs/specs/traceability.md`
- `docs/specs/requirements.md` (add NFR-OBS if referenced by SPEC-012)
- `scripts/test_health.py` or a new `scripts/check_docs_drift.py`
- `scripts/run_quality_gates.py` (if wiring is needed)

**Out-of-scope (explicit):**

- Writing brand new tutorials.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Identify drift: `rg "src/processing/document_processor.py|process_document\\(" docs/`.
2. [ ] Update `docs/specs/spec-002-ingestion-pipeline.md` to match the canonical ingestion pipeline + API.
3. [ ] Update `docs/developers/developer-handbook.md` ingestion examples to the canonical API (SPEC-026).
4. [ ] Update `docs/developers/system-architecture.md` to reflect actual modules.
5. [ ] Update `docs/specs/spec-012-observability.md` to match code and ensure SRS has referenced NFR-OBS requirements.
6. [ ] Implement drift checker:
   - scan non-archived docs for `src/<...>.py` references
   - fail if referenced files don’t exist
7. [ ] Wire drift checker into quality gates.
8. [ ] Update RTM row and run quality gates.

Commands:

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run pylint --fail-under=9.5 src/ tests/ scripts/
uv run python scripts/run_tests.py --fast
uv run python scripts/run_quality_gates.py --ci --report
```

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes                    |
| ----------- | ------ | -------------------------------- |
| Docs        |        | no references to missing modules |
| Drift check |        | runs in quality gates            |
| RTM         |        | updated                          |

**EXECUTE UNTIL COMPLETE.**
