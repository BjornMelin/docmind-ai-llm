# Implementation Prompt — Remove Legacy `src/main.py` Entrypoint

Implements `ADR-046` + `SPEC-027`.

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
