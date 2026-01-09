# Implementation Prompt — Settings UI Hardening + Safe Provider Badge

Implements `ADR-041` + `SPEC-022`.

## Tooling & Skill Strategy (fresh Codex sessions)

**Use skill:** `$streamlit-master-architect`

Load and follow its workflows for:
- rerun discipline + `st.session_state` correctness
- AppTest patterns
- security-by-default (no unsafe HTML)

Skill references to consult (as needed):
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/security.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/testing_apptest.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/widget_keys_and_reruns.md`

**Streamlit preflight (version + docs + audit):**

```bash
uv run python -c "import streamlit as st; print(st.__version__)"
uv run python /home/bjorn/.codex/skills/streamlit-master-architect/scripts/audit_streamlit_project.py --root . --format md
uv run python /home/bjorn/.codex/skills/streamlit-master-architect/scripts/sync_streamlit_docs.py --out /tmp/streamlit-docs
```

**MCP tool sequence (use when it adds signal):**

1. `functions.mcp__zen__planner` → plan the UI + tests steps.
2. Context7 API verification:
   - `functions.mcp__context7__resolve-library-id` → `streamlit`, `pydantic`, `python-dotenv`
   - `functions.mcp__context7__query-docs` → `st.badge`, `st.form_submit_button`, `st.session_state`, dotenv `set_key/unset_key`
3. Exa (official docs only) for Streamlit forms + best practices if uncertain.
4. `functions.mcp__zen__secaudit` → validate no new injection/logging sinks.

**opensrc (only if subtle behavior):**

```bash
cat opensrc/sources.json | rg -n "python-dotenv|streamlit|pydantic" || true
npx opensrc pypi:python-dotenv
```

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

**Primary Task:** Harden the Streamlit Settings page with pre-validation and safe `.env` persistence, and replace the unsafe provider badge HTML with Streamlit-native UI.

**Why now:** Current Settings can persist invalid `.env` values and includes an avoidable XSS-class sink via `unsafe_allow_html=True`. This is a ship blocker for v1.

**Definition of Done (DoD):**

- `src/ui/components/provider_badge.py` contains **no** `unsafe_allow_html=True` usage.
- Settings Save/Apply validate a candidate settings payload before mutating global `settings` or writing `.env`.
- `.env` persistence uses `python-dotenv` (`set_key`/`unset_key`) and is covered by tests.
- AppTest/pytest coverage verifies invalid settings disable actions and valid settings apply/persist successfully.
- `docs/specs/traceability.md` updated (planned → implemented row for FR-021).

**In-scope modules/files (initial):**

- `src/pages/04_settings.py`
- `src/ui/components/provider_badge.py`
- `tests/integration/test_settings_page.py`
- `docs/developers/adrs/ADR-041-settings-ui-hardening-and-safe-badges.md`
- `docs/specs/spec-022-settings-ui-hardening.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Adding new providers/backends.
- Enabling remote endpoints by default.

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python version must remain **3.11.x** (respect `pyproject.toml`).
- Use **uv only**.

#### 2) Style, Types, and Lint

Must pass:

- `uv run ruff format .`
- `uv run ruff check . --fix`
- `uv run pyright`
- `uv run pylint --fail-under=9.5 src/ tests/ scripts/`

#### 3) Streamlit UI Discipline

- No expensive work at import time.
- Use Streamlit-native components; avoid unsafe HTML.

#### 4) Config Discipline

- Validate using `DocMindSettings` before persistence.
- Do not scatter `os.getenv` in UI code.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Baseline scan: confirm only `src/ui/components/provider_badge.py` uses `unsafe_allow_html=True`.

   - Run: `rg -n \"unsafe_allow_html\\s*=\\s*True\" -S src`

2. [ ] Replace provider badge with Streamlit-native UI (`st.badge` + `st.caption`).

   - Files: `src/ui/components/provider_badge.py`
   - Run: `uv run ruff format src/ui/components/provider_badge.py`

3. [ ] Implement Settings candidate validation (pre-validate before apply/save).

   - Files: `src/pages/04_settings.py`
   - Use `DocMindSettings.model_validate(...)` to validate a candidate payload.

4. [ ] Replace custom `.env` writer with python-dotenv `set_key`/`unset_key`.

   - Files: `src/pages/04_settings.py`

5. [ ] Update/add tests (AppTest + unit) for settings validation and persistence.

   - Files: `tests/integration/test_settings_page.py` (+ new files if needed)

6. [ ] Update RTM and verify quality gates.

   - Files: `docs/specs/traceability.md`
   - Run:

     ```bash
     uv run ruff format .
     uv run ruff check . --fix
     uv run pyright
     uv run pylint --fail-under=9.5 src/ tests/ scripts/
     uv run python scripts/run_tests.py --fast
     ```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. `unsafe_allow_html=True` rendering of dynamic content.
2. Persisting `.env` without Pydantic validation first.
3. Silent exception swallowing around Apply/Save.
4. Import-time IO in Streamlit pages/components.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes                      |
| ----------- | ------ | ---------------------------------- |
| Packaging   |        | `uv sync` clean                    |
| Formatting  |        | `ruff format`                      |
| Lint        |        | `ruff check` clean                 |
| Types       |        | `pyright` clean                    |
| Pylint      |        | meets threshold                    |
| Tests       |        | settings tests green               |
| Docs        |        | ADR/SPEC/RTM updated               |
| Security    |        | no unsafe HTML; allowlist enforced |
| Tech Debt   |        | no TODO/FIXME introduced           |

**EXECUTE UNTIL COMPLETE.**
