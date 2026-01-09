# Implementation Prompt — Documents Snapshot Service Boundary

Implements `ADR-051` + `SPEC-032`.

## Tooling & Skill Strategy (fresh Codex sessions)

This is a cross-cutting refactor (UI → persistence boundary). Use analysis + review tools.

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Use skill:** `$streamlit-master-architect` (for the Documents page wiring + AppTest), but keep the service boundary Streamlit-free.

Skill references to consult (as needed):
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/testing_apptest.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/caching_and_fragments.md`

**Primary tools to leverage:**

- `rg` to locate all snapshot rebuild/export code paths and tests.
- Context7 for any subtle LlamaIndex persistence APIs (if needed).
- `functions.mcp__zen__analyze` before refactor to avoid accidental behavior changes.
- `functions.mcp__zen__codereview` after refactor to ensure the boundary is clean and tests moved correctly.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Inventory call sites + tests:
  - `rg -n \"rebuild_snapshot|SnapshotManager|manifest\\.jsonl|graph_exports\" -S src tests`
  - `rg -n \"02_documents\\.py\" -S src tests docs`
- Read in parallel:
  - `src/pages/02_documents.py`
  - `src/persistence/snapshot_manager.py` (and related persistence modules)
  - `src/retrieval/graph_config.py` (exports packaging, if involved)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local “persistence/snapshot” resources before web search.

**Architecture gate (required):**

- Run `functions.mcp__zen__analyze` before the refactor to document current behavior boundaries and identify duplication.

**Long-running UI validation (use native capabilities):**

- If you run `streamlit run src/app.py`, keep it running and use `functions.write_stdin` to fetch logs.
- Attach screenshots of the Documents page state changes with `functions.view_image` if verification is visual.
- Optional E2E smoke (skill script): `/home/bjorn/.codex/skills/streamlit-master-architect/scripts/mcp/run_playwright_mcp_e2e.py`

**Review gate (required):**

- Run `functions.mcp__zen__codereview` after tests pass to ensure the service boundary is clean and Streamlit-free.

**opensrc (optional):**

Use only if you must understand LlamaIndex persistence internals; prefer repo code + tests first.

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

**Primary Task:** Extract snapshot rebuild + GraphRAG export packaging logic from `src/pages/02_documents.py` into a persistence-layer service module (`src/persistence/snapshot_service.py`) and keep the Documents page as UI wiring only.

**Why now:** The Documents page currently embeds domain logic (snapshot lifecycle, hashing, export metadata) which is hard to test and easy to regress during UI changes. A service boundary improves correctness and maintainability for v1.

**Definition of Done (DoD):**

- New module `src/persistence/snapshot_service.py` exists and contains the canonical snapshot rebuild orchestration.
- `src/pages/02_documents.py` no longer contains the full `rebuild_snapshot` implementation (a thin wrapper delegating to the service is acceptable temporarily for UI wiring).
- Unit tests cover snapshot rebuild/export metadata via the service module (not via page-local logic).
- AppTest integration test stubs/patches the service boundary (not page-internal exports).
- RTM updated: `FR-009` references the new service module.

**In-scope modules/files (initial):**

- `src/persistence/snapshot_service.py` (new)
- `src/pages/02_documents.py`
- `tests/unit/ui/test_documents_snapshot_utils.py` (move/update)
- `tests/integration/ui/test_documents_snapshot_button.py`
- `docs/developers/adrs/ADR-051-documents-snapshot-service-boundary.md`
- `docs/specs/spec-032-documents-snapshot-service-boundary.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Background ingestion/progress/cancellation (handled separately).
- Changing snapshot schema or lock semantics.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Identify the snapshot rebuild/export logic currently living in `src/pages/02_documents.py` and the tests that assert it.
2. [ ] Implement `src/persistence/snapshot_service.py`:
   - typed result model (dataclass or Pydantic)
   - snapshot workspace lifecycle + manifest writing
   - packaged `graph_exports` metadata hashing (sha256)
3. [ ] Refactor `src/pages/02_documents.py` to call the service and only render results/errors.
4. [ ] Update tests:
   - move unit tests to cover `src/persistence/snapshot_service.py`
   - update AppTest integration to patch the service boundary instead of per-export functions
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

1. Snapshot persistence logic embedded in Streamlit pages.
2. Import-time heavy dependencies in `src/pages/*` that break smoke tests.
3. Broad exception swallowing that hides snapshot corruption.
4. Non-deterministic export metadata (timestamps only; missing sha256).

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes         |
| ----------- | ------ | --------------------- |
| Formatting  |        | `ruff format`         |
| Lint        |        | `ruff check` clean    |
| Types       |        | `pyright` clean       |
| Pylint      |        | meets threshold       |
| Tests       |        | fast tier green       |
| Docs        |        | ADR/SPEC/RTM updated  |
| Security    |        | no new write surfaces |

**EXECUTE UNTIL COMPLETE.**
