# Implementation Prompt — Ingestion API + Legacy Facade Cleanup

Implements `ADR-045` + `SPEC-026`.

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

**Primary Task:** Replace `src/utils/document.py` placeholders with a canonical ingestion API under `src/processing/` and keep `src.utils.document` as a thin forwarding facade (no duplicate ingestion logic).

**Why now:** `src/utils/document.py` currently raises `NotImplementedError` and contains multiple TODOs; docs/tests reference these functions. This is a v1 ship blocker and undermines trust in the repo.

**Definition of Done (DoD):**

- No `TODO(...)` or `NotImplementedError` remains in `src/utils/document.py` or `src/processing/__init__.py`.
- A typed canonical API exists at `src/processing/ingestion_api.py` and is used by the facade.
- Directory ingestion blocks symlink traversal and is deterministic.
- `clear_document_cache()` only touches `settings.cache_dir / "ingestion"` and is safe.
- Unit tests cover path validation, hashing-based IDs, and facade behavior.
- RTM updated: FR-024 planned → implemented.

**In-scope modules/files (initial):**

- `src/processing/ingestion_api.py` (new)
- `src/processing/__init__.py`
- `src/utils/document.py`
- `tests/unit/processing/` (new tests)
- `tests/unit/utils/document/` (new/updated tests)
- `docs/specs/spec-026-ingestion-api-facade.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Large docs rewrites (handled in WP08).
- Adding remote ingestion sources (HTTP/S3).
- spaCy model downloading (offline-first).

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Inspect current ingestion pipeline (`src/processing/ingestion_pipeline.py`) and Streamlit adapter (`src/ui/_ingest_adapter_impl.py`) to reuse stable-ID and config patterns.
2. [ ] Implement `src/processing/ingestion_api.py`:
   - deterministic path collection for directories
   - symlink traversal prevention
   - streaming SHA-256 hashing + `document_id = doc-<sha[:16]>`
   - config construction from `settings`
   - `ingest_paths` + `ingest_paths_sync`
3. [ ] Update `src/processing/__init__.py`:
   - remove placeholder TODO
   - export canonical API functions
4. [ ] Replace `src/utils/document.py` stubs with forwarding facade:
   - preserve names/signatures where feasible
   - emit `DeprecationWarning` on call
   - implement safe cache clear/stats constrained to `settings.cache_dir / "ingestion"`
5. [ ] Add/adjust unit tests to cover:
   - symlink rejection
   - deterministic ordering
   - hashing-based IDs stable
   - facade no longer raises
6. [ ] Update RTM + run quality gates.

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

1. Duplicate ingestion logic in UI + processing + utils.
2. “Accept any path” ingestion without symlink checks.
3. Hashing based on filenames/mtimes only (must hash bytes for stable IDs).
4. Cache clear that deletes outside `settings.cache_dir`.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes             |
| ----------- | ------ | ------------------------- |
| Formatting  |        | `ruff format`             |
| Lint        |        | `ruff check` clean        |
| Types       |        | `pyright` clean           |
| Pylint      |        | meets threshold           |
| Tests       |        | fast tier green           |
| Docs        |        | SPEC/RTM updated          |
| Security    |        | symlink traversal blocked |

**EXECUTE UNTIL COMPLETE.**
