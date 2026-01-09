# Implementation Prompt — Ingestion API + Legacy Facade Cleanup

Implements `ADR-045` + `SPEC-026`.

**Read first (repo truth):**
- ADR: `docs/developers/adrs/ADR-045-ingestion-api-and-legacy-facade.md`
- SPEC: `docs/specs/spec-026-ingestion-api-facade.md`
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/ — LlamaIndex ingestion pipeline overview.
- https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/ — Document/node concepts and file readers.
- https://docs.unstructured.io/ — Unstructured docs (partitioning strategies; offline-first cautions).
- https://pymupdf.readthedocs.io/ — PyMuPDF docs (PDF parsing/rendering primitives used by the repo).
- https://docs.python.org/3/library/pathlib.html — `pathlib` path handling patterns (safe path joins, normalization).

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` for placeholder/legacy discovery (TODOs, NotImplemented stubs, old doc references).
- Context7 for authoritative API signatures (LlamaIndex IngestionPipeline, UnstructuredReader, Pydantic models).
- Exa for official LlamaIndex ingestion guidance if behavior is unclear.
- `opensrc/` to confirm internal behavior (LlamaIndex + unstructured) when subtle (caching, docstore persist).

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Locate placeholders and call sites:
  - `rg -n \"(NotImplementedError|ingestion-phase-2|load_documents_|clear_document_cache)\" -S src tests`
  - `rg -n \"src\\.utils\\.document\" -S src tests`
- Read in parallel:
  - `src/utils/document.py`
  - `src/processing/ingestion_pipeline.py`
  - `src/ui/_ingest_adapter_impl.py` (if present; reuse stable patterns)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for LlamaIndex/unstructured docs resources.
- `functions.read_mcp_resource` → prefer local resources before web search.

**API verification (Context7):**

- `functions.mcp__context7__resolve-library-id` → `llama-index` (and optionally `unstructured`)
- `functions.mcp__context7__query-docs` → confirm:
  - ingestion pipeline + file reader APIs you plan to use
  - any recommended patterns for deterministic ingestion / caching

**Security gate (required):**

- `functions.mcp__zen__secaudit` must cover:
  - path traversal and symlink escape prevention
  - directory ingestion determinism
  - offline-first posture (no implicit network calls)

**Review gate (recommended):**

- `functions.mcp__zen__codereview` after implementation to ensure one canonical ingestion API and no legacy code paths remain.

**MCP tool sequence (use when it adds signal):**

1. `functions.mcp__zen__planner` → plan: new canonical API + facade + tests + doc updates.
2. Context7:
   - resolve `llama-index` (and optionally `unstructured`) and query docs for ingestion pipeline + readers
3. `functions.mcp__zen__secaudit` → file/path traversal protection + symlink blocking (directory ingestion).
4. `functions.mcp__zen__codereview` after implementation to ensure the facade is thin and no legacy paths remain.

**opensrc (recommended):**

```bash
cat opensrc/sources.json | rg -n "llama-index|unstructured" || true
# Fetch only if missing and behavior is surprising; treat opensrc/ as read-only.
npx opensrc pypi:llama-index
npx opensrc pypi:unstructured
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

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

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
uv run python scripts/run_tests.py
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Duplicate ingestion logic in UI + processing + utils.
2. “Accept any path” ingestion without symlink checks.
3. Hashing based on filenames/mtimes only (must hash bytes for stable IDs).
4. Cache clear that deletes outside `settings.cache_dir`.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (API + facade + tests).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → official LlamaIndex/unstructured docs if subtle behavior is unclear.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API details (`llama-index`, optional `unstructured`).
4. `functions.mcp__gh_grep__searchGitHub` → real-world ingestion/facade patterns (optional).
5. `functions.mcp__zen__analyze` → only if multiple ingestion paths emerge; keep one canonical API.
6. `functions.mcp__zen__codereview` → post-implementation review (ensure facade is thin; no legacy path remains).
7. `functions.mcp__zen__secaudit` → required security audit (path traversal, symlink escape).

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes |
|---|---|---|
| **Packaging** |  | `uv sync` clean |
| **Formatting** |  | `uv run ruff format .` |
| **Lint** |  | `uv run ruff check .` clean |
| **Types** |  | `uv run pyright` clean |
| **Pylint** |  | meets threshold |
| **Tests** |  | `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs** |  | SPEC/RTM updated |
| **Security** |  | symlink traversal blocked; cache clear constrained to `settings.cache_dir` |
| **Tech Debt** |  | zero TODO/FIXME introduced; no placeholder stubs remain |
| **Performance** |  | hashing is streaming; no import-time heavy work |

**EXECUTE UNTIL COMPLETE.**
