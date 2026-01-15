---
prompt: PROMPT-026
title: Ingestion API Cleanup
date: 2026-01-15
version: 2.0
related_adrs: ["ADR-045"]
related_specs: ["SPEC-026"]
---
<!--
Implemented-by: DocMind Agent
Date: 2026-01-15
Version: 2.0
-->

## Implementation Prompt — Unified Ingestion API (Refactor)

Implements `ADR-045` + `SPEC-026`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-045-ingestion-api-and-legacy-facade.md`
- SPEC: `docs/specs/spec-026-ingestion-api-facade.md` (Version 2.0 - Strict Unification)
- RTM: `docs/specs/traceability.md`
- Requirements: `docs/specs/requirements.md` (FR-024, NFR-SEC-001)

## Official docs (research during implementation)

- <https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/> — LlamaIndex ingestion overview.
- <https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/> — Document/node concepts.
- <https://docs.unstructured.io/> — Unstructured docs (partitioning strategies).
- <https://docs.python.org/3/library/pathlib.html> — PathLib best practices.

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` / `ls` for rigorous path validation and verifying cleanup.
- Context7 for authoritative API signatures (LlamaIndex, Unstructured).
- `mv` (shell) for moving test files.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Locate legacy code to extract:
  - `rg -n "load_documents" src/utils/document.py`
  - `ls -F src/utils/document.py src/processing/`
- Locate all call sites to update:
  - `rg -n "src\.utils\.document" src tests`
- Check existing tests to move:
  - `ls tests/unit/utils/document/`

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for LlamaIndex/unstructured resources.
- `functions.read_mcp_resource` → prefer local resources before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs`
- Context7: `functions.mcp__context7__query-docs` → `llama-index`

**Security gate (required):**

- `functions.mcp__zen__secaudit` must cover:
  - path traversal and symlink escape (input validation in `collect_paths` / `load_documents`).
  - deterministic ordering of directory ingestion.

---

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### YOU ARE

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes (Refactor/Move)
- tests (Migration/Update)
- documentation updates (RTM)
- **Deletion** of the legacy module

You must keep changes minimal, library-first, and maintainable.

---

### FEATURE CONTEXT (REFACTOR & UNIFY)

**Primary Task:** Consolidate all ingestion loading logic into a single canonical module `src/processing/ingestion_api.py` and **DELETE** `src/utils/document.py`.

**Why now:** ADR-045 v2 mandates a unified "Greenfield" architecture. The current split between `utils` and `processing` is ambiguous and technical debt. We are unifying the stack for a 2026-ready production baseline.

**Definition of Done (DoD):**

- `src/processing/ingestion_api.py` exists and is the **only** place where file loading/path sanitization happens.
- `src/utils/document.py` is **DELETED**.
- All imports of the removed legacy document module in `src/` and `tests/` are updated to `src.processing.ingestion_api` (or `src.processing`).
- `tests/unit/utils/document/` are moved to `tests/unit/processing/` and pass.
- Symlink traversal is explicitly blocked.
- Project passes all linters and tests.

**In-scope modules/files:**

- `src/processing/ingestion_api.py` (New - Extraction Target)
- `src/utils/document.py` (Source - To Be Deleted)
- `src/processing/ingestion_pipeline.py` (Update consumer)
- `src/ui/_ingest_adapter_impl.py` (Update consumer)
- `src/processing/__init__.py` (Export new API)
- `tests/` (Refactor imports and file locations)

**Out-of-scope (explicit):**

- Adding remote ingestion sources (S3/HTTP).
- Changing the behavior of `UnstructuredReader` (keep existing logic, just move it).

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python version must remain **3.11.x**.
- Use **uv only**:
  - `uv sync`
  - `uv run <cmd>`

#### 2) Style, Types, and Lint

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright` (Must be clean after refactor)

#### 3) Security & Offline-first

- **Symlinks**: `path.resolve(strict=True)` must not resolve to an external path or be a symlink itself if `follow_symlinks=False`. Block traversal.
- **Cache**: `clear_ingestion_cache` must only delete from `settings.cache_dir / "ingestion"`.

---

### STEP-BY-STEP EXECUTION PLAN

You MUST produce a plan and keep exactly one step “in_progress” at a time.

1. [x] **Inspect & Plan**: Verify file paths and existing logic in `src/utils/document.py`.
2. [x] **Create Canonical API**:
   - Create `src/processing/ingestion_api.py`.
   - **Extract/Move** logic from `src/utils/document.py` (copy first).
   - Ensure imports are updated relative to new location.
   - Refine types and docstrings.
3. [x] **Refactor Consumers**:
   - Update `src/processing/ingestion_pipeline.py` to use `src.processing.ingestion_api`.
   - Update `src/ui/_ingest_adapter_impl.py`.
   - Update `src/utils/__init__.py` (remove exports) and `src/processing/__init__.py` (add exports).
4. [x] **Migrate Tests**:
   - Move `tests/unit/utils/document/*.py` to `tests/unit/processing/`.
   - Update imports in all tests.
   - Verify tests pass: `uv run python scripts/run_tests.py`.
5. [x] **Delete Legacy**:
   - Delete `src/utils/document.py`.
   - Remove `src/utils/document.py` references from any other files (check with `rg`).
6. [x] **Docs & Cleanup**:
   - **Update References**: The following files reference the removed legacy document module and MUST be updated:
     - `docs/developers/developer-handbook.md` (mock patches)
     - `docs/developers/system-architecture.md` (imports/diagrams)
     - `docs/developers/testing-notes.md`
     - `docs/specs/spec-002-ingestion-pipeline.md`
     - `docs/specs/spec-029-docs-consistency-pass.md`
   - **Update Metadata**:
     - `docs/developers/adrs/ADR-045-ingestion-api-and-legacy-facade.md`: Set `Status: Implemented`.
     - `docs/specs/spec-026-ingestion-api-facade.md`: Set `status: Implemented`.
   - **Traceability**: Update `docs/specs/traceability.md` (FR-024) to `Implemented`.
   - **Archive Prompt**: Move this file to `docs/developers/prompts/implemented/` and prepend metadata:

     ```markdown
     <!--
     Implemented-by: [Agent Name]
     Date: YYYY-MM-DD
     Version: [Resulting Version]
     -->
     ```

**Commands (required):**

```bash
uv sync
uv run ruff format .
uv run ruff check .
uv run pyright
uv run python scripts/run_tests.py
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. **Facade/Shim**: Do NOT leave `src/utils/document.py` behind as a proxy. Delete it.
2. **Circular Imports**: Watch out when moving logic; if `ingestion_api` imports `ingestion_pipeline`, ensure `ingestion_pipeline` doesn't import `ingestion_api` in a way that cycles. (Input -> Pipeline is the correct flow).
3. **Typos in Refactor**: Use IDE tools or careful search-replace to ensure `load_documents_unstructured` calls are correctly updated.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement     | Status | Proof / Notes                                                                 |
| --------------- | ------ | ----------------------------------------------------------------------------- |
| **Packaging**   | Done   | `uv sync`                                                                     |
| **Formatting**  | Done   | `uv run ruff format .`                                                        |
| **Lint**        | Done   | `uv run ruff check .`                                                         |
| **Types**       | Done   | `uv run pyright --threads 4`                                                  |
| **Tests**       | Done   | `uv run python scripts/run_tests.py`                                          |
| **Refactor**    | Done   | `src/utils/document.py` deleted                                               |
| **Docs**        | Done   | `docs/` updated (no legacy import-path refs outside ADRs)                     |
| **Prompt**      | Done   | Archived under `docs/developers/prompts/implemented/`                         |
| **Security**    | Done   | Symlinks blocked; validation centralized in `src/processing/ingestion_api.py` |

**EXECUTE UNTIL COMPLETE.**
