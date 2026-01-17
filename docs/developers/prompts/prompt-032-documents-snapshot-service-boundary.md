# Implementation Prompt — Documents Snapshot Service Boundary

Implements `ADR-051` + `SPEC-032`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-051-documents-snapshot-service-boundary.md`
- SPEC: `docs/specs/spec-032-documents-snapshot-service-boundary.md`
- Requirements: `docs/specs/requirements.md` (NFR-MAINT-001, NFR-MAINT-003)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.python.org/3/library/tempfile.html> — Safe temporary workspace patterns.
- <https://docs.python.org/3/library/hashlib.html> — SHA-256 hashing primitives (stable IDs/manifest hashes).
- <https://docs.llamaindex.ai/en/stable/> — LlamaIndex stable docs index (only if persistence behaviors are subtle).

## Tooling & Skill Strategy (fresh Codex sessions)

This is a cross-cutting refactor (UI → persistence boundary). Use analysis + review tools.

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Note:** Paths that reference `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}` assume a local Codex skill install. Adjust the prefix to your team’s skill library location or a repo-local copy if needed. Codex skills are optional and not part of the DocMind release.

**Adapting without Codex:** If Codex skills are not available:

- Use `rg` + your editor to review `src/pages/02_documents.py` and the persistence modules.
- Read LlamaIndex persistence docs directly from <https://docs.llamaindex.ai/en/stable/> if persistence behavior is unclear.
- Use standard Python tooling (`uv run pyright`, `uv run python scripts/run_tests.py`) without MCP extensions.

**Use skill:** `$streamlit-master-architect` (for the Documents page wiring + AppTest), but keep the service boundary Streamlit-free.

Skill references to consult (as needed):

- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/testing_apptest.md`
- `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/references/caching_and_fragments.md`

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
  - `src/persistence/snapshot.py` (SnapshotManager) plus `snapshot_writer.py` and `snapshot_utils.py`
  - `src/retrieval/graph_config.py` (exports packaging, if involved)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local “persistence/snapshot” resources before web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc` (persistence, storage, node metadata)
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (rare for this package)

**Architecture gate (required):**

- Run `functions.mcp__zen__analyze` before the refactor to document current behavior boundaries and identify duplication.

**Long-running UI validation (use native capabilities):**

- If you run `uv run streamlit run app.py`, keep it running and use `functions.write_stdin` to fetch logs.
- Attach screenshots of the Documents page state changes with `functions.view_image` if verification is visual.
- Optional E2E smoke (skill script): `${CODEX_SKILLS_HOME:-$CODEX_HOME/skills}/streamlit-master-architect/scripts/mcp/run_playwright_mcp_e2e.py`

**Review gate (required):**

- Run `functions.mcp__zen__codereview` after tests pass to ensure the service boundary is clean and Streamlit-free.

**opensrc (local reference only):**

The `opensrc/` directory is a local development artifact (excluded from version control) containing offline snapshots of dependencies. Use it only for quick reference of LlamaIndex internals during development and not for production or critical implementation decisions. Prefer repo code, tests, and official LlamaIndex docs instead.

## **Implementation Executor Template (DocMind / Python)**

### You Are

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes
- tests
- documentation updates (ADR/SPEC/RTM)
- deletion of dead code and removal of legacy/backcompat shims within scope

You must keep changes minimal, library-first, and maintainable.

---

### **Feature Context (Filled)**

**Primary Task:** Extract snapshot rebuild + GraphRAG export packaging logic from `src/pages/02_documents.py` into a persistence-layer service module (`src/persistence/snapshot_service.py`) and keep the Documents page as UI wiring only. `snapshot_service.py` is the orchestration layer that coordinates `SnapshotManager` (`snapshot.py`), `SnapshotWriter` (`snapshot_writer.py`), and pure helpers (`snapshot_utils.py`); state/lifecycle stays in `SnapshotManager`, low-level I/O stays in `SnapshotWriter`, and helpers remain in `snapshot_utils.py`.

**Why now:** The Documents page currently embeds domain logic (snapshot lifecycle, hashing, export metadata) which is difficult to test and easy to regress during UI changes. A service boundary improves correctness and maintainability for v1.

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

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Identify the snapshot rebuild/export logic currently living in `src/pages/02_documents.py` and the tests that assert it.
2. [ ] Implement `src/persistence/snapshot_service.py`:
   - typed result model (dataclass or Pydantic)
   - snapshot workspace lifecycle + manifest writing
   - manifest schema fields: required `index_id`, `graph_store_type`, `vector_store_type`, `corpus_hash`, `config_hash`, `versions`; optional `graph_exports` (with sha256 metadata hash)
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
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
```

---

### **Anti-Pattern Kill List (Immediate Deletion/Rewrite)**

1. Snapshot persistence logic embedded in Streamlit pages.
2. Import-time heavy dependencies in `src/pages/*` that break smoke tests.
3. Broad exception swallowing that hides snapshot corruption.
4. Non-deterministic export metadata (timestamps only; missing sha256).

---

### **MCP Tool Strategy (For Implementation Run)**

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (service boundary + tests).
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → only if subtle LlamaIndex persistence behavior needs confirmation.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API details (LlamaIndex persistence, if needed).
4. `functions.mcp__gh_grep__searchGitHub` → optional patterns for service-layer boundaries around Streamlit apps.
5. `functions.mcp__zen__analyze` → required pre-refactor architecture assessment.
6. `functions.mcp__zen__codereview` → required post-implementation review (boundary cleanliness).
7. `functions.mcp__zen__secaudit` → optional unless new write surfaces are introduced.

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel file reads.

---

### **Final Verification Checklist (Must Complete)**

All items below must be completed (Status = ✓) before the implementation is ready for code review.

| Requirement     | Status | Gate      | Proof / Notes                                                                                                                        |
| --------------- | ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Packaging**   |        | Must pass | `uv sync` clean                                                                                                                      |
| **Formatting**  |        | Must pass | `uv run ruff format .`                                                                                                               |
| **Lint**        |        | Must pass | `uv run ruff check .` clean                                                                                                          |
| **Types**       |        | Must pass | `uv run pyright` clean                                                                                                               |
| **Tests**       |        | Must pass | service + UI wiring tests; `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py`                        |
| **Docs**        |        | Must pass | ADR/SPEC/RTM updated; verify `scripts/performance_monitor.py` details in `scripts/README.md`                                         |
| **Security**    |        | Must pass | snapshot writes remain atomic; no new write surfaces                                                                                 |
| **Tech Debt**   |        | Must pass | zero work-marker placeholders introduced                                                                                             |
| **Performance** |        | Advisory  | service layer keeps Streamlit pages import-light; target ≤10% regression vs. baseline (measure via `scripts/performance_monitor.py`) |

**EXECUTE UNTIL COMPLETE.**
