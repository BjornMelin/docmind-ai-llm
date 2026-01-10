# Implementation Prompt — Docs Consistency Pass (Specs/Handbook/RTM)

Implements `ADR-048` + `SPEC-029`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-048-docs-consistency-pass.md`
- SPEC: `docs/specs/spec-029-docs-consistency-pass.md`
- Requirements: `docs/specs/requirements.md` (NFR-MAINT-003)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.streamlit.io/develop> — Streamlit official docs index (use when docs reference Streamlit APIs).
- <https://docs.llamaindex.ai/en/stable/> — LlamaIndex stable docs index (use when docs reference ingestion/retrieval/memory).
- <https://qdrant.tech/documentation/> — Qdrant docs index (hybrid queries, query points API).
- <https://opentelemetry.io/docs/languages/python/> — OpenTelemetry Python docs (when aligning observability specs).

## Tooling & Skill Strategy (fresh Codex sessions)

Docs drift is best solved with repo-local tooling + grep first.

**Read first:** `docs/developers/prompts/README.md` and `${CODEX_PROMPT_LIBRARY:-$HOME/prompt_library}/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` for finding stale references and validating replacements.
- Use `multi_tool_use.parallel` for independent drift scans across docs/specs/scripts.
- Prefer `opensrc/` only when a doc claim depends on library internals (rare).
- Exa/Context7 only when a doc section references subtle library behavior that must be correct.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel drift scan (use `multi_tool_use.parallel`):**

- Identify file references and drift hotspots:
  - `rg -n \"src/[^\\s\\)\\]\\}]+\\.py\" docs -S`
  - `rg -n \"(NotImplementedError|ingestion-phase-2|phase 2|placeholder)\" docs -S`
  - `rg -n \"spec-012-observability|ObservabilityConfig\" docs src -S`
- Use scripts where possible:
  - `uv run python scripts/test_health.py` (if it already checks TODO/drift patterns)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → read any local docs indexes/templates for consistency policies.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc`
- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain`
- OpenAI API docs: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (only if updating docs that cite OpenAI API semantics)

**API verification (only when needed):**

- Context7 (`functions.mcp__context7__resolve-library-id` → `functions.mcp__context7__query-docs`) for subtle library behavior referenced by docs.
- Web tools (`functions.mcp__exa__web_search_exa` / `web.run`) only if you must cite a current upstream behavior or changelog.

**Review gate (recommended):**

- `functions.mcp__zen__codereview` after drift checker wiring to reduce false positives/negatives.

**MCP tool sequence (optional):**

1. `functions.mcp__zen__planner` → scope the doc drift changes + drift-checker script changes.
2. `functions.mcp__zen__codereview` → sanity check after drift checker is added (avoid false positives).

## IMPLEMENTATION EXECUTOR TEMPLATE (DOCMIND / PYTHON)

### YOU ARE

You are an autonomous implementation agent for the **DocMind AI LLM** repository.

You will implement the feature described below end-to-end, including:

- code changes (scripts/checkers as needed)
- tests
- documentation updates (SPEC/RTM)
- deletion of dead code and removal of legacy/backcompat shims within scope

You must keep changes minimal, library-first, and maintainable.

---

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

### HARD RULES (EXECUTION)

#### 1) Docs truth discipline

- Prefer repo-truth over prose: verify paths with `rg`, validate configs against `src/config/settings.py`, and keep examples runnable.
- Avoid “future work” placeholders; delete stale placeholders instead of rewording them.

#### 2) Packaging + quality gates

- Use **uv only**: `uv sync`, `uv run ...`
- If you add a drift-checker script, wire it into existing quality gates in `scripts/run_quality_gates.py`.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Identify drift: `rg "src/processing/document_processor.py|process_document\\(" docs/`.
2. [ ] Update `docs/specs/spec-002-ingestion-pipeline.md` to match the canonical ingestion pipeline + API.
3. [ ] Update `docs/developers/developer-handbook.md` ingestion examples to the canonical API (SPEC-026).
4. [ ] Update `docs/developers/system-architecture.md` to reflect actual modules.
5. [ ] Update `docs/specs/spec-012-observability.md` to match code and ensure SRS has referenced NFR-OBS requirements.
6. [ ] Implement drift checker:
   - scan non-archived docs for `src/<...>.py` references
   - fail if referenced files don’t exist
   - classify findings:
     - hard failures: direct code refs to missing `src/...` paths
     - soft warnings: historical examples, comments, or external docs references
     - allowlist: intentional external cross-references
   - report severity and track false positives (keep soft warnings non-blocking)
7. [ ] Wire drift checker into quality gates.
8. [ ] Update RTM row and run quality gates.

Commands:

```bash
uv sync
uv run ruff format .
uv run ruff check . --fix
uv run pyright
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py
uv run python scripts/run_quality_gates.py --ci --report
```

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

1. Docs referencing non-existent files/modules (must be caught by the drift checker).
2. “Phase 2”/placeholder claims for features that are not shipped.
3. Drift checker that is too strict (false positives) or too lax (misses broken references).
4. Web citations for stable facts that are already derivable from repo truth (prefer local).

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → plan doc changes + drift checker scope.
2. `functions.mcp__exa__web_search_exa` / `web.run` → only if a doc section depends on time-sensitive upstream behavior.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → only if docs must mention subtle library behavior precisely.
4. `functions.mcp__gh_grep__searchGitHub` → generally unnecessary.
5. `functions.mcp__zen__analyze` → optional if drift touches architecture boundaries.
6. `functions.mcp__zen__codereview` → recommended to avoid checker false positives/negatives.
7. `functions.mcp__zen__secaudit` → only if docs change security-sensitive guidance.

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local drift scans (`rg`) and parallel file reads.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement     | Status | Proof / Notes                                                                      |
| --------------- | ------ | ---------------------------------------------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                                                    |
| **Formatting**  |        | `uv run ruff format .` (if Python scripts changed)                                 |
| **Lint**        |        | `uv run ruff check .` clean                                                        |
| **Types**       |        | `uv run pyright` clean                                                             |
| **Tests**       |        | `uv run python scripts/run_tests.py --fast` + `uv run python scripts/run_tests.py` |
| **Docs**        |        | no references to missing modules                                                   |
| **Security**    |        | security guidance matches repo posture (offline-first)                             |
| **Tech Debt**   |        | zero TODO/FIXME introduced                                                         |
| **Performance** |        | drift checker runs fast (O(files) scans; no heavy IO)                              |

**EXECUTE UNTIL COMPLETE.**
