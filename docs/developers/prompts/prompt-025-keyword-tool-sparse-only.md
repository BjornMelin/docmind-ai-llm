# Implementation Prompt — Keyword Tool (Sparse-only Qdrant)

Implements `ADR-044` + `SPEC-025`.

**Read first (repo truth):**
- ADR: `docs/developers/adrs/ADR-044-keyword-tool-sparse-only-qdrant.md`
- SPEC: `docs/specs/spec-025-keyword-tool-sparse-only.md`
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- https://qdrant.tech/documentation/concepts/search/#query-points — Qdrant Query Points API (sparse vectors, filters, payloads).
- https://qdrant.tech/documentation/concepts/hybrid-queries/ — Hybrid query concepts (dense+sparse fusion).
- https://python-client.qdrant.tech/ — Qdrant Python client docs (types, requests).
- https://docs.llamaindex.ai/en/stable/api_reference/vector_stores/qdrant/ — LlamaIndex Qdrant vector store integration.
- https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/ — LlamaIndex retriever patterns and contracts.

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Primary tools to leverage:**

- `rg` for local code search and call-site discovery.
- Context7 + Exa for Qdrant/LlamaIndex API verification (especially sparse vectors / named vectors).
- `functions.mcp__gh_grep__searchGitHub` for idiomatic sparse-only query patterns.
- `opensrc/` for inspecting exact dependency behavior (Qdrant client + LlamaIndex) when in doubt.

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Find all current keyword tool references:
  - `rg -n \"create_keyword_tool|keyword_search|keyword tool\" -S src/agents src/retrieval tests`
  - `rg -n \"text-sparse|Sparse|BM25|BM42|fastembed\" -S src/retrieval src/config`
- Read in parallel:
  - `src/agents/tool_factory.py`
  - `src/retrieval/sparse_query.py`
  - any Qdrant client helpers (`src/retrieval/*qdrant*`, `src/config/*qdrant*`)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for Qdrant/LlamaIndex resources; read them before web search.

**API correctness (Context7 + web):**

- `functions.mcp__context7__resolve-library-id` → `qdrant-client`, `llama-index`
- `functions.mcp__context7__query-docs` → confirm:
  - named vector and sparse vector payload shapes
  - Query API / `Prefetch` usage patterns (if used)
- Use `functions.mcp__exa__deep_search_exa` (or `web.run` when you need citations/dates) for Qdrant “Query API sparse vector” docs and recent changes.

**Real-world patterns (GitHub grep):**

- `functions.mcp__gh_grep__searchGitHub` for patterns like `Prefetch(`, `SparseVector(`, `NamedVector`, `QueryRequest(`.

**Architecture gate (recommended):**

- Use `functions.mcp__zen__analyze` before implementing if you find multiple retrieval paths; keep one canonical sparse-query helper.

**Security gate (required):**

- `functions.mcp__zen__secaudit` must confirm no raw query text or document content is emitted to logs/telemetry.

### MCP tool sequence (use when it adds signal)

1. `functions.mcp__zen__planner` → plan: new retriever + tool wiring + tests.
2. Context7:
   - resolve `qdrant-client` and `llama-index`
   - query docs for Query API / named vectors usage and payload formats
3. Exa search for “Qdrant Query API prefetch sparse named vector text-sparse” (official sources preferred).
4. `functions.mcp__gh_grep__searchGitHub` for patterns like `Prefetch(` / `FusionQuery(` / `SparseVector`.
5. `functions.mcp__zen__secaudit` → confirm no query text/PII is logged in telemetry.

**opensrc (recommended):**

```bash
cat opensrc/sources.json | rg -n "qdrant-client|llama-index" || true
# Fetch only if missing and behavior is surprising; treat opensrc/ as read-only.
npx opensrc pypi:qdrant-client
npx opensrc pypi:llama-index
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

**Primary Task:** Replace the placeholder `keyword_search` tool with a real implementation using sparse-only Qdrant queries against the `text-sparse` named vector.

**Why now:** Tests and code contain a TODO placeholder, and agent routing benefits from a distinct exact-term tool without adding BM25 dependencies.

**Definition of Done (DoD):**

- `create_keyword_tool` no longer uses a vector query engine placeholder.
- A sparse-only retriever exists and is unit-tested.
- Tool registration remains gated behind `settings.retrieval.enable_keyword_tool` (default false).
- RTM updated for FR-023.

**In-scope modules/files (initial):**

- `src/agents/tool_factory.py`
- `src/retrieval/keyword.py` (new)
- `src/retrieval/sparse_query.py` (reuse only; minimal/no changes preferred)
- `tests/unit/agents/test_tool_factory_keyword.py`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- Adding rank_bm25 or other BM25 packages.

---

### HARD RULES (EXECUTION)

- Reuse existing Qdrant client config helpers and sparse encoder caching.
- Fail open when sparse encoder unavailable.
- Do not log raw query text.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

0. [ ] Read ADR/SPEC/RTM and restate DoD in your plan.

1. [ ] Inspect current placeholder implementation and tests.
2. [ ] Implement `src/retrieval/keyword.py` sparse-only retriever.
3. [ ] Wire ToolFactory keyword tool to use the new retriever query engine.
4. [ ] Add/update unit tests (mock qdrant client + sparse encoder).
5. [ ] Run quality gates and update RTM.

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

1. Adding BM25 dependencies (`rank_bm25`, etc.) for this prompt (explicitly out-of-scope).
2. Logging raw query strings or retrieved content in logs/telemetry.
3. Duplicating sparse query logic across multiple modules (prefer one canonical helper).
4. Silent broad `except Exception` fallbacks without telemetry/error classification.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Follow the “Prompt-specific Tool Playbook” above. Use these tools as needed:

1. `functions.mcp__zen__planner` → implementation plan (retriever + tool wiring + tests).
2. `functions.mcp__exa__deep_search_exa` / `functions.mcp__exa__crawling_exa` → official Qdrant Query API docs when subtle behavior matters.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API details (`qdrant-client`, `llama-index`).
4. `functions.mcp__gh_grep__searchGitHub` → real-world sparse/named-vector query patterns.
5. `functions.mcp__zen__analyze` → ensure one canonical sparse-query path.
6. `functions.mcp__zen__codereview` → post-implementation review.
7. `functions.mcp__zen__secaudit` → ensure no raw query/content logging.

Also use `functions.exec_command` + `multi_tool_use.parallel` for repo-local discovery (`rg`) and parallel doc lookups.

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
| **Docs** |  | RTM updated |
| **Security** |  | no raw query logs/content; allowlist posture unchanged |
| **Tech Debt** |  | zero TODO/FIXME introduced |
| **Performance** |  | no new import-time heavy work; bounded query payloads |

**EXECUTE UNTIL COMPLETE.**
