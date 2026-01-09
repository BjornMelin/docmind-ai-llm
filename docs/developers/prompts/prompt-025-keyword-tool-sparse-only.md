# Implementation Prompt — Keyword Tool (Sparse-only Qdrant)

Implements `ADR-044` + `SPEC-025`.

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
```

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes            |
| ----------- | ------ | ------------------------ |
| Tests       |        | keyword tool tests green |
| Security    |        | no raw query logs        |
| Docs        |        | RTM updated              |

**EXECUTE UNTIL COMPLETE.**
