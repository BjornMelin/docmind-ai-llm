# Implementation Prompt — Keyword Tool (Sparse-only Qdrant)

Implements `ADR-044` + `SPEC-025`.

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
