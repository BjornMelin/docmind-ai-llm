# Implementation Prompt — Release Readiness v1 (Execute All Work Packages)

Use this prompt to implement **every** work package defined in `docs/specs/spec-021-release-readiness-v1.md`.

**Read first (repo truth):**

- SPEC: `docs/specs/spec-021-release-readiness-v1.md`
- Requirements: `docs/specs/requirements.md`
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.streamlit.io/develop/concepts/multipage-apps/overview> — Streamlit multipage fundamentals and routing patterns.
- <https://docs.streamlit.io/develop/api-reference/execution-flow/st.navigation> — Streamlit navigation API used by this repo.
- <https://docs.streamlit.io/develop/api-reference/execution-flow/st.page> — Streamlit page definition API used by this repo.
- <https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest> — Streamlit AppTest reference (for deterministic UI tests).
- <https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/> — LlamaIndex ingestion pipeline overview.
- <https://qdrant.tech/documentation/concepts/hybrid-queries/> — Qdrant hybrid query concepts (dense+sparse).
- <https://qdrant.tech/documentation/concepts/search/#query-points> — Qdrant Query Points API reference (Prefetch/fusion patterns).
- <https://qdrant.tech/documentation/concepts/snapshots/> — Qdrant snapshots (backup/restore).
- <https://docs.python.org/3/library/sqlite3.html> — stdlib sqlite3 (ops metadata store, migrations, WAL).
- <https://www.sqlite.org/wal.html> — WAL mode semantics and concurrency tradeoffs.
- <https://langchain-ai.github.io/langgraph/how-tos/streaming/> — LangGraph streaming (supervisor loop integration).
- <https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph/interrupts> — LangGraph interrupts/stop patterns (cooperative cancellation).
- <https://docs.astral.sh/uv/guides/integration/docker/> — `uv` Docker build guidance (reproducible installs).
- <https://docs.astral.sh/ruff/formatter/> — Ruff formatter reference (repo uses `ruff format`).
- <https://docs.astral.sh/ruff/linter/> — Ruff linter reference (repo uses `ruff check`).
- <https://github.com/microsoft/pyright/blob/main/docs/configuration.md> — Pyright configuration reference.
- <https://docs.docker.com/build/building/best-practices/> — Docker build best practices.
- <https://opentelemetry.io/docs/languages/python/> — OpenTelemetry Python docs (optional in this repo).
- <https://docs.llamaindex.ai/en/stable/module_guides/storing/chat_stores/> — LlamaIndex chat store persistence patterns.

## Tooling & Skill Strategy (fresh Codex sessions)

**Skills to load during execution (by package):**

- `$streamlit-master-architect` for any Streamlit page/UI work (WP01, WP03, WP11–WP13).
- `$streamlit-master-architect` also applies to WP14 (analysis modes) and any Streamlit-side UX work for WP15/16.
- `$docker-architect` for container/compose work (WP02).

**Mandatory preflight in each new session:**

```bash
cd /home/bjorn/repos/agents/docmind-ai-llm
rg -n "\\b(TODO|FIXME|XXX)\\b" src tests docs scripts tools || true
uv run python -c "import streamlit as st; print(st.__version__)"
```

### Required tool inventory

Use the repo’s tool inventory guidance:

- `docs/developers/prompts/README.md` (tool inventory + opensrc rules)
- `~/prompt_library/assistant/codex-inventory.md` (complete MCP + skill list)

### Parallelization rules (mandatory)

When you have independent discovery or research steps, batch them:

- Use `multi_tool_use.parallel` for independent function calls (multiple searches, multiple Context7 resolves, multiple shell reads).
- Avoid serial “one tool call at a time” loops when results don’t depend on each other.

Examples of “parallel-safe” bundles (use as patterns):

- `functions.exec_command` running `rg` searches + `functions.list_mcp_resources` + `functions.mcp__context7__resolve-library-id` (independent).
- Multiple Exa searches (`functions.mcp__exa__web_search_exa`) for different libraries/pages (independent).

**MCP tool sequence (use when it adds signal):**

1. `functions.mcp__zen__planner` → plan the current WP only.
2. `functions.list_mcp_resources` → discover preloaded docs/indexes; `functions.read_mcp_resource` to load any relevant ones.
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → authoritative API signatures/snippets (Streamlit, Pydantic, LlamaIndex, Qdrant, DuckDB, OpenTelemetry).
4. Prefer MCP doc corpora before general web when applicable:
   - LlamaIndex: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc`
   - LangChain/LangGraph: `functions.mcp__langchain-docs__SearchDocsByLangChain`
   - OpenAI API: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (only when needed)
5. `functions.mcp__exa__web_search_exa` (+ `functions.mcp__exa__crawling_exa`) → official docs/changelogs when behavior is subtle or “latest” matters.
6. `functions.mcp__gh_grep__searchGitHub` → real-world usage patterns for tricky APIs.
7. `functions.mcp__zen__analyze` → architecture sanity check before cross-cutting refactors.
8. `functions.mcp__zen__secaudit` → security audit for new/changed surfaces (config, path handling, logging, network).
9. `functions.mcp__zen__codereview` → final quality gate after each WP.

**opensrc usage (dependency internals, repo-truth):**

- Inspect `opensrc/sources.json` and fetch missing sources only when needed:

```bash
cat opensrc/sources.json | rg -n "\"name\": \"(pypi:)?(llama-index|streamlit|qdrant-client|duckdb|python-dotenv|opentelemetry)\"" || true
npx opensrc pypi:python-dotenv
npx opensrc pypi:llama-index
npx opensrc open-telemetry/opentelemetry-python
```

### Long-running processes

Use `functions.write_stdin` if you start long-running commands (examples: `streamlit run ...`, `docker compose up ...`) and need to interact or fetch more output without restarting.

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

**Primary Task:** Ship DocMind v1 by executing all work packages listed in `docs/specs/spec-021-release-readiness-v1.md`.

**Why now:** The repo contains P0 ship blockers (broken Docker artifacts, unsafe HTML sink, placeholder modules, and documentation drift). A first finished release requires runnable deploy artifacts + consistent docs + passing quality gates.

**Definition of Done (DoD):**

- All work package prompts in `docs/developers/prompts/README.md` are completed in order, each passing its quality gates.
- No TODO/FIXME/XXX placeholders remain in `src/` (except explicitly allowed in tests or archived docs).
- `docs/specs/traceability.md` reflects all shipped changes (status updated to Implemented).
- Dockerfile + compose run successfully and respect Python 3.11 + `DOCMIND_*` env contract.

**In-scope modules/files (initial):**

- `docs/specs/spec-021-release-readiness-v1.md`
- `docs/developers/prompts/*`
- Work package scopes per each prompt.

**Out-of-scope (explicit):**

- Major model stack changes (Torch/vLLM/Transformers pin shifts).
- Enabling remote endpoints by default.

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python version must remain **3.11.x** (respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`
- Do not introduce new dependency groups/extras unless an ADR/SPEC requires it.

#### 2) Style, Types, and Lint

Your code must pass:

- `uv run ruff format .`
- `uv run ruff check . --fix`
- `uv run pyright`

Rules:

- Prefer typed dataclasses / Pydantic models over untyped dicts.
- Avoid `Any` unless you can justify it (and isolate it behind a narrow boundary).
- No silent exception swallowing. Catch specific exceptions and log meaningfully.

#### 3) Streamlit UI Discipline

- `src/app.py` stays a thin shell (no business logic).
- Pages in `src/pages/*` should:
  - keep UI concerns local
  - call domain-layer services/helpers (do not rebuild pipelines in UI)
  - use `st.session_state` intentionally and avoid hidden global state
- Avoid expensive work at import time; Streamlit reruns frequently.

#### 4) Config Discipline (Pydantic Settings v2)

- Configuration source of truth is `src/config/settings.py`.
- Do not scatter `os.getenv` in domain code.
- If a new config knob is needed:
  - add it to settings
  - document it
  - add tests for mapping behavior
  - update docs (configuration reference if present)

#### 5) LlamaIndex + LangGraph Alignment

- Prefer LlamaIndex maintained primitives over custom ingestion/retrieval glue.
- Prefer LangGraph supervisor patterns for orchestration logic.
- Preserve offline-first operation:
  - do not add implicit network calls
  - gate network/exporters behind config flags

#### 6) Persistence & Caching Correctness

- Respect snapshot locking/atomicity semantics.
- Ensure ingestion/idempotency rules remain deterministic:
  - stable IDs/hashes
  - explicit cache invalidation behavior

#### 7) Observability & Security

- Emit telemetry events where the SPEC requires:
  - local JSONL telemetry and/or OTel spans/metrics (when enabled)
- Never log secrets.
- Enforce endpoint allowlist for any remote surface.
- Validate filesystem paths (no traversal, no symlink escape).

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

You MUST produce a plan and keep exactly one step “in_progress” at a time.

0. [ ] Read `docs/specs/spec-021-release-readiness-v1.md` + `docs/specs/traceability.md` and restate DoD in your plan.

1. [ ] Load `docs/specs/spec-021-release-readiness-v1.md` and enumerate all work packages.
2. [ ] Execute prompts in `docs/developers/prompts/README.md` order, one at a time.
3. [ ] After each package: run its scoped tests + full quality gates when required.
4. [ ] Update `docs/specs/traceability.md` statuses to Implemented for shipped changes.
5. [ ] Run final full quality gates and produce a release readiness report.

Commands you will run at minimum:

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

Scan the feature scope and delete or refactor immediately if found:

1. **God Modules:** single file > 400 LOC without clear layering.
2. **Import-time side effects:** heavy IO/model loads at import (breaks Streamlit reruns/tests).
3. **Config sprawl:** repeated `os.getenv` usage outside settings module.
4. **Swallowed errors:** broad `except Exception` without re-raise or explicit handling.
5. **Async misuse:** blocking calls inside async paths; mixed event loops without control.
6. **Unbounded caches:** files growing without rotation/limits; missing TTL/invalidation.
7. **Security footguns:** path traversal, unsafe temp files, remote endpoints ungated.
8. **Dead code:** unused exports, unreferenced entrypoints, obsolete compatibility layers.
9. **Undocumented behavior:** feature ships without SPEC/ADR/RTM updates.

---

### MCP TOOL STRATEGY (FOR IMPLEMENTATION RUN)

Use these tools as needed:

1. `functions.mcp__zen__planner` → Implementation plan
2. `functions.mcp__exa__web_search_exa` / `functions.mcp__exa__crawling_exa` → Official docs for touched libs
3. `functions.mcp__context7__resolve-library-id` + `functions.mcp__context7__query-docs` → API details
4. `functions.mcp__gh_grep__searchGitHub` → idiomatic patterns
5. `functions.mcp__zen__analyze` → architecture assessment of modified areas
6. `functions.mcp__zen__codereview` → post-implementation review
7. `functions.mcp__zen__secaudit` → security audit of changed surfaces

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement     | Status | Proof / Notes                                     |
| --------------- | ------ | ------------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                   |
| **Formatting**  |        | `ruff format`                                     |
| **Lint**        |        | `ruff check` clean                                |
| **Types**       |        | `pyright` clean                                   |
| **Pylint**      |        | meets threshold                                   |
| **Tests**       |        | `pytest` green (scoped + full tiers as required)  |
| **Docs**        |        | ADR/SPEC/RTM updated                              |
| **Security**    |        | allowlist + path validation + no secret logs      |
| **Tech Debt**   |        | zero TODO/FIXME introduced                        |
| **Performance** |        | no new import-time heavy work; key flows measured |

**EXECUTE UNTIL COMPLETE.**
