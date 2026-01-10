# Implementation Prompt — Chat Persistence + Hybrid Agentic Memory (LangGraph SQLite)

Implements `ADR-057` + `SPEC-041`.

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-057-chat-persistence-langgraph-sqlite-hybrid-memory.md`
- SPEC: `docs/specs/spec-041-chat-persistence-agentic-memory-langgraph-sqlite.md`
- Requirements: `docs/specs/requirements.md` (FR-022, FR-030..032, NFR-SEC-001/002/003/004)
- RTM: `docs/specs/traceability.md`

## Official docs (research during implementation)

- <https://docs.langchain.com/oss/python/langgraph/persistence> — Threads/checkpoints and how configs (`thread_id`) drive durable state.
- <https://docs.langchain.com/oss/python/langgraph/add-memory> — Short-term vs long-term memory, and DB-backed implementations.
- <https://docs.langchain.com/oss/python/langgraph/use-time-travel> — `get_state_history`, `update_state`, and resuming from a fork.
- <https://pypi.org/project/langgraph-checkpoint-sqlite/> — `langgraph-checkpoint-sqlite==3.0.1` (SqliteSaver + SqliteStore, sqlite-vec dependency).
- <https://docs.streamlit.io/develop/api-reference/chat/st.chat_input> — Chat input widget behavior.
- <https://docs.streamlit.io/develop/api-reference/chat/st.chat_message> — Chat message rendering patterns.
- <https://docs.streamlit.io/develop/api-reference/caching-and-state/st.query_params> — Shareable URLs (`?chat=<id>`), repeated keys, and multipage clearing behavior.
- <https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest> — AppTest for Streamlit integration tests.

SOTA memory research (context, informs design):

- <https://www.letta.com/blog/letta-leaderboard> — Memory capability benchmark suite for LLMs (read/write/update memory).
- <https://www.letta.com/blog/benchmarking-ai-agent-memory> — LoCoMo benchmark discussion; compares filesystem/tooling memory vs specialized memory systems.
- <https://mem0.ai/blog/ai-agent-memory-benchmark/> — Mem0 benchmark comparisons; describes extraction + update pipeline patterns.
- <https://arxiv.org/abs/2501.13956> — Zep/Graphiti temporal knowledge graph approach for agent memory.

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Use skill:** `$streamlit-master-architect`

Mandatory workflow steps from the skill:

```bash
uv sync
uv run python -c "import streamlit as st; print(st.__version__)"
uv run python /home/bjorn/.codex/skills/streamlit-master-architect/scripts/audit_streamlit_project.py --root . --format md
```

Skill references to consult (as needed):

- `/home/bjorn/.codex/skills/streamlit-master-architect/references/architecture_state.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/widget_keys_and_reruns.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/caching_and_fragments.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/testing_apptest.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/security.md`

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Repo truth scan:
  - `rg -n \"InMemorySaver\\b|SqliteSaver\\b|SqliteStore\\b|thread_id\\b\" -S src/agents/coordinator.py src/pages/01_chat.py`
  - `rg -n \"st\\.session_state\\.(messages|\\w+)\" -S src/pages/01_chat.py`
  - `rg -n \"query_params|st\\.query_params\" -S src/pages/01_chat.py src/app.py src/pages`
  - `rg -n \"ChatMemoryBuffer\\b\" -S src/agents`
- Read key files (in parallel): `src/pages/01_chat.py`, `src/agents/coordinator.py`, `src/agents/models.py`, `src/agents/registry/tool_registry.py`, `src/config/settings.py`.

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for preloaded LangGraph/Streamlit resources.
- `functions.read_mcp_resource` → read before doing web search.

**Authoritative library docs (MCP, prefer over general web when applicable):**

- LangChain/LangGraph docs: `functions.mcp__langchain-docs__SearchDocsByLangChain` (persistence, checkpoints, interrupts/time travel)
- LlamaIndex docs: `functions.mcp__llama_index_docs__search_docs` / `functions.mcp__llama_index_docs__grep_docs` / `functions.mcp__llama_index_docs__read_doc` (if you bridge any LlamaIndex chat stores/tools)
- OpenAI API docs: `functions.mcp__openaiDeveloperDocs__search_openai_docs` → `functions.mcp__openaiDeveloperDocs__fetch_openai_doc` (only if this work package touches OpenAI API semantics)

**API verification (Context7, only when uncertain):**

- `functions.mcp__context7__resolve-library-id` → `streamlit`, `langgraph`, `langchain-core` (and `pydantic` if state schema changes).
- `functions.mcp__context7__query-docs` → confirm exact signatures for:
  - Streamlit `st.query_params` behavior on multipage apps
  - LangGraph `get_state_history` / `update_state` / persistence config structure

**Real-world patterns (GitHub grep):**

- `functions.mcp__gh_grep__searchGitHub` → find Streamlit Chat session pickers and LangGraph `thread_id` usage patterns.

**Dependency internals (opensrc):**

- Check `opensrc/sources.json` and prefer existing snapshots.
- If a dependency is ambiguous, fetch sources with `npx opensrc pypi:<pkg>@<ver> --modify=false`.
- Cite exact `opensrc/...` paths in code review notes when using internals.

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` focused on:
  - DB path validation (no traversal/symlink escape)
  - SQL injection surfaces (metadata filter keys and any dynamic queries)
  - memory poisoning (prompt injection stored as “fact”)
  - telemetry safety (no raw message content)

**Review gate (recommended):**

- Run `functions.mcp__zen__codereview` after implementation (large blast radius: coordinator + chat page + persistence).

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

**Primary Task:** Replace the superseded SimpleChatStore plan with a final-release chat persistence + hybrid agentic memory system using LangGraph SQLite (`SqliteSaver` + `SqliteStore` with `sqlite-vec` semantic search), plus a deterministic “extract + ADD/UPDATE/DELETE/NOOP” memory consolidation pipeline, with Streamlit UI session management and time-travel branching.

**Why now:** The current Chat UI is session-only and the coordinator uses `InMemorySaver`, so there is no durable memory or time travel. The final release requires multi-session persistence, branching, and long-term memory without violating offline-first security posture.

**Definition of Done (DoD):**

- Chat sessions persist across refresh/restart and support create/rename/delete/select.
- Coordinator uses `langgraph-checkpoint-sqlite` for durable checkpoints, enabling time travel with `get_state_history` + `update_state`.
- Long-term memory store uses `SqliteStore` vector indexing; memories are scoped by `user_id` and `thread_id`, support metadata-filtered recall, and are user-reviewable/deletable.
- Consolidation is implemented with explicit `ADD/UPDATE/DELETE/NOOP` operations and bounded retention/TTL (no unbounded growth).
- No raw message content is emitted in telemetry/logs; DB paths are validated; remote endpoints remain blocked by default.
- Tests added: unit tests for chat DB/session registry + AppTest integration for session restore and time travel fork.
- Docs remain aligned: ADR-057 + SPEC-041 + requirements/RTM updated; superseded docs remain in `*/superseded/`.

**In-scope modules/files (initial):**

- `src/pages/01_chat.py`
- `src/agents/coordinator.py`
- `src/agents/models.py`
- `src/agents/registry/tool_registry.py`
- `src/agents/tools/memory.py` (new)
- `src/persistence/chat_db.py` (new)
- `src/ui/chat_sessions.py` (new)
- `docs/specs/traceability.md` (update to Implemented)

**Out-of-scope (explicit):**

- Cloud sync and multi-tenant auth.
- Replacing Qdrant document retrieval or SnapshotManager.

---

### HARD RULES (EXECUTION)

#### 1) Python + Packaging

- Python version must remain **3.11.x** (respect `pyproject.toml`).
- Use **uv only**:
  - install/sync: `uv sync`
  - run tools: `uv run <cmd>`

#### 2) Style, Types, and Lint

Your code must pass:

- `uv run ruff format .`
- `uv run ruff check .`
- `uv run pyright`

Rules:

- Prefer typed dataclasses / TypedDict / Pydantic models with strict types; avoid `Any` unless isolated behind a narrow boundary.
- No silent exception swallowing. Catch specific exceptions and log meaningfully.

#### 3) Streamlit UI Discipline

- `src/app.py` stays a thin shell (no business logic).
- `src/pages/01_chat.py` should be UI wiring; move DB/session logic to `src/persistence/*` and `src/ui/*` helpers.
- Avoid expensive work at import time; Streamlit reruns frequently.

#### 4) Config Discipline (Pydantic Settings v2)

- Configuration source of truth is `src/config/settings.py`.
- Do not add new `os.getenv` usage in domain code.
- If new config is needed (e.g., `chat.user_id`), add it to settings and document it.

#### 5) LangGraph Store Alignment

- Use LangGraph checkpointer and store interfaces (`setup()`, `thread_id` config).
- Ensure persisted state is serializable (avoid storing live objects in state).
- Memory namespaces must be scoped to `user_id` and `thread_id` (no cross-user bleed).
- Enforce bounded memory growth: TTL/retention and delete/purge operations are first-class.

#### 6) Observability & Security

- Never log secrets or raw chat messages.
- Validate DB paths (no traversal, no symlink escape).
- Keep remote endpoints blocked by default and respect endpoint allowlist.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

You MUST produce a plan and keep exactly one step “in_progress” at a time.

1. [ ] Add dependency + wire `langgraph-checkpoint-sqlite` (pin versions in `pyproject.toml` and `uv.lock`).
   - Commands:
     - `uv sync`
     - `uv run python -c \"import langgraph; print(langgraph.__version__)\"`
     - `uv run python -c \"import langgraph.checkpoint.sqlite\"`
2. [ ] Implement `src/persistence/chat_db.py` (path validation, connections, `chat_session` CRUD).
   - Commands:
     - `uv run ruff check src/persistence/chat_db.py`
     - `uv run pyright`
3. [ ] Add LangGraph persistence in `src/agents/coordinator.py`:
   - replace `InMemorySaver` with `SqliteSaver` and ensure `setup()` called
   - ensure state schema supports multi-turn message accumulation and serialization (migrate to `AnyMessage` + reducer semantics as needed)
   - Commands:
     - `uv run ruff check src/agents/coordinator.py src/agents/models.py`
4. [ ] Add long-term memory primitives:
   - `SqliteStore` vector index initialization (embedding adapter, dims, fields)
   - memory extract + `ADD/UPDATE/DELETE/NOOP` consolidation step
   - memory search tool (vector search + metadata filters)
   - register tools in `src/agents/registry/tool_registry.py`
   - Commands:
     - `uv run ruff check src/agents/tools/memory.py src/agents/registry/tool_registry.py`
5. [ ] Update Chat UI (`src/pages/01_chat.py`) to:
   - session picker (create/rename/delete)
   - `thread_id` + `user_id` propagation into coordinator calls
   - time travel UI (list checkpoints; fork; resume)
   - memory review + purge (per-user/per-session)
   - Commands:
     - `uv run ruff check src/pages/01_chat.py src/ui/chat_sessions.py`
6. [ ] Add tests:
   - unit: `chat_db` CRUD + namespace scoping
   - integration: AppTest session persistence + time travel fork
   - Commands:
     - `uv run python -m pytest -q tests/unit/persistence/test_chat_db.py`
     - `uv run python -m pytest -q tests/integration/ui/test_chat_persistence_langgraph.py`
7. [ ] Run repo quality gates (no placeholders left in scope).
   - Commands:
     - `uv run ruff format .`
     - `uv run ruff check .`
     - `uv run pyright`
     - `uv run python scripts/run_tests.py --fast`
     - `uv run python scripts/run_quality_gates.py --ci --report`

---

### ANTI-PATTERN KILL LIST (IMMEDIATE DELETION/REWRITE)

Scan the feature scope and delete or refactor immediately if found:

1. **Import-time side effects:** heavy IO/model loads at import (breaks Streamlit reruns/tests).
2. **Config sprawl:** repeated `os.getenv` usage outside settings module.
3. **Swallowed errors:** broad `except Exception` without re-raise or explicit handling.
4. **Security footguns:** path traversal, unsafe temp files, remote endpoints ungated.
5. **Non-serializable state:** storing live clients/objects in LangGraph persisted state.
6. **Undocumented behavior:** drift between implementation and ADR/SPEC/RTM.

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement     | Status | Proof / Notes                                  |
| --------------- | ------ | ---------------------------------------------- |
| **Packaging**   |        | `uv sync` clean                                |
| **Formatting**  |        | `uv run ruff format .`                         |
| **Lint**        |        | `uv run ruff check .`                          |
| **Types**       |        | `uv run pyright`                               |
| **Tests**       |        | `uv run python scripts/run_tests.py --fast`    |
| **Docs**        |        | ADR/SPEC/RTM updated                           |
| **Security**    |        | allowlist + path validation + no raw chat logs |
| **Tech Debt**   |        | zero TODO/FIXME introduced                     |
| **Performance** |        | no import-time heavy work; key spans measured  |

**EXECUTE UNTIL COMPLETE.**
