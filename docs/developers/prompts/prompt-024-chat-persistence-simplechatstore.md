# Implementation Prompt — Chat Persistence (SimpleChatStore JSON)

Implements `ADR-043` + `SPEC-024`.

## Tooling & Skill Strategy (fresh Codex sessions)

**Read first:** `docs/developers/prompts/README.md` and `~/prompt_library/assistant/codex-inventory.md`.

**Use skill:** `$streamlit-master-architect` (Streamlit reruns, AppTest, session state discipline)

Skill references to consult (as needed):
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/architecture_state.md`
- `/home/bjorn/.codex/skills/streamlit-master-architect/references/testing_apptest.md`

**Preflight:**

```bash
uv sync
uv run python -c "import streamlit as st; print(st.__version__)"
rg -n "st\\.session_state\\.|st\\.chat_" src/pages/01_chat.py
```

### Prompt-specific Tool Playbook (optimize tool usage)

**Planning discipline (required):** Use `functions.update_plan` to track execution steps and keep exactly one step `in_progress`.

**Parallel preflight (use `multi_tool_use.parallel`):**

- Code search and scope discovery:
  - `rg -n \"SimpleChatStore|ChatMemoryBuffer|chat_store\" -S src`
  - `rg -n \"messages\\b|st\\.session_state\\b\" -S src/pages/01_chat.py src/ui`
- Read in parallel:
  - `src/pages/01_chat.py`
  - `src/config/settings.py` (data_dir location)
  - `src/utils/security.py` (path validation helpers, if any)

**MCP resources first (when available):**

- `functions.list_mcp_resources` → look for LlamaIndex/storage/chat-store resources; read them before web search.

**API verification (Context7):**

- `functions.mcp__context7__resolve-library-id` → `llama-index`
- `functions.mcp__context7__query-docs` → confirm the exact module paths/usage for:
  - `SimpleChatStore` JSON persistence
  - `ChatMemoryBuffer` wiring and how to persist updates deterministically

**Real-world patterns (GitHub grep):**

- `functions.mcp__gh_grep__searchGitHub` → confirm common patterns for “load/store JSON + chat memory buffer” and safe file IO boundaries.

**Long-running UI validation (use native capabilities):**

- If you run `streamlit run src/app.py`, keep the process alive; use `functions.write_stdin` to pull logs.
- If you capture screenshots during manual verification, attach them with `functions.view_image`.
- For E2E smoke, use the skill’s Playwright flow:
  - `/home/bjorn/.codex/skills/streamlit-master-architect/scripts/mcp/run_playwright_mcp_e2e.py`

**Security gate (required):**

- Run `functions.mcp__zen__secaudit` focused on:
  - safe path handling under `settings.data_dir`
  - no raw message-content logging

### MCP tool sequence (use when it adds signal)

1. `functions.mcp__zen__planner` → plan persistence wiring + tests.
2. Context7:
   - resolve `llama-index` and query docs for `SimpleChatStore` + `ChatMemoryBuffer` usage patterns.
3. `functions.mcp__gh_grep__searchGitHub` → confirm real-world `SimpleChatStore` JSON persistence patterns (if behavior unclear).
4. `functions.mcp__zen__secaudit` → path validation + no message-content logging.

**opensrc (only if subtle LlamaIndex behavior):**

```bash
cat opensrc/sources.json | rg -n "llama-index" || true
# Fetch only if missing and behavior is surprising; treat opensrc/ as read-only.
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

**Primary Task:** Persist Streamlit chat history locally using `llama_index.core.storage.chat_store.SimpleChatStore` persisted to JSON and wired into `ChatMemoryBuffer`.

**Why now:** Chat history currently resets on refresh/restart, contradicting expected UX and ADR-021 intent. We must ship persistence without adding new dependencies.

**Definition of Done (DoD):**

- Chat page loads prior messages on refresh when a chat store file exists.
- Messages persist to `settings.data_dir / \"chat\"` per session id.
- “Clear chat” removes persisted history for the session safely.
- No network calls are introduced.
- Tests: AppTest integration proves persistence across runs.
- RTM updated (FR-022 row planned → implemented).

**In-scope modules/files (initial):**

- `src/pages/01_chat.py`
- `src/ui/chat_persistence.py` (new)
- `tests/integration/ui/` (new AppTest file)
- `docs/developers/adrs/ADR-043-chat-persistence-simplechatstore.md`
- `docs/specs/spec-024-chat-persistence-simplechatstore.md`
- `docs/specs/traceability.md`

**Out-of-scope (explicit):**

- SQLite chat store integration packages.
- Multi-user server concurrency guarantees.

---

### HARD RULES (EXECUTION)

- Persist only under `settings.data_dir` (validate paths; block symlink escape).
- Never log raw message content.
- Keep helper minimal and typed.

---

### STEP-BY-STEP EXECUTION PLAN (FILLED)

1. [ ] Inspect current chat page flow and how it stores messages in `st.session_state`.
2. [ ] Add `src/ui/chat_persistence.py` helper:
   - session id generation
   - load/create SimpleChatStore from disk
   - return ChatMemoryBuffer wired to store
   - persist store after updates
3. [ ] Wire `src/pages/01_chat.py` to load persisted history at startup and persist on new messages.
4. [ ] Add “Clear chat” action to delete the session store and clear UI state.
5. [ ] Add AppTest integration tests for persistence across runs.
6. [ ] Run quality gates and update RTM.

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

1. Logging raw chat messages in telemetry/logs.
2. Persisting chat outside `settings.data_dir`.
3. Heavy imports or IO at import time (must be lazy in Streamlit).

---

### FINAL VERIFICATION CHECKLIST (MUST COMPLETE)

| Requirement | Status | Proof / Notes                    |
| ----------- | ------ | -------------------------------- |
| Tests       |        | AppTest proves persistence       |
| Security    |        | path validation; no content logs |
| Docs        |        | ADR/SPEC/RTM updated             |

**EXECUTE UNTIL COMPLETE.**
