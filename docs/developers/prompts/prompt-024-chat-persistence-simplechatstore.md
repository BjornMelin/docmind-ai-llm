# Implementation Prompt — Chat Persistence (SimpleChatStore JSON)

Implements `ADR-043` + `SPEC-024`.

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
