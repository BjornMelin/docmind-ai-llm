---
spec: SPEC-024
title: Chat Persistence — SimpleChatStore JSON + ChatMemoryBuffer in Streamlit
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Superseded
related_requirements:
  - FR-022: Chat history persists locally across refresh/restart.
  - NFR-SEC-001: Default egress disabled; only local endpoints allowed unless explicitly configured.
  - NFR-SEC-002: Local data remains on device; logging excludes sensitive content.
related_adrs: ["ADR-043", "ADR-024", "ADR-047", "ADR-057"]
---

> Status notice (2026-01-09): Superseded by SPEC-041 / ADR-057 (LangGraph SQLite checkpointer + store with time travel + hybrid long-term memory).

## Objective

Persist Streamlit Chat history locally using LlamaIndex core primitives:

- `SimpleChatStore` persisted to JSON under `settings.data_dir`
- `ChatMemoryBuffer` bound to the store per session

## Non-goals

- Multi-user server deployments with concurrent write coordination
- Cloud chat history sync

## User stories

1) As a user, I refresh the browser and my prior chat history remains.
2) As a user, I can clear chat history for the current session.
3) As a user, chat persistence never triggers network access.

## Technical design

### Storage layout

- Directory: `settings.data_dir / "chat"`
- Files: `chat_store_<session_id>.json`

Session id policy:

- If no explicit session id exists, generate and store one in `st.session_state["chat_session_id"]`.

### Implementation plan

1) Add a small helper module (suggested):
   - `src/ui/chat_persistence.py`
   - Responsibilities:
     - resolve chat session id
     - load/create `SimpleChatStore` from disk
     - return `ChatMemoryBuffer` bound to the store
     - persist store on message append

2) Wire Chat page (`src/pages/01_chat.py`) to:
   - initialize memory/store once per session
   - on each user message:
     - append to store via memory/chat store APIs
     - persist to disk

3) Provide “Clear chat” button:
   - delete store file for this session (validated path under data_dir)
   - clear `st.session_state.messages`

### Notes on token limits

- Use `ChatMemoryBuffer.from_defaults(token_limit=...)` for prompt window trimming.
- Persistence stores the full history; memory is responsible for limiting prompt window.

## Observability

- Emit a local JSONL event on:
  - `chat.persist` { session_id, message_count, size_bytes? }
  - `chat.clear` { session_id }
- Do not include message content.

## Security

- Validate file paths stay under `settings.data_dir / "chat"` and block symlink escape.
- Local-only; no remote sync.

## Testing strategy

### Integration (Streamlit AppTest)

- Start Chat page (`AppTest.from_file("src/pages/01_chat.py")`) in a temp data_dir.
- Simulate sending a message.
- Restart a fresh AppTest instance with the same data_dir and assert prior messages load.
- Test “Clear chat” deletes file and clears UI state.

### Unit

- Chat persistence helper:
  - load when file exists
  - create when missing
  - persist writes JSON to expected path

## Rollout / migration

- Default-on for local runs.
- No migration required (new file under data_dir).

## RTM updates (docs/specs/traceability.md)

Add:

- FR-022: “Chat persistence (SimpleChatStore)”
  - Code: `src/pages/01_chat.py`, `src/ui/chat_persistence.py` (new)
  - Tests: new AppTest integration + unit tests
  - Verification: test
  - Status: Planned → Implemented
