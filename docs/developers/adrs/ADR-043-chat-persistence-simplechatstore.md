---
ADR: 043
Title: Chat Persistence via LlamaIndex SimpleChatStore (JSON) + ChatMemoryBuffer
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes: 021
Superseded-by:
Related: 012, 024
Tags: chat, memory, streamlit, persistence
References:
  - https://docs.llamaindex.ai/en/stable/api_reference/memory/chat_memory_buffer/
  - https://docs.llamaindex.ai/en/v0.10.22/module_guides/storing/chat_stores/
---

## Description

Persist chat history locally for the Streamlit Chat page using LlamaIndex **SimpleChatStore** (JSON) wired into **ChatMemoryBuffer**.

## Context

ADR-021 proposes persistence via SQLite-backed chat stores, but:

- current code (`src/pages/01_chat.py`) uses only `st.session_state.messages` (no persistence)
- the repo does not currently depend on a SQLite chat store integration package
- for v1 ship-readiness, the lowest-risk approach is to use LlamaIndex core primitives already installed

## Decision Drivers

- Offline-first, local-only persistence
- No new dependencies for v1
- Predictable, testable behavior in Streamlit rerun model
- Minimal operational burden (single JSON file per session)

## Alternatives

- A: Session-only (`st.session_state`) — loses history on restart
- B: SimpleChatStore JSON persistence (Selected)
- C: Add SQLite chat store integration — more deps + migration surface

### Decision Framework (≥9.0)

| Option                      | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| --------------------------- | ---------------: | ---------: | --------------: | ------: |
| **B: SimpleChatStore JSON** |                9 |          9 |              10 | **9.3** |
| A: Session-only             |               10 |          8 |               5 |     7.9 |
| C: SQLite integration       |                6 |          9 |               7 |     7.2 |

## Decision

Implement chat history persistence as:

- `SimpleChatStore.from_persist_path(persist_path=...)` (create new store if missing)
- `ChatMemoryBuffer.from_defaults(chat_store=store, chat_store_key=session_id, token_limit=...)`
- persist store to disk on message append and on clear/reset

Use per-session separation to avoid collisions:

- file: `data/chat/chat_store_<session_id>.json`
- key: `chat_store_key=<session_id>`

## Security & Privacy

- Chat history files are local-only under `settings.data_dir`.
- Do not log raw chat content in telemetry.
- Provide a “Clear chat” action that deletes only the session’s file (validated path).

## Consequences

### Positive Outcomes

- Conversations survive refresh and server restarts.
- No new dependencies required.

### Trade-offs

- JSON persistence is not designed for high concurrency; mitigate via per-session file and atomic writes.

## Changelog

- 1.0 (2026-01-09): Proposed for v1; supersedes ADR-021’s SQLite-centric plan.
