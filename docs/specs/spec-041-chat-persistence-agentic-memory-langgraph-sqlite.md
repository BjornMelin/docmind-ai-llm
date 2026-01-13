---
spec: SPEC-041
title: Chat Persistence + Hybrid Agentic Memory (LangGraph SQLite Checkpointer + Store)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Partial
related_requirements:
  - FR-022: Persist chat history locally across refresh/restart with per-session clear/purge.
  - NFR-SEC-001: Offline-first; remote endpoints blocked by default.
  - NFR-SEC-002: Local data remains on device; logging excludes sensitive content.
  - NFR-REL-001: Crash-safe persistence and recovery.
related_adrs: ["ADR-057", "ADR-011", "ADR-016", "ADR-024", "ADR-047", "ADR-055"]
notes: "ADR-055 is currently 'Proposed' status; update reference when it advances to 'Accepted'."
---

## Goals

1. **Durable chat sessions** across refresh/restart with explicit user controls:
   - create/rename/delete sessions
   - clear/purge per-session history
2. **Branching / time travel** for agent workflows:
   - list checkpoints for a session
   - fork state from a checkpoint
   - resume execution from the fork
3. **Hybrid agentic memory**:
   - short-term memory: thread state (messages) via LangGraph checkpointer
   - long-term memory: facts/preferences stored in a DocMind-managed SQLite memory table (with `sqlite-vec` for semantic search)
   - background consolidation using an explicit extract + update policy (reduces duplicates/contradictions)
4. **Offline-first + security-first**:
   - no new network surfaces required
   - no raw chat content in telemetry/logs
   - strict user/session isolation in memory namespaces

## Non-goals

- Cloud chat history sync.
- Multi-tenant authentication (DocMind remains local-first; “user_id” is a local concept).
- Replacing the document RAG persistence architecture (Qdrant + snapshots) (ADR-031 / SPEC-014).

## User Stories (Streamlit flows)

1. Session management
   - As a user, I can create a new chat session and give it a name.
   - As a user, I can switch between sessions and see their history immediately.
   - As a user, I can delete a session and purge its persisted data.
2. Time travel / branching
   - As a user, I can select a past checkpoint of a session and fork a new branch.
   - As a user, I can “resume from fork” and keep both histories.
3. Long-term memory
   - As a user, I can say “Remember I prefer dark mode” and later ask “What are my preferences?” and the agent recalls it.
   - As a user, I can review and delete stored memories for my local profile.

## Technical Design

### Dependencies (new)

Add (pinned):

- `langgraph-checkpoint-sqlite==3.0.1` (includes `sqlite-vec>=0.1.6`)

Rationale:

- Enables durable checkpoints and time travel; the long-term memory store is a DocMind-managed SQLite schema in the same Chat DB (no extra service).

### Storage boundaries

DocMind distinguishes:

1. **Ops DB** (metadata-only; ADR-055 / SPEC-039)
2. **Chat DB** (user data; this spec)

This spec introduces the Chat DB.

### Chat DB location

- Path: `settings.chat.sqlite_path`
- Default: `./data/chat.db` (separate from Ops DB for data isolation and independent retention/backup)
- Must be under `settings.data_dir` by default.
- WAL enabled by `SqliteSaver.setup()` and Chat DB initialization.

### Components

#### A) Durable checkpoints (short-term memory)

- Use `langgraph.checkpoint.sqlite.SqliteSaver`
- Required config keys per invoke:
  - `config={"configurable": {"thread_id": <thread_id>, "user_id": <user_id>}}`
- Checkpoint lifecycle:
  - each graph super-step persists a checkpoint
  - checkpoints enable replay + time travel + fault tolerance

#### B) Long-term memory store

- Use a DocMind-managed memory table in the Chat DB (SQLite + `sqlite-vec`), with a small adapter surface modeled after LangGraph store semantics (get/put/list + metadata filter).
- Configure vector index:
  - dims: `settings.embedding.dimension` (1024)
  - embed: **LlamaIndex embed model adapter** implementing `langchain_core.embeddings.Embeddings`
  - fields: `["content"]` (memory value payload field to embed)

Adapter sketch:

```python
from langchain_core.embeddings import Embeddings
from llama_index.core import Settings as LISettings

class LlamaIndexEmbeddingsAdapter(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return LISettings.embed_model.get_text_embedding_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return LISettings.embed_model.get_text_embedding(text)
```

#### C) Memory extraction + consolidation (final-release)

Long-term memory must not be an unbounded append-only log. Implement a deterministic, testable consolidation pipeline inspired by state-of-the-art “ADD/UPDATE/DELETE/NOOP” memory update patterns (see internal implementation in [src/agents/tools/memory.py](src/agents/tools/memory.py) and [src/persistence/memory_store.py](src/persistence/memory_store.py)):

1. **Extract candidates** from the most recent conversation turn into a fixed schema:
   - memories must be small, user-relevant facts/preferences
   - each candidate includes: `content`, `kind` (`fact|preference|todo|project_state`), `importance` (0..1), `source_checkpoint_id`, and optional `tags`
2. **Update policy**:
   - retrieve the top-N nearest existing memories in the same namespace (vector search)
   - decide per candidate: `ADD`, `UPDATE(existing_id)`, `DELETE(existing_id)`, or `NOOP`
   - write changes back to the store
3. **Retention / TTL**:
   - apply TTL for low-importance items
   - enforce max per-namespace counts (evict oldest/lowest-importance)
4. **User controls**:
   - UI must provide memory review and delete/purge per-user and per-session

Configuration defaults (settings.chat):

- `memory_similarity_threshold` (0.85)
- `memory_low_importance_threshold` (0.3)
- `memory_low_importance_ttl_days` (14)
- `memory_max_items_per_namespace` (200)
- `memory_max_candidates_per_turn` (8)

Namespace conventions:

- per-session: `("memories", "{user_id}", "{thread_id}")`
- per-user global (optional): `("memories", "{user_id}")`

##### Memory candidate schema

Pydantic model (canonical for validation/tests):

```python
from pydantic import BaseModel, Field

class MemoryCandidate(BaseModel):
    content: str
    kind: str = Field(pattern="^(fact|preference|todo|project_state)$")
    importance: float = Field(ge=0.0, le=1.0)
    source_checkpoint_id: str
    tags: list[str] | None = None
```

Example payloads:

```json
{
  "content": "Prefers dark mode",
  "kind": "preference",
  "importance": 0.7,
  "source_checkpoint_id": "chkpt_01"
}
```

```json
{
  "content": "Project roadmap doc lives in /docs/specs",
  "kind": "fact",
  "importance": 1.0,
  "source_checkpoint_id": "chkpt_99",
  "tags": ["docs", "project_state"]
}
```

### Release scope (v1.0 vs. roadmap)

**Release 1.0 (v1) scope:**

- Durable checkpoints + session registry
- Long-term memory storage with explicit `ADD/UPDATE/DELETE/NOOP` policy
- Basic TTL and per-namespace caps with user-visible review/delete

**Roadmap (post-v1):**

- More advanced importance scoring and automated conflict resolution
- Background consolidation scheduling with adaptive cadence

#### D) Streamlit state + URL sync

Use Streamlit-native state per ADR-016:

- `st.session_state["chat_thread_id"]` (current session)
- `st.session_state["chat_user_id"]` (local profile id)

Use `st.query_params` for shareable links:

- `?chat=<thread_id>` and optionally `?branch=<checkpoint_id>`

Important Streamlit behavior:

- Query parameters are cleared when navigating between pages in a multipage app.
  - Therefore, query params are treated as an **entry hint**, not the sole source of truth.
  - On first load of a shared link, hydrate `st.session_state["chat_thread_id"]` (and a branch candidate from `?branch=`) and proceed from session state afterward. If a user navigates away and back, resume the thread from `st.session_state` and prompt the user to re-select/confirm the branch if the URL no longer contains `?branch=`.
  - If persistent shareable URLs are required, consider a hash-based fragment or a server-side share token that rehydrates session state on load.

#### E) Session registry

The system must list sessions and store user-friendly titles. The LangGraph checkpointer does not provide “list all thread_ids” as a stable public API.

Implement a small session registry table in the Chat DB (see [src/persistence/chat_db.py](src/persistence/chat_db.py)):

- `chat_session`
  - `thread_id TEXT PRIMARY KEY`
  - `title TEXT NOT NULL`
  - `created_at_ms INTEGER NOT NULL`
  - `updated_at_ms INTEGER NOT NULL`
  - `last_checkpoint_id TEXT` (nullable; convenience only)
  - `deleted_at_ms INTEGER` (nullable; for soft delete)

Indexes:

- `(updated_at_ms)`
- `(deleted_at_ms)`

### Time travel UX semantics

Expose in Chat sidebar (orchestrated by [src/agents/coordinator.py](src/agents/coordinator.py)):

1. List checkpoints for current `thread_id` (reverse chronological).
2. User selects one checkpoint to fork.
3. Fork operation:
   - call `graph.update_state(selected_state.config, values=…)` to create a new checkpoint id.
4. Resume:
   - call `graph.invoke(None, new_config)` via the coordinator’s compiled graph (e.g., `coordinator.compiled_graph.invoke`) or add a small public wrapper (e.g., `coordinator.resume_from_fork(...)`) and persist the new branch head.
5. Ensure both histories remain accessible (session remains the same `thread_id` but different checkpoint lineage).

## Observability

Emit local JSONL telemetry events (no message content):

- `chat.session_created` `{thread_id, title}`
- `chat.session_selected` `{thread_id}`
- `chat.session_deleted` `{thread_id}`
- `chat.checkpoint_forked` `{thread_id, from_checkpoint_id, new_checkpoint_id}`
- `chat.memory_saved` `{thread_id, user_id, count}`
- `chat.memory_searched` `{thread_id, user_id, top_k, latency_ms}`

If OpenTelemetry is enabled, add spans around:

- `chat.invoke` (overall)
- `chat.time_travel.list_history`
- `chat.time_travel.update_state`
- `chat.memory.search`

## Security

Threats and controls:

1. **PII retention**
   - Provide explicit UI controls: memory review + delete + session purge.
   - Default to “store only salient facts/preferences” (explicit extract + update policy) rather than raw logs.
2. **Prompt-injection memory poisoning**
   - Treat memories as _untrusted_ facts: store provenance and allow user review.
   - Use fixed tool schemas; do not allow arbitrary system prompt writes.
3. **SQL injection**
   - Never interpolate user-provided keys into SQL.
   - Rely on `langgraph-checkpoint-sqlite` filter-key validation and keep package pinned.
4. **Filesystem escape**
   - Validate the Chat DB path remains under `settings.data_dir` unless user explicitly overrides.
5. **Secret leakage**
   - Never log API keys; never include raw messages in telemetry.

## Testing Strategy

### Unit tests

- Session registry CRUD
- Embeddings adapter produces correct vector dims and deterministic outputs (using Mock embed model)
- Namespace scoping and filters
- Path validation (reject `..` / symlink escapes)

### Integration tests (Streamlit AppTest)

- Create session → chat → restart AppTest → history persists
- Time travel: fork from checkpoint → resume → UI shows new branch head
- Long-term memory: store preference → recall it later

### Security tests

- Telemetry payloads never include raw message content (assert keys only)
- Filter keys are not user-controlled in any query builder

## Rollout / Migration Notes

- New tables/files are created on demand at startup or first Chat interaction.
- No migration from the previous SimpleChatStore plan is required (it was not shipped).
- If a legacy JSON chat store file exists (from experimental runs), it is ignored by default.

## Performance Expectations / Guardrails

- No import-time heavy work in Streamlit pages; DB connections and coordinator initialization must be cached (`st.cache_resource`) or lazy.
- Recommended pattern:
  - `@st.cache_resource` factory for DB/checkpointer (keyed by `user_id` if relevant) to avoid repeated connections on Streamlit reruns
  - lazy session-state init for the coordinator (only create when missing)
- Add a performance test simulating rapid messages + session switches to assert DB connection counts remain bounded (no leaks across reruns).
- Memory consolidation runs in the background (debounced) to avoid blocking the hot path (Streamlit fragments `run_every` per ADR-052 when available).

## RTM Updates

Update `docs/specs/requirements.md`:

- Replace FR-022 “Source: SPEC-024/ADR-043” with “Source: SPEC-041/ADR-057”.
- Add new FRs for session management, branching/time travel, and long-term memory management (review/purge).

Update `docs/specs/traceability.md`:

- FR-022: point to new modules/tests and mark as `Planned`.
- Add new rows for session management + time travel + memory review.
