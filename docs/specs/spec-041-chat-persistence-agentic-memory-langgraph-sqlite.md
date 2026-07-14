---
spec: SPEC-041
title: Chat Persistence + Hybrid Agentic Memory (LangGraph SQLite Checkpointer + Store)
version: 2.0.0
date: 2026-07-13
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-022: Persist chat history locally across refresh/restart with per-session clear/purge.
  - NFR-SEC-001: Offline-first; remote endpoints blocked by default.
  - NFR-SEC-002: Local data remains on device; logging excludes sensitive content.
  - NFR-REL-001: Crash-safe persistence and recovery.
related_adrs: ["ADR-058", "ADR-057", "ADR-011", "ADR-016", "ADR-024", "ADR-047", "ADR-055"]
notes: "ADR-058 is the integrated source of truth (supersedes ADR-057). ADR-055 is currently 'Proposed' status; update reference when it advances to 'Accepted'."
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
   - long-term memory: facts/preferences stored by LangGraph's native SQLite store (with `sqlite-vec` for semantic search)
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

- `langgraph-checkpoint-sqlite>=3.0.3,<4.0.0` (with `sqlite-vec>=0.1.6`)

Rationale:

- Enables durable checkpoints, time travel, and the native LangGraph long-term memory store in the same Chat DB (no extra service or custom store adapter).

### Storage boundaries

DocMind distinguishes:

1. **Ops DB** (metadata-only; ADR-055 / SPEC-039)
2. **Chat DB** (user data; this spec)

This spec introduces the Chat DB.

### Chat DB location

- Path: `settings.chat.sqlite_path`
- Default: `./data/chat.db` (separate from Ops DB for data isolation and independent retention/backup)
- Must be under `settings.data_dir` by default.
- WAL enabled by `AsyncSqliteSaver.setup()` and Chat DB initialization.

### Components

#### A) Durable checkpoints (short-term memory)

- Use `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` because graph execution
  is asynchronous.
- The coordinator creates, initializes, uses, and closes the saver on its owned
  persistent event loop. Synchronous Streamlit state/list/fork APIs use the
  saver's native cross-thread bridge and are never called from the saver loop.
- A process-wide coordinator serializes first-use setup. It builds model and
  graph components locally, then publishes them under the same lifecycle lock
  that marks the coordinator closed. An owned saver opens on an unpublished
  runner; close wins publication races, and failed publication closes both.
- Close captures published saver and runner ownership without waiting for the
  setup lock or checkpointer initialization lock. Shutdown remains bounded when
  setup or a persistence bridge does not return.
- Every operation maps the public `(user_id, thread_id)` pair to the same opaque
  `docmind:<sha256>` persistence `thread_id`; `user_id` remains runtime context,
  and `checkpoint_ns` remains empty because LangGraph reserves non-empty
  namespaces for subgraphs.
- Startup rejects any LangGraph checkpoint table that contains a raw or
  structurally incompatible thread ID. V2 does not guess which user owns a v1
  checkpoint.
- Active-run fencing and hard purge use that same canonical identity. Two users
  may therefore use the same public thread ID without sharing state, while purge
  removes the current user's matching checkpoint rows.
- Checkpoint lifecycle:
  - each graph super-step persists a checkpoint
  - checkpoints enable replay + time travel + fault tolerance

#### B) Long-term memory store

- Use `langgraph.store.sqlite.SqliteStore` directly in the Chat DB. DocMind owns only the connection lifecycle and a one-shot migration from the removed `docmind_store_items` / `docmind_store_vec` schema; reads, writes, filters, TTL, namespace listing, and semantic search use native LangGraph APIs.
- Migration copies active rows as deterministic upserts, infers a missing `origin` as `consolidation` when `source_checkpoint_id` exists and `explicit` otherwise, reads every copied value back, and only then drops both legacy tables. A failed validation leaves the legacy source tables available for a retry.
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

Long-term memory must not be an unbounded append-only log. Implement a deterministic, testable consolidation pipeline inspired by state-of-the-art “ADD/UPDATE/DELETE/NOOP” memory update patterns (see internal implementation in [src/agents/tools/memory.py](../../src/agents/tools/memory.py) and [src/persistence/memory_store.py](../../src/persistence/memory_store.py)):

1. **Extract candidates** from the most recent conversation turn into a fixed schema:
   - memories must be small, user-relevant facts/preferences
   - each candidate includes: `content`, `kind` (`fact|preference|todo|project_state`), `importance` (0..1), `source_checkpoint_id`, and optional `tags`
2. **Update policy**:
   - retrieve the top-N nearest existing memories in the same namespace (vector search)
   - decide per candidate: `ADD`, `UPDATE(existing_id)`, `DELETE(existing_id)`, or `NOOP`
   - write changes back to the store
3. **Retention / TTL**:
   - apply TTL for low-importance items
   - enforce max per-namespace counts by evicting only derived consolidation
     records; never evict explicit user memories
   - reject a genuinely new explicit memory at the cap while allowing an
     idempotent update or a rekey that replaces an existing record
4. **User controls**:
   - UI must provide memory review and delete/purge per-user and per-session
   - failed save, search, delete, and purge operations show stable sanitized
     errors; save input and delete/purge confirmation remain available for retry
   - a rejected save, including a tombstoned namespace, never clears input or
     triggers a success rerun

Configuration defaults (settings.chat):

- `memory_similarity_threshold` (0.85)
- `memory_low_importance_threshold` (0.3)
- `memory_low_importance_ttl_days` (14)
- `memory_max_items_per_namespace` (200)
- `memory_max_candidates_per_turn` (8)

Namespace conventions:

- per-session: `("memories", "session", "u-<sha256>", "t-<sha256>")`
- per-user global (optional): `("memories", "user", "u-<sha256>")`

The explicit scope segment makes these namespace trees disjoint. Native
LangGraph prefix search, capacity enforcement, sidebar review, and purge
therefore cannot leak session memories into user-global results or cross
between sessions. Public user and thread identifiers never enter the native
store schema.

##### Mutation identity and deadlines

- Tools, consolidation, and the sidebar use one save path. Memory IDs are
  `mem-` plus the SHA-256 digest of the memory kind and stripped, case-folded
  content. Namespace components supply user, thread, and scope identity.
- Explicit `remember` and `forget_memory` tools require native LangGraph
  `InjectedState`. They fail closed with stable, user-safe JSON unless
  `deadline_ts` is a finite future absolute `time.monotonic()` timestamp at the
  check immediately before `put` or `delete`.
- Each admitted graph turn captures both its session and user namespace
  generations in typed graph state. Explicit mutations require the matching
  scope generation, so a manual purge rejects writes or deletes from a turn
  that started before the purge; the next turn captures the new generation.
- Background consolidation inherits the admitted session generation from the
  terminal graph state. Scheduling rejects missing or stale provenance instead
  of recapturing a post-purge generation.
- A bounded set of process-local namespace lock stripes serializes all memory
  writers. Consolidation holds the lock across search, decision, and apply;
  explicit writes take precedence over extracted metadata.
- Consolidation uses the same deterministic key. `UPDATE` writes the canonical
  key before deleting the old key. A failed write preserves the old record; a
  failed delete may leave both records, and retrying the same `UPDATE` safely
  converges.
- One absolute consolidation deadline covers decision, every mutation, and
  eviction. Extraction passes the remaining budget as the provider request
  timeout. Once the deadline expires, no later mutation starts, and bounded
  worker capacity remains occupied until the provider call exits.
- Consolidation work starts only after a worker slot is admitted and runs on an
  owned daemon thread. Close rejects new work and invalidates outstanding
  generations; a provider that ignores its timeout cannot hold interpreter
  shutdown through Python's global thread-pool exit join.
- Python cannot cancel an already-started synchronous store commit. Such a
  commit may finish after the deadline, but its deterministic key prevents a
  retry from creating a duplicate record.
- The coordinator owns hard purge as one mutation. It installs a permanent fence
  for the canonical `(user_id, public_thread_id)` identity, drains active graph
  work, then holds the persistence lock through the final fence check and durable
  deletion. A concurrent fork uses the same lock, so purge deletes a completed
  fork or the fork fails closed. A consolidation already mutating completes
  before deletion; queued or extracting work sees the fence and cannot recreate
  purged memory afterward.
- Historical checkpoint forks reserve the exact thread/user active-run fence
  while holding the persistence lock across checkpoint read and branch write.
  A new turn or another fork for that identity fails closed until the write
  finishes; independent thread or user identities remain available.

##### Memory candidate schema

Pydantic model (canonical for validation/tests):

```python
from typing import Literal

from pydantic import BaseModel, Field

class MemoryCandidate(BaseModel):
    content: str
    kind: Literal["fact", "preference", "todo", "project_state"]
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

Implement a small session registry table in the Chat DB (see [src/persistence/chat_db.py](../../src/persistence/chat_db.py)):

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

Expose in Chat sidebar (orchestrated by [src/agents/coordinator.py](../../src/agents/coordinator.py)):

1. List checkpoints for current `thread_id` (reverse chronological).
2. User selects one checkpoint to fork.
3. Fork operation:
   - call `coordinator.fork_from_checkpoint(...)`, which uses the same
     user-scoped config and `graph.update_state(...)` to create a new checkpoint
     id
   - hold the coordinator persistence mutation lock across checkpoint read and
     branch creation
   - reserve the same active-run fence used by graph turns until branch creation
     finishes; reject forks after hard purge or while that identity is active
4. Resume:
   - persist the forked checkpoint ID as the branch head and pass it through the
     coordinator's next async graph run.
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
   - Rely on `langgraph-checkpoint-sqlite` filter-key validation and keep the package within the `>=3.0.3,<4.0.0` constraint.
4. **Filesystem escape**
   - Validate the Chat DB path remains under `settings.data_dir` unless user explicitly overrides.
5. **Secret leakage**
   - Never log API keys; never include raw messages in telemetry.

## Testing Strategy

### Unit tests

- Session registry CRUD
- Embeddings adapter produces correct vector dims and deterministic outputs (using Mock embed model)
- Native store migration canonicalizes origin, preserves active TTL, validates every copied value, and retains legacy tables on failure
- Namespace scoping and filters
- Exact hard purge removes only the matching user and public thread namespace
- Deadline-expired memory mutations perform no write or delete
- Explicit and consolidation retries resolve to one deterministic memory key
- Stale consolidation decisions cannot overwrite later explicit writes
- Sidebar and tool writes resolve to the same key
- Deadline expiry stops later apply and eviction mutations
- Consolidation logs exclude raw identifiers, keys, content, and exceptions
- Purge fencing blocks queued, extracting, and future consolidation for the deleted session
- Consolidation request timeouts consume the remaining deadline, and worker capacity is held until provider exit
- Path validation (reject `..` / symlink escapes)

### Integration tests (Streamlit AppTest)

- A real temporary SQLite database proves concurrent async graph persistence,
  same-public-thread user isolation, sync state/list/fork bridging, WAL setup,
  and connection closure on the owning loop.
- Create session → chat → restart AppTest → history persists
- Time travel: fork from checkpoint → resume → UI shows new branch head
- Long-term memory: store preference → recall it later
- Use `tests/helpers/apptest_utils.py` (`apptest_timeout_sec()`) for AppTest
  timeouts; keep tests offline by stubbing non-persistence seams (coordinator
  factory, provider badge health checks) unless explicitly under test.

### Security tests

- Telemetry payloads never include raw message content (assert keys only)
- Filter keys are not user-controlled in any query builder

## Rollout / Migration Notes

- V2 has no migration from v1 raw checkpoint thread IDs to the hashed `(user_id, public_thread_id)` identity. Startup rejects raw or structurally incompatible identities before graph compilation. Operators must stop DocMind, archive `data/chat.db` with `data/chat.db-wal` and `data/chat.db-shm`, and start v2 with a fresh Chat DB. V1 chat history remains available only in the archive.
- Native LangGraph tables are created on demand at startup or first Chat interaction.
- For databases without retained v1 checkpoint history, startup migrates active rows from the previous custom SQLite memory tables, validates the native copies, and drops the legacy item/vector tables. The migration is idempotent after an interrupted copy, but it does not migrate checkpoints.
- If a legacy JSON chat store file exists (from experimental runs), it is ignored by default.

## Performance Expectations / Guardrails

- No import-time heavy work in Streamlit pages; DB connections and coordinator initialization must be cached (`st.cache_resource`) or lazy.
- Recommended pattern:
  - `@st.cache_resource` factories for the Chat DB and coordinator to avoid
    repeated connections on Streamlit reruns
  - pass `settings.chat.sqlite_path` to the coordinator; do not create a second
    synchronous saver in the UI
- Add a performance test simulating rapid messages + session switches to assert DB connection counts remain bounded (no leaks across reruns).
- Memory consolidation runs in the background (debounced) to avoid blocking the hot path (Streamlit fragments `run_every` per ADR-052 when available).

## RTM Updates

Completed:

- `docs/specs/requirements.md` now cites `SPEC-041/ADR-058` for FR-022 and includes FR-030..032 as implemented.
- `docs/specs/traceability.md` includes FR-022 + FR-030..032 rows pointing to the current modules and tests.
