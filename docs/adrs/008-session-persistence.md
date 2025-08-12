# ADR-008: Session Persistence

## Title

Persistence and Caching for Sessions and Data

## Version/Date

3.0 / July 25, 2025

## Status

Accepted

## Context

Offline persistence for chat/state/documents/KV/KG/vector (SQLite queryable/scalable ~GBs, diskcache fast embeds). For local multi-process (e.g., concurrent indexing/agents), use SQLite WAL mode/diskcache locks—no distributed scaling (reassess later if needed).

## Related Requirements

- Offline/local (SQLite primary, diskcache supplement—toggle AppSettings.cache_backend).
- Unified sharing (StorageContext for LlamaIndex, checkpointer for LangGraph).
- Local multi-process (SQLite WAL for concurrent writes).

## Alternatives

- In-memory: Volatile (rejected).
- Redis: Overhead (local server)—not needed for local app.

## Decision

SQLite for structured stores (KV/chat/KG via StorageContext with WAL), diskcache for quick embeds. LangGraph: MemorySaver (in-memory) or SqliteSaver (persistent). No distributed—local multi-process sufficient.

## Related Decisions

- ADR-001 (Integrates across components).

## Design

- **Stores**: In utils.py: from llama_index.core.storage import StorageContext; context = StorageContext.from_defaults(kv_store=SimpleKVStore.from_sqlite(AppSettings.cache_db_path or "docmind.db", wal=True)); index = VectorStoreIndex(..., storage_context=context).
- **Caching**: Wrap with diskcache.memoize (thread-safe locks for multi-process).
- **Agents**: In agent_factory.py: from langgraph.checkpoint.sqlite import SqliteSaver; checkpointer=SqliteSaver.from_conn_string(AppSettings.cache_db_path).
- **Integration**: Use context in indexes/pipelines (e.g., MultiModalVectorStoreIndex(storage_context=context)). checkpointer in workflow.compile(). For multi-process: Wrap Pool in if heavy load (e.g., pool.map(embed, chunks))—SQLite WAL enables concurrent.
- **Implementation Notes**: Wrap Pool in try/except logger.error(e). No Redis—local sufficient.
- **Testing**: tests/test_real_validation.py: def test_persistence_multi_process(): context.persist(); with multiprocessing.Pool(): results = pool.map(load_context, [AppSettings.cache_db_path]); assert results[0] == original; def test_local_concurrent(): conn = sqlite3.connect(AppSettings.cache_db_path, check_same_thread=False); # Simulate concurrent writes; assert no errors.

## Consequences

- Offline/local (SQLite/diskcache for all).
- Scalable locally (multi-process via WAL/locks).
- Maintainable (no distributed complexity for MVP).

- If distributed needed later: Reassess with new ADR.
- Future: Expand Pool usage if benchmarks show gains.

**Changelog:**  

- 3.0 (July 25, 2025): Removed Redis/distributed_mode (overengineering); Emphasized local multi-process with SQLite WAL/diskcache locks; Enhanced integrations/testing for dev.
- 2.0: Previous hybrid details.
