# ADR-008: Session Persistence

## Title

Persistence and Caching for Sessions and Data

## Version/Date

5.0 / August 13, 2025

## Status

Accepted

## Context

Offline persistence for chat/state/documents/KV/KG/vector (SQLite queryable/scalable ~GBs, native IngestionCache for document processing). For local multi-process (e.g., concurrent indexing), use SQLite WAL mode—no distributed scaling (reassess later if needed). Single ReAct agent simplifies session persistence through ChatMemoryBuffer vs complex multi-agent state management.

## Related Requirements

- Offline/local (SQLite primary, native IngestionCache for document processing).

- Unified sharing (StorageContext for LlamaIndex, ChatMemoryBuffer for single agent).

- Local multi-process (SQLite WAL for concurrent writes).

- Simplified session management (single agent vs multi-agent coordination state).

## Alternatives

- In-memory: Volatile (rejected).

- Redis: Overhead (local server)—not needed for local app.

- Complex LangGraph state management: Over-engineered for single agent.

## Decision

SQLite for structured stores (KV/chat/KG via StorageContext with WAL), native IngestionCache for document processing (80-95% re-processing reduction). Single ReActAgent: ChatMemoryBuffer(token_limit=8192) for session persistence—simpler than multi-agent SqliteSaver coordination. No distributed—local multi-process sufficient.

## Related Decisions

- ADR-001 (Integrates across components).

- ADR-020 (LlamaIndex Settings Migration - unified configuration for storage and caching).

- ADR-022 (Tenacity Resilience Integration - robust database operations with retry patterns).

## Design

- **Stores**: In utils.py: from llama_index.core.storage import StorageContext; context = StorageContext.from_defaults(kv_store=SimpleKVStore.from_sqlite(AppSettings.cache_db_path or "docmind.db", wal=True)); index = VectorStoreIndex(..., storage_context=context).

- **Caching**: Native IngestionCache for document transformations (automatic caching of embeddings, chunks, metadata without custom code).

- **Agent Memory**: In agent_factory.py: from llama_index.core.memory import ChatMemoryBuffer; memory = ChatMemoryBuffer.from_defaults(token_limit=8192). ReActAgent.from_tools(memory=memory) for session persistence—simpler than multi-agent checkpointers.

- **Integration**: Use context in indexes/pipelines (e.g., MultiModalVectorStoreIndex(storage_context=context)). IngestionPipeline with IngestionCache for automatic document processing optimization. Single agent with ChatMemoryBuffer vs complex workflow compilation. For multi-process: SQLite WAL enables concurrent access.

- **Implementation Notes**: IngestionCache automatically handles document fingerprinting and transformation caching. No Redis or complex custom caching—native LlamaIndex features sufficient.

- **Testing**: tests/test_real_validation.py: def test_persistence_ingestion_cache(): cache = IngestionCache(); pipeline = IngestionPipeline(cache=cache); # Process same document twice; assert second processing uses cache; def test_local_concurrent(): conn = sqlite3.connect(AppSettings.cache_db_path, check_same_thread=False); # Simulate concurrent writes; assert no errors.

## Consequences

- Offline/local (SQLite for structured data, native IngestionCache for document processing).

- Scalable locally (multi-process via WAL, automatic cache optimization via IngestionCache).

- Maintainable (no distributed complexity for MVP).

- If distributed needed later: Reassess with new ADR.

- Future: Expand Pool usage if benchmarks show gains.

**Changelog:**  

- 5.0 (August 13, 2025): Replaced diskcache with native IngestionCache for 80-95% re-processing reduction without custom code. Aligned with simplified architecture using LlamaIndex native features.

- 4.0 (August 12, 2025): Simplified agent session persistence using ChatMemoryBuffer vs complex multi-agent SqliteSaver coordination. Single agent memory management pattern aligned with library-first principles.

- 3.0 (July 25, 2025): Removed Redis/distributed_mode (overengineering); Emphasized local multi-process with SQLite WAL; Enhanced integrations/testing for dev.

- 2.0: Previous hybrid details.
