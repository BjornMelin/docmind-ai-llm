# ADR-001: Architecture Overview

## Title

High-Level Architecture for DocMind AI

## Version/Date

3.0 / July 25, 2025

## Status

Accepted

## Context

DocMind AI is a local, offline RAG application for document analysis, emphasizing high-performance retrieval, multimodal support, agentic workflows, and efficiency without API keys or internet. The architecture uses LlamaIndex for RAG pipelines, Unstructured for parsing, LangGraph for agents. Local multi-process supported via SQLite WAL/diskcache locks (no distributed scaling for MVP—reassess later if needed). Key needs: hybrid search (dense/sparse), KG for relations, multimodal (text/images/tables), multi-agent coordination, and caching/persistence for sessions.

## Related Requirements

- Offline/local operation (no cloud APIs like LlamaParse).
- Hybrid retrieval with SPLADE++/BGE-Large/Jina v4.
- Multimodal parsing (PDFs with images/tables via Unstructured).
- Agentic RAG with LangGraph supervisor (handoffs, human-in-loop).
- Efficient chunking (1024/200 overlap via SentenceSplitter).
- Multi-stage querying (QueryPipeline).
- Persistence/caching (SQLite/diskcache).
- Local multi-process (SQLite WAL for concurrent).

## Alternatives

- LangChain: Heavier, less offline (deprecated).
- Custom: High maintenance.
- Distributed with Redis: Overengineering for local/single-user (optional later).

## Decision

LlamaIndex for indexing/retrieval/pipelines (VectorStoreIndex/Qdrant, MultiModalVectorStoreIndex/Jina v4, KGIndex/spaCy, QueryPipeline multi-stage). LangGraph for agents (supervisor/create_react_agent with local Ollama, checkpointer=MemorySaver, interrupts/human-in-loop, handoffs/Send API). Unstructured for parsing (hi_res). SentenceSplitter in IngestionPipeline for chunking. SQLite/diskcache for persistence (StorageContext).

## Related Decisions

- ADR-015 (LlamaIndex migration).
- ADR-011 (LangGraph agents).
- ADR-016 (Multimodal Jina v4/Unstructured).
- ADR-008 (Persistence with SQLite/diskcache).

## Design

- **Ingestion**: UnstructuredReader.load_data(file_path, strategy="hi_res") → IngestionPipeline([SentenceSplitter(AppSettings.chunk_size/overlap), MetadataExtractor()]) → nodes.
- **Indexing/Retrieval**: HybridFusionRetriever (dense=FastEmbedEmbedding(AppSettings.dense_embedding_model), sparse=SparseTextEmbedding(AppSettings.sparse_embedding_model), alpha=AppSettings.rrf_fusion_alpha) with QdrantVectorStore. MultiModalVectorStoreIndex (image_embed_model=HuggingFaceEmbedding("jinaai/jina-embeddings-v4")). KGIndex.from_documents(nodes, extractor=spaCy).
- **Querying**: QueryPipeline(chain=[retriever, ColbertRerank, synthesizer], async_mode=True, parallel=True).
- **Agents**: LangGraph StateGraph with supervisor (routing via analyze_query_complexity), create_react_agent workers (local Ollama LLM/LlamaIndex tools), checkpointer=MemorySaver for persistence, interrupts for human-in-loop, handoffs via Send API.
- **Caching/Persistence**: StorageContext with SQLite backend (kv_store=SimpleKVStore.from_sqlite(AppSettings.cache_db_path)); Wrap embeds with diskcache.
- **Local Multi-Process**: SQLite WAL mode/diskcache locks for concurrent (e.g., multiprocessing.Pool for parallel indexing if heavy load—no distributed).
- **UI**: Streamlit: Toggles (st.checkbox for gpu_acceleration/chunk_size); No distributed toggle (reassess later).
- **Diagram** (Mermaid):

```mermaid
graph TD
    A[Streamlit UI (Toggles: AppSettings)] --> B[Upload (async upload_section, st.status/progress/error)]
    B --> C[Parse (UnstructuredReader hi_res → docs)]
    C --> D[Chunk/Extract (IngestionPipeline: SentenceSplitter(AppSettings.chunk_size/overlap) + MetadataExtractor → nodes)]
    D --> E[Index/Embed (HybridFusionRetriever dense/sparse + MultiModalVectorStoreIndex Jina v4 512D + KGIndex spaCy)]
    E --> F[Query (QueryPipeline: chain=[retriever, ColbertRerank, synthesizer], async/parallel)]
    F --> G[Agents (LangGraph supervisor/create_react_agent workers with local Ollama LLM/tools, checkpointer=MemorySaver, interrupts/human-in-loop, handoffs/Send API)]
    G --> A[Response/Chat (markdown for sources/multimodal)]
    H[SQLite/Diskcache (StorageContext for stores, @memoize for embeds; WAL/locks for local multi-process)] <--> E & F & G
```

- **Implementation Notes**: If heavy load, use multiprocessing.Pool(map=embed, chunks)—SQLite WAL enables concurrent writes. No Redis (overengineering—local sufficient).
- **Testing**: tests/test_performance_integration.py: def test_local_multi_process(): with multiprocessing.Pool(): results = pool.map(query, queries); assert no locks/errors; latency < threshold.

## Consequences

- Offline/local focus (SQLite/diskcache for all).
- Scalable locally (multi-process via WAL/locks).
- Maintainable (no distributed complexity for MVP).

- If distributed needed later: Reassess with new ADR.
- Future: Expand Pool usage if benchmarks show gains.

**Changelog:**  

- 3.0 (July 25, 2025): Removed distributed_mode/Redis (overengineering); Emphasized local multi-process with SQLite WAL/diskcache locks; Enhanced integrations/diagram/testing for dev.
- 2.0: Previous updates for Unstructured/Jina/pipelines.
