# ADR-015: LlamaIndex Migration

## Title

Migration from LangChain to LlamaIndex

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

LangChain initial but LlamaIndex superior for offline RAG (built-in pipelines/multimodal/hybrid/chunking/stores—e.g., QueryPipeline async, MultiModalIndex local VLM, ReActAgent document workflows, native Qdrant integration).

## Related Requirements

- Offline integrations (local parsing/pipelines)

- Advanced RAG (hybrid/multimodal/KG)

- Performance optimization for document processing

## Alternatives

- Stay LangChain: Less offline features (e.g., no native Unstructured)

- Partial Migration: Inefficient and increases maintenance complexity

## Decision

Full migration to LlamaIndex (indexing/retrieval/pipelines/multimodal/KG/chunking/stores). Keep LangGraph for agents (integrates via tools).

## Related Decisions

- ADR-001 (Uses LlamaIndex core)

- ADR-010 (Deprecates LangChain)

## Design

- **Migration Steps**: Replace chains with QueryPipeline, loaders with UnstructuredReader, agents with LangGraph (tools from LlamaIndex retrievers)

- **Integration**: utils.py/app.py/agent_factory.py LlamaIndex-centric (e.g., VectorStoreIndex, QueryPipeline in tools). LangGraph workers use LlamaIndex tools

- **Implementation Notes**: Ensure offline (local Ollama in LlamaIndex LLM). No LangChain imports

- **Testing**: tests/test_real_validation.py: def test_migration_offline(): from llama_index import *; assert no LangChain; test_pipeline(); test_offline_query()

## Consequences

- Offline-optimized RAG (pipelines/multimodal)
- Simpler/maintainable (unified library)

- Initial refactor (but complete now)

**Changelog:**  

- 2.0 (July 25, 2025): Detailed offline reasons/integrations/tests; Kept LangGraph hybrid.
