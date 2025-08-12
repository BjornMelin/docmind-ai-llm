# ADR-015: LlamaIndex Migration

## Title

Migration from LangChain to LlamaIndex

## Version/Date

3.0 / August 12, 2025

## Status

Accepted

## Context

LangChain initial but LlamaIndex superior for offline RAG (built-in pipelines/multimodal/hybrid/chunking/stores—e.g., QueryPipeline async, MultiModalIndex local VLM, ReActAgent document workflows, native Qdrant integration). Complete transition to single ReActAgent architecture eliminates multi-agent coordination complexity while maintaining all agentic capabilities through pure LlamaIndex implementation.

## Related Requirements

- Offline integrations (local parsing/pipelines)

- Advanced RAG (hybrid/multimodal/KG)

- Performance optimization for document processing  

- Simplified agent architecture (single ReAct agent vs multi-agent coordination)

- Library-first principle compliance (KISS > DRY > YAGNI)

## Alternatives

- Stay LangChain: Less offline features (e.g., no native Unstructured)

- Partial Migration: Inefficient and increases maintenance complexity

- LangGraph Multi-Agent: Over-engineered coordination for document Q&A workflows

## Decision

Complete migration to pure LlamaIndex stack (indexing/retrieval/pipelines/multimodal/KG/chunking/stores/agents). Single ReActAgent.from_tools() replaces all multi-agent patterns while providing complete agentic capabilities: reasoning, tool selection, query decomposition, and adaptive retrieval.

## Related Decisions

- ADR-001 (Uses LlamaIndex core architecture)

- ADR-010 (Deprecates LangChain completely)

- ADR-011 (Single ReAct agent implementation)

- ADR-020 (LlamaIndex native Settings adoption completes ecosystem migration)

## Design

- **Migration Steps**: Replace chains with QueryPipeline, loaders with UnstructuredReader, multi-agent system with single ReActAgent.from_tools()

- **Integration**: src/agents/agent_factory.py (77 lines) creates ReActAgent with QueryEngineTool from LlamaIndex indices. Pure LlamaIndex ecosystem - no LangGraph coordination packages.

- **Implementation Notes**: Single agent factory pattern. Ensure offline (local Ollama in LlamaIndex LLM). No LangChain or LangGraph imports. ~17 fewer dependency packages.

- **Testing**: tests/test_real_validation.py: def test_migration_offline(): from llama_index.core.agent import ReActAgent; assert no LangChain/LangGraph; test_single_agent(); test_offline_query()

## Consequences

- Offline-optimized RAG (pipelines/multimodal) with simplified agent architecture

- Dramatically simpler/maintainable (unified pure LlamaIndex library, 85% code reduction)

- Complete library-first migration (KISS principle compliance)

- Dependency reduction (~17 fewer packages to maintain)

- Single codebase easier to debug and extend

**Changelog:**  

- 3.0 (August 12, 2025): Complete migration to pure LlamaIndex stack including single ReActAgent replacement of LangGraph multi-agent system. 85% agent code reduction, ~17 dependency package reduction, full library-first compliance.

- 2.0 (July 25, 2025): Detailed offline reasons/integrations/tests; Kept LangGraph hybrid.
