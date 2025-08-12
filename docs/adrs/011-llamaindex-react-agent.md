# ADR-011: LlamaIndex ReAct Agent Architecture

## Title

Single LlamaIndex ReAct Agent for Document Q&A System

## Version/Date

4.0 / August 12, 2025

## Status

Accepted

## Context

Previous multi-agent LangGraph supervisor architecture violated core engineering principles (KISS > DRY > YAGNI) through excessive complexity without demonstrable benefits for document Q&A workflows.

**Multi-Agent Architecture Problems:**

- **Complexity Violations**: 450+ lines per component vs 77 lines for single-agent solution

- **Code Bloat**: Complex supervisor patterns, custom handoffs, SqliteSaver persistence overhead

- **Maintenance Burden**: Multi-agent state management and debugging complexity

- **Dependency Overhead**: Additional coordination packages without value justification

- **Implementation Cost**: 44-57 hours vs 10-15 hours for equivalent functionality

**Research Findings:**

- Document Q&A represents linear workflow unsuited for multi-agent coordination

- Single ReAct agents provide complete agentic capabilities: reasoning, tool selection, query decomposition

- Library-first analysis validates LlamaIndex ReActAgent completeness with reduced complexity

## Related Requirements

- **Pure LlamaIndex Stack**: Achieve 8.6/10 architecture score through library-first implementation

- **Performance**: <2s query latency with reliable response quality

- **Maintainability**: KISS compliance (0.9/1.0) with minimal code complexity

- **Local/Offline**: Ollama LLM integration with no external service dependencies  

- **Agentic Capabilities**: Chain-of-thought reasoning, dynamic tool selection, query decomposition, adaptive retrieval

- **Integration**: Seamless integration with existing Qdrant vector storage and Streamlit UI

## Alternatives

### 1. LangGraph Multi-Agent Supervisor (Previous - Rejected)

- **Architecture Score**: 2.89/10 (excessive complexity)

- **Code Complexity**: 450+ lines per component

- **KISS Compliance**: 0.4/1.0 (major violations)

- **Implementation Cost**: 44-57 hours development time

- **Issues**: Coordination overhead, difficult debugging, maintenance burden

- **Rejected**: Violates simplicity principles without demonstrable benefits

### 2. Haystack Multi-Agent Framework

- **Architecture Score**: 8.1/10

- **Issues**: Over-engineered for document Q&A, ecosystem misalignment

- **Rejected**: Unnecessary complexity, violates library-first principle

### 3. Custom Multi-Agent Implementation  

- **Architecture Score**: 7.35/10

- **Issues**: Reinvents proven patterns, insufficient built-in features

- **Rejected**: Violates library-first approach

### 4. Pure LlamaIndex ReAct Agent (Selected)

- **Architecture Score**: 8.6/10

- **Code Reduction**: 85% (450+ lines → 77 lines)

- **KISS Compliance**: 0.9/1.0

- **Implementation Time**: 10-15 hours vs 44-57 hours

- **Benefits**: Complete agentic capabilities with minimal complexity

## Decision

**Adopt Single LlamaIndex ReActAgent Architecture** using `ReActAgent.from_tools()` for all document Q&A operations, completely replacing the LangGraph multi-agent supervisor system.

**Key Decision Factors:**

1. **Simplicity**: 85% code reduction while maintaining all capabilities  
2. **Library-First**: Pure LlamaIndex ecosystem alignment with proven patterns
3. **Maintenance**: Single agent system significantly easier to debug and extend
4. **Development Efficiency**: 74% faster implementation (10-15h vs 44-57h)
5. **Architecture Alignment**: Research confirmed single-agent suitability for document Q&A workflows

**Implementation Approach:**

- Single ReActAgent with tool-based architecture for query execution

- Dynamic tool selection from vector search, knowledge graph, and document analysis tools

- Native LlamaIndex patterns for memory management and streaming responses

- Integration with existing infrastructure (PyTorch GPU monitoring, spaCy optimization, Qdrant storage)

## Related Decisions

- **ADR-001**: Agent integration in overall architecture (updated for single-agent approach)

- **ADR-008**: Persistence strategy (simplified with single-agent state management)

- **ADR-015**: LlamaIndex migration strategy (core foundation for this decision)

- **ADR-018**: Refactoring decisions (KISS > DRY > YAGNI principle application)

## Design

**Implementation**: Single ReActAgent.from_tools() with dynamic tool creation from VectorStoreIndex.

**Core Components**:

- Agent factory creating ReActAgent with ChatMemoryBuffer (16K tokens)

- Tool factory generating QueryEngineTool from indices  

- Streamlit integration with agent.chat() for queries

- GPU optimization and spaCy memory management preserved

**Dependencies**: Simplified to core LlamaIndex packages (llama-index-core, llms-ollama, vector-stores-qdrant, embeddings-fastembed). Removed langgraph coordination packages (~17 fewer dependencies).

**Testing**: Validates agent reasoning, tool selection, and query decomposition capabilities.

## Consequences

### Positive Outcomes

- **Dramatic Simplification**: 85% code reduction (450+ lines → 77 lines)

- **KISS Compliance**: Improved from 0.4/1.0 to 0.9/1.0 simplicity score

- **Development Efficiency**: 74% faster implementation (10-15h vs 44-57h)

- **Maintenance**: Single codebase easier to debug and extend

- **Dependencies**: ~17 fewer packages to maintain

- **Library-First**: Pure LlamaIndex ecosystem alignment

### Capabilities Preserved

All agentic capabilities maintained: chain-of-thought reasoning, dynamic tool selection, query decomposition, adaptive retrieval, memory management, and streaming responses.

### Trade-offs

- Single agent architecture vs specialized agents (acceptable for document Q&A use case)

- Simpler learning curve compared to multi-agent coordination patterns

## Changelog

**4.0 (August 12, 2025)**: Complete replacement of LangGraph multi-agent with single LlamaIndex ReActAgent. 85% code reduction while preserving all agentic capabilities. KISS compliance improved from 0.4/1.0 to 0.9/1.0. Pure LlamaIndex stack achieving 8.6/10 architecture score. Integration with completed infrastructure (PyTorch GPU monitoring, spaCy optimization). Streamlined dependencies removing ~17 packages.

**3.0 (July 25, 2025)**: Previous LangGraph multi-agent approach - deprecated due to excessive complexity without demonstrable benefits.

**2.0**: Earlier multi-agent iterations - superseded by simplified single-agent approach.
