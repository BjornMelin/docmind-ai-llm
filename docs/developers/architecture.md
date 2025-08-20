# DocMind AI Architecture Overview

## High-Level Components

- **Frontend**: Streamlit UI for document uploads, configuration, results, and chat interface
- **LLM Backend**: vLLM with FlashInfer attention for Qwen3-4B-Instruct-2507-FP8 model inference
- **Multi-Agent Orchestration**: 5-agent LangGraph supervisor system with specialized agents
- **Vector Storage**: Qdrant for hybrid search with dense/sparse embeddings
- **Context Management**: 128K context window optimization with FP8 KV cache
- **Performance Layer**: FP8 quantization, parallel tool execution, CUDA 12.8+ optimization

## Multi-Agent Coordination System

DocMind AI implements a sophisticated multi-agent architecture using LangGraph's supervisor pattern to coordinate specialized agents for complex document analysis tasks.

### Agent Architecture

```mermaid
graph TD
    A[User Query] --> B[MultiAgentCoordinator]
    B --> C[RouterAgent]
    C --> D{Complexity Analysis}
    D -->|Simple| E[Direct Retrieval]
    D -->|Complex| F[PlannerAgent]
    F --> G[RetrievalAgent]
    G --> H[SynthesisAgent]
    H --> I[ValidationAgent]
    I --> J[Validated Response]
    E --> J
    
    K[Fallback to Basic RAG] -.-> J
    B -.-> K
```

### Specialized Agents

| Agent | Responsibility | Performance Target | Key Features |
|-------|----------------|-------------------|--------------|
| **Query Router Agent** | Query analysis and retrieval strategy selection | <50ms | Strategy caching, confidence scoring |
| **Query Planner Agent** | Complex query decomposition into manageable sub-tasks | <100ms | Dependency mapping, resource allocation |
| **Retrieval Expert Agent** | Multi-modal retrieval with 128K context utilization | <150ms | Hybrid search, DSPy optimization, reranking |
| **Result Synthesizer Agent** | Multi-source result integration and conflict resolution | <100ms | Evidence ranking, citation generation |
| **Response Validator Agent** | Quality assurance and accuracy validation | <75ms | Consistency checking, confidence assessment |

### Agent Communication Patterns

- **Supervisor Orchestration**: LangGraph supervisor coordinates agent interactions
- **Shared State**: Context and metadata flow through MessagesState
- **Tool Integration**: Standardized @tool functions for consistent agent interfaces
- **Fallback Mechanisms**: Graceful degradation to basic RAG on agent failures

## Data Flow

### Document Processing Flow
1. User uploads docs → Loaded/split in src/utils/.
2. Indexed in Qdrant with hybrid embeddings (Jina v4 dense, FastEmbed sparse).
3. Analysis: Multi-agent system processes with specialized agents → Structured via Pydantic.
4. Chat: Multi-agent coordination with context preservation and validation.
5. GPU: torch.cuda for embeddings/reranking if enabled.

### Multi-Agent Query Processing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as Coordinator
    participant R as Router
    participant P as Planner
    participant RA as Retrieval
    participant S as Synthesis
    participant V as Validator
    
    U->>C: Document query
    C->>R: Analyze complexity
    alt Simple Query
        R->>RA: Direct retrieval
        RA->>V: Validate response
        V->>C: Quality-checked result
    else Complex Query
        R->>P: Plan decomposition
        P->>RA: Execute sub-tasks
        RA->>S: Combine results
        S->>V: Validate synthesis
        V->>C: Comprehensive response
    end
    C->>U: Final response
```

## Key Technologies

### Core Infrastructure

- **Embeddings**: HuggingFace (Jina v4), FastEmbed (SPLADE++).

- **Multi-Agent Framework**: LangGraph supervisor pattern with specialized agents.

- **Optimization**: PEFT for efficiency, late chunking with NLTK, DSPy for query optimization.

- **Error Handling**: Tenacity for retry logic with exponential backoff.

- **Logging**: Loguru for structured logging with automatic rotation.

- **Caching**: Diskcache for document processing (90% performance improvement).

### Multi-Agent Technologies

- **LangGraph**: Native supervisor pattern for agent orchestration
- **Tool Integration**: @tool decorator with InjectedState for agent communication  
- **State Management**: MessagesState for context preservation across agents
- **Memory Systems**: InMemorySaver for conversation continuity
- **Performance Monitoring**: Built-in timing and quality metrics
- **Fallback Systems**: Graceful degradation to basic RAG on failures

## Architecture Design

DocMind AI follows modern library-first principles for reliability and maintainability:

### Core Design Principles

- **Robust Error Handling**: Tenacity-based retry logic with exponential backoff

- **Structured Logging**: Loguru integration for comprehensive monitoring and debugging

- **Type-Safe Configuration**: Pydantic BaseSettings for validated configuration management

- **High-Performance Caching**: Document processing cache layer delivers 90% speed improvement

- **Production-Ready Components**: Robust reliability with comprehensive error recovery

### Multi-Agent Design Principles

- **Agent Specialization**: Each agent has a focused responsibility and optimized performance
- **Supervisor Coordination**: LangGraph supervisor manages agent interactions and state
- **Performance Budgets**: Strict timing constraints (<300ms total overhead)
- **Graceful Degradation**: Automatic fallback to basic RAG when agents fail
- **Context Preservation**: Conversation continuity across multi-turn interactions
- **Quality Assurance**: Built-in validation and quality scoring for all responses

## Model Update Architecture Changes

### Qwen3-4B-Instruct-2507-FP8 Integration

The architecture has been enhanced with the Qwen3-4B-Instruct-2507-FP8 model integration, providing significant improvements:

#### Model Layer Architecture

```mermaid
graph TD
    A[DocMind AI Core] --> B[vLLM Backend]
    B --> C[Qwen3-4B-Instruct-2507-FP8]
    C --> D[FlashInfer Attention]
    D --> E[FP8 Quantization Engine]
    E --> F[128K Context Manager]
    F --> G[RTX 4090 Hardware]
    
    H[Multi-Agent Coordinator] --> B
    I[LlamaIndex Pipeline] --> B
    
    subgraph "Performance Optimizations"
        D --> J[2x Speedup vs Standard]
        E --> K[~50% Memory Reduction]
        F --> L[131,072 Token Context]
    end
    
    subgraph "Hardware Layer"
        G --> M[CUDA 12.8+]
        M --> N[16GB VRAM Optimized]
    end
```

#### Performance Improvements

| Metric | Previous (Qwen3-14B) | Current (Qwen3-4B-FP8) | Improvement |
|--------|---------------------|----------------------|-------------|
| **Model Size** | 14B parameters | 4.23B parameters | 70% reduction |
| **Context Window** | 32K tokens | 128K tokens | 4x increase |
| **Decode Speed** | 60-100 tok/s | 120-180 tok/s | 2x improvement |
| **Prefill Speed** | 400-600 tok/s | 900-1400 tok/s | 2.3x improvement |
| **VRAM Usage** | 20-24GB | 12-14GB | 40% reduction |
| **Context Efficiency** | Limited 32K | Full 128K supported | 400% increase |

#### Architectural Benefits

- **Memory Efficiency**: FP8 quantization with FP8 KV cache reduces memory footprint by ~50%
- **Context Scalability**: 128K context window enables processing of large documents without chunking
- **Performance Optimization**: FlashInfer attention backend provides RTX 4090-specific optimizations
- **Multi-Agent Enhancement**: Larger context window improves agent coordination and information synthesis

### Enhanced Context Management

The 128K context window integration requires sophisticated context management:

```mermaid
graph LR
    A[Input Context] --> B[Context Analyzer]
    B --> C{Size > 120K?}
    C -->|No| D[Direct Processing]
    C -->|Yes| E[Context Optimizer]
    
    E --> F[Priority Content]
    E --> G[Conversation History]
    E --> H[Retrieval Results]
    
    F --> I[128K Context Window]
    G --> I
    H --> I
    
    I --> J[Qwen3-4B Model]
    J --> K[Response Generation]
    D --> J
```

## Performance Characteristics

### Multi-Agent System Performance

| Operation | Target Time | Fallback Time | Model Update Impact |
|-----------|-------------|---------------|-------------------|
| Simple query routing | <50ms | N/A | Unchanged |
| Query planning | <100ms | Bypass | Improved with 128K context |
| Document retrieval | <150ms | <500ms | Faster with larger context |
| Result synthesis | <100ms | Skip | Better quality with FP8 |
| Response validation | <75ms | Basic check | Enhanced accuracy |
| **Total coordination overhead** | **<300ms** | **<3s fallback** | **20% improvement** |

### Model Performance Targets

| Metric | Target Range | Expected with FlashInfer |
|--------|--------------|--------------------------|
| **Decode Speed** | 100-160 tok/s | 120-180 tok/s |
| **Prefill Speed** | 800-1300 tok/s | 900-1400 tok/s |
| **VRAM Usage** | 12-14GB target | 12-14GB for 128K context |
| **Context Utilization** | Up to 120K tokens | 131,072 tokens supported |
| **Multi-Agent Overhead** | <300ms | <250ms with optimization |

### Integration Benefits

- **Document Processing**: Large documents can be processed without aggressive chunking
- **Conversation Memory**: Extended conversation history with 128K context buffer
- **Multi-Agent Coordination**: Better information synthesis across specialized agents
- **Performance Scalability**: FP8 optimizations enable efficient scaling within hardware constraints

See [model-update-implementation.md](model-update-implementation.md) for detailed implementation guide and [../adrs/](../adrs/) for all architectural decisions, including [ADR-011](../adrs/ADR-011-agent-orchestration-framework.md) for detailed multi-agent design decisions.
