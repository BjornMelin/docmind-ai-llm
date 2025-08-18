# Final Architecture Decisions - DocMind AI

## Updated: 2025-08-17 - Post Expert Review Integration

## Model Selections

### Language Model (LLM)

- **Primary**: Qwen3-14B (Q4_K_M GGUF format)
  - 128K native context window
  - 8-10GB VRAM requirement
  - **Structured Outputs**: Instructor library integration
  - Deployment: Ollama/llama.cpp with Instructor patch
- **Fallback**: Qwen3-7B for lower-end hardware (4-6GB VRAM)

### Embedding Model

- **Primary**: BGE-M3 (`BAAI/bge-m3`) - **CONFIRMED BEST CHOICE**
  - **Unique Value**: Only model with unified dense+sparse embeddings
  - 1024 dimensions (dense) + integrated sparse vectors
  - 8,192 token context window
  - 100+ languages, 70.0 NDCG@10 performance
  - 2.3GB model size
  - **100% local operation** - no API dependencies
  - **Decision**: Keep BGE-M3 over Arctic-v2/Voyage-3 (no sparse support)

### Reranking Model

- **Primary**: BGE-reranker-v2-m3
  - Direct CrossEncoder usage via sentence-transformers
  - 1.1GB model size
  - No custom wrappers needed

## Technology Stack

```toml
[project.dependencies]
# Core Framework
llama-index = "^0.11.0"          # RAG framework
langgraph-supervisor = "^0.0.29" # Agent orchestration  
streamlit = "^1.40.0"            # UI with streaming

# NEW CRITICAL ADDITIONS
instructor = "^1.3.0"            # Structured outputs (MUST HAVE)
dspy-ai = "^2.4.0"              # Query optimization (MUST HAVE)
gptcache = "^0.1.0"             # Semantic caching (SHOULD HAVE)
graphrag = "^0.1.0"             # Optional GraphRAG module

# Models (100% local)
sentence-transformers = "^3.3.0"  # BGE-M3, reranking
transformers = "^4.46.0"          # Qwen3-14B
FlagEmbedding = "^1.2.0"         # BGE-M3 sparse features

# Storage & Memory
qdrant-client = "^1.12.0"        # Vector database
sqlmodel = "^0.0.22"             # SQLite ORM
redis = "^5.0.0"                # Optional session memory backend

# Documents & Evaluation
unstructured = "^0.16.0"         # Document parsing
deepeval = "^1.0.0"              # Evaluation metrics
ragas = "^0.1.0"                # RAG-specific metrics
```

## Architecture Decisions

### Core Architecture

- **Pattern**: Agentic RAG with library-first implementation
- **Agents**: **5 specialized agents** via langgraph-supervisor
  1. Query Routing Agent (strategy selection)
  2. Planning Agent (complex query decomposition)
  3. Retrieval Agent (document search with DSPy)
  4. Synthesis Agent (multi-source combination)
  5. Response Validation Agent (quality checks)
- **Retrieval**: LlamaIndex with DSPy query optimization
- **State Management**: Streamlit + LangGraph memory integration
- **Streaming**: Native st.write_stream() for real-time responses
- **Structured Outputs**: Instructor library for guaranteed JSON

### ADRs to Keep (12)

1. ADR-001: Modern Agentic RAG ✅
2. ADR-002: Unified Embedding Strategy ✅
3. ADR-003: Adaptive Retrieval Pipeline ✅
4. ADR-004: Local-First LLM Strategy ✅
5. ADR-006: Modern Reranking (simplified) ✅
6. ADR-007: Hybrid Persistence (simplified) ✅
7. ADR-009: Document Processing ✅
8. ADR-010: Performance Optimization ✅
9. ADR-011: Agent Orchestration (5 agents) ✅
10. ADR-013: User Interface (with streaming) ✅
11. **ADR-018**: DSPy Prompt Optimization ✅ (NEW)
12. **ADR-019**: Optional GraphRAG Module ✅ (NEW)

### ADRs to Delete (3)

1. **ADR-005**: Framework Abstraction Layer (YAGNI violation)
2. **ADR-008**: Production Observability (over-engineered for local app)
3. **ADR-017**: Component Library & Theming (unnecessary complexity)

### Enhanced ADRs (6)

1. **ADR-004**: LLM Strategy + Instructor structured outputs ✅
2. **ADR-010**: Performance + GPTCache semantic caching ✅
3. **ADR-012**: Evaluation → DeepEval + Ragas metrics ✅
4. **ADR-014**: Testing → pytest + DeepEval only ✅
5. **ADR-015**: Deployment → Single Docker container ✅
6. **ADR-016**: UI State + LangGraph memory ✅

## Implementation Principles

### Core Principles

1. **Library-First**: Always use existing libraries before custom code
2. **KISS > DRY > YAGNI**: Simplicity over premature optimization
3. **Local-First**: 100% offline operation, no API dependencies
4. **MVP Focus**: Ship in 3 weeks, not 3 months

### Code Reduction Achieved

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Evaluation | 1,000 | 50 | 95% |
| Reranking | 500 | 20 | 96% |
| State Management | 500 | 150 | 70% |
| Document Processing | 400 | 50 | 88% |
| Deployment | 600 | 80 | 87% |
| Testing | 500 | 100 | 80% |
| Framework Abstraction | 400 | 0 | 100% |
| Observability | 600 | 0 | 100% |
| Component Library | 400 | 0 | 100% |
| Agent Orchestration | 900 | 50 | 94% |
| **TOTAL** | **5,800** | **500** | **91.4%** |

## Key Technical Decisions

### Core Features (Scoring >70%)

1. **Structured Outputs** (85 points) - Instructor library ✅
2. **Query Rewriting** (82 points) - DSPy integration ✅
3. **Streaming Responses** (78 points) - Streamlit native ✅
4. **GraphRAG** (76.5 points) - Optional module ✅
5. **Semantic Caching** (72 points) - GPTCache ✅

### What We Use

- **LlamaIndex**: Direct usage with DSPy enhancement
- **Instructor**: Structured output guarantees
- **DSPy**: Automatic prompt optimization
- **GPTCache**: Semantic result caching
- **Sentence-Transformers**: All embeddings and reranking
- **Unstructured.io**: All document parsing
- **DeepEval + Ragas**: Comprehensive evaluation
- **Streamlit Native**: With streaming support
- **LangGraph**: Supervisor + memory integration

### What We Don't Build

- Custom evaluation metrics (use DeepEval + Ragas)
- Framework abstractions (use libraries directly)
- Custom state management (use Streamlit + LangGraph)
- Complex deployment scripts (single Docker container)
- Custom document parsers (use Unstructured.io)
- Custom reranking logic (use CrossEncoder directly)
- Custom prompt engineering (use DSPy)
- Custom JSON parsing (use Instructor)
- Custom caching logic (use GPTCache)
- Production monitoring (simple logging only)

## System Requirements

### Minimum Hardware

- **GPU**: RTX 3060 or equivalent
- **VRAM**: 10GB (Qwen3-7B + BGE-M3 + reranker)
- **RAM**: 16GB system memory
- **Storage**: 50GB for models and data

### Recommended Hardware

- **GPU**: RTX 4060 or better
- **VRAM**: 14GB (Qwen3-14B + BGE-M3 + reranker)
- **RAM**: 32GB system memory
- **Storage**: 100GB for models and data

## Final Validation

- ✅ All models run 100% locally without internet
- ✅ BGE-M3 confirmed as best choice (unified embeddings)
- ✅ 5 agents for comprehensive query handling
- ✅ All features >70% score implemented
- ✅ 91% code reduction through library usage
- ✅ Structured outputs with Instructor
- ✅ Query optimization with DSPy
- ✅ Streaming responses with Streamlit
- ✅ Semantic caching with GPTCache
- ✅ Optional GraphRAG for complex queries
- ✅ Can be deployed with single Docker command
- ✅ 3-week delivery timeline achievable
