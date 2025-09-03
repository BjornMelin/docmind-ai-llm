# ADR Index (Curated by Topic)

This page provides a quick, opinionated index of all Architectural Decision Records (ADRs), grouped by topic. For the full file list, view this directory.

---

## Architecture & Agents

- **[ADR-001 — Modern Agentic RAG Architecture](ADR-001-modern-agentic-rag-architecture.md)**  
  Scope: Multi-agent, modern RAG architecture and capabilities
- **[ADR-011 — Agent Orchestration Framework](ADR-011-agent-orchestration-framework.md)**  
  Scope: Supervisor/orchestration patterns and agent tool integration
- **[ADR-015 — Docker-First Local Deployment](ADR-015-deployment-strategy.md)**  
  Scope: Local deployment strategy and considerations
- **[ADR-027 — Implementation Experience (meta)](ADR-027-implementation-experience.md)**  
  Scope: Cross-ADR integration learnings and validation

## Retrieval, Embeddings & Graph

- **[ADR-002 — Unified Embedding Strategy with BGE-M3](ADR-002-unified-embedding-strategy.md)**  
  Scope: Unified dense/sparse embeddings centered on BGE-M3
- **[ADR-003 — Adaptive Retrieval Pipeline](ADR-003-adaptive-retrieval-pipeline.md)**  
  Scope: Router/strategy selection and hierarchical retrieval
- **[ADR-006 — Modern Reranking Architecture](ADR-006-reranking-architecture.md)**  
  Scope: CrossEncoder reranking (BGE-reranker-v2-m3), library-first usage
- **[ADR-019 — Optional GraphRAG Module](ADR-019-optional-graphrag.md)**  
  Scope: Property graph/graph retrieval integrations
- **[ADR-034 — Idempotent Indexing & Embedding Reuse](ADR-034-idempotent-indexing-and-embedding-reuse.md)**  
  Scope: Idempotent ingestion and reusable embeddings

## UI & State

- **[ADR-013 — User Interface Architecture](ADR-013-user-interface-architecture.md)**  
  Scope: Streamlit multipage architecture and native component usage
- **[ADR-016 — Streamlit Native State Management](ADR-016-ui-state-management.md)**  
  Scope: Session state and cache primitives without custom layers
- **[ADR-036 — Reranker UI Controls (normalize_scores and top_n)](ADR-036-reranker-ui-controls-normalize-topn.md)**  
  Scope: Minimal sidebar controls for CrossEncoder reranker

## Configuration

- **[ADR-024 — Unified Settings Architecture](ADR-024-configuration-architecture.md)**  
  Scope: Pydantic BaseSettings + LlamaIndex Settings integration

## Caching & Persistence

- **[ADR-030 — Cache Unification via LlamaIndex IngestionCache (DuckDBKVStore)](ADR-030-cache-unification-ingestioncache-duckdbkvstore.md)**  
  Scope: Document-processing cache, single-file DuckDB KV store
- **[ADR-031 — Local-First Persistence Architecture (Vectors, Cache, Operational Data)](ADR-031-local-first-persistence-architecture.md)**  
  Scope: Separation of vectors (Qdrant), processing cache (DuckDB), ops data (SQLite)
- **[ADR-035 — Application-Level Semantic Cache v1.1 with GPTCache (SQLite + FAISS)](ADR-035-semantic-cache-gptcache-sqlite-faiss.md)**  
  Scope: Feature-flagged application-level semantic cache (v1.1.0)

## Document Processing

- **[ADR-009 — Document Processing Pipeline](ADR-009-document-processing-pipeline.md)**  
  Scope: Unstructured-first chunking, parameterization, table isolation

## Performance

- **[ADR-010 — Performance Optimization Strategy](ADR-010-performance-optimization-strategy.md)**  
  Scope: FP8/vLLM optimizations and latency targets

## Testing & Quality

- **[ADR-012 — Evaluation Strategy](ADR-012-evaluation-strategy.md)**  
  Scope: Evaluation methodologies and metrics
- **[ADR-014 — Testing and Quality Validation Framework](ADR-014-testing-quality-validation.md)**  
  Scope: DeepEval + pytest strategy and CI integration
- **[ADR-029 — Modern Testing Strategy with Boundary Testing](ADR-029-testing-strategy.md)**  
  Scope: Unit/integration boundary tests and coverage focus

## Prompt & LLM Optimization

- **[ADR-018 — Automatic Prompt Optimization with DSPy](ADR-018-prompt-optimization-dspy.md)**  
  Scope: DSPy-based prompt optimization patterns
- **[ADR-020 — Dynamic Prompt Template & Response Configuration](ADR-020-prompt-template-system.md)**  
  Scope: Template system, response shaping and configuration
- **[ADR-004 — Local-First LLM Strategy](ADR-004-local-first-llm-strategy.md)**  
  Scope: Local-first backends and performance principles

## Export & Output

- **[ADR-022 — Export & Structured Output System](ADR-022-export-output-formatting.md)**  
  Scope: Structured output/export patterns

## Analytics & Backup

- **[ADR-032 — Local Analytics & Metrics (DuckDB)](ADR-032-local-analytics-and-metrics.md)**  
  Scope: Local analytics DB and retention
- **[ADR-033 — Local Backup & Retention](ADR-033-local-backup-and-retention.md)**  
  Scope: Manual backup + rotation policies

## Analysis Modes

- **[ADR-023 — Document Analysis Mode Strategy](ADR-023-analysis-mode-strategy.md)**  
  Scope: Analysis modes and switching strategy

## Memory & Chat

- **[ADR-021 — Conversational Memory & Context Management](ADR-021-chat-memory-context-management.md)**  
  Scope: Chat history, memory stores, context management

## Archived ADRs

- **[ADR-007 — Hybrid Persistence Strategy (archived)](archived/ADR-007-hybrid-persistence-strategy.md)**  
  Scope: Early persistence approach (archived)
- **[ADR-025 — Caching Strategy (Superseded)](archived/ADR-025-caching-strategy.md)**  
  Scope: Legacy caching strategy (superseded)

---

Latest additions:

- **ADR-035 (Accepted, v1.1.0)** — Application-level semantic cache with GPTCache (SQLite + FAISS)
- **ADR-036 (Accepted, v1.0.0)** — Streamlit UI controls for reranker (normalize_scores, top_n)
