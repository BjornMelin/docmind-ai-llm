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
- ADR-006 — Modern Reranking Architecture (Superseded by ADR‑037)  
  Scope: Historical text-only CrossEncoder reranking; see ADR‑037 for current design
- **[ADR-037 — Multimodal Reranking with ColPali (visual) and BGE v2‑m3 (text)](ADR-037-multimodal-reranking-architecture.md)**  
  Scope: Modality-aware reranking; ColPali for visuals, BGE v2‑m3 for text
- **[ADR-019 — Optional GraphRAG Module](ADR-019-optional-graphrag.md)**  
  Scope: Property graph/graph retrieval integrations
- **[ADR-038 — GraphRAG Persistence and Router Integration](ADR-038-graphrag-persistence-and-router.md)**  
  Scope: Router composition (vector+graph), snapshot persistence, exports, and UI staleness
- **[ADR-034 — Idempotent Indexing & Embedding Reuse](ADR-034-idempotent-indexing-and-embedding-reuse.md)**  
  Scope: Idempotent ingestion and reusable embeddings

## UI & State

- **[ADR-013 — User Interface Architecture](ADR-013-user-interface-architecture.md)**  
  Scope: Streamlit multipage architecture and native component usage
- **[ADR-016 — Streamlit Native State Management](ADR-016-ui-state-management.md)**  
  Scope: Session state and cache primitives without custom layers
- **[ADR-036 — Reranker UI Controls (Superseded)](archived/ADR-036-SUPERSEDED-reranker-ui-controls-normalize-topn.md)**  
  Scope: Historical UI controls; superseded by always-on rerank + env-only overrides

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
- **[ADR-029 — Boundary‑First Testing Strategy](ADR-029-testing-strategy.md)**  
  Scope: Unit/integration boundary tests and coverage focus (see ADR‑014 for gates)

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

- **ADR-039 (Accepted)** — Offline evaluation CLIs
- **ADR-040 (Accepted)** — Model predownload CLI
- **ADR-041–ADR-054 (Proposed)** — Release Readiness v1 work packages

## Release Readiness v1 (2026-01-09)

- **[ADR-041 — Settings UI Hardening: Pre-validation, Safe Badges, and .env Persistence](ADR-041-settings-ui-hardening-and-safe-badges.md)**
- **[ADR-042 — Containerization Hardening](ADR-042-containerization-hardening.md)**
- **[ADR-043 — Chat Persistence (SimpleChatStore JSON)](ADR-043-chat-persistence-simplechatstore.md)**
- **[ADR-044 — Keyword Tool (Sparse-only Qdrant)](ADR-044-keyword-tool-sparse-only-qdrant.md)**
- **[ADR-045 — Programmatic Ingestion API + Legacy Facade](ADR-045-ingestion-api-and-legacy-facade.md)**
- **[ADR-046 — Remove Legacy `src/main.py` Entrypoint](ADR-046-remove-legacy-main-entrypoint.md)**
- **[ADR-047 — Safe Logging Policy (No PII Redactor Stub)](ADR-047-safe-logging-policy-no-pii-redactor.md)**
- **[ADR-048 — Docs Consistency Pass](ADR-048-docs-consistency-pass.md)**
- **[ADR-049 — Multimodal Helper Cleanup](ADR-049-multimodal-helper-cleanup.md)**
- **[ADR-050 — Config Discipline (Remove `os.getenv` Sprawl)](ADR-050-config-discipline-env-bridges.md)**
- **[ADR-051 — Documents Snapshot Service Boundary](ADR-051-documents-snapshot-service-boundary.md)**
- **[ADR-052 — Background Ingestion Jobs (Threads + Fragments)](ADR-052-background-ingestion-jobs.md)**
- **[ADR-053 — Analytics Page Hardening](ADR-053-analytics-page-hardening.md)**
- **[ADR-054 — Config Surface Pruning (Unused Knobs)](ADR-054-config-surface-pruning-unused-knobs.md)**
