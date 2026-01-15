# ADR Index (Curated by Topic)

This page provides a quick, opinionated index of all Architectural Decision Records (ADRs), grouped by topic. For the full file list, view this directory.

---

## Architecture & Agents

- **[ADR-001 — Modern Agentic RAG Architecture](ADR-001-modern-agentic-rag-architecture.md)**<br>
  Scope: Multi-agent, modern RAG architecture and capabilities
- **[ADR-011 — Agent Orchestration Framework](ADR-011-agent-orchestration-framework.md)**<br>
  Scope: Supervisor/orchestration patterns and agent tool integration
- **[ADR-056 — Agent Deadline Propagation + Router Injection](ADR-056-agent-deadline-propagation-and-router-injection.md)**<br>
  Scope: Cooperative time budgets, injected router engines, and retrieval contract correctness
- **[ADR-027 — Implementation Experience (meta)](ADR-027-implementation-experience.md)**<br>
  Scope: Cross-ADR integration learnings and validation

## Retrieval, Embeddings & Graph

- **[ADR-002 — Unified Embedding Strategy with BGE-M3](ADR-002-unified-embedding-strategy.md)**<br>
  Scope: Unified dense/sparse embeddings centered on BGE-M3
- **[ADR-003 — Adaptive Retrieval Pipeline](ADR-003-adaptive-retrieval-pipeline.md)**<br>
  Scope: Router/strategy selection and hierarchical retrieval
- **[ADR-037 — Multimodal Reranking with ColPali (visual) and BGE v2‑m3 (text)](ADR-037-multimodal-reranking-architecture.md)**<br>
  Scope: Modality-aware reranking; ColPali for visuals, BGE v2‑m3 for text
- **[ADR-019 — Optional GraphRAG Module](ADR-019-optional-graphrag.md)**<br>
  Scope: Property graph/graph retrieval integrations
- **[ADR-038 — GraphRAG Persistence and Router Integration](ADR-038-graphrag-persistence-and-router.md)**<br>
  Scope: Router composition (vector+graph), snapshot persistence, exports, and UI staleness
- **[ADR-034 — Idempotent Indexing & Embedding Reuse](ADR-034-idempotent-indexing-and-embedding-reuse.md)**<br>
  Scope: Idempotent ingestion and reusable embeddings

## UI & State

- **[ADR-013 — User Interface Architecture](ADR-013-user-interface-architecture.md)**<br>
  Scope: Streamlit multipage architecture and native component usage
- **[ADR-016 — Streamlit Native State Management](ADR-016-ui-state-management.md)**<br>
  Scope: Session state and cache primitives without custom layers
- **[ADR-023 — Document Analysis Modes (Separate / Combined / Auto)](ADR-023-analysis-mode-strategy.md)**<br>
  Scope: User-visible analysis modes and domain-layer routing/aggregation

## Configuration

- **[ADR-024 — Unified Settings Architecture](ADR-024-configuration-architecture.md)**<br>
  Scope: Pydantic BaseSettings + LlamaIndex Settings integration

## Caching & Persistence

- **[ADR-030 — Cache Unification via LlamaIndex IngestionCache (DuckDBKVStore)](ADR-030-cache-unification-ingestioncache-duckdbkvstore.md)**<br>
  Scope: Document-processing cache, single-file DuckDB KV store
- **[ADR-031 — Local-First Persistence Architecture (Vectors, Cache, Operational Data)](ADR-031-local-first-persistence-architecture.md)**<br>
  Scope: Separation of vectors (Qdrant), processing cache (DuckDB), ops data (SQLite)
- **[ADR-055 — Operational Metadata Store (SQLite WAL)](ADR-055-operational-metadata-sqlite-wal.md)**<br>
  Scope: Transactional, local-only ops state for background jobs and snapshot events
- **[ADR-035 — Application-Level Semantic Cache](ADR-035-semantic-cache-qdrant.md)**<br>
  Scope: Optional semantic response caching (backend pluggable; offline-first)
- **[ADR-058 — Final Multimodal Pipeline + Cognitive Persistence](ADR-058-final-multimodal-pipeline-and-persistence.md)**<br>
  Scope: End-to-end multimodal (PDF images → retrieval → UI) + durability invariants (no blobs/paths in durable stores)

## Document Processing

- **[ADR-009 — Document Processing Pipeline (Superseded by ADR-058)](superseded/ADR-009-document-processing-pipeline.md)**<br>
  Scope: Historical (superseded by ADR-058)

## Performance

- **[ADR-010 — Performance Optimization Strategy](ADR-010-performance-optimization-strategy.md)**<br>
  Scope: FP8/vLLM optimizations and latency targets

## Testing & Quality

- **[ADR-014 — Testing and Quality Validation Framework](ADR-014-testing-quality-validation.md)**<br>
  Scope: DeepEval + pytest strategy and CI integration
- **[ADR-029 — Boundary‑First Testing Strategy](ADR-029-testing-strategy.md)**<br>
  Scope: Unit/integration boundary tests and coverage focus (see ADR‑014 for gates)

## Prompt & LLM Optimization

- **[ADR-018 — Automatic Prompt Optimization with DSPy](ADR-018-prompt-optimization-dspy.md)**<br>
  Scope: DSPy-based prompt optimization patterns
- **[ADR-020 — Dynamic Prompt Template & Response Configuration](ADR-020-prompt-template-system.md)**<br>
  Scope: Template system, response shaping and configuration
- **[ADR-004 — Local-First LLM Strategy](ADR-004-local-first-llm-strategy.md)**<br>
  Scope: Local-first backends and performance principles
- **[ADR-059 — Ollama Native Capabilities and Cloud Gating](ADR-059-ollama-native-capabilities-and-cloud-gating.md)**<br>
  Scope: Ollama-native `/api/*` capabilities, cloud web tools gating, and explicit streaming semantics

## Export & Output

- **[ADR-022 — Export & Structured Output System](ADR-022-export-output-formatting.md)**<br>
  Scope: Structured output/export patterns

## Analytics & Backup

- **[ADR-032 — Local Analytics & Metrics (DuckDB)](ADR-032-local-analytics-and-metrics.md)**<br>
  Scope: Local analytics DB and retention
- **[ADR-033 — Local Backup & Retention](ADR-033-local-backup-and-retention.md)**<br>
  Scope: Manual backups with rotation (local-only)

## Superseded ADRs

These ADRs are kept for historical context only and MUST NOT be implemented. See files under `docs/developers/adrs/superseded/`.

- **[ADR-006 — Modern Reranking Architecture (Superseded by ADR-037)](superseded/ADR-006-reranking-architecture.md)**
- **[ADR-007 — Hybrid Persistence Strategy (Superseded by ADR-031)](superseded/ADR-007-hybrid-persistence-strategy.md)**
- **[ADR-012 — Evaluation with DeepEval (Superseded by ADR-039)](superseded/ADR-012-evaluation-strategy.md)**
- **[ADR-015 — Docker-First Local Deployment (Superseded by ADR-042)](superseded/ADR-015-deployment-strategy.md)**
- **[ADR-021 — Conversational Memory & Context Management (Superseded by ADR-058)](superseded/ADR-021-chat-memory-context-management.md)**
- **[ADR-043 — Chat Persistence via SimpleChatStore (Superseded by ADR-058)](superseded/ADR-043-chat-persistence-simplechatstore.md)**
- **[ADR-025 — Caching Strategy (Superseded by ADR-030)](superseded/ADR-025-caching-strategy.md)**
- **[ADR-036 — Reranker UI Controls (Superseded)](superseded/ADR-036-reranker-ui-controls-normalize-topn.md)**
- **[ADR-054 — Config Surface Pruning (Unused Knobs) (Superseded)](superseded/ADR-054-config-surface-pruning-unused-knobs.md)**
- **[ADR-057 — Chat Persistence + Hybrid Agentic Memory (LangGraph SQLite) (Superseded by ADR-058)](superseded/ADR-057-chat-persistence-langgraph-sqlite-hybrid-memory.md)**
- **[ADR-013.1 — User Interface Architecture (Full) (Superseded by ADR-013)](superseded/ADR-013.1-user-interface-architecture-full.md)**

---

Latest additions:

- **ADR-039 (Accepted)** — Offline evaluation CLIs
- **ADR-040 (Accepted)** — Model predownload CLI
- **ADR-041–ADR-053 (Proposed)** — Release Readiness work packages
- **ADR-055–ADR-056 (Proposed)** — Ops metadata + agent deadline propagation work packages
- **ADR-059 (Accepted)** — Ollama native capabilities + cloud gating

## Release Readiness (2026-01-09)

- **[ADR-041 — Settings UI Hardening: Pre-validation, Safe Badges, and .env Persistence](ADR-041-settings-ui-hardening-and-safe-badges.md)**
- **[ADR-042 — Containerization Hardening](ADR-042-containerization-hardening.md)**
- **[ADR-044 — Keyword Tool (Sparse-only Qdrant)](ADR-044-keyword-tool-sparse-only-qdrant.md)**
- **[ADR-045 — Programmatic Ingestion API + Legacy Facade](ADR-045-ingestion-api-and-legacy-facade.md)**
- **[ADR-046 — Remove Legacy `src/main.py` Entrypoint](ADR-046-remove-legacy-main-entrypoint.md)**
- **[ADR-047 — Safe Logging Policy (Metadata-only + Keyed Fingerprints + Backstop)](ADR-047-safe-logging-policy-no-pii-redactor.md)**
- **[ADR-048 — Docs Consistency Pass](ADR-048-docs-consistency-pass.md)**
- **[ADR-049 — Multimodal Helper Cleanup](ADR-049-multimodal-helper-cleanup.md)**
- **[ADR-050 — Config Discipline (Remove `os.getenv` Sprawl)](ADR-050-config-discipline-env-bridges.md)**
- **[ADR-051 — Documents Snapshot Service Boundary](ADR-051-documents-snapshot-service-boundary.md)**
- **[ADR-052 — Background Ingestion Jobs (Threads + Fragments)](ADR-052-background-ingestion-jobs.md)**
- **[ADR-053 — Analytics Page Hardening](ADR-053-analytics-page-hardening.md)**
- **[ADR-055 — Operational Metadata Store (SQLite WAL)](ADR-055-operational-metadata-sqlite-wal.md)**
- **[ADR-056 — Agent Deadline Propagation + Router Injection](ADR-056-agent-deadline-propagation-and-router-injection.md)**
