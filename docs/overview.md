# DocMind AI Overview

**Version:** 2.0.0  
**Status:** Alpha (v2.0 Revision)

DocMind AI is an advanced, local-first agentic RAG (Retrieval-Augmented Generation) system.

## Documentation Navigation

- [Technical Requirements](specs/requirements.md)
- [System Architecture](developers/system-architecture.md)
- [Contributing Guide](../CONTRIBUTING.md)

## Project Summary

DocMind AI is an offline-first document analysis system featuring a 5-agent LangGraph supervisor coordination system, hybrid retrieval, and local processing for privacy.

Ingestion is handled by a library-first LlamaIndex `IngestionPipeline` with DuckDB-backed caching and deterministic hashing, while snapshot persistence relies on a portalocker-based SnapshotManager that produces tri-file manifests and timestamps GraphRAG exports. Observability is unified through OpenTelemetry with OTLP exporters (or console fallback) so ingestion, snapshot promotion, GraphRAG exports, and Streamlit UI flows emit consistent spans, metrics, and telemetry events.

Retrieval/Router overview: Queries are routed through a RouterQueryEngine composed via `router_factory` with tools `semantic_search`, `hybrid_search` (Qdrant server‑side fusion), and (optionally) `knowledge_graph` when a PropertyGraphIndex is present and healthy. The selector prefers `PydanticSingleSelector` and falls back to `LLMSingleSelector`.

Snapshots & Staleness: Index snapshots persist vector and graph stores with an enriched manifest (schema/persist versions, versions map, hashes). Chat auto‑loads the latest non‑stale snapshot and surfaces a staleness badge if current corpus/config differ.

GraphRAG Exports: Graph exports preserve relation labels from `get_rel_map` (fallback `related`) and are available as JSONL (baseline) and Parquet (optional). Export seeding follows a retriever‑first policy (graph → vector → deterministic fallback).

## Quick Router Example

```python
from src.retrieval.router_factory import build_router_engine

# Assume vector_index is available; graph_index may be None when GraphRAG is disabled
router = build_router_engine(vector_index, graph_index, settings)
response = router.query("What connects topic X and entity Y?")
print(response)
```

Notes:

- The router registers tools `semantic_search`, `hybrid_search` (Qdrant server‑side fusion), and (optionally) `knowledge_graph` when a PropertyGraphIndex is present and healthy.
- Selector preference: `PydanticSingleSelector` when available; otherwise `LLMSingleSelector`.

## Documentation Structure

- **[Product Requirements Document (PRD)](PRD.md)**: Complete system requirements with validated performance specifications
- **[User Documentation](user/)**:
  - [Getting Started](user/getting-started.md): Installation and setup
  - [Configuration](user/configuration.md): Basic settings and examples
  - [Troubleshooting & FAQ](user/troubleshooting-faq.md): Common issues and answers
- **[Developer Documentation](developers/)**:
  - [Getting Started](developers/getting-started.md): Development setup and first run
  - [Architecture Overview](developers/architecture-overview.md): High-level system view
  - [System Architecture](developers/system-architecture.md): Technical diagrams and flows
  - [Cache Implementation](developers/cache.md): Cache wiring and operations
  - [Configuration Guide](developers/configuration.md): Developer configuration details
  - [CI/CD Pipeline](developers/ci-cd-pipeline.md): GitHub Actions pipeline (3.11 + 3.10)
  - [Operations Guide](developers/operations-guide.md): Deployment and operations
  - [Developer Handbook](developers/developer-handbook.md): Patterns and workflows
  - [Testing Guide](testing/testing-guide.md): How to write and run tests
- **[API Documentation](api/)**:
  - [API Reference](api/api.md): REST and Python API with examples
- **Architectural Decision Records (ADRs)**:
  - [ADR Index](developers/adrs/): Complete collection of architectural decisions

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

The project is licensed under MIT—see [LICENSE](../LICENSE).

## Support

For issues or questions, open an issue on [GitHub](https://github.com/BjornMelin/docmind-ai) or check [user/troubleshooting-faq.md](user/troubleshooting-faq.md).
