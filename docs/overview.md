# DocMind AI Overview

- **Released version:** 1.0.0
- **Documentation revision:** v2 modernization, unreleased

DocMind AI is an advanced, local-first agentic RAG (Retrieval-Augmented Generation) system.

## Documentation Navigation

- [Technical Requirements](specs/requirements.md)
- [System Architecture](developers/system-architecture.md)
- [Contributing Guide](../CONTRIBUTING.md)

## Project Summary

DocMind AI is an offline-first document analysis system with four LangGraph
worker roles (planner, retrieval, synthesis, and validation), hybrid retrieval,
and local processing for privacy. The retrieval worker delegates strategy
selection to LlamaIndex's native `RouterQueryEngine`.

Ingestion is handled by a library-first LlamaIndex `IngestionPipeline` with
DuckDB-backed caching and deterministic hashing, while snapshot persistence uses
a portalocker-based SnapshotManager with tri-file manifests and timestamped
GraphRAG exports. Optional OpenTelemetry uses configured OTLP exporters;
metadata-only JSONL telemetry remains local.

Retrieval/Router overview: queries pass through the native LlamaIndex
`RouterQueryEngine` composed by `router_factory`. It always includes
`semantic_search` and conditionally includes `hybrid_search`, `keyword_search`,
`multimodal_search`, and `knowledge_graph` when their configuration and indexes
are ready.

Snapshots & Staleness: Qdrant owns vector data. App snapshots identify immutable
physical text/image collections and package optional property-graph artifacts in
a checksum-verified manifest. Chat activates only the non-stale manifest named by
`CURRENT`; it never infers activation from directory ordering.

GraphRAG Exports: Graph exports preserve relation labels from `get_rel_map` (fallback `related`) and are available as JSONL (baseline) and Parquet (optional). Export seeding follows a retriever‑first policy (graph → vector → deterministic fallback).

## Quick Router Example

```python
from src.retrieval.router_factory import build_router_engine

# Assume vector_index is available; graph_index may be None when GraphRAG is disabled
router = build_router_engine(vector_index, graph_index, settings)
try:
    response = router.query("What connects topic X and entity Y?")
    print(response)
finally:
    router.close()
```

Notes:

- The router always registers `semantic_search`; configured optional tools are
  `hybrid_search`, `keyword_search`, `multimodal_search`, and
  `knowledge_graph`.
- LlamaIndex's native `RouterQueryEngine` owns tool selection.

## Documentation Structure

- **[Product Requirements Document (PRD)](PRD.md)**: Complete system requirements with validated performance specifications
- **[User Documentation](user/getting-started.md)**:
  - [Getting Started](user/getting-started.md): Installation and setup
  - [Configuration](user/configuration.md): Basic settings and examples
  - [Troubleshooting & FAQ](user/troubleshooting-faq.md): Common issues and answers
- **[Developer Documentation](developers/)**:
  - [Getting Started](developers/getting-started.md): Development setup and first run
  - [Architecture Overview](developers/architecture-overview.md): High-level system view
  - [System Architecture](developers/system-architecture.md): Technical diagrams and flows
  - [Cache Implementation](developers/cache.md): Cache wiring and operations
  - [Configuration Guide](developers/configuration.md): Developer configuration details
  - [CI/CD Pipeline](developers/ci-cd-pipeline.md): GitHub Actions pipeline (Python 3.12.13)
  - [Operations Guide](developers/operations-guide.md): Deployment and operations
  - [Developer Handbook](developers/developer-handbook.md): Patterns and workflows
  - [Testing Guide](testing/testing-guide.md): How to write and run tests
- **[API Documentation](api/api.md)**:
  - [API Reference](api/api.md): Repository-local Python retrieval and coordinator interfaces
- **Architectural Decision Records (ADRs)**:
  - [ADR Index](developers/adrs/): Complete collection of architectural decisions

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

The project is licensed under MIT—see [LICENSE](../LICENSE).

## Support

For issues or questions, open an issue on [GitHub](https://github.com/BjornMelin/docmind-ai-llm) or check [user/troubleshooting-faq.md](user/troubleshooting-faq.md).
