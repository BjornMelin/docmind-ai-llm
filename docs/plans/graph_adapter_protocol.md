# GraphRAG Adapter Protocol

This note captures the adapter contract introduced on 2025-09-19 alongside the
GraphRAG modernization effort.

## Adapter Interfaces

Adapters must implement the following runtime-checkable protocols:

- `GraphIndexBuilderProtocol` – optional factory for constructing
  `PropertyGraphIndex` instances from documents.
- `GraphRetrieverProtocol` – exposes a `retrieve()` method returning graph-aware
  nodes.
- `GraphQueryEngineProtocol` – provides synchronous and asynchronous
  query execution (`query()`/`aquery()`).
- `GraphExporterProtocol` – optional exporter surface for JSONL/Parquet
  persistence.
- `TelemetryHooksProtocol` – instrumentation hooks for router construction and
  export events.
- `AdapterFactoryProtocol` – central factory returning
  `GraphQueryArtifacts` bundles (retriever, query engine, exporter, telemetry)
  and exposing metadata (`name`, `version`, `supports_graphrag`,
  `dependency_hint`).

See `src/retrieval/adapters/protocols.py` for exact type signatures.

## LlamaIndex Reference Implementation

`LlamaIndexAdapterFactory` (`src/retrieval/llama_index_adapter.py`) lazily imports
LlamaIndex modules and validates a minimum version of `0.10.0`. It instantiates
`KnowledgeGraphRAGRetriever` and `RetrieverQueryEngine.from_args(...)` as
recommended in the official GraphRAG guides. Missing dependencies raise
`MissingLlamaIndexError` with actionable install hints
(`pip install docmind_ai_llm[graphrag]`).

Telemetry hooks default to a no-op implementation and will be extended in
Phase B for OpenTelemetry spans.

## Registry

`src/retrieval/adapter_registry.py` provides a lightweight in-memory registry.
It registers the LlamaIndex adapter eagerly when dependencies are present and
falls back to vector-only routing otherwise.

Key helpers:

- `register_adapter(factory)` – register additional adapters.
- `get_adapter(name=None)` – fetch adapters, defaulting to LlamaIndex.
- `ensure_default_adapter()` – attempt registration of the default adapter.
- `list_adapters()` – introspect registered names.

## Migration Notes

- `build_graph_query_engine`, `build_graph_retriever`, and
  `get_export_seed_ids` now accept an optional `adapter` argument; they resolve
  the default adapter via the registry when omitted.
- Legacy stub construction (`_build_stub_adapter`) has been removed. If
  `llama-index` is absent the router factory logs a warning and returns a
  vector-only router.
- Compatibility helpers (`get_llama_index_adapter`, `set_llama_index_adapter`)
  remain for tests but now always expose real LlamaIndex classes (no stubs).
