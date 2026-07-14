# Configure GraphRAG

DocMind uses LlamaIndex core's `PropertyGraphIndex` directly. There is no
GraphRAG adapter registry or optional graph dependency lane.

Normative requirements live in:

- `docs/specs/spec-006-graphrag.md`
- `docs/developers/adrs/ADR-019-optional-graphrag.md`
- `docs/developers/adrs/ADR-038-graphrag-persistence-and-router.md`

## Install required dependencies

```bash
uv sync --frozen
```

`llama-index-core` is a required dependency. A missing `PropertyGraphIndex` API
is a broken installation, not a feature-specific extra to install.

## Understand runtime gates

The router adds `knowledge_graph` only when all runtime conditions pass:

1. The ingestion run produced a property graph index.
2. The index exposes a property graph store.
3. LlamaIndex can construct its graph retriever and query engine.

`DOCMIND_GRAPHRAG_CFG__ENABLED` sets the default state of the per-ingestion
GraphRAG control. The submitted control decides whether that run builds an
index. A supplied healthy index is the router's sole GraphRAG availability
signal.

The router retains semantic search and any configured hybrid, keyword, or
multimodal tools when a graph index is unavailable. `get_graphrag_health()`
reports whether the required LlamaIndex Property Graph API is available for the
Streamlit status display. Availability does not imply enablement.

## Runtime ownership

- `src/retrieval/router_factory.py` imports the required LlamaIndex router and
  tool classes directly; tests patch those constructor boundaries when needed.
- `src/retrieval/llama_index_adapter.py` retains only the settings-facing
  GraphRAG health result.
- `src/retrieval/graph_config.py` calls `PropertyGraphIndex.as_retriever()` and
  builds `RetrieverQueryEngine` directly.
- Snapshot persistence writes and reloads the complete native property-graph
  `StorageContext`. Its graph-local vector store is required for semantic graph
  retrieval after restart; it is separate from the live Qdrant corpus vectors.
- Graph exports use the property graph store's `get()` and `get_rel_map()` APIs.
- Graph helpers emit OpenTelemetry `graph_export.<format>` spans. Snapshot and
  manual exports also emit metadata-only local `export_performed` JSONL events
  and record optional OpenTelemetry metrics when a meter is configured.

Do not add a second graph backend, registry, factory, or no-op telemetry hook
without a shipped product requirement.

## Verify GraphRAG

```bash
uv run pytest --no-cov \
  tests/unit/retrieval/test_graph_rag_factory.py \
  tests/unit/retrieval/test_router_factory_contract.py \
  tests/integration/test_graph.py \
  tests/integration/test_graphrag_exports.py
```

The `requires_llama` marker identifies tests that exercise the required
`llama_index.core` installation directly.
