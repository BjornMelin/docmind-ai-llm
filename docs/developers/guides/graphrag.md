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

The router adds `knowledge_graph` only when all gates pass:

1. `DOCMIND_ENABLE_GRAPHRAG` is true.
2. `DOCMIND_GRAPHRAG_CFG__ENABLED` is true.
3. A property graph index is present.
4. LlamaIndex can construct its graph retriever and query engine.

The application falls back to vector and hybrid retrieval when a graph index is
unavailable. `get_graphrag_health()` reports whether the required LlamaIndex
Property Graph API is importable for the Streamlit status badge.

## Runtime ownership

- `src/retrieval/llama_index_adapter.py` lazily imports the LlamaIndex router
  classes and owns the GraphRAG health check.
- `src/retrieval/graph_config.py` calls `PropertyGraphIndex.as_retriever()` and
  builds `RetrieverQueryEngine` directly.
- Graph exports use the property graph store's `get()` and `get_rel_map()` APIs.
- OpenTelemetry export spans remain the single graph-export telemetry path.

Do not add a second graph backend, registry, factory, or no-op telemetry hook
without a shipped product requirement.

## Verify GraphRAG

```bash
uv run pytest --no-cov \
  tests/unit/retrieval/test_llama_index_adapter.py \
  tests/unit/retrieval/test_graph_rag_factory.py \
  tests/unit/retrieval/test_router_factory_contract.py \
  tests/integration/test_graph.py \
  tests/integration/test_graphrag_exports.py
```

The `requires_llama` marker disables the default test stub and exercises the
required `llama_index.core` installation.
