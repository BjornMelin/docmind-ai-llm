# Developer Guide — GraphRAG Adapters & Optional Dependency Lanes

This guide documents the **GraphRAG adapter contract** (runtime-checkable
protocols + registry) and how **optional dependencies** affect runtime behavior
and test execution.

This is an implementation/operations guide. For the normative requirements and
architecture decisions, see:

- `docs/specs/spec-006-graphrag.md` (SPEC‑006)
- `docs/developers/adrs/ADR-019-optional-graphrag.md` (ADR‑019)
- `docs/developers/adrs/ADR-038-graphrag-persistence-and-router.md` (ADR‑038)

## Capability matrix (install states)

DocMind is designed to **degrade gracefully**: GraphRAG wiring may be enabled,
but will fall back to vector-only behavior when GraphRAG-specific dependencies
or adapters are unavailable.

| Installation state | Available features | Behavior notes |
|---|---|---|
| **Baseline** (`uv sync` / `pip install docmind_ai_llm`) | Vector + hybrid retrieval | GraphRAG paths may be requested/configured, but the router will remain vector-only when the GraphRAG adapter cannot be registered (missing graph-store deps, health check failures). A single guidance warning is emitted when GraphRAG is requested but unavailable. |
| **GraphRAG extra** (`uv sync --extra graph` / `pip install docmind_ai_llm[graph]`) | Property-graph retrieval (Kùzu-backed graph store) | The default GraphRAG adapter can register, enabling the `knowledge_graph` tool when a property graph index is present and healthy. |
| **Multimodal extra** (`uv sync --extra multimodal` / `pip install docmind_ai_llm[multimodal]`) | ColPali reranker + vision deps | Optional lane for image-heavy workloads; behavior downgrades when absent. |

## Runtime behavior and signals

GraphRAG enablement is a combination of **config**, **adapter availability**,
and **index presence**:

- Config gate: `DOCMIND_ENABLE_GRAPHRAG=true|false` (top-level feature flag).
- Adapter gate: `src/retrieval/adapter_registry.py` tries to register a default
  adapter and exposes health via `get_default_adapter_health()`.
- Index gate: router construction only adds the `knowledge_graph` tool when a
  property graph index is present and the adapter supports GraphRAG.

The Streamlit UI surfaces adapter health via a badge (see
`src/ui/components/provider_badge.py`) and the retrieval/router stack emits
guidance messages when GraphRAG is requested but unavailable.

## Adapter protocol (what you must implement)

The GraphRAG adapter contract is expressed as runtime-checkable protocols in:

- `src/retrieval/adapters/protocols.py`

Adapters are expected to implement (directly or via thin wrappers):

- `GraphIndexBuilderProtocol`: optional factory for constructing a property
  graph index from documents (used by ingestion paths).
- `GraphRetrieverProtocol`: `retrieve(query, ...) -> Sequence[...]`.
- `GraphQueryEngineProtocol`: `query()` and `aquery()` for sync/async queries.
- `GraphExporterProtocol`: optional export surface (`export_jsonl`,
  `export_parquet`) used by snapshot/export flows.
- `TelemetryHooksProtocol`: hooks for router construction and export events.
- `AdapterFactoryProtocol`: the central factory that bundles runtime artifacts
  (`GraphQueryArtifacts`) and exposes metadata:
  - `name`, `version`
  - `supports_graphrag` (bool)
  - `dependency_hint` (actionable install/config guidance)

The router and export flows consume a `GraphQueryArtifacts` bundle:

- `retriever`, `query_engine`, `exporter` (optional), `telemetry`

## Registry (how adapters are discovered)

The adapter registry lives in:

- `src/retrieval/adapter_registry.py`

Key behaviors:

- `ensure_default_adapter()` attempts to register the default adapter (currently
  the LlamaIndex implementation) and silently skips registration when optional
  dependencies are missing.
- `resolve_adapter(adapter=None)` returns the provided adapter or resolves the
  default registry adapter.
- `get_default_adapter_health()` returns `(supported, adapter_name, guidance)`
  for UI and runtime messaging.

The registry is intentionally lightweight (in-memory) and is designed so the
application can fall back to vector-only routing when GraphRAG cannot be
enabled.

## Reference implementation (LlamaIndex)

The default adapter implementation is:

- `src/retrieval/llama_index_adapter.py`

Highlights:

- Lazy imports: avoids importing LlamaIndex modules at import time.
- Version guard: validates the installed LlamaIndex version against a minimum.
- Uses documented LlamaIndex APIs for GraphRAG wiring:
  - `PropertyGraphIndex.as_retriever(...)`
  - `RetrieverQueryEngine.from_args(...)`
- Missing dependencies are surfaced as `MissingLlamaIndexError` with actionable
  installation guidance.

## Adding a new adapter (checklist)

1. Implement `AdapterFactoryProtocol` (and optionally
   `GraphIndexBuilderProtocol`) in a new module under `src/retrieval/`.
2. Provide a clear `dependency_hint` that tells contributors how to install the
   required extras or system dependencies.
3. Register the adapter with `register_adapter(factory)` during app startup
   (or expose a helper that callers can invoke prior to resolution).
4. Add tests:
   - If tests require optional deps, mark them with `@pytest.mark.requires_llama`
     so they can run in the optional dependency lane.

## Testing optional dependency lanes

The tiered test runner provides an optional dependency lane:

```bash
# Tiered fast path (unit → integration)
uv run python scripts/run_tests.py --fast

# Optional dependency lane (runs pytest -m requires_llama; skips when deps missing)
uv run python scripts/run_tests.py --extras
```

`--extras` currently gates execution on the presence of
`llama_index.program.openai` and will automatically skip when it is not
installed.

