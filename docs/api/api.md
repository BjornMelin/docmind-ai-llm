# API Reference (Overview)

DocMind AI exposes a Python API for programmatic use. The core retrieval entrypoint is a Router/Query Engine composed via `router_factory`.

## Retrieval Examples

```python
from src.retrieval.router_factory import build_router_engine

# Assume you have built/persisted indices elsewhere
router = build_router_engine(vector_index, graph_index, settings)
response = router.query("What relates X and Y across the corpus?")
print(response)
```

## GraphRAG Exports

```python
from src.retrieval.graph_config import export_graph_jsonl, get_export_seed_ids

seeds = get_export_seed_ids(pg_index, vector_index, cap=32)
export_graph_jsonl(
    property_graph_index=pg_index,
    output_path="./exports/graph.jsonl",
    seed_node_ids=seeds,
)
```

See also:

- docs/api/examples/python-example.py for end‑to‑end examples
- README.md (Environment Variables)
- [Configuration Guide](../developers/configuration.md)
