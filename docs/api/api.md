# Use DocMind's internal Python APIs

DocMind exposes repository-local Python interfaces for its application code. It does not publish a stable library package or HTTP API. The core retrieval entrypoint is a router query engine composed through `router_factory`.

## Retrieval Examples

```python
from src.retrieval.router_factory import build_router_engine

# Assume you have built/persisted indices elsewhere
router = build_router_engine(vector_index, graph_index, settings)
try:
    response = router.query("What relates X and Y across the corpus?")
    print(response)
finally:
    router.close()
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

- [Multi-agent coordinator](agent-api.md)
- [Internal Python examples](examples/python-example.py)
- [Environment variables](../../README.md#environment-variables)
- [Configuration Guide](../developers/configuration.md)
