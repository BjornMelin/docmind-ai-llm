# Test Plan

## Unit Tests

- `agents/test_settings_override_router.py`: settings override includes router_engine; handles absence gracefully.
- `retrieval/test_graph_helpers.py`: traverse_relations uses get_rel_map; export JSONL correctness; Parquet guarded.
- `persistence/test_snapshot_manager.py`: atomic rename behavior; manifest fields; lock guards; hash functions stable.

## Integration Tests

- `integration/test_graphrag_exports.py`: run export on small stub store; validate JSONL lines; Parquet conditional when pyarrow present.
- `integration/test_ingest_router_flow.py`: end-to-end ingest path sets up router tools and routes a simple query.

## E2E Smoke

- `e2e/test_chat_graphrag_smoke.py`: seed one or two tiny docs; ensure router responds and includes sources.

## Guidelines

- Avoid external network calls; use stubs/mocks
- Deterministic seeds; small fixtures
- Keep path_depth=1 to bound latency
