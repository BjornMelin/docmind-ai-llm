# ADR-012: Async Performance Optimization

## Title

Async and Parallel Processing Strategy

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Async for non-blocking UI (uploads/indexing/querying), parallel for speed (e.g., multi-stage pipeline).

## Related Requirements

- Async in indexing/loading/querying.
- Parallel in QueryPipeline/GPU streams.

## Alternatives

- Sync: UI blocking.
- Threading: GIL limits for Python.

## Decision

Use asyncio for ops (create_index_async, QueryPipeline async_mode=True, parallel=True). CUDA streams for GPU parallel in async functions.

## Related Decisions

- ADR-003 (GPU streams in async).
- ADR-006 (Async in pipeline).

## Design

- **Async Setup**: In app.py: import asyncio; async def upload_section(): await create_index_async(...); asyncio.run(upload_section()).
- **Integration**: QueryPipeline(async_mode=True, parallel=True) for querying. In create_index_async: with torch.cuda.Stream(): await ...
- **Implementation Notes**: Use asyncio.to_thread for sync parts (e.g., non-async libs). Error: await with try/except.
- **Testing**: tests/test_performance_integration.py: @pytest.mark.asyncio def test_async_parallel(): await qp.run("query"); assert latency < sync; def test_streams(): if gpu: assert stream used in index.

## Consequences

- Responsive UI (non-blocking).
- Scalable (parallel for multi-core/GPU).

- Async debugging (use tests for coroutines).
- Deps: asyncio (built-in).

**Changelog:**  

- 2.0 (July 25, 2025): Added QueryPipeline async/parallel; Integrated with GPU streams; Enhanced testing for dev.
