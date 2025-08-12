# ADR-006: Analysis Pipeline

## Title

Multi-Stage Query and Analysis Pipeline

## Version/Date

3.0 / August 12, 2025

## Status

Accepted

## Context

Multi-stage for efficient querying (retrieve → rerank → synthesize), async/parallel for speed, caching for reuse. Single ReAct agent eliminates complex routing logic while maintaining intelligent query processing through built-in reasoning capabilities.

## Related Requirements

- Phase 3.3: Multi-stage/routing/caching.

- Integrate hybrid/retrievers with single intelligent agent.

- Simplified pipeline routing through agent reasoning vs complex coordination.

## Alternatives

- Custom loops: Error-prone.

- Sequential: Slow.

- Multi-agent routing: Over-engineered for pipeline complexity.

## Decision

Use QueryPipeline (chain=[retriever, ColbertRerank, synthesizer], async_mode=True, parallel=True) with diskcache for caching. Single ReActAgent handles query complexity routing through intelligent reasoning vs external coordination.

## Related Decisions

- ADR-013 (RRF in retriever stage).

- ADR-001 (Core pipeline).

## Design

- **Pipeline**: In utils.py: from llama_index.core.query_pipeline import QueryPipeline; qp = QueryPipeline(chain=[HybridFusionRetriever(...), ColbertRerank(...), synthesizer], async_mode=True, parallel=True).

- **Caching**: Wrap qp components with diskcache.memoize.

- **Integration**: qp.as_query_engine() converted to QueryEngineTool for ReActAgent. Agent handles complexity routing through reasoning: agent.chat("analyze complex query") vs external routing logic.

- **Implementation Notes**: Single agent determines retrieval strategy through reasoning vs hardcoded complexity analysis. Error handling: Try/except in chain.

- **Testing**: tests/test_performance_integration.py: def test_pipeline_multi_stage(): results = qp.run("query"); assert len(results) > 0; measure latency < 2s with async/parallel; def test_caching(): time1 = measure(qp.run("query")); time2 = measure(qp.run("query")); assert time2 < time1 / 2.

## Consequences

- Efficient/modular (async/parallel chaining, caching reuse).

- Scalable (route via agent reasoning vs external complexity analysis).

- Simplified integration (single agent vs multi-agent coordination).

- Complexity (manage chain errors).

- Deps: diskcache==5.6.3.

**Changelog:**  

- 3.0 (August 12, 2025): Updated pipeline integration to use single ReActAgent for intelligent query routing vs external multi-agent coordination. Simplified complexity routing through agent reasoning capabilities.

- 2.0 (July 25, 2025): Switched to QueryPipeline for multi-stage/async/parallel/caching; Integrated with hybrid/rerank/agents; Enhanced testing for dev.
