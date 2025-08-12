# ADR-006: Analysis Pipeline

## Title

Multi-Stage Query and Analysis Pipeline

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Multi-stage for efficient querying (retrieve → rerank → synthesize), async/parallel for speed, caching for reuse.

## Related Requirements

- Phase 3.3: Multi-stage/routing/caching.
- Integrate hybrid/retrievers/agents.

## Alternatives

- Custom loops: Error-prone.
- Sequential: Slow.

## Decision

Use QueryPipeline (chain=[retriever, ColbertRerank, synthesizer], async_mode=True, parallel=True) with diskcache for caching. Route complexity via LangGraph.

## Related Decisions

- ADR-013 (RRF in retriever stage).
- ADR-001 (Core pipeline).

## Design

- **Pipeline**: In utils.py: from llama_index.core.query_pipeline import QueryPipeline; qp = QueryPipeline(chain=[HybridFusionRetriever(...), ColbertRerank(...), synthesizer], async_mode=True, parallel=True).
- **Caching**: Wrap qp components with diskcache.memoize.
- **Integration**: qp.as_query_engine() in tools. For agents, agent.chat(qp.run("query")).
- **Implementation Notes**: Add routing (e.g., if complexity=="complex": use KG retriever). Error handling: Try/except in chain.
- **Testing**: tests/test_performance_integration.py: def test_pipeline_multi_stage(): results = qp.run("query"); assert len(results) > 0; measure latency < 2s with async/parallel; def test_caching(): time1 = measure(qp.run("query")); time2 = measure(qp.run("query")); assert time2 < time1 / 2.

## Consequences

- Efficient/modular (async/parallel chaining, caching reuse).
- Scalable (route via complexity).

- Complexity (manage chain errors).
- Deps: diskcache==5.6.3.

**Changelog:**  

- 2.0 (July 25, 2025): Switched to QueryPipeline for multi-stage/async/parallel/caching; Integrated with hybrid/rerank/agents; Enhanced testing for dev.
