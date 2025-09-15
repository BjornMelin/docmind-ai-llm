# Integration Tests (Offline, Deterministic)

## Overview

- The integration tier validates component interactions without touching real networks or GPUs.
- All tests are strictly offline by default: a session‑autouse fixture forces `Settings.llm = MockLLM` for the entire integration session.
- Reranking and routing use mocks and fixtures to prevent heavyweight model calls.

## Key Isolation Fixture

- **File**: `tests/integration/conftest.py`
- **What it does**: Sets `Settings.llm = MockLLM(max_tokens=256)` for the duration of the integration session, then restores the original LLM.
- **Why**: Prevents accidental Ollama/OpenAI calls and removes environment drift from global settings.

## Guidelines for Adding New Integration Tests

1) **Always keep it offline**
   - Do not rely on remote model backends or network calls.
   - Use provided fixtures: `mock_llm_for_routing`, `mock_vector_index`, `mock_hybrid_retriever`, `mock_property_graph`, `mock_memory_monitor`, etc.

2) **Use public APIs**

   - Prefer high‑level entry points (e.g., build_router_engine via router_factory, RouterQueryEngine.query/aquery, build_text_reranker(...).postprocess_nodes).
     - Avoid private/underscored internals in production code.

3) **Routing with LLMSingleSelector**
   - The selector reads `Settings.llm` (MockLLM). For deterministic routing, set the response text on the test LLM via `mock_llm_for_routing.response_text = "semantic_search"` (or `"hybrid_search"`, etc.).
   - When needed, stub `router_engine.router_engine.query = MagicMock(return_value=...)` to control underlying outcomes.

4) **Reranking (text via SentenceTransformerRerank)**
   - Patch `src.retrieval.reranking.build_text_reranker` to return a stubbed `SentenceTransformerRerank` that emits deterministic scores.
   - Prefer asserting ordering and `top_n` length rather than exact floats.

5) **Memory/Performance checks**
   - Use `mock_memory_monitor` to simulate memory usage; do not import system monitors.
   - Keep timing assertions loose and property‑based (monotonicity, max bounds).

6) **Keep tests independent and deterministic**
   - Avoid hidden dependencies across tests; use fixtures for shared state.
   - No sleeps; if needed, use the provided boundary fixtures that patch sleep/perf counters.

## Examples

### Example: Deterministic router test

```python
def test_selects_hybrid(mock_vector_index, mock_hybrid_retriever, mock_llm_for_routing):
    mock_llm_for_routing.response_text = "hybrid_search"
    engine = build_router_engine(
        vector_index=mock_vector_index,
        hybrid_retriever=mock_hybrid_retriever,
        llm=mock_llm_for_routing,
    )
    # Stub underlying router call to control output
    engine.router_engine.query = MagicMock(return_value=MagicMock(response="ok"))
    assert engine.query("broad query").response == "ok"
```

### Example: Deterministic reranker test (text via SentenceTransformerRerank)

```python
from unittest.mock import patch
import numpy as np

@patch("src.retrieval.reranking.build_text_reranker")
def test_rerank_order(mock_build):
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

    class _StubRerank:
        def __init__(self, top_n: int):
            self.top_n = top_n

        def postprocess_nodes(self, nodes, query_str: str):
            scores = [0.9, 0.8, 0.7]
            for node, s in zip(nodes, scores, strict=False):
                node.score = s
            return nodes[: self.top_n]

    mock_build.return_value = _StubRerank(top_n=2)

    nodes = [
        NodeWithScore(node=TextNode(text="a", id_="1"), score=0.1),
        NodeWithScore(node=TextNode(text="b", id_="2"), score=0.1),
        NodeWithScore(node=TextNode(text="c", id_="3"), score=0.1),
    ]
    qb = QueryBundle(query_str="q")
    out = mock_build().postprocess_nodes(nodes, qb.query_str)
    assert len(out) == 2
    assert out[0].score >= out[1].score
```

## What Not to Do

- Do not patch global `Settings.llm` outside fixtures; always pass `llm=mock_llm_for_routing` to constructors/factories when needed.
- Do not assert against unstable logs or exact floating point values that may change with library updates.

## CI Notes

- Integration runs in CI as a separate job, without coverage gating. Unit coverage thresholds remain stable.
