# Optional GraphRAG Capability Matrix

This plan captures how DocMind behaves when GraphRAG is installed via the
optional `graph` extra, while still allowing developers to opt into heavier
optional extras (e.g., ColPali multimodal reranking). The goal is to keep the
router predictable and provide clear guidance to contributors and CI jobs.

## Dependency Levels

| Installation State | Available Features | Behaviour Notes |
|--------------------|--------------------|-----------------|
| **Baseline** (`pip install docmind_ai_llm`) | Vector + hybrid retrieval | GraphRAG is unavailable; router omits the knowledge-graph tool and emits a single guidance warning when GraphRAG is requested without dependencies. |
| **GraphRAG Extras** (`pip install docmind_ai_llm[graph]`) | Adds property-graph retrieval (Kùzu graph store) | GraphRAG adapter registers; router can build the knowledge-graph tool when a property graph index is present. |
| **Multimodal Extras** (`pip install docmind_ai_llm[multimodal]`) | Adds ColPali reranker + vision dependencies | Optional lane used for image-heavy workloads. The router downgrades gracefully when these extras are absent. |

## Runtime Signals

* `router_factory` exposes `adapter.supports_graphrag` so the router emits a
  single warning if GraphRAG is requested but the graph extra is unavailable
  (or if the graph adapter fails health checks).
* Telemetry resets cached histograms between tests via `shutdown_metrics()`
  to keep deterministic runs when optional extras are toggled on/off.
* The retrieval agent fallback wrapper normalises LangChain tool call
  signatures, retrying automatically when adapters expect payload dictionaries.

## Test Matrix

Use the updated tiered runner to coordinate coverage:

```bash
# Tiered fast path (unit → integration)
uv run python scripts/run_tests.py --fast

# Multimodal lane (skips automatically when optional extras are missing)
uv run python scripts/run_tests.py --extras
```

Targeted `pytest` invocations no longer apply coverage gates by default. CI
jobs that need coverage should call the scripted `--coverage` workflow which
still generates `htmlcov`, `coverage.xml`, and `coverage.json` artifacts.

## Developer Workflow

1. Install optional extras only when you need multimodal reranking or other
   advanced features.
   - For GraphRAG/property-graph retrieval: `uv sync --extra graph` (or
     `pip install docmind_ai_llm[graph]`)
   - For multimodal reranking: `uv sync --extra multimodal` (or
     `pip install docmind_ai_llm[multimodal]`)
2. Run `scripts/run_tests.py --fast` for the default validation and
   `scripts/run_tests.py --extras` if you are touching optional graph or
   multimodal paths.
3. Treat GraphRAG warnings as a signal to inspect the local environment (or
   install the missing extras), rather than papering over failures in code.
