# Optional GraphRAG Capability Matrix

This plan captures how DocMind behaves now that the Kùzu-backed GraphRAG stack
ships in the default installation while still allowing developers to opt into
heavier optional extras (e.g., ColPali multimodal reranking). The goal is to
keep the router predictable and provide clear guidance to contributors and CI
jobs.

## Dependency Levels

| Installation State | Available Features | Behaviour Notes |
|--------------------|--------------------|-----------------|
| **Baseline** (`pip install docmind_ai_llm`) | Vector + hybrid + property-graph retrieval (Kùzu + DuckDB) | GraphRAG adapter registers automatically; router builds the knowledge-graph tool by default. Tests tagged `requires_llama` run without additional setup. |
| **Multimodal Extras** (`pip install docmind_ai_llm[multimodal]`) | Adds ColPali reranker + vision dependencies | Optional lane used for image-heavy workloads. The router downgrades gracefully when these extras are absent. |

## Runtime Signals

* `router_factory` exposes `adapter.supports_graphrag` so the router emits a
  single warning if the Kùzu adapter fails health checks.
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
2. Run `scripts/run_tests.py --fast` for the default validation and
   `scripts/run_tests.py --extras` if you are touching optional multimodal
   paths.
3. The router raises a warning only if the embedded Kùzu adapter is unhealthy;
   treat this as a signal to inspect the local environment rather than install
   extra packages.
