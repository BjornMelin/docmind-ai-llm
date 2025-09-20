# Optional GraphRAG Capability Matrix

This plan documents how DocMind aligns behaviour when optional `llama_index` extras
are present versus the default lean installation. The goal is to keep the
retrieval router predictable while providing clear guidance to developers and CI
jobs.

## Dependency Levels

| Installation State | Available Features | Behaviour Notes |
|--------------------|--------------------|-----------------|
| **Baseline** (`pip install docmind_ai_llm`) | Vector + hybrid retrieval, telemetry | GraphRAG tools are disabled and a structured warning is emitted explaining how to enable support. Tests marked `requires_llama` are skipped. |
| **Graph Extras** (`pip install docmind_ai_llm[llama]`) | Adds `llama_index.core` + adapter surface | Router builds vector + hybrid tools. GraphRAG remains disabled until the program extras are installed. |
| **Full GraphRAG** (`pip install docmind_ai_llm[llama] llama-index-program-openai`) | Vector + hybrid + knowledge-graph (GraphRAG) | Router builds the knowledge graph tool path, reranking injectors run end-to-end, and optional `requires_llama` test suite is executed. |

## Runtime Signals

* `router_factory` exposes `adapter.supports_graphrag` so the router can emit a
  single warning message when graph support is unavailable.
* Telemetry now resets cached histograms between tests via
  `shutdown_metrics()` ensuring deterministic test runs.
* The retrieval agent fallback wrapper normalises LangChain tool call
  signatures, retrying automatically when tools expect payload dictionaries.

## Test Matrix

Use the updated tiered runner to coordinate coverage:

```bash
# Tiered fast path (unit → integration)
uv run python scripts/run_tests.py --fast

# Extras lane (skips automatically when llama_index extras are missing)
uv run python scripts/run_tests.py --extras
```

Targeted `pytest` invocations no longer apply coverage gates by default. CI
jobs that need coverage should call the scripted `--coverage` workflow which
still generates `htmlcov`, `coverage.xml`, and `coverage.json` artifacts.

## Developer Workflow

1. Install the appropriate extras for the feature you are working on.
2. Run `scripts/run_tests.py --fast` for the default validation and
   `scripts/run_tests.py --extras` if you are touching GraphRAG paths.
3. The router will raise a warning when GraphRAG is requested without installed
   dependencies—treat this as a signal to add the extras locally or adjust test
   expectations via the provided pytest markers.
