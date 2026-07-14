---
ADR: 023
Title: Document Analysis Modes (Separate / Combined / Auto)
Status: Accepted
Version: 3.0
Date: 2026-07-13
Supersedes:
Superseded-by:
Related: 001, 003, 009, 013, 016, 024, 052
Tags: analysis, modes, routing, parallel
References:
- [Streamlit Tabs](https://docs.streamlit.io/develop/api-reference/layout/st.tabs)
- [Python concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
---

## Description

DocMind supports three analysis modes over the active LlamaIndex vector index:

- **Separate**: query each selected document concurrently
- **Combined**: query the selected documents as one corpus
- **Auto**: select separate or combined mode from the document count and worker limit

`src/analysis/service.py` owns mode selection and execution. Streamlit owns selection and rendering, not analysis behavior.

## Context

Document-specific questions need metadata filters, while corpus questions need one combined query. The service must keep Streamlit calls off worker threads, bound concurrency, preserve citations, and stop cooperatively when cancellation is requested.

## Decision drivers

- Preserve document-level filtering
- Bound per-document concurrency
- Keep one typed result contract
- Reuse LlamaIndex vector query engines
- Keep user-interface code free of analysis logic

## Decision framework

Weights: solution leverage 35%, application value 30%, maintenance and cognitive load 25%, architectural adaptability 10%.

| Option | Leverage | Value | Maintenance | Adaptability | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bounded service over the active vector index | 9.5 | 9.2 | 9.4 | 8.8 | **9.31** |
| Sequential per-document queries | 8.5 | 6.5 | 9.2 | 7.5 | 7.98 |
| A second agent graph for analysis modes | 5.5 | 8.0 | 4.0 | 7.0 | 6.03 |

The bounded service is selected.

## Decision

`run_analysis(...)` accepts the query, requested mode, active vector index, selected document references, settings, and optional cancellation and progress callbacks.

Auto mode applies the rules in `auto_select_mode(...)`:

- no selected documents or one selected document resolves to combined mode
- two or three selected documents resolve to separate mode when at least two workers are available
- larger selections resolve to combined mode

Combined mode creates one vector query engine with an OR filter over selected document IDs. It may run without filters only when the index does not accept the filter argument.

Separate mode creates one filtered query per document and uses a bounded `ThreadPoolExecutor`. It does not fall back to an unfiltered query because that could mix document results.

Both modes return `AnalysisResult` values from `src/analysis/models.py`. The service emits metadata-only selection, completion, cancellation, and failure events.

## Configuration

```env
DOCMIND_ANALYSIS__MODE=auto
DOCMIND_ANALYSIS__MAX_WORKERS=4
```

## Verification

- `tests/unit/analysis/test_analysis_service.py` covers mode selection, filters, concurrency, cancellation, and result contracts
- `tests/integration/ui/test_analysis_modes.py` covers Streamlit selection and rendering with the service stubbed

## Consequences

### Positive outcomes

- One service owns analysis-mode behavior
- Separate mode preserves per-document attribution
- Combined mode avoids an unnecessary second orchestration graph
- Tests can use a compatible vector-index stub without network access

### Trade-offs

- Separate mode consumes one worker per active document up to the configured limit
- Cooperative cancellation cannot terminate dependency work that ignores cancellation
- Combined mode may query the full corpus when an index does not accept metadata filters

## Changelog

- 3.0 (2026-07-13): Replaced QueryPipeline and illustrative contracts with the shipped vector-index service.
- 2.0 (2026-01-10): Added the original tier-two decision gate.
