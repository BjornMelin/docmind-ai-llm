---
spec: SPEC-040
title: Agent Deadlines and Canonical Router Retrieval
date: 2026-07-13
version: 2.0.0
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-014: LangGraph-supervised multi-agent flow.
  - FR-029: Agent deadline propagation and canonical router retrieval.
  - NFR-PERF-001: Chat latency budgets remain within targets.
  - NFR-REL-002: Timed-out runs cannot publish late graph state.
related_adrs: ["ADR-056", "ADR-011", "ADR-024", "ADR-013"]
---

## Goals

1. Make the coordinator deadline authoritative for each graph run.
2. Prevent a timed-out run from publishing late checkpoints or overlapping a
   replacement run for the same thread.
3. Use one prebuilt LlamaIndex `RouterQueryEngine` as the sole owner of document
   retrieval strategy selection.
4. Return sanitized, structured documents for synthesis, validation, and UI
   source display.

## Non-goals

- Forcefully terminating arbitrary in-process Python threads or external side
  effects.
- Persisting query engines, indexes, retrievers, or clients in LangGraph state.
- Maintaining the v1 raw-index, `ToolFactory`, or retrieval-strategy interfaces.
- Logging prompts, queries, document text, paths, or credentials in telemetry.

## Coordinator Deadline

The coordinator seeds an absolute monotonic deadline in every `MultiAgentState`.
It rejects a missing or non-finite deadline before graph execution. An explicit
`max_agent_timeout` can lower the configured `settings.agents.decision_timeout`
for one coordinator instance.

The coordinator executes `astream(...)` on one bounded, persistent event-loop
runner. For each run it:

1. calculates the remaining wall-clock budget;
2. copies the compiled graph with public `Pregel.copy(...)` and sets that copy's
   `step_timeout` to the remaining budget;
3. supplies a public `langgraph.runtime.RunControl`;
4. waits on `Future.result(remaining)` as the authoritative sync boundary; and
5. fences the canonical user-scoped persistence key until the async wrapper
   exits.

On timeout, the coordinator requests drain, cancels the wrapper future, and
returns one stable timeout state. Only `deadline_exceeded` and
`dependency_timeout` set `timed_out=true`; capacity, closure, overlapping-run,
and generic cancellation stops set `workflow_stopped=true` with
`timed_out=false`. No stopped run is cached, consolidated into memory, or
counted as successful. A late synchronous node result is neither accepted nor
checkpointed.

### Provider timeout alignment

Coordinator-owned model calls use the smallest applicable timeout from:

- the provider request timeout;
- `settings.agents.decision_timeout`; and
- the coordinator's explicit per-run cap.

When an explicit coordinator cap is present, `ChatOpenAI.max_retries` is zero so
provider retries cannot multiply the overall budget. Qdrant calls are similarly
bounded by the smaller configured database and decision timeouts.

### Cooperative limitation

LangGraph may offload synchronous nodes and tools to worker threads. Python
cannot safely kill such a thread, so an external call may finish its own side
effect after the caller deadline if that dependency ignores cancellation. Every
provider and tool must therefore use a native timeout and make externally visible
operations idempotent where practical. Process-per-run isolation is out of scope.

## Canonical Retrieval Boundary

The Documents and Chat pages build or restore one LlamaIndex
`RouterQueryEngine`. Chat passes only that object through
`ToolRuntime.context["router_engine"]`; it is never checkpointed.

`src/agents/tools/retrieval.py::retrieve_documents` accepts only the user query
plus injected state/runtime parameters. It calls `router_engine.query(query)`
exactly once. The router owns selection among the tools assembled by
`src/retrieval/router_factory.py`:

- semantic vector search (required);
- server-side hybrid search (optional);
- sparse keyword search (optional);
- multimodal fusion (optional); and
- knowledge-graph retrieval (optional).

The v2 boundary deliberately has no `strategy`, `use_graphrag`, raw vector
index, raw graph index, hybrid retriever, or router-enable flag. Missing routers
and query failures return stable, user-safe error JSON; they do not reconstruct
an alternate retrieval stack.

Successful output contains:

- `documents` with sanitized content, score, and metadata;
- `strategy_used: "router"`;
- original and optimized query fields;
- result count and processing time; and
- `router_used: true`.

Raw filesystem paths and base64 payloads are removed before documents enter
graph state. Empty contextual follow-ups may reuse the most recent sanitized
sources from persisted state.

## Tool Invocation

The retrieval tool remains synchronously invokable for LangGraph and Streamlit.
Parallelism belongs to graph orchestration, not hidden `asyncio.gather(...)`
calls inside tools.

## Observability

Metadata-only events include:

- `agent_deadline_exceeded=true` only for actual deadline overruns;
- `agent_workflow_stopped` for every canonical stop; and
- `retrieval.backend="llama_index_router"` with stable outcomes
  `success`, `router_missing`, or `query_failed`.

## Verification

- `tests/unit/agents/tools/test_retrieval.py` proves the sole-router contract,
  one query call, document normalization, deduplication, contextual recall, and
  safe failures.
- `tests/unit/agents/tools/test_tool_registry.py` proves the registry
  exposes the structured retrieval tool.
- `tests/unit/agents/test_deadline_seeded.py` and
  `tests/unit/config/test_timeout_caps.py` prove deadline and provider caps.
- `tests/unit/agents/test_coordinator_additional_coverage.py` proves isolated
  graph copies, same-thread fencing, and prompt runner shutdown.
- `tests/integration/test_agent_timeout_behavior.py` runs a real sleeping sync
  node and proves prompt return without a late checkpoint.

All tests are deterministic and offline.

## v2 Migration

This is a forward-only hard cut. v2 removes
`AgentConfig.enable_deadline_propagation`, `AgentConfig.enable_router_injection`,
`src/agents/tool_factory.py`, raw retrieval objects in coordinator overrides, and
the retrieval tool's manual strategy parameters. Deadline propagation is
unconditional; no deadline feature flag remains. Callers must build a router
with `build_router_engine(...)` and pass only `{"router_engine": router}`.
