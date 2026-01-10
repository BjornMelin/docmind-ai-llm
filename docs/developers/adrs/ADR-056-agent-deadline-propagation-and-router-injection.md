---
ADR: 056
Title: Agent Deadline Propagation + Router Injection (Cooperative, Local-First)
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 011, 024, 031, 052
Tags: agents, langgraph, timeouts, cancellation, retrieval, streamlit
References:
  - https://langchain-ai.github.io/langgraph/how-tos/streaming/
  - https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph/interrupts
  - https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig
  - https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI/
---

## Description

Implement **cooperative deadline propagation** for the LangGraph supervisor workflow and make `AgentConfig.enable_router_injection` meaningful by using injected router engines when available, while restoring a consistent retrieval tool contract.

## Context

DocMind’s multi-agent coordinator currently:

- runs `compiled_graph.stream(…)` synchronously (Streamlit-compatible)
- enforces a wall-clock timeout only **between** streamed state transitions
- defines config flags:
  - `agents.enable_deadline_propagation`
  - `agents.enable_router_injection`
    but does not use them

Additionally, there is a **retrieval wiring mismatch**:

- The supervisor prompt describes `retrieval_agent` as “Document search with DSPy optimization”.
- The default tool registry wires the retrieval agent to a router tool that returns only a text response and does not expose retrieved documents/source nodes.

This drift limits correctness (missing sources), makes timeouts less effective (blocking calls can exceed the supervisor budget), and prevents operators from relying on configuration flags.

## Decision Drivers

- Streamlit constraints: synchronous execution, avoid async event loop complexity
- Offline-first: no new network surfaces; bounded timeouts for local endpoints
- Correctness: retrieval outputs must include documents/sources for synthesis/validation
- Reliability: prevent “runaway retries” when time budget is exhausted
- Maintainability: one clear contract for “retrieval tool result”

## Alternatives

- A: **Cooperative deadline propagation + injected router engine path** (Selected)
  - Store absolute deadline in agent state
  - Cap request timeouts (LLM/Qdrant) to the decision timeout
  - Prefer injected router engine for retrieval when enabled
  - Restore retrieval tool contract to return structured docs (sources)
  - Pros: works with sync Streamlit, minimal re-platforming, improves correctness
  - Cons: cannot preempt a truly blocking call; relies on bounded timeouts
- B: Coordinator-only hard timeout (thread/future wrapper)
  - Pros: quick UX stopgap
  - Cons: does not stop nested calls; can waste compute and leave background work running
- C: Full async refactor to `astream` + cancellation
  - Pros: strongest cancellation semantics
  - Cons: high-risk in Streamlit; large migration surface
- D: Interrupt/stop-flag “pseudo cancellation” only
  - Pros: leverages LangGraph primitives for control flow
  - Cons: still requires cooperative timeouts; more scattered logic without explicit budgets

### Decision Framework (≥9.0)

| Option                                         | Complexity (40%) | Perf (30%) | Alignment (30%) |    Total | Decision    |
| ---------------------------------------------- | ---------------: | ---------: | --------------: | -------: | ----------- |
| **A: Deadline propagation + router injection** |              8.8 |        9.2 |             9.6 | **9.16** | ✅ Selected |
| B: Coordinator-only hard timeout               |              7.5 |        7.0 |             7.5 |     7.35 | Rejected    |
| C: Full async cancellation refactor            |              4.0 |        9.0 |             6.0 |     6.10 | Rejected    |
| D: Stop-flag/interrupt only                    |              6.5 |        7.5 |             7.0 |     6.95 | Rejected    |

## Decision

We will implement a **budget-aware, cooperative timeout model**:

1. Add an **absolute deadline** to `MultiAgentState` when `settings.agents.enable_deadline_propagation` is enabled.
2. Cap request timeouts (LLM and Qdrant client timeouts) so individual calls cannot exceed the agent decision timeout.
3. Restore a consistent retrieval tool contract by using the structured `retrieve_documents` tool for the retrieval agent.
4. If `settings.agents.enable_router_injection` is enabled and a `router_engine` is injected in state/tools_data, prefer it as the retrieval path (fail-open to explicit retrieval).

## High-Level Architecture

```mermaid
flowchart TD
  COORD[Coordinator] -->|seed deadline_ts| STATE[MultiAgentState]
  STATE -->|InjectedState| TOOLS[Tools]
  TOOLS -->|remaining budget| CALLS[LLM/Qdrant calls with capped timeouts]
  STATE -->|router_engine injected| ROUTE[RouterQueryEngine (optional)]
  ROUTE -->|fallback| EXPL[Explicit retrieval via vector/hybrid tools]
```

## Security & Privacy

- No new network surfaces; remote endpoints remain gated by the allowlist policy.
- Deadline propagation reduces runaway calls but must not log raw user content.
- Router injection must not bypass allowlist enforcement (it should only route within local stores).

## Testing

- Unit:
  - `deadline_ts` seeded in state when enabled
  - LLM and Qdrant timeouts are capped to the decision timeout
  - retrieval tool returns structured payload with documents
  - router injection preferred when enabled and router_engine is present
- Integration:
  - run a short decision timeout and verify the coordinator returns a timeout response without waiting for long retries

## Consequences

### Positive Outcomes

- Retrieval results consistently include sources for downstream synthesis/validation and UI display.
- Timeouts become more meaningful (per-call timeouts align to the overall decision budget).
- Config flags `enable_deadline_propagation` and `enable_router_injection` become real controls.

### Negative Consequences / Trade-offs

- Cooperative timeouts cannot force-cancel a call that ignores timeouts.
- Slightly more complexity in tool wiring and state schema.
