---
ADR: 011
Title: Agent Orchestration with LangGraph Supervisor
Status: Accepted (Amended)
Version: 6.3
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 001, 003, 004, 010, 015, 016, 024
Tags: orchestration, agents, langgraph, supervisor
References:
- [LangGraph Supervisor — GitHub](https://github.com/langchain-ai/langgraph-supervisor-py)
- [LangGraph — Official Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Multi‑Agent Patterns (Supervisor)](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#multi-agent)
---

## Description

Use LangGraph Supervisor to coordinate five agent roles (router, planner, retrieval, synthesis, validation). Replace custom orchestration logic with proven, library‑first patterns that are easy to test and maintain.

## Context

DocMind AI relies on multiple agent roles that must coordinate while preserving context and working fully offline. A prebuilt supervisor pattern avoids bespoke state machines and reduces orchestration bugs.

## Decision Drivers

- Simplicity over bespoke orchestration
- Local‑first operation (no external services)
- Clear observability of handoffs and outcomes
- Compatibility with adaptive retrieval (ADR‑003)

## Alternatives

- Monolithic agent — simple but inflexible; weak error recovery
- Manual orchestration — complex state and error handling
- Heavy multi‑agent frameworks — overkill for local desktop app
- LangGraph Supervisor (Selected) — prebuilt, testable patterns

### Decision Framework

| Option               | Simplicity (40%) | Reliability (30%) | Capability (20%) | Effort (10%) | Total | Decision    |
| -------------------- | ---------------- | ----------------- | ---------------- | ------------ | ----- | ----------- |
| LangGraph Supervisor | 10               | 9                 | 9                | 9            | 9.5   | ✅ Selected |
| Manual orchestration | 3                | 5                 | 8                | 4            | 4.9   | Rejected    |
| Monolithic agent     | 9                | 4                 | 4                | 8            | 6.1   | Rejected    |

## Decision

Adopt LangGraph Supervisor for five‑agent coordination with minimal customization. Surface small, explicit configuration for logging, parallel tool calls (ADR‑010), and guardrails.

### Agent Roles and Coordinator

- `Supervisor/Coordinator`: Central controller that routes, hands off, and collects outcomes; owns prompt and guardrails
- `Router`: Chooses retrieval path (hybrid, hierarchical, graph) based on query/metadata
- `Planner`: Decomposes complex queries into sub‑tasks (optional early‑exit for simple queries)
- `Retrieval`: Executes adaptive retrieval (ADR‑003), including multimodal reranking (ADR‑037)
- `Synthesis`: Aggregates evidence into a coherent answer with citations
- `Validation`: Checks relevance/faithfulness; may trigger correction or re‑route

## High-Level Architecture

User → Supervisor → {Router → Planner → Retrieval → Synthesis → Validation} → Response

```mermaid
graph TD
  U["User"] --> S["Supervisor"]
  S --> R["Router"]
  R -->|complex| P["Planner"]
  R -->|simple| T["Retrieval"]
  P --> T
  T --> Y["Synthesis"]
  Y --> V["Validation"]
  V --> O["Response"]
```

## Related Requirements

### Functional Requirements

- FR‑1: Orchestrate multi‑agent workflows with conditional execution
- FR‑2: Maintain conversation context and pass state
- FR‑3: Provide fallback and retries for failed steps

### Non-Functional Requirements

- NFR‑1: Coordination overhead ≤500ms
- NFR‑2: Local‑first; no external services
- NFR‑3: Clear boundaries; small testable units

### Performance Requirements

- PR‑1: Coordination overhead ≤500ms at P95
- PR‑2: Parallel tool calls available where safe (see ADR‑010)

### Integration Requirements

- IR‑1: Integrates with ADR‑003 retrieval and ADR‑004 model
- IR‑2: Exposes minimal settings via ADR‑024 config

## Design

### Architecture Overview

- Five roles implemented with prebuilt agents; early exits where applicable
- Minimal prompts; rely on Supervisor primitives

### Implementation Details

```python
# src/agents/coordinator.py (skeleton)
from langgraph_supervisor import create_supervisor

SUPERVISOR_PROMPT = (
    "You are a supervisor coordinating a modern 5‑agent RAG system. "
    "Route, plan, retrieve, synthesize, and validate answers. Prefer minimal steps."
)

def create_app(llm, tools):
    agents = make_agents(llm, tools)  # router, planner, retrieval, synthesis, validation
    return create_supervisor(
        agents=agents,
        model=llm,
        prompt=SUPERVISOR_PROMPT,
        parallel_tool_calls=True,          # concurrent agent/tool paths
        output_mode="last_message",        # structured data stays in state
        add_handoff_messages=True,         # track coordination handoffs
        # Optional hooks for context/window management
        pre_model_hook=trim_context_hook,  # enforce 128K cap (ADR‑004/010)
        post_model_hook=collect_metrics_hook,
        tools=[create_forward_message_tool("supervisor")],
    )
```

### Supervisor Configuration (Key Options)

- `parallel_tool_calls`: Enable concurrent tool/agent branches to reduce tokens (50–87%)
- `output_mode`: Controls message history added by the supervisor. Supported values:
  - `"last_message"` (default): Add only the final agent message
  - `"full_history"`: Add the entire agent message history
- `create_forward_message_tool`: Allow direct message passthrough when no processing is needed
- `add_handoff_messages`: Emit coordination breadcrumbs for debugging and audits
- `pre_model_hook`/`post_model_hook`: Trim context (e.g., at ~120K) and attach metrics (always-on)

### Configuration

- Flags: `parallel_tool_calls`, `max_parallel_calls`, log level
- Expose as part of unified settings (ADR‑024)

```env
DOCMIND_AGENTS__ENABLE_PARALLEL_TOOL_EXECUTION=true
DOCMIND_AGENTS__MAX_CONCURRENT_AGENTS=3
DOCMIND_LOG_LEVEL=INFO
```

### Deprecations

- `output_mode="structured"` is not supported by the Supervisor. Structured metadata belongs in state and response models. Use `output_mode="last_message"` or `"full_history"` and rely on hooks/state to record metrics.
- `add_handoff_back_messages` is no longer relied upon; use `add_handoff_messages=True` to include handoff traces.

## Testing

```python
def test_supervisor_boots_with_agents(supervisor_app):
    result = supervisor_app.invoke({"messages": [{"role": "user", "content": "hi"}]})
    assert "messages" in result
```

## Limitations / Future Improvements

- Deadline propagation: The supervisor’s wall-clock timeout (decision timeout) is enforced at the coordinator boundary. Deadlines are not yet propagated to nested LLM/tool calls, so a long-running subcall may continue even after the overall timeout has elapsed. In practice, the coordinator returns a timeout fallback quickly and stops streaming to the UI, but subcalls may continue in the background. A future enhancement should propagate an absolute deadline or remaining time budget through agent graph calls and tools to enable cooperative cancellation.

## Consequences

### Positive Outcomes

- Simpler code; fewer orchestration bugs
- Clear observability and testability

### Negative Consequences / Trade-offs

- Some constraints from the framework’s control flow

### Dependencies

- Python: `langgraph>=1.0.5`, `langchain-core>=1.2.6`, `langgraph-supervisor>=0.0.31`

### Ongoing Maintenance & Considerations

- Track LangGraph breaking changes; pin versions in lockfile
- Periodically review agent prompts and routing heuristics with logs
- Keep supervisor config minimal; avoid custom state machines

## Changelog

- 6.2 (2025‑09‑04): Restored explicit agent role list and supervisor configuration details (parallel_tool_calls, output_mode, create_forward_message_tool, add_handoff_back_messages) with hook notes and prompt
- 6.1 (2025‑09‑04): Standardized to template; added diagram, PR/IR, config/tests
- 6.0 (2025‑08‑19): Accepted Supervisor implementation; integrates ADR‑003/004/010

## Supervisor Configuration Updates

- Supervisor `output_mode` MUST be one of `last_message` (default) or `full_history`. A custom `structured` output mode is NOT supported; structured metadata SHOULD be carried in state/response models.
- Rename legacy `add_handoff_back_messages` to `add_handoff_messages`.
- Prefer `create_forward_message_tool("supervisor")` when summarization is unnecessary.

## Performance & Timeouts

- Coordinators SHOULD enforce elapsed‑time guards around streaming and tool execution; publish sane defaults and document override points.

## Streaming Fallback & Analytics

- The Chat UI MUST implement streaming fallback using chunked `write_stream`.
- Analytics MUST be best‑effort and non‑blocking; failures MUST NOT impact the user experience.
