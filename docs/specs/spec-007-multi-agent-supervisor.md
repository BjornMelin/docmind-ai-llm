---
spec: SPEC-007
title: LangGraph Supervisor Orchestrator with Deterministic JSON-Schema Outputs
version: 1.2.0
date: 2026-05-01
owners: ["ai-arch"]
status: Revised
related_requirements:
  - FR-AGENT-001: Use a graph-native LangGraph `StateGraph` supervisor to coordinate specialized agents (no external supervisor wrapper dependency).
  - FR-AGENT-002: Enforce JSON schema outputs where backend supports structured decoding.
  - FR-AGENT-003: Provide stop conditions and max step limits.
related_adrs: ["ADR-001","ADR-011","ADR-066"]
---


## Objective

Implement and integrate DocMind’s **multi-agent supervisor** with the RAG pipeline and tools using a **graph-native** LangGraph `StateGraph` supervisor. Use schema-guided decoding for deterministic outputs when the selected LLM provider supports it.

ADR-066 confirms that LangGraph remains DocMind's default agent runtime. Modern
LlamaIndex Workflows may only be evaluated as a future contained pilot and must
not replace this supervisor spec without a separate approved hard-cut decision.

## Provider Capability Notes (Structured Outputs / Tools)

- vLLM may support schema-guided decoding via server-side guided JSON (see [SPEC-001](./spec-001-llm-runtime.md)).
- Ollama supports native structured outputs via `format` and optional "thinking" and tool-calling metadata; when using Ollama-native APIs, follow [SPEC-043](./spec-043-ollama-native-capabilities.md) and treat these fields as optional metadata (do not assume presence or log verbatim traces).
- Cloud web tools (web search/fetch) MUST remain opt-in and gated by security allowlists ([SPEC-011](./spec-011-security-privacy.md) + [SPEC-043](./spec-043-ollama-native-capabilities.md)).

## Libraries and Imports

Use `langchain.agents.create_agent` (LangChain v1) for agent construction and keep
DocMind’s repo-local `StateGraph` supervisor (`src/agents/supervisor_graph.py`) for orchestration.

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from src.agents.supervisor_graph import build_multi_agent_supervisor_graph
from src.retrieval.router_factory import build_router_engine
```

## File Operations

### CREATE

- `src/agents/supervisor_graph.py`: repo-local supervisor graph builder and handoff tools.
- `src/agents/coordinator.py`: coordinator wiring + compilation with checkpointer/store + streaming.
- Agent tools live under `src/agents/tools/` and are registered via the tool registry/factory.

### UPDATE

- `src/app.py` and `src/pages/01_chat.py`: send queries via the coordinator; stream responses.

## Runtime Replacement Non-Goals

- Do not replace the LangGraph supervisor in issue #86.
- Do not add the old `llama-agents` PyPI package.
- Do not add `llama-agents-server`, `llama-agents-client`, or `llamactl` as
  runtime dependencies for the default app path.
- Do not add a dual-runtime abstraction or orchestration feature flag.
- Do not introduce adapters that translate between LangGraph state and
  LlamaIndex Workflow context unless a future approved hard-cut requires it.

## Future LlamaIndex Workflows Pilot Gate

A future pilot MAY evaluate `llama-index-workflows` on synthesis plus
validation only. The pilot MUST remain isolated from the default runtime and
MUST prove:

- checkpoint, resume, and time-travel parity or a documented blocking gap
- store and persistence compatibility
- streaming event shape compatibility
- handoff and backflow semantics
- timeout and deadline propagation
- semantic cache policy preservation
- metadata-only telemetry and log safety
- offline-first behavior with local/degraded LLM backends
- deterministic failure and retry behavior
- focused regression tests
- net code deletion potential before replacement is considered

## Acceptance Criteria

```gherkin
Feature: Supervisor routing
  Scenario: Retrieval tool use
    Given a user query
    Then the supervisor SHALL call the retrieval tool
    And compose a final answer following the JSON schema
```

## References

- LangGraph supervisor official tutorials and repo.
- [ADR-066](../developers/adrs/ADR-066-llamaindex-workflows-orchestration-evaluation.md)
- [SPEC-043](./spec-043-ollama-native-capabilities.md) (Ollama native SDK integration and capability gating).
