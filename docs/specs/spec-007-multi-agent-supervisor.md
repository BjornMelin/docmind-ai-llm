---
spec: SPEC-007
title: LangGraph Supervisor Orchestrator with Deterministic JSON-Schema Outputs
version: 1.1.0
date: 2026-01-17
owners: ["ai-arch"]
status: Revised
related_requirements:
  - FR-AGENT-001: Use a graph-native LangGraph `StateGraph` supervisor to coordinate specialized agents (no external supervisor wrapper dependency).
  - FR-AGENT-002: Enforce JSON schema outputs where backend supports structured decoding.
  - FR-AGENT-003: Provide stop conditions and max step limits.
related_adrs: ["ADR-001","ADR-011"]
---


## Objective

Implement and integrate DocMind’s **multi-agent supervisor** with the RAG pipeline and tools using a **graph-native** LangGraph `StateGraph` supervisor. Use schema-guided decoding for deterministic outputs when the selected LLM provider supports it.

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
- [SPEC-043](./spec-043-ollama-native-capabilities.md) (Ollama native SDK integration and capability gating).
