---
spec: SPEC-007
title: LangGraph Supervisor Orchestrator with Deterministic JSON-Schema Outputs
version: 1.0.1
date: 2026-01-09
owners: ["ai-arch"]
status: Revised
related_requirements:
  - FR-AGENT-001: Use langgraph-supervisor-py to coordinate specialized agents.
  - FR-AGENT-002: Enforce JSON schema outputs where backend supports structured decoding.
  - FR-AGENT-003: Provide stop conditions and max step limits.
related_adrs: ["ADR-001","ADR-011"]
---


## Objective

Restore and integrate your **langgraph-supervisor-py** multi-agent system with the RAG pipeline and tools. Use schema-guided decoding for deterministic outputs when LLM provider supports it.

## Provider Capability Notes (Structured Outputs / Tools)

- vLLM may support schema-guided decoding via server-side guided JSON (see SPEC-001).
- Ollama supports native structured outputs via `format` and optional “thinking” and tool calling metadata; when using Ollama-native APIs, follow SPEC-043 and treat these fields as optional metadata (do not assume presence or log verbatim traces).
- Cloud web tools (web search/fetch) MUST remain opt-in and gated by security allowlists (SPEC-011 + SPEC-043).

## Libraries and Imports

LangGraph v1 deprecates `langgraph.prebuilt.create_react_agent`. Use
`langchain.agents.create_agent` (LangChain v1) for agent construction and keep
`langgraph-supervisor` for orchestration.

```python
from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
from langchain_core.tools import tool
from src.retrieval.router_factory import build_router_engine
```

## File Operations

### CREATE

- `src/agents/coordinator.py`: supervisor coordinator using `langgraph-supervisor` and registered tools.
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
- SPEC-043 (Ollama native SDK integration and capability gating).
