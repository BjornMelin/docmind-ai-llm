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

## Libraries and Imports

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
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
