---
spec: SPEC-007
title: LangGraph Supervisor Orchestrator with Deterministic JSON-Schema Outputs
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
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
from src.retrieval.query_engine import build_hybrid_retriever
```

## File Operations

### CREATE

- `src/agents/supervisor.py`: supervisor factory with config flags.
- `src/agents/tools/router_tool.py`, `retrieval.py`, `synthesis.py`, `validation.py` already exist; ensure registration with supervisor.

### UPDATE

- `src/app.py` and `src/pages/chat.py`: send queries via supervisor; stream responses.

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
