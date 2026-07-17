---
spec: SPEC-008
title: Streamlit UI: Programmatic Multipage, Truthful Chat, Settings, Documents, Analytics
version: 1.1.0
date: 2026-07-16
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-010: Streamlit multipage navigation.
  - FR-011: Truthful native Chat status and terminal rendering.
  - FR-012: Multi-provider settings in UI.
  - FR-001: Backend document ingestion through the CPU-safe parser boundary.
related_adrs: ["ADR-013","ADR-016","ADR-041","ADR-052","ADR-057","ADR-009","ADR-002"]
---


## Objective

Build a clean Streamlit UI using programmatic pages: Chat, Documents, Analytics,
and Settings. Use native Chat elements and truthful generation status, forms to
avoid rerun churn, and status/toast/dialog primitives for UX.

The current public coordinator API is synchronous and returns one completed
`AgentResponse`. Chat therefore shows a native spinner only while that call is
running and renders the completed answer normally. It MUST NOT split a completed
answer into delayed chunks or claim incremental token delivery. Future real
streaming requires a public coordinator event API and an end-to-end test proving
output arrives before completion.

## Libraries and Imports

```python
import streamlit as st
from src.config import settings, setup_llamaindex
from src.ui.components.provider_badge import provider_badge
from src.utils.storage import human_size
```

## File Operations

### CREATE

- `src/pages/01_chat.py`, `src/pages/02_documents.py`, `src/pages/03_analytics.py`, `src/pages/04_settings.py`.
- `src/ui/components/provider_badge.py`.

### UPDATE

- `src/app.py`: programmatic navigation via `st.Page`/`st.navigation`.

## Acceptance Criteria

```gherkin
Feature: Truthful Chat state
  Scenario: Ask a question
    Given an indexed corpus
    When I ask in Chat
    Then a native generation spinner SHALL wrap the synchronous coordinator call
    And the completed answer SHALL render normally after the call returns
    And the UI SHALL NOT simulate streaming by chunking completed text

  Scenario: Chat history is empty
    Given the history read succeeds with no messages
    Then the UI SHALL identify the conversation as new

  Scenario: Chat history cannot be read
    Given the public history read raises or runtime maintenance is active
    Then the UI SHALL distinguish that state from an empty conversation
    And offer a retry without exposing raw errors or user/thread identifiers

Feature: Settings save
  Scenario: Change provider
    Given Settings page
    When I change provider and Save
    Then the provider badge SHALL update immediately
```

## References

- Streamlit docs for chat, spinner, forms, status, and navigation.
