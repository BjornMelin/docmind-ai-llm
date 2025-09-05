---
spec: SPEC-008
title: Streamlit UI: Programmatic Multipage, Native Chat, Settings, Documents, Analytics
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - FR-UI-001: Use st.Page/st.navigation for multipage nav.
  - FR-UI-002: Use st.chat_message/input and st.write_stream for chat.
  - FR-UI-003: Provide Settings form for providers, models, retrieval knobs.
  - FR-UI-004: Documents page to ingest files and show status/progress.
related_adrs: ["ADR-012","ADR-016","ADR-013"]
---


## Objective

Build a clean Streamlit UI using programmatic pages: Chat, Documents, Analytics, Settings. Use native chat and streaming, forms to avoid rerun churn, and status/toast/dialog for UX.

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
Feature: Chat streaming
  Scenario: Ask a question
    Given an indexed corpus
    When I ask in Chat
    Then tokens SHALL stream using st.write_stream

Feature: Settings save
  Scenario: Change provider
    Given Settings page
    When I change provider and Save
    Then the provider badge SHALL update immediately
```

## References

- Streamlit docs for chat, write_stream, forms, status, navigation.
