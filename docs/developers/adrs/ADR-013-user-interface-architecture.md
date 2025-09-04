---
ADR: 013
Title: Streamlit UI Architecture (Multipage, Native Components)
Status: Accepted
Version: 3.2
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 001, 009, 016, 021, 032, 036
Tags: ui, streamlit, state, analytics
References:
- [Streamlit — Official Docs](https://docs.streamlit.io/)
- [Streamlit State & Caching](https://docs.streamlit.io/develop/api-reference/caching)
---

## Description

Adopt a clean, multipage Streamlit UI using native components for navigation, state, streaming, and caching. Avoid custom UI frameworks and heavy component stacks; keep pages small and focused.

## Context

The app needs agentic chat, document management, analytics, and settings in a local‑first desktop flow. Streamlit’s native features cover these without extra layers.

## Decision Drivers

- Simplicity and performance with native components
- Clear session state and caching model
- Minimal dependencies; fast iteration

## Alternatives

- Directory‑based pages only — limited UX/state
- Gradio — quick, but constrained for multipage apps
- FastAPI+React — overkill for local desktop app

### Decision Framework

| Option               | Dev Speed (40%) | UX (30%) | Simplicity (20%) | Perf (10%) | Total | Decision      |
| -------------------- | --------------- | -------- | ---------------- | ---------- | ----- | ------------- |
| Streamlit native     | 10              | 9        | 9                | 8          | 9.3   | ✅ Selected    |
| FastAPI+React        | 5               | 10       | 4                | 9          | 6.7   | Rejected      |
| Gradio               | 9               | 6        | 8                | 8          | 7.8   | Rejected      |

## Decision

Use native Streamlit (pages, session_state, cache, chat streaming) for all UI needs. Keep files short and cohesive.

## High-Level Architecture

Entry (`app.py`) → Navigation → {Chat, Documents, Analytics, Settings}

## Related Requirements

### Functional Requirements

- FR‑1: Multipage navigation and persistent session state
- FR‑2: Chat with streaming responses and source display
- FR‑3: Documents table with filters and bulk actions
- FR‑4: Analytics dashboard with core metrics

### Non-Functional Requirements

- NFR‑1: <2s page load; <100ms UI interactions
- NFR‑2: Minimal dependencies; native components first

### Performance Requirements

- PR‑1: Chat streaming token updates render within 50ms/frame
- PR‑2: Table interactions update under 150ms P95 for 1k rows

### Integration Requirements

- IR‑1: `.streamlit/config.toml` governs theme
- IR‑2: UI toggles map to ADR‑024 settings

## Design

### Architecture Overview

- `app.py` bootstraps pages and theme
- Pages: `pages/chat.py`, `pages/documents.py`, `pages/analytics.py`, `pages/settings.py`
- State: `st.session_state` with a light dataclass wrapper (optional)

### Implementation Details

```python
# pages/chat.py (skeleton)
import streamlit as st

def stream_tokens(question: str):
    for t in ("Thinking... ", "Answer ", "here."):
        yield t

st.title("Chat")
if prompt := st.chat_input("Ask a question"):
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        full = st.write_stream(stream_tokens(prompt))
        st.session_state.setdefault("messages", []).append({"role": "assistant", "content": full})
```

### Configuration

- `.streamlit/config.toml` for theme
- Env toggles map to ADR‑024 settings

```toml
# .streamlit/config.toml
[theme]
base = "light"
primaryColor = "#4A90E2"
```

```env
DOCMIND_UI__SHOW_ADVANCED=false
```

## Testing

- Smoke tests for page boot and basic interactions
- Snapshot tests for key components when feasible

```python
def test_chat_page_boot(app_runner):
    resp = app_runner.open("/chat")
    assert resp.status_code == 200
```

## Consequences

### Positive Outcomes

- Fast iteration, minimal code, native reliability
- Clear state/caching patterns

### Negative Consequences / Trade-offs

- Less control than a full web stack

### Dependencies

- Python: `streamlit>=1.36`
- Optional: `plotly>=5.17`

## Changelog

- 3.2 (2025‑09‑04): Standardized to template; added PR/IR, config/tests
- 3.1 (2025‑09‑03): Accepted modernized native multipage UI
