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

### Extended Implementation Guide

```python
# Streaming with sources and status
import streamlit as st
import time
from typing import Iterator, AsyncGenerator

def stream_llm_response(q: str) -> Iterator[str]:
    # connect to LLM client in app code
    for t in ["Thinking… ", "answer ", "tokens."]:
        yield t
        time.sleep(0.01)

def stream_with_sources(query: str, sources: list[dict]):
    col1, col2 = st.columns([2, 1])
    with col1:
        buf = ""
        slot = st.empty()
        for chunk in stream_llm_response(query):
            buf += chunk
            slot.markdown(buf)
    with col2:
        st.subheader("Sources")
        for src in sources:
            with st.expander(src.get("title", "Source")):
                st.write((src.get("text") or "")[:200] + "…")

def process_documents_with_status(uploaded_files):
    with st.status("Processing documents…", expanded=True) as status:
        try:
            st.write("🔍 Validating files…")
            time.sleep(0.2)
            st.write("📄 Parsing and extracting content…")
            st.write("🧮 Creating embeddings…")
            st.write("💾 Storing in vector database…")
            status.update(label="✅ Processing complete!", state="complete")
        except Exception as e:  # replace with specific exceptions in app code
            status.update(label="🚨 Error processing documents", state="error")
            st.error(f"Processing failed: {e}")

def safe_operation_with_feedback(name: str):
    def deco(fn):
        def wrapper(*args, **kwargs):
            try:
                with st.spinner(f"{name}…"):
                    return fn(*args, **kwargs)
            except Exception as e:  # replace with specific exceptions in app code
                st.error(f"{name} failed: {e}")
                return None
        return wrapper
    return deco
```

### Entry Point Skeleton (`app.py`)

```python
import streamlit as st

st.set_page_config(page_title="DocMind AI", page_icon="📄", layout="wide")

pages = {
    "Chat": st.Page("pages/chat.py", title="Chat", icon="💬", default=True),
    "Documents": st.Page("pages/documents.py", title="Documents", icon="📁"),
    "Analytics": st.Page("pages/analytics.py", title="Analytics", icon="📊"),
    "Settings": st.Page("pages/settings.py", title="Settings", icon="⚙️"),
}

st.navigation({"Main": [pages["Chat"], pages["Documents"]], "System": [pages["Analytics"], pages["Settings"]]}).run()
```

### Implementation Phases

- Foundation: app.py, base pages, theme, session schema
- Core: chat streaming, document upload/table, settings, basic analytics
- Integration: hook ADR‑001/003/009/032 flows
- Polish: loading/error states, a11y, caching, tests

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

### Ongoing Maintenance & Considerations

- Track Streamlit release notes for changes to chat/streaming APIs
- Keep pages small and cohesive; factor shared bits into tiny helpers
- Avoid non‑native components unless a clear, measurable need emerges

### Monitoring Metrics

- Page load and interaction latency
- Cache hit rate and session state size
- Table render performance on large datasets

## Changelog

- 3.2 (2025‑09‑04): Standardized to template; added PR/IR, config/tests
- 3.1 (2025‑09‑03): Accepted modernized native multipage UI
