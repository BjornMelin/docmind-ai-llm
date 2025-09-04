---
ADR: 016
Title: Streamlit Native State Management
Status: Accepted
Version: 4.3
Date: 2025-08-18
Supersedes:
Superseded-by:
Related: 013, 021
Tags: ui, state, streamlit
References:
- [Streamlit — Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/session-state)
- [Streamlit — Caching](https://docs.streamlit.io/develop/api-reference/caching)
---

## Description

Use Streamlit’s native `st.session_state`, `st.cache_data`, and `st.cache_resource` directly. Avoid custom state/caching layers; keep the UI simple and traceable.

## Context

Earlier designs introduced custom state managers and wrappers. They added complexity without clear benefit for a local, single‑user app.

## Decision Drivers

- KISS: fewer layers, easier debugging
- Good enough performance with native cache
- Clear page/state mental model

## Alternatives

- Custom state/cache managers — harder to reason about
- External state backends — overkill for local app

### Decision Framework

| Option                 | Simplicity (50%) | Perf (20%) | Maintainability (20%) | DX (10%) | Total | Decision      |
| ---------------------- | ---------------- | ---------- | --------------------- | -------- | ----- | ------------- |
| Native Streamlit (Sel) | 10               | 8          | 10                    | 9        | 9.4   | ✅ Selected    |
| Custom managers        | 4                | 8          | 5                     | 6        | 5.5   | Rejected      |

## Decision

Adopt Streamlit natives for state and caching; only add external store if future multi‑user features require it.

## High-Level Architecture

Pages → session_state (dict) + cache_data/resource → UI

## Related Requirements

### Functional Requirements

- FR‑1: Persist chat history within a session
- FR‑2: Cache expensive computations/resources

### Non-Functional Requirements

- NFR‑1: Simple code paths; <200 LOC per page

### Performance Requirements

- PR‑1: State updates render under 50ms typical

### Integration Requirements

- IR‑1: Aligns with ADR‑013 page structure; no custom managers

## Design

### Architecture Overview

- Pages use `session_state`; expensive ops cached; partial updates via `fragment`

### Implementation Details

```python
# pages/chat.py (state skeleton)
import streamlit as st

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if prompt := st.chat_input("Ask a question"):
    st.session_state['messages'].append({"role":"user","content":prompt})
    st.session_state['messages'].append({"role":"assistant","content":"Hello!"})
    st.rerun()
```

### Extended Patterns

```python
# Cache examples
import streamlit as st
from datetime import timedelta

@st.cache_data(ttl=timedelta(minutes=5))
def search_documents(query: str, filters: dict):
    return vector_db.search(query, filters)

@st.cache_resource
def get_vector_db():
    from qdrant_client import QdrantClient
    return QdrantClient(path="./data/qdrant")

# Fragments for partial updates
@st.fragment(run_every=5)
def metrics_display():
    m = get_metrics()
    st.metric("Documents", m['doc_count'])
```

### Performance Tips

- Prefer `cache_data` for pure data; `cache_resource` for singletons
- Use `fragment` to avoid full rerenders for fast dashboards
- Clear caches selectively during heavy updates

### Configuration

- No extra config beyond Streamlit defaults

## Testing

- UI smoke tests and basic state assertions

## Consequences

### Positive Outcomes

- Less code; easier to maintain
- Predictable behavior using documented Streamlit patterns

### Negative Consequences / Trade-offs

- Fewer knobs than custom layers

### Ongoing Maintenance & Considerations

- Avoid storing large payloads in session_state; prefer lightweight refs
- Clear caches selectively after heavy updates

### Dependencies

- Python: `streamlit>=1.36`

## Changelog

- 4.3 (2025‑09‑04): Standardized to template; clarified requirements/tests

- 4.2 (2025‑08‑18): Accepted native state/caching; removed custom layers
