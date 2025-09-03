---
ADR: 036
Title: Reranker UI Controls (normalize_scores, top_n, mode)
Status: Accepted
Version: 1.1.0
Date: 2025-09-03
Supersedes:
Superseded-by:
Related: 003, 024, 037
Tags: ui, streamlit, reranking
References:
- [Streamlit docs](https://docs.streamlit.io/)
---

## Description

Expose CrossEncoder/multimodal reranker controls in Streamlit UI: normalize_scores, top_n, and Reranker Mode

Expose three controls in the sidebar to tune reranking behavior: 1) a checkbox to enable sigmoid score normalization, 2) a number input to set `top_n` (1–20), and 3) a radio for `Reranker Mode` (`auto|text|multimodal`). Values persist through the settings model and are consumed by the reranker factory.

## Context

Our reranking uses a sentence-transformers CrossEncoder (`BAAI/bge-reranker-v2-m3`) for text and ColPali for visual nodes (ADR‑037). Operators need lightweight control to adjust reranked list size, normalization, and mode selection without editing environment variables.

## Decision Drivers

- KISS: two controls only; no extra knobs
- Library-first: reuse existing reranker and settings
- Operator ergonomics: UI-first tuning; settings persistence
- Offline determinism: UI tests via Streamlit AppTest

## Alternatives

- **A**: No UI; config-only via env
  - Pros: zero UI work; stable
  - Cons: poor operator feedback loop; less discoverable
- **B**: Full advanced panel (thresholds, model, device)
  - Pros: more control
  - Cons: over-engineered; contradicts KISS and v1 scope
- **C**: Two controls only [Selected]
  - Pros: minimal, high-value, testable
  - Cons: fewer options exposed (by design)

### Decision Framework

| Option | Simplicity (40%) | Operator Value (30%) | Testability (20%) | Alignment (10%) | Total | Decision |
|-------|------------------|----------------------|-------------------|-----------------|-------|----------|
| **C** | 1.0              | 0.9                  | 0.9               | 0.9             | **0.94** | ✅ Selected |
| A     | 0.9              | 0.4                  | 0.9               | 0.9             | 0.78  | Rejected |
| B     | 0.5              | 0.9                  | 0.6               | 0.8             | 0.67  | Rejected |

## Decision

- Add in `src/app.py` (sidebar):
  - Checkbox `"Reranker: Normalize scores"` (default True)
  - Number input `"Reranker: Top N"` (min=1, max=20; default from settings)
  - Radio `"Reranker Mode"` with options `["auto","text","multimodal"]` (default `auto`)
- Wiring: update `settings.retrieval.reranker_normalize_scores`, `settings.retrieval.reranking_top_k`, and `settings.retrieval.reranker_mode` before constructing reranker(s).
- Bounds and validation: rely on Streamlit `number_input` and `radio` built-in constraints.

## High-Level Architecture

```mermaid
graph TD
  UI[Streamlit Sidebar Controls]
  UI --> S[Settings.retrieval]
  S --> F[Reranker Factory]
  F --> CE[CrossEncoder (BGE-v2-m3) + ColPali]
```

## Related Requirements

### Functional Requirements

- **FR-1**: Toggle sigmoid normalization on/off
- **FR-2**: Set top_n in [1,20] to control reranked list size

### Non-Functional Requirements

- **NFR-1**: Minimal UI surface; no advanced controls in v1.0.0
- **NFR-2**: Deterministic E2E testing via AppTest

### Performance Requirements

- **PR-1**: No measurable overhead beyond UI read/write

### Integration Requirements

- **IR-1**: Read/write via existing settings singleton (ADR-024)
- **IR-2**: Works with modality-aware reranking (ADR‑037) and UI architecture (ADR-013/ADR-016)

## Related Decisions

- **ADR-006**: Reranking architecture—this ADR adds operator controls for normalization and list size
- **ADR-013**: UI architecture—controls live in Streamlit sidebar
- **ADR-016**: UI state management—session_state initialization/persistence
- **ADR-024**: Unified configuration—fields already defined in RetrievalConfig

## Design

### Implementation Details

**In `src/app.py`:**

```python
with st.sidebar:
    st.markdown("### Retrieval & Reranking")
    if "reranker_normalize_scores" not in st.session_state:
        st.session_state["reranker_normalize_scores"] = settings.retrieval.reranker_normalize_scores
    if "reranking_top_k" not in st.session_state:
        st.session_state["reranking_top_k"] = settings.retrieval.reranking_top_k

    norm = st.checkbox(
        "Reranker: Normalize scores",
        key="reranker_normalize_scores",
        value=st.session_state["reranker_normalize_scores"],
    )
    top_n = st.number_input(
        "Reranker: Top N",
        key="reranking_top_k",
        min_value=1,
        max_value=20,
        value=st.session_state["reranking_top_k"],
    )

# Persist to settings before constructing reranker
settings.retrieval.reranker_normalize_scores = bool(st.session_state["reranker_normalize_scores"])
settings.retrieval.reranking_top_k = int(st.session_state["reranking_top_k"])

reranker = create_bge_cross_encoder_reranker(
    top_n=settings.retrieval.reranking_top_k
)
```

### Configuration

No new settings; uses existing `RetrievalConfig.reranker_normalize_scores` and `RetrievalConfig.reranking_top_k` (ADR-024).

## Testing

- **E2E (Streamlit AppTest)**: programmatically set checkbox/number and `run()`; assert settings mutated and reranker constructed with expected top_n.
- **Unit**: factory reads updated settings and applies normalization flag.

## Consequences

### Positive Outcomes

- Minimal operator control that improves practical reranking results
- Deterministic UI tests; no new API surfaces

### Negative Consequences / Trade-offs

- Only two knobs; deeper tuning (thresholds, device) deferred by design

### Ongoing Maintenance & Considerations

- Consider exposing additional controls only if justified by operator demand

### Dependencies

- Streamlit (existing), sentence-transformers (existing)

## References

- BGE reranker semantics: <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- Streamlit AppTest patterns: <https://github.com/streamlit/docs/tree/main/content/develop/concepts/app-testing>
- Final research plan: agent-logs/2025-09-02/processing/002_semantic_cache_and_reranker_ui_final_plan.md

## Changelog

- **1.0.0 (2025-09-03)**: Initial accepted version.
