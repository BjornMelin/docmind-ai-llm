# ADR-009: UI Framework

## Title

User Interface Framework Selection

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Simple/local UI for uploads/analysis/chat (Streamlit fast/low-maintenance, supports async/fragments).

## Related Requirements

- Responsive (progress/errors for async processing).
- Toggles (AppSettings for gpu/chunk_size).

## Alternatives

- Flask: More boilerplate.
- Gradio: ML-focused, less for data apps.

## Decision

Streamlit (pinned 1.47.1) for UI, with fragments for reactivity, st.status/progress/error for loading.

## Related Decisions

- ADR-001 (UI orchestrates pipeline/agents).

## Design

- **Components**: In app.py: st.file_uploader for upload, st.button for analyze, st.chat_input for chat, st.markdown for results/sources/multimodal images.
- **Integration**: Async calls (asyncio.run(upload_section())); Toggles: st.checkbox("GPU", value=AppSettings.gpu_acceleration); Update AppSettings on change. Progress: st.progress for indexing steps.
- **Implementation Notes**: Handle errors (st.error(e)); Display multimodal (st.image if "image" in metadata).
- **Testing**: tests/test_app.py: def test_ui_toggle(): app.checkbox.set(True); app.run(); assert AppSettings.gpu_acceleration == True; def test_progress_error(): simulate error; assert "Error" in app.error.value.

## Consequences

+ Quick/responsive (fragments/async).
- User-friendly (toggles for configs).

- Limited advanced UI (but sufficient for local RAG app).
- Future: Add pages for multi-view if needed.

**Changelog:**  

- 2.0 (July 25, 2025): Added async progress/errors/display; Toggles via AppSettings; Enhanced testing for dev.
