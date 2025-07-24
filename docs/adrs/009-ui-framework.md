# ADR 009: UI Framework Selection

## Version/Date

v1.0 / July 22, 2025

## Status

Accepted

## Context

DocMind AI needs a user-friendly, lightweight UI for document uploads, configs, and results display.

## Decision

- Use **Streamlit** for the frontend, implemented in `app.py`.
- Features: File upload, sidebar configs, expandable previews, chat interface, theming (light/dark/auto).
- Leverage Streamlit's session_state for state management.

## Rationale

- Streamlit is Python-native, simple for rapid prototyping.
- Supports reactive UI, file uploads, and theming out of the box.
- Minimal setup compared to web frameworks.

## Alternatives Considered

- Flask/Dash: More complex for simple apps.
- Gradio: Less flexible for custom layouts.

## Consequences

- Pros: Fast development, intuitive UI.
- Cons: Limited customization; sufficient for current scope.
