# ADR 008: Session Persistence

## Version/Date

v1.0 / July 22, 2025

## Status

Accepted

## Context

Users need to save analysis/chat sessions for continuity across interactions.

## Decision

- Use Streamlit's **session_state** for in-memory state (model configs, history).
- Persist to disk via **pickle** for saving/loading sessions.
- Implement in `app.py` with save/load buttons and warnings for pickle security.

## Rationale

- session_state is native to Streamlit, simple to use.
- Pickle is lightweight for local apps, supports complex objects.
- Security warnings mitigate risks for local use.

## Alternatives Considered

- JSON storage: Limited for complex objects.
- Database (SQLite): Overkill for local app.

## Consequences

- Pros: Simple, effective for local use.
- Cons: Pickle security risks; mitigated by warnings and local context.
