---
ADR: 041
Title: Settings UI Hardening: Pre-validation, Safe Badges, and .env Persistence
Status: Implemented
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 024, 013, 016, 031
Tags: streamlit, settings, configuration, security
References:
- https://docs.streamlit.io/develop/api-reference/text/st.badge
- https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest
- https://pypi.org/project/python-dotenv/
---

## Description

Harden the Streamlit Settings page to prevent invalid persistent configuration, remove unsafe HTML rendering, and persist updates to `.env` safely using `python-dotenv`.

## Context

Current Settings behavior allows:

- persisting invalid configuration into `.env` without pre-validation
- applying runtime changes even when the underlying LlamaIndex `Settings.llm` is not successfully bound
- rendering a provider badge using `st.markdown(..., unsafe_allow_html=True)` which is an avoidable XSS sink

This creates a “startup bricking” risk and violates Streamlit Master Architect security-by-default guidance.

## Decision Drivers

- Prevent invalid persisted config from breaking subsequent restarts
- Remove unsafe HTML rendering in UI
- Keep offline-first + endpoint allowlist posture intact
- Prefer maintained libraries over custom `.env` writers

## Alternatives

- A: Keep current behavior — risk of invalid `.env`, unsafe HTML sink
- B: Use Streamlit forms + Pydantic pre-validation + Streamlit-native badges (Selected)
- C: Add a bespoke settings persistence layer (SQLite/JSON) — unnecessary complexity

### Decision Framework

| Model / Option                                 | Leverage (35%) | Value (25%) | Risk Reduction (25%) | Maint (15%) |    Total | Decision    |
| ---------------------------------------------- | -------------: | ----------: | -------------------: | ----------: | -------: | ----------- |
| **B: Pre-validate + st.badge + python-dotenv** |              9 |           9 |                   10 |           9 | **9.25** | ✅ Selected |
| A: Status quo                                  |              2 |           4 |                    2 |           7 |     3.15 | Rejected    |
| C: Custom persistence                          |              5 |           6 |                    7 |           3 |     5.55 | Rejected    |

## Decision

We will:

1. Replace unsafe HTML badges with Streamlit-native UI (`st.badge`, `st.caption`, `st.columns`) and never use `unsafe_allow_html=True` for provider status UI.

2. Implement "proposed settings" validation:
   - Build a candidate env/settings mapping from widget values.
   - Validate via `DocMindSettings.model_validate(…)` (or equivalent safe construction) **before** mutating the global `settings` singleton or writing `.env`.
   - Disable Apply/Save actions when validation fails; show errors to the user.

3. Persist `.env` updates using `python-dotenv` `set_key` / `unset_key`:
   - Use `quote_mode="auto"` to preserve readability.
   - Use `unset_key` for empty values where appropriate.
   - Note: `.env` is intentionally lower precedence than already-exported environment variables (Pydantic Settings default). If an env var like `DOCMIND_LMSTUDIO_BASE_URL` is exported in the shell/process, it will override the `.env` value on the next load.

## High-Level Architecture

```mermaid
flowchart TD
  UI[Settings Page Widgets] --> C[Candidate env mapping]
  C --> V[Pydantic pre-validation]
  V -->|valid| A[Apply runtime (initialize_integrations)]
  V -->|valid| P[Persist .env (python-dotenv)]
  V -->|invalid| E[UI errors; disable actions]
```

## Security & Privacy

- Settings must not allow remote endpoints unless `security.allow_remote_endpoints=true` and host is allowlisted.
- UI must not introduce new HTML/JS sinks.
- `.env` persistence must not log secrets.

## Testing

- Unit tests for helpers (validation + `.env` persistence):
  - URL validation helpers and allowlist checks
  - `.env` persistence round-trip (set/unset, missing file behavior)
- AppTest-based integration tests for Settings page:
  - invalid URL blocks Save/Apply and renders errors
  - valid change persists to `.env` and runtime apply binds LlamaIndex Settings.llm
  - provider badge renders without using `unsafe_allow_html=True`
  - global settings are reset/mutated in-place between tests (avoid rebinding `src.config.settings.settings` to keep the suite order-independent)
- Cross-reference SPEC-022 for the detailed test matrix and separation of unit vs. integration coverage.

## Consequences

### Positive Outcomes

- Settings cannot persist invalid configuration, preventing “bricked” restarts.
- Removes XSS sink from provider badge.
- `.env` updates become predictable and less error-prone.

### Trade-offs

- Slightly more UI code for validation and error rendering.
- We intentionally avoid `st.form` for Settings inputs so validation and action disabling stay reactive; forms can be revisited if validation-on-submit is acceptable.

## Changelog

- 1.0 (2026-01-09): Proposed for v1 release hardening.
- 1.1 (2026-01-10): Implemented in code + tests (Settings pre-validation, safe badges, dotenv persistence).
