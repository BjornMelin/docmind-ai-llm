---
ADR: 060
Title: Dotenv Override Modes and Allowlisted Env Mask/Overlay
Status: Implemented
Version: 1.0
Date: 2026-01-15
Supersedes:
Superseded-by:
Related: 024, 041, 050
Tags: configuration, dotenv, pydantic-settings, security
References:
  - https://docs.pydantic.dev/latest/concepts/pydantic_settings/
  - https://saurabh-kumar.com/python-dotenv/reference/
  - https://12factor.net/config
---

## Description

Provide an explicit, safe way for developers/users to avoid accidental use of
global machine environment variables (e.g., `OPENAI_API_KEY`) while keeping the
default 12-factor precedence (exported env > `.env`) unchanged.

This ADR introduces:

- An opt-in mode where repo `.env` can override exported env vars **for
  `DocMindSettings` only**, with security guardrails.
- An allowlist-based mechanism to mask and/or overlay selected `os.environ` keys
  for dependencies that read non-`DOCMIND_*` variables.

## Context

DocMind is local-first and uses Pydantic Settings v2 as the source of truth for
configuration. By default, environment variables have higher priority than
dotenv values, which aligns with production best practices and CI behavior.

However, local developer machines commonly have globally exported variables
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). Even if DocMind does not use
those variables directly, dependencies may, leading to surprising behavior.

## Decision Drivers

- Keep default precedence and production safety intact.
- Provide explicit opt-in overrides for local development.
- Avoid new config layers; keep behavior close to Pydantic Settings and
  python-dotenv primitives.
- Prevent accidental weakening of the offline-first security posture.

## Alternatives

1. Reorder Pydantic Settings sources globally so dotenv always wins.
2. Load `.env` into `os.environ` with `override=True` semantics (broad override).
3. Hybrid approach:
   - Optional dotenv-first behavior for `DocMindSettings` only, plus
   - Allowlisted mask/overlay for selected `os.environ` keys.
4. “Sandbox env”: clear most of `os.environ` and rebuild from `.env`.

### Decision Framework (Tier 2; ≥ 9.0 required)

Weights:

- Complexity & Maintenance (40%)
- Performance (30%)
- Ecosystem Alignment (30%)

|Option|Complexity (40%)|Perf (30%)|Alignment (30%)|Total|
|-----:|--------------:|--------:|-------------:|----:|
|1|8.5|10.0|9.0|9.05|
|2|6.5|9.5|6.5|7.55|
|**3**|**8.8**|**9.8**|**9.4**|**9.35**|
|4|3.0|8.0|3.0|4.70|

Selected: **Option 3**.

## Decision

### A) Dotenv-first (DocMind settings only; optional)

Add an opt-in mode that changes precedence for `DocMindSettings` via
`settings_customise_sources`:

- `DOCMIND_CONFIG__DOTENV_PRIORITY=env_first|dotenv_first` (default: `env_first`)

Guardrail:

- Even in `dotenv_first`, `settings.security.*` remains env-first so a local
  `.env` cannot accidentally weaken endpoint allowlisting/offline-first posture.

### B) Allowlisted env masking and overlay (optional)

To influence dependencies that read `os.environ` directly:

- `DOCMIND_CONFIG__ENV_MASK_KEYS=OPENAI_API_KEY,ANTHROPIC_API_KEY`
  - Removes the listed keys from `os.environ` at startup.
- `DOCMIND_CONFIG__ENV_OVERLAY=OPENAI_API_KEY:openai.api_key`
  - After settings validation, sets `os.environ[OPENAI_API_KEY]` from
    `settings.openai.api_key`.

Constraints:

- Mask/overlay are allowlist-only (no “override everything” mode).
- Values are never logged; invalid overlay mappings fail fast with clear errors.

## Consequences

### Positive Outcomes

- Default precedence remains production-safe.
- Local development can be made deterministic even on machines with global keys.
- Third-party libraries can be constrained without adding a new config layer.

### Trade-offs

- Opt-in complexity: additional knobs exist, but are strictly gated and
  documented as “local-dev only”.
