---
spec: SPEC-043
title: Ollama Native SDK Integration and Optional Capabilities
version: 1.0.0
date: 2026-01-15
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-LLM-002: The app SHALL support Ollama as a provider.
  - FR-LLM-003: Structured outputs SHALL be supported when provider allows.
  - NFR-SEC-001: Remote endpoints SHALL be disabled by default.
  - NFR-OBS-001: Secrets SHALL NOT be logged.
related_adrs: ["ADR-004","ADR-024","ADR-047","ADR-059"]
---

## Objective

Define DocMind’s canonical integration for Ollama-native APIs and capabilities using the official `ollama` Python SDK, including explicit security/egress policy and explicit streaming semantics.

This spec complements (does not replace) the LlamaIndex-based runtime selection described in SPEC-001.

## Scope

In scope:

- Ollama host/auth/timeout configuration and security gating.
- Ollama-native capability usage via `/api/chat`, `/api/generate`, `/api/embed`.
- Optional capabilities (feature-flagged): logprobs, embed dimensions, thinking, structured outputs, tool calling, cloud web tools.

Out of scope:

- Changing the primary chat UI runtime away from LlamaIndex adapters (SPEC-001).
- Enabling network/remote endpoints by default.
- Adding new agent orchestration abstractions.

## Canonical Code Locations

- Client/config entrypoint: `src/config/ollama_client.py`
- Settings fields: `src/config/settings.py` (see `ollama_*` and `security.*`)
- Optional agent tool loop example (feature-flagged): `src/agents/tools/ollama_web_tools.py`

## Configuration

Canonical settings (Pydantic Settings, prefix `DOCMIND_`):

- `DOCMIND_OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `DOCMIND_OLLAMA_API_KEY` (optional; required for Ollama Cloud and web tools)
- `DOCMIND_OLLAMA_ENABLE_WEB_SEARCH` (default: `false`)
- `DOCMIND_OLLAMA_EMBED_DIMENSIONS` (optional int)
- `DOCMIND_OLLAMA_ENABLE_LOGPROBS` (default: `false`)
- `DOCMIND_OLLAMA_TOP_LOGPROBS` (default: `0`)
- `DOCMIND_LLM_REQUEST_TIMEOUT_SECONDS` (default: `120`)
- `DOCMIND_LLM_STREAMING_ENABLED` (default: `true`)

Remote endpoint policy (canonical):

- `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false` by default.
- Enabling Ollama Cloud features (web tools) requires `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true`.
- For defense-in-depth, also include `https://ollama.com` in `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST` (note: allowlist enforcement applies when `ALLOW_REMOTE_ENDPOINTS=false`).

### Config surface decision

DocMind uses `DOCMIND_*` for all app-level configuration (including Ollama) to avoid conflicts with provider/daemon env vars such as `OLLAMA_*`. No `OLLAMA_*` aliases are supported unless a future ADR explicitly overrides this.

## Required Behavioral Guarantees

### Explicit streaming semantics

All Ollama chat/generate calls initiated by DocMind MUST pass `stream=` explicitly (derived from `settings.llm_streaming_enabled`). Do not rely on server defaults.

### Optional fields must be treated as optional

Callers MUST NOT assume these response fields exist unless enabled/requested and supported by the model/server:

- `logprobs` / token alternatives
- thinking traces (model-dependent)
- `usage` fields (present for most responses; still treat as optional)

## Capability Support

### Logprobs (ollama-python >= 0.6.1)

- When enabled, requests may include:
  - `logprobs: true`
  - `top_logprobs: K` (0–20)
- Default posture is **off** unless explicitly enabled.
- Response parsing must tolerate missing `logprobs`.

### Embed dimensions

- `/api/embed` optionally accepts `dimensions` to truncate embeddings for supported models.
- When configured, DocMind passes `dimensions` and surfaces a clear error if the server/model rejects it.

### Structured outputs

DocMind should prefer Ollama-native structured outputs when using Ollama-native `/api/*`:

- `format="json"` for “JSON mode”
- `format=<json-schema-dict>` for schema-constrained outputs

When a Pydantic model is available, DocMind should pass `format=<Model>.model_json_schema()` and validate output with Pydantic on receipt.

### Thinking

DocMind may pass `think` when a model supports it:

- boolean on/off, or a supported level string (e.g., `"low" | "medium" | "high"` depending on model)

Thinking traces MUST be treated as optional metadata and MUST NOT be logged verbatim.

### Tool-calling + cloud web tools

When `ollama_enable_web_search` is enabled and the endpoint policy allows `https://ollama.com`:

- Tools MAY include Ollama Cloud web tools (web search/fetch) via the SDK and tool-calling capability.
- If disabled, the app MUST NOT inject these tools or call cloud endpoints.

## Testing Guidance

- Unit tests should mock Ollama SDK calls (no network by default).
- Any tests that hit real Ollama Cloud endpoints MUST be marked with the project’s `requires_network` marker and MUST require an API key.

## Acceptance Criteria

- All Ollama SDK usage flows through `src/config/ollama_client.py`.
- Remote endpoints remain blocked by default; enabling web tools requires explicit config + allowlist.
- Streaming vs non-streaming behavior is explicitly controlled by settings.
- Docs for enabling logprobs, embed dimensions, thinking, structured outputs, and web tools are consistent across specs/ADRs/prompts.
