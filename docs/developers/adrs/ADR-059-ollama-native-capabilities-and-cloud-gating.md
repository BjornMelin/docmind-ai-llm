---
ADR: 059
Title: Ollama Native Capabilities and Cloud Gating
Status: Accepted
Version: 1.0
Date: 2026-01-15
Supersedes:
Superseded-by:
Related: ADR-004, ADR-024, ADR-047, ADR-050, SPEC-001, SPEC-011, SPEC-043
Tags: llm, ollama, security, tooling, structured-outputs
References:
  - https://docs.ollama.com/api/introduction
  - https://docs.ollama.com/api/chat
  - https://docs.ollama.com/api/generate
  - https://docs.ollama.com/api/embed
  - https://docs.ollama.com/api/streaming
  - https://docs.ollama.com/api/usage
  - https://docs.ollama.com/capabilities/tools
  - https://docs.ollama.com/capabilities/thinking
  - https://docs.ollama.com/capabilities/structured-outputs
  - https://docs.ollama.com/capabilities/web-search
  - https://docs.ollama.com/cloud
  - https://github.com/ollama/ollama-python
  - https://github.com/ollama/ollama-python/releases
---

## Description

Adopt the official `ollama` Python SDK as the canonical integration for Ollama-native `/api/*` capabilities (structured outputs, thinking, tool calling, logprobs, embed dimensions, and cloud web tools) while enforcing DocMind’s offline-first remote endpoint policy.

## Context

DocMind is local-first by default (ADR-004) and blocks remote endpoints unless explicitly enabled/allowlisted (SPEC-011, `security.allow_remote_endpoints`). Historically, “Ollama support” in the runtime focused on provider selection via LlamaIndex’s adapters (SPEC-001). This covers baseline chat/generation, but does not provide a single, app-owned surface for rapidly evolving Ollama-native capabilities (e.g., `format`, `think`, `logprobs`, `/api/embed` `dimensions`, and cloud `web_search`/`web_fetch`).

We need an integration that:

- Preserves current runtime behavior (especially streaming vs non-streaming) explicitly.
- Makes new Ollama capabilities easy to adopt without spreading ad-hoc calls/config across the codebase.
- Keeps remote egress opt-in and auditable (no implicit network use).

## Decision Drivers

- **Local-first security posture**: remote endpoints disabled by default; secrets never logged.
- **Minimal surface area**: one canonical configuration point; avoid new abstractions.
- **Feature parity with upstream**: structured outputs, tool calling, thinking, web tools, logprobs, embed dimensions.
- **Operational clarity**: explicit streaming behavior; stable error handling via SDK exceptions.

## Alternatives

- A: Use only LlamaIndex `llama_index.llms.ollama.Ollama` for all Ollama use cases.
  - Pros: fewer direct deps; single abstraction.
  - Cons: slower adoption of Ollama-native capabilities; unclear mapping of newer `/api/*` features; harder to gate cloud web tools precisely.
- B: Raw HTTP calls to `http://localhost:11434/api/...` and `https://ollama.com/api/...`.
  - Pros: maximal control; no SDK dependency.
  - Cons: duplicated request/stream parsing; higher maintenance; easier to accidentally bypass security policy; inconsistent error handling.
- C: Official `ollama` Python SDK as canonical, with a small central client module and feature flags.
  - Pros: matches upstream semantics; minimal app-owned glue; consistent exceptions; easiest to keep current with new capabilities.
  - Cons: adds a direct dependency; must ensure explicit streaming semantics are preserved.

### Decision Framework

Weights: **Solution leverage (35%)**, **Application value (30%)**, **Maintenance (25%)**, **Adaptability (10%)**. Scores are 1–10.

| Option | Leverage (35%) | Value (30%) | Maintenance (25%) | Adaptability (10%) | Total | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **C: Official SDK + central module + flags** | 9.5 | 9.2 | 9.0 | 9.0 | **9.24** | ✅ **Selected** |
| A: LlamaIndex wrapper only | 7.5 | 7.8 | 8.2 | 6.5 | 7.73 | Rejected |
| B: Raw HTTP everywhere | 6.0 | 7.0 | 6.0 | 7.0 | 6.40 | Rejected |

## Decision

We will use the official `ollama` Python SDK (ollama-python) as DocMind’s canonical integration for Ollama-native `/api/*` features and (optionally) Ollama Cloud web tools.

Implementation is centralized in `src/config/ollama_client.py` and MUST:

- Resolve host/auth/timeouts in one place, driven by `DocMindSettings` with opt-in env overrides.
- Use DocMind’s `DOCMIND_*` config surface exclusively; do not introduce `OLLAMA_*` aliases unless a future ADR explicitly overrides this.
- Enforce `security.allow_remote_endpoints` / allowlist checks before allowing non-local hosts.
- Require an API key for cloud web tools and never log that key.
- Preserve streaming semantics by always passing `stream=` explicitly from settings (no reliance on server defaults).

## Consequences

### Positive Outcomes

- Feature adoption stays aligned with upstream capability docs (`format`, `think`, tool calling, logprobs, embed dimensions).
- Cloud web tools are explicit, gated, and testable without enabling network by default.
- Less duplicated HTTP/stream handling logic; consistent SDK error behavior.

### Trade-offs

- Direct dependency on `ollama` SDK means we must monitor upstream releases for behavior changes.
- Some capability differences exist across models; callers must treat optional fields (`logprobs`, thinking traces) as absent unless requested and supported.
