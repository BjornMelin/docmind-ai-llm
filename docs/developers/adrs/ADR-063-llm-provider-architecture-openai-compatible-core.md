---
ADR: 063
Title: LLM Provider Architecture (OpenAI-Compatible Core + Open Responses Alignment)
Status: Accepted
Version: 1.0
Date: 2026-01-17
Supersedes:
Superseded-by:
Related: ADR-004, ADR-024, ADR-059, ADR-062
Tags: architecture, configuration, llm, providers, streaming
References:
  - [Open Responses specification](https://www.openresponses.org/specification)
  - [Open Responses reference](https://www.openresponses.org/reference)
  - [OpenAI Python SDK (Responses API)](https://github.com/openai/openai-python)
  - [LiteLLM proxy: Docker quick start](https://docs.litellm.ai/docs/proxy/docker_quick_start)
  - [LiteLLM: Responses API (SDK + Proxy)](https://litellm.vercel.app/docs/providers/openai/responses_api)
  - [OpenRouter API overview](https://openrouter.ai/docs/api/reference/overview)
  - [OpenRouter Responses API (beta)](https://openrouter.ai/docs/api-reference/responses-api/overview)
  - [OpenRouter Responses: tool calling](https://openrouter.ai/docs/api-reference/responses-api/tool-calling)
  - [xAI REST API reference](https://docs.x.ai/docs/api-reference)
  - [Vercel AI Gateway (OpenAI compatibility)](https://vercel.com/docs/ai-gateway/openai-compat)
  - [Vercel AI Gateway (OpenResponses)](https://vercel.com/docs/ai-gateway/openresponses)
  - [Ollama OpenAI compatibility](https://docs.ollama.com/openai)
  - [LM Studio OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat)
---

## Description

Adopt an **OpenAI-compatible “core provider interface”** (base URL + headers + model) for both local and cloud LLMs, with an opt-in **Responses API** mode aligned with the Open Responses ecosystem.

## Context

DocMind is **offline-first by default** and must not silently make outbound calls. At the same time, users expect to bring their own provider and model:

- Local servers (Ollama, LM Studio, vLLM, llama.cpp server) commonly expose OpenAI-compatible endpoints.
- Cloud providers and gateways (OpenAI, OpenRouter, xAI, Vercel AI Gateway) are OpenAI-compatible (with small quirks like attribution headers).
- Non-OpenAI-native providers (Anthropic/Gemini/etc.) are frequently accessed via an OpenAI-compatible gateway (OpenRouter, LiteLLM Proxy, Vercel AI Gateway).

Open Responses defines a shared schema and streaming event taxonomy around the **OpenAI Responses API** (`POST /v1/responses`). We want the benefits (more structured semantics, tool/event streaming) while keeping the provider layer small and leveraging existing SDKs we already depend on.

## Decision Drivers

- **Offline-first + security policy**: no implicit egress; strict endpoint validation/allowlist must remain the guardrail.
- **User-first interoperability**: “paste base URL + key” should work for most providers.
- **Maintenance**: minimize duplicated provider logic and reduce version churn risk.
- **Streaming/tool calling correctness**: preserve semantics across providers.
- **Runtime/tooling coherence**: keep provider behavior consistent under the single Python 3.13 baseline (ADR-064).

## Alternatives

- **A: OpenAI-compatible-first core (selected)** — one provider interface across local + cloud; support Responses API where available; use gateways (OpenRouter/LiteLLM Proxy/etc.) for non-native providers.
  - Pros: smallest surface area; best alignment with Open Responses; easiest UX (“works with anything OpenAI-compatible”); least dependency churn.
  - Cons: some providers require a gateway/proxy for best support; Responses API not universally implemented.
- **B: Many native SDKs** — add first-class Anthropic/Gemini/HF/etc. SDKs and keep OpenAI-compatible as just one path.
  - Pros: best feature parity per provider; fewer “edge” incompatibilities.
  - Cons: high maintenance; more auth/config surfaces; more dependency conflicts; harder to keep security posture consistent.
- **C: LiteLLM everywhere (in-process SDK)** — call all providers via LiteLLM’s Python SDK.
  - Pros: broad provider coverage without extra infrastructure; one interface.
  - Cons: increases dependency surface area and churn risk; still needs careful endpoint/egress controls; not all providers map perfectly to identical semantics.

### Decision Framework

Weights match the project “upgrade” rubric:

| Option | Complexity & Maintenance (40%) | Performance & Scale (30%) | Ecosystem Alignment (30%) | Total |
| --- | ---: | ---: | ---: | ---: |
| **A: OpenAI-compatible-first core** | 9.5 | 8.5 | 9.5 | **9.2 ✅** |
| B: Many native SDKs | 6.5 | 9.0 | 7.0 | 7.4 |
| C: LiteLLM everywhere (in-process SDK) | 7.5 | 8.5 | 9.0 | 8.3 |

## Decision

We will standardize the provider layer on an **OpenAI-compatible core interface**:

- **One base URL** (`DOCMIND_OPENAI__BASE_URL`) + **optional headers** (`DOCMIND_OPENAI__DEFAULT_HEADERS`) + **API key**.
- Default to **Chat Completions** for broad compatibility, with an explicit opt-in to **Responses API** mode (`DOCMIND_OPENAI__API_MODE=responses`) when a provider supports it.
- Keep **offline-first endpoint validation** as the hard boundary: loopback-only by default, allowlisted remote endpoints when explicitly enabled.
- Prefer **gateways/proxies** (OpenRouter, LiteLLM Proxy, Vercel AI Gateway) for providers that are not natively OpenAI-compatible.

We do **not** adopt the upstream `openresponses` Python package at this time. DocMind aligns with Open Responses semantics by using the OpenAI SDK v2 Responses event model (via LlamaIndex/LangChain) against OpenAI-compatible endpoints. If adopting `openresponses` provides concrete integration value (e.g., compliance tooling or interop adapters we can’t get otherwise), we can revisit under the Python 3.13-only baseline (ADR-064).

## High-Level Architecture

- **UI/Env settings** → normalize/validate endpoint + headers
- **LLM factory** → selects:
  - Chat Completions (OpenAI-like) for compatibility
  - Responses API for Open Responses-aligned semantics where supported
- **Agents/RAG** consume a single LLM interface; provider differences are encapsulated at the boundary.

## Related Requirements

### Functional Requirements

- Users can configure **any OpenAI-compatible** provider with base URL + API key + optional headers.
- Users can opt into **Responses API** mode when supported.

### Non-Functional Requirements

- **Security**: no implicit egress; validated and allowlisted endpoints only.
- **Maintainability**: avoid duplicated provider code paths; keep provider presets as data where possible.

## Consequences

### Positive Outcomes

- “Works with most providers” becomes the default UX: local servers and common gateways work by just setting a base URL.
- Open Responses alignment is achievable without constraining the runtime to Python 3.13-only.

### Negative Consequences / Trade-offs

- Some providers require a gateway (OpenRouter/LiteLLM Proxy) to be used via OpenAI-compatible semantics.
- Responses API features may vary by provider and must remain opt-in and well-tested.
