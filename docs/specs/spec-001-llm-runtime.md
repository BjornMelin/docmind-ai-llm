---
spec: SPEC-001
title: Multi-provider LLM Runtime with UI Selection and Hardware-Aware Paths
version: 1.1.0
date: 2026-01-17
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - FR-LLM-001: Users SHALL select provider and model at runtime via UI.
  - FR-LLM-002: The app SHALL support llama.cpp, vLLM, Ollama, and LM Studio.
  - FR-LLM-003: Structured outputs SHALL be supported when provider allows.
  - NFR-PERF-001: p50 token throughput ≥ 20 tok/s on mid-GPU.
  - NFR-PORT-001: Windows, macOS, Linux support.
related_adrs: ["ADR-001","ADR-004","ADR-009","ADR-010","ADR-024","ADR-059","ADR-063"]
---


## Objective

Provide a single, definitive LLM runtime with **five providers** selectable in the UI:

- **Ollama** (default)
- **llama.cpp** (local GGUF via library)
- **vLLM** (external server; OpenAI-compatible HTTP)
- **LM Studio** (external server; OpenAI-compatible HTTP)
- **OpenAI-compatible** (generic provider for OpenAI/OpenRouter/xAI/Vercel AI Gateway/LiteLLM Proxy/any compatible endpoint)

Persist selection to settings. Expose model id/path, context, streaming, and safe endpoint configuration. Enable schema-guided outputs when available.

## Architecture Notes

- Use **LlamaIndex** official adapters:
  - `llama_index.llms.llama_cpp.LlamaCPP` for GGUF local models.
  - `llama_index.llms.openai_like.OpenAILike` for **vLLM**/**LM Studio**/**llama.cpp server** (OpenAI-compatible endpoints).
  - `llama_index.llms.openai.OpenAIResponses` for providers that support the OpenAI Responses API (opt-in, OpenAI-compatible base URL).
  - `llama_index.llms.ollama.Ollama` for **Ollama**.
- For Ollama-native `/api/*` capabilities, use the official Ollama SDK helpers defined in `src/config/ollama_client.py` and follow SPEC-043. Ollama-native capabilities include:
  - Structured outputs via the `format` parameter
  - Thinking via the `think` parameter
  - Tool-calling support
  - Logprobs for token probabilities
  - Embed `dimensions` for truncation
  - Optional cloud web tools (web_search/web_fetch)
- Central factory: `src/config/llm_factory.py`.
- UI wiring: `src/pages/04_settings.py` controls provider, model, base URLs, and advanced knobs.
- Respect environment via `src/config/settings.py` and surfacing in UI.

## Libraries and Imports

```python
from llama_index.core import Settings
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAIResponses
from llama_index.llms.ollama import Ollama
from src.config.settings import DocMindSettings
from src.config.llm_factory import build_llm
```

## Integration Points

- `src/config/integrations.py::setup_llamaindex()` to inject Settings.llm and context caps.
- UI: `src/pages/04_settings.py` provider select, model text input, server URL fields.
- Chat page uses `Settings.llm` directly.

## File Operations

### CREATE

- `src/ui/components/provider_badge.py`: small UI chip showing active provider.

### UPDATE

- `src/config/settings.py`: Add fields for all providers (urls, flags), validation, defaults.
- `src/config/llm_factory.py`: Ensure correct backend mapping, timeouts, context_window handling, GPU offload for llama.cpp.
- `src/config/integrations.py`: Do not overwrite pre-set test LLMs. Set `Settings.context_window` and `Settings.num_output`.
- `src/pages/04_settings.py`: Add controls to change provider, model id/path, URLs, and save.

### DELETE

- Remove any duplicate factory helpers under `src/utils/` if present.

## Determinism & Structured Outputs

- When provider is **vLLM** and server supports **xgrammar/guided_json**, enable schema-guided decoding in agent layer (see SPEC-007).
- When using Ollama-native APIs, prefer Ollama’s `format` parameter for structured outputs (SPEC-043) and validate with Pydantic where applicable.

## Performance Budgets

- p50 end-to-first-token ≤ 1.2 s on mid-GPU vLLM.
- p50 streaming ≥ 20 tok/s on mid-GPU, ≥ 8 tok/s CPU.

## Observability

- Log provider, model, base_url at INFO once. No secrets.
- Emit counters: provider_used, streaming_enabled.

## Acceptance Criteria

```gherkin
Feature: LLM provider selection
  Scenario: Switch provider in settings
    Given the app is running
    And I open Settings
    When I select 'vllm' and set model 'Qwen2.5-7B-Instruct'
    And I click Save
    Then Settings.llm SHALL be OpenAILike
    And chat replies stream successfully

  Scenario: OpenAI-compatible cloud endpoint (Responses API)
    Given I select 'openai_compatible'
    And I set base_url to https://api.openai.com/v1
    And I set API mode to 'responses'
    Then Settings.llm SHALL be OpenAIResponses
    And chat replies successfully

  Scenario: Use llama.cpp with GGUF
    Given I select 'llamacpp' and model_path points to a GGUF file
    Then Settings.llm SHALL be LlamaCPP
    And generation SHALL not raise exceptions

  Scenario: LM Studio endpoint
    Given I set base_url to http://localhost:1234/v1
    Then Settings.llm SHALL be OpenAILike with is_chat_model True
```

## Detailed Checklist

- [x] Add UI select for provider with options: ollama, vllm, lmstudio, llamacpp.
- [x] Add UI select for provider option: openai_compatible (generic OpenAI-compatible endpoint).
- [x] Validate base URLs: vLLM may be raw server or OpenAI-compatible; LM Studio requires `/v1`.
- [x] Support opt-in Responses API mode for OpenAI-compatible endpoints.
- [x] LlamaCPP uses `model_kwargs={"n_gpu_layers": -1 if GPU else 0}`.
- [x] Hook `Settings.llm` inside `setup_llamaindex()` only if not already set (allow force rebind).
- [x] Persist settings to `.env` via existing settings save util (minimal updater implemented).

## Git Plan

- Branch: `feat/llm-runtime`
- Commits:
  1. `feat(llm): add provider selection UI and settings fields`
  2. `feat(llm): implement unified llm_factory with 4 providers`
  3. `chore(llm): log provider badge and env validation`
  4. `refactor(ui): wire Settings to chat engine`

## References

- vLLM structured outputs and AMD ROCm.  
- llama.cpp repo features.  
- LM Studio OpenAI API compat; Ollama OpenAI compat.

## Settings Scope & Validation

- The Settings page MUST include: provider selection (Ollama, vLLM, LM Studio, llama.cpp), model ID/path, context window, timeout, and GPU toggle.
- Retrieval/reranking/hybrid toggles MUST NOT appear in Settings; these remain environment‑only.
- URL validation/allowlist:
  - LM Studio endpoints MUST end with `/v1`.
  - Remote endpoints MUST be disabled by default and only allowed via explicit allowlist policy.

## Selector Policy

- Preferred selector: `PydanticSingleSelector`.
- Fallback: `LLMSingleSelector` based on provider/model capability.
- Selector choice impacts routing only (not answer generation) and MUST be covered by unit tests.
