---
ADR: 010
Title: Performance measurement and optimization strategy
Status: Implemented (Amended)
Version: 10.0
Date: 2026-07-13
Supersedes:
Superseded-by:
Related: 004, 011, 024, 030
Tags: performance, kv-cache, fp8, parallel
References:
- [vLLM — Engine Args](https://docs.vllm.ai/en/latest/models/engine_args.html)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
---

## Description

Measure the full retrieval-augmented generation path before optimizing it. Use
LangGraph's native handoff semantics instead of an inert application concurrency
toggle. Treat FP8 key-value cache and extended context as optional settings for
an external vLLM server. Application-level document cache remains separate
(ADR-030).

## Context

Local multi-agent RAG performance depends on the model, server, hardware, context size, retrieval work, and tool workload. Parallel tools can reduce wall-clock time when calls are independent. They can also increase contention. FP8 key-value cache can reduce cache storage on supported vLLM deployments, but the repository has no reproducible LLM benchmark that proves a fixed latency, token, or memory improvement.

## Decision Drivers

- Measure end-to-end behavior in the target environment
- Bound parallel work through the orchestration layer
- Keep optional vLLM tuning in server configuration
- Keep configuration library-first

## Alternatives

- A: Server-default cache format. Broad compatibility, but it may use more memory
- B: Smaller context. Lower memory use, but less room for retrieved evidence
- C: Sequential tools. Predictable resource use, but independent calls cannot overlap
- D: Optional FP8 plus native graph handoffs (Selected). Keeps server controls
  at their real owner without claiming universal gains

### Historical decision framework

The 2025 scores were planning judgments. They are not benchmark results.

| Model / Option | Latency (40%) | VRAM (30%) | Simplicity (20%) | Stability (10%) | Total | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| FP8+parallel (Selected) | 9 | 9 | 8 | 8 | **8.7** | Selected |
| FP16 KV | 6 | 5 | 9 | 9 | 6.7 | Rejected |
| 32K context | 7 | 9 | 9 | 9 | 8.0 | Rejected |

## Decision

Use the compiled supervisor graph's native handoff behavior. Do not expose an
application concurrency setting until a live orchestration boundary can consume
it. When an operator selects an external vLLM server, expose the FP8 cache and
FlashInfer settings as an optional deployment profile. Do not treat either
setting as a performance guarantee.

Measure time to first token, end-to-end latency, prefill and decode throughput, peak host memory, peak accelerator memory, and failure rate. Record the model, backend, hardware, context size, concurrency, and corpus with each result.

## High-Level Architecture

```mermaid
graph TD
  M["Configured LLM endpoint"] --> S["Supervisor"]
  S -->|parallel_tool_calls| T1["Tool A"]
  S --> T2["Tool B"]
  S --> T3["Tool C"]
```

## Related Requirements

### Functional Requirements

- FR‑1: Validate and record external vLLM context, cache, and attention settings
  for operator-managed server launch; inference requests do not forward
  server-only cache or attention controls
- FR‑2: Execute graph handoffs through LangGraph's native routing primitives

### Non-Functional Requirements

- NFR‑1: Publish performance figures only with reproducible environment metadata
- NFR‑2: Avoid host or accelerator out-of-memory failures at the configured workload

### Performance Requirements

- PR‑1: Compare parallel and sequential tool execution on the same workload
- PR‑2: Compare cache formats on the same vLLM version, model, and hardware

### Integration Requirements

- IR‑1: Compatible with supervisor orchestration (ADR‑011)
- IR‑2: Separate from application cache (ADR‑030)

## Design

### Architecture Overview

- The external vLLM process owns cache and attention settings
- LangGraph owns supervisor handoff scheduling

### Implementation Details

Inspect external vLLM launch configuration independently from DocMind:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8_e5m2
```

Notes:

- Prefer LangGraph's native orchestration primitives. Avoid hiding
  `asyncio.gather(...)` inside tool implementations.
- Keep tools sync-callable (`BaseTool.invoke(...)`) unless the full orchestration is
  migrated end-to-end to async. This preserves compatibility with the current
  `agent.invoke(...)` execution model (ADR-011).

### Configuration

```env
DOCMIND_LLM_REQUEST__CONTEXT_WINDOW=131072
DOCMIND_VLLM_BASE_URL=http://localhost:8000/v1
```

## Testing

```bash
uv run pytest tests/unit/config/test_integrations.py \
  tests/unit/config/test_llm_factory.py tests/unit/agents/test_coordinator.py -q
```

## Consequences

### Positive Outcomes

- Operators configure vLLM deployment values on the external server; the application does not install or launch vLLM or FlashInfer
- Handoff scheduling has one owner in the compiled graph
- Performance claims require reproducible evidence

### Negative Consequences / Trade-offs

- FP8 requires backend and hardware compatibility
- Native parallel handoffs can increase contention or provider load

### Ongoing Maintenance & Considerations

- Track vLLM and FlashInfer releases when that optional profile is in use
- Re-run the same benchmark after model, dependency, driver, or hardware changes

### Dependencies

- Server-side (optional): vLLM with a compatible attention backend and accelerator runtime.
- Python app: no vLLM or FlashInfer dependencies. The provider factory uses
  OpenAI-compatible HTTP through LlamaIndex and LangChain; those integrations
  own the transitive OpenAI SDK dependency.

## Changelog

- 10.0 (2026-07-13): Remove the no-op parallel-execution setting and keep handoff scheduling at the LangGraph boundary
- 9.2 (2026-07-10): Replace unverified performance guarantees with environment-specific measurements; clarify that vLLM is an optional external server and parallel-tool gains require comparison
- 9.0 (2025-08-26): Separated from application cache (ADR‑030); FP8 + parallel tools focus
- 8.1 (2025-08-20): Verified supervisor parameters for parallelism
- 8.0 (2025-08-19): Record the FP8, extended-context, and parallel-tool planning assumptions; quantitative gains were not backed by a reproducible repository benchmark
- 7.0 (2025-08-19): INT8 KV cache attempt for 262K context
- 6.0 (2025-08-18): Hardware upgrade for 128K via YaRN
