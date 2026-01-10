---
spec: SPEC-040
title: Agent Deadline Propagation + Router Injection (Cooperative Cancellation)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - FR-014: LangGraph-supervised multi-agent flow.
  - FR-029: Agent deadline propagation/cooperative cancellation.
  - NFR-PERF-001: Chat latency budgets remain within targets.
  - NFR-REL-002: Fail-open behavior on timeouts (no corruption).
related_adrs: ["ADR-056", "ADR-011", "ADR-024", "ADR-013"]
---

## Goals

1. Make `settings.agents.enable_deadline_propagation` effective:
   - seed an absolute deadline in agent state
   - align per-call timeouts (LLM/Qdrant) to the decision timeout budget
   - prevent runaway retries when budget is exhausted
2. Make `settings.agents.enable_router_injection` effective:
   - prefer an injected `router_engine` when provided by the UI (session snapshot)
   - fail-open to explicit vector/hybrid retrieval when absent
3. Restore a consistent retrieval tool contract:
   - retrieval outputs must include **documents/sources** suitable for synthesis/validation and UI.

## Non-goals

- Full async cancellation refactor (`astream` + task cancellation) for Streamlit.
- Introducing new retrieval backends or modifying Qdrant schema.
- Logging raw prompts or document text in telemetry for debugging.

## Technical Design

### State: Deadline Fields

Extend `src/agents/models.py::MultiAgentState` with:

- `deadline_ts: float | None` — absolute timestamp in `time.monotonic()` seconds
- `cancel_reason: str | None` — e.g. `deadline_exceeded`

### Coordinator: Budget Seeding

In `src/agents/coordinator.py::process_query`:

- If `settings.agents.enable_deadline_propagation` is true:
  - compute `deadline_ts = time.monotonic() + settings.agents.decision_timeout`
  - seed it into the initial state

In `_run_agent_workflow`:

- Keep the existing “timeout check between yielded states”
- When the deadline is exceeded:
  - set `timed_out=True` + `deadline_s=<decision_timeout>` on the final state dict
  - exit the loop and trigger coordinator fallback policy (existing behavior)

### Timeout Alignment (Per-call caps)

When deadline propagation is enabled, ensure that **individual call timeouts cannot exceed the overall budget**:

- In `src/config/langchain_factory.py::build_chat_model`:
  - set `timeout = min(cfg.llm_request_timeout_seconds, cfg.agents.decision_timeout)`
- In `src/config/llm_factory.py::build_llm`:
  - similarly cap `timeout_s` and `request_timeout` values
- In Qdrant client construction (where applicable):
  - cap request timeout using `min(settings.database.qdrant_timeout, settings.agents.decision_timeout)`

This does not provide perfect cancellation, but prevents the worst case where an individual call exceeds the supervisor budget.

### Retrieval Tool Contract + Router Injection

Fix default tool wiring:

- In `src/agents/registry/tool_registry.py::DefaultToolRegistry.get_retrieval_tools`:
  - return `retrieve_documents` (from `src/agents/tools/retrieval.py`) as the retrieval tool for the supervisor graph.

Enhance `retrieve_documents` to support router injection:

- If `settings.agents.enable_router_injection` is true and `router_engine` is present in `InjectedState`:
  - call `router_engine.query(query)` and extract:
    - response text
    - `source_nodes` → normalized `documents` list with `{content/text, score, metadata}`
  - return a JSON payload matching the existing retrieval tool shape:
    - `documents`, `strategy_used`, `processing_time_ms`, etc.
- Otherwise:
  - use the existing explicit hybrid/vector/GraphRAG tool paths.

### Observability

Emit JSONL events:

- `agent_deadline_exceeded` with `{decision_timeout_s, elapsed_s}`
- `router_injection_used` with `{used: bool, reason}`

Never include raw prompt/doc text.

## Testing Strategy

### Unit

- `tests/unit/agents/test_tool_registry_retrieval_tool.py`:
  - registry returns `retrieve_documents` (not `router_tool`) for retrieval agent.
- `tests/unit/agents/test_deadline_seeded.py`:
  - deadline_ts exists in initial state when enabled.
- `tests/unit/config/test_timeout_caps.py`:
  - build_chat_model/build_llm timeouts are capped when enable_deadline_propagation is true.
- `tests/unit/agents/test_router_injection_path.py`:
  - retrieve_documents uses injected router_engine when enabled and present.

### Integration

- `tests/integration/test_agent_timeout_behavior.py`:
  - configure a small decision_timeout and a slow mocked call, assert coordinator returns timeout fallback without crashing.

All tests must be deterministic and offline.

## Rollout / Migration

- Default behavior unchanged unless flags are enabled.
- Safe to ship incrementally; fail-open on errors.

## RTM Updates

Update `docs/specs/traceability.md`:

- Add `FR-029` row (and update `FR-014` row if needed) mapping to code/tests above.
