---
ADR: 066
Title: LlamaIndex Workflows Evaluation Without Runtime Replacement
Status: Accepted
Version: 1.0
Date: 2026-05-01
Supersedes:
Superseded-by:
Related: 001, 003, 011, 024, 035, 056, 058, 063
Tags: orchestration, agents, langgraph, llamaindex, workflows
References:
- [Issue #86](https://github.com/BjornMelin/docmind-ai-llm/issues/86)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Durable Execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- [LlamaIndex Workflows](https://developers.llamaindex.ai/python/llamaagents/workflows/)
- [LlamaIndex Multi-Agent Patterns](https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/)
- [LlamaIndex Workflow Deployment](https://developers.llamaindex.ai/python/llamaagents/workflows/deployment/)
---

## Description

Evaluate modern LlamaIndex Workflows and LlamaAgents options without replacing
DocMind's default LangGraph v1 supervisor runtime.

## Context

Issue #86 asked whether DocMind should adopt modern LlamaIndex Workflows or the
newer LlamaAgents ecosystem before any agent runtime replacement. DocMind already
uses a graph-native LangGraph supervisor for agent coordination, LangChain v1
`create_agent` for role agents, and LlamaIndex for retrieval/indexing boundaries.

The current runtime has production behavior tied to LangGraph semantics:

- `src/agents/coordinator.py` compiles the supervisor graph with a checkpointer
  and store, streams graph values, enforces coordinator-level deadlines, and owns
  semantic cache policy.
- `src/agents/supervisor_graph.py` owns the local LangGraph v1 `StateGraph`
  supervisor, handoff tools, forward-to-user behavior, and handoff-back messages.
- `src/pages/01_chat.py` wires the runtime to a SQLite LangGraph checkpointer.
- `src/persistence/memory_store.py` provides the LangGraph `BaseStore`
  implementation for local-first memory and optional semantic search.

Modern LlamaIndex Workflows are active and event-driven. They support branching,
parallelism, human-in-the-loop events, context serialization, observability, and
service deployment options. They are not, by default, the same architecture as
DocMind's in-process checkpointed LangGraph supervisor.

## Package And Ecosystem Facts

Repository snapshot facts were verified from `uv.lock` and `pyproject.toml`.
Local installed source on 2026-04-30 was used only as a cross-check, not as the
authoritative source for this ADR.

- The repo snapshot pins `llama-index==0.14.21` and `llama-index-core==0.14.21`.
- The repo snapshot already contains `llama-index-workflows==2.20.0`
  transitively through `llama-index-core==0.14.21`.
- The repo snapshot pins `langgraph==1.1.10`,
  `langgraph-checkpoint==4.0.3`, and
  `langgraph-checkpoint-sqlite==3.0.3`.
- The repo snapshot pins `langchain==1.2.16` and `langchain-core==1.3.2`.

PyPI metadata and the local install cross-check confirmed the broader ecosystem
posture at the time of review:

- `llama-agents-server==0.5.0`, Python `>=3.10`
- `llama-agents-client==0.3.7`, Python `>=3.10`
- `llamactl==0.9.1`, Python `>=3.10,<4`
- old `llama-agents==0.0.14`, uploaded in 2024

The old `llama-agents` package MUST NOT be added.

## Decision Drivers

- Preserve the current default LangGraph runtime.
- Avoid service orchestration unless a validated product need requires it.
- Avoid permanent dual-runtime abstractions and compatibility shims.
- Prefer one canonical runtime path.
- Require a future pilot to prove parity and net code deletion potential.
- Keep DocMind local-first and metadata-only in telemetry/logging.

## Weighted Decision Framework

Architecture options were scored from 1.0 to 10.0 using these weights:

- Runtime parity and correctness: 25%
- Persistence, checkpointing, resume, and time-travel fit: 20%
- Streaming, events, handoff, and deadline semantics: 15%
- Offline-first behavior and local/degraded-mode compatibility: 10%
- Telemetry, cache, and log-safety fit: 10%
- Dependency risk, ecosystem maturity, and operational simplicity: 10%
- Code deletion potential and entropy reduction: 10%

Thresholds:

- `9.0+`: eligible for approved pilot or adoption path
- `8.5-8.9`: promising, but requires more evidence before implementation
- `7.0-8.4`: defer unless it unlocks a higher-scoring path
- `<7.0`: reject or skip

| Option | Score | Decision | Rationale |
| --- | ---: | --- | --- |
| Keep current LangGraph v1 supervisor with no runtime pilot now | 9.0 | Adopt | Best current parity for checkpointing, store, time travel, deadlines, telemetry, cache, and offline behavior. Lower deletion upside, but lowest risk. |
| Add ADR/research note and open a future pilot issue | 9.3 | Adopt | Captures issue #86 cleanly, preserves default runtime, creates explicit gates, and adds no runtime entropy. |
| Pilot `llama-index-workflows` on one contained in-process flow | 8.7 | Defer | Promising future pilot candidate because Workflows are active and already transitive. Needs parity proof for checkpoint/time travel and must stay isolated. |
| Pilot LlamaIndex `AgentWorkflow` on one contained flow | 7.8 | Defer | Useful built-in handoff/state model, but overlaps more directly with the current supervisor and would likely require translation. |
| Pilot `llama-agents-server` as a service boundary | 6.1 | Reject | Adds HTTP service, deployment, persistence, auth, and operational complexity without a validated DocMind product need. |
| Replace the current LangGraph supervisor now | 3.2 | Reject | Violates issue #86 constraints and lacks parity evidence across persistence, deadlines, cache, telemetry, and offline behavior. |
| Add a dual-runtime abstraction layer | 3.0 | Reject | Creates permanent adapters, duplicated tests, and default-path ambiguity without a proven external contract. |

## Decision

Keep the current LangGraph v1 supervisor as DocMind's default and only runtime
path for issue #86.

Do not replace LangGraph in this issue. Do not add the old `llama-agents`
package. Do not add `llama-agents-server`, `llama-agents-client`, or `llamactl`
to the application runtime. Do not add a dual-runtime abstraction layer.

Modern LlamaIndex Workflows MAY be revisited only as a future contained,
in-process pilot. The candidate pilot scope is synthesis plus validation, not
retrieval, persistence, or UI routing. Retrieval remains out of scope because it
touches Qdrant, GraphRAG, router injection, semantic cache, and more durable
DocMind contracts.

## Future Pilot Conditions

A future pilot must satisfy all of these conditions before implementation:

- It remains isolated from the default runtime.
- It does not add a runtime selection flag.
- It does not introduce a LangGraph-to-LlamaIndex translation adapter.
- It does not add service orchestration.
- It proves checkpoint/resume/time-travel parity or documents a blocking gap.
- It preserves store/persistence expectations.
- It compares streaming event shape and handoff/backflow semantics.
- It preserves timeout and deadline propagation.
- It preserves semantic cache policy or proves cache is outside pilot scope.
- It preserves metadata-only telemetry and log safety.
- It works offline and with local/degraded LLM backends.
- It has deterministic failure and retry behavior.
- It uses focused regression tests.
- It identifies more custom code to delete than it adds before replacement is
  considered.

If any condition fails, the pilot code and tests should be deleted and the ADR
should record the failed evidence.

## Hard-Cut And Entropy Controls

The runtime must remain single-path. A pilot is not a backdoor to permanent
runtime complexity.

Do not add:

- legacy `llama-agents`
- service orchestration packages to the app runtime
- runtime feature flags for orchestration selection
- LangGraph/LlamaIndex coercion layers
- compatibility branches for old and new event/state shapes
- duplicated test suites for parallel orchestration paths

If a replacement is ever approved, it must be a hard cut: migrate call sites,
delete the old runtime path, delete obsolete tests and docs, and keep one
canonical shape.

## Consequences

### Positive Outcomes

- Preserves the production runtime and local-first behavior.
- Documents the modern LlamaIndex option with current evidence.
- Creates a precise future pilot gate without changing runtime code.
- Avoids permanent dual-runtime entropy.

### Trade-offs

- DocMind does not immediately benefit from LlamaIndex Workflow ergonomics.
- Any future pilot must first build a parity harness before runtime adoption can
  be considered.

## Validation

This ADR is a docs-only decision. Future docs updates should run markdown and
link validation. Future pilot code must also run focused agent tests, type
checking, and broader quality gates before adoption.

## Changelog

- 1.0 (2026-05-01): Accepted issue #86 decision: keep LangGraph as default,
  defer runtime replacement, and define strict future LlamaIndex Workflows pilot
  conditions.
