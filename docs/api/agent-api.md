# Use DocMind's multi-agent coordinator

This reference explains DocMind's internal Python coordination API, its response contract, and its persistence helpers. DocMind does not expose an HTTP, REST, or WebSocket agent API.

## Run a query

`MultiAgentCoordinator` compiles the LangGraph workflow on first use and reads
provider, model, context-window, and feature configuration from
`src.config.settings`. Build the active snapshot's router as shown in
[API usage](api.md#retrieval-examples) before querying.

```python
from src.agents.coordinator import MultiAgentCoordinator
from src.config import settings
from src.retrieval.router_factory import build_router_engine

# vector_index is the active persisted or newly built LlamaIndex vector index.
# graph_index is optional.
router = build_router_engine(vector_index, graph_index, settings)

coordinator = MultiAgentCoordinator()
try:
    result = coordinator.process_query(
        "Compare the indexed quarterly reports",
        settings_override={"router_engine": router},
        thread_id="quarterly-analysis",
        user_id="local",
    )
    print(result.content)
    print(result.validation_score)
    print(result.sources)
finally:
    # The router is bound to the coordinator loop after its first async query.
    router.close()
    coordinator.close()
```

Close the router before its coordinator. Coordinator shutdown releases the graph
runner and drains background memory work.

## Configure the coordinator

The constructor accepts runtime dependencies, not a second configuration surface:

```python
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.coordinator import MultiAgentCoordinator

coordinator = MultiAgentCoordinator(
    max_agent_timeout=120,
    checkpointer=InMemorySaver(),
)
```

Constructor parameters:

- `max_agent_timeout`: Optional workflow deadline in seconds. The default comes from `settings.agents.decision_timeout`.
- `checkpointer`: Optional LangGraph checkpointer. The default is `InMemorySaver`.
- `checkpointer_path`: Optional SQLite path for a coordinator-owned
  `AsyncSqliteSaver`. It is mutually exclusive with `checkpointer`; `close()`
  closes the saver and its connection. Startup rejects checkpoint tables that
  contain raw v1 thread IDs; follow the [v2 upgrade procedure](../../README.md#upgrade-from-v1)
  before opening that database.
- `store`: Optional LangGraph store for long-term memory.

Model names, context windows, and backend selection belong in application settings. The constructor does not accept overrides for those values.

## Pass request options

`process_query` accepts one query and keyword-only execution options:

```python
result = coordinator.process_query(
    "Summarize the indexed evidence",
    settings_override={"router_engine": injected_router},
    thread_id="analysis-42",
    user_id="local",
    checkpoint_id=None,
)
```

Parameters:

- `query`: Required query text.
- `settings_override`: Optional transient runtime objects. Document retrieval
  accepts only the prebuilt `router_engine`; do not use this mapping as a second
  application-settings tree.
- `thread_id`: Conversation identifier used by the checkpointer.
- `user_id`: Persistence namespace for conversation and memory state.
- `checkpoint_id`: Optional checkpoint to resume.

Conversation history belongs to the LangGraph checkpointer. `process_query` does not accept a separate context buffer.

## Read the response

Every call returns `AgentResponse`:

```python
from src.agents.models import AgentResponse

response: AgentResponse = coordinator.process_query(
    "Find the key risks",
    settings_override={"router_engine": router},
)

print(response.content)
print(response.sources)
print(response.metadata)
print(response.validation_score)
print(response.processing_time)
print(response.optimization_metrics)
print(response.agent_decisions)
```

Fields:

- `content`: Generated response text or the canonical timeout/error message.
- `sources`: Source-document dictionaries selected by retrieval or synthesis.
- `metadata`: Planning, timing, validation, and error metadata.
- `validation_score`: Confidence from `0.0` through `1.0`.
- `processing_time`: Total processing time in seconds.
- `optimization_metrics`: Context and coordination measurements.
- `agent_decisions`: Structured decision records when the graph emits them.

The response schema has no fallback flag. DocMind does not run a second hidden retrieval pipeline when coordination fails.

## Handle timeouts and errors

Timeouts and workflow failures remain explicit in response metadata:

```python
result = coordinator.process_query(
    "Analyze the indexed corpus",
    settings_override={"router_engine": router},
)
reason = result.metadata.get("reason")

if reason == "timeout":
    print("The coordinator exceeded its workflow deadline.")
elif reason in {"initialization_failed", "execution_failed"}:
    print(result.content)
else:
    print(result.content)
```

A timeout response has `metadata["reason"] == "timeout"`, validation score `0.0`, and `optimization_metrics["timeout"] is True`. Initialization and execution failures use the stable reasons `initialization_failed` and `execution_failed`, respectively, with `optimization_metrics["error"] is True`. These responses do not expose exception text.

## Inspect persisted state

Use the coordinator's checkpoint helpers to inspect a conversation without reaching into the compiled graph:

```python
state = coordinator.get_state_values(
    thread_id="analysis-42",
    user_id="local",
)

checkpoints = coordinator.list_checkpoints(
    thread_id="analysis-42",
    user_id="local",
    limit=20,
)
```

`get_state_values` returns the selected checkpoint's state dictionary or an empty dictionary when no state exists. `list_checkpoints` returns checkpoint metadata in reverse chronological order.

## Fork a checkpoint

Create a non-destructive branch from a prior checkpoint:

```python
new_checkpoint_id = coordinator.fork_from_checkpoint(
    thread_id="analysis-42",
    user_id="local",
    checkpoint_id="checkpoint-id-from-list",
)
```

The method returns the new checkpoint identifier or `None` when the source checkpoint cannot be read or copied.

Fork and hard-purge mutations share one coordinator lock. If they overlap, the fork either completes before deletion or fails after the purge fence is installed. Session-management callers must use `coordinator.purge_session(...)` instead of deleting persistence rows directly.

## Treat role tools as internal

The planner, retrieval, synthesis, and validation functions under `src.agents.tools` are graph implementation details. Their state and runtime contracts may change with the coordinator. Application code should call `process_query` instead of invoking individual role tools as public APIs.

## Record metadata-only telemetry

Emit timing without logging query or response content:

```python
import time

from src.utils.telemetry import log_jsonl

started = time.perf_counter()
result = coordinator.process_query(
    "Analyze the indexed documents",
    settings_override={"router_engine": router},
)
log_jsonl(
    {
        "event": "agent_coordination",
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "result_length": len(result.content),
    }
)
```

Do not add raw queries, document text, provider credentials, or exception messages to telemetry payloads.
