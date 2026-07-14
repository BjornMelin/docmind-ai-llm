"""Graph-native multi-agent supervisor orchestration (LangGraph v1).

This module replaces the deprecated third-party supervisor wrapper by providing a
small, repo-local implementation of the same high-level pattern:

- A `supervisor` agent decides which subagent to run next.
- Handoffs are implemented as tools that return `langgraph.types.Command`
  targeting the *parent* graph (`Command.PARENT`).
- Subagents are regular asynchronous `langchain.agents.create_agent(...)`
  graphs (or compatible LangGraph runnables).

The implementation intentionally stays small and dependency-light while keeping
the DocMind coordinator's async execution, checkpointer/store, and runtime
context injection seams.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from pydantic import BaseModel
from typing_extensions import TypedDict

OutputMode = Literal["full_history", "last_message"]

_WHITESPACE_RE = re.compile(r"\s+")
_METADATA_KEY_IS_HANDOFF_BACK = "__is_handoff_back"
_DISPATCH_TOOL_NAME = "dispatch_agents"
_WORKER_DOMAIN_OUTPUT_KEYS = (
    "planning_output",
    "synthesis_result",
    "validation_result",
)


def _normalize_agent_name(agent_name: str) -> str:
    """Normalizes an agent name to lowercase with underscores.

    Args:
        agent_name: The raw agent name string.

    Returns:
        The normalized lowercase agent name suitable for tool naming.
    """
    return _WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def _dispatch_message(
    last_ai_message: AIMessage,
    *,
    tool_call_id: str,
    destinations: list[str],
) -> AIMessage:
    """Collapse one or more dispatch calls into one resolved message."""
    content: str | list[str | dict[str, Any]] = last_ai_message.content
    if isinstance(content, list):
        content = [
            block
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") == tool_call_id
            )
            or (not isinstance(block, dict) or block.get("type") != "tool_use")
        ]
    return AIMessage(
        content=content,
        tool_calls=[
            ToolCall(
                name=_DISPATCH_TOOL_NAME,
                args={"destinations": destinations},
                id=tool_call_id,
            )
        ],
        name=last_ai_message.name,
        id=str(uuid.uuid4()),
    )


def create_dispatch_tool(
    *,
    agent_names: Sequence[str],
    add_handoff_messages: bool = True,
) -> BaseTool:
    """Create one canonical tool that atomically dispatches unique workers."""
    allowed = tuple(agent_names)
    allowed_set = set(allowed)
    description = (
        "Dispatch one or more independent workers in a single call. Pass a unique "
        f"list chosen only from: {', '.join(allowed)}."
    )

    @tool(_DISPATCH_TOOL_NAME, description=description)
    def dispatch_agents(
        destinations: list[str],
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Dispatch all requested destinations in one parent graph command."""
        messages = state.get("messages") or []
        if not messages:
            raise ValueError("Dispatch requires an AI tool-call message")
        last_ai_message = cast(AIMessage, messages[-1])

        dispatch_calls = [
            tool_call
            for tool_call in (getattr(last_ai_message, "tool_calls", []) or [])
            if tool_call.get("name") == _DISPATCH_TOOL_NAME
        ]
        first_call_id = (
            str(dispatch_calls[0].get("id")) if dispatch_calls else tool_call_id
        )
        if tool_call_id != first_call_id:
            return Command(
                graph=Command.PARENT,
                update={
                    "messages": [
                        ToolMessage(
                            content="Additional dispatch call merged into the first",
                            name=_DISPATCH_TOOL_NAME,
                            tool_call_id=tool_call_id,
                        )
                    ]
                },
            )

        requested: list[str] = []
        raw_destination_lists: list[object] = [destinations]
        raw_destination_lists.extend(
            call.get("args", {}).get("destinations")
            for call in dispatch_calls[1:]
            if isinstance(call.get("args"), dict)
        )
        for raw_destinations in raw_destination_lists:
            if not isinstance(raw_destinations, list) or not all(
                isinstance(destination, str) for destination in raw_destinations
            ):
                raise ValueError("Dispatch destinations must be a list of agent names")
            for destination in raw_destinations:
                if destination not in allowed_set:
                    raise ValueError(f"Unknown dispatch destination: {destination}")
                if destination not in requested:
                    requested.append(destination)
        if not requested:
            raise ValueError("Dispatch requires at least one destination")

        tool_message = ToolMessage(
            content=f"Dispatched to: {', '.join(requested)}",
            name=_DISPATCH_TOOL_NAME,
            tool_call_id=first_call_id,
        )
        if add_handoff_messages:
            handoff_messages: list[AnyMessage] = [*state["messages"][:-1]]
            handoff_messages.extend(
                (
                    _dispatch_message(
                        last_ai_message,
                        tool_call_id=first_call_id,
                        destinations=requested,
                    ),
                    tool_message,
                )
            )
        else:
            handoff_messages = list(state["messages"][:-1])
        return Command(
            graph=Command.PARENT,
            goto=[
                Send(
                    destination,
                    {**state, "messages": handoff_messages},
                )
                for destination in requested
            ],
        )

    return dispatch_agents


def create_handoff_back_messages(
    agent_name: str, supervisor_name: str
) -> tuple[AIMessage, ToolMessage]:
    """Generates messages to signal a return of control to the supervisor.

    Args:
        agent_name: The name of the agent yielding control.
        supervisor_name: The name of the supervisor expecting control.

    Returns:
        A tuple containing an AIMessage (tool call) and ToolMessage (result)
        marked with handoff metadata.
    """
    tool_call_id = str(uuid.uuid4())
    tool_name = f"transfer_back_to_{_normalize_agent_name(supervisor_name)}"
    tool_calls = [ToolCall(name=tool_name, args={}, id=tool_call_id)]
    return (
        AIMessage(
            content=f"Transferring back to {supervisor_name}",
            tool_calls=tool_calls,
            name=agent_name,
            id=str(uuid.uuid4()),
            response_metadata={_METADATA_KEY_IS_HANDOFF_BACK: True},
        ),
        ToolMessage(
            content=f"Successfully transferred back to {supervisor_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
            response_metadata={_METADATA_KEY_IS_HANDOFF_BACK: True},
        ),
    )


def _select_output_messages(
    messages: Sequence[AnyMessage], *, output_mode: OutputMode
) -> list[AnyMessage]:
    """Selects messages for the output state based on the configured mode.

    Args:
        messages: The list of messages produced by an agent.
        output_mode: 'full_history' returns all messages; 'last_message' returns
            only the final message (or last two if the final is a ToolMessage).

    Returns:
        A list of selected messages.

    Raises:
        ValueError: If an unknown output_mode is provided.
    """
    if not messages:
        return []
    if output_mode == "full_history":
        return list(messages)
    if output_mode != "last_message":
        raise ValueError(f"Invalid output_mode: {output_mode}")

    last = messages[-1]
    if isinstance(last, ToolMessage) and len(messages) >= 2:
        return list(messages[-2:])
    return [last]


def _new_messages(
    previous: Sequence[AnyMessage], updated: Sequence[AnyMessage]
) -> list[AnyMessage]:
    """Return only messages created by a nested graph invocation."""
    if list(updated[: len(previous)]) == list(previous):
        return list(updated[len(previous) :])

    previous_ids = {
        message.id for message in previous if isinstance(message.id, str) and message.id
    }
    return [
        message
        for message in updated
        if not (
            isinstance(message.id, str) and message.id and message.id in previous_ids
        )
        and message not in previous
    ]


def _retrieval_batch_key(value: object) -> tuple[str, str] | None:
    """Return a validated turn-scoped retrieval batch identity."""
    if not isinstance(value, dict):
        return None
    turn_id = value.get("turn_id")
    retrieval_id = value.get("retrieval_id")
    if (
        not isinstance(turn_id, str)
        or not turn_id.strip()
        or not isinstance(retrieval_id, str)
        or not retrieval_id.strip()
    ):
        return None
    return turn_id, retrieval_id


def _new_retrieval_results(
    previous: list[dict[str, Any]], updated: object, *, turn_id: str
) -> list[dict[str, Any]]:
    """Reconcile the current turn by immutable, turn-scoped batch identity."""
    if not turn_id.strip():
        raise ValueError("Retrieval reconciliation requires a turn identity")
    if not isinstance(updated, list):
        raise ValueError("Worker returned invalid retrieval results")

    previous_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for batch in previous:
        if batch.get("turn_id") != turn_id:
            continue
        retrieval_id = _retrieval_batch_key(batch)
        if retrieval_id is None or retrieval_id in previous_by_id:
            raise ValueError("Current retrieval results have invalid identities")
        previous_by_id[retrieval_id] = batch

    new_batches: list[dict[str, Any]] = []
    updated_ids: set[tuple[str, str]] = set()
    for batch in updated:
        if not isinstance(batch, dict) or batch.get("turn_id") != turn_id:
            continue
        retrieval_id = _retrieval_batch_key(batch)
        if retrieval_id is None or retrieval_id in updated_ids:
            raise ValueError("Worker returned invalid retrieval result identities")
        updated_ids.add(retrieval_id)
        typed_batch = cast(dict[str, Any], batch)
        existing = previous_by_id.get(retrieval_id)
        if existing is not None:
            if typed_batch != existing:
                raise ValueError("Worker mutated an existing retrieval result")
            continue
        new_batches.append(typed_batch)
    return new_batches


class _OuterState(TypedDict):
    """Minimum required state for the supervisor workflow.

    Attributes:
        messages: A list of messages with a reducer to append new ones.
    """

    messages: Annotated[list[AnyMessage], add_messages]


class SupervisorBuildParams(BaseModel):
    """Strict parameter object for building the supervisor graph.

    Attributes:
        supervisor_name: The node name for the supervisor agent.
        output_mode: filtering strategy for agent outputs ('full_history' or
            'last_message').
        add_handoff_messages: Whether to add handoff tool messages to history.
        add_handoff_back_messages: Whether to add return-control messages to history.
    """

    supervisor_name: str = "supervisor"
    output_mode: OutputMode = "last_message"
    add_handoff_messages: bool = True
    add_handoff_back_messages: bool = True


def build_multi_agent_supervisor_graph(
    agents: Sequence[Any],
    *,
    model: str | BaseChatModel,
    prompt: str,
    state_schema: type[AgentState[Any]] | None = None,
    context_schema: type[Any] | None = None,
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
    params: SupervisorBuildParams | None = None,
) -> StateGraph:
    """Build a parent StateGraph coordinating a supervisor + subagents.

    Args:
        agents: Identifying sub-agents (must be runnables with a .name attribute).
        model: The LLM instance for the supervisor agent.
        prompt: System prompt for the supervisor.
        state_schema: Optional custom schema for graph state.
        context_schema: Optional schema for runtime context.
        middleware: List of agent middlewares to apply.
        params: Configuration object for build options.

    Returns:
        An uncompiled StateGraph builder. Call `.compile(checkpointer=..., store=...)`
        to obtain the executable graph.

    Raises:
        ValueError: If agents lack names or have duplicate names.
    """
    cfg = params or SupervisorBuildParams()
    agent_names: list[str] = []
    for agent in agents:
        name = getattr(agent, "name", None)
        if not name:
            raise ValueError(
                "All subagents must have a non-empty `.name` attribute. "
                "Pass `name=...` when creating agents via `create_agent`."
            )
        agent_names.append(str(name))
    if len(set(agent_names)) != len(agent_names):
        raise ValueError("Subagent names must be unique")

    dispatch_tool = create_dispatch_tool(
        agent_names=agent_names,
        add_handoff_messages=cfg.add_handoff_messages,
    )
    supervisor_agent = create_agent(
        model,
        tools=[dispatch_tool],
        system_prompt=prompt,
        middleware=middleware,
        state_schema=state_schema,
        context_schema=context_schema,
        name=cfg.supervisor_name,
    )

    workflow_schema: type[Any] = state_schema or _OuterState
    builder = StateGraph(workflow_schema, context_schema=context_schema)

    async def call_supervisor(
        state: dict[str, Any],
        config: RunnableConfig,
        runtime: Runtime[Any],
    ) -> Command | dict[str, Any]:
        """Invoke the nested supervisor without echoing parent graph state."""
        output = await supervisor_agent.ainvoke(
            state,
            config=config,
            context=runtime.context,
        )
        if isinstance(output, Command):
            return output
        if not isinstance(output, dict):
            raise TypeError("Supervisor returned an invalid graph result")
        previous_messages = cast(Sequence[AnyMessage], state.get("messages", []))
        updated_messages = output.get("messages", [])
        if not isinstance(updated_messages, list):
            raise TypeError("Supervisor returned invalid messages")
        messages = _select_output_messages(
            _new_messages(
                previous_messages,
                cast(Sequence[AnyMessage], updated_messages),
            ),
            output_mode=cfg.output_mode,
        )
        return {"messages": messages}

    builder.add_node(
        cfg.supervisor_name,
        call_supervisor,
        destinations=(*agent_names, END),
    )
    builder.add_edge(START, cfg.supervisor_name)

    def _make_call_agent(
        agent: Any,
    ) -> Callable[
        [dict[str, Any], RunnableConfig, Runtime[Any]],
        Awaitable[dict[str, Any]],
    ]:
        """Creates a wrapper function to invoke a specific agent."""

        async def call_agent(
            state: dict[str, Any],
            config: RunnableConfig,
            runtime: Runtime[Any],
        ) -> dict[str, Any]:
            """Invokes the agent and processes its output messages.

            Args:
                state: The input state for the agent.
                config: Runtime configuration options.
                runtime: Parent graph runtime carrying transient tool context.

            Returns:
                A dictionary containing the agent's output state with processed
                messages.
            """
            output = await agent.ainvoke(
                state,
                config=config,
                context=runtime.context,
            )
            messages = _select_output_messages(
                _new_messages(
                    cast(Sequence[AnyMessage], state.get("messages", [])),
                    cast(Sequence[AnyMessage], output.get("messages", [])),
                ),
                output_mode=cfg.output_mode,
            )
            if cfg.add_handoff_back_messages:
                messages.extend(
                    create_handoff_back_messages(str(agent.name), cfg.supervisor_name)
                )
            # Nested agents return their full input state. Forward only worker-owned
            # deltas: echoing parent LastValue channels breaks parallel handoffs and
            # lets nested `remaining_steps` overwrite the supervisor's budget.
            node_output: dict[str, Any] = {"messages": messages}
            # Nested agents return full state; additive parent channels need only
            # the worker delta or every handoff re-appends prior retrieval batches.
            previous_retrieval = state.get("retrieval_results", [])
            if "retrieval_results" in output:
                if not isinstance(previous_retrieval, list) or not all(
                    isinstance(batch, dict) for batch in previous_retrieval
                ):
                    raise ValueError("Existing retrieval results are invalid")
                new_retrieval = _new_retrieval_results(
                    cast(list[dict[str, Any]], previous_retrieval),
                    output["retrieval_results"],
                    turn_id=(
                        state["turn_id"]
                        if isinstance(state.get("turn_id"), str)
                        else ""
                    ),
                )
                if new_retrieval:
                    node_output["retrieval_results"] = new_retrieval
            for key in _WORKER_DOMAIN_OUTPUT_KEYS:
                if key in output and output[key] != state.get(key):
                    node_output[key] = output[key]
            return node_output

        return call_agent

    for agent in agents:
        builder.add_node(str(agent.name), _make_call_agent(agent))
        builder.add_edge(str(agent.name), cfg.supervisor_name)

    return builder
