"""Graph-native multi-agent supervisor orchestration (LangGraph v1).

This module replaces the deprecated third-party supervisor wrapper by providing a
small, repo-local implementation of the same high-level pattern:

- A `supervisor` agent decides which subagent to run next.
- Handoffs are implemented as tools that return `langgraph.types.Command`
  targeting the *parent* graph (`Command.PARENT`).
- Subagents are regular `langchain.agents.create_agent(...)` graphs (or any
  runnable compatible with LangGraph).

The implementation intentionally stays small and dependency-light while keeping
the DocMind coordinator's existing seams (`compiled_graph.stream(..., ...)`),
checkpointer/store, and runtime context injection.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Literal, TypeGuard, cast

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from pydantic import BaseModel
from typing_extensions import TypedDict

OutputMode = Literal["full_history", "last_message"]

_WHITESPACE_RE = re.compile(r"\s+")
_METADATA_KEY_IS_HANDOFF_BACK = "__is_handoff_back"


def _normalize_agent_name(agent_name: str) -> str:
    """Normalizes an agent name to lowercase with underscores.

    Args:
        agent_name: The raw agent name string.

    Returns:
        The normalized lowercase agent name suitable for tool naming.
    """
    return _WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def _has_multiple_content_blocks(
    content: str | list[str | dict[str, Any]],
) -> TypeGuard[list[dict[str, Any]]]:
    """Checks if message content consists of multiple blocks.

    Args:
        content: The message content to check, which can be a string or list.

    Returns:
        True if content is a list with at least two elements and the first is a
        dictionary.
    """
    return (
        isinstance(content, list) and len(content) > 1 and isinstance(content[0], dict)
    )


def _remove_non_handoff_tool_calls(
    last_ai_message: AIMessage, handoff_tool_call_id: str
) -> AIMessage:
    """Filters parallel tool calls to retain only the specified handoff call.

    Args:
        last_ai_message: The message containing potential parallel tool calls.
        handoff_tool_call_id: The ID of the handoff tool call to preserve.

    Returns:
        A new AIMessage containing only the target handoff tool call and its content.
    """
    content: str | list[str | dict[str, Any]] = last_ai_message.content
    if _has_multiple_content_blocks(content):
        content = [
            block
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") == handoff_tool_call_id
            )
            or (not isinstance(block, dict) or block.get("type") != "tool_use")
        ]
    return AIMessage(
        content=content,
        tool_calls=[
            tool_call
            for tool_call in last_ai_message.tool_calls
            if tool_call.get("id") == handoff_tool_call_id
        ],
        name=last_ai_message.name,
        id=str(uuid.uuid4()),
    )


def create_handoff_tool(
    *,
    agent_name: str,
    name: str | None = None,
    description: str | None = None,
    add_handoff_messages: bool = True,
) -> BaseTool:
    """Creates a tool that hands control to the requested agent node.

    Args:
        agent_name: The destination agent's identifier.
        name: Optional custom name for the tool. Defaults to 'transfer_to_{agent_name}'.
        description: Optional tool description.
        add_handoff_messages: Whether to append tool messages to the state history.

    Returns:
        A BaseTool instance configured to return a Command for graph navigation.
    """
    if name is None:
        name = f"transfer_to_{_normalize_agent_name(agent_name)}"
    if description is None:
        description = f"Ask agent '{agent_name}' for help"

    @tool(name, description=description)
    def handoff_to_agent(
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Executes the handoff logic, updating state and directing graph flow.

        Args:
            state: The current graph state.
            tool_call_id: The unique identifier for this tool execution.

        Returns:
            A Command object that directs the parent graph to the target agent.
        """
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        last_ai_message = cast(AIMessage, state["messages"][-1])

        # Parallel handoffs: use Send to allow ToolNode aggregation.
        if len(getattr(last_ai_message, "tool_calls", []) or []) > 1:
            handoff_messages: list[AnyMessage] = list(state["messages"][:-1])
            if add_handoff_messages:
                handoff_messages.extend(
                    (
                        _remove_non_handoff_tool_calls(last_ai_message, tool_call_id),
                        tool_message,
                    )
                )
            return Command(
                graph=Command.PARENT,
                goto=[Send(agent_name, {**state, "messages": handoff_messages})],
            )

        # Single handoff: update parent state and jump.
        if add_handoff_messages:
            handoff_messages = [*state["messages"], tool_message]
        else:
            handoff_messages = list(state["messages"][:-1])
        return Command(
            graph=Command.PARENT,
            goto=agent_name,
            update={"messages": handoff_messages},
        )

    return handoff_to_agent


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


def create_forward_message_tool(*, supervisor_name: str = "supervisor") -> BaseTool:
    """Creates a tool the supervisor can use to forward a worker message verbatim.

    Args:
        supervisor_name: The name of the supervisor agent.

    Returns:
        A BaseTool that searches history and returns a Command to update state.
    """

    @tool(
        "forward_message",
        description=(
            "Forwards the latest message from the specified agent to the user "
            "without changes. Use this to preserve information fidelity."
        ),
    )
    def forward_message(
        from_agent: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> Command | str:
        """Locates and forwards the most recent message from a specific agent.

        Args:
            from_agent: The name of the agent whose message should be forwarded.
            state: The current graph state containing message history.

        Returns:
            A Command that updates the state with the forwarded message, or an
            error string if the agent/message is not found.
        """
        target = next(
            (
                m
                for m in reversed(state.get("messages", []))
                if isinstance(m, AIMessage)
                and (m.name or "").lower() == str(from_agent).lower()
                and not bool(m.response_metadata.get(_METADATA_KEY_IS_HANDOFF_BACK))
            ),
            None,
        )
        if target is None:
            found = {
                m.name
                for m in state.get("messages", [])
                if isinstance(m, AIMessage) and m.name
            }
            return (
                f"Could not find message from source agent {from_agent}. "
                f"Found names: {sorted(found)}"
            )

        update_messages = [
            AIMessage(
                content=target.content,
                name=supervisor_name,
                id=str(uuid.uuid4()),
            )
        ]
        return Command(
            graph=Command.PARENT,
            goto=END,
            update={"messages": update_messages},
        )

    return forward_message


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
    extra_tools: Sequence[BaseTool | Callable[..., Any]] = (),
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
        extra_tools: Additional tools available to the supervisor.
        params: Configuration object for build options.

    Returns:
        A compiled StateGraph ready for execution.

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

    handoff_tools: list[BaseTool | Callable[..., Any]] = [
        create_handoff_tool(
            agent_name=name,
            add_handoff_messages=cfg.add_handoff_messages,
        )
        for name in agent_names
    ]
    all_tools = [*handoff_tools, *list(extra_tools)]

    supervisor_agent = create_agent(
        model,
        tools=all_tools,
        system_prompt=prompt,
        middleware=middleware,
        state_schema=state_schema,
        context_schema=context_schema,
        name=cfg.supervisor_name,
    )

    workflow_schema: type[Any] = state_schema or _OuterState
    builder = StateGraph(workflow_schema, context_schema=context_schema)
    builder.add_node(
        cfg.supervisor_name,
        supervisor_agent,
        destinations=(*agent_names, END),
    )
    builder.add_edge(START, cfg.supervisor_name)

    def _make_call_agent(
        agent: Any,
    ) -> Callable[[dict[str, Any], RunnableConfig], dict[str, Any]]:
        """Creates a wrapper function to invoke a specific agent."""

        def call_agent(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            """Invokes the agent and processes its output messages.

            Args:
                state: The input state for the agent.
                config: Runtime configuration options.

            Returns:
                A dictionary containing the agent's output state with processed
                messages.
            """
            output = agent.invoke(state, config=config)
            messages = _select_output_messages(
                cast(Sequence[AnyMessage], output.get("messages", [])),
                output_mode=cfg.output_mode,
            )
            if cfg.add_handoff_back_messages:
                messages.extend(
                    create_handoff_back_messages(str(agent.name), cfg.supervisor_name)
                )
            return {**output, "messages": messages}

        return call_agent

    for agent in agents:
        builder.add_node(str(agent.name), _make_call_agent(agent))
        builder.add_edge(str(agent.name), cfg.supervisor_name)

    return builder
