# ADR-011: LangGraph Multi-Agent

## Title

Multi-Agent Coordination with LangGraph Supervisor

## Version/Date

3.0 / July 25, 2025

## Status

Accepted

## Context

Agentic RAG requires a central supervisor for intelligent routing based on query complexity and type (via analyze_query_complexity), persistence for state/memory across sessions, human-in-loop for approvals, and seamless handoffs between specialty agents—all fully offline with local Ollama LLM. Use langgraph-supervisor-py for prebuilt supervisor to minimize custom code while allowing custom agents (e.g., doc/kg/multimodal specialists).

## Related Requirements

- Phase 4: Supervisor with custom specialty agents (doc for retrieval, kg for relations, multimodal for images/tables).
- Offline/local (Ollama LLM in agents, no external services).
- Persistence (checkpointer for agent state/memory).
- Human-in-loop (interrupts for approvals).
- Handoffs (via LangGraph Handoff primitive for context transfer).

## Alternatives

- Custom StateGraph without supervisor-py: More boilerplate, harder to maintain prebuilt patterns.
- CrewAI: Less integrated with LlamaIndex tools, no native handoff primitive.
- Basic LangGraph without handoffs: Lacks seamless delegation (e.g., no context-preserving transfers).

## Decision

Use langgraph-supervisor-py (from <https://github.com/langchain-ai/langgraph-supervisor-py>) for prebuilt multi-agent system: Central supervisor agent coordinates custom specialty agents (built with create_react_agent using local Ollama LLM and LlamaIndex tools). Supervisor controls all communication flow and task delegation, deciding invocation based on current context/task requirements. Integrate handoffs via LangGraph's Handoff primitive[](https://langchain-ai.github.io/langgraph/agents/multi-agent/) for transferring control/state between agents. Add MemorySaver checkpointer for persistence, interrupts for human-in-loop.

## Related Decisions

- ADR-001 (Agents in overall architecture, integrated with LlamaIndex pipelines).
- ADR-008 (Persistence with checkpointer for agent state).

## Design

- **Setup with Supervisor-Py**: Install langgraph-supervisor-py (add to pyproject.toml: "langgraph-supervisor-py==latest"). In agent_factory.py: from langgraph_supervisor import Supervisor; supervisor = Supervisor(agents=[doc_agent, kg_agent, multimodal_agent], llm=Ollama(AppSettings.default_model), tools=from utils.py create_tools_from_index).
- **Custom Specialty Agents**: Use create_react_agent for each: doc_agent = create_react_agent(llm=Ollama(AppSettings.default_model), tools=retriever_tools, state_modifier="Specialize in document retrieval and summarization"). Similar for kg (relations) and multimodal (images/tables).
- **Supervisor Control**: Supervisor decides routing (e.g., if context has "image": invoke multimodal_agent). Controls flow: Receives inputs, delegates tasks, aggregates outputs.
- **Handoffs**: Use Handoff primitive: supervisor.handoff(to="kg_agent", state={"context": current_state}) for seamless transfer (preserves messages/memory).
- **Persistence/Human-in-Loop**: Add checkpointer=MemorySaver(); supervisor.compile(checkpointer=checkpointer). Interrupts: interrupt_before=["tool_call"] (pause for human input via UI toggle in app.py).
- **Integration**: In app.py: agent_system = get_agent_system(tools, llm, enable_multi_agent); response = process_query_with_agent_system(agent_system, query, mode) — invokes supervisor for delegation/handoffs. Persist via checkpointer (thread_id for sessions).
- **Implementation Notes**: Local LLM: Ollama in all agents/workers. Tools from utils.py (e.g., QueryEngineTool for retrievers). Error handling: Retry on handoff failures (max 3, log errors). UI toggle for multi-agent/human-in-loop.
- **Testing**: tests/test_agents.py: def test_supervisor_delegation_handoff(): supervisor = Supervisor(...); state = supervisor.invoke({"query": "complex image query"}); assert "multimodal_agent" in state["invoked"]; assert handoff transferred context; def test_persistence_interrupt(): state1 = supervisor.invoke(input, config={"thread_id": "1"}); resumed = supervisor.invoke({"human_input": "approve"}, config={"thread_id": "1"}); assert resumed["persisted"] and "approved" in resumed.

## Consequences

- Efficient agentic system (prebuilt supervisor reduces code, handoffs for seamless flow, persistence for offline sessions, interrupts for control).
- Modular/scalable (custom agents easy to add, Send API for data handoffs).
- Offline (local Ollama LLM/tools).

- Deps: langgraph-supervisor-py (add to pyproject.toml).
- Complexity (manage state/handoffs—mitigated by prebuilt library and tests).

**Changelog:**  

- 3.0 (July 25, 2025): Integrated langgraph-supervisor-py for prebuilt; Added custom specialty agents/central control/delegation; Incorporated Handoff primitive; Enhanced persistence/interrupts/offline integrations/testing for streamlined dev.
- 2.0: Previous evolutions with create_react_agent/checkpointer.
