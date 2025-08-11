# ADR-011: LangGraph Multi-Agent

## Title

Multi-Agent Coordination with LangGraph Supervisor

## Version/Date

3.0 / July 25, 2025

## Status

Accepted

## Context

Agentic RAG requires a central supervisor for task delegation based on query complexity and type (via analyze_query_complexity), persistence for state/memory, human-in-loop for approvals, and dynamic handoffs between specialty agents—all offline with local Ollama LLM. Leverage langgraph-supervisor-py for prebuilt supervisor to reduce custom code, with custom agents for specialties (doc/kg/multimodal/planning). Research shows value in adding a planning agent (decomposes complex queries), better handoffs via Send primitive (dynamic state pass), and SqliteSaver for offline persistence (integrates with LlamaIndex).

## Related Requirements

- Phase 4: Supervisor with custom specialty agents (doc for retrieval, kg for relations, multimodal for images/tables, planning for query decomposition).

- Offline/local (Ollama LLM in agents, no external services).

- Persistence (SqliteSaver for checkpointer).

- Human-in-loop (interrupts for approvals).

- Handoffs (via Send primitive for context/state transfer).

## Alternatives

- Custom StateGraph without supervisor-py: More boilerplate, harder to leverage prebuilts like ToolNode/Send.

- CrewAI: Less integrated with LlamaIndex tools, no native Send primitive or SqliteSaver.

- Basic LangGraph without Send/planning: Lacks dynamic handoffs (e.g., no context-preserving transfers) and query decomposition for complex RAG.

## Decision

Use langgraph-supervisor-py (<https://github.com/langchain-ai/langgraph-supervisor-py>) for prebuilt multi-agent: Central supervisor coordinates custom specialty agents (built with create_react_agent using local Ollama LLM and LlamaIndex tools). Supervisor controls communication flow and delegates tasks based on context/requirements. Add planning specialist (decomposes queries). Integrate handoffs via Send primitive (pass state/messages dynamically). Use SqliteSaver checkpointer for persistence, interrupts for human-in-loop, ToolNode for tool execution.

## Related Decisions

- ADR-001 (Agents in overall architecture, integrated with LlamaIndex pipelines).

- ADR-008 (Persistence with SqliteSaver for agent state).

## Design

- **Setup with Supervisor-Py**: In src/agent_factory.py: from langgraph_supervisor import Supervisor; supervisor = Supervisor(agents=[doc_agent, kg_agent, multimodal_agent, planning_agent], llm=Ollama(AppSettings.default_model), tools=from src/utils/ create_tools_from_index).

- **Custom Specialty Agents**: Use create_react_agent for each: planning_agent = create_react_agent(llm=Ollama(AppSettings.default_model), tools=query_tools, state_modifier="Decompose complex queries into sub-tasks"); doc_agent = create_react_agent(..., state_modifier="Specialize in document retrieval/summarization"); Similar for kg (relations), multimodal (images/tables).

- **Supervisor Control**: Central supervisor receives inputs, analyzes context (complexity/type), delegates (e.g., if "analyze images": multimodal_agent), aggregates outputs.

- **Handoffs**: Use Send primitive: supervisor.send(to="kg_agent", state={"context": current_state, "messages": messages}) for dynamic transfer (preserves state without edges).

- **Persistence/Human-in-Loop**: from langgraph.checkpoint.sqlite import SqliteSaver; checkpointer=SqliteSaver.from_conn_string(AppSettings.cache_db_path); workflow.compile(checkpointer=checkpointer). Interrupts: interrupt_before=["tool_call"] (pause for human input via UI toggle in src/app.py).

- **Prebuilts/Optimizations**: Use ToolNode for tool execution in workers; Templates for hierarchies (e.g., planning → supervisor → workers).

- **Integration**: In src/app.py: agent_system = get_agent_system(tools, llm, enable_multi_agent); response = process_query_with_agent_system(agent_system, query, mode) — invokes supervisor for delegation/handoffs. Persist via checkpointer (thread_id for sessions). Integrate LlamaIndex tools (e.g., QueryEngineTool for retrievers) in workers.

- **Implementation Notes**: Local LLM: Ollama in all agents/workers. Tools from src/utils/ (offline RAG). Error handling: Retry on handoff failures (max 3, log errors). UI toggle for multi-agent/human-in-loop.

- **Testing**: tests/test_agents.py: def test_supervisor_delegation_handoff_planning(): supervisor = Supervisor(...); state = supervisor.invoke({"query": "complex image analysis"}); assert "planning_agent" in state["invoked"] and "decomposed" in state; assert "multimodal_agent" in state["handoff_to"]; assert handoff transferred context via Send; def test_persistence_interrupt(): state1 = supervisor.invoke(input, config={"thread_id": "1"}); resumed = supervisor.invoke({"human_input": "approve"}, config={"thread_id": "1"}); assert resumed["persisted"] and "approved" in resumed.

## Consequences

- Efficient agentic system (prebuilt supervisor reduces code, Send for dynamic handoffs, planning for complex query decomposition, SqliteSaver for offline persistence, interrupts for control).

- Modular/scalable (custom agents easy to add, ToolNode for tools).

- Offline (local Ollama LLM/tools).

- Deps: langgraph-supervisor-py (add to pyproject.toml).

- Complexity (manage state/handoffs—mitigate with prebuilts/templates and tests).

**Changelog:**  

- 3.0 (July 25, 2025): Integrated langgraph-supervisor-py for prebuilt; Added custom specialty agents including planning (decomposition); Emphasized central control/delegation; Incorporated Send primitive for handoffs; Optimized with SqliteSaver/ToolNode; Enhanced integrations/testing for dev.

- 2.0: Previous evolutions with create_react_agent/checkpointer.
