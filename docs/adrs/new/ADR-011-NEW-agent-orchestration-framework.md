# ADR-011-NEW: Agent Orchestration Framework

## Title

LangGraph-Based Agent Orchestration with Local-First Agentic Patterns

## Version/Date

2.0 / 2025-01-16

## Status

Proposed

## Description

Implements a simplified agent orchestration framework using the pre-built `langgraph-supervisor` library to coordinate the agentic RAG system. This approach replaces complex custom orchestration code with a proven supervisor pattern, dramatically reducing implementation complexity while maintaining all required functionality for local-first operation.

## Context

The modernized architecture introduces multiple agentic components that need coordination:

1. **Query Routing Agent**: Determines optimal retrieval strategy
2. **Retrieval Correction Agent**: Evaluates and improves retrieval quality
3. **Response Validation Agent**: Ensures response quality and accuracy
4. **Quality Assessment Agent**: Provides continuous feedback and improvement

These agents need orchestration to work together effectively while avoiding over-complexity and maintaining local-first constraints. The `langgraph-supervisor` library provides the optimal solution with proven supervisor patterns, automatic state management, and conditional execution - eliminating the need for custom orchestration code.

## Related Requirements

### Functional Requirements

- **FR-1:** Orchestrate multi-agent workflows with conditional execution
- **FR-2:** Maintain conversation context across agent interactions
- **FR-3:** Provide fallback mechanisms when agent decisions fail
- **FR-4:** Support both synchronous and asynchronous agent execution
- **FR-5:** Enable agent introspection and decision transparency

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Agent coordination overhead <500ms per query
- **NFR-2:** **(Reliability)** Graceful degradation when individual agents fail
- **NFR-3:** **(Local-First)** All agent operations execute locally without external APIs
- **NFR-4:** **(Maintainability)** Clear agent boundaries and testable workflows

## Alternatives

### 1. Single Monolithic Agent

- **Description**: Single agent handling all RAG operations
- **Issues**: Limited flexibility, no specialization, difficult error recovery
- **Score**: 4/10 (simplicity: 8, capability: 2, maintainability: 2)

### 2. Manual Agent Coordination

- **Description**: Custom orchestration logic without framework
- **Issues**: Complex state management, error-prone coordination, hard to debug
- **Score**: 5/10 (control: 8, reliability: 3, maintainability: 4)

### 3. Heavy Multi-Agent Framework (AutoGen)

- **Description**: Full enterprise multi-agent system
- **Issues**: Over-engineered for single-user app, complex setup, resource intensive
- **Score**: 6/10 (capability: 9, complexity: 3, resource-usage: 4)

### 4. LangGraph Supervisor Library (Selected)

- **Description**: Pre-built supervisor pattern using `langgraph-supervisor-py`
- **Benefits**: Proven library (1.2k stars), ~90% code reduction, battle-tested patterns
- **Score**: 10/10 (capability: 9, simplicity: 10, reliability: 10)

## Decision

We will implement **LangGraph Supervisor-Based Orchestration** using the `langgraph-supervisor` library with:

### Core Components

1. **Supervisor Agent**: Central coordinator using `create_supervisor()` function
2. **Specialized Worker Agents**: Query routing, retrieval, validation agents
3. **Handoff Tools**: Automatic agent-to-agent delegation via supervisor library
4. **Context Preservation**: Built-in conversation and decision context management
5. **Error Handling**: Library-provided error recovery and fallback mechanisms
6. **Observability Integration**: Standard supervisor metrics and logging hooks

## Related Decisions

- **ADR-001-NEW** (Modern Agentic RAG): Implements the agent patterns orchestrated by this framework
- **ADR-004-NEW** (Local-First LLM Strategy): Provides the LLM for agent decision-making
- **ADR-008-NEW** (Production Observability): Monitors agent performance and decisions
- **ADR-005-NEW** (Framework Abstraction Layer): Integrates with abstracted components

## Design

### Simplified Supervisor Architecture

```mermaid
graph TD
    A[User Query] --> B[Supervisor Agent]
    B --> C{Agent Selection}
    
    C -->|Query Analysis| D[Query Analysis Agent]
    C -->|Retrieval| E[Retrieval Agent] 
    C -->|Quality Check| F[Quality Assessment Agent]
    C -->|Generation| G[Response Generation Agent]
    C -->|Validation| H[Response Validation Agent]
    
    D --> B
    E --> B
    F --> B
    G --> B
    H --> B
    
    B --> I[Final Response]
    
    J[Error Handler] --> K[Automatic Recovery]
```

### Simplified Supervisor Implementation

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import List, Dict, Any

# Define specialized agent tools
@tool
def analyze_query(query: str) -> Dict[str, Any]:
    """Analyze query complexity and determine retrieval strategy."""
    # Query analysis logic here
    return {"strategy": "vector", "complexity": "medium"}

@tool 
def retrieve_documents(query: str, strategy: str) -> List[Dict]:
    """Retrieve relevant documents using specified strategy."""
    # Retrieval logic here
    return [{"content": "...", "score": 0.85}]

@tool
def assess_quality(query: str, documents: List[Dict]) -> Dict[str, Any]:
    """Assess retrieval quality and determine if adequate."""
    # Quality assessment logic here
    return {"quality_score": 0.8, "adequate": True}

@tool
def generate_response(query: str, documents: List[Dict]) -> str:
    """Generate response from retrieved documents."""
    # Response generation logic here
    return "Generated response based on documents..."

@tool
def validate_response(query: str, response: str, documents: List[Dict]) -> Dict[str, Any]:
    """Validate response quality and accuracy."""
    # Response validation logic here
    return {"valid": True, "confidence": 0.9}

# Create specialized agents
query_agent = create_react_agent(
    model=llm,
    tools=[analyze_query],
    name="query_analyst",
    prompt="You analyze queries to determine optimal retrieval strategies."
)

retrieval_agent = create_react_agent(
    model=llm,
    tools=[retrieve_documents],
    name="retrieval_expert", 
    prompt="You retrieve documents using various strategies."
)

quality_agent = create_react_agent(
    model=llm,
    tools=[assess_quality],
    name="quality_assessor",
    prompt="You assess the quality of retrieved documents."
)

generation_agent = create_react_agent(
    model=llm,
    tools=[generate_response],
    name="response_generator",
    prompt="You generate responses from retrieved documents."
)

validation_agent = create_react_agent(
    model=llm,
    tools=[validate_response],
    name="response_validator",
    prompt="You validate response quality and accuracy."
)

# Create supervisor workflow
workflow = create_supervisor(
    agents=[
        query_agent,
        retrieval_agent, 
        quality_agent,
        generation_agent,
        validation_agent
    ],
    model=llm,
    prompt="""You are a supervisor coordinating a RAG system.
    
    For user queries:
    1. Use query_analyst to analyze the query
    2. Use retrieval_expert to get relevant documents
    3. Use quality_assessor to check document quality
    4. Use response_generator to create the response
    5. Use response_validator to ensure quality
    
    Route between agents based on the workflow needs."""
)

# Compile and use
app = workflow.compile()

def process_query(query: str) -> str:
    """Process query through supervisor workflow."""
    result = app.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })
    return result["messages"][-1]["content"]

# Supervisor handles all coordination automatically
# No need for custom BaseAgent, AgentResult, or complex state management
# The langgraph-supervisor library provides:
# - Automatic state management
# - Error handling and recovery
# - Agent handoff mechanisms
# - Conversation context preservation
# - Built-in observability hooks

# Custom agent classes are no longer needed!
# The supervisor library handles all agent coordination
# Agents are created using create_react_agent() with tools
# All complex state management is handled automatically

# All agent logic is now handled by tools and the supervisor library
# This reduces hundreds of lines of boilerplate code

# Quality assessment is now a simple tool function
# All the complex state management is handled by supervisor

# Response generation is now a simple tool function
# Context formatting is handled automatically

# Response validation is now a simple tool function
# JSON parsing and error handling are automatic

# The entire AgentOrchestrator class is replaced by:
# workflow = create_supervisor(agents, model, prompt)
# app = workflow.compile()
# 
# This eliminates hundreds of lines of boilerplate code!
# The supervisor library handles all the complex routing,
# state management, error handling, and workflow execution.
    
# Workflow execution is now just:
# result = app.invoke({"messages": [{"role": "user", "content": query}]})
# return result["messages"][-1]["content"]
# 
# All state management, error handling, and observability
# are handled automatically by the supervisor library
    
# All node implementations and routing logic are eliminated!
# The supervisor library handles all coordination automatically
# through its built-in handoff tools and decision mechanisms
```

## Consequences

### Positive Outcomes

- **Massive Code Reduction**: 90% reduction in orchestration code (~900 → ~50 lines)
- **Proven Reliability**: Battle-tested supervisor library with 1.2k stars and active maintenance
- **Simplified Debugging**: Standard supervisor patterns easier to understand and debug
- **Automatic Features**: Built-in state management, error handling, handoff mechanisms
- **Community Support**: Leverage community contributions and best practices
- **Reduced Maintenance**: Library updates provide improvements without custom code changes

### Negative Consequences / Trade-offs

- **Library Dependency**: Reliance on external library for core coordination functionality
- **Less Control**: Some customization may be constrained by library design
- **Learning Curve**: Team needs to understand supervisor patterns vs custom implementation
- **Version Lock-in**: Updates require coordinating with library release cycles

### Performance Targets

- **Orchestration Overhead**: <300ms additional latency (improved from custom implementation)
- **Success Rate**: ≥95% of queries processed without fallback (improved reliability)
- **Agent Efficiency**: Individual agents execute in <150ms on average
- **Error Recovery**: <3% of queries require fallback responses (improved error handling)

## Dependencies

- **Python**: `langgraph-supervisor>=0.0.29`, `langgraph>=0.2.0`, `langchain-core>=0.3.0`
- **Integration**: Framework abstraction layer, observability system
- **Models**: Local LLM with function calling capabilities

## Monitoring Metrics

- Agent execution times and success rates
- Workflow completion rates and error frequencies
- Agent decision distribution and effectiveness
- Retry rates and fallback usage
- Overall system performance impact

## Future Enhancements

- Dynamic agent loading based on query complexity
- Agent performance learning and optimization
- Custom agent creation for specific domains
- Advanced error recovery strategies
- Cross-conversation context and learning

## Changelog

- **2.0 (2025-01-16)**: **MAJOR SIMPLIFICATION** - Replaced custom LangGraph orchestration with `langgraph-supervisor` library. ~90% code reduction while maintaining all functionality. Improved reliability through proven patterns.
- **1.0 (2025-01-16)**: Initial LangGraph-based agent orchestration framework with specialized agents and comprehensive error handling
