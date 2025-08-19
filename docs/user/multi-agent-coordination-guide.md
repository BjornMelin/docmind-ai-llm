# Multi-Agent Coordination System User Guide

## Overview

The Multi-Agent Coordination System enhances DocMind AI's document analysis capabilities by using specialized AI agents that work together to understand and respond to complex queries. This system automatically analyzes your questions and coordinates the best combination of agents to provide accurate, comprehensive responses.

## What is Multi-Agent Coordination?

Instead of using a single AI model to handle all queries, the multi-agent system breaks down complex questions into specialized tasks and assigns them to expert agents:

- **Router Agent**: Determines the complexity of your query and selects the best processing strategy
- **Planner Agent**: Breaks down complex queries into manageable sub-tasks
- **Retrieval Agent**: Finds the most relevant documents using advanced search strategies
- **Synthesis Agent**: Combines information from multiple sources into coherent responses
- **Validation Agent**: Checks response quality and accuracy before delivery

## How It Works

### Simple Queries

For straightforward questions like "What is the main topic of document X?", the system:

1. **Analyzes** your query complexity (typically <50ms)
2. **Routes** directly to document retrieval
3. **Validates** the response quality
4. **Delivers** the answer (usually within 1.5 seconds)

### Complex Queries

For multi-part questions like "Compare the methodologies in documents A and B, then explain which approach would work better for scenario C", the system:

1. **Analyzes** query complexity and identifies it as complex
2. **Plans** the analysis by breaking it into sub-tasks:
   - Extract methodology from document A
   - Extract methodology from document B
   - Compare the methodologies
   - Evaluate applicability to scenario C
3. **Retrieves** relevant information for each sub-task
4. **Synthesizes** all findings into a comprehensive response
5. **Validates** accuracy and completeness
6. **Delivers** the final answer

## Enabling/Disabling Multi-Agent System

### Environment Configuration

Add these settings to your `.env` file:

```bash
# Enable multi-agent coordination (default: true)
ENABLE_MULTI_AGENT=true

# Individual agent timeouts (default: 30 seconds)
AGENT_TIMEOUT_SECONDS=30

# Maximum context to preserve (default: 65,000 tokens)
MAX_CONTEXT_TOKENS=65000

# Enable DSPy query optimization (default: true)
ENABLE_DSPY_OPTIMIZATION=true

# Fallback strategy when agents fail (default: basic_rag)
FALLBACK_STRATEGY=basic_rag
```

### Runtime Configuration

You can also configure the system programmatically:

```python
from src.agents import MultiAgentCoordinator

# Create coordinator with custom settings
coordinator = MultiAgentCoordinator(
    llm=your_llm,
    tools_data=your_tools,
    config={
        "enable_planning": True,
        "enable_synthesis": True,
        "enable_validation": True,
        "performance_mode": "balanced"  # options: fast, balanced, thorough
    }
)
```

### Disabling Multi-Agent System

To use the traditional single-agent approach:

```bash
# In your .env file
ENABLE_MULTI_AGENT=false
```

Or programmatically:

```python
from src.agents import create_agent

# Create basic agent instead
agent = create_agent(agent_type="basic")
```

## Configuration Options

### Performance Modes

| Mode | Description | Use Case | Speed | Accuracy |
|------|-------------|----------|-------|----------|
| **fast** | Minimal agent coordination, quick responses | Simple Q&A, basic document lookup | Fastest | Good |
| **balanced** | Standard multi-agent processing | Most document analysis tasks | Moderate | Better |
| **thorough** | Full agent pipeline with extensive validation | Complex analysis, research tasks | Slower | Best |

### Advanced Settings

```bash
# Performance tuning
AGENT_CONCURRENCY_LIMIT=5        # Max parallel agent operations
RETRY_ATTEMPTS=3                 # Retry failed operations
CACHE_TTL_SECONDS=300           # Cache results for 5 minutes

# Fallback configuration
FALLBACK_THRESHOLD_MS=3000      # Switch to basic mode after 3 seconds
CONTEXT_PRESERVATION=true       # Maintain conversation context

# Quality settings
MIN_VALIDATION_SCORE=0.7        # Minimum acceptable response quality
ENABLE_HALLUCINATION_CHECK=true # Check for fabricated information
```

## Performance Expectations

### Response Times

| Query Type | Expected Time | Agent Pipeline |
|------------|---------------|----------------|
| Simple lookup | 0.5-1.5 seconds | Router → Retrieval → Validation |
| Medium complexity | 1.5-3 seconds | Router → Planner → Retrieval → Synthesis → Validation |
| Complex analysis | 2-5 seconds | Full multi-agent coordination |
| Fallback mode | <3 seconds | Basic RAG processing |

### Quality Improvements

With multi-agent coordination enabled, you can expect:

- **Better Accuracy**: Specialized agents focus on their strengths
- **More Comprehensive**: Complex queries are properly decomposed
- **Source Attribution**: Clear tracking of information sources
- **Quality Validation**: Automatic checking for errors and inconsistencies

## Understanding System Responses

### Response Structure

Multi-agent responses include additional metadata:

```json
{
  "content": "Your answer here...",
  "sources": ["document1.pdf", "document2.pdf"],
  "validation_score": 0.95,
  "processing_time": 2.3,
  "metadata": {
    "agents_used": ["router", "planner", "retrieval", "synthesis", "validator"],
    "strategy": "complex_decomposition",
    "sub_tasks": 4,
    "fallback_triggered": false
  }
}
```

### Quality Indicators

- **Validation Score** (0.0-1.0): Higher scores indicate better response quality
- **Source Count**: More sources often mean more comprehensive analysis
- **Processing Time**: Indicates system effort and complexity handled
- **Agents Used**: Shows which specialized agents contributed

### System Status Messages

The interface may show status indicators:

- 🔄 **Analyzing query complexity...**
- 📋 **Planning analysis strategy...**
- 🔍 **Retrieving relevant documents...**
- 🔧 **Synthesizing findings...**
- ✅ **Validating response quality...**
- ⚡ **Using fallback mode** (when agents encounter issues)

## Best Practices

### Query Formulation

**For Best Results:**
- Be specific about what you want to analyze
- Mention specific documents when relevant
- Ask follow-up questions to drill down into details

**Examples:**

✅ **Good**: "Compare the data collection methods in the Johnson 2023 study with the Smith 2022 approach, focusing on sample size and bias mitigation."

✅ **Good**: "What are the main findings in document X, and how do they relate to the conclusions in document Y?"

❌ **Avoid**: "Tell me about the documents" (too vague)

❌ **Avoid**: "What does everything say?" (too broad)

### Context Management

- **Multi-turn Conversations**: The system remembers previous questions and builds on context
- **Token Limits**: Very long conversations may need context summarization
- **Context Reset**: Start fresh conversations for unrelated topics

### Performance Optimization

- **Simple First**: Start with basic questions before drilling into complexity
- **Batch Related**: Ask follow-up questions in the same conversation
- **Document Focus**: Specify documents when you know what you're looking for

## Monitoring Your Usage

### Performance Feedback

Pay attention to:
- **Response Times**: Consistently slow responses may indicate system load
- **Validation Scores**: Low scores (<0.7) may indicate unclear queries
- **Fallback Frequency**: Frequent fallbacks suggest configuration issues

### System Health

The system provides health indicators:
- 🟢 **All agents operational**
- 🟡 **Some agents experiencing delays**
- 🔴 **Fallback mode active**

## When to Use Each Mode

### Use Standard Multi-Agent Mode When:
- Analyzing complex documents or datasets
- Comparing multiple sources
- Performing multi-step analysis
- Needing high accuracy and comprehensive coverage

### Use Fast Mode When:
- Simple document lookups
- Quick fact-checking
- Time-sensitive queries
- System resources are limited

### Use Fallback Mode When:
- Multi-agent system is experiencing issues
- Simple queries only
- Debugging configuration problems

## Next Steps

- **Explore**: Try different types of queries to see how the system adapts
- **Monitor**: Watch response times and quality scores to optimize usage
- **Configure**: Adjust settings based on your specific use cases
- **Troubleshoot**: Check the troubleshooting guide if you encounter issues

For technical details and advanced configuration, see the [Multi-Agent Coordination Implementation Guide](../developers/multi-agent-coordination-implementation.md).

For common issues and solutions, see the [Troubleshooting Guide](troubleshooting.md).