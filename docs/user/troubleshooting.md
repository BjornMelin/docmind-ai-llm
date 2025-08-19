# Troubleshooting DocMind AI

This guide helps resolve common issues when using DocMind AI.

## Common Issues and Solutions

### 1. Ollama Not Running

- **Symptoms**: "Connection refused" or "Cannot connect to Ollama" errors.
- **Solution**:
  1. Ensure Ollama is installed: `ollama --version`.
  2. Start Ollama: `ollama serve`.
  3. Verify URL: Default is `http://localhost:11434`. Update in sidebar if different.
  4. Check logs: `logs/app.log` for errors.

### 2. Dependency Installation Fails

- **Symptoms**: `uv sync` errors or missing packages.
- **Solution**:
  - Ensure Python 3.9+: `python --version`.
  - Update uv: `pip install -U uv`.
  - Retry: `uv sync` or `uv sync --extra gpu` for GPU support.
  - Check for conflicts: Use `uv pip list` to inspect installed versions.

### 3. GPU Not Detected

- **Symptoms**: GPU toggle ineffective; slow performance.
- **Solution**:
  - Verify NVIDIA drivers: `nvidia-smi`.
  - Install GPU dependencies: `uv sync --extra gpu`.
  - Ensure CUDA compatibility: Requires CUDA 12.x for FastEmbed.
  - Check VRAM: `utils.py:detect_hardware()` logs available VRAM.

### 4. Unsupported File Formats

- **Symptoms**: "Unsupported file type" error.
- **Solution**:
  - Supported formats: PDF, DOCX, TXT, XLSX, CSV, JSON, XML, MD, RTF, MSG, PPTX, ODT, EPUB, code files.
  - Convert unsupported files (e.g., to PDF) before uploading.
  - Log issue on GitHub for new format support.

### 5. Analysis Errors

- **Symptoms**: "Error parsing output" or incomplete results.
- **Solution**:
  - Check document size: Enable chunking for large files.
  - Verify context size: Adjust in sidebar (e.g., 8192 for larger models).
  - Review raw output in `logs/app.log`.

### 6. Chat Interface Issues

- **Symptoms**: Irrelevant responses or retrieval failures.
- **Solution**:
  - Ensure vectorstore is created: Re-upload documents if needed.
  - Check hybrid search settings: Toggle multi-vector embeddings.
  - Review Qdrant logs: Ensure `QdrantClient` is running (`:memory:` mode).

## Multi-Agent System Troubleshooting

### 7. Multi-Agent Coordination Issues

#### Agent Timeout Problems
- **Symptoms**: "Agent timeout" errors, slow responses, fallback mode activated.
- **Solution**:
  - Check agent timeout settings: `AGENT_TIMEOUT_SECONDS=30` in `.env`.
  - Monitor system resources: High CPU/memory usage can cause timeouts.
  - Review agent performance in logs: Look for individual agent timing.
  - Consider performance mode: Switch to `fast` mode for simpler queries.

#### Frequent Fallback to Basic RAG
- **Symptoms**: System consistently uses fallback mode instead of multi-agent coordination.
- **Solution**:
  - Check multi-agent enable flag: `ENABLE_MULTI_AGENT=true` in `.env`.
  - Verify LangGraph dependencies: Ensure `langgraph` package is installed.
  - Review fallback threshold: `FALLBACK_THRESHOLD_MS=3000` may be too low.
  - Check agent initialization: Look for startup errors in logs.

#### Poor Response Quality with Multi-Agent System
- **Symptoms**: Low validation scores, incomplete responses, inconsistent results.
- **Solution**:
  - Enable DSPy optimization: `ENABLE_DSPY_OPTIMIZATION=true`.
  - Increase validation threshold: `MIN_VALIDATION_SCORE=0.7`.
  - Check agent coordination logs: Look for synthesis or validation issues.
  - Verify context preservation: `CONTEXT_PRESERVATION=true`.

#### Memory/Performance Issues
- **Symptoms**: High memory usage, slow performance, system crashes.
- **Solution**:
  - Adjust context limits: `MAX_CONTEXT_TOKENS=65000` or lower.
  - Reduce concurrency: `AGENT_CONCURRENCY_LIMIT=3` instead of 5.
  - Enable caching: `CACHE_TTL_SECONDS=300` for repeated queries.
  - Monitor agent overhead: Should be <300ms total.

### 8. Configuration Problems

#### Multi-Agent System Not Starting
- **Symptoms**: System defaults to basic agent, no multi-agent coordination.
- **Solution**:
  - Verify `.env` configuration:
    ```bash
    ENABLE_MULTI_AGENT=true
    AGENT_TIMEOUT_SECONDS=30
    MAX_CONTEXT_TOKENS=65000
    ```
  - Check Python imports: Ensure `from src.agents import MultiAgentCoordinator` works.
  - Review initialization logs: Look for agent creation errors.

#### Invalid Agent Configuration
- **Symptoms**: "Invalid agent configuration" or "Agent not found" errors.
- **Solution**:
  - Check agent types: Valid options are `router`, `planner`, `retrieval`, `synthesis`, `validator`.
  - Verify tool availability: Agents need access to document tools and LLM.
  - Review performance mode: Use `fast`, `balanced`, or `thorough`.

### 9. Context and Memory Issues

#### Context Not Preserved Across Conversations
- **Symptoms**: Agents don't remember previous questions or context.
- **Solution**:
  - Enable context preservation: `CONTEXT_PRESERVATION=true`.
  - Check token limits: Reduce `MAX_CONTEXT_TOKENS` if hitting limits.
  - Review memory initialization: Ensure `ChatMemoryBuffer` is properly configured.
  - Monitor context truncation: Look for "context truncated" messages.

#### Context Overflow Errors
- **Symptoms**: "Context too long" or "Token limit exceeded" errors.
- **Solution**:
  - Reduce context window: Lower `MAX_CONTEXT_TOKENS` to 32000 or less.
  - Enable automatic truncation: System should handle this automatically.
  - Use context summarization: Consider shorter conversation threads.
  - Check document sizes: Very large documents may need chunking.

### 10. Development and Testing Issues

#### Tests Failing for Multi-Agent System
- **Symptoms**: `pytest tests/test_agents/` failures.
- **Solution**:
  - Check test dependencies: Ensure `pytest-asyncio` is installed.
  - Run specific tests: `pytest tests/test_agents/test_multi_agent_coordination_spec.py -v`.
  - Review mock configuration: Tests use deterministic mocks.
  - Check async compatibility: Use `@pytest.mark.asyncio` for async tests.

#### Agent Development and Debugging
- **Symptoms**: Need to debug agent behavior or add custom agents.
- **Solution**:
  - Enable debug logging: Set log level to DEBUG in configuration.
  - Use agent demo: Run `python src/agents/demo.py` for testing.
  - Review agent interfaces: Check `@tool` function signatures.
  - Monitor state flow: Use LangGraph debugger if available.

## Performance Monitoring

### Multi-Agent Performance Metrics

Check these metrics to identify performance issues:

- **Agent Timing**: Individual agent response times
- **Coordination Overhead**: Total multi-agent coordination time
- **Validation Scores**: Response quality indicators (0.0-1.0)
- **Fallback Rate**: Frequency of fallback to basic RAG
- **Context Usage**: Token consumption per conversation
- **Cache Hit Rate**: Effectiveness of result caching

### Log Analysis for Multi-Agent Issues

Key log patterns to watch for:

```bash
# Agent timing issues
grep "agent_timeout" logs/app.log

# Fallback activations  
grep "fallback_triggered" logs/app.log

# Validation problems
grep "validation_score.*0\.[0-6]" logs/app.log

# Context management
grep "context_truncated" logs/app.log

# Performance monitoring
grep "coordination_complete" logs/app.log
```

## Environment Configuration Reference

### Complete Multi-Agent Configuration

```bash
# Core multi-agent settings
ENABLE_MULTI_AGENT=true
AGENT_TIMEOUT_SECONDS=30
MAX_CONTEXT_TOKENS=65000
ENABLE_DSPY_OPTIMIZATION=true
FALLBACK_STRATEGY=basic_rag

# Performance tuning
AGENT_CONCURRENCY_LIMIT=5
RETRY_ATTEMPTS=3
CACHE_TTL_SECONDS=300
FALLBACK_THRESHOLD_MS=3000
CONTEXT_PRESERVATION=true

# Quality settings
MIN_VALIDATION_SCORE=0.7
ENABLE_HALLUCINATION_CHECK=true
```

## Getting Help

- Check `logs/app.log` for detailed errors.
- Search or open issues on [GitHub](https://github.com/BjornMelin/docmind-ai).
- Include: Steps to reproduce, logs, system details (OS, Python version, GPU).
