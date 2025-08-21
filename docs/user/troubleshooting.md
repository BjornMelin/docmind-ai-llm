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

## Model-Specific Troubleshooting

### 11. Qwen3-4B-Instruct-2507-FP8 Model Issues

#### Model Loading Problems
- **Symptoms**: "Model failed to load" or CUDA errors during initialization.
- **Solution**:
  ```bash
  # Check CUDA compatibility (need 12.8+ for PyTorch 2.7.1)
  nvidia-smi
  nvcc --version
  
  # Verify PyTorch installation
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
  
  # Reinstall with correct CUDA version if needed
  uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128
  ```

#### FP8 Quantization Issues
- **Symptoms**: Model loads but performance is poor or VRAM usage is high.
- **Solution**:
  ```bash
  # Check FP8 support
  python -c "
  import torch
  compute_cap = torch.cuda.get_device_capability()
  print(f'GPU Compute Capability: {compute_cap[0]}.{compute_cap[1]}')
  print(f'FP8 Support: {\"Optimal\" if compute_cap[0] >= 8 else \"Limited\"}')
  "
  
  # Verify environment variables
  echo "VLLM_QUANTIZATION: $VLLM_QUANTIZATION"
  echo "VLLM_KV_CACHE_DTYPE: $VLLM_KV_CACHE_DTYPE"
  ```

#### vLLM FlashInfer Backend Problems  
- **Symptoms**: "FlashInfer not available" or slower than expected performance.
- **Solution**:
  ```bash
  # Verify FlashInfer installation
  python -c "
  import vllm
  print(f'vLLM version: {vllm.__version__}')
  print(f'FlashInfer backend: {\"Available\" if \"flashinfer\" in str(vllm.__version__) else \"Not Available\"}')
  "
  
  # Reinstall vLLM with FlashInfer if needed
  uv pip install "vllm[flashinfer]>=0.10.1" --extra-index-url https://download.pytorch.org/whl/cu128
  
  # Check environment configuration
  export VLLM_ATTENTION_BACKEND=FLASHINFER
  export VLLM_USE_CUDNN_PREFILL=1
  ```

#### 128K Context Window Issues
- **Symptoms**: Context overflow despite large context window, or poor performance with large contexts.
- **Solution**:
  ```bash
  # Check context configuration
  echo "DOCMIND_CONTEXT_LENGTH: $DOCMIND_CONTEXT_LENGTH"  # Should be 131072
  
  # Monitor context utilization
  python -c "
  from docs.developers.model_update_implementation import Qwen3ContextManager
  manager = Qwen3ContextManager()
  print(f'Max context: {manager.max_context_tokens}')
  print(f'Safe context: {manager.safe_context_tokens}')
  "
  ```

#### Performance Below Targets
- **Symptoms**: Decode speed < 100 tok/s or prefill speed < 800 tok/s.
- **Solution**:
  ```bash
  # Run performance validation
  python scripts/performance_validation.py
  
  # Check optimization settings
  export VLLM_ATTENTION_BACKEND=FLASHINFER
  export VLLM_USE_CUDNN_PREFILL=1
  export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
  
  # Verify GPU memory utilization
  echo "VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
  ```

#### Memory Issues with RTX 4090 (16GB)
- **Symptoms**: CUDA out of memory errors or VRAM usage > 16GB.
- **Solution**:
  ```bash
  # Reduce GPU memory utilization
  export VLLM_GPU_MEMORY_UTILIZATION=0.75  # Reduce from 0.85
  
  # Monitor GPU memory
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1
  
  # Clear GPU cache if needed
  python -c "import torch; torch.cuda.empty_cache()"
  
  # Verify FP8 optimizations are enabled
  python -c "
  import os
  print('FP8 Quantization:', os.environ.get('VLLM_QUANTIZATION', 'Not Set'))
  print('FP8 KV Cache:', os.environ.get('VLLM_KV_CACHE_DTYPE', 'Not Set'))
  "
  ```

### 12. vLLM Backend Troubleshooting

#### vLLM Server Connection Issues
- **Symptoms**: "Connection refused" to vLLM server or server won't start.
- **Solution**:
  ```bash
  # Check if vLLM server is running
  curl http://localhost:8000/health
  
  # Start vLLM server manually for debugging
  vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072 \
    --quantization fp8 \
    --kv-cache-dtype fp8_e5m2
  ```

#### Model Download/Loading Issues
- **Symptoms**: "Model not found" or download failures.
- **Solution**:
  ```bash
  # Download model manually (if available)
  huggingface-cli download Qwen/Qwen3-4B-Instruct-2507-FP8
  
  # Check available models
  ls -la ~/.cache/huggingface/hub/
  
  # Verify model path in configuration
  python -c "
  from models import AppSettings
  settings = AppSettings()
  print(f'Configured model: {settings.default_model}')
  "
  ```

## Performance Monitoring

### Model Performance Metrics

Monitor these metrics for the Qwen3-4B model:

- **Decode Speed**: Target 100-160 tok/s (measure with benchmark script)
- **Prefill Speed**: Target 800-1300 tok/s (measure during long context processing)
- **VRAM Usage**: Target 12-14GB (monitor with nvidia-smi)
- **Context Utilization**: Up to 128K tokens (track in logs)
- **FP8 Memory Savings**: ~50% vs FP16 (compare memory usage)

### Multi-Agent Performance Metrics

Check these metrics to identify performance issues:

- **Agent Timing**: Individual agent response times  
- **Coordination Overhead**: Total multi-agent coordination time
- **Validation Scores**: Response quality indicators (0.0-1.0)
- **Fallback Rate**: Frequency of fallback to basic RAG
- **Context Usage**: Token consumption per conversation
- **Cache Hit Rate**: Effectiveness of result caching

### Performance Validation Steps

Run these commands to validate Qwen3-4B model performance:

```bash
# 1. Quick environment validation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else \"N/A\")
"

# 2. vLLM installation check
python -c "
import vllm
print(f'vLLM Version: {vllm.__version__}')
"

# 3. FlashInfer backend validation
python -c "
import os
print(f'FlashInfer Backend: {os.environ.get(\"VLLM_ATTENTION_BACKEND\", \"Not Set\")}')
print(f'cuDNN Prefill: {os.environ.get(\"VLLM_USE_CUDNN_PREFILL\", \"Not Set\")}')
"

# 4. Model configuration check
python -c "
from models import AppSettings
settings = AppSettings()
print(f'Model: {settings.default_model}')
print(f'Context Length: {getattr(settings, \"context_length\", \"Not Set\")}')
"

# 5. Full performance benchmark (if available)
python scripts/performance_validation.py
```

Expected results for optimal configuration:
- PyTorch: 2.7.1+cu128
- vLLM: >=0.10.1
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- VRAM: 16.0GB
- FlashInfer Backend: FLASHINFER
- Model: Qwen/Qwen3-4B-Instruct-2507-FP8

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
