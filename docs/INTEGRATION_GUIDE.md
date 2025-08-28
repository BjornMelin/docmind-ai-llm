# DocMind AI Integration Guide

This guide covers the complete integration of all DocMind AI components after the Qwen3-4B-Instruct-2507-FP8 model update and multi-agent system implementation.

## Quick Start

### 1. Environment Setup

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Verify vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# Run environment validation
python scripts/validate_requirements.py
```

### 2. Basic Usage

```python
import asyncio
from src.config import settings
from src.agents.coordinator import MultiAgentCoordinator

async def main():
    # Initialize the multi-agent system with unified configuration
    coordinator = MultiAgentCoordinator(settings)
    
    # Process a query
    response = coordinator.process_query(
        "What are the key benefits of machine learning in healthcare?"
    )
    
    print(f"Response: {response.content}")
    print(f"Validation score: {response.validation_score:.2f}")
    print(f"Processing time: {response.processing_time:.3f}s")

# Run the example
asyncio.run(main())
```

## Architecture Overview

### Core Components

1. **vLLM Backend** (`src/config/vllm_config.py`)
   - Qwen3-4B-Instruct-2507 with FP8 quantization
   - 128K context window support
   - FlashInfer attention optimization
   - Memory-efficient KV cache

2. **Multi-Agent Supervisor** (`src/agents/coordinator.py`)
   - LangGraph-based workflow coordination
   - 5-agent pipeline: Router → Planner → Retrieval → Synthesis → Validator
   - Error handling and fallback mechanisms
   - <300ms coordination latency

3. **Individual Agents**
   - **Router**: Query analysis and strategy selection
   - **Planner**: Complex query decomposition
   - **Retrieval**: Multi-strategy document retrieval
   - **Synthesis**: Result combination and deduplication
   - **Validator**: Quality assurance and validation

### Performance Specifications

- **Model Performance**: 100-160 tok/s decode, 800-1300 tok/s prefill
- **Memory Constraints**: <4GB RAM, <16GB VRAM
- **Context Window**: 131,072 tokens (128K)
- **Agent Coordination**: <300ms end-to-end
- **Quantization**: FP8 weights + FP8 KV cache

## Configuration

### Settings Overview

Key configuration options in `src/config/settings.py`:

```python
# Import unified settings
from src.config import settings

# Access configuration values
print(f"LLM Backend: {settings.llm_backend}")
print(f"Model: {settings.vllm.model}")
print(f"Quantization: {settings.vllm.kv_cache_dtype}")
print(f"Context Window: {settings.vllm.context_window}")
print(f"Multi-Agent Enabled: {settings.agents.enable_multi_agent}")
print(f"GPU Memory Util: {settings.vllm.gpu_memory_utilization}")
```

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Core Configuration
DOCMIND_LLM_BACKEND=vllm
DOCMIND_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
DOCMIND_QUANTIZATION=fp8
DOCMIND_KV_CACHE_DTYPE=fp8

# Performance Settings
DOCMIND_MAX_VRAM_GB=14.0
DOCMIND_VLLM_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_AGENT_DECISION_TIMEOUT=300

# Multi-Agent Settings
DOCMIND_ENABLE_MULTI_AGENT=true
DOCMIND_ENABLE_FALLBACK_RAG=true
DOCMIND_MAX_AGENT_RETRIES=2
```

## Advanced Usage

### Custom Agent Configuration

```python
from src.config import settings
from src.agents.coordinator import MultiAgentCoordinator
from llama_index.core.memory import ChatMemoryBuffer
import os

# Override settings via environment variables
os.environ["DOCMIND_MODEL_NAME"] = "Qwen/Qwen3-4B-Instruct-2507-FP8"
os.environ["DOCMIND_CONTEXT_WINDOW_SIZE"] = "131072"
os.environ["DOCMIND_ENABLE_FALLBACK_RAG"] = "true"
os.environ["DOCMIND_AGENT_DECISION_TIMEOUT"] = "5000"  # 5 seconds in ms

# Create coordinator with unified configuration
coordinator = MultiAgentCoordinator(settings)

# Create context for conversation continuity
context = ChatMemoryBuffer.from_defaults()

# Process query with context
response = coordinator.process_query(
    query="Compare different ML algorithms",
    context=context
)
```

### Direct vLLM Backend Usage

```python
from src.config import settings
from src.retrieval.query_engine import get_query_engine

# Initialize query engine with unified configuration
query_engine = get_query_engine(settings)

# Generate response using the configured backend
response = query_engine.query(
    "Explain the benefits of FP8 quantization"
)

print(response.response)
```

### Streaming Responses

```python
import asyncio
from src.config import settings
from src.retrieval.query_engine import get_streaming_query_engine

async def stream_example():
    # Create streaming query engine with unified configuration
    query_engine = get_streaming_query_engine(settings)
    
    # Stream response
    response_stream = query_engine.query_stream(
        "Explain machine learning in detail:"
    )
    
    for chunk in response_stream:
        print(chunk.response_delta, end="", flush=True)

asyncio.run(stream_example())
```

## Performance Validation

### Automated Testing

```bash
# Run comprehensive performance validation
python scripts/performance_validation.py

# Run end-to-end integration test
python scripts/end_to_end_test.py

# Validate all 100 requirements
python scripts/validate_requirements.py
```

### Performance Monitoring

```python
from src.utils.monitoring import get_performance_monitor, get_memory_usage

# Monitor system resources
monitor = get_performance_monitor()
ram_gb, vram_gb = get_memory_usage()

print(f"RAM: {ram_gb:.2f}GB, VRAM: {vram_gb:.2f}GB")
```

## Error Handling

### Graceful Degradation

The system includes multiple fallback mechanisms:

1. **Agent Failures**: Automatic retry with exponential backoff
2. **Model Failures**: Fallback to basic RAG pipeline
3. **Memory Issues**: Dynamic batch size adjustment
4. **Network Issues**: Local-first operation continues

### Error Recovery

```python
try:
    response = await supervisor.process_query(query)
    if response.get("error_occurred"):
        print(f"Error: {response['error_message']}")
        # Implement custom error handling
except Exception as e:
    print(f"System error: {e}")
    # Fallback to basic processing
```

## Integration Patterns

### Batch Processing

```python
async def process_batch(queries: list[str]):
    coordinator = MultiAgentCoordinator(settings)
    results = []
    
    for query in queries:
        response = coordinator.process_query(query)
        results.append(response)
    
    return results

# Process multiple queries
queries = [
    "What is machine learning?",
    "How does neural networks work?", 
    "Compare supervised vs unsupervised learning"
]

results = await process_batch(queries)
```

### Document Processing Pipeline

```python
from src.config import settings
from src.utils.document import load_documents_unstructured
from src.utils.embedding import create_index_async
from src.agents.coordinator import MultiAgentCoordinator
from pathlib import Path

async def setup_document_pipeline(doc_directory: str):
    # Load documents using unified configuration
    doc_paths = list(Path(doc_directory).rglob("*.*"))
    documents = await load_documents_unstructured(doc_paths, settings)
    
    # Create vector index
    index = await create_index_async(documents, settings)
    
    return index

# Use in multi-agent system
index = await setup_document_pipeline("./data/documents")

# Initialize coordinator with unified configuration
coordinator = MultiAgentCoordinator(settings)
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**

   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Verify compute capability (need >=8.0 for FP8)
   python -c "import torch; print(torch.cuda.get_device_capability())"
   ```

2. **Memory Issues**

   ```python
   # Reduce GPU memory utilization
   from src.config import settings
   settings.vllm_gpu_memory_utilization = 0.7  # Reduce from 0.85
   ```

3. **Performance Issues**

   ```bash
   # Check attention backend
   python -c "import os; print(os.environ.get('VLLM_ATTENTION_BACKEND', 'Not set'))"
   
   # Should output: FLASHINFER
   ```

4. **Agent Coordination Issues**

   ```python
   # Enable debug logging
   import logging
   logging.getLogger('src.agents').setLevel(logging.DEBUG)
   ```

### Performance Tuning

1. **Memory Optimization**
   - Use FP8 quantization: `quantization = "fp8"`
   - Enable FP8 KV cache: `kv_cache_dtype = "fp8"`
   - Adjust GPU memory utilization: `gpu_memory_utilization = 0.85`

2. **Latency Optimization**  
   - Enable chunked prefill: `enable_chunked_prefill = True`
   - Use FlashInfer attention: `attention_backend = "FLASHINFER"`
   - Optimize batch sizes: `max_num_batched_tokens = 8192`

3. **Context Optimization**
   - Use appropriate context window: `max_model_len = 131072`
   - Enable conversation memory: `enable_conversation_memory = True`
   - Configure context buffer: `context_buffer_size = 131072`

## Production Deployment

### Resource Requirements

**Minimum System Requirements:**

- GPU: RTX 4090 or equivalent (16GB VRAM)
- RAM: 32GB system memory
- Storage: 50GB available space
- CUDA: 12.0 or later

**Recommended Configuration:**

- GPU: RTX 4090 24GB or H100
- RAM: 64GB system memory  
- Storage: 100GB SSD
- Network: High-bandwidth for model downloads

### Monitoring and Logging

```python
from src.utils import setup_logging, log_performance

# Configure logging
setup_logging()

# Log performance metrics
@log_performance
async def tracked_query(query: str):
    coordinator = MultiAgentCoordinator()
    return coordinator.process_query(query)
```

### Health Checks

```python
async def health_check():
    try:
        # Check system configuration
        from src.config import settings
        
        # Check agent system
        coordinator = MultiAgentCoordinator(settings)
        
        # Test query
        response = coordinator.process_query("Test query")
        
        return {
            "status": "healthy",
            "model": settings.vllm.model,
            "quantization": settings.vllm.kv_cache_dtype,
            "agents": "operational",
            "processing_time_s": response.processing_time,
            "validation_score": response.validation_score,
            "configuration": "unified"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Next Steps

1. **Extend Agent Capabilities**: Add domain-specific agents
2. **Custom Model Support**: Integrate additional models
3. **Advanced Analytics**: Implement usage analytics
4. **API Integration**: Add REST/GraphQL APIs
5. **UI Enhancement**: Develop advanced user interfaces

For additional support and documentation, see:

- [Architecture Overview](./adrs/ARCHITECTURE-OVERVIEW.md)
- [Performance Optimization](./adrs/ADR-010-performance-optimization-strategy.md)
- [Multi-Agent Coordination](./specs/001-multi-agent-coordination.spec.md)
- [API Documentation](./api/)

## Conclusion

DocMind AI now provides a complete, high-performance document analysis system with:

- ✅ FP8-quantized Qwen3-4B-Instruct-2507 model
- ✅ Multi-agent coordination with <300ms latency
- ✅ 128K context window support
- ✅ Memory-efficient operation (<16GB VRAM)
- ✅ Comprehensive error handling and fallbacks
- ✅ Full requirements compliance (100/100)

The system is production-ready and optimized for RTX 4090 hardware with enterprise-grade reliability and performance.
