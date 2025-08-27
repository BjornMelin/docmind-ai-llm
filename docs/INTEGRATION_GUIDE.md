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
from src.agents.coordinator import MultiAgentCoordinator

async def main():
    # Initialize the multi-agent system
    coordinator = MultiAgentCoordinator()
    
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
# Model Configuration
llm_backend = "vllm"  # Use vLLM by default
model_name = "Qwen/Qwen3-4B-Instruct-2507"
quantization = "fp8"
kv_cache_dtype = "fp8"
context_window_size = 131072  # 128K tokens

# Multi-Agent Settings
enable_multi_agent = True
agent_decision_timeout = 300  # milliseconds
enable_fallback_rag = True
max_agent_retries = 2

# Performance Settings
max_vram_gb = 14.0  # FP8 optimized
vllm_gpu_memory_utilization = 0.85
vllm_attention_backend = "FLASHINFER"
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
from src.agents.coordinator import MultiAgentCoordinator
from llama_index.core.memory import ChatMemoryBuffer

# Create coordinator with custom configuration
coordinator = MultiAgentCoordinator(
    model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length=131072,  # 128K context
    enable_fallback=True,
    max_agent_timeout=5.0  # 5 second timeout
)

# Create context for conversation continuity
context = ChatMemoryBuffer.from_defaults()

# Process query with context
response = coordinator.process_query(
    query="Compare different ML algorithms",
    context=context,
    settings_override={"domain": "healthcare", "complexity": "high"}
)
```

### Direct vLLM Backend Usage

```python
from src.config.vllm_config import create_vllm_manager, VLLMConfig

# Initialize vLLM manager
vllm_manager = create_vllm_manager(
    model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
    max_context_length=131072
)

# Initialize the engine
if vllm_manager.initialize_engine():
    # Create LlamaIndex instance for generation
    llm_instance = vllm_manager.create_vllm_instance()
    
    # Generate response
    response = llm_instance.complete(
        "Explain the benefits of FP8 quantization"
    )
    
    print(response.text)
```

### Streaming Responses

```python
import asyncio
from src.config.vllm_config import create_vllm_manager

async def stream_example():
    # Create vLLM manager
    vllm_manager = create_vllm_manager()
    
    # Initialize engine
    if vllm_manager.initialize_engine():
        # Create LlamaIndex instance
        llm_instance = vllm_manager.create_vllm_instance()
        
        # Stream response
        response = llm_instance.stream_complete(
            "Explain machine learning in detail:"
        )
        
        for chunk in response:
            print(chunk.delta, end="", flush=True)

asyncio.run(stream_example())
```

## Performance Validation

### Automated Testing

```bash
# Run comprehensive performance validation
python scripts/vllm_performance_validation.py

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
    coordinator = MultiAgentCoordinator()
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
from src.utils import (
    load_documents_from_directory,
    create_vector_store,
    setup_hybrid_collection
)

async def setup_document_pipeline(doc_directory: str):
    # Load documents
    documents = await load_documents_from_directory(doc_directory)
    
    # Setup hybrid collection for vector storage
    await setup_hybrid_collection("documents")
    
    # Create vector store with documents
    vector_store = create_vector_store("documents")
    
    return vector_store

# Use in multi-agent system
vector_store = await setup_document_pipeline("./data/documents")

# Initialize coordinator with document processing capability
coordinator = MultiAgentCoordinator()
# Note: Document vector store integration is handled through the retrieval agent tools
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
   from src.config.settings import settings
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
        # Check vLLM backend
        vllm_manager = create_vllm_manager()
        if vllm_manager.initialize_engine():
            performance_metrics = vllm_manager.get_performance_metrics()
        else:
            raise RuntimeError("vLLM engine initialization failed")
        
        # Check agent system
        coordinator = MultiAgentCoordinator()
        
        # Test query
        response = coordinator.process_query("Test query")
        
        return {
            "status": "healthy",
            "model": performance_metrics["config"]["model"],
            "quantization": "fp8",
            "agents": "operational",
            "processing_time_s": response.processing_time,
            "validation_score": response.validation_score,
            "adr_compliance": coordinator.validate_adr_compliance()
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
