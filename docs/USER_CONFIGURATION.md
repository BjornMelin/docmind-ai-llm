# User Configuration Guide

## Overview

DocMind AI is designed as a **local-first AI application** that adapts to your hardware and privacy preferences. Whether you're running on a student laptop or a high-end research workstation, DocMind AI provides flexible configuration options to maximize performance on your specific setup.

## Quick Start User Scenarios

### Scenario 1: Student/Budget Setup (CPU-Only)

Perfect for older laptops or privacy-focused users who prefer CPU-only operation:

```bash
# .env configuration
DOCMIND_ENABLE_GPU_ACCELERATION=false
DOCMIND_LLM_BACKEND=ollama
DOCMIND_CONTEXT_WINDOW_SIZE=4096
DOCMIND_MAX_MEMORY_GB=4.0
DOCMIND_PERFORMANCE_TIER=low
```

**Hardware Requirements**: 8GB RAM, any CPU  
**Expected Performance**: 10-20 seconds per query, basic document analysis  
**Privacy Level**: Complete offline operation

### Scenario 2: Mid-Range Gaming Setup (RTX 3060/4060)

Good balance of performance and accessibility:

```bash
# .env configuration  
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_LLM_BACKEND=vllm
DOCMIND_CONTEXT_WINDOW_SIZE=32768
DOCMIND_MAX_VRAM_GB=12.0
DOCMIND_MAX_MEMORY_GB=16.0
DOCMIND_PERFORMANCE_TIER=medium
```

**Hardware Requirements**: RTX 3060/4060 (12GB VRAM), 16GB RAM  
**Expected Performance**: 3-7 seconds per query, good document analysis  
**Privacy Level**: Complete offline operation with GPU acceleration

### Scenario 3: High-End Research Setup (RTX 4090)

Maximum performance for professional research and complex documents:

```bash
# .env configuration
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_LLM_BACKEND=vllm
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_MAX_VRAM_GB=24.0
DOCMIND_MAX_MEMORY_GB=32.0
DOCMIND_PERFORMANCE_TIER=high
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true
```

**Hardware Requirements**: RTX 4090 (24GB VRAM), 32GB RAM  
**Expected Performance**: 1-3 seconds per query, advanced analysis with 128K context  
**Privacy Level**: Complete offline operation with maximum features

### Scenario 4: Privacy-Focused Professional

Maximum privacy with local models and no performance logging:

```bash
# .env configuration
DOCMIND_LLM_BACKEND=ollama
DOCMIND_LOCAL_MODEL_PATH=/home/user/models/
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CONTEXT_WINDOW_SIZE=65536
```

**Hardware Requirements**: RTX 4070+ (16GB VRAM), 16GB RAM  
**Expected Performance**: 2-5 seconds per query, good analysis  
**Privacy Level**: Maximum privacy with local model storage

### Scenario 5: Development/Testing Setup

Optimized for fast iteration during development:

```bash
# .env configuration
DOCMIND_DEBUG=true
DOCMIND_LOG_LEVEL=DEBUG
DOCMIND_CONTEXT_WINDOW_SIZE=8192
DOCMIND_ENABLE_DOCUMENT_CACHING=false
DOCMIND_AGENT_DECISION_TIMEOUT=100
```

**Hardware Requirements**: Any modern setup  
**Expected Performance**: Fast startup, detailed logging  
**Privacy Level**: Development-focused with debug information

## Hardware Configuration Options

### GPU Acceleration Control

DocMind AI automatically detects your hardware but respects your preferences:

```bash
# Enable/disable GPU acceleration
DOCMIND_ENABLE_GPU_ACCELERATION=true    # Use GPU if available
DOCMIND_ENABLE_GPU_ACCELERATION=false   # Force CPU-only operation

# Automatic device selection
DOCMIND_DEVICE=auto    # Automatically detect best device
DOCMIND_DEVICE=cuda    # Force CUDA GPU usage
DOCMIND_DEVICE=cpu     # Force CPU usage
```

### Memory Limits

Protect your system by setting memory limits based on your hardware:

```bash
# System memory limits (protects against OOM)
DOCMIND_MAX_MEMORY_GB=4.0     # Conservative for 8GB system
DOCMIND_MAX_MEMORY_GB=8.0     # Good for 16GB system  
DOCMIND_MAX_MEMORY_GB=16.0    # High-end 32GB system

# GPU memory limits (prevents VRAM overflow)
DOCMIND_MAX_VRAM_GB=8.0       # RTX 3060 Ti
DOCMIND_MAX_VRAM_GB=12.0      # RTX 3060/4060
DOCMIND_MAX_VRAM_GB=16.0      # RTX 4070/4080
DOCMIND_MAX_VRAM_GB=24.0      # RTX 4090
```

### Performance Tiers

DocMind AI automatically optimizes based on your hardware:

```bash
# Automatic performance scaling
DOCMIND_PERFORMANCE_TIER=auto     # Auto-detect optimal settings
DOCMIND_PERFORMANCE_TIER=low      # Optimize for modest hardware
DOCMIND_PERFORMANCE_TIER=medium   # Balance performance and resources
DOCMIND_PERFORMANCE_TIER=high     # Maximum performance for capable hardware
```

## LLM Backend Configuration

### Ollama Backend (Default - Privacy-Focused)

Best for users who prioritize privacy and offline operation:

```bash
DOCMIND_LLM_BACKEND=ollama
DOCMIND_OLLAMA_BASE_URL=http://localhost:11434
DOCMIND_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507-FP8
```

**Benefits**: Complete privacy, easy model management, no internet required  
**Use Cases**: Personal research, sensitive documents, air-gapped systems

### vLLM Backend (High Performance)

Best for users with capable hardware who want maximum speed:

```bash
DOCMIND_LLM_BACKEND=vllm
DOCMIND_VLLM_BASE_URL=http://localhost:8000
DOCMIND_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507-FP8

# vLLM Performance Optimization
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
VLLM_GPU_MEMORY_UTILIZATION=0.85
```

**Benefits**: 2-3x faster inference, FP8 optimization, advanced caching  
**Use Cases**: High-throughput analysis, real-time applications

### OpenAI-Compatible Endpoints

For users who want to use custom endpoints or other providers:

```bash
DOCMIND_LLM_BACKEND=openai
DOCMIND_OPENAI_BASE_URL=http://localhost:8080
# Or use external service: https://api.openai.com/v1
```

**Benefits**: Flexibility, can use any OpenAI-compatible service  
**Use Cases**: Custom model servers, specific provider requirements

### llama.cpp Backend (CPU Optimized)

Best for CPU-only setups or specific quantization needs:

```bash
DOCMIND_LLM_BACKEND=llama_cpp
DOCMIND_LLM_BASE_URL=http://localhost:8080
```

**Benefits**: Excellent CPU performance, custom quantization options  
**Use Cases**: CPU-only systems, specific quantization requirements

## Context Window Configuration

### Conservative Settings (Broad Compatibility)

For modest hardware or when running multiple applications:

```bash
DOCMIND_CONTEXT_WINDOW_SIZE=4096   # 4K tokens - works on any system
DOCMIND_CONTEXT_WINDOW_SIZE=8192   # 8K tokens - good balance
```

### Moderate Settings (Good Performance)

For mid-range systems with dedicated GPU:

```bash
DOCMIND_CONTEXT_WINDOW_SIZE=32768  # 32K tokens - good for most documents
DOCMIND_CONTEXT_WINDOW_SIZE=65536  # 64K tokens - handles large documents
```

### High-End Settings (Maximum Capability)

For RTX 4080/4090 and extensive document analysis:

```bash
DOCMIND_CONTEXT_WINDOW_SIZE=131072 # 128K tokens - maximum context
```

**Memory Impact**: Higher context windows require more VRAM:

- 4K context: ~2GB additional VRAM
- 32K context: ~6GB additional VRAM  
- 128K context: ~12GB additional VRAM

## Privacy and Security Configuration

### Maximum Privacy Setup

For users who prioritize complete privacy and offline operation:

```bash
# Disable all external connections
DOCMIND_LLM_BACKEND=ollama
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false

# Use local model storage
DOCMIND_LOCAL_MODEL_PATH=/home/user/.local/share/docmind/models/

# Disable caching if desired
DOCMIND_ENABLE_DOCUMENT_CACHING=false
```

### Offline-First Configuration

Ensure DocMind AI works without internet access:

```bash
# Local model storage
DOCMIND_LOCAL_MODEL_PATH=/opt/models/

# Disable external model downloads
DOCMIND_EMBEDDING_MODEL=BAAI/bge-m3  # Must be pre-downloaded

# Local vector database
DOCMIND_VECTOR_STORE_TYPE=qdrant
DOCMIND_QDRANT_URL=http://localhost:6333
```

## Advanced Feature Configuration

### Multi-Agent System

Control the intelligent agent coordination system:

```bash
# Enable advanced multi-agent features
DOCMIND_ENABLE_MULTI_AGENT=true
DOCMIND_MAX_CONCURRENT_AGENTS=3
DOCMIND_AGENT_DECISION_TIMEOUT=200
DOCMIND_ENABLE_FALLBACK_RAG=true
```

### Document Processing

Optimize for your document types and performance needs:

```bash
# Conservative settings for modest hardware
DOCMIND_CHUNK_SIZE=1024
DOCMIND_CHUNK_OVERLAP=100
DOCMIND_MAX_DOCUMENT_SIZE_MB=25

# Aggressive settings for powerful hardware
DOCMIND_CHUNK_SIZE=2048
DOCMIND_CHUNK_OVERLAP=200
DOCMIND_MAX_DOCUMENT_SIZE_MB=200
```

### Advanced AI Features

Enable experimental features for advanced users:

```bash
# DSPy query optimization (20-30% quality improvement)
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true

# GraphRAG for relationship mapping
DOCMIND_ENABLE_GRAPHRAG=true
```

## Environment Configuration

### Development Setup

```bash
DOCMIND_DEBUG=true
DOCMIND_LOG_LEVEL=DEBUG
DOCMIND_ENABLE_PERFORMANCE_LOGGING=true
```

### Production Setup

```bash
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=INFO
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false
```

## Configuration Validation

### Verify Your Setup

Use DocMind AI's built-in validation to ensure optimal configuration:

```bash
# Check hardware compatibility
uv run python -c "from src.config import settings; print(settings.get_user_hardware_info())"

# Validate configuration
uv run python -c "from src.config import settings; print(settings.get_user_scenario_config())"

# Test performance
uv run python scripts/performance_validation.py
```

### Expected Output Examples

**GPU Setup Validation**:

```json
{
  "enable_gpu_acceleration": true,
  "device": "cuda",
  "performance_tier": "high",
  "max_memory_gb": 32.0,
  "max_vram_gb": 24.0,
  "embedding_device": "cuda",
  "embedding_batch_size": 12,
  "backend_url": "http://localhost:8000"
}
```

**User Scenario Summary**:

```json
{
  "scenario": "GPU with 32.0GB RAM, 24.0GB VRAM",
  "backend": "VLLM backend", 
  "performance": "high tier, 128K context",
  "features": "Multi-agent: True, Caching: True"
}
```

## Common Configuration Patterns

### Pattern 1: Maximize Privacy

```bash
DOCMIND_LLM_BACKEND=ollama
DOCMIND_LOCAL_MODEL_PATH=/private/models/
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false
DOCMIND_ENABLE_DOCUMENT_CACHING=false
DOCMIND_LOG_LEVEL=ERROR
```

### Pattern 2: Maximize Performance  

```bash
DOCMIND_LLM_BACKEND=vllm
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true
DOCMIND_PERFORMANCE_TIER=high
VLLM_ATTENTION_BACKEND=FLASHINFER
```

### Pattern 3: Balance Performance and Resources

```bash
DOCMIND_LLM_BACKEND=ollama
DOCMIND_CONTEXT_WINDOW_SIZE=32768
DOCMIND_MAX_VRAM_GB=12.0
DOCMIND_PERFORMANCE_TIER=medium
DOCMIND_ENABLE_GPU_ACCELERATION=true
```

### Pattern 4: Development and Testing

```bash
DOCMIND_DEBUG=true
DOCMIND_LOG_LEVEL=DEBUG
DOCMIND_CONTEXT_WINDOW_SIZE=8192
DOCMIND_AGENT_DECISION_TIMEOUT=100
DOCMIND_ENABLE_DOCUMENT_CACHING=false
```

## Troubleshooting Configuration Issues

### GPU Detection Problems

```bash
# Force GPU detection
DOCMIND_DEVICE=cuda
DOCMIND_ENABLE_GPU_ACCELERATION=true

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues

```bash
# Reduce memory usage
DOCMIND_CONTEXT_WINDOW_SIZE=4096
DOCMIND_MAX_VRAM_GB=8.0
DOCMIND_BGE_M3_BATCH_SIZE_GPU=4
```

### Performance Issues

```bash
# Optimize for your hardware
DOCMIND_PERFORMANCE_TIER=auto
DOCMIND_LLM_BACKEND=vllm  # If you have capable GPU
VLLM_ATTENTION_BACKEND=FLASHINFER
```

### Connection Issues

```bash
# Use local-only configuration
DOCMIND_LLM_BACKEND=ollama
DOCMIND_OLLAMA_BASE_URL=http://localhost:11434
DOCMIND_QDRANT_URL=http://localhost:6333
```

## Hardware Recommendations

### Minimum Requirements

- **CPU**: Any modern processor
- **RAM**: 8GB system memory
- **GPU**: Optional (CPU-only operation supported)
- **Storage**: 20GB free space for models

### Recommended Setup

- **CPU**: Modern multi-core processor
- **RAM**: 16GB system memory
- **GPU**: RTX 4060 (12GB VRAM) or equivalent
- **Storage**: 50GB SSD space for models and cache

### Optimal Setup

- **CPU**: High-end desktop processor
- **RAM**: 32GB+ system memory
- **GPU**: RTX 4080/4090 (16GB+ VRAM)
- **Storage**: 100GB+ NVMe SSD

## Summary

DocMind AI's configuration system is designed to **respect user choice** and **adapt to your specific hardware and privacy requirements**. Whether you're running on modest hardware with CPU-only operation or maximizing performance on a high-end workstation, DocMind AI provides the flexibility to optimize for your specific needs while maintaining complete local control and privacy.

The key principle is **local-first operation** - DocMind AI works entirely offline with no external dependencies, giving you complete control over your data and computing resources.
