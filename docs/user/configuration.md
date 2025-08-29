# Configuration Guide

DocMind AI adapts to your hardware and privacy preferences through simple configuration profiles. Whether you're running on a student laptop or a high-end workstation, these settings optimize performance for your specific setup.

## Quick Configuration Profiles

### Student/Budget Setup (CPU-Only)
Perfect for older laptops or privacy-focused users who prefer CPU-only operation:

```bash
# Copy base configuration
cp .env.example .env

# Add CPU-only settings
cat >> .env << 'EOF'
DOCMIND_ENABLE_GPU_ACCELERATION=false
DOCMIND_CONTEXT_WINDOW_SIZE=32768
DOCMIND_MAX_MEMORY_GB=4.0
DOCMIND_PERFORMANCE_TIER=low
AGENT_TIMEOUT_SECONDS=60
EOF
```

**Expected Performance**: 10-20 seconds per query, basic document analysis  
**Hardware Requirements**: 8GB RAM, any CPU  
**Privacy Level**: Complete offline operation

### Mid-Range Gaming Setup (RTX 4060/4070)
Good balance of performance and accessibility:

```bash
cp .env.example .env
cat >> .env << 'EOF'
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CONTEXT_WINDOW_SIZE=65536
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_MAX_MEMORY_GB=16.0
DOCMIND_PERFORMANCE_TIER=medium
VLLM_ATTENTION_BACKEND=FLASHINFER
EOF
```

**Expected Performance**: 3-7 seconds per query, good document analysis  
**Hardware Requirements**: RTX 4060/4070 (12-16GB VRAM), 16GB RAM  
**Privacy Level**: Complete offline operation with GPU acceleration

### High-End Research Setup (RTX 4090)
Maximum performance for professional research and complex documents:

```bash
cp .env.example .env
cat >> .env << 'EOF'
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_MAX_MEMORY_GB=32.0
DOCMIND_PERFORMANCE_TIER=high
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
ENABLE_DSPY_OPTIMIZATION=true
EOF
```

**Expected Performance**: 1-3 seconds per query, advanced analysis with 128K context  
**Hardware Requirements**: RTX 4090 (16-24GB VRAM), 32GB RAM  
**Privacy Level**: Complete offline operation with maximum features

### Privacy-Focused Setup
Maximum privacy with local models and no performance logging:

```bash
cp .env.example .env
cat >> .env << 'EOF'
DOCMIND_LOCAL_MODEL_PATH=/home/user/models/
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CONTEXT_WINDOW_SIZE=65536
DOCMIND_LOG_LEVEL=ERROR
EOF
```

**Hardware Requirements**: RTX 4070+ (16GB VRAM), 16GB RAM  
**Expected Performance**: 2-5 seconds per query, good analysis  
**Privacy Level**: Maximum privacy with local model storage

## Hardware-Specific Optimization

### GPU Memory Management

Control GPU memory usage based on your hardware:

```bash
# Conservative settings (12GB VRAM)
DOCMIND_GPU_MEMORY_UTILIZATION=0.75
DOCMIND_CONTEXT_WINDOW_SIZE=65536

# Balanced settings (16GB VRAM)
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_CONTEXT_WINDOW_SIZE=131072

# Aggressive settings (24GB VRAM)
DOCMIND_GPU_MEMORY_UTILIZATION=0.90
DOCMIND_CONTEXT_WINDOW_SIZE=131072
```

### System Memory Limits

Protect your system from out-of-memory errors:

```bash
# For 8GB system RAM
DOCMIND_MAX_MEMORY_GB=4.0
DOCMIND_MAX_DOCUMENT_SIZE_MB=25

# For 16GB system RAM
DOCMIND_MAX_MEMORY_GB=8.0
DOCMIND_MAX_DOCUMENT_SIZE_MB=50

# For 32GB+ system RAM
DOCMIND_MAX_MEMORY_GB=16.0
DOCMIND_MAX_DOCUMENT_SIZE_MB=100
```

### Performance Optimization

Enable advanced optimizations based on your hardware:

```bash
# RTX 4080/4090 optimizations
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_CUDNN_PREFILL=1
VLLM_KV_CACHE_DTYPE=fp8_e5m2

# CPU optimization
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=4

# Memory management
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Multi-Agent System Configuration

### Basic Agent Settings

```bash
# Enable 5-agent coordination
ENABLE_MULTI_AGENT=true
AGENT_TIMEOUT_SECONDS=30
MAX_CONTEXT_TOKENS=65000

# Agent performance tuning
AGENT_CONCURRENCY_LIMIT=5
MAX_AGENT_RETRIES=2
ENABLE_FALLBACK_RAG=true
```

### Quality and Validation

```bash
# Response quality controls
MIN_VALIDATION_SCORE=0.7
ENABLE_HALLUCINATION_CHECK=true
SOURCE_ATTRIBUTION_REQUIRED=true

# Cross-validation settings
ENABLE_CROSS_VALIDATION=true
RESPONSE_CONSISTENCY_CHECK=true
```

## Document Processing Settings

### Processing Optimization

```bash
# Document chunking
DOCMIND_CHUNK_SIZE=512
DOCMIND_CHUNK_OVERLAP=50

# Parallel processing
DOCMIND_PARALLEL_PROCESSING=true
DOCMIND_MAX_DOCUMENT_SIZE_MB=100

# Advanced features
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true
DOCMIND_ENABLE_GRAPHRAG=false  # Optional relationship mapping
```

### Search and Retrieval

```bash
# Embedding models (BGE-M3 unified)
DOCMIND_EMBEDDING_MODEL=BAAI/bge-m3
DOCMIND_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Retrieval parameters
RETRIEVAL_TOP_K=20
RERANKING_TOP_K=10
RRF_ALPHA=0.7

# Search optimization
DOCMIND_USE_RERANKING=true
ENABLE_QUERY_EXPANSION=true
```

## Performance Profiles

### Speed Optimized
For fast responses when accuracy can be slightly lower:

```bash
DOCMIND_PERFORMANCE_TIER=fast
AGENT_TIMEOUT_SECONDS=15
MIN_VALIDATION_SCORE=0.6
RETRIEVAL_TOP_K=10
CONTEXT_WINDOW_SIZE=32768
```

### Balanced (Default)
Good balance of speed and accuracy:

```bash
DOCMIND_PERFORMANCE_TIER=balanced
AGENT_TIMEOUT_SECONDS=30
MIN_VALIDATION_SCORE=0.7
RETRIEVAL_TOP_K=20
CONTEXT_WINDOW_SIZE=65536
```

### Quality Optimized
Best accuracy for important analysis:

```bash
DOCMIND_PERFORMANCE_TIER=thorough
AGENT_TIMEOUT_SECONDS=60
MIN_VALIDATION_SCORE=0.85
RETRIEVAL_TOP_K=30
CONTEXT_WINDOW_SIZE=131072
ENABLE_DSPY_OPTIMIZATION=true
```

## Model and Backend Configuration

### Ollama Backend (Recommended)
Best for privacy and offline operation:

```bash
DOCMIND_LLM_BACKEND=ollama
DOCMIND_OLLAMA_BASE_URL=http://localhost:11434
DOCMIND_MODEL_NAME=qwen3-4b-instruct-2507
```

### vLLM Backend (High Performance)
For users who want maximum speed:

```bash
DOCMIND_LLM_BACKEND=vllm
DOCMIND_VLLM_BASE_URL=http://localhost:8000
DOCMIND_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507-FP8

# Performance optimizations
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
VLLM_GPU_MEMORY_UTILIZATION=0.85
```

## Context Window Configuration

Choose based on your hardware and use case:

### Conservative (Works on Most Hardware)
```bash
DOCMIND_CONTEXT_WINDOW_SIZE=32768  # 32K tokens
```
- **Memory**: ~6GB additional VRAM
- **Use Case**: Basic document analysis
- **Performance**: Fast processing

### Moderate (Recommended)
```bash
DOCMIND_CONTEXT_WINDOW_SIZE=65536  # 64K tokens
```
- **Memory**: ~10GB additional VRAM
- **Use Case**: Most document types
- **Performance**: Good balance

### Maximum (High-End Hardware)
```bash
DOCMIND_CONTEXT_WINDOW_SIZE=131072  # 128K tokens
```
- **Memory**: ~12-14GB additional VRAM
- **Use Case**: Large documents, complex analysis
- **Performance**: Full context processing

## Privacy and Security Settings

### Maximum Privacy
For users who prioritize complete privacy:

```bash
# Disable external connections
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false
DOCMIND_LOG_LEVEL=ERROR

# Use local model storage
DOCMIND_LOCAL_MODEL_PATH=/private/models/

# Disable caching (optional)
DOCMIND_ENABLE_DOCUMENT_CACHING=false
```

### Offline-First Configuration
Ensure complete offline operation:

```bash
# Local model storage
DOCMIND_LOCAL_MODEL_PATH=/opt/models/

# Embedding models (must be pre-downloaded)
DOCMIND_EMBEDDING_MODEL=BAAI/bge-m3

# Vector database
DOCMIND_VECTOR_STORE_TYPE=qdrant
DOCMIND_QDRANT_URL=http://localhost:6333
```

## Validation and Testing

### Verify Your Configuration

```bash
# Test system configuration
python -c "
from src.config import settings
print(f'GPU Enabled: {settings.enable_gpu_acceleration}')
print(f'Context Size: {settings.context_window_size}')
print(f'Multi-Agent: {settings.agents.enable_multi_agent}')
print(f'Model: {settings.vllm.model}')
"
```

### Performance Testing

```bash
# Run performance validation
python scripts/performance_validation.py

# Test GPU functionality
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
else:
    print('GPU: Not available')
"
```

## Troubleshooting Configuration

### GPU Detection Issues
```bash
# Force GPU detection
DOCMIND_DEVICE=cuda
DOCMIND_ENABLE_GPU_ACCELERATION=true

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
```bash
# Reduce memory usage
DOCMIND_GPU_MEMORY_UTILIZATION=0.75
DOCMIND_CONTEXT_WINDOW_SIZE=32768
DOCMIND_MAX_DOCUMENT_SIZE_MB=50
```

### Performance Issues
```bash
# Enable optimizations
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_CUDNN_PREFILL=1
DOCMIND_PERFORMANCE_TIER=high
```

## Environment Variables Reference

### Core Settings
```bash
# Essential configuration
DOCMIND_ENABLE_GPU_ACCELERATION=true/false
DOCMIND_CONTEXT_WINDOW_SIZE=32768|65536|131072
DOCMIND_GPU_MEMORY_UTILIZATION=0.75-0.90
DOCMIND_MAX_MEMORY_GB=4.0|8.0|16.0|32.0

# Multi-agent system
ENABLE_MULTI_AGENT=true
AGENT_TIMEOUT_SECONDS=15|30|60
MAX_CONTEXT_TOKENS=32000|65000|128000

# Performance optimization
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
DOCMIND_PERFORMANCE_TIER=low|medium|high

# Document processing
DOCMIND_CHUNK_SIZE=256|512|1024
DOCMIND_CHUNK_OVERLAP=25|50|100
DOCMIND_MAX_DOCUMENT_SIZE_MB=25|50|100|200

# Quality and validation
MIN_VALIDATION_SCORE=0.6|0.7|0.8|0.85
ENABLE_HALLUCINATION_CHECK=true|false
SOURCE_ATTRIBUTION_REQUIRED=true|false
```

## Configuration Best Practices

### Start Simple
1. Use a pre-configured profile that matches your hardware
2. Test with a small document first
3. Monitor performance and memory usage
4. Adjust settings gradually based on results

### Hardware-Specific Optimization
1. **RTX 4060**: Use medium profile with 65K context
2. **RTX 4080**: Use high profile with 128K context  
3. **RTX 4090**: Use thorough profile with all optimizations
4. **CPU-Only**: Use low profile with conservative settings

### Memory Management
1. Start with conservative memory settings
2. Increase gradually if system is stable
3. Monitor GPU memory usage with `nvidia-smi`
4. Use smaller context windows if encountering OOM errors

### Performance Tuning
1. Enable FlashInfer backend for RTX 4080/4090
2. Use FP8 quantization for memory efficiency
3. Adjust agent timeouts based on your patience level
4. Enable DSPy optimization for better response quality

---

**Summary**: DocMind AI's configuration system adapts to your specific hardware and privacy requirements. Start with a pre-configured profile, test performance, and adjust settings gradually. The system is designed to work well with defaults while providing flexibility for optimization.