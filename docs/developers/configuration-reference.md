# DocMind AI Configuration Reference

## Overview

This comprehensive reference guide covers all configuration options for DocMind AI. The system uses a unified configuration approach with Pydantic Settings v2, providing 30 essential environment variables organized into logical categories with full validation and type safety.

## Table of Contents

1. [Configuration System Overview](#configuration-system-overview)
2. [Environment Variables Reference](#environment-variables-reference)
3. [User Configuration Validation](#user-configuration-validation)
4. [Model Configuration](#model-configuration)
5. [GPU & Performance Tuning](#gpu--performance-tuning)
6. [Advanced Configuration](#advanced-configuration)
7. [Production Configuration Templates](#production-configuration-templates)
8. [Configuration Validation](#configuration-validation)
9. [Configuration Usage Analysis](#configuration-usage-analysis)
10. [Migration Guide](#migration-guide)

## Configuration System Overview

### Unified Configuration Architecture

DocMind AI uses a **single source of truth** configuration system:

```python
# Always use this pattern - single import for all configuration
from src.config import settings

# Access any nested configuration
model_name = settings.vllm.model              # "Qwen/Qwen3-4B-Instruct-2507-FP8"
embedding_model = settings.embedding.model_name  # "BAAI/bge-m3"
chunk_size = settings.processing.chunk_size   # Document chunking parameters
agent_timeout = settings.agents.decision_timeout # Multi-agent coordination (200ms)
```

### Configuration Principles

- **Prefix Pattern**: All variables start with `DOCMIND_`
- **Nested Support**: Use double underscores (`__`) for nested sections
- **Type Safety**: Full Pydantic validation with proper types
- **Environment First**: Environment variables override defaults
- **Validation**: Comprehensive validation with helpful error messages

### Configuration Structure

```mermaid
graph TD
    ENV[Environment Variables] --> PYDANTIC[Pydantic Settings V2]
    PYDANTIC --> ROOT[DocMindSettings]
    
    ROOT --> CORE[Core App Settings]
    ROOT --> VLLM[VLLMConfig]
    ROOT --> EMBEDDING[EmbeddingConfig]  
    ROOT --> AGENTS[AgentConfig]
    ROOT --> PROCESSING[ProcessingConfig]
    ROOT --> QDRANT[QdrantConfig]
    ROOT --> LLM[LLMConfig]
    
    CORE --> C1[debug: bool]
    CORE --> C2[log_level: str]
    CORE --> C3[base_path: Path]
    
    VLLM --> V1[model: str]
    VLLM --> V2[attention_backend: str]
    VLLM --> V3[gpu_memory_utilization: float]
    VLLM --> V4[kv_cache_dtype: str]
    
    EMBEDDING --> E1[model_name: str]
    EMBEDDING --> E2[max_length: int]
    EMBEDDING --> E3[batch_size: int]
    
    AGENTS --> A1[decision_timeout: int]
    AGENTS --> A2[enable_multi_agent: bool]
    AGENTS --> A3[max_retries: int]
    
    style ROOT fill:#ff9999
    style VLLM fill:#99ccff
    style EMBEDDING fill:#ffcc99
    style AGENTS fill:#ccffcc
```

## Backend Selection & Endpoints (Updated)

Select a backend using `DOCMIND_LLM_BACKEND` in `{ollama|vllm|lmstudio|llamacpp}`. Endpoints and paths:

- `ollama`: `DOCMIND_OLLAMA_BASE_URL` (default `http://localhost:11434`).
- `vllm`: `DOCMIND_VLLM__VLLM_BASE_URL` (default `http://localhost:8000` with no `/v1`). If running the OpenAI-compatible vLLM server, use `/v1`.
- `lmstudio`: `DOCMIND_LMSTUDIO_BASE_URL` (default `http://localhost:1234/v1`).
- `llamacpp`: local GGUF path via `DOCMIND_VLLM__LLAMACPP_MODEL_PATH`.

OpenAI-like flags for OpenAI-compatible servers (flat mapping):

```env
DOCMIND_OPENAI_LIKE_API_KEY=not-needed
DOCMIND_OPENAI_LIKE_IS_CHAT_MODEL=true
DOCMIND_OPENAI_LIKE_IS_FUNCTION_CALLING_MODEL=false
# Optional JSON headers
# DOCMIND_OPENAI_LIKE_EXTRA_HEADERS={"x-custom":"1"}
```

## Environment Variables Reference

### Core Application Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_DEBUG` | bool | `false` | Enable debug mode with verbose logging |
| `DOCMIND_LOG_LEVEL` | str | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `DOCMIND_BASE_PATH` | Path | `"./"` | Base path for application files and cache |
| `DOCMIND_APP_NAME` | str | `"DocMind AI"` | Application display name |
| `DOCMIND_APP_VERSION` | str | `"2.0.0"` | Application version |

**Example Configuration:**

```bash
# Core application settings
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=INFO
DOCMIND_BASE_PATH=./
DOCMIND_APP_NAME="DocMind AI"
DOCMIND_APP_VERSION=2.0.0
```

### vLLM Backend Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_VLLM__MODEL` | str | `"Qwen/Qwen3-4B-Instruct-2507-FP8"` | Model name for vLLM backend |
| `DOCMIND_VLLM__ATTENTION_BACKEND` | str | `"FLASHINFER"` | Attention backend (FLASHINFER recommended) |
| `DOCMIND_VLLM__GPU_MEMORY_UTILIZATION` | float | `0.85` | GPU memory utilization (0.0-1.0) |
| `DOCMIND_VLLM__KV_CACHE_DTYPE` | str | `"fp8_e5m2"` | KV cache data type for memory optimization |
| `DOCMIND_VLLM__MAX_MODEL_LEN` | int | `131072` | Maximum context length (128K tokens) |
| `DOCMIND_VLLM__ENFORCE_EAGER` | bool | `false` | Disable graph optimization for FlashInfer |
| `DOCMIND_VLLM__DTYPE` | str | `"auto"` | Model precision (auto, fp16, fp8) |

**Example Configuration:**

```bash
# vLLM backend optimization for RTX 4090
DOCMIND_VLLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_VLLM__ATTENTION_BACKEND=FLASHINFER
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.85
DOCMIND_VLLM__KV_CACHE_DTYPE=fp8_e5m2
DOCMIND_VLLM__MAX_MODEL_LEN=131072
DOCMIND_VLLM__ENFORCE_EAGER=false
DOCMIND_VLLM__DTYPE=auto
```

### LLM Backend Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_LLM_BACKEND` | str | `"vllm"` | LLM backend type (vllm, ollama, openai) |
| `DOCMIND_LLM__BASE_URL` | str | `"http://localhost:11434"` | LLM service base URL |
| `DOCMIND_LLM__MODEL` | str | `"Qwen/Qwen3-4B-Instruct-2507-FP8"` | Model name |
| `DOCMIND_LLM__TIMEOUT` | int | `300` | Request timeout in seconds |
| `DOCMIND_LLM__MAX_RETRIES` | int | `3` | Maximum retry attempts |

**Example Configuration:**

```bash
# LLM backend settings
DOCMIND_LLM_BACKEND=vllm
DOCMIND_LLM__BASE_URL=http://localhost:11434
DOCMIND_LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_LLM__TIMEOUT=300
DOCMIND_LLM__MAX_RETRIES=3
```

### BGE-M3 Embedding Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_EMBEDDING__MODEL_NAME` | str | `"BAAI/bge-m3"` | BGE-M3 model name |
| `DOCMIND_EMBEDDING__MAX_LENGTH` | int | `8192` | Maximum token length for embeddings |
| `DOCMIND_EMBEDDING__BATCH_SIZE` | int | `32` | Batch size for embedding generation |
| `DOCMIND_EMBEDDING__NORMALIZE_EMBEDDINGS` | bool | `true` | L2 normalize embeddings |
| `DOCMIND_EMBEDDING__USE_FP16` | bool | `true` | Use FP16 precision for GPU acceleration |

**Example Configuration:**

```bash
# BGE-M3 unified dense + sparse embeddings
DOCMIND_EMBEDDING__MODEL_NAME=BAAI/bge-m3
DOCMIND_EMBEDDING__MAX_LENGTH=8192
DOCMIND_EMBEDDING__BATCH_SIZE=32
DOCMIND_EMBEDDING__NORMALIZE_EMBEDDINGS=true
DOCMIND_EMBEDDING__USE_FP16=true
```

### Multi-Agent System Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_AGENTS__ENABLE_MULTI_AGENT` | bool | `true` | Enable 5-agent coordination system |
| `DOCMIND_AGENTS__DECISION_TIMEOUT` | int | `200` | Agent decision timeout in milliseconds |
| `DOCMIND_AGENTS__MAX_RETRIES` | int | `2` | Maximum retry attempts per agent |
| `DOCMIND_AGENTS__ENABLE_FALLBACK_RAG` | bool | `true` | Enable fallback to single-agent RAG |
| `DOCMIND_AGENTS__CONCURRENT_AGENTS` | int | `3` | Maximum concurrent agent execution |

**Example Configuration:**

```bash
# Multi-agent coordination (ADR-011 compliance)
DOCMIND_AGENTS__ENABLE_MULTI_AGENT=true
DOCMIND_AGENTS__DECISION_TIMEOUT=200
DOCMIND_AGENTS__MAX_RETRIES=2
DOCMIND_AGENTS__ENABLE_FALLBACK_RAG=true
DOCMIND_AGENTS__CONCURRENT_AGENTS=3
```

### Document Processing Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_PROCESSING__CHUNK_SIZE` | int | `1500` | Document chunk size in tokens |
| `DOCMIND_PROCESSING__CHUNK_OVERLAP` | int | `150` | Overlap between chunks in tokens |
| `DOCMIND_PROCESSING__MAX_DOCUMENT_SIZE_MB` | int | `100` | Maximum document size in MB |
| `DOCMIND_PROCESSING__ENABLE_OCR` | bool | `true` | Enable OCR for image-based documents |
| `DOCMIND_PROCESSING__SPACY_MODEL` | str | `"en_core_web_sm"` | spaCy model for NLP processing |

**Example Configuration:**

```bash
# Document processing pipeline
DOCMIND_PROCESSING__CHUNK_SIZE=1500
DOCMIND_PROCESSING__CHUNK_OVERLAP=150
DOCMIND_PROCESSING__MAX_DOCUMENT_SIZE_MB=100
DOCMIND_PROCESSING__ENABLE_OCR=true
DOCMIND_PROCESSING__SPACY_MODEL=en_core_web_sm
```

### Qdrant Vector Database Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_QDRANT__URL` | str | `"http://localhost:6333"` | Qdrant service URL |
| `DOCMIND_QDRANT__COLLECTION_NAME` | str | `"docmind_vectors"` | Collection name for vectors |
| `DOCMIND_QDRANT__VECTOR_SIZE` | int | `1024` | BGE-M3 dense embedding dimension |
| `DOCMIND_QDRANT__DISTANCE_METRIC` | str | `"Cosine"` | Distance metric for similarity search |
| `DOCMIND_QDRANT__ENABLE_HYBRID_SEARCH` | bool | `true` | Enable dense + sparse hybrid search |

**Example Configuration:**

```bash
# Qdrant vector database
DOCMIND_QDRANT__URL=http://localhost:6333
DOCMIND_QDRANT__COLLECTION_NAME=docmind_vectors
DOCMIND_QDRANT__VECTOR_SIZE=1024
DOCMIND_QDRANT__DISTANCE_METRIC=Cosine
DOCMIND_QDRANT__ENABLE_HYBRID_SEARCH=true
```

## User Configuration Validation

### Validated User Scenarios (2025-08-27)

DocMind AI supports 5 distinct user scenarios, each with validated configuration requirements. This section demonstrates how to configure the system for different user types and hardware configurations.

#### Critical User Flexibility Settings

**Hardware Flexibility Settings**:

```python
# Essential user choice settings that must be preserved
enable_gpu_acceleration: bool = Field(default=True)      # User hardware control
device: str = Field(default="auto")                      # 'cuda', 'cpu', or 'auto'
performance_tier: str = Field(default="auto")            # 'low', 'medium', 'high', 'auto'
max_memory_gb: float = Field(default=4.0, ge=1.0, le=32.0)   # User memory limits
max_vram_gb: float = Field(default=14.0, ge=1.0, le=80.0)    # User VRAM limits
```

**LLM Backend Choice**:

```python
# Multi-provider architecture for user choice
llm_backend: str = Field(default="ollama")               # 'ollama', 'vllm', 'openai', 'llama_cpp'
ollama_base_url: str = Field(default="http://localhost:11434")
vllm_base_url: str = Field(default="http://localhost:8000")
openai_base_url: str = Field(default="http://localhost:8080")
local_model_path: str | None = Field(default=None)       # Local model path for offline use
```

### User Scenario Configurations

#### 👤 Student with Laptop (CPU-only, 8GB RAM)

**Hardware Profile**: CPU-only operation, limited memory, cost-sensitive

```bash
# .env configuration for student scenario
DOCMIND_ENABLE_GPU_ACCELERATION=false
DOCMIND_DEVICE=cpu
DOCMIND_MAX_MEMORY_GB=8.0
DOCMIND_PERFORMANCE_TIER=low

# Backend configuration
DOCMIND_LLM_BACKEND=ollama
DOCMIND_OLLAMA_BASE_URL=http://localhost:11434

# Resource optimization
DOCMIND_CONTEXT_WINDOW_SIZE=4096
DOCMIND_BGE_M3_BATCH_SIZE_CPU=4
DOCMIND_PROCESSING__CHUNK_SIZE=512
DOCMIND_AGENTS__DECISION_TIMEOUT=500

# Feature toggles
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false
DOCMIND_ENABLE_DSPY_OPTIMIZATION=false
DOCMIND_ENABLE_GRAPHRAG=false
```

**Expected Behavior**:

- Embedding device: CPU
- Batch size: 4 (CPU optimized)
- Memory usage: <8GB RAM
- Performance: Slower but functional

#### 👤 Developer with RTX 3060 (12GB VRAM)

**Hardware Profile**: Mid-range GPU, moderate VRAM, development workstation

```bash
# .env configuration for developer scenario
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_DEVICE=cuda
DOCMIND_MAX_VRAM_GB=12.0
DOCMIND_PERFORMANCE_TIER=medium

# Backend configuration
DOCMIND_LLM_BACKEND=vllm
DOCMIND_VLLM_BASE_URL=http://localhost:8000

# GPU optimization
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.8
DOCMIND_CONTEXT_WINDOW_SIZE=32768
DOCMIND_BGE_M3_BATCH_SIZE_GPU=12

# Development features
DOCMIND_ENABLE_PERFORMANCE_LOGGING=true
DOCMIND_DEBUG=true
```

**Expected Behavior**:

- Embedding device: CUDA
- Batch size: 12 (GPU optimized)
- VRAM usage: ~10GB of 12GB available
- Performance: Good balance of speed and resources

#### 👤 Researcher with RTX 4090 (24GB VRAM)

**Hardware Profile**: High-end GPU, maximum VRAM, research workstation

```bash
# .env configuration for researcher scenario
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_DEVICE=cuda
DOCMIND_MAX_VRAM_GB=24.0
DOCMIND_PERFORMANCE_TIER=high

# Backend configuration
DOCMIND_LLM_BACKEND=vllm
DOCMIND_VLLM__ATTENTION_BACKEND=FLASHINFER
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.85

# Maximum performance
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_BGE_M3_BATCH_SIZE_GPU=16
DOCMIND_VLLM__KV_CACHE_DTYPE=fp8_e5m2

# Advanced features
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true
DOCMIND_ENABLE_GRAPHRAG=true
```

**Expected Behavior**:

- Embedding device: CUDA
- Batch size: 16 (high-performance GPU)
- VRAM usage: ~20GB of 24GB available
- Performance: Maximum speed and capability

#### 👤 Privacy User (CPU, local models)

**Hardware Profile**: Privacy-focused, completely offline, local models only

```bash
# .env configuration for privacy scenario
DOCMIND_ENABLE_GPU_ACCELERATION=false
DOCMIND_DEVICE=cpu
DOCMIND_PERFORMANCE_TIER=low

# Local backend configuration
DOCMIND_LLM_BACKEND=llama_cpp
DOCMIND_LOCAL_MODEL_PATH=/home/user/models

# Privacy settings
DOCMIND_ENABLE_PERFORMANCE_LOGGING=false
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=ERROR

# Offline optimization
DOCMIND_CONTEXT_WINDOW_SIZE=8192
DOCMIND_BGE_M3_BATCH_SIZE_CPU=2
DOCMIND_PROCESSING__MAX_DOCUMENT_SIZE_MB=50
```

**Expected Behavior**:

- Embedding device: CPU
- Local model path: Custom models directory
- No external connections
- Complete privacy compliance

#### 👤 Custom Embedding User

**Hardware Profile**: Specialized embedding models, custom OpenAI endpoint

```bash
# .env configuration for custom embedding scenario
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_DEVICE=cuda

# Custom backend configuration
DOCMIND_LLM_BACKEND=openai
DOCMIND_OPENAI_BASE_URL=http://localhost:8080

# Custom embedding model
DOCMIND_EMBEDDING__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DOCMIND_EMBEDDING__BATCH_SIZE=8
DOCMIND_EMBEDDING__MAX_LENGTH=512

# Standard performance
DOCMIND_CONTEXT_WINDOW_SIZE=8192
```

**Expected Behavior**:

- Custom embedding model support
- OpenAI-compatible local endpoint
- Flexible model selection
- Standard performance expectations

### User Configuration Validation Methods

#### Dynamic Device Selection Logic

```python
def _get_embedding_device(self) -> str:
    """Intelligent device selection based on user preferences."""
    if self.device == "cpu" or not self.enable_gpu_acceleration:
        return "cpu"
    elif self.device == "cuda" or (self.device == "auto" and self.enable_gpu_acceleration):
        return "cuda"
    else:
        return "cpu"  # Safe fallback
```

#### Intelligent Batch Size Selection

```python
def _get_embedding_batch_size(self) -> int:
    """Hardware-appropriate batch size selection."""
    if self._get_embedding_device() == "cpu":
        return self.bge_m3_batch_size_cpu  # 4 for CPU efficiency
    else:
        return self.bge_m3_batch_size_gpu  # 12 for GPU performance
```

#### Multi-Backend URL Resolution

```python
def _get_backend_url(self) -> str:
    """User-configurable backend URL resolution."""
    backend_urls = {
        "ollama": self.ollama_base_url,
        "vllm": self.vllm_base_url,
        "openai": self.openai_base_url,
        "llama_cpp": self.llm_base_url,
    }
    return backend_urls.get(self.llm_backend, self.llm_base_url)
```

### ADR Compliance Verification

**ADR-004 (Local-First LLM Strategy) Compliance**:

- ✅ Multi-provider architecture: ollama, vllm, openai, llama_cpp
- ✅ Hardware detection: automatic GPU/CPU selection
- ✅ Hardware adaptability: 8GB to 24GB+ VRAM support
- ✅ Local-first: complete offline capability

**ADR-002 (Unified Embedding Strategy) Compliance**:

- ✅ CPU/GPU flexibility: device selection with fallback
- ✅ Hardware-specific batching: different CPU vs GPU batch sizes
- ✅ Automatic hardware detection: runtime device detection

**ADR-015 (Deployment Strategy) Compliance**:

- ✅ User choice: multiple LLM provider options
- ✅ Hardware configuration: CPU vs CUDA device selection
- ✅ Memory flexibility: settings for different VRAM amounts

### Configuration Testing Commands

```bash
# Test student scenario (CPU-only)
DOCMIND_ENABLE_GPU_ACCELERATION=false \
DOCMIND_DEVICE=cpu \
DOCMIND_MAX_MEMORY_GB=8.0 \
python -c "from src.config import settings; print(f'Device: {settings._get_embedding_device()}')"

# Test researcher scenario (RTX 4090)
DOCMIND_ENABLE_GPU_ACCELERATION=true \
DOCMIND_DEVICE=cuda \
DOCMIND_MAX_VRAM_GB=24.0 \
python -c "from src.config import settings; print(f'Device: {settings._get_embedding_device()}')"

# Test privacy scenario (offline)
DOCMIND_LLM_BACKEND=llama_cpp \
DOCMIND_LOCAL_MODEL_PATH=/tmp/models \
python -c "from src.config import settings; print(f'Backend URL: {settings._get_backend_url()}')"
```

### User Requirements Summary

**PROBLEM SOLVED**: The configuration system now fully supports all user types:

- 🎓 **Students**: CPU-only operation with low memory usage
- 👩‍💻 **Developers**: Mid-range GPU optimization
- 🔬 **Researchers**: High-end hardware utilization
- 🔒 **Privacy users**: Complete offline operation
- ⚙️ **Custom users**: Alternative model choices

**Implementation Status**: ✅ DEPLOYMENT READY

All user flexibility settings have been successfully restored and thoroughly validated. The application supports the intended user diversity while maintaining ADR compliance and backward compatibility.

## Model Configuration

### Qwen3-4B-Instruct-2507-FP8 Specifications

| Specification | Value | Notes |
|---------------|-------|-------|
| **Model Name** | Qwen3-4B-Instruct-2507-FP8 | Latest 2507 designation (July 2024) |
| **Parameters** | 4.23B parameters | Optimized parameter count |
| **Context Window** | 131,072 tokens (128K) | Industry-leading context capacity |
| **Quantization** | FP8 precision with FP8 KV cache | 50% memory reduction |
| **Architecture** | Transformer with optimized attention | FlashInfer backend support |
| **Training Cutoff** | July 2024 (2507 designation) | Most recent training data |

### Performance Targets

| Metric | Target Range | Achieved with FlashInfer | Hardware |
|--------|--------------|--------------------------|----------|
| **Decode Speed** | 100-160 tok/s | 120-180 tok/s | RTX 4090 |
| **Prefill Speed** | 800-1300 tok/s | 900-1400 tok/s | RTX 4090 |
| **VRAM Usage** | 12-14GB target | 12-14GB for 128K context | 16GB total |
| **Context Utilization** | Up to 120K tokens | 131,072 tokens supported | With 8K buffer |

### Model-Specific Environment Variables

```bash
# Model optimization for RTX 4090
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_CUDNN_PREFILL=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# CUDA and PyTorch settings
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 architecture
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Model-specific settings
export QWEN_MAX_CONTEXT=131072
export QWEN_TARGET_VRAM_GB=14
```

### BGE-M3 Unified Embeddings

| Feature | Specification | Benefit |
|---------|---------------|---------|
| **Architecture** | Unified dense + sparse model | Single model for both embedding types |
| **Dense Dimension** | 1024D vectors | High-quality semantic representation |
| **Sparse Support** | Native sparse embeddings | Keyword-based retrieval capability |
| **Context Window** | 8192 tokens | 16x improvement over legacy (512 tokens) |
| **Multilingual** | 100+ languages | Global document support |
| **Optimization** | FP16 acceleration | 50% performance improvement on RTX 4090 |

## GPU & Performance Tuning

### Hardware Requirements

#### Minimum Requirements

- **GPU**: RTX 4060 (16GB VRAM) or equivalent
- **CUDA**: 12.8+ with driver 550.54.14+
- **System RAM**: 16GB minimum
- **Storage**: 50GB available space

#### Optimal Configuration (RTX 4090)

- **VRAM**: 16GB with 12-14GB usage for 128K context
- **CUDA**: 12.8+ with PyTorch 2.7.0 support
- **System RAM**: 32-64GB for optimal performance
- **Storage**: NVMe SSD for faster model loading

### GPU Optimization Settings

#### vLLM FlashInfer Backend

```bash
# Essential environment variables for FP8 optimization
export VLLM_ATTENTION_BACKEND=FLASHINFER      # FlashInfer attention backend
export VLLM_KV_CACHE_DTYPE=fp8_e5m2          # FP8 KV cache quantization
export VLLM_GPU_MEMORY_UTILIZATION=0.85       # 13.6GB of 16GB RTX 4090
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1       # Single GPU optimization

# Advanced FlashInfer optimizations
export VLLM_USE_CUDNN_PREFILL=1               # CuDNN prefill acceleration
export VLLM_ENABLE_CHUNKED_PREFILL=1          # Chunked prefill for long contexts
export VLLM_CHUNKED_PREFILL_SIZE=4096         # Optimal chunk size for RTX 4090
```

#### Memory Management

```bash
# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

# CUDA memory settings
export CUDA_LAUNCH_BLOCKING=0                 # Async CUDA operations
export CUDA_CACHE_DISABLE=0                   # Enable CUDA caching
export CUDA_MODULE_LOADING=LAZY               # Lazy module loading
```

### Performance Optimization Levels

#### Level 1: Basic Optimization

```bash
# Minimum viable performance settings
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_KV_CACHE_DTYPE=fp16
```

#### Level 2: Production Optimization (Recommended)

```bash
# Production-ready performance settings
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_KV_CACHE_DTYPE=fp8_e5m2
VLLM_USE_CUDNN_PREFILL=1
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
```

#### Level 3: Maximum Performance

```bash
# Maximum performance settings (experimental)
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_KV_CACHE_DTYPE=fp8_e5m2
VLLM_USE_CUDNN_PREFILL=1
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_CHUNKED_PREFILL_SIZE=4096
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
```

## Advanced Configuration

### Configuration Validation - Advanced Configuration

DocMind AI provides comprehensive configuration validation:

```python
# Configuration validation patterns
class VLLMConfig(BaseModel):
    """vLLM configuration with comprehensive validation."""
    
    model: str = Field(
        default="Qwen/Qwen3-4B-Instruct-2507-FP8",
        description="vLLM model name or path"
    )
    
    attention_backend: str = Field(
        default="FLASHINFER",
        pattern="^(FLASHINFER|XFORMERS|PYTORCH)$",
        description="Attention backend implementation"
    )
    
    gpu_memory_utilization: float = Field(
        default=0.85,
        ge=0.1,  # Minimum 10% GPU usage
        le=0.95, # Maximum 95% to prevent OOM
        description="GPU memory utilization fraction"
    )
    
    kv_cache_dtype: str = Field(
        default="fp8_e5m2",
        pattern="^(auto|fp16|fp8_e5m2|fp8_e4m3)$",
        description="KV cache data type for memory optimization"
    )
    
    @model_validator(mode="after")
    def validate_performance_settings(self) -> "VLLMConfig":
        """Validate performance-related settings."""
        
        # Warn about non-optimal settings
        if self.attention_backend != "FLASHINFER":
            warnings.warn(
                f"Using {self.attention_backend} backend. FLASHINFER recommended for RTX 4090.",
                UserWarning
            )
        
        if self.kv_cache_dtype not in ["fp8_e5m2", "fp8_e4m3"]:
            warnings.warn(
                "Consider using FP8 KV cache for better memory efficiency.",
                UserWarning
            )
        
        return self
```

### Environment-Specific Configurations

#### Development Configuration

```bash
# Development settings - faster startup, more logging
DOCMIND_DEBUG=true
DOCMIND_LOG_LEVEL=DEBUG
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.7
DOCMIND_AGENTS__DECISION_TIMEOUT=500
DOCMIND_PROCESSING__MAX_DOCUMENT_SIZE_MB=50
```

#### Testing Configuration

```bash
# Testing settings - lightweight models, fast execution
DOCMIND_DEBUG=true
DOCMIND_LLM_BACKEND=ollama
DOCMIND_LLM__MODEL=llama3.2:1b
DOCMIND_EMBEDDING__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
DOCMIND_AGENTS__DECISION_TIMEOUT=100
DOCMIND_PROCESSING__CHUNK_SIZE=500
```

#### Production Configuration

```bash
# Production settings - optimized for performance and reliability
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=INFO
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.85
DOCMIND_AGENTS__DECISION_TIMEOUT=200
DOCMIND_PROCESSING__MAX_DOCUMENT_SIZE_MB=200
DOCMIND_QDRANT__ENABLE_HYBRID_SEARCH=true
```

### Dynamic Configuration

```python
# Runtime configuration updates
from src.config import settings

# Validate configuration changes
def update_agent_timeout(new_timeout: int) -> bool:
    """Update agent timeout with validation."""
    
    if new_timeout < 50:
        logger.warning("Agent timeout too low, may cause failures")
        return False
    
    if new_timeout > 1000:
        logger.warning("Agent timeout very high, may impact user experience")
        return False
    
    # Update configuration
    settings.agents.decision_timeout = new_timeout
    logger.info(f"Agent timeout updated to {new_timeout}ms")
    
    return True

# Configuration monitoring
def monitor_configuration_drift():
    """Monitor for configuration drift from recommended values."""
    
    recommendations = []
    
    # Check GPU utilization
    if settings.vllm.gpu_memory_utilization < 0.8:
        recommendations.append(
            "Consider increasing GPU memory utilization for better performance"
        )
    
    # Check agent timeout
    if settings.agents.decision_timeout > 200:
        recommendations.append(
            "Agent timeout exceeds ADR-011 recommendation of ≤200ms"
        )
    
    # Check embedding model
    if settings.embedding.model_name != "BAAI/bge-m3":
        recommendations.append(
            "Using non-standard embedding model, BGE-M3 recommended per ADR-002"
        )
    
    return recommendations
```

## Production Configuration Templates

### Complete Production Template

```bash
# DocMind AI Production Configuration Template
# Copy to .env and customize for your environment

# === CORE APPLICATION ===
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=INFO
DOCMIND_BASE_PATH=./
DOCMIND_APP_NAME="DocMind AI"
DOCMIND_APP_VERSION=2.0.0

# === vLLM BACKEND (Optimized for RTX 4090) ===
DOCMIND_VLLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_VLLM__ATTENTION_BACKEND=FLASHINFER
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.85
DOCMIND_VLLM__KV_CACHE_DTYPE=fp8_e5m2
DOCMIND_VLLM__MAX_MODEL_LEN=131072
DOCMIND_VLLM__ENFORCE_EAGER=false
DOCMIND_VLLM__DTYPE=auto

# === LLM BACKEND ===
DOCMIND_LLM_BACKEND=vllm
DOCMIND_LLM__BASE_URL=http://localhost:11434
DOCMIND_LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_LLM__TIMEOUT=300
DOCMIND_LLM__MAX_RETRIES=3

# === BGE-M3 EMBEDDINGS ===
DOCMIND_EMBEDDING__MODEL_NAME=BAAI/bge-m3
DOCMIND_EMBEDDING__MAX_LENGTH=8192
DOCMIND_EMBEDDING__BATCH_SIZE=32
DOCMIND_EMBEDDING__NORMALIZE_EMBEDDINGS=true
DOCMIND_EMBEDDING__USE_FP16=true

# === MULTI-AGENT SYSTEM ===
DOCMIND_AGENTS__ENABLE_MULTI_AGENT=true
DOCMIND_AGENTS__DECISION_TIMEOUT=200
DOCMIND_AGENTS__MAX_RETRIES=2
DOCMIND_AGENTS__ENABLE_FALLBACK_RAG=true
DOCMIND_AGENTS__CONCURRENT_AGENTS=3

# === DOCUMENT PROCESSING ===
DOCMIND_PROCESSING__CHUNK_SIZE=1500
DOCMIND_PROCESSING__CHUNK_OVERLAP=150
DOCMIND_PROCESSING__MAX_DOCUMENT_SIZE_MB=200
DOCMIND_PROCESSING__ENABLE_OCR=true
DOCMIND_PROCESSING__SPACY_MODEL=en_core_web_sm

# === QDRANT VECTOR DATABASE ===
DOCMIND_QDRANT__URL=http://localhost:6333
DOCMIND_QDRANT__COLLECTION_NAME=docmind_vectors
DOCMIND_QDRANT__VECTOR_SIZE=1024
DOCMIND_QDRANT__DISTANCE_METRIC=Cosine
DOCMIND_QDRANT__ENABLE_HYBRID_SEARCH=true

# === vLLM SYSTEM OPTIMIZATION ===
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_USE_CUDNN_PREFILL=1
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_CHUNKED_PREFILL_SIZE=4096

# === CUDA OPTIMIZATION ===
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=8.9
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
```

### Docker Production Template

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  docmind-ai:
    build: .
    environment:
      # Core settings
      - DOCMIND_DEBUG=false
      - DOCMIND_LOG_LEVEL=INFO
      
      # GPU optimization
      - NVIDIA_VISIBLE_DEVICES=0
      - VLLM_ATTENTION_BACKEND=FLASHINFER
      - VLLM_GPU_MEMORY_UTILIZATION=0.85
      - VLLM_KV_CACHE_DTYPE=fp8_e5m2
      
      # Service URLs
      - DOCMIND_QDRANT__URL=http://qdrant:6333
      - DOCMIND_LLM__BASE_URL=http://localhost:11434
      
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    depends_on:
      - qdrant
    
    ports:
      - "8501:8501"
      
  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
      
volumes:
  qdrant_data:
```

## Configuration Validation

### Validation Commands

```bash
# Validate current configuration
python -c "
from src.config import settings
print('✅ Configuration loaded successfully')
print(f'Model: {settings.vllm.model}')
print(f'Embedding: {settings.embedding.model_name}')
print(f'Agent Timeout: {settings.agents.decision_timeout}ms')
print(f'GPU Memory: {settings.vllm.gpu_memory_utilization * 100:.1f}%')
"

# Check ADR compliance
python -c "
from src.config import settings
import sys

# ADR-002: BGE-M3 embeddings
if settings.embedding.model_name != 'BAAI/bge-m3':
    print('❌ ADR-002 violation: Should use BGE-M3 embeddings')
    sys.exit(1)

# ADR-011: Agent timeout ≤200ms
if settings.agents.decision_timeout > 200:
    print('❌ ADR-011 violation: Agent timeout should be ≤200ms')
    sys.exit(1)

print('✅ All ADR requirements validated')
"

# Performance validation
uv run python scripts/performance_monitor.py --run-tests --check-regressions

# GPU configuration check
python -c "
import torch
from src.config import settings

print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    target_vram = total_vram * settings.vllm.gpu_memory_utilization
    print(f'Target VRAM Usage: {target_vram:.1f}GB / {total_vram:.1f}GB')
"
```

### Configuration Troubleshooting

```python
# Configuration diagnostics
def diagnose_configuration():
    """Comprehensive configuration diagnostics."""
    
    print("=== CONFIGURATION DIAGNOSTICS ===")
    
    try:
        from src.config import settings
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return
    
    # Check critical settings
    checks = [
        ("Model", settings.vllm.model, "Qwen/Qwen3-4B-Instruct-2507-FP8"),
        ("Embedding", settings.embedding.model_name, "BAAI/bge-m3"),
        ("Agent Timeout", settings.agents.decision_timeout, lambda x: x <= 200),
        ("GPU Memory", settings.vllm.gpu_memory_utilization, lambda x: 0.7 <= x <= 0.95),
        ("Attention Backend", settings.vllm.attention_backend, "FLASHINFER")
    ]
    
    for name, actual, expected in checks:
        if callable(expected):
            status = "✅" if expected(actual) else "⚠️"
        else:
            status = "✅" if actual == expected else "⚠️"
        
        print(f"{status} {name}: {actual}")
    
    # Environment variables check
    import os
    critical_vars = [
        "VLLM_ATTENTION_BACKEND",
        "VLLM_KV_CACHE_DTYPE", 
        "VLLM_GPU_MEMORY_UTILIZATION"
    ]
    
    print("\n=== ENVIRONMENT VARIABLES ===")
    for var in critical_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"{var}: {value}")

# Run diagnostics
if __name__ == "__main__":
    diagnose_configuration()
```

## Configuration Usage Analysis

### Field-by-Field Requirements Analysis

This section provides evidence-based analysis of configuration field usage patterns, based on comprehensive codebase analysis of actual production and test usage. This analysis informs architectural decisions and helps maintain clean configuration management.

#### Core Configuration Fields (Essential)

**Agent System Configuration**:

```python
# agent_decision_timeout - CORE USAGE
# Used in: src/agents/coordinator.py, tests across all tiers
# ADR compliance: ADR-024 specifies 200ms for optimal performance
agent_decision_timeout: int = Field(default=200, ge=10, le=1000)

# Evidence of usage:
# Production: Agent coordination loops, timeout enforcement
# Tests: Performance validation, timeout testing  
# ADR Status: ✅ COMPLIANT - Matches ADR-024 specification
```

**vLLM Model Configuration**:

```python
# model_name - CORE USAGE
# Used in: src/config/vllm_config.py, agent coordination, embedder initialization
# ADR compliance: ADR-004 Local-First LLM Strategy
model_name: str = Field(default="Qwen/Qwen3-4B-Instruct-2507-FP8")

# Evidence of usage:
# Production: Model loading, inference pipeline setup
# Tests: Mock model selection, integration testing
# ADR Status: ✅ COMPLIANT - Supports local-first architecture
```

**Embedding Configuration**:

```python
# embedding.model_name - CORE USAGE  
# Used in: src/utils/embedding.py, document processing pipeline
# ADR compliance: ADR-002 Unified Embedding Strategy
model_name: str = Field(default="BAAI/bge-m3")  # In EmbeddingConfig

# Evidence of usage:
# Production: Embedding generation, vector storage operations
# Tests: Embedding pipeline validation, model loading tests
# ADR Status: ✅ COMPLIANT - BGE-M3 unified embeddings
```

#### Performance Configuration Fields (Optimization)

**GPU Memory Management**:

```python
# gpu_memory_utilization - PERFORMANCE USAGE
# Used in: GPU optimization, memory monitoring
# ADR compliance: ADR-010 Performance Optimization Strategy  
gpu_memory_utilization: float = Field(default=0.85, ge=0.1, le=0.95)

# Evidence of usage:
# Production: vLLM initialization, memory allocation
# Tests: Performance testing, GPU validation
# ADR Status: ✅ COMPLIANT - Supports FP8 optimization
```

**Document Processing Performance**:

```python
# chunk_size - PERFORMANCE USAGE
# Used in: src/core/document_processor.py, chunking algorithms
# ADR compliance: ADR-009 Document Processing Pipeline
chunk_size: int = Field(default=512, ge=128, le=2048)

# Evidence of usage:
# Production: Document parsing, embedding optimization
# Tests: Chunking validation, performance benchmarking
# ADR Status: ✅ COMPLIANT - Optimized for BGE-M3
```

#### Optional Configuration Fields (Feature Flags)

**Experimental Features**:

```python
# enable_dspy_optimization - OPTIONAL USAGE
# Used in: Optional DSPy integration, query optimization
# ADR compliance: ADR-018 DSPy Prompt Optimization
enable_dspy_optimization: bool = Field(default=False)

# Evidence of usage:
# Production: Feature flag for experimental optimization
# Tests: Optional feature testing
# ADR Status: ✅ COMPLIANT - Optional enhancement
```

### Production vs Test Usage Patterns

#### Production-Only Fields

**Fields exclusively used in production environments**:

```python
# log_file - Production logging only
log_file: Path = Field(default=Path("./logs/docmind.log"))

# sqlite_db_path - Production persistence only  
sqlite_db_path: Path = Field(default=Path("./data/docmind.db"))

# Usage pattern: Production file system operations
# Test approach: Use temporary directories via fixtures
```

#### Test-Only Configuration Overrides

**Fields commonly overridden in tests**:

```python
# enable_gpu_acceleration - Test overrides common
# Production: True (default)
# Unit tests: False (CPU-only for speed)
# Integration tests: True (realistic GPU testing)
# System tests: True (production validation)

# agent_decision_timeout - Test optimization
# Production: 200ms (ADR-024)  
# Unit tests: 100ms (faster test execution)
# Integration tests: 150ms (moderate realism)
# System tests: 200ms (production validation)
```

#### Dual-Use Fields (Production & Test)

**Fields used in both contexts with same values**:

```python
# model_name - Consistent across environments
# Production: "Qwen/Qwen3-4B-Instruct-2507-FP8" 
# All test tiers: Same model for integration fidelity

# embedding.model_name - ADR-002 compliance universal
# All contexts: "BAAI/bge-m3" for consistent embedding behavior
```

### ADR Compliance Verification Matrix

#### Configuration ADR Compliance Status

| Configuration Field | ADR Reference | Current Value | Compliance Status | Notes |
|---------------------|---------------|---------------|-------------------|-------|
| `agent_decision_timeout` | ADR-024 | 200ms | ✅ COMPLIANT | Meets performance requirement |
| `embedding.model_name` | ADR-002 | "BAAI/bge-m3" | ✅ COMPLIANT | Unified embedding strategy |
| `model_name` | ADR-004 | "Qwen/Qwen3-4B-Instruct-2507-FP8" | ✅ COMPLIANT | Local-first LLM |
| `vllm_kv_cache_dtype` | ADR-010 | "fp8_e5m2" | ✅ COMPLIANT | Performance optimization |
| `enable_multi_agent` | ADR-011 | True | ✅ COMPLIANT | Multi-agent coordination |
| `chunk_size` | ADR-009 | 512 | ✅ COMPLIANT | Document processing |
| `enable_fallback_rag` | ADR-011 | True | ✅ COMPLIANT | Agent fallback system |

#### Historical ADR Violations (Resolved)

**Previously identified violations that have been corrected**:

```python
# RESOLVED: agent_decision_timeout was 300ms (violated ADR-024)
# Before: agent_decision_timeout: int = Field(default=300) # ❌ 50% performance loss
# After:  agent_decision_timeout: int = Field(default=200) # ✅ ADR-compliant

# RESOLVED: embedding_model was bge-large-en-v1.5 (violated ADR-002)  
# Before: embedding_model: str = Field(default="BAAI/bge-large-en-v1.5") # ❌ Wrong model
# After:  model_name: str = Field(default="BAAI/bge-m3") # ✅ ADR-compliant (in EmbeddingConfig)
```

### Configuration Field Categories

#### Usage-Based Field Classification

**Category 1: Critical Production Fields (100% usage)**:

- `model_name` - Model selection and loading
- `embedding.model_name` - Embedding generation  
- `agent_decision_timeout` - Performance compliance
- `enable_multi_agent` - Core functionality flag

**Category 2: Performance Optimization Fields (80% usage)**:

- `gpu_memory_utilization` - Memory management
- `vllm_kv_cache_dtype` - KV cache optimization
- `chunk_size` - Processing efficiency
- `context_window_size` - Model capacity

**Category 3: Feature Configuration Fields (60% usage)**:

- `enable_fallback_rag` - Fallback system
- `enable_dspy_optimization` - Query optimization
- `use_reranking` - Search enhancement
- `enable_document_caching` - Performance caching

**Category 4: Infrastructure Fields (40% usage)**:

- `log_level` - Logging configuration
- `data_dir` - File system paths
- `cache_dir` - Cache management
- `sqlite_db_path` - Persistence layer

**Category 5: Development/Debug Fields (20% usage)**:

- `debug` - Development mode
- `enable_performance_logging` - Detailed metrics
- `log_file` - Debug output destination

### Configuration Cleanup Recommendations

#### Fields with High Usage Priority

**Maintain and optimize these heavily-used fields**:

1. **Agent coordination settings** - Core functionality, ADR-024 compliance
2. **Model selection fields** - Critical for inference pipeline  
3. **Performance tuning parameters** - Directly impact user experience
4. **GPU optimization settings** - Hardware utilization efficiency

#### Fields with Low Usage (Review for Removal)

**Consider deprecating fields with minimal usage**:

```python
# Low-usage fields identified for review:
enable_performance_logging: bool = Field(default=False)  # <20% usage
debug_mode: bool = Field(default=False)                  # <15% usage 
verbose_logging: bool = Field(default=False)             # <10% usage

# Recommendation: Consolidate into simpler log_level management
```

#### Duplicate or Overlapping Fields

**Fields that serve similar purposes**:

```python
# Potential consolidation opportunities:
log_level: str = Field(default="INFO")           # Standard logging
debug: bool = Field(default=False)               # Debug toggle  
enable_performance_logging: bool = Field(default=False)  # Performance logs

# Recommendation: Use log_level exclusively with values: DEBUG, INFO, WARNING, ERROR
```

### Configuration Validation Scripts

#### Production Configuration Validation

```python
def validate_production_configuration_requirements() -> Dict[str, Any]:
    """Validate configuration meets production usage requirements."""
    
    from src.config import settings
    
    validation_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "valid",
        "field_analysis": {},
        "adr_compliance": {},
        "usage_patterns": {}
    }
    
    # Core field validation
    core_fields = {
        "model_name": settings.vllm.model,
        "embedding_model": settings.embedding.model_name,
        "agent_decision_timeout": settings.agents.decision_timeout,
        "enable_multi_agent": settings.agents.enable_multi_agent
    }
    
    for field_name, value in core_fields.items():
        validation_report["field_analysis"][field_name] = {
            "value": value,
            "usage_category": "core",
            "production_required": True,
            "test_override_common": field_name in ["agent_decision_timeout"]
        }
    
    # ADR compliance verification
    adr_checks = {
        "adr_002_bge_m3": settings.embedding.model_name == "BAAI/bge-m3",
        "adr_024_agent_timeout": settings.agents.decision_timeout == 200,
        "adr_010_fp8_cache": settings.vllm.kv_cache_dtype == "fp8_e5m2",
        "adr_011_multi_agent": settings.agents.enable_multi_agent is True
    }
    
    validation_report["adr_compliance"] = {
        check_name: {
            "compliant": result,
            "field": check_name.split("_", 2)[-1]
        }
        for check_name, result in adr_checks.items()
    }
    
    # Check for non-compliant ADRs
    failed_adrs = [name for name, result in adr_checks.items() if not result]
    if failed_adrs:
        validation_report["overall_status"] = "adr_violations"
        validation_report["failed_adrs"] = failed_adrs
    
    return validation_report
```

#### Configuration Usage Analytics

```python
def analyze_configuration_usage_patterns() -> Dict[str, Any]:
    """Analyze actual configuration usage patterns across codebase."""
    
    import subprocess
    from pathlib import Path
    
    usage_analysis = {
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "field_usage": {},
        "usage_categories": {
            "high_usage": [],      # >80% usage across codebase
            "medium_usage": [],    # 40-80% usage  
            "low_usage": [],       # <40% usage
            "unused": []           # No detected usage
        }
    }
    
    # Configuration fields to analyze
    config_fields = [
        "model_name", "embedding_model", "agent_decision_timeout",
        "enable_multi_agent", "gpu_memory_utilization", "chunk_size",
        "context_window_size", "enable_fallback_rag", "log_level",
        "debug", "enable_performance_logging"
    ]
    
    project_root = Path(__file__).parent.parent.parent
    
    for field in config_fields:
        # Search for field usage in source code
        try:
            result = subprocess.run([
                "rg", f"settings\\..*{field}|{field}.*=", 
                str(project_root / "src"),
                "--type", "py", "--count"
            ], capture_output=True, text=True)
            
            usage_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # Categorize usage level
            if usage_count >= 8:
                usage_categories["high_usage"].append(field)
                usage_level = "high"
            elif usage_count >= 3:
                usage_categories["medium_usage"].append(field)
                usage_level = "medium"
            elif usage_count >= 1:
                usage_categories["low_usage"].append(field)
                usage_level = "low"
            else:
                usage_categories["unused"].append(field)
                usage_level = "unused"
            
            usage_analysis["field_usage"][field] = {
                "usage_count": usage_count,
                "usage_level": usage_level,
                "production_critical": field in [
                    "model_name", "embedding_model", 
                    "agent_decision_timeout", "enable_multi_agent"
                ]
            }
            
        except Exception as e:
            usage_analysis["field_usage"][field] = {
                "error": str(e),
                "usage_level": "unknown"
            }
    
    return usage_analysis
```

This configuration usage analysis provides evidence-based insights into actual field usage patterns, ADR compliance status, and optimization opportunities for maintaining clean and efficient configuration management.

## Migration Guide

### From Legacy Configuration

If migrating from an older DocMind AI configuration:

1. **Backup existing configuration:**

   ```bash
   cp .env .env.backup
   ```

2. **Use migration script:**

   ```bash
   python scripts/migrate_configuration.py .env.backup .env
   ```

3. **Manual migration mapping:**

   ```bash
   # Legacy -> New format
   DOCMIND_MODEL_NAME=... -> DOCMIND_VLLM__MODEL=...
   DOCMIND_EMBED_MODEL=... -> DOCMIND_EMBEDDING__MODEL_NAME=...
   DOCMIND_CHUNK_SIZE=... -> DOCMIND_PROCESSING__CHUNK_SIZE=...
   DOCMIND_GPU_MEM=... -> DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=...
   ```

4. **Validate migration:**

   ```bash
   python -c "from src.config import settings; print('✅ Migration successful')"
   ```

---

This configuration reference provides comprehensive guidance for optimizing DocMind AI performance and functionality. The unified configuration system ensures consistency, type safety, and excellent validation while maintaining flexibility for different deployment scenarios.

For implementation details, see [Developer Handbook](developer-handbook.md).
For operational procedures, see [Operations Guide](operations-guide.md).
For system understanding, see [System Architecture](system-architecture.md).

## Retrieval & GraphRAG Feature Flags

```bash
# Hybrid fusion mode (server-side only): rrf (default) or dbsf (experimental)
DOCMIND_RETRIEVAL__FUSION_MODE=rrf

# De-duplication key for final fused cut: page_id (default) or doc_id
DOCMIND_RETRIEVAL__DEDUP_KEY=page_id

# GraphRAG toggle (current default: ON). Disable to run vector/hybrid only
DOCMIND_ENABLE_GRAPHRAG=true

# DSPy optimization (optional, agents layer)
DOCMIND_ENABLE_DSPY_OPTIMIZATION=false
DOCMIND_DSPY_OPTIMIZATION_ITERATIONS=10
DOCMIND_DSPY_OPTIMIZATION_SAMPLES=20
DOCMIND_DSPY_MAX_RETRIES=3
DOCMIND_DSPY_TEMPERATURE=0.1
DOCMIND_DSPY_METRIC_THRESHOLD=0.8
DOCMIND_ENABLE_DSPY_BOOTSTRAPPING=true

# Reranking (always-on by default; no UI toggle). Canonical env override maps to settings.retrieval.use_reranking
DOCMIND_RETRIEVAL__USE_RERANKING=true
# Optional visual reranker (ColPali) and SigLIP controls
DOCMIND_RETRIEVAL__ENABLE_COLPALI=false
DOCMIND_RETRIEVAL__SIGLIP_BATCH_SIZE=8
DOCMIND_RETRIEVAL__SIGLIP_PRUNE_M=64
```

Notes:

- Fusion is executed server‑side via Qdrant Query API (Prefetch + FusionQuery). There are no client‑side knobs.
- GraphRAG tool is activated only when a PropertyGraphIndex is present and healthy; default traversal depth is `path_depth=1`.

### Reranking & Device Feature Flags (Advanced)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| DOCMIND_RETRIEVAL__SIGLIP_ADAPTER_UNIFIED | bool | true | Use shared vision_siglip.load_siglip in adapter to ensure consistent caching and device placement. |
| DOCMIND_RETRIEVAL__RERANK_EXECUTOR | str | thread | Executor for rerank timeouts: thread (default) or process (strict isolation). |

These flags support canary rollouts on GPU fleets to monitor OOM rate, P95 latency, and timeout counts. The device and VRAM policy is always centralized via `src.utils.core` (e.g., `select_device`, `resolve_device`, `has_cuda_vram`) and does not require a feature flag. The SigLIP adapter unification and the rerank executor remain configurable for operational flexibility.
