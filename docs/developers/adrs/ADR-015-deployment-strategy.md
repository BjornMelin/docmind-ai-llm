# ADR-015: Docker-First Local Deployment

## Title

Simple Docker Deployment for Local Streamlit App

## Version/Date

5.0 / 2025-08-19

## Status

Accepted

## Description

Single Docker image deployment with docker-compose for easy local setup leveraging 128K context windows through Qwen3-4B-Instruct-2507-FP8 with FP8 KV cache optimization. Users run `docker-compose up` and the app starts with constrained context capability at ~12-14GB VRAM usage (hardware-limited from 262K native). No complex profiles, just environment variables for 128K context configuration with vLLM/FlashInfer support.

## Context

DocMind AI is a **local Streamlit app**, not an enterprise cloud service. Users want to:

1. Download the repository
2. Run `docker-compose up`
3. Use the app

That's it. No need for multi-stage builds, hardware profiling, or complex orchestration.

## Implementation Reference

For detailed vLLM service configuration, systemd templates, and production deployment procedures, see:

- [FP8 Supervisor Integration Guide](../archived/implementation-plans/fp8-supervisor-integration-guide-v1.0.md)

## Decision

We will implement **dead-simple Docker deployment**:

### Single Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p /app/data/{models,documents,cache}

# Environment variables with RTX 4090 Laptop optimized defaults
ENV PYTHONPATH=/app
ENV DOCMIND_MODEL=${DOCMIND_MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}
ENV DOCMIND_CONTEXT_LENGTH=${DOCMIND_CONTEXT_LENGTH:-131072}
ENV DOCMIND_KV_CACHE_DTYPE=${DOCMIND_KV_CACHE_DTYPE:-fp8_e5m2}
ENV DOCMIND_LLM_PROVIDER=${DOCMIND_LLM_PROVIDER:-vllm}
ENV DOCMIND_DEVICE=${DOCMIND_DEVICE:-cuda}
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Simple docker-compose.yml

```yaml
version: '3.8'

services:
  docmind:
    build: .
    container_name: docmind-ai
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - DOCMIND_MODEL=${DOCMIND_MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}
      - DOCMIND_DEVICE=${DOCMIND_DEVICE:-cuda}
      - DOCMIND_CONTEXT_LENGTH=${DOCMIND_CONTEXT_LENGTH:-131072}
      - DOCMIND_KV_CACHE_DTYPE=${DOCMIND_KV_CACHE_DTYPE:-fp8_e5m2}
      - DOCMIND_LLM_PROVIDER=${DOCMIND_LLM_PROVIDER:-vllm}
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_ATTENTION_BACKEND=FLASHINFER
      - VLLM_USE_CUDNN_PREFILL=1
    restart: unless-stopped
    
    # GPU support for RTX 4090 Laptop (16GB VRAM)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Complete Configuration Reference (.env.example)

```bash
# Copy to .env and adjust as needed
# NOTE: This is the COMPLETE configuration reference for DocMind AI
# Centralized configuration was deliberately rejected (see archived ADR-024)

# ========================================
# Core Model Settings - 128K CONTEXT (Hardware Constrained)
# ========================================
DOCMIND_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_CONTEXT_LENGTH=131072  # 128K context (hardware-constrained from 262K native)
DOCMIND_KV_CACHE_DTYPE=fp8_e5m2    # FP8 KV cache enables 128K context
DOCMIND_QUANTIZATION=fp8       # FP8 quantization

# LLM Provider (vllm, lmdeploy, llamacpp, ollama)
DOCMIND_LLM_PROVIDER=vllm  # Recommended for FP8 KV cache with FlashInfer

# Hardware (cpu or cuda)
DOCMIND_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# ========================================
# Database & Storage
# ========================================
DATABASE_URL=sqlite:///data/docmind.db
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

# ========================================
# Feature Flags (Experimental)
# ========================================
ENABLE_DSPY=false        # DSPy prompt optimization (ADR-018)
ENABLE_GRAPHRAG=false    # GraphRAG functionality (ADR-019)
ENABLE_MULTIMODAL=true   # Multimodal document processing

# ========================================
# Performance Settings
# ========================================
MAX_WORKERS=4                    # Thread pool size
CACHE_SIZE_LIMIT_MB=1000        # Cache size limit
MEMORY_LIMIT_MB=8192            # Memory usage limit
BATCH_SIZE=32                   # Embedding batch size

# ========================================
# Provider-Specific Optimizations - FP8 KV Cache
# ========================================
# vLLM (RECOMMENDED with FlashInfer)
VLLM_ATTENTION_BACKEND=FLASHINFER    # FlashInfer backend for optimized attention
VLLM_USE_CUDNN_PREFILL=1            # Use cuDNN for prefill operations
VLLM_KV_CACHE_DTYPE=fp8_e5m2        # FP8 KV cache quantization
VLLM_GPU_MEMORY_UTILIZATION=0.95    # High memory utilization for 16GB VRAM

# LMDeploy (alternative with FP8 support)
LMDEPLOY_QUANT_POLICY=8         # Close to FP8 quantization
LMDEPLOY_CACHE_MAX_ENTRY=0.9    # Use 90% of cache capacity

# llama.cpp (if using GGUF fallback)
LLAMA_TYPE_K=8                  # 8-bit quantization for keys
LLAMA_TYPE_V=8                  # 8-bit quantization for values

# Ollama (if using local models)
OLLAMA_KV_CACHE_TYPE=q8_0       # 8-bit cache quantization

# ========================================
# Document Processing
# ========================================
CHUNK_SIZE=1024                 # Text chunk size in tokens
CHUNK_OVERLAP=200              # Chunk overlap in tokens
MAX_FILE_SIZE_MB=100           # Maximum file size
SUPPORTED_EXTENSIONS=".pdf,.docx,.txt,.md,.json,.xml,.rtf,.csv,.msg,.pptx,.odt,.epub"

# ========================================
# Retrieval Settings
# ========================================
SIMILARITY_TOP_K=10            # Number of documents to retrieve
SIMILARITY_CUTOFF=0.7          # Similarity threshold
ENABLE_HYBRID_SEARCH=true      # Dense + sparse retrieval
ENABLE_RERANKING=true          # ColBERT reranking
RRF_ALPHA=0.7                 # RRF fusion weight

# ========================================
# Security & Logging
# ========================================
LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
ENABLE_AUTH=false              # User authentication
SECRET_KEY=your-secret-key-change-this
RATE_LIMIT_PER_MINUTE=60       # API rate limiting
```

### One-Line Setup

```bash
# Clone and run
git clone https://github.com/user/docmind-ai
cd docmind-ai
cp .env.example .env
docker-compose up
```

### Multi-Provider Docker Profiles

For users who want specific LLM providers, we offer optional Docker profiles:

```yaml
# docker-compose.providers.yml
version: '3.8'

services:
  # Profile for vLLM users (RECOMMENDED with FlashInfer)
  docmind-vllm:
    profiles: ["vllm"]
    extends:
      file: docker-compose.yml
      service: docmind
    environment:
      - DOCMIND_LLM_PROVIDER=vllm
      - VLLM_ATTENTION_BACKEND=FLASHINFER
      - VLLM_USE_CUDNN_PREFILL=1
      - VLLM_KV_CACHE_DTYPE=fp8_e5m2  # FP8 KV cache
      - VLLM_GPU_MEMORY_UTILIZATION=0.95
    volumes:
      - ./models:/app/models  # For FP8 models
      
  # Profile for LMDeploy users (alternative)
  docmind-lmdeploy:
    profiles: ["lmdeploy"]
    extends:
      file: docker-compose.yml
      service: docmind
    environment:
      - DOCMIND_LLM_PROVIDER=lmdeploy
      - LMDEPLOY_QUANT_POLICY=8  # 8-bit quantization
      - LMDEPLOY_CACHE_MAX_ENTRY=0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # Single GPU for Qwen3-4B
              capabilities: [gpu]
```

Usage:

```bash
# Use default (vLLM with FlashInfer + FP8 KV cache)
docker-compose up

# Use vLLM specifically
docker-compose --profile vllm up

# Use LMDeploy alternative
docker-compose --profile lmdeploy up
```

## Related Decisions

- **ADR-001** (Modern Agentic RAG Architecture): Deploys the complete 5-agent system architecture
- **ADR-004** (Local-First LLM Strategy): Configures Qwen3-4B-Instruct-2507-FP8 with 128K context through FP8 KV cache
- **ADR-010** (Performance Optimization Strategy): Cache configuration affects Docker deployment requirements
- **ADR-011** (Agent Orchestration Framework): Deploys the langgraph-supervisor 5-agent orchestration
- **ADR-016** (UI State Management): Deploys Streamlit UI with proper state management
- **ADR-024** (Configuration Architecture): Provides unified configuration system for deployment
- **ADR-026** (Test-Production Separation): Enables clean production deployment without test contamination

## Configuration Philosophy

DocMind AI uses **distributed, simple configuration by design**:

- **Environment Variables**: All runtime configuration through simple `.env` file
- **Streamlit Native Config**: UI configuration via `.streamlit/config.toml` (ADR-013)
- **Library Defaults**: Most components use sensible library defaults
- **No Centralization**: Centralized configuration was considered and rejected as over-engineering (see archived ADR-024)

This approach aligns with the project's core principles:

- **KISS over DRY**: Simple environment variables over complex abstractions
- **Library-first**: Use native approaches, avoid unnecessary wrappers
- **Local-first**: Optimized for single-user desktop application, not enterprise deployment

## Benefits of Simplification

- **User-Friendly**: One command to start, no complex setup
- **Maintainable**: Single Dockerfile, easy to debug
- **Flexible**: Environment variables for all configuration
- **Fast**: No multi-stage builds or complex scripts
- **Clear**: Anyone can understand what's happening

## What We Removed

- ❌ Multi-stage Docker builds (unnecessary complexity)
- ❌ Hardware detection scripts (users know their hardware)
- ❌ Multiple deployment profiles (environment variables are enough)
- ❌ Complex volume management (just map data and models)
- ❌ Elaborate health checks (simple curl is sufficient)
- ❌ 600+ lines of deployment code (now <100 lines total)

## Dependencies

- Docker
- Docker Compose
- (Optional) NVIDIA Container Toolkit for GPU

## Performance Targets (RTX 4090 Laptop)

- **Startup Time**: <45 seconds with cached FP8 models
- **Image Size**: <2GB (using python:3.11-slim base)
- **Memory Usage**: ~12-14GB VRAM for Qwen3-4B-Instruct-2507-FP8 with 128K context (hardware-constrained)
- **Model Loading**: <15 seconds for FP8 quantization from NVMe SSD
- **Performance**: 100-160 tok/s decode, 800-1300 tok/s prefill with vLLM + FlashInfer

## Changelog

- **5.1 (2025-08-20)**: **FP8 DEPLOYMENT CONFIGURATION** - Updated for Qwen3-4B-Instruct-2507-FP8 with FP8 quantization and 128K context (hardware-constrained from 262K native due to 16GB VRAM limitation). Docker configuration updated with vLLM + FlashInfer backend as recommended provider. Added VLLM_ATTENTION_BACKEND=FLASHINFER and VLLM_USE_CUDNN_PREFILL=1 environment variables. Memory usage: ~12-14GB VRAM. Performance: 100-160 tok/s decode, 800-1300 tok/s prefill. Updated all configuration files for FP8 model with accurate technical specifications.
- **5.0 (2025-08-19)**: **CONTEXT WINDOW DEPLOYMENT** - Updated for Qwen3-4B-Instruct-2507 with AWQ quantization enabling 262K context on RTX 4090 Laptop (16GB VRAM). Docker configuration updated with INT8 KV cache support through LMDeploy (default) and vLLM. Memory usage optimized to ~12.2GB VRAM. Added LMDeploy as recommended provider with --quant-policy 8. Performance improvements: <45s startup, +30% throughput with INT8. Removed YaRN dependencies, replaced with native 262K context capability.
- **4.0 (2025-08-18)**: **MAJOR HARDWARE UPGRADE** - Enhanced for RTX 4090 Laptop GPU (16GB VRAM). Updated Docker configuration with YaRN context scaling defaults (128K), Q5_K_M quantization, and CUDA as default device. Added proper GPU deployment configuration. Updated performance targets for high-end hardware.
- **3.2 (2025-08-18)**: CORRECTED - Updated Qwen3-14B-Instruct to correct official name Qwen3-14B (no separate instruct variant exists)
- **3.1 (2025-08-18)**: Enhanced deployment configuration to support 5-agent system with DSPy optimization and optional GraphRAG capabilities while maintaining single Docker image simplicity
- **3.0 (2025-08-17)**: FINALIZED - Updated with Qwen3-14B and 128K context, accepted status
- **2.0 (2025-08-17)**: SIMPLIFIED - Removed all complexity, single image deployment
- **1.0 (2025-01-16)**: Original over-engineered version with profiles and detection
