# ADR-015: Docker-First Local Deployment

## Title

Simple Docker Deployment for Local Streamlit App

## Version/Date

4.1 / 2025-08-18

## Status

Accepted

## Description

Single Docker image deployment with docker-compose for easy local setup optimized for 32K native context and intelligent retrieval. Users run `docker-compose up` and the app starts with optimized memory configuration. No complex profiles, no hardware detection scripts, just environment variables for 32K context configuration.

## Context

DocMind AI is a **local Streamlit app**, not an enterprise cloud service. Users want to:

1. Download the repository
2. Run `docker-compose up`
3. Use the app

That's it. No need for multi-stage builds, hardware profiling, or complex orchestration.

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
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p /app/data/{models,documents,cache}

# Environment variables with RTX 4090 Laptop optimized defaults
ENV PYTHONPATH=/app
ENV DOCMIND_MODEL=${DOCMIND_MODEL:-Qwen/Qwen3-14B}
ENV DOCMIND_CONTEXT_LENGTH=${DOCMIND_CONTEXT_LENGTH:-131072}
ENV DOCMIND_ENABLE_YARN=${DOCMIND_ENABLE_YARN:-true}
ENV DOCMIND_QUANTIZATION=${DOCMIND_QUANTIZATION:-Q5_K_M}
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
      - DOCMIND_MODEL=${DOCMIND_MODEL:-Qwen/Qwen3-14B}
      - DOCMIND_DEVICE=${DOCMIND_DEVICE:-cuda}
      - DOCMIND_CONTEXT_LENGTH=${DOCMIND_CONTEXT_LENGTH:-131072}
      - DOCMIND_ENABLE_YARN=${DOCMIND_ENABLE_YARN:-true}
      - DOCMIND_QUANTIZATION=${DOCMIND_QUANTIZATION:-Q5_K_M}
      - CUDA_VISIBLE_DEVICES=0
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
# Core Model Settings
# ========================================
DOCMIND_MODEL=Qwen/Qwen3-14B
DOCMIND_CONTEXT_LENGTH=131072  # 128K native context
DOCMIND_ENABLE_YARN=true       # YaRN context extension
DOCMIND_QUANTIZATION=Q5_K_M    # Model quantization level

# LLM Provider (auto, ollama, llamacpp, vllm)
DOCMIND_LLM_PROVIDER=auto  # Automatic selection based on hardware

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
# Provider-Specific Optimizations
# ========================================
# Ollama
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0

# llama.cpp
LLAMA_CUBLAS=1
LLAMA_FLASH_ATTN=1

# vLLM (for multi-GPU)
# VLLM_ATTENTION_BACKEND=FLASH_ATTN
# CUDA_VISIBLE_DEVICES=0,1

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
  # Profile for llama.cpp users
  docmind-llamacpp:
    profiles: ["llamacpp"]
    extends:
      file: docker-compose.yml
      service: docmind
    environment:
      - DOCMIND_LLM_PROVIDER=llamacpp
      - LLAMA_CUBLAS=1
      - LLAMA_FLASH_ATTN=1
    volumes:
      - ./models:/app/models  # For GGUF models
      
  # Profile for vLLM users (multi-GPU)
  docmind-vllm:
    profiles: ["vllm"]
    extends:
      file: docker-compose.yml
      service: docmind
    environment:
      - DOCMIND_LLM_PROVIDER=vllm
      - VLLM_ATTENTION_BACKEND=FLASH_ATTN
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # Use all available GPUs
              capabilities: [gpu]
```

Usage:

```bash
# Use default (auto-select provider)
docker-compose up

# Use llama.cpp specifically
docker-compose --profile llamacpp up

# Use vLLM for multi-GPU
docker-compose --profile vllm up
```

## Related Decisions

- **ADR-001** (Modern Agentic RAG Architecture): Deploys the complete 5-agent system architecture
- **ADR-010** (Performance Optimization Strategy): Cache configuration affects Docker deployment requirements
- **ADR-011** (Agent Orchestration Framework): Deploys the langgraph-supervisor 5-agent orchestration
- **ADR-004** (Local-First LLM Strategy): Configures Qwen3-14B with 128K context support
- **ADR-016** (UI State Management): Deploys Streamlit UI with proper state management

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

- **Startup Time**: <1 minute with cached models
- **Image Size**: <2GB (using python:3.11-slim base)
- **Memory Usage**: 14GB VRAM for Qwen3-14B with 128K context
- **Model Loading**: <30 seconds for Q5_K_M quantization from SSD

## Changelog

- **4.0 (2025-08-18)**: **MAJOR HARDWARE UPGRADE** - Enhanced for RTX 4090 Laptop GPU (16GB VRAM). Updated Docker configuration with YaRN context scaling defaults (128K), Q5_K_M quantization, and CUDA as default device. Added proper GPU deployment configuration. Updated performance targets for high-end hardware.
- **3.2 (2025-08-18)**: CORRECTED - Updated Qwen3-14B-Instruct to correct official name Qwen3-14B (no separate instruct variant exists)
- **3.1 (2025-08-18)**: Enhanced deployment configuration to support 5-agent system with DSPy optimization and optional GraphRAG capabilities while maintaining single Docker image simplicity
- **3.0 (2025-08-17)**: FINALIZED - Updated with Qwen3-14B and 128K context, accepted status
- **2.0 (2025-08-17)**: SIMPLIFIED - Removed all complexity, single image deployment
- **1.0 (2025-01-16)**: Original over-engineered version with profiles and detection
