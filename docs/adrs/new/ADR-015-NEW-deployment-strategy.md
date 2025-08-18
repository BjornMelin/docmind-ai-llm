# ADR-015-NEW: Docker-First Local Deployment

## Title

Simple Docker Deployment for Local Streamlit App

## Version/Date

3.0 / 2025-08-17

## Status

Accepted

## Description

Single Docker image deployment with docker-compose for easy local setup. Users run `docker-compose up` and the app starts. No complex profiles, no hardware detection scripts, just environment variables for configuration.

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

# Environment variables with sensible defaults
ENV PYTHONPATH=/app
ENV DOCMIND_MODEL=${DOCMIND_MODEL:-Qwen/Qwen3-14B-Instruct}
ENV DOCMIND_CONTEXT_LENGTH=${DOCMIND_CONTEXT_LENGTH:-32768}
ENV DOCMIND_DEVICE=${DOCMIND_DEVICE:-cpu}
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
      - DOCMIND_MODEL=${DOCMIND_MODEL:-Qwen/Qwen3-14B-Instruct}
      - DOCMIND_DEVICE=${DOCMIND_DEVICE:-cpu}
      - DOCMIND_CONTEXT_LENGTH=${DOCMIND_CONTEXT_LENGTH:-32768}
    restart: unless-stopped
    
    # Optional: GPU support (uncomment if needed)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

### Simple .env.example

```bash
# Copy to .env and adjust as needed

# Model settings
DOCMIND_MODEL=Qwen/Qwen3-14B-Instruct
DOCMIND_CONTEXT_LENGTH=131072  # 128K native context

# LLM Provider (auto, ollama, llamacpp, vllm)
DOCMIND_LLM_PROVIDER=auto  # Automatic selection based on hardware

# Hardware (cpu or cuda)
DOCMIND_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# Provider-specific optimizations
# Ollama
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0

# llama.cpp
LLAMA_CUBLAS=1
LLAMA_FLASH_ATTN=1

# vLLM (for multi-GPU)
# VLLM_ATTENTION_BACKEND=FLASH_ATTN
# CUDA_VISIBLE_DEVICES=0,1
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

## Performance Targets

- **Startup Time**: <2 minutes with cached models
- **Image Size**: <2GB (using python:3.11-slim base)
- **Memory Usage**: Depends on model (8GB for small, 16GB for large)

## Changelog

- **3.0 (2025-08-17)**: FINALIZED - Updated with Qwen3-14B-Instruct and 128K context, accepted status
- **2.0 (2025-08-17)**: SIMPLIFIED - Removed all complexity, single image deployment
- **1.0 (2025-01-16)**: Original over-engineered version with profiles and detection
