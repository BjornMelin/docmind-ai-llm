# GPU Setup and Optimization

## Overview

This guide provides comprehensive GPU configuration for DocMind AI with FastEmbed, Qdrant GPU acceleration, and vLLM integration. The setup targets **100x+ performance improvements** for embedding generation and vector search workloads.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability 6.0+ 

- Ubuntu 24.04 (recommended) or compatible Linux distribution

- NVIDIA drivers version 545.0+ (CUDA 12.0+ support)

- Docker with NVIDIA Container Toolkit (for containerized deployment)

## Quick Start

### 1. Automated Setup

Run the provided setup script to install all GPU components:

```bash

# Make executable and run
chmod +x gpu_setup.sh
./gpu_setup.sh
```

### 2. Validation

After installation, validate your setup:

```bash

# Run comprehensive validation
python3 gpu_validation.py

# Test performance improvements
python3 test_gpu_performance.py
```

### 3. Environment Configuration

Copy and customize the GPU environment settings:

```bash
cp .env.example .env

# Edit .env with your GPU-specific settings
```

## Installation Options

### Option 1: Basic GPU Setup (Recommended)

```bash

# Install with basic GPU acceleration
uv sync --extra gpu
```

Includes:

- PyTorch 2.7.0 with CUDA support

- vLLM 0.9.2 with GPU acceleration

- FastEmbed GPU support

- llama-cpp-python (configure with CMAKE_ARGS)

### Option 2: Advanced GPU Setup

```bash

# Install with GPU monitoring and advanced tools
uv sync --extra gpu-full
```

Additional features:

- GPU utilization monitoring (gpustat)

- NVIDIA ML Python bindings

- Performance tracking tools

### Option 3: Custom CUDA Installation

For specific CUDA versions or manual control:

```bash

# Manual PyTorch with specific CUDA version
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install vLLM with CUDA support
uv pip install vllm~=0.9.2 --extra-index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
uv sync --extra gpu
```

## Manual Installation Steps

### 1. CUDA Toolkit

```bash

# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.8 toolkit
sudo apt-get install -y cuda-toolkit-12-8
```

### 2. cuDNN Installation

```bash

# Install cuDNN development libraries
sudo apt-get install -y libcudnn9-dev libcudnn9-cuda-12
```

### 3. NVIDIA Container Toolkit

```bash

# Add repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Environment Configuration

### CUDA Environment Variables

Add to your `.bashrc` or `.zshrc`:

```bash

# CUDA Configuration
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# vLLM Optimization
export VLLM_TORCH_BACKEND=cu128
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# FastEmbed GPU Settings
export FASTEMBED_CACHE_PATH=/tmp/fastembed_cache
export ONNXRUNTIME_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

### Application Environment (.env)

```bash

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
VLLM_TORCH_BACKEND=cu128
VLLM_ATTENTION_BACKEND=FLASH_ATTN
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# FastEmbed GPU Settings
FASTEMBED_CACHE_PATH=/tmp/fastembed_cache
ONNXRUNTIME_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
FASTEMBED_GPU_BATCH_SIZE=512

# Qdrant GPU Settings
QDRANT_GPU_INDEXING=1
QDRANT_GPU_FORCE_HALF_PRECISION=true
QDRANT_GPU_GROUPS_COUNT=512

# Performance Monitoring
GPU_MEMORY_UTILIZATION=0.9
ENABLE_GPU_MONITORING=true
```

## Docker Configuration

### Basic Docker Compose

The project includes a GPU-optimized `docker-compose.yml`:

```yaml
version: "3.8"

services:
  app:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_TORCH_BACKEND=cu128
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ipc: host
    shm_size: '2gb'

  qdrant:
    image: qdrant/qdrant:gpu-nvidia-latest
    runtime: nvidia
    environment:
      - QDRANT__GPU__INDEXING=1
      - QDRANT__GPU__FORCE_HALF_PRECISION=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Starting GPU Services

```bash

# Start with GPU acceleration
docker compose up

# Verify GPU access in containers
docker exec -it docmind-ai-llm-app-1 nvidia-smi
```

## Verification and Testing

### Basic Verification Commands

```bash

# Check NVIDIA driver
nvidia-smi

# Verify CUDA toolkit
nvcc --version

# Test PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Test vLLM import
python3 -c "import vllm; print('vLLM successfully imported')"

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
```

### Performance Benchmarking

The included validation script runs comprehensive performance tests:

```bash

# Run full validation suite
python3 gpu_validation.py

# Expected output:

# 🚀 DocMind AI GPU Infrastructure Validation

# ✅ NVIDIA Driver: Working

# ✅ CUDA Toolkit: Version 12.8 installed

# ✅ PyTorch CUDA: Available with device access

# ✅ FastEmbed GPU: Working, 10x+ speedup

# ✅ Docker GPU: Container access confirmed

# 🎉 GPU infrastructure ready for 100x performance improvements!
```

## Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch

```bash

# Check versions
nvidia-smi  # Driver CUDA version
nvcc --version  # Toolkit version

# Reinstall with correct CUDA
uv pip uninstall torch torchvision torchaudio vllm
uv sync --extra gpu
```

#### 2. Docker GPU Access Denied

```bash

# Restart Docker service
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
```

#### 3. Memory Issues

```bash

# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### 4. Import Errors

```bash

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -r {} +

# Reinstall dependencies
uv pip install --force-reinstall torch fastembed-gpu onnxruntime-gpu
```

### Performance Monitoring

```bash

# Real-time GPU monitoring
gpustat -i 1

# NVIDIA system monitoring
watch -n 1 nvidia-smi

# Memory usage tracking
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Expected Performance Improvements

With this optimized configuration, you should expect:

- **100x+ faster embedding generation** with FastEmbed GPU vs CPU

- **25-40% faster inference** compared to default PyTorch installation

- **Better memory utilization** with proper CUDA memory management

- **Improved stability** with matched PyTorch/vLLM versions

- **Enhanced monitoring** capabilities with gpu-full setup

### Benchmarking Results

Typical performance improvements on NVIDIA RTX 4090:

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| FastEmbed (500 docs) | 4.33s | 43.4ms | 100x |
| PyTorch Matrix Ops | 2.1s | 45ms | 47x |
| Vector Search | 850ms | 12ms | 71x |

## Model Recommendations

### Optimized Models for GPU

- **Embedding**: `BAAI/bge-large-en-v1.5` (GPU-optimized)

- **Reranker**: `jinaai/jina-reranker-v1-turbo-en` (8K context)

- **Sparse**: `prithvida/Splade_PP_en_v1` (term expansion)

### vLLM Configuration

```python
from vllm import LLM, SamplingParams

# Production-ready configuration
llm = LLM(
    model="your-model-name",
    tensor_parallel_size=1,          # Adjust for multi-GPU
    gpu_memory_utilization=0.9,      # Use 90% of GPU memory
    trust_remote_code=True,
    max_model_len=4096,             # Adjust based on needs
    enforce_eager=False,            # Enable CUDA graphs
    quantization=None,              # Add "awq" or "gptq" if needed
    attention_backend="FLASH_ATTN", # Use FlashAttention
)
```

## Support and Resources

For issues with GPU setup:

1. Run the validation commands above
2. Check environment variables are properly set
3. Test with the provided performance scripts
4. Consult the troubleshooting section
5. Review vLLM documentation: https://docs.vllm.ai/

## Related Documentation

- [Architecture Overview](../adrs/003-gpu-optimization.md) - GPU optimization decisions

- [Deployment Guide](deployment.md) - Production deployment with GPU

- [Testing Guide](testing.md) - GPU-specific testing procedures
