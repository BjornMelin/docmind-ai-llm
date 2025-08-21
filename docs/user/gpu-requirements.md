# GPU Requirements and Setup Guide

## Overview

DocMind AI requires a powerful GPU to run the Qwen3-4B-Instruct-2507-FP8 model with 128K context capability. This guide outlines the GPU requirements, setup procedures, and optimization strategies for optimal performance.

## Minimum Hardware Requirements

### GPU Specifications

#### Supported GPUs
- **RTX 4090** (24GB VRAM) - Recommended
- **RTX 4090 Laptop** (16GB VRAM) - Validated configuration
- **RTX 4080** (16GB VRAM) - Supported with optimization
- **RTX 3090** (24GB VRAM) - Legacy support
- **A6000** (48GB VRAM) - Professional workstation
- **H100** (80GB VRAM) - Enterprise/research

#### Minimum Requirements
- **VRAM**: 16GB minimum (RTX 4090 Laptop validated)
- **Compute Capability**: 8.0 or higher (Ampere/Ada Lovelace)
- **Memory Bandwidth**: 500+ GB/s
- **CUDA Cores**: 8,000+ recommended

### System Requirements

#### Operating System
- **Linux**: Ubuntu 20.04+ (recommended)
- **Windows**: Windows 10/11 with WSL2
- **macOS**: Not supported (CUDA required)

#### System Memory
- **Minimum**: 32GB RAM
- **Recommended**: 64GB RAM for optimal performance
- **Swap**: 16GB+ swap space recommended

#### Storage
- **Model Storage**: 50GB+ free space
- **Vector Database**: 10-100GB depending on document corpus
- **SSD**: NVMe SSD recommended for best performance

## GPU Setup Instructions

### Step 1: Verify GPU Compatibility

```bash
# Check GPU model and VRAM
nvidia-smi

# Expected output should show:
# - GPU name (RTX 4090, RTX 4090 Laptop, etc.)
# - Memory: 16384MiB or 24576MiB
# - CUDA Version: 12.8 or higher

# Check compute capability
nvidia-ml-py3 -c "
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
print(f'Compute Capability: {major}.{minor}')
"
```

### Step 2: Install CUDA 12.8+

#### Ubuntu/Linux
```bash
# Download and install CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.14_linux.run
sudo sh cuda_12.8.0_550.54.14_linux.run

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
```

#### Windows with WSL2
```bash
# Install CUDA in WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# Verify WSL2 CUDA support
nvidia-smi
nvcc --version
```

### Step 3: Install NVIDIA Driver

#### Minimum Driver Version
- **Driver Version**: 550.54.14 or higher
- **CUDA Compatibility**: Must support CUDA 12.8+

```bash
# Ubuntu - Install recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify driver installation
nvidia-smi
# Should show driver version 550.54.14+

# Alternative: Install specific driver version
sudo apt install nvidia-driver-550
```

### Step 4: Configure GPU Settings

#### GPU Performance Mode
```bash
# Set maximum performance mode
sudo nvidia-smi -pm 1

# Set maximum power limit (adjust for your GPU)
sudo nvidia-smi -pl 450  # RTX 4090: 450W, RTX 4090 Laptop: varies

# Set memory and GPU clocks to maximum
sudo nvidia-smi -ac 10001,2520  # Memory,GPU clocks (adjust as needed)

# Verify settings
nvidia-smi
```

#### Persistence Mode
```bash
# Enable persistence mode for better performance
sudo nvidia-smi -pm 1

# Verify persistence mode is enabled
nvidia-smi -q -d PERFORMANCE
```

## Performance Validation

### Quick GPU Test

```bash
# Test basic CUDA functionality
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# Simple performance test
x = torch.randn(10000, 10000).cuda()
y = torch.randn(10000, 10000).cuda()
result = torch.matmul(x, y)
print('GPU computation successful')
"
```

### DocMind AI GPU Validation

```python
# Run DocMind-specific GPU validation
from scripts.gpu_validation import validate_gpu_setup

# This will test:
# - CUDA compatibility
# - VRAM availability  
# - vLLM functionality
# - Performance benchmarks
results = validate_gpu_setup()
print(f"GPU Validation: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
```

## Optimization Strategies

### Memory Optimization

#### For 16GB VRAM (RTX 4090 Laptop)
```bash
# Environment settings for 16GB VRAM
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_CUDNN_PREFILL=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# Memory utilization settings
export DOCMIND_GPU_MEMORY_UTILIZATION=0.85
export DOCMIND_MAX_CONTEXT_LENGTH=131072
export DOCMIND_ENABLE_FP8_KV_CACHE=true
```

#### For 24GB VRAM (RTX 4090)
```bash
# More aggressive settings for higher VRAM
export DOCMIND_GPU_MEMORY_UTILIZATION=0.90
export DOCMIND_MAX_CONTEXT_LENGTH=131072
export DOCMIND_ENABLE_FP8_KV_CACHE=true
export DOCMIND_BATCH_SIZE=2  # Can handle small batches
```

### Performance Tuning

#### Power and Thermal Management
```bash
# Monitor GPU temperature
watch -n 1 nvidia-smi

# If temperatures > 85°C, consider:
# 1. Improve case ventilation
# 2. Reduce power limit: sudo nvidia-smi -pl 400
# 3. Custom fan curves
# 4. Undervolting (advanced users)
```

#### Memory Bandwidth Optimization
```bash
# Optimize memory clocks for your specific GPU
# RTX 4090: Memory can often run at 10751 MHz
# RTX 4090 Laptop: Usually limited to lower speeds

# Test memory stability with higher clocks
sudo nvidia-smi -ac 10751,2520  # Memory,GPU clocks

# Monitor for artifacts or crashes
nvidia-smi dmon -s pucvmet
```

## Multi-GPU Support (Future)

### Planned Multi-GPU Features
Currently, DocMind AI is optimized for single-GPU setups. Multi-GPU support is planned for future releases:

#### Supported Configurations (Planned)
- **2x RTX 4090**: 48GB total VRAM, parallel agent processing
- **4x RTX 3090**: 96GB total VRAM, massive context windows
- **Mixed GPU**: Different models for different tasks

#### Current Limitations
- Single GPU only
- No tensor parallelism across GPUs
- No model sharding support

## Troubleshooting

### Common GPU Issues

#### Issue: CUDA Out of Memory
```bash
# Symptoms:
# RuntimeError: CUDA out of memory

# Solutions:
# 1. Reduce memory utilization
export DOCMIND_GPU_MEMORY_UTILIZATION=0.75

# 2. Enable FP8 optimization
export DOCMIND_ENABLE_FP8_KV_CACHE=true

# 3. Reduce context window if not needed
export DOCMIND_MAX_CONTEXT_LENGTH=65536  # 64K instead of 128K

# 4. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### Issue: Low Performance
```bash
# Check GPU utilization
nvidia-smi dmon

# Should see high GPU utilization (>80%) during inference
# If low utilization:

# 1. Check power limit
nvidia-smi -q -d POWER

# 2. Check thermal throttling  
nvidia-smi -q -d TEMPERATURE

# 3. Verify driver version
nvidia-smi | grep "Driver Version"

# 4. Check for background processes
nvidia-smi pmon
```

#### Issue: Driver Compatibility
```bash
# Symptoms:
# - CUDA version mismatch
# - vLLM import errors
# - Performance degradation

# Solutions:
# 1. Update to minimum driver version
sudo ubuntu-drivers install nvidia:550

# 2. Verify CUDA compatibility
nvidia-smi | grep "CUDA Version"
# Should show 12.8 or higher

# 3. Reinstall CUDA if necessary
sudo apt remove --purge nvidia-*
sudo apt install cuda-toolkit-12-8
```

#### Issue: Temperature Problems
```bash
# Monitor temperature continuously
watch -n 1 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits'

# If consistently > 85°C:
# 1. Reduce power limit
sudo nvidia-smi -pl 350  # Reduce from default

# 2. Improve cooling
# - Check case ventilation
# - Clean dust from GPU fans
# - Consider aftermarket cooling

# 3. Reduce performance slightly
export DOCMIND_GPU_MEMORY_UTILIZATION=0.80
```

### Performance Benchmarking

#### Expected Performance Targets

For **RTX 4090 Laptop (16GB VRAM)**:
- **Decode Speed**: 100-160 tokens/second
- **Prefill Speed**: 800-1300 tokens/second  
- **VRAM Usage**: 12-14GB typical, <16GB maximum
- **Context Window**: Full 128K tokens supported
- **Temperature**: <85°C under sustained load

For **RTX 4090 (24GB VRAM)**:
- **Decode Speed**: 120-180 tokens/second
- **Prefill Speed**: 1000-1500 tokens/second
- **VRAM Usage**: 12-16GB typical
- **Context Window**: Full 128K tokens with room for batching
- **Temperature**: <80°C under sustained load

#### Benchmark Commands
```bash
# Run comprehensive GPU benchmark
python -m scripts.performance_validation --gpu-only

# Quick performance test
python -c "
from docs.developers.qwen3_fp8_configuration import run_qwen3_benchmark
import asyncio
results = asyncio.run(run_qwen3_benchmark())
"

# Monitor during benchmark
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv'
```

## Recommended GPU Configurations

### Budget Option: RTX 4090 Laptop (16GB)
- **Cost**: $2,500-3,500 (in laptop)
- **VRAM**: 16GB (meets minimum requirements)
- **Performance**: Meets all targets with FP8 optimization
- **Limitations**: No room for future expansion

### Performance Option: RTX 4090 (24GB)  
- **Cost**: $1,600-2,000
- **VRAM**: 24GB (comfortable headroom)
- **Performance**: Exceeds all targets
- **Headroom**: Can handle larger models and small batches

### Professional Option: RTX A6000 (48GB)
- **Cost**: $4,000-6,000
- **VRAM**: 48GB (massive headroom)
- **Performance**: Excellent for development and research
- **Features**: ECC memory, enterprise drivers

### Enterprise Option: H100 (80GB)
- **Cost**: $25,000-30,000
- **VRAM**: 80GB (handles any workload)
- **Performance**: Maximum possible performance
- **Use Cases**: Research, development, production deployment

## Future GPU Support

### Upcoming GPU Support
- **RTX 5090**: Expected Q1 2025, likely 32GB VRAM
- **RTX 5080**: Expected Q1 2025, likely 16-20GB VRAM
- **AMD RDNA4**: Pending CUDA alternative support

### Architecture Improvements
- **Better FP8 Support**: Next-gen GPUs will have improved FP8 performance
- **Larger Context Windows**: Hardware improvements enabling 256K+ contexts
- **Multi-GPU Scaling**: Improved tensor parallelism support

For additional GPU setup help, see [developers/gpu-setup.md](../developers/gpu-setup.md) and [troubleshooting.md](troubleshooting.md).