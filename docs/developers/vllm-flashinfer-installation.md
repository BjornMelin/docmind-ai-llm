# vLLM FlashInfer Installation Guide for RTX 4090

This guide provides comprehensive installation instructions for the finalized vLLM + FlashInfer stack optimized for Qwen3-4B-Instruct-2507-FP8 on NVIDIA RTX 4090 with 16GB VRAM.

## Quick Start (Recommended)

**Target Performance**: 100-160 tok/s decode, 800-1300 tok/s prefill with 128K context support

### 1. Prerequisites Check

```bash
# Verify CUDA 12.8+ installation
nvcc --version  # Should show CUDA 12.8+
nvidia-smi     # Should show RTX 4090 and driver 550.54.14+
```

### 2. Install vLLM FlashInfer Stack

```bash
# Phase 1: Install PyTorch 2.7.1 with CUDA 12.8 (FINALIZED APPROACH)
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Phase 2: Install vLLM with FlashInfer support (includes FlashInfer automatically)
uv pip install "vllm[flashinfer]>=0.10.1" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Phase 3: Install remaining GPU dependencies
uv sync --extra gpu
```

### 3. Validate Installation

```bash
# Comprehensive validation
python scripts/performance_validation.py

# Quick environment check (no model loading)
SKIP_MODEL_TEST=true python scripts/performance_validation.py
```

**Expected Result**:  SUCCESS: vLLM FlashInfer stack is properly installed and configured!

## Detailed Installation Process

### Hardware Requirements

- **GPU**: NVIDIA RTX 4090 (16GB VRAM minimum)
- **CUDA Toolkit**: 12.8+ (required for PyTorch 2.7.1)
- **NVIDIA Driver**: 550.54.14+ (confirmed compatible)
- **Compute Capability**: 8.9 (RTX 4090 specific)

### Environment Configuration

**Add to `.env` file:**

```bash
# vLLM FlashInfer Configuration
VLLM_TORCH_BACKEND=cu128
VLLM_ATTENTION_BACKEND=FLASHINFER
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false

# vLLM Server Configuration for RTX 4090 16GB
VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.85      # 13.6GB of 16GB
VLLM_MAX_MODEL_LEN=131072             # 128K context for Qwen3
VLLM_ENABLE_PREFIX_CACHING=true
VLLM_USE_V2_BLOCK_MANAGER=true
VLLM_KV_CACHE_DTYPE=fp8_e5m2         # FP8 KV cache optimization
VLLM_QUANTIZATION=fp8                 # Enable FP8 quantization
VLLM_MAX_NUM_BATCHED_TOKENS=8192     # Optimize for throughput
VLLM_MAX_NUM_SEQS=256                # Concurrent sequences

# CUDA Environment Variables
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
CUDA_HOME=/usr/local/cuda
```

## Performance Targets & Features

### Key Performance Improvements

- **Up to 2x FP8 speedup** with FlashInfer vs standard CUDA backend
- **50% RTX 4090-specific performance enhancement** with allow_fp16_qk_reduction
- **128K context support** with FP8 KV cache (12-14GB VRAM usage)
- **PyTorch 2.7.1 compatibility** confirmed in vLLM v0.10.0+ (Issue #20566 resolved)

### Target Performance Metrics

| Metric | Target Range | Expected with FlashInfer |
|--------|--------------|---------------------------|
| **Decode Speed** | 100-160 tok/s | 120-180 tok/s |
| **Prefill Speed** | 800-1300 tok/s | 900-1400 tok/s |
| **VRAM Usage** | 12-14GB | 12-14GB for 128K context |
| **Context Length** | 128K tokens | 131,072 tokens supported |

## Fallback Installation

If FlashInfer installation fails:

```bash
# Fallback: vLLM CUDA-only installation with PyTorch 2.7.1
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
uv sync --extra gpu

# Configure for standard CUDA backend
export VLLM_ATTENTION_BACKEND=XFORMERS  # or FLASH_ATTN
```

## Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch

```bash
# Check versions (need CUDA 12.8+ for PyTorch 2.7.1)
nvidia-smi  # Driver CUDA version (should be 12.8+)
nvcc --version  # Toolkit version (should be 12.8+)

# Clean reinstall with correct CUDA
uv pip uninstall torch torchvision torchaudio vllm flashinfer-python -y
# Then follow installation steps above
```

#### 2. FlashInfer Compatibility Issues

```bash
# Verify GPU compute capability (need SM 8.9 for RTX 4090)
python -c "
import torch
print(f'GPU Compute Capability: {torch.cuda.get_device_capability(0)}')
"

# Should show: (8, 9) for RTX 4090
```

#### 3. PyTorch 2.7.1 Compatibility

**CONFIRMED WORKING**: vLLM Issue #20566 was resolved in v0.10.0+ (July 2025)

```bash
# Verify compatible versions
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"  # Should be >=0.10.1
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # Should be 2.7.1
```

#### 4. Memory Issues (16GB RTX 4090)

```bash
# Reduce GPU memory utilization
export VLLM_GPU_MEMORY_UTILIZATION=0.75  # Reduce from 0.85

# Monitor GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1

# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache()"
```

## Version Compatibility Matrix

### Core Stack (FINALIZED)

| Component | Version | Compatibility Status |
|-----------|---------|---------------------|
| **PyTorch** | 2.7.1 |  Confirmed compatible |
| **vLLM** | >=0.10.1 |  FlashInfer supported |
| **FlashInfer** | Auto-installed |  RTX 4090 compatible |
| **CUDA** | 12.8+ |  Required for PyTorch 2.7.1 |
| **Driver** | 550.54.14+ |  RTX 4090 compatible |

### Dependency Resolution

**RESOLVED CONCERNS:**

-  **Version Compatibility**: Issue #20566 was resolved in vLLM v0.10.0+ (July 2025)
-  **PyTorch 2.7.1 Support**: Official vLLM documentation confirms compatibility
-  **FlashInfer Integration**: Automatic installation with `vllm[flashinfer]`
-  **RTX 4090 Support**: Compute capability 8.9 fully supported

## Performance Validation Script

The project includes a comprehensive validation script based on the research findings:

```bash
# Full validation with performance testing
python scripts/performance_validation.py

# Expected output sections:
# =� CUDA Environment - Hardware compatibility
# =% PyTorch - Version 2.7.1 validation  
# � vLLM - Version >=0.10.1 with FlashInfer
# <� FlashInfer Backend Test - Functional testing
# =� Performance Results - Throughput measurements
# <� Overall Status - Success/failure summary
```

## Next Steps After Installation

1. **Configure your model** in `.env`: `VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8`
2. **Set context length**: `VLLM_MAX_MODEL_LEN=131072` (128K context)
3. **Optimize GPU memory**: `VLLM_GPU_MEMORY_UTILIZATION=0.85` (13.6GB of 16GB)
4. **Enable FP8 optimization**: `VLLM_QUANTIZATION=fp8` and `VLLM_KV_CACHE_DTYPE=fp8_e5m2`
5. **Run validation**: `python scripts/performance_validation.py`

## Related Documentation

- [GPU Setup Guide](gpu-setup.md) - Comprehensive GPU configuration
- [Performance Validation](performance-validation.md) - Detailed benchmarks
- [Developer Setup](setup.md) - General development environment
- [Troubleshooting](../user/troubleshooting.md) - Common issues and solutions

---

**Based on**: ai-research/2025-08-20/002-vllm-cuda-stack-finalization.md  
**Target Model**: Qwen3-4B-Instruct-2507-FP8 with 128K context  
**Hardware**: NVIDIA RTX 4090 Laptop GPU (16GB VRAM)  
**Status**: Production-ready installation process
