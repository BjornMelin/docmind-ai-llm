# System Tests for DocMind AI

This directory contains comprehensive system-level tests for the DocMind AI pipeline, designed to validate the complete system with real models and GPU acceleration.

## Test Structure

### test_full_pipeline.py
Complete end-to-end pipeline validation with real GPU resources:

**TestGPUPipelineValidation:**
- `test_full_multimodal_pipeline_gpu`: Full pipeline with real BGE-M3 embeddings and multi-agent system
- `test_vram_usage_bounds`: Memory usage validation (14GB limit for RTX 4090)
- `test_performance_targets`: Decode (120+ tok/s) and prefill (900+ tok/s) throughput validation
- `test_resource_cleanup_under_load`: Concurrent operations and memory management

**TestSystemConfiguration:**
- `test_gpu_detection_and_configuration`: Hardware detection and GPU setup
- `test_settings_validation_for_gpu`: Configuration validation for GPU operations
- `test_gpu_memory_allocation`: GPU memory allocation and cleanup

**TestSystemIntegration:**
- `test_graceful_degradation_on_errors`: Error handling and recovery
- `test_configuration_compatibility`: Cross-component configuration validation

### test_model_loading.py
Model-specific loading and optimization tests:

**TestCLIPModelLoading:**
- `test_clip_model_loading`: CLIP model loading and memory footprint (<2GB)
- `test_clip_inference_performance`: Inference throughput validation (10+ texts/sec)
- `test_clip_memory_optimization`: FP16/BF16 precision optimization

**TestVLLMModelLoading:**
- `test_vllm_model_loading`: vLLM engine initialization and memory usage (<12GB)
- `test_vllm_fp8_optimization`: FP8 quantization validation (40-60% memory reduction)
- `test_vllm_context_scaling`: Memory scaling with context sizes (16K-128K)

**TestEmbeddingModelLoading:**
- `test_bge_m3_embedding_loading`: BGE-M3 model loading and memory usage (<4GB)
- `test_embedding_batch_processing`: Batch processing performance optimization

**TestModelResourceManagement:**
- `test_model_memory_cleanup`: Memory cleanup validation across multiple models
- `test_concurrent_model_loading`: Concurrent loading and resource contention
- `test_model_configuration_validation`: Configuration validation and error handling

## Running System Tests

### Prerequisites
- GPU with CUDA support (RTX 4090 recommended)
- At least 16GB VRAM
- CUDA drivers and PyTorch with CUDA support

### Test Execution

**Run all system tests (GPU required):**
```bash
uv run pytest tests/system/ -v -s -m system
```

**Run only GPU pipeline tests:**
```bash
uv run pytest tests/system/test_full_pipeline.py -v -s -m "system and requires_gpu"
```

**Run only model loading tests:**
```bash
uv run pytest tests/system/test_model_loading.py -v -s -m "system and slow"
```

**Skip GPU-dependent tests (CPU only):**
```bash
uv run pytest tests/system/ -v -s -m "system and not requires_gpu"
```

**Run with specific timeout:**
```bash
uv run pytest tests/system/ -v -s --timeout=600  # 10 minute timeout
```

### Performance Targets

These tests validate the following performance targets:

**RTX 4090 Targets:**
- vLLM Decode Throughput: 120-180 tokens/second
- vLLM Prefill Throughput: 900-1400 tokens/second
- Total VRAM Usage: <14GB
- Model Loading Time: <90 seconds per model
- CLIP Inference: >10 texts/second
- BGE-M3 Batch Processing: Variable based on batch size

**Memory Limits:**
- CLIP Model: <2GB VRAM
- BGE-M3 Embeddings: <4GB VRAM
- vLLM Engine: <12GB VRAM
- Total Peak Usage: <15GB VRAM

### Test Markers

System tests use the following pytest markers:

- `@pytest.mark.system`: Core system-level tests
- `@pytest.mark.requires_gpu`: Requires GPU hardware
- `@pytest.mark.slow`: Long-running tests (>30 seconds)
- `@pytest.mark.timeout(N)`: Specific timeout in seconds

### Automatic Skipping

Tests are automatically skipped when:
- GPU is not available (`torch.cuda.is_available() == False`)
- Required models are not accessible
- Insufficient VRAM detected
- Network connectivity issues

### CI/CD Integration

For CI/CD environments without GPU:
```bash
# Skip GPU tests in CI
uv run pytest tests/system/ -m "system and not requires_gpu"

# Or skip system tests entirely
uv run pytest tests/ -m "not system"
```

### Troubleshooting

**Common Issues:**

1. **GPU Out of Memory:**
   ```bash
   # Reduce GPU memory utilization in settings
   export VLLM_GPU_MEMORY_UTILIZATION=0.6
   ```

2. **Model Loading Timeouts:**
   ```bash
   # Increase timeout for slow networks
   uv run pytest tests/system/ --timeout=900
   ```

3. **Import Errors:**
   ```bash
   # Ensure all dependencies are installed
   uv sync --extra gpu
   ```

### Performance Monitoring

System tests include built-in performance monitoring:
- GPU memory usage tracking
- Inference throughput measurement
- Model loading time validation
- Resource cleanup verification

These metrics are logged during test execution and can be used for performance regression detection.

## Architecture Validation

System tests validate the complete DocMind AI architecture:

1. **Multi-Agent Coordination:** 5-agent system with LangGraph supervisor
2. **Hybrid Retrieval:** BGE-M3 unified embeddings with RRF fusion
3. **GPU Optimization:** FP8 quantization and FlashInfer backend
4. **Memory Management:** Efficient VRAM utilization and cleanup
5. **Performance Targets:** Real-world throughput and latency requirements

These tests ensure the system meets production requirements for local AI document analysis with RTX 4090 hardware.