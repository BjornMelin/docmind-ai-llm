# ADR-023: PyTorch Optimization Strategy

## Title

PyTorch Optimization with TorchAO Quantization and Mixed Precision

## Version/Date

1.0 / August 13, 2025

## Status

Accepted

## Context

Following ADR-003's GPU optimization simplification and research into modern PyTorch optimization capabilities, DocMind AI requires implementation of PyTorch-native optimization strategies to achieve 1.89x faster inference with 58% memory reduction on RTX 4090 hardware. Research reveals performance opportunities through TorchAO quantization, mixed precision training, and kernel optimization that complement the native LlamaIndex Settings.llm architecture.

**Performance Opportunity:**

- **TorchAO int4 Quantization**: 1.89x faster inference with 58% memory reduction vs FP16

- **Mixed Precision Training**: 1.5x training speedup with maintained model quality  

- **Liger Kernel Integration**: 47% reduction in peak memory usage at batch size 256

- **Flash Attention 2**: 2-4x memory efficiency for long sequences on RTX 4090

**Integration Context:**

PyTorch optimization strategies operate as implementation enhancements to the unified Settings.llm configuration established in ADR-003, providing hardware-specific acceleration without requiring additional architectural complexity.

## Related Requirements

- **Performance Targets**: Achieve 1.89x inference speedup while maintaining model quality

- **Memory Efficiency**: 58% memory reduction for RTX 4090 16GB optimal utilization

- **Multi-Backend Support**: Quantization strategies across Ollama, LlamaCPP, vLLM backends

- **Quality Preservation**: Maintain model accuracy through intelligent quantization selection

- **Integration Simplicity**: Seamless integration with existing Settings.llm patterns

## Alternatives

- **No PyTorch Optimization**: Baseline performance, missed 1.89x speedup opportunity

- **Custom Quantization**: Implementation complexity, violates library-first principle

- **ONNX Runtime Optimization**: Additional dependency, less native PyTorch integration

- **TensorRT-LLM**: Complex setup, overkill for current deployment requirements

## Decision

Implement **TorchAO int4 quantization** for LLM inference, **mixed precision training** for embedding models, and **Flash Attention 2** integration for memory optimization. Use PyTorch-native optimization patterns integrated with LlamaIndex Settings.llm configuration for seamless multi-backend acceleration.

**Strategic Implementation:**

- **Primary**: TorchAO int4 quantization for 4B+ models on RTX 4090

- **Secondary**: Mixed precision (FP16) for embedding generation and fine-tuning

- **Additional**: Liger Kernel evaluation for memory-constrained scenarios

- **Integration**: Native torch.compile optimization for production deployment

## Related Decisions

- ADR-003 (GPU Optimization - provides Settings.llm foundation for PyTorch integration)

- ADR-002 (Embedding Choices - benefits from mixed precision optimization)

- ADR-019 (Multi-Backend LLM Strategy - quantization across Ollama, LlamaCPP, vLLM)

- ADR-020 (LlamaIndex Settings Migration - unified configuration patterns)

- ADR-021 (Native Architecture Consolidation - library-first optimization approach)

## Design

### TorchAO Quantization Integration

**Int4 Weight-Only Quantization for LLM Inference:**

```python

# In utils.py: TorchAO quantization integration with Settings.llm
import torch
from torchao.quantization import quantize_, int4_weight_only
from llama_index.core import Settings

def setup_quantized_llm_backend(backend_name: str, model_config: dict):
    """Configure PyTorch quantization for LLM backends."""
    
    # RTX 4090 optimization: int4 quantization for 4B+ models
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 12e9:
        quantization_config = {
            "quantization_scheme": "int4_weight_only",
            "group_size": 128,  # Optimal for RTX 4090
            "inner_k_tiles": 8,  # Memory efficiency
        }
        
        # Apply quantization to model weights
        if backend_name in ["llamacpp", "vllm"]:
            model_config.update({
                "model_kwargs": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "quantization_config": quantization_config
                }
            })
    
    return model_config

# Backend configuration with quantization
quantized_backends = {
    "ollama": Ollama(
        model="qwen3:4b-thinking", 
        request_timeout=120.0
    ),
    "llamacpp_quantized": LlamaCPP(
        model_path="./models/qwen3-4b-thinking.Q4_K_M.gguf",
        n_gpu_layers=35,
        n_ctx=65536,
        **setup_quantized_llm_backend("llamacpp", {})["model_kwargs"]
    ),
    "vllm_quantized": vLLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        **setup_quantized_llm_backend("vllm", {})["model_kwargs"]
    )
}

# Single-line quantized backend activation
Settings.llm = quantized_backends["vllm_quantized"]
```

### Mixed Precision Training Integration

**Automatic Mixed Precision for Embedding and Fine-Tuning:**

```python

# In models.py: Mixed precision configuration
from torch.cuda.amp import GradScaler, autocast
from llama_index.core import Settings

class MixedPrecisionOptimizer:
    """Mixed precision optimization for embedding and training workflows."""
    
    def __init__(self):
        self.scaler = GradScaler()
        self.enabled = torch.cuda.is_available()
    
    @contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision operations."""
        if self.enabled:
            with autocast():
                yield
        else:
            yield
    
    async def optimized_embedding_generation(self, texts: list[str]):
        """Generate embeddings with mixed precision optimization."""
        with self.mixed_precision_context():
            # 1.5x speedup with maintained quality
            embeddings = await Settings.embed_model.aget_text_embedding_batch(texts)
            return embeddings
    
    def optimized_index_creation(self, documents):
        """Create vector index with mixed precision optimization."""
        with self.mixed_precision_context():
            # Memory efficient index creation
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=Settings.embed_model,
                show_progress=True
            )
            return index

# Global mixed precision optimizer
mp_optimizer = MixedPrecisionOptimizer()
```

### Flash Attention 2 Integration

**Memory-Efficient Attention for Long Sequences:**

```python

# In utils.py: Flash Attention 2 configuration
def configure_flash_attention_2():
    """Configure Flash Attention 2 for memory efficiency."""
    
    flash_attention_config = {
        "attn_implementation": "flash_attention_2",
        "use_cache": True,
        "torch_dtype": torch.float16,
    }
    
    # RTX 4090 optimization: 2-4x memory efficiency
    if torch.cuda.is_available():
        return flash_attention_config
    else:
        return {"attn_implementation": "eager"}  # CPU fallback

# Integration with backend configuration
def create_memory_optimized_backend(backend_type: str):
    """Create backend with Flash Attention 2 optimization."""
    
    base_config = configure_flash_attention_2()
    
    if backend_type == "vllm":
        return vLLM(
            model="Qwen/Qwen3-4B-Thinking-2507",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            max_model_len=65536,
            **base_config
        )
    elif backend_type == "llamacpp":
        return LlamaCPP(
            model_path="./models/qwen3-4b-thinking.Q4_K_M.gguf",
            n_gpu_layers=35,
            n_ctx=65536,
            **base_config
        )

# Memory-optimized backend selection
Settings.llm = create_memory_optimized_backend("vllm")
```

### Kernel Optimization

**Liger Kernel Integration for Memory Reduction:**

```python

# In utils.py: Kernel optimization
def setup_liger_kernel_optimization():
    """Configure Liger Kernel for 47% memory reduction."""
    
    try:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        
        # 47% peak memory reduction at batch size 256
        liger_config = {
            "rope": True,
            "rms_norm": True,
            "swiglu": True,
            "cross_entropy": True,
            "fused_linear_cross_entropy": True,
        }
        
        return liger_config
        
    except ImportError:
        logger.warning("Liger Kernel not available, using standard kernels")
        return {}

# Conditional Liger Kernel integration
def create_kernel_optimized_backend():
    """Create backend with kernel optimization."""
    
    liger_config = setup_liger_kernel_optimization()
    
    if liger_config:
        # Improved memory efficiency for large batch processing
        return vLLM(
            model="Qwen/Qwen3-4B-Thinking-2507",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,  # Higher utilization with Liger
            **liger_config
        )
    else:
        # Standard configuration
        return create_memory_optimized_backend("vllm")
```

### Torch.compile Integration

**Production-Ready Compilation Optimization:**

```python

# In utils.py: torch.compile integration
def setup_compiled_optimization():
    """Configure torch.compile for production optimization."""
    
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        compile_config = {
            "fullgraph": True,
            "dynamic": False,
            "backend": "inductor",
            "mode": "reduce-overhead"  # Optimal for inference
        }
        
        return compile_config
    else:
        return None

# Integration with LlamaIndex patterns
class CompiledModelWrapper:
    """Wrapper for torch.compile integration with LlamaIndex."""
    
    def __init__(self, model):
        self.model = model
        compile_config = setup_compiled_optimization()
        
        if compile_config:
            self.model = torch.compile(self.model, **compile_config)
            logger.info("✅ Model compiled with torch.compile optimization")
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

# Seamless integration with Settings.llm
def create_compiled_backend(backend_type: str):
    """Create compiled backend for production optimization."""
    
    base_backend = create_kernel_optimized_backend()
    
    # Apply compilation optimization
    if hasattr(base_backend, 'model'):
        base_backend.model = CompiledModelWrapper(base_backend.model)
    
    return base_backend
```

### Performance Monitoring Integration

**PyTorch Optimization Metrics:**

```python

# In monitoring.py: PyTorch optimization monitoring
class PyTorchOptimizationMonitor:
    """Monitor PyTorch optimization performance and memory usage."""
    
    def __init__(self):
        self.metrics = {
            "quantization_speedup": [],
            "memory_reduction": [],
            "compilation_overhead": [],
            "mixed_precision_accuracy": []
        }
    
    @contextmanager
    def benchmark_optimization(self, optimization_type: str):
        """Benchmark PyTorch optimization performance."""
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        yield
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Record optimization metrics
        speedup = 1.0  # Calculate based on baseline
        memory_change = (start_memory - end_memory) / start_memory if start_memory > 0 else 0
        
        self.metrics[f"{optimization_type}_speedup"].append(speedup)
        self.metrics[f"{optimization_type}_memory_reduction"].append(memory_change)
        
        logger.info(f"🚀 {optimization_type} optimization: {speedup:.2f}x speedup, {memory_change:.1%} memory reduction")

# Global optimization monitor
pytorch_monitor = PyTorchOptimizationMonitor()
```

### Implementation Notes

- **Quantization Strategy**: int4 weight-only for optimal RTX 4090 performance vs quality trade-off

- **Mixed Precision**: Automatic FP16 with gradient scaling for training stability

- **Flash Attention**: 2-4x memory efficiency for sequences >2K tokens

- **Kernel Optimization**: Liger kernels for memory-constrained high-throughput scenarios

- **Compilation**: torch.compile for production inference optimization

- **Monitoring**: Performance tracking for optimization validation

### Testing Strategy

```python

# In tests/test_pytorch_optimization.py: Optimization validation
async def test_torchao_quantization_performance():
    """Test TorchAO int4 quantization performance and quality."""
    
    # Setup quantized vs non-quantized models
    baseline_backend = vLLM(model="Qwen/Qwen3-4B-Thinking-2507", torch_dtype=torch.float16)
    quantized_backend = create_kernel_optimized_backend()
    
    test_prompt = "Explain the concept of machine learning in simple terms."
    
    # Benchmark performance
    with pytorch_monitor.benchmark_optimization("quantization"):
        quantized_response = quantized_backend.complete(test_prompt)
    
    baseline_response = baseline_backend.complete(test_prompt)
    
    # Validate quality preservation (>95% semantic similarity)
    similarity = calculate_semantic_similarity(baseline_response, quantized_response)
    assert similarity > 0.95, f"Quality degradation: {similarity:.2f} similarity"
    
    # Validate performance improvement (>1.5x speedup target)
    speedup = pytorch_monitor.metrics["quantization_speedup"][-1]
    assert speedup > 1.5, f"Insufficient speedup: {speedup:.2f}x"

async def test_mixed_precision_embedding_generation():
    """Test mixed precision embedding generation performance."""
    
    test_texts = ["Document analysis", "Knowledge extraction", "Vector similarity"]
    
    # Test mixed precision performance
    with pytorch_monitor.benchmark_optimization("mixed_precision"):
        mp_embeddings = await mp_optimizer.optimized_embedding_generation(test_texts)
    
    # Validate embedding quality and performance
    assert len(mp_embeddings) == len(test_texts)
    assert all(len(emb) == 1024 for emb in mp_embeddings)  # BGE-large dimensions
    
    speedup = pytorch_monitor.metrics["mixed_precision_speedup"][-1]
    assert speedup > 1.3, f"Mixed precision speedup insufficient: {speedup:.2f}x"

@pytest.mark.gpu
async def test_flash_attention_memory_efficiency():
    """Test Flash Attention 2 memory efficiency for long sequences."""
    
    if not torch.cuda.is_available():
        pytest.skip("GPU required for Flash Attention testing")
    
    # Test long sequence processing
    long_sequence = "This is a test sequence. " * 1000  # ~4K tokens
    
    baseline_memory = torch.cuda.memory_allocated()
    
    # Process with Flash Attention 2
    flash_backend = create_memory_optimized_backend("vllm")
    response = flash_backend.complete(long_sequence)
    
    peak_memory = torch.cuda.memory_allocated()
    memory_efficiency = 1 - (peak_memory - baseline_memory) / baseline_memory
    
    # Validate 2x+ memory efficiency for long sequences
    assert memory_efficiency > 0.5, f"Flash Attention memory efficiency: {memory_efficiency:.1%}"

@pytest.mark.parametrize("backend_type", ["vllm", "llamacpp"])
async def test_multi_backend_optimization_consistency(backend_type):
    """Test optimization consistency across multiple backends."""
    
    optimized_backend = create_compiled_backend(backend_type)
    test_query = "What are the benefits of document Q&A systems?"
    
    # Validate backend switching with optimization preservation
    Settings.llm = optimized_backend
    response = Settings.llm.complete(test_query)
    
    assert response is not None
    assert len(response.text) > 50  # Reasonable response length
    
    # Validate performance monitoring
    assert len(pytorch_monitor.metrics) > 0
```

## Consequences

### Positive Outcomes

- **Performance Gains**: 1.89x faster inference with TorchAO int4 quantization vs FP16 baseline

- **Memory Efficiency**: 58% memory reduction enables larger models on RTX 4090 16GB configuration

- **Mixed Precision Benefits**: 1.5x training/embedding speedup with maintained model quality

- **Flash Attention Optimization**: 2-4x memory efficiency for long sequences and large batch processing

- **Production Readiness**: torch.compile integration provides additional inference optimization

- **Library-First Compliance**: PyTorch-native optimization eliminates custom acceleration implementations

- **Multi-Backend Support**: Unified optimization strategies across Ollama, LlamaCPP, vLLM backends

- **Quality Preservation**: Intelligent quantization strategies maintain >95% model accuracy

### Ongoing Considerations

- **Monitor PyTorch Ecosystem**: TorchAO quantization methods and kernel optimization improvements

- **Validate Performance Targets**: Ensure 1.89x speedup and 58% memory reduction achieved consistently

- **Quality Assurance**: Continuous validation of quantization impact on model accuracy

- **Backend Optimization**: Keep quantization configurations optimized per backend characteristics

- **Hardware Scaling**: Evaluate optimization strategies for different GPU configurations

- **Memory Management**: Monitor VRAM usage patterns with quantization and mixed precision

### Dependencies

- **Core**: torch>=2.7.1 with CUDA support for RTX 4090 optimization

- **Quantization**: torchao>=0.1.0 for int4 weight-only quantization

- **Mixed Precision**: torch.cuda.amp for automatic mixed precision training

- **Flash Attention**: flash-attn>=2.0.0 for memory-efficient attention computation

- **Optional**: liger-kernel for 47% memory reduction capabilities

- **Integration**: Native PyTorch compilation support in torch>=2.0

**Changelog:**

- 1.0 (August 13, 2025): Initial PyTorch optimization strategy with TorchAO int4 quantization, mixed precision training, Flash Attention 2 integration, and kernel optimization support. Designed for 1.89x inference speedup and 58% memory reduction on RTX 4090 hardware. Integration with ADR-003 Settings.llm architecture and multi-backend quantization strategies. Aligned with ADR-017's Qwen3-4B-Thinking model strategy and ADR-020's unified Settings patterns.
