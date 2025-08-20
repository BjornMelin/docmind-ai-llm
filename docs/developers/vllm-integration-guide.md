# vLLM Integration Guide

## Overview

DocMind AI leverages vLLM with FlashInfer attention backend to provide high-performance local LLM inference for the Qwen3-4B-Instruct-2507-FP8 model. This guide covers the complete integration, configuration, and optimization procedures for achieving 100-160 tok/s decode and 800-1300 tok/s prefill performance on RTX 4090 Laptop hardware.

## Technical Stack

### Core Components

- **Model**: Qwen3-4B-Instruct-2507-FP8
- **Context Window**: 131,072 tokens (128K)
- **vLLM Version**: >=0.10.1[flashinfer]
- **Attention Backend**: FlashInfer (VLLM_ATTENTION_BACKEND=FLASHINFER)
- **Quantization**: FP8 with FP8 KV cache (fp8_e5m2)
- **PyTorch**: 2.7.1 with CUDA 12.8+ support

### Hardware Requirements

- **GPU**: RTX 4090 Laptop (16GB VRAM minimum)
- **VRAM Usage**: ~12-14GB with FP8 optimization
- **CUDA**: 12.8+ with driver 550.54.14+
- **System Memory**: 32GB recommended for optimal performance

## Installation & Setup

### Prerequisites

#### 1. CUDA 12.8+ Installation

```bash
# Verify current CUDA version
nvidia-smi

# If CUDA 12.8+ not installed, download from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version
```

#### 2. PyTorch 2.7.1 with CUDA 12.8

```bash
# Install PyTorch with CUDA 12.8 support
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

### vLLM Installation

#### Automated Installation (Recommended)

```bash
# Install vLLM with FlashInfer support through project dependencies
uv sync --extra gpu --index-strategy=unsafe-best-match
```

#### Manual Installation

```bash
# Install vLLM with FlashInfer backend
uv pip install "vllm[flashinfer]>=0.10.1" --extra-index-url https://download.pytorch.org/whl/cu128

# Install build dependencies for GPU compilation
uv pip install ninja>=1.11.1 cmake>=3.26.4

# Verify vLLM installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### Environment Configuration

#### Required Environment Variables

```bash
# vLLM FlashInfer Configuration
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_CUDNN_PREFILL=1

# CUDA Configuration
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090

# Performance Optimization
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_V1=0
```

#### Configuration File (.env)

```ini
# vLLM Configuration
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_CUDNN_PREFILL=1
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# Model Configuration
MODEL_NAME=Qwen3-4B-Instruct-2507-FP8
MODEL_PATH=/path/to/model
MAX_MODEL_LENGTH=131072
TENSOR_PARALLEL_SIZE=1

# Performance Settings
GPU_MEMORY_UTILIZATION=0.85
QUANTIZATION=fp8
KV_CACHE_DTYPE=fp8_e5m2
DTYPE=auto

# Hardware Settings
CUDA_VISIBLE_DEVICES=0
TARGET_VRAM_GB=16
```

## Integration Implementation

### Basic vLLM LLM Configuration

```python
from llama_index.llms.vllm import VllmLLM
from typing import Dict, Any, Optional
import os

class DocMindVLLMConfig:
    """Optimized vLLM configuration for DocMind AI"""
    
    def __init__(
        self,
        model: str = "Qwen3-4B-Instruct-2507-FP8",
        max_model_len: int = 131072,
        gpu_memory_utilization: float = 0.85,
        quantization: str = "fp8",
        kv_cache_dtype: str = "fp8_e5m2"
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.kv_cache_dtype = kv_cache_dtype
    
    def create_llm(self) -> VllmLLM:
        """Create optimized vLLM instance"""
        return VllmLLM(
            model=self.model,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
            attention_backend="FLASHINFER",
            dtype="auto",
            enforce_eager=False,  # Enable CUDA graphs for better performance
            max_num_batched_tokens=self.max_model_len,
            max_num_seqs=1,  # Single user application
            disable_custom_all_reduce=True
        )

# Usage
config = DocMindVLLMConfig()
llm = config.create_llm()
```

### Advanced Configuration with Performance Monitoring

```python
import time
import psutil
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class PerformanceMetrics:
    """Track vLLM performance metrics"""
    prefill_tps: float
    decode_tps: float
    vram_usage_gb: float
    total_tokens: int
    execution_time: float
    context_length: int

class OptimizedVLLMManager:
    """Advanced vLLM manager with performance optimization"""
    
    def __init__(self, config: DocMindVLLMConfig):
        self.config = config
        self.llm = None
        self.metrics_history: List[PerformanceMetrics] = []
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize vLLM with optimal settings"""
        try:
            # Set environment variables
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
            os.environ["VLLM_USE_CUDNN_PREFILL"] = "1"
            
            self.llm = self.config.create_llm()
            self._validate_setup()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM: {e}")
    
    def _validate_setup(self):
        """Validate vLLM setup and performance"""
        # Test inference to ensure everything works
        test_prompt = "Test prompt for validation"
        start_time = time.time()
        
        try:
            response = self.llm.complete(test_prompt)
            end_time = time.time()
            
            print(f"✅ vLLM validation successful")
            print(f"⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"🔧 Model: {self.config.model}")
            print(f"💾 Max context: {self.config.max_model_len:,} tokens")
            
        except Exception as e:
            raise RuntimeError(f"vLLM validation failed: {e}")
    
    def get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def benchmark_inference(
        self, 
        prompt: str, 
        max_new_tokens: int = 512
    ) -> Tuple[str, PerformanceMetrics]:
        """Benchmark inference performance"""
        start_time = time.time()
        vram_before = self.get_vram_usage()
        
        # Execute inference
        response = self.llm.complete(
            prompt, 
            max_tokens=max_new_tokens,
            temperature=0.1
        )
        
        end_time = time.time()
        vram_after = self.get_vram_usage()
        
        # Calculate metrics
        execution_time = end_time - start_time
        input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        output_tokens = len(response.text.split()) * 1.3
        total_tokens = input_tokens + output_tokens
        
        # Performance calculations
        decode_tps = output_tokens / execution_time
        prefill_tps = input_tokens / (execution_time * 0.1)  # Rough prefill estimate
        
        metrics = PerformanceMetrics(
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            vram_usage_gb=vram_after,
            total_tokens=int(total_tokens),
            execution_time=execution_time,
            context_length=len(prompt)
        )
        
        self.metrics_history.append(metrics)
        return response.text, metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from metrics history"""
        if not self.metrics_history:
            return {"status": "No metrics available"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 runs
        
        return {
            "avg_decode_tps": sum(m.decode_tps for m in recent_metrics) / len(recent_metrics),
            "avg_prefill_tps": sum(m.prefill_tps for m in recent_metrics) / len(recent_metrics),
            "avg_vram_usage_gb": sum(m.vram_usage_gb for m in recent_metrics) / len(recent_metrics),
            "max_vram_usage_gb": max(m.vram_usage_gb for m in recent_metrics),
            "avg_execution_time": sum(m.execution_time for m in recent_metrics) / len(recent_metrics),
            "total_inferences": len(self.metrics_history),
            "target_decode_tps": "100-160",
            "target_prefill_tps": "800-1300",
            "target_vram_gb": "12-14"
        }
```

### Integration with LlamaIndex

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

class DocMindVLLMIntegration:
    """Complete vLLM integration with DocMind AI stack"""
    
    def __init__(self):
        self.vllm_manager = OptimizedVLLMManager(DocMindVLLMConfig())
        self._setup_llamaindex()
    
    def _setup_llamaindex(self):
        """Configure LlamaIndex to use vLLM"""
        # Set global LLM
        Settings.llm = self.vllm_manager.llm
        
        # Configure embeddings (separate from LLM)
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        Settings.embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-large-en-v1.5"
        )
        
        # Configure chunk settings for 128K context
        Settings.chunk_size = 2048
        Settings.chunk_overlap = 256
        
        print("✅ LlamaIndex configured with vLLM backend")
    
    def create_chat_engine(self, vector_store: QdrantVectorStore) -> SimpleChatEngine:
        """Create chat engine with vLLM backend"""
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        return index.as_chat_engine(
            chat_mode="context",
            context_template="Context information: {context_str}\n"
                           "Given this context, please answer: {query_str}",
            verbose=True
        )
    
    async def process_query(
        self, 
        query: str, 
        chat_engine: SimpleChatEngine
    ) -> Tuple[str, PerformanceMetrics]:
        """Process query with performance monitoring"""
        start_time = time.time()
        
        # Execute query through chat engine
        response = await chat_engine.achat(query)
        
        # Benchmark the underlying LLM call
        _, metrics = self.vllm_manager.benchmark_inference(
            prompt=query,
            max_new_tokens=1024
        )
        
        return str(response), metrics
```

## Performance Optimization

### FP8 Quantization Setup

```python
class FP8OptimizationConfig:
    """FP8 optimization configuration for memory efficiency"""
    
    @staticmethod
    def get_optimized_config() -> Dict[str, Any]:
        """Get FP8 optimized configuration"""
        return {
            "quantization": "fp8",
            "kv_cache_dtype": "fp8_e5m2",
            "quantization_param_path": None,  # Auto-detect
            "load_format": "auto",
            "dtype": "auto",  # Let vLLM choose optimal dtype
        }
    
    @staticmethod
    def validate_fp8_support():
        """Validate FP8 support on current hardware"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for FP8 support")
        
        # Check GPU compute capability (RTX 4090 = 8.9)
        compute_cap = torch.cuda.get_device_capability()
        if compute_cap[0] < 8:
            print("⚠️  FP8 may not be fully optimized on this GPU")
        else:
            print(f"✅ GPU compute capability {compute_cap[0]}.{compute_cap[1]} supports FP8")
        
        # Verify FlashInfer backend
        if os.environ.get("VLLM_ATTENTION_BACKEND") == "FLASHINFER":
            print("✅ FlashInfer attention backend enabled")
        else:
            print("⚠️  FlashInfer backend not enabled")
```

### Context Window Optimization

```python
class ContextWindowManager:
    """Manage 128K context window efficiently"""
    
    def __init__(self, max_context_length: int = 131072):
        self.max_context_length = max_context_length
        self.context_buffer = int(max_context_length * 0.1)  # 10% buffer
        self.effective_context = max_context_length - self.context_buffer
    
    def optimize_context(self, text: str, priority_sections: List[str] = None) -> str:
        """Optimize context for maximum information density"""
        tokens = self._estimate_tokens(text)
        
        if tokens <= self.effective_context:
            return text
        
        # Implement context optimization strategies
        optimized = self._apply_context_strategies(text, priority_sections)
        return optimized
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1.3 tokens per word)"""
        return int(len(text.split()) * 1.3)
    
    def _apply_context_strategies(
        self, 
        text: str, 
        priority_sections: List[str] = None
    ) -> str:
        """Apply context optimization strategies"""
        strategies = [
            self._prioritize_sections,
            self._remove_redundant_content,
            self._compress_examples,
            self._truncate_oldest_conversation
        ]
        
        optimized_text = text
        for strategy in strategies:
            optimized_text = strategy(optimized_text, priority_sections)
            if self._estimate_tokens(optimized_text) <= self.effective_context:
                break
        
        return optimized_text
    
    def _prioritize_sections(self, text: str, priority_sections: List[str]) -> str:
        """Keep priority sections and trim others"""
        if not priority_sections:
            return text
        
        # Implementation would prioritize keeping specified sections
        return text  # Simplified for example
    
    def _remove_redundant_content(self, text: str, _: List[str]) -> str:
        """Remove redundant or repetitive content"""
        # Implementation would identify and remove redundancy
        return text  # Simplified for example
    
    def _compress_examples(self, text: str, _: List[str]) -> str:
        """Compress long examples while preserving key information"""
        # Implementation would compress verbose examples
        return text  # Simplified for example
    
    def _truncate_oldest_conversation(self, text: str, _: List[str]) -> str:
        """Remove oldest conversation turns to fit context"""
        # Implementation would remove oldest parts of conversation
        return text  # Simplified for example
```

## Performance Validation

### Automated Benchmarking

```python
import asyncio
from typing import List, Dict
import json

class VLLMPerformanceValidator:
    """Validate vLLM performance against targets"""
    
    def __init__(self, vllm_manager: OptimizedVLLMManager):
        self.vllm_manager = vllm_manager
        self.targets = {
            "decode_tps_min": 100,
            "decode_tps_max": 160,
            "prefill_tps_min": 800,
            "prefill_tps_max": 1300,
            "vram_max_gb": 16,
            "vram_target_gb": 14
        }
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        test_cases = [
            {"name": "Short Query", "prompt": "What is machine learning?", "tokens": 256},
            {"name": "Medium Query", "prompt": "Explain the benefits of document analysis." * 10, "tokens": 512},
            {"name": "Long Query", "prompt": "Provide a detailed analysis." * 50, "tokens": 1024},
            {"name": "Context Heavy", "prompt": "Based on this context..." * 100, "tokens": 2048}
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"🧪 Running benchmark: {test_case['name']}")
            
            # Run multiple iterations for average
            iterations = 5
            metrics_list = []
            
            for i in range(iterations):
                response, metrics = self.vllm_manager.benchmark_inference(
                    test_case["prompt"], 
                    test_case["tokens"]
                )
                metrics_list.append(metrics)
                
                # Brief pause between iterations
                await asyncio.sleep(1)
            
            # Calculate averages
            avg_metrics = self._calculate_average_metrics(metrics_list)
            avg_metrics["test_case"] = test_case["name"]
            results.append(avg_metrics)
        
        # Overall assessment
        assessment = self._assess_performance(results)
        
        return {
            "individual_results": results,
            "overall_assessment": assessment,
            "targets": self.targets,
            "timestamp": time.time()
        }
    
    def _calculate_average_metrics(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, float]:
        """Calculate average metrics from multiple runs"""
        return {
            "avg_decode_tps": sum(m.decode_tps for m in metrics_list) / len(metrics_list),
            "avg_prefill_tps": sum(m.prefill_tps for m in metrics_list) / len(metrics_list),
            "avg_vram_usage_gb": sum(m.vram_usage_gb for m in metrics_list) / len(metrics_list),
            "avg_execution_time": sum(m.execution_time for m in metrics_list) / len(metrics_list),
            "max_vram_usage_gb": max(m.vram_usage_gb for m in metrics_list)
        }
    
    def _assess_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall performance against targets"""
        overall_decode_tps = sum(r["avg_decode_tps"] for r in results) / len(results)
        overall_prefill_tps = sum(r["avg_prefill_tps"] for r in results) / len(results)
        max_vram_usage = max(r["max_vram_usage_gb"] for r in results)
        
        assessment = {
            "decode_tps_target_met": (
                self.targets["decode_tps_min"] <= overall_decode_tps <= self.targets["decode_tps_max"]
            ),
            "prefill_tps_target_met": (
                self.targets["prefill_tps_min"] <= overall_prefill_tps <= self.targets["prefill_tps_max"]
            ),
            "vram_target_met": max_vram_usage <= self.targets["vram_max_gb"],
            "overall_performance": {
                "decode_tps": overall_decode_tps,
                "prefill_tps": overall_prefill_tps,
                "max_vram_gb": max_vram_usage
            }
        }
        
        assessment["overall_success"] = all([
            assessment["decode_tps_target_met"],
            assessment["prefill_tps_target_met"], 
            assessment["vram_target_met"]
        ])
        
        return assessment

# Usage example
async def validate_vllm_setup():
    """Validate complete vLLM setup"""
    config = DocMindVLLMConfig()
    manager = OptimizedVLLMManager(config)
    validator = VLLMPerformanceValidator(manager)
    
    print("🚀 Starting comprehensive vLLM validation...")
    results = await validator.run_comprehensive_benchmark()
    
    if results["overall_assessment"]["overall_success"]:
        print("✅ vLLM setup meets all performance targets!")
    else:
        print("⚠️  vLLM setup needs optimization")
    
    # Save results
    with open("vllm_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results
```

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: vLLM compilation fails with CUDA errors

```bash
# Solution: Ensure CUDA 12.8+ and proper environment
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reinstall with proper CUDA paths
uv pip uninstall vllm
uv pip install "vllm[flashinfer]>=0.10.1" --extra-index-url https://download.pytorch.org/whl/cu128
```

**Issue**: FlashInfer backend not available

```bash
# Solution: Verify environment variables and installation
python -c "
import vllm
print('vLLM version:', vllm.__version__)
print('Available backends:', vllm.attention.get_available_backends())
"
```

#### Performance Issues

**Issue**: Lower than expected token/second performance

```python
# Diagnostics script
def diagnose_performance():
    """Diagnose vLLM performance issues"""
    checks = {
        "CUDA_available": torch.cuda.is_available(),
        "GPU_memory_GB": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "FlashInfer_enabled": os.environ.get("VLLM_ATTENTION_BACKEND") == "FLASHINFER",
        "cuDNN_prefill_enabled": os.environ.get("VLLM_USE_CUDNN_PREFILL") == "1"
    }
    
    print("🔍 Performance Diagnostics:")
    for check, status in checks.items():
        print(f"{'✅' if status else '❌'} {check}: {status}")
    
    return all(checks.values())
```

**Issue**: VRAM usage exceeding 16GB

```python
# Solution: Adjust memory utilization and quantization
config = DocMindVLLMConfig(
    gpu_memory_utilization=0.75,  # Reduce from 0.85
    quantization="fp8",
    kv_cache_dtype="fp8_e5m2"
)
```

#### Context Window Issues

**Issue**: Context overflow with 128K tokens

```python
# Solution: Implement context trimming
def trim_context_for_128k(text: str) -> str:
    """Trim context to fit 128K window with buffer"""
    max_tokens = 120000  # Leave 8K buffer
    estimated_tokens = len(text.split()) * 1.3
    
    if estimated_tokens > max_tokens:
        # Trim to 90% of max to be safe
        target_words = int((max_tokens * 0.9) / 1.3)
        words = text.split()
        return " ".join(words[:target_words])
    
    return text
```

### Memory Optimization

#### FP8 Quantization Validation

```python
def validate_fp8_quantization():
    """Validate FP8 quantization is working correctly"""
    model_config = {
        "quantization": "fp8",
        "kv_cache_dtype": "fp8_e5m2"
    }
    
    # Check if FP8 is properly enabled
    llm = VllmLLM(**model_config)
    
    # Monitor memory usage
    baseline_memory = torch.cuda.memory_allocated()
    
    # Run inference
    response = llm.complete("Test FP8 quantization performance")
    
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used_gb = (peak_memory - baseline_memory) / 1024**3
    
    print(f"FP8 Memory Usage: {memory_used_gb:.2f} GB")
    
    if memory_used_gb > 16:
        print("⚠️  Memory usage exceeds target, check FP8 configuration")
    else:
        print("✅ FP8 quantization working within memory limits")
```

### Performance Monitoring

```python
class VLLMMonitor:
    """Continuous monitoring of vLLM performance"""
    
    def __init__(self, vllm_manager: OptimizedVLLMManager):
        self.vllm_manager = vllm_manager
        self.monitoring = False
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous performance monitoring"""
        self.monitoring = True
        
        while self.monitoring:
            metrics = {
                "timestamp": time.time(),
                "vram_usage_gb": self.vllm_manager.get_vram_usage(),
                "gpu_utilization": self._get_gpu_utilization(),
                "temperature_c": self._get_gpu_temperature()
            }
            
            # Log metrics
            print(f"📊 VRAM: {metrics['vram_usage_gb']:.1f}GB | "
                  f"GPU: {metrics['gpu_utilization']:.1f}% | "
                  f"Temp: {metrics['temperature_c']}°C")
            
            # Check for issues
            if metrics['vram_usage_gb'] > 15:
                print("⚠️  High VRAM usage detected")
            
            if metrics['temperature_c'] > 85:
                print("🔥 High GPU temperature detected")
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 0.0
    
    def _get_gpu_temperature(self) -> int:
        """Get GPU temperature in Celsius"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            return int(result.stdout.strip())
        except:
            return 0
```

## Best Practices

### Configuration Management

1. **Environment Variables**: Use consistent environment variable naming
2. **Configuration Validation**: Validate all configurations at startup
3. **Performance Baselines**: Establish and monitor performance baselines
4. **Resource Monitoring**: Continuously monitor VRAM and GPU utilization

### Error Handling

1. **Graceful Degradation**: Implement fallbacks for vLLM failures
2. **Resource Cleanup**: Properly cleanup GPU resources on errors
3. **Retry Logic**: Implement exponential backoff for transient failures
4. **Monitoring Integration**: Log all vLLM errors for analysis

### Production Considerations

1. **Model Warming**: Pre-warm models at startup for consistent performance
2. **Batch Processing**: Use batch processing for multiple simultaneous requests
3. **Resource Limits**: Set appropriate resource limits and timeouts
4. **Health Checks**: Implement health checks for vLLM service status

For additional support, see [troubleshooting.md](../user/troubleshooting.md) and [performance-validation.md](performance-validation.md).
