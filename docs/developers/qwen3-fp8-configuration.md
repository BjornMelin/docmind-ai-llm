# Qwen3-4B-Instruct-2507-FP8 Configuration Guide

## Overview

This guide provides comprehensive configuration instructions for the Qwen3-4B-Instruct-2507-FP8 model, which serves as the core LLM for DocMind AI. The model is optimized for 131,072 token context windows with FP8 quantization to achieve optimal performance on RTX 4090 hardware within 12-14GB VRAM constraints.

## Model Specifications

### Core Model Parameters

- **Model Name**: Qwen3-4B-Instruct-2507-FP8
- **Context Window**: 131,072 tokens (128K)
- **Model Size**: 4.23B parameters
- **Quantization**: FP8 precision with FP8 KV cache
- **Architecture**: Transformer-based with optimized attention mechanisms
- **Training Cutoff**: July 2024 with 2507 designation

### Performance Targets

- **Decode Speed**: 100-160 tokens/second
- **Prefill Speed**: 800-1300 tokens/second
- **VRAM Usage**: 12-14GB (target) / 16GB (maximum)
- **Context Utilization**: Up to 120K tokens (with 8K buffer)

## Hardware Requirements

### Minimum Requirements

- **GPU**: RTX 4090 (16GB VRAM) or equivalent
- **CUDA**: 12.8 or higher
- **Driver**: 550.54.14 or higher
- **System RAM**: 32GB recommended
- **Storage**: 50GB available space for model files

### Optimal Configuration

- **GPU**: RTX 4090 Laptop (16GB VRAM)
- **CUDA**: 12.8+
- **PyTorch**: 2.7.1 with CUDA support
- **vLLM**: 0.10.1+ with FlashInfer backend
- **System RAM**: 64GB for optimal performance

## Model Installation & Setup

### Model Acquisition

```bash
# Option 1: Download from Hugging Face (if available)
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507-FP8

# Option 2: Use model loading utilities
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
print('✅ Tokenizer loaded successfully')
"
```

### Environment Configuration

```bash
# Essential environment variables for FP8 optimization
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_CUDNN_PREFILL=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# CUDA and PyTorch settings
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 architecture
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Model-specific settings
export QWEN_MODEL_PATH="/path/to/qwen3-4b-instruct-2507-fp8"
export QWEN_MAX_CONTEXT=131072
export QWEN_TARGET_VRAM_GB=14
```

## vLLM Configuration

### Basic Configuration

```python
from llama_index.llms.vllm import VllmLLM
from typing import Dict, Any, Optional
import os

class Qwen3FP8Config:
    """Optimized configuration for Qwen3-4B-Instruct-2507-FP8"""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
        max_model_len: int = 131072,
        gpu_memory_utilization: float = 0.85,
        enable_fp8_kv_cache: bool = True
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_fp8_kv_cache = enable_fp8_kv_cache
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.max_model_len > 131072:
            raise ValueError(f"Max model length {self.max_model_len} exceeds Qwen3 limit of 131,072")
        
        if self.gpu_memory_utilization > 0.95:
            raise ValueError("GPU memory utilization should not exceed 95%")
        
        # Check CUDA availability
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen3-FP8 model")
        
        # Verify GPU compute capability
        compute_cap = torch.cuda.get_device_capability()
        if compute_cap[0] < 8:
            print(f"⚠️  GPU compute capability {compute_cap} may not fully support FP8 optimization")
    
    def create_vllm_config(self) -> Dict[str, Any]:
        """Create vLLM configuration dictionary"""
        config = {
            "model": self.model_path,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "quantization": "fp8",
            "attention_backend": "FLASHINFER",
            "dtype": "auto",
            "enforce_eager": False,
            "max_num_batched_tokens": self.max_model_len,
            "max_num_seqs": 1,  # Single-user application
            "disable_custom_all_reduce": True,
        }
        
        if self.enable_fp8_kv_cache:
            config.update({
                "kv_cache_dtype": "fp8_e5m2",
                "quantization_param_path": None  # Auto-detect
            })
        
        return config
    
    def create_llm(self) -> VllmLLM:
        """Create optimized VllmLLM instance"""
        config = self.create_vllm_config()
        return VllmLLM(**config)

# Usage example
config = Qwen3FP8Config()
llm = config.create_llm()
```

### Advanced Configuration with Performance Monitoring

```python
import time
import torch
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

@dataclass
class Qwen3PerformanceMetrics:
    """Performance metrics for Qwen3 model"""
    model_name: str
    context_length: int
    input_tokens: int
    output_tokens: int
    decode_tps: float
    prefill_tps: float
    vram_usage_gb: float
    execution_time: float
    timestamp: float

class Qwen3OptimizedManager:
    """Advanced management for Qwen3-FP8 with performance optimization"""
    
    def __init__(self, config: Qwen3FP8Config):
        self.config = config
        self.llm = None
        self.performance_history: List[Qwen3PerformanceMetrics] = []
        self.context_cache = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Qwen3 model with validation"""
        print("🚀 Initializing Qwen3-4B-Instruct-2507-FP8...")
        
        try:
            # Set environment variables
            os.environ.update({
                "VLLM_ATTENTION_BACKEND": "FLASHINFER",
                "VLLM_USE_CUDNN_PREFILL": "1",
                "VLLM_DISABLE_CUSTOM_ALL_REDUCE": "1"
            })
            
            # Create LLM instance
            start_time = time.time()
            self.llm = self.config.create_llm()
            initialization_time = time.time() - start_time
            
            print(f"✅ Qwen3 model initialized in {initialization_time:.2f}s")
            
            # Run validation
            self._validate_model_setup()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qwen3 model: {e}")
    
    def _validate_model_setup(self):
        """Validate model setup and basic functionality"""
        print("🔍 Validating Qwen3 model setup...")
        
        # Test basic inference
        test_prompt = "Hello! Please respond to test model functionality."
        
        try:
            start_time = time.time()
            response = self.llm.complete(test_prompt, max_tokens=50)
            validation_time = time.time() - start_time
            
            if response and response.text:
                print(f"✅ Model validation successful ({validation_time:.2f}s)")
                print(f"📝 Test response: {response.text[:100]}...")
                
                # Check VRAM usage
                vram_usage = self._get_vram_usage()
                print(f"💾 VRAM usage: {vram_usage:.2f} GB")
                
                if vram_usage > 16:
                    print("⚠️  VRAM usage exceeds 16GB target")
                elif vram_usage > 14:
                    print("⚠️  VRAM usage above optimal 14GB target")
                else:
                    print("✅ VRAM usage within target range")
            else:
                raise RuntimeError("Model returned empty response")
                
        except Exception as e:
            raise RuntimeError(f"Model validation failed: {e}")
    
    def _get_vram_usage(self) -> float:
        """Get current VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _get_max_vram_usage(self) -> float:
        """Get peak VRAM usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
        return 0.0
    
    def process_with_metrics(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        temperature: float = 0.1,
        context_id: Optional[str] = None
    ) -> Tuple[str, Qwen3PerformanceMetrics]:
        """Process prompt with comprehensive performance tracking"""
        
        # Estimate input tokens
        input_tokens = self._estimate_tokens(prompt)
        
        # Check context cache
        if context_id and context_id in self.context_cache:
            cached_context = self.context_cache[context_id]
            prompt = f"{cached_context}\n\n{prompt}"
        
        # Execute inference with timing
        start_time = time.time()
        vram_before = self._get_vram_usage()
        
        response = self.llm.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        vram_after = self._get_max_vram_usage()
        
        # Calculate performance metrics
        output_tokens = self._estimate_tokens(response.text)
        total_tokens = input_tokens + output_tokens
        
        # Performance calculations
        decode_tps = output_tokens / execution_time if execution_time > 0 else 0
        prefill_tps = input_tokens / (execution_time * 0.1) if execution_time > 0 else 0  # Rough estimate
        
        # Create metrics record
        metrics = Qwen3PerformanceMetrics(
            model_name="Qwen3-4B-Instruct-2507-FP8",
            context_length=len(prompt),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
            vram_usage_gb=vram_after,
            execution_time=execution_time,
            timestamp=time.time()
        )
        
        # Store metrics
        self.performance_history.append(metrics)
        
        # Update context cache
        if context_id:
            self.context_cache[context_id] = f"{prompt}\n{response.text}"
            # Trim cache if too large
            if len(self.context_cache[context_id]) > 100000:  # Rough character limit
                self.context_cache[context_id] = self.context_cache[context_id][-80000:]
        
        # Log performance
        self._log_performance(metrics)
        
        return response.text, metrics
    
    def _estimate_tokens(self, text: str) -> float:
        """Estimate token count (rough approximation)"""
        # Qwen3 typically has ~1.3 tokens per word
        return len(text.split()) * 1.3
    
    def _log_performance(self, metrics: Qwen3PerformanceMetrics):
        """Log performance metrics"""
        print(f"📊 Qwen3 Performance:")
        print(f"   🎯 Decode: {metrics.decode_tps:.1f} TPS (target: 100-160)")
        print(f"   ⚡ Prefill: {metrics.prefill_tps:.1f} TPS (target: 800-1300)")
        print(f"   💾 VRAM: {metrics.vram_usage_gb:.2f} GB (target: <14)")
        print(f"   ⏱️  Time: {metrics.execution_time:.3f}s")
        
        # Performance warnings
        if metrics.decode_tps < 100:
            print("⚠️  Decode speed below target (100 TPS)")
        if metrics.prefill_tps < 800:
            print("⚠️  Prefill speed below target (800 TPS)")
        if metrics.vram_usage_gb > 14:
            print("⚠️  VRAM usage above optimal target (14GB)")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return {"status": "No performance data available"}
        
        recent_metrics = self.performance_history[-10:]  # Last 10 runs
        
        # Calculate averages
        avg_decode_tps = sum(m.decode_tps for m in recent_metrics) / len(recent_metrics)
        avg_prefill_tps = sum(m.prefill_tps for m in recent_metrics) / len(recent_metrics)
        avg_vram_usage = sum(m.vram_usage_gb for m in recent_metrics) / len(recent_metrics)
        avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        
        # Performance assessment
        performance_grade = self._calculate_performance_grade(
            avg_decode_tps, avg_prefill_tps, avg_vram_usage
        )
        
        return {
            "model_name": "Qwen3-4B-Instruct-2507-FP8",
            "total_inferences": len(self.performance_history),
            "recent_performance": {
                "avg_decode_tps": round(avg_decode_tps, 1),
                "avg_prefill_tps": round(avg_prefill_tps, 1),
                "avg_vram_usage_gb": round(avg_vram_usage, 2),
                "avg_execution_time": round(avg_execution_time, 3)
            },
            "performance_targets": {
                "decode_tps_range": [100, 160],
                "prefill_tps_range": [800, 1300],
                "vram_target_gb": 14,
                "vram_max_gb": 16
            },
            "performance_grade": performance_grade,
            "target_compliance": {
                "decode_speed": avg_decode_tps >= 100,
                "prefill_speed": avg_prefill_tps >= 800,
                "vram_usage": avg_vram_usage <= 14
            }
        }
    
    def _calculate_performance_grade(
        self, 
        decode_tps: float, 
        prefill_tps: float, 
        vram_usage: float
    ) -> str:
        """Calculate overall performance grade"""
        
        score = 0
        
        # Decode speed scoring (40 points max)
        if decode_tps >= 160:
            score += 40
        elif decode_tps >= 130:
            score += 35
        elif decode_tps >= 100:
            score += 30
        elif decode_tps >= 80:
            score += 20
        else:
            score += 10
        
        # Prefill speed scoring (40 points max)  
        if prefill_tps >= 1300:
            score += 40
        elif prefill_tps >= 1000:
            score += 35
        elif prefill_tps >= 800:
            score += 30
        elif prefill_tps >= 600:
            score += 20
        else:
            score += 10
        
        # VRAM usage scoring (20 points max)
        if vram_usage <= 12:
            score += 20
        elif vram_usage <= 14:
            score += 15
        elif vram_usage <= 16:
            score += 10
        else:
            score += 5
        
        # Convert to letter grade
        if score >= 90:
            return "A+ Excellent"
        elif score >= 80:
            return "A Good"
        elif score >= 70:
            return "B Average"
        elif score >= 60:
            return "C Below Average"
        else:
            return "D Poor"
    
    def optimize_context_window(self, text: str, target_tokens: int = 120000) -> str:
        """Optimize text for Qwen3's 128K context window"""
        
        estimated_tokens = self._estimate_tokens(text)
        
        if estimated_tokens <= target_tokens:
            return text
        
        print(f"🔄 Context optimization: {estimated_tokens:.0f} → {target_tokens:.0f} tokens")
        
        # Calculate reduction ratio
        reduction_ratio = target_tokens / estimated_tokens
        
        # Split into sentences and trim proportionally
        sentences = text.split('. ')
        target_sentences = int(len(sentences) * reduction_ratio)
        
        if target_sentences < len(sentences):
            # Keep first and last portions, trim middle
            keep_start = target_sentences // 2
            keep_end = target_sentences - keep_start
            
            optimized_sentences = sentences[:keep_start]
            optimized_sentences.append("[... content trimmed for context window ...]")
            optimized_sentences.extend(sentences[-keep_end:])
            
            return '. '.join(optimized_sentences)
        
        return text
    
    def clear_performance_history(self):
        """Clear performance history"""
        self.performance_history.clear()
        print("🗑️  Performance history cleared")
    
    def export_performance_metrics(self, filename: str = None) -> str:
        """Export performance metrics to JSON"""
        import json
        
        if not filename:
            filename = f"qwen3_performance_{int(time.time())}.json"
        
        export_data = {
            "model_config": asdict(self.config),
            "metrics_count": len(self.performance_history),
            "metrics": [asdict(m) for m in self.performance_history],
            "summary": self.get_performance_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Performance metrics exported to: {filename}")
        return filename
```

## Context Window Optimization

### 128K Context Management

```python
class Qwen3ContextManager:
    """Specialized context management for Qwen3's 128K window"""
    
    def __init__(self):
        self.max_context_tokens = 131072  # Model's actual limit
        self.safe_context_tokens = 120000  # Safe limit with buffer
        self.token_buffer = 8192          # Buffer for response generation
    
    def optimize_for_qwen3(
        self, 
        conversation_history: List[Dict[str, str]],
        retrieval_results: List[Dict[str, Any]],
        current_query: str
    ) -> Dict[str, Any]:
        """Optimize context specifically for Qwen3's capabilities"""
        
        # Estimate current token usage
        current_tokens = self._estimate_total_tokens(
            conversation_history, retrieval_results, current_query
        )
        
        print(f"📊 Context analysis: {current_tokens:,} tokens")
        
        if current_tokens <= self.safe_context_tokens:
            return {
                "conversation_history": conversation_history,
                "retrieval_results": retrieval_results,
                "current_query": current_query,
                "optimization_applied": False
            }
        
        print(f"🔄 Context optimization needed: {current_tokens:,} > {self.safe_context_tokens:,}")
        
        # Apply Qwen3-specific optimization strategies
        optimized_context = self._apply_qwen3_optimization(
            conversation_history, retrieval_results, current_query, current_tokens
        )
        
        # Verify optimization
        final_tokens = self._estimate_total_tokens(
            optimized_context["conversation_history"],
            optimized_context["retrieval_results"],
            optimized_context["current_query"]
        )
        
        savings = current_tokens - final_tokens
        print(f"✅ Context optimized: {savings:,} tokens saved ({savings/current_tokens*100:.1f}%)")
        
        return optimized_context
    
    def _estimate_total_tokens(
        self,
        conversation_history: List[Dict[str, str]],
        retrieval_results: List[Dict[str, Any]],
        current_query: str
    ) -> int:
        """Estimate total token usage"""
        
        # Conversation tokens
        conv_text = ""
        for turn in conversation_history:
            conv_text += f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}\n"
        
        # Retrieval results tokens
        retrieval_text = ""
        for result in retrieval_results:
            retrieval_text += result.get('content', '') + "\n"
        
        # Total text
        total_text = conv_text + retrieval_text + current_query
        
        # Qwen3 token estimation (roughly 1.3 tokens per word for English)
        return int(len(total_text.split()) * 1.3)
    
    def _apply_qwen3_optimization(
        self,
        conversation_history: List[Dict[str, str]],
        retrieval_results: List[Dict[str, Any]],
        current_query: str,
        current_tokens: int
    ) -> Dict[str, Any]:
        """Apply Qwen3-specific context optimization"""
        
        target_tokens = self.safe_context_tokens
        tokens_to_save = current_tokens - target_tokens
        
        # Optimization priority for Qwen3:
        # 1. Trim older conversation (keep recent context)
        # 2. Compress retrieval results (keep most relevant)
        # 3. Summarize middle conversation turns
        # 4. Reduce retrieval result verbosity
        
        optimized_conv = self._optimize_conversation_for_qwen3(
            conversation_history, tokens_to_save * 0.4  # 40% of savings from conversation
        )
        
        optimized_retrieval = self._optimize_retrieval_for_qwen3(
            retrieval_results, tokens_to_save * 0.6  # 60% of savings from retrieval
        )
        
        return {
            "conversation_history": optimized_conv,
            "retrieval_results": optimized_retrieval,
            "current_query": current_query,
            "optimization_applied": True,
            "tokens_saved": tokens_to_save
        }
    
    def _optimize_conversation_for_qwen3(
        self,
        conversation_history: List[Dict[str, str]],
        target_token_savings: float
    ) -> List[Dict[str, str]]:
        """Optimize conversation history for Qwen3's context preferences"""
        
        if not conversation_history:
            return conversation_history
        
        # Qwen3 performs well with recent context, so keep last few turns
        # and summarize or remove older turns
        
        if len(conversation_history) <= 3:
            return conversation_history  # Keep short conversations intact
        
        # Keep last 2 turns always (most recent context)
        recent_turns = conversation_history[-2:]
        older_turns = conversation_history[:-2]
        
        # If we need significant savings, summarize older turns
        if target_token_savings > 1000 and older_turns:
            # Create summary of older conversation
            summary_turn = {
                "user": "[Previous conversation summarized]",
                "assistant": self._summarize_conversation_turns(older_turns)
            }
            return [summary_turn] + recent_turns
        else:
            # Just trim some older turns
            keep_turns = max(1, len(older_turns) - int(target_token_savings / 200))
            return older_turns[-keep_turns:] + recent_turns
    
    def _optimize_retrieval_for_qwen3(
        self,
        retrieval_results: List[Dict[str, Any]],
        target_token_savings: float
    ) -> List[Dict[str, Any]]:
        """Optimize retrieval results for Qwen3's processing preferences"""
        
        if not retrieval_results:
            return retrieval_results
        
        # Qwen3 can handle dense information well, so focus on keeping
        # high-quality, relevant results and compressing less relevant ones
        
        optimized_results = []
        
        for i, result in enumerate(retrieval_results):
            if i < 5:  # Keep top 5 results with full content
                optimized_results.append(result)
            else:
                # Compress remaining results
                compressed_result = result.copy()
                content = result.get('content', '')
                
                if len(content) > 500:
                    # Keep first 300 and last 100 characters for context
                    compressed_content = content[:300] + "\n[...content trimmed...]\n" + content[-100:]
                    compressed_result['content'] = compressed_content
                
                optimized_results.append(compressed_result)
        
        return optimized_results
    
    def _summarize_conversation_turns(self, turns: List[Dict[str, str]]) -> str:
        """Create a summary of conversation turns"""
        
        # Simple summarization - in production, this could use the LLM itself
        topics = []
        for turn in turns:
            user_msg = turn.get('user', '')
            if len(user_msg) > 20:
                # Extract key topics/questions
                if '?' in user_msg:
                    topics.append(f"User asked about: {user_msg[:50]}...")
                else:
                    topics.append(f"User mentioned: {user_msg[:50]}...")
        
        if topics:
            return f"Previous discussion covered: {'; '.join(topics[:3])}"
        else:
            return "Previous conversation context available but summarized for space."
```

## Performance Benchmarking

### Comprehensive Benchmark Suite

```python
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor

class Qwen3BenchmarkSuite:
    """Comprehensive benchmarking for Qwen3-FP8 performance"""
    
    def __init__(self, manager: Qwen3OptimizedManager):
        self.manager = manager
        self.benchmark_results = {}
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        
        print("🧪 Starting Qwen3 Comprehensive Benchmark Suite")
        print("=" * 60)
        
        # Test scenarios
        test_scenarios = [
            ("short_query", "What is machine learning?", 100),
            ("medium_query", "Explain the benefits and challenges of implementing artificial intelligence in healthcare systems.", 300),
            ("long_query", "Provide a detailed analysis of the economic impacts of renewable energy adoption, including market trends, policy implications, and technological developments." * 3, 500),
            ("context_heavy", "Based on the following context..." + "This is important context information. " * 100 + "Please analyze this data.", 800)
        ]
        
        results = {}
        
        for scenario_name, prompt, max_tokens in test_scenarios:
            print(f"\n🎯 Running benchmark: {scenario_name}")
            scenario_results = await self._benchmark_scenario(scenario_name, prompt, max_tokens)
            results[scenario_name] = scenario_results
        
        # Run context window stress test
        print(f"\n🔥 Running context window stress test")
        context_results = await self._benchmark_context_window()
        results["context_window_stress"] = context_results
        
        # Run parallel execution test
        print(f"\n⚡ Running parallel execution test")
        parallel_results = await self._benchmark_parallel_execution()
        results["parallel_execution"] = parallel_results
        
        # Calculate overall assessment
        overall_assessment = self._calculate_overall_assessment(results)
        
        final_results = {
            "benchmark_timestamp": time.time(),
            "model_name": "Qwen3-4B-Instruct-2507-FP8",
            "individual_scenarios": results,
            "overall_assessment": overall_assessment,
            "recommendations": self._generate_recommendations(results)
        }
        
        self.benchmark_results = final_results
        return final_results
    
    async def _benchmark_scenario(
        self, 
        scenario_name: str, 
        prompt: str, 
        max_tokens: int,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark a specific scenario multiple times"""
        
        results = []
        
        for i in range(iterations):
            print(f"  📊 Iteration {i+1}/{iterations}")
            
            try:
                response, metrics = self.manager.process_with_metrics(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                
                results.append({
                    "iteration": i + 1,
                    "success": True,
                    "metrics": asdict(metrics),
                    "response_length": len(response)
                })
                
            except Exception as e:
                results.append({
                    "iteration": i + 1,
                    "success": False,
                    "error": str(e),
                    "metrics": None
                })
                print(f"    ❌ Iteration {i+1} failed: {e}")
            
            # Brief pause between iterations
            await asyncio.sleep(0.5)
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            decode_speeds = [r["metrics"]["decode_tps"] for r in successful_results]
            prefill_speeds = [r["metrics"]["prefill_tps"] for r in successful_results]
            vram_usage = [r["metrics"]["vram_usage_gb"] for r in successful_results]
            execution_times = [r["metrics"]["execution_time"] for r in successful_results]
            
            stats = {
                "success_rate": len(successful_results) / len(results),
                "avg_decode_tps": statistics.mean(decode_speeds),
                "avg_prefill_tps": statistics.mean(prefill_speeds),
                "avg_vram_gb": statistics.mean(vram_usage),
                "avg_execution_time": statistics.mean(execution_times),
                "decode_tps_std": statistics.stdev(decode_speeds) if len(decode_speeds) > 1 else 0,
                "min_decode_tps": min(decode_speeds),
                "max_decode_tps": max(decode_speeds),
                "target_compliance": {
                    "decode_speed": statistics.mean(decode_speeds) >= 100,
                    "prefill_speed": statistics.mean(prefill_speeds) >= 800,
                    "vram_usage": statistics.mean(vram_usage) <= 14
                }
            }
        else:
            stats = {
                "success_rate": 0,
                "error": "All iterations failed"
            }
        
        return {
            "scenario_name": scenario_name,
            "iterations": len(results),
            "individual_results": results,
            "statistics": stats
        }
    
    async def _benchmark_context_window(self) -> Dict[str, Any]:
        """Benchmark context window utilization"""
        
        # Create prompts of varying context lengths
        context_tests = [
            ("32K", "Context test: " + "Important information. " * 6000),  # ~32K tokens
            ("64K", "Context test: " + "Important information. " * 12000), # ~64K tokens  
            ("96K", "Context test: " + "Important information. " + "Additional context. " * 18000), # ~96K tokens
            ("120K", "Context test: " + "Maximum context utilization. " * 22000) # ~120K tokens
        ]
        
        results = []
        
        for test_name, prompt in context_tests:
            print(f"  🧪 Testing {test_name} context")
            
            try:
                # Optimize context first
                optimized_prompt = self.manager.optimize_context_window(prompt)
                
                response, metrics = self.manager.process_with_metrics(
                    prompt=optimized_prompt,
                    max_tokens=512,  # Short response to focus on context handling
                    temperature=0.1
                )
                
                results.append({
                    "context_size": test_name,
                    "success": True,
                    "context_tokens": metrics.input_tokens,
                    "decode_tps": metrics.decode_tps,
                    "vram_usage_gb": metrics.vram_usage_gb,
                    "execution_time": metrics.execution_time
                })
                
                print(f"    ✅ {test_name}: {metrics.input_tokens} tokens, {metrics.decode_tps:.1f} TPS")
                
            except Exception as e:
                results.append({
                    "context_size": test_name,
                    "success": False,
                    "error": str(e)
                })
                print(f"    ❌ {test_name} failed: {e}")
        
        return {
            "test_type": "context_window_stress",
            "results": results,
            "max_successful_context": max([
                r["context_tokens"] for r in results if r["success"]
            ], default=0)
        }
    
    async def _benchmark_parallel_execution(self) -> Dict[str, Any]:
        """Benchmark parallel processing capabilities"""
        
        # Test concurrent requests (simulate multi-agent scenario)
        prompts = [
            "Analyze this document for key insights.",
            "Summarize the main findings.",
            "Extract important dates and numbers.",
            "Identify potential issues or concerns.",
            "Generate follow-up questions."
        ]
        
        # Sequential execution
        start_time = time.time()
        sequential_results = []
        
        for prompt in prompts:
            response, metrics = self.manager.process_with_metrics(prompt, max_tokens=200)
            sequential_results.append(metrics.execution_time)
        
        sequential_time = time.time() - start_time
        
        # Parallel execution (simulated - actual parallel would require multiple model instances)
        # For this benchmark, we measure the overhead of rapid sequential calls
        start_time = time.time()
        rapid_results = []
        
        for prompt in prompts:
            response, metrics = self.manager.process_with_metrics(prompt, max_tokens=200)
            rapid_results.append(metrics.execution_time)
        
        rapid_time = time.time() - start_time
        
        return {
            "test_type": "parallel_execution",
            "sequential_time": sequential_time,
            "rapid_sequential_time": rapid_time,
            "efficiency_ratio": sequential_time / rapid_time,
            "avg_individual_time": statistics.mean(sequential_results),
            "overhead_per_call": (rapid_time - sum(rapid_results)) / len(prompts)
        }
    
    def _calculate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance assessment"""
        
        # Collect all successful metrics
        all_decode_speeds = []
        all_prefill_speeds = []
        all_vram_usage = []
        
        for scenario_name, scenario_data in results.items():
            if scenario_name in ["context_window_stress", "parallel_execution"]:
                continue
                
            if scenario_data.get("statistics", {}).get("success_rate", 0) > 0:
                stats = scenario_data["statistics"]
                all_decode_speeds.append(stats["avg_decode_tps"])
                all_prefill_speeds.append(stats["avg_prefill_tps"])
                all_vram_usage.append(stats["avg_vram_gb"])
        
        if not all_decode_speeds:
            return {"status": "No successful benchmarks to assess"}
        
        # Overall averages
        overall_decode = statistics.mean(all_decode_speeds)
        overall_prefill = statistics.mean(all_prefill_speeds)
        overall_vram = statistics.mean(all_vram_usage)
        
        # Performance grade calculation
        grade_points = 0
        
        # Decode performance (40 points)
        if overall_decode >= 160:
            grade_points += 40
        elif overall_decode >= 130:
            grade_points += 35
        elif overall_decode >= 100:
            grade_points += 30
        else:
            grade_points += 20
        
        # Prefill performance (40 points)
        if overall_prefill >= 1300:
            grade_points += 40
        elif overall_prefill >= 1000:
            grade_points += 35
        elif overall_prefill >= 800:
            grade_points += 30
        else:
            grade_points += 20
        
        # Memory efficiency (20 points)
        if overall_vram <= 12:
            grade_points += 20
        elif overall_vram <= 14:
            grade_points += 15
        elif overall_vram <= 16:
            grade_points += 10
        else:
            grade_points += 5
        
        # Grade assignment
        if grade_points >= 95:
            grade = "A+ Exceptional"
        elif grade_points >= 85:
            grade = "A Excellent"
        elif grade_points >= 75:
            grade = "B Good"
        elif grade_points >= 65:
            grade = "C Average"
        else:
            grade = "D Needs Improvement"
        
        return {
            "overall_decode_tps": round(overall_decode, 1),
            "overall_prefill_tps": round(overall_prefill, 1),
            "overall_vram_gb": round(overall_vram, 2),
            "performance_grade": grade,
            "grade_points": grade_points,
            "target_compliance": {
                "decode_target_met": overall_decode >= 100,
                "prefill_target_met": overall_prefill >= 800,
                "vram_target_met": overall_vram <= 14
            }
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on benchmark results"""
        
        recommendations = []
        
        # Analyze results for recommendations
        overall = results.get("overall_assessment", {})
        
        if not overall.get("target_compliance", {}).get("decode_target_met", True):
            recommendations.append("Consider optimizing FP8 quantization settings to improve decode speed")
        
        if not overall.get("target_compliance", {}).get("prefill_target_met", True):
            recommendations.append("Enable cuDNN prefill optimization (VLLM_USE_CUDNN_PREFILL=1)")
        
        if not overall.get("target_compliance", {}).get("vram_target_met", True):
            recommendations.append("Reduce gpu_memory_utilization or enable additional FP8 optimizations")
        
        # Context-specific recommendations
        if "context_window_stress" in results:
            max_context = results["context_window_stress"].get("max_successful_context", 0)
            if max_context < 100000:
                recommendations.append("Context window performance may benefit from FlashInfer optimization")
        
        if not recommendations:
            recommendations.append("Performance meets all targets - system is optimally configured")
        
        return recommendations

# Usage example
async def run_qwen3_benchmark():
    """Run complete Qwen3 benchmark suite"""
    
    # Initialize configuration and manager
    config = Qwen3FP8Config()
    manager = Qwen3OptimizedManager(config)
    
    # Run benchmark suite
    benchmark_suite = Qwen3BenchmarkSuite(manager)
    results = await benchmark_suite.run_full_benchmark()
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 QWEN3 BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    overall = results["overall_assessment"]
    print(f"Overall Performance Grade: {overall['performance_grade']}")
    print(f"Average Decode Speed: {overall['overall_decode_tps']} TPS (target: 100-160)")
    print(f"Average Prefill Speed: {overall['overall_prefill_tps']} TPS (target: 800-1300)")
    print(f"Average VRAM Usage: {overall['overall_vram_gb']} GB (target: <14)")
    
    print("\n📋 RECOMMENDATIONS:")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Export results
    manager.export_performance_metrics("qwen3_benchmark_results.json")
    
    return results
```

## Troubleshooting

### Common Issues and Solutions

#### Model Loading Issues

**Issue**: Model fails to load with CUDA errors

```bash
# Solution: Verify CUDA installation and compatibility
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128
```

**Issue**: FP8 quantization not working correctly

```python
# Diagnostic script
def diagnose_fp8_support():
    import torch
    
    print("🔍 FP8 Diagnostic Report:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        compute_cap = torch.cuda.get_device_capability()
        print(f"GPU Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        print(f"FP8 Support: {'✅ Yes' if compute_cap[0] >= 8 else '❌ No'}")
        
    print(f"FlashInfer Backend: {os.environ.get('VLLM_ATTENTION_BACKEND', 'Not Set')}")
    print(f"cuDNN Prefill: {os.environ.get('VLLM_USE_CUDNN_PREFILL', 'Not Set')}")

diagnose_fp8_support()
```

#### Performance Issues

**Issue**: Lower than expected token/second performance

```python
# Performance optimization checklist
def optimize_qwen3_performance():
    """Optimize Qwen3 performance systematically"""
    
    optimizations = [
        ("Enable FlashInfer", "export VLLM_ATTENTION_BACKEND=FLASHINFER"),
        ("Enable cuDNN Prefill", "export VLLM_USE_CUDNN_PREFILL=1"),  
        ("Disable Custom All-Reduce", "export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1"),
        ("Set GPU Memory Utilization", "gpu_memory_utilization=0.85 in config"),
        ("Enable FP8 KV Cache", "kv_cache_dtype='fp8_e5m2' in config"),
        ("Use CUDA Graphs", "enforce_eager=False in config")
    ]
    
    print("🔧 Qwen3 Performance Optimization Checklist:")
    for desc, command in optimizations:
        print(f"• {desc}: {command}")
```

#### Memory Issues

**Issue**: VRAM usage exceeding 16GB

```python
# Memory optimization strategies
def optimize_memory_usage():
    """Optimize memory usage for RTX 4090"""
    
    strategies = [
        "Reduce gpu_memory_utilization to 0.75",
        "Enable FP8 KV cache: kv_cache_dtype='fp8_e5m2'", 
        "Reduce max_model_len if not using full 128K context",
        "Set max_num_seqs=1 for single-user applications",
        "Enable quantization='fp8' for model weights"
    ]
    
    print("💾 Memory Optimization Strategies:")
    for strategy in strategies:
        print(f"• {strategy}")
```

#### Context Window Issues

**Issue**: Context overflow or truncation

```python
# Context management solution
def manage_context_overflow():
    """Handle context window overflow gracefully"""
    
    context_manager = Qwen3ContextManager()
    
    # Example usage
    large_context = "Very large context..." * 10000
    optimized_context = context_manager.optimize_context_window(large_context)
    
    print(f"Original tokens: {context_manager._estimate_tokens(large_context)}")
    print(f"Optimized tokens: {context_manager._estimate_tokens(optimized_context)}")
```

For additional troubleshooting support, see [troubleshooting.md](../user/troubleshooting.md) and [vllm-integration-guide.md](vllm-integration-guide.md).
