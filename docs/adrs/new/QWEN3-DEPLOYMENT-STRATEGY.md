# Qwen3-14B Deployment Strategy (CORRECTED)

## Overview

This document provides comprehensive deployment guidance for Qwen3-14B in DocMind AI architecture. After critical review, the 30B MoE model was found impractical for consumer hardware, requiring 24GB+ VRAM and delivering <1 token/sec performance. Qwen3-14B provides reliable performance on actual consumer hardware.

**Target Model**: Qwen3-14B
**Architecture**: Dense (14.8B parameters)
**Context**: 32,768 tokens native (64K with sliding window)
**Memory**: 8GB VRAM with Q4_K_M quantization

### ⚠️ Previous Recommendation FAILED

~~Qwen3-30B-A3B-Instruct-2507~~ was theoretically promising but:

- **Memory**: Required 24GB+ VRAM, not 6-8GB as claimed
- **Performance**: <1 token/sec with large contexts on consumer GPUs
- **Complexity**: MoE expert offloading too complex for consumer deployment
- **Availability**: No GGUF quantization support

## Hardware Requirements

### Minimum Requirements (REALISTIC)

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | RTX 3060 (8GB VRAM) | Q4_K_M model storage |
| **System RAM** | 8GB DDR4 | Model loading |
| **Storage** | 20GB available | GGUF model + cache |
| **CPU** | 4 cores, 2.5GHz+ | Context processing |

### Recommended Requirements (RTX 4090 LAPTOP)

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | RTX 4090 Laptop (16GB VRAM) | YaRN 128K context |
| **System RAM** | 64GB DDR5 | Large batch processing |
| **Storage** | 2TB NVMe SSD | Fast model loading |
| **CPU** | Intel i9-14900HX (24 cores) | Multi-agent processing |

### High-End Setup

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | RTX 4070/4080 (12-16GB VRAM) | Sliding window 64K context |
| **System RAM** | 32GB DDR5 | Large document processing |
| **Storage** | 100GB NVMe SSD | Multiple model variants |
| **CPU** | 12+ cores, 3.5GHz+ | Advanced optimizations |

## Deployment Options

### Option 1: llama.cpp Deployment with YaRN (RECOMMENDED)

**Best for**: Production environments, YaRN context scaling, GGUF quantization

```bash
# Download Q5_K_M model for RTX 4090
huggingface-cli download bartowski/Qwen3-14B-GGUF --include "*Q5_K_M.gguf" --local-dir ./models

# Production deployment with YaRN (128K context)
./llama-server \
  -m ./models/Qwen3-14B-Q5_K_M.gguf \
  -c 131072 \
  --rope-scaling yarn \
  --rope-scale 4.0 \
  --yarn-orig-ctx 32768 \
  --yarn-ext-factor 1.0 \
  --yarn-attn-factor 1.0 \
  --yarn-beta-fast 32 \
  --yarn-beta-slow 1 \
  -ngl 99 \
  -b 1024 \
  -t 24 \
  --host 0.0.0.0 \
  --port 8000
```

**Expected Performance**: 40-60 tokens/sec on RTX 4090 Laptop

### Option 2: vLLM with YaRN Support

**Best for**: AWQ models, multi-GPU setups, YaRN configuration

```bash
# Install vLLM
pip install vllm>=0.8.5

# vLLM deployment with YaRN
vllm serve Qwen/Qwen3-14B-AWQ \
  --max-model-len 131072 \
  --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype int8 \
  --trust-remote-code \
  --port 8001
```

**Expected Performance**: 50-70 tokens/sec on RTX 4090 Laptop

### Option 3: Ollama Development Setup

**Best for**: Development, testing, limited MoE support

```bash
# Simple setup (when available)
ollama pull qwen3:30b-a3b-instruct
ollama run qwen3:30b-a3b-instruct

# Note: Limited MoE optimization in Ollama
```

**Expected Performance**: 80-100 tokens/sec on RTX 4060

## Step-by-Step Deployment Guide

### Step 1: Environment Preparation

```bash
# Create virtual environment
python -m venv qwen3_env
source qwen3_env/bin/activate  # Linux/Mac
# qwen3_env\Scripts\activate  # Windows

# Install base dependencies
pip install torch>=2.0.0 transformers>=4.51.0
pip install vllm>=0.8.5  # or sglang>=0.4.6.post1

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

### Step 2: Model Download

```bash
# Using Hugging Face Hub
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local-dir ./models/qwen3-30b-a3b-instruct-2507-fp8

# Verify download
ls -la ./models/qwen3-30b-a3b-instruct-2507-fp8/
# Should see: config.json, model safetensors files, tokenizer files
```

### Step 3: Configuration Setup

```python
# config.py
QWEN3_CONFIG = {
    "model_path": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "max_model_len": 262144,
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.85,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 32768,
    "enforce_eager": True,
    "max_num_seqs": 8,
    "kv_cache_dtype": "int8",
    "trust_remote_code": True,
    "dtype": "float8_e4m3fn"
}

# Environment variables
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Multi-GPU if available
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
```

### Step 4: Health Check Script

```python
# health_check.py
import requests
import json
import time

def test_qwen3_deployment():
    """Test Qwen3 deployment health."""
    
    # Basic connectivity test
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        print("✓ Service is running")
    except Exception as e:
        print(f"✗ Service connection failed: {e}")
        return False
    
    # Function calling test
    try:
        payload = {
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "messages": [
                {"role": "user", "content": "What is 2+2? Use a calculator tool if needed."}
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        start_time = time.time()
        response = requests.post("http://localhost:8000/v1/chat/completions", 
                               json=payload, timeout=30)
        latency = time.time() - start_time
        
        assert response.status_code == 200, f"Chat completion failed: {response.status_code}"
        result = response.json()
        
        print(f"✓ Function calling test passed")
        print(f"✓ Response latency: {latency:.2f}s")
        print(f"✓ Response: {result['choices'][0]['message']['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Function calling test failed: {e}")
        return False

if __name__ == "__main__":
    test_qwen3_deployment()
```

### Step 5: Load Testing

```python
# load_test.py
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def test_concurrent_requests():
    """Test concurrent request handling."""
    
    async def make_request(session, request_id):
        payload = {
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "messages": [
                {"role": "user", "content": f"Analyze this data: request {request_id}"}
            ],
            "max_tokens": 200
        }
        
        start_time = time.time()
        async with session.post("http://localhost:8000/v1/chat/completions", 
                               json=payload) as response:
            result = await response.json()
            latency = time.time() - start_time
            return request_id, latency, len(result['choices'][0]['message']['content'])
    
    # Test 5 concurrent requests
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        total_time = max(r[1] for r in results)
        avg_latency = sum(r[1] for r in results) / len(results)
        
        print(f"Concurrent requests: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average latency: {avg_latency:.2f}s")
        print(f"Throughput: {len(results)/total_time:.2f} req/s")

if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())
```

## Integration with DocMind AI

### LlamaIndex Integration

```python
# docmind_llm.py
from llama_index.llms.vllm import Vllm
from llama_index.core import Settings
from typing import Optional

class DocMindLLM:
    """DocMind AI optimized Qwen3 integration."""
    
    def __init__(self, model_url: str = "http://localhost:8000"):
        self.model_url = model_url
        self.llm = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LlamaIndex LLM interface."""
        # Use OpenAI-compatible API
        from llama_index.llms.openai_like import OpenAILike
        
        self.llm = OpenAILike(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            api_base=f"{self.model_url}/v1",
            api_key="EMPTY",
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=262144,
            max_tokens=16384,
            temperature=0.7
        )
        
        # Set global LlamaIndex configuration
        Settings.llm = self.llm
        Settings.context_window = 262144
        Settings.num_output = 16384
    
    async def analyze_document(self, content: str, focus: Optional[str] = None) -> str:
        """Analyze document with native 256K context."""
        
        prompt = f"""
        Analyze the following document comprehensively:
        
        {content}
        
        {"Focus on: " + focus if focus else ""}
        
        Provide:
        1. Key themes and insights
        2. Important details and facts  
        3. Actionable conclusions
        4. Relevant questions for follow-up
        """
        
        response = await self.llm.acomplete(prompt)
        return response.text
    
    async def multi_agent_query(self, query: str, context_docs: list) -> dict:
        """Execute multi-agent query with function calling."""
        
        # Combine documents within 256K context
        combined_context = "\n\n---\n\n".join(context_docs)
        
        prompt = f"""
        You are part of a multi-agent RAG system. Analyze this query and context.
        
        Query: {query}
        
        Context Documents:
        {combined_context}
        
        Respond with structured analysis including:
        - Direct answer to the query
        - Supporting evidence from documents
        - Confidence level
        - Suggested follow-up actions
        """
        
        response = await self.llm.acomplete(prompt)
        
        return {
            "answer": response.text,
            "model": "qwen3-30b-a3b-instruct-2507",
            "context_tokens": len(combined_context.split()),
            "response_tokens": len(response.text.split())
        }
```

### Function Calling Setup

```python
# function_calling.py
from qwen_agent.agents import Assistant

def setup_docmind_agent():
    """Setup DocMind AI agent with Qwen3."""
    
    # LLM configuration
    llm_cfg = {
        'model': 'Qwen3-30B-A3B-Instruct-2507',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'max_tokens': 16384,
            'temperature': 0.7,
            'top_p': 0.8
        }
    }
    
    # Tool definitions for DocMind AI
    tools = [
        {
            'mcpServers': {
                'time': {
                    'command': 'uvx',
                    'args': ['mcp-server-time']
                },
                'fetch': {
                    'command': 'uvx',
                    'args': ['mcp-server-fetch']
                },
                'qdrant': {
                    'command': 'uvx',
                    'args': ['mcp-server-qdrant']
                }
            }
        },
        'code_interpreter',
        'file_browser'
    ]
    
    # Initialize agent
    agent = Assistant(llm=llm_cfg, function_list=tools)
    
    return agent

# Example usage
async def run_docmind_query():
    """Example DocMind AI query execution."""
    
    agent = setup_docmind_agent()
    
    query = """
    Search for documents related to 'machine learning optimization' 
    and provide a comprehensive summary of the latest techniques.
    """
    
    messages = [{'role': 'user', 'content': query}]
    
    responses = []
    async for response in agent.run(messages=messages):
        responses.append(response)
    
    return responses[-1]  # Final response
```

## Monitoring and Maintenance

### Performance Monitoring

```python
# monitoring.py
import psutil
import GPUtil
import time
import requests
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class SystemMetrics:
    vram_used_gb: float
    vram_total_gb: float
    ram_used_gb: float
    ram_total_gb: float
    cpu_percent: float
    gpu_utilization: float

@dataclass  
class ModelMetrics:
    requests_per_minute: float
    avg_latency_ms: float
    active_experts: int
    expert_swaps_per_minute: float
    context_utilization: float

class Qwen3Monitor:
    """Monitor Qwen3 deployment health and performance."""
    
    def __init__(self, model_url: str = "http://localhost:8000"):
        self.model_url = model_url
        self.metrics_history: List[Dict] = []
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics."""
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        gpu = gpus[0] if gpus else None
        
        vram_used = gpu.memoryUsed / 1024 if gpu else 0  # MB to GB
        vram_total = gpu.memoryTotal / 1024 if gpu else 0
        gpu_util = gpu.load * 100 if gpu else 0
        
        # RAM metrics
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)  # Bytes to GB
        ram_total = ram.total / (1024**3)
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return SystemMetrics(
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
            cpu_percent=cpu_percent,
            gpu_utilization=gpu_util
        )
    
    def get_model_metrics(self) -> ModelMetrics:
        """Collect model-specific metrics."""
        
        try:
            # Get model stats from vLLM
            response = requests.get(f"{self.model_url}/metrics", timeout=5)
            stats = response.json() if response.status_code == 200 else {}
            
            return ModelMetrics(
                requests_per_minute=stats.get('requests_per_minute', 0),
                avg_latency_ms=stats.get('avg_latency_ms', 0),
                active_experts=stats.get('active_experts', 0),
                expert_swaps_per_minute=stats.get('expert_swaps_per_minute', 0),
                context_utilization=stats.get('context_utilization', 0)
            )
            
        except Exception as e:
            print(f"Error collecting model metrics: {e}")
            return ModelMetrics(0, 0, 0, 0, 0)
    
    def check_alerts(self, system: SystemMetrics, model: ModelMetrics) -> List[str]:
        """Check for alert conditions."""
        
        alerts = []
        
        # Resource alerts
        if system.vram_used_gb / system.vram_total_gb > 0.9:
            alerts.append(f"High VRAM usage: {system.vram_used_gb:.1f}GB / {system.vram_total_gb:.1f}GB")
        
        if system.ram_used_gb / system.ram_total_gb > 0.9:
            alerts.append(f"High RAM usage: {system.ram_used_gb:.1f}GB / {system.ram_total_gb:.1f}GB")
        
        # Performance alerts
        if model.avg_latency_ms > 5000:
            alerts.append(f"High latency: {model.avg_latency_ms:.0f}ms")
        
        if model.expert_swaps_per_minute > 100:
            alerts.append(f"Excessive expert swaps: {model.expert_swaps_per_minute:.0f}/min")
        
        return alerts
    
    def run_monitoring_loop(self, interval_seconds: int = 60):
        """Run continuous monitoring loop."""
        
        print("Starting Qwen3 monitoring...")
        
        while True:
            try:
                system_metrics = self.get_system_metrics()
                model_metrics = self.get_model_metrics()
                
                # Check for alerts
                alerts = self.check_alerts(system_metrics, model_metrics)
                
                # Log metrics
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Qwen3 Status:")
                print(f"  VRAM: {system_metrics.vram_used_gb:.1f}GB / {system_metrics.vram_total_gb:.1f}GB")
                print(f"  RAM: {system_metrics.ram_used_gb:.1f}GB / {system_metrics.ram_total_gb:.1f}GB")
                print(f"  GPU: {system_metrics.gpu_utilization:.1f}%")
                print(f"  Requests/min: {model_metrics.requests_per_minute:.1f}")
                print(f"  Avg latency: {model_metrics.avg_latency_ms:.0f}ms")
                
                # Show alerts
                if alerts:
                    print("  ALERTS:")
                    for alert in alerts:
                        print(f"    ⚠️  {alert}")
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval_seconds)

if __name__ == "__main__":
    monitor = Qwen3Monitor()
    monitor.run_monitoring_loop()
```

## Troubleshooting Guide

### Common Issues

#### 1. Out of Memory Errors

**Symptoms**: CUDA OOM, model loading failures
**Solutions**:

```bash
# Reduce tensor parallel size
--tensor-parallel-size 1

# Reduce memory utilization
--gpu-memory-utilization 0.7

# Enable CPU offloading
--cpu-offload-gb 8
```

#### 2. Slow Expert Loading

**Symptoms**: High latency spikes, timeout errors
**Solutions**:

```bash
# Use NVMe storage for model
mv models/ /nvme/models/

# Increase system RAM
# Add RAM for expert caching

# Reduce batch size
--max-num-batched-tokens 16384
```

#### 3. Connection Issues

**Symptoms**: Health check failures, connection refused
**Solutions**:

```bash
# Check service status
curl http://localhost:8000/health

# Verify port binding
netstat -tlnp | grep 8000

# Check logs
tail -f vllm.log
```

#### 4. Poor Performance

**Symptoms**: Low throughput, high latency
**Solutions**:

```bash
# Enable flash attention
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Optimize chunked prefill
--chunked-prefill-size 16384

# Reduce context length if not needed
--max-model-len 131072
```

## Security Considerations

### Network Security

```bash
# Bind to localhost only (default)
--host 127.0.0.1 --port 8000

# Use reverse proxy for external access
# Configure nginx/Apache with SSL

# API key authentication (if needed)
--api-key your_secure_api_key
```

### Model Security

```bash
# Verify model checksums
sha256sum models/qwen3-30b-a3b-instruct-2507-fp8/*

# Use trusted model sources only
# Monitor for unauthorized model modifications

# Implement request rate limiting
# Log all API requests for audit
```

## Performance Optimization

### Memory Optimization

```python
# Optimize expert offloading
OPTIMIZATION_CONFIG = {
    "expert_cache_size": "16GB",  # System RAM for expert caching
    "expert_swap_threshold": 0.8,  # VRAM threshold for swapping
    "prefetch_experts": True,      # Predictive expert loading
    "kv_cache_dtype": "int8",      # Reduce KV cache memory
    "attention_backend": "flash_attn"  # Optimized attention
}
```

### Throughput Optimization

```python
# Batching optimization
THROUGHPUT_CONFIG = {
    "max_num_seqs": 16,           # Increase batch size
    "max_num_batched_tokens": 65536,  # Larger token batches
    "enable_chunked_prefill": True,    # Better memory usage
    "tensor_parallel_size": 2,         # Multi-GPU parallelism
}
```

## Conclusion

Qwen3-30B-A3B-Instruct-2507 deployment requires careful attention to MoE architecture specifics, but provides exceptional value through enterprise-grade capabilities on consumer hardware. The native 256K context and agent optimization make it ideal for DocMind AI's multi-agent RAG requirements.

**Key Success Factors**:

1. Adequate system RAM for expert offloading (32GB+)
2. Proper vLLM/SGLang configuration for MoE
3. Monitoring of expert activation patterns
4. Optimization of memory utilization across VRAM/RAM

Follow this guide systematically for successful deployment and optimal performance.

---

*This deployment strategy is optimized for DocMind AI architecture and Qwen3-30B-A3B-Instruct-2507 specific requirements.*
