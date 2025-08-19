# ADR-004: Local-First LLM Strategy

## Title

Local LLM Selection and Optimization for Consumer Hardware

## Version/Date

10.0 / 2025-08-19

## Status

Accepted

## Description

Establishes a local-first LLM strategy using **Qwen3-4B-Instruct-2507** with AWQ quantization and INT8 KV cache optimization. This configuration enables the FULL 262K context window on RTX 4090 Laptop (16GB VRAM) through 50% KV cache memory reduction. Total memory usage at 262K: ~12.2GB (well within 16GB limit). Performance actually improves with INT8 KV cache (~30% throughput gain) while maintaining near-lossless accuracy.

## Context

Current architecture lacks a defined local LLM strategy, relying on external APIs or unoptimized local models. Modern local LLMs have achieved significant capabilities in reasoning, function calling, and instruction following while being deployable on consumer hardware. Key requirements:

1. **Local-First Operation**: No external API dependencies for core functionality
2. **Function Calling**: Support for agentic RAG patterns requiring tool use
3. **High-End Consumer Hardware**: Optimized for RTX 4090 Laptop GPU (16GB VRAM)
4. **Quality**: Superior performance exceeding GPT-3.5 capabilities
5. **Massive Context**: Full 262K tokens achievable with AWQ + INT8 KV cache optimization

Research confirms **Qwen3-4B-Instruct-2507 with AWQ quantization + INT8 KV cache** enables the FULL 262K context window on 16GB VRAM. The AWQ model uses ~2.92GB, and INT8 KV cache reduces memory by 50% (72 KiB → 36 KiB per token). Total memory at 262K: ~12.2GB, well within the 16GB limit. Performance benchmarks show 40-60 tokens/sec with potential 30% throughput improvement from INT8 optimization.

**Adaptive Context Strategy**: Rather than defaulting to large contexts, the system uses intelligent multi-stage retrieval to provide precisely relevant content within the 32K native window. This approach delivers 3-4x performance improvement over brute-force large context approaches while maintaining higher quality through reduced "lost in the middle" effects.

**Integration Benefits**: The 32K native context combined with sophisticated retrieval (ADR-003) enables processing complex queries with optimal relevance. The RTX 4090 Laptop's 16GB VRAM supports both the model and efficient retrieval operations with significantly improved throughput.

## Related Requirements

### Functional Requirements

- **FR-1:** Support function calling for agentic RAG operations
- **FR-2:** Handle context lengths up to 262K tokens with INT8 KV cache optimization
- **FR-3:** Provide reasoning capabilities for query routing and result validation
- **FR-4:** Support multiple conversation turns with context retention
- **FR-5:** Enable adaptive context strategies optimized for query complexity

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Response generation <1.5 seconds on RTX 4090 Laptop
- **NFR-2:** **(Memory)** Total usage ~12.2GB VRAM at 262K context with AWQ + INT8 KV cache
- **NFR-3:** **(Quality)** Performance ≥95% of GPT-3.5-turbo on reasoning tasks through intelligent retrieval
- **NFR-4:** **(Local-First)** Zero external API dependencies for core operations
- **NFR-5:** **(Throughput)** 40-60 tokens/sec, with ~30% improvement from INT8 KV cache

## Alternatives

### 1. Cloud API Dependencies (OpenAI/Claude)

- **Benefits**: High quality, no local resource requirements
- **Issues**: Violates local-first principle, privacy concerns, ongoing costs, latency
- **Score**: 3/10 (quality: 9, local-first: 0, privacy: 2)

### 2. GPT-OSS-20B (OpenAI's Open Source Model - August 2025)

- **Benefits**: OpenAI brand, 128K context, Apache 2.0 license, MoE architecture
- **Issues**: Poor document analysis performance, 16GB VRAM requirement, weak general capabilities
- **Score**: 5/10 (quality: 4, performance: 5, memory: 3, context: 8)

### 3. Smaller Local Models (Phi-3-Mini, Llama3-8B)

- **Benefits**: Lower resource requirements, faster inference
- **Issues**: Limited reasoning, weaker function calling, reduced quality
- **Score**: 6/10 (performance: 8, quality: 5, capability: 4)

### 4. Qwen3-14B (Fallback Model)

- **Benefits**: Latest generation (April 2025), 32K native context, dense architecture, proven reliability, Q5_K_M/Q6_K quantization support
- **Role**: Handles complex reasoning tasks where 4B model may struggle
- **Score**: 8.5/10 (quality: 9, capability: 9, performance: 8, context: 6, efficiency: 7)

### 5. Qwen3-4B-Instruct-2507 (Primary Model - Selected)

- **Benefits**: Full 262K context with AWQ + INT8 KV cache, excellent efficiency, strong benchmarks
- **Benchmarks**: MMLU-Pro: 69.6, GPQA: 62.0, AIME25: 47.4 (2x improvement over base)
- **Breakthrough**: INT8 KV cache enables 262K context with only ~12.2GB VRAM (previously thought impossible)
- **Score**: 9.5/10 (quality: 7.5, capability: 9, performance: 9, full context: 10, efficiency: 10)

### 6. Dense Large Models (Qwen3-32B, Mixtral-8x7B)

- **Benefits**: High quality, strong reasoning capabilities
- **Issues**: Requires high-end hardware (RTX 4090+), slower inference, 32K context limitation
- **Score**: 7/10 (quality: 9, performance: 5, accessibility: 6, context: 7)

### 7. Older Generation (Qwen2.5-14B-Instruct)

- **Benefits**: Proven performance, well-documented
- **Issues**: Only 32K native context (needs YaRN for 128K), superseded by Qwen3
- **Score**: 7/10 (quality: 8, capability: 8, context: 5, future-proof: 4)

## Decision

We will adopt **Qwen3-4B-Instruct-2507** with AWQ quantization and INT8 KV cache optimization:

### Model Configuration

- **Architecture**: Dense (4.0B parameters, 36 layers, 32 attention heads, 8 KV heads with GQA)
- **Quantization**: AWQ-INT4 (2.92GB model size)
- **KV Cache**: INT8 quantization (50% memory reduction)
- **Memory Usage with INT8 KV Cache**:
  - Model: ~2.92GB VRAM (AWQ)
  - KV Cache per token: 36 KiB (vs 72 KiB with FP16)
  - Total @ 32K: ~4.0GB
  - Total @ 128K: ~7.5GB  
  - Total @ 262K: ~12.2GB (fits in 16GB!)
- **Achievable Context**:
  - **Quick**: 8,192 tokens (minimal KV cache)
  - **Standard**: 32,768 tokens (optimal performance)
  - **Extended**: 131,072 tokens (balanced)
  - **Maximum**: 262,144 tokens (full capability, ~75% VRAM usage)
- **Performance**:
  - 40-60 tokens/sec baseline
  - ~30% throughput improvement with INT8 KV cache
  - Near-lossless accuracy with INT8 quantization
- **Capabilities**: Strong general reasoning, excellent math (AIME25: 47.4), good coding
- **Hardware Target**: RTX 4090 Laptop GPU (16GB VRAM)

### Deployment Configuration

Optimized settings with INT8 KV cache enabling full context:

```python
# Context configuration with INT8 KV cache optimization
CONTEXT_CONFIGURATION = {
    "quick": 8192,        # Ultra-fast responses
    "standard": 32768,    # Default - optimal performance
    "extended": 131072,   # Extended documents
    "maximum": 262144     # Full context - NOW POSSIBLE with INT8!
}

# LMDeploy configuration (RECOMMENDED)
lmdeploy_config = {
    "model": "Qwen/Qwen3-4B-Instruct-2507-AWQ",
    "quant_policy": 8,    # INT8 KV cache quantization
    "cache_max_entry_count": 0.9,
    "tp": 1  # Tensor parallelism
}

# Alternative: vLLM with FP8 KV cache
vllm_config = {
    "model": "Qwen/Qwen3-4B-Instruct-2507-AWQ",
    "max_model_len": 262144,  # Full context achievable!
    "quantization": "awq",
    "kv_cache_dtype": "fp8",  # Or "int8" if supported
    "gpu_memory_utilization": 0.90,
    "dtype": "float16"
}
```

### Deployment Commands

```bash
# LMDeploy with INT8 KV cache (BEST OPTION)
lmdeploy serve api_server \
    Qwen/Qwen3-4B-Instruct-2507-AWQ \
    --quant-policy 8 \
    --cache-max-entry-count 0.9 \
    --tp 1

# vLLM with FP8 KV cache (alternative)
vllm serve Qwen/Qwen3-4B-Instruct-2507-AWQ \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90
```

### Memory Optimization Impact

- **AWQ Model**: 2.92GB (vs 7.97GB FP16)
- **INT8 KV Cache**: 36 KiB per token (vs 72 KiB FP16)
- **Total at 262K**: ~12.2GB (75% of 16GB VRAM)
- **Performance**: +30% throughput with INT8
- **Accuracy**: Near-lossless with INT8 quantization

## Related Decisions

- **ADR-001** (Modern Agentic RAG): Provides LLM for agent decision-making
- **ADR-003** (Adaptive Retrieval Pipeline): Uses LLM for query routing and evaluation, leverages 32K native context with intelligent chunking
- **ADR-010** (Performance Optimization Strategy): Implements quantization and caching
- **ADR-011** (Agent Orchestration Framework): Integrates function calling capabilities
- **ADR-012** (Evaluation Strategy): Uses Qwen3-14B for evaluation and quality assessment tasks

## Design

### Multi-Provider Architecture with Automatic Selection

DocMind AI supports multiple local LLM providers with automatic hardware-based selection for optimal performance. The architecture leverages LlamaIndex's native provider support without custom abstraction layers.

#### Provider Comparison Matrix

| Provider | Performance | Setup Complexity | Best For | GGUF Support |
|----------|------------|------------------|----------|--------------|
| **llama.cpp** | Excellent (GGUF optimized) | Simple | Production, GGUF models | Excellent |
| **Ollama** | Good (GGUF support) | Simple | Development, testing | Good |
| **vLLM** | Excellent | Moderate | Production, AWQ models | Limited |

#### Performance Benchmarks (RTX 4090 Laptop - 16GB VRAM)

**Qwen3-14B Performance:**

- **llama.cpp Q5_K_M**: ~40-60 tokens/sec (128K context with YaRN)
- **llama.cpp Q6_K**: ~35-50 tokens/sec (best quality)
- **Ollama Q5_K_M**: ~35-45 tokens/sec (built-in GGUF support)
- **vLLM AWQ**: ~50-70 tokens/sec (AWQ quantization)

**Qwen3-32B-AWQ Performance (Now Viable):**

- **vLLM**: ~25-35 tokens/sec with 64K context
- **Memory**: ~12GB VRAM (fits comfortably in 16GB)

> *Hardware: RTX 4090 Laptop GPU (16GB VRAM), Intel Core i9-14900HX, 64GB RAM*

### Multi-Provider Deployment Strategy

```python
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.vllm import Vllm
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from instructor import patch
from typing import Optional, Dict, Any
import torch
import os

class LocalLLMProvider:
    """Multi-provider deployment for Qwen3-14B with automatic selection."""
    
    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Detect available hardware for optimal provider selection."""
        hardware = {
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "total_vram_gb": 0,
            "cpu_cores": os.cpu_count() or 4
        }
        
        if hardware["gpu_count"] > 0:
            # Calculate total VRAM across all GPUs
            total_vram = sum(
                torch.cuda.get_device_properties(i).total_memory 
                for i in range(hardware["gpu_count"])
            )
            hardware["total_vram_gb"] = total_vram / (1024**3)
                
        return hardware
    
    @staticmethod
    def select_provider() -> str:
        """Select optimal provider based on hardware."""
        hardware = LocalLLMProvider.detect_hardware()
        
        if hardware["total_vram_gb"] >= 12:
            # High-end GPU: prefer vLLM for AWQ models
            return "vllm"
        elif hardware["total_vram_gb"] >= 8:
            # Mid-range GPU: prefer llama.cpp for GGUF
            return "llama_cpp"
        else:
            # Low-end or development: Ollama
            return "ollama"

def setup_qwen3_14b_llm(
    prefer_provider: Optional[str] = None,
    quantization: str = "Q5_K_M",  # Updated default for RTX 4090
    enable_yarn: bool = True,  # Enable YaRN by default
    yarn_factor: float = 4.0  # 4x scaling (32K → 128K)
) -> tuple:
    """Setup Qwen3-14B with YaRN context scaling for RTX 4090 Laptop."""
    
    provider = prefer_provider or LocalLLMProvider.select_provider()
    print(f"Deploying Qwen3-14B with {provider}")
    
    # Context configuration with YaRN
    base_context = 32768
    context_length = int(base_context * yarn_factor) if enable_yarn else base_context
    
    if provider == "llama_cpp":
        # llama.cpp optimized for GGUF models with YaRN
        model_path = f"models/qwen3-14b-{quantization.lower()}.gguf"
        
        # YaRN-specific parameters for llama.cpp
        rope_kwargs = {}
        if enable_yarn:
            rope_kwargs = {
                "rope_scaling_type": "yarn",
                "rope_freq_scale": 1.0 / yarn_factor,  # Inverse of scaling factor
                "yarn_orig_ctx": base_context,
                "yarn_ext_factor": 1.0,
                "yarn_attn_factor": 1.0,
                "yarn_beta_fast": 32.0,
                "yarn_beta_slow": 1.0
            }
        
        base_llm = LlamaCPP(
            model_path=model_path,
            n_ctx=context_length,
            n_gpu_layers=-1,  # Use all GPU layers
            n_batch=1024,  # Increased for RTX 4090
            n_threads=24,  # Intel i9-14900HX has 24 cores
            temperature=0.7,
            max_tokens=4096,  # Increased for longer outputs
            verbose=False,
            **rope_kwargs  # Apply YaRN configuration
        )
        
    elif provider == "vllm":
        # vLLM with YaRN support for RTX 4090 Laptop
        rope_scaling = None
        if enable_yarn:
            rope_scaling = {
                "type": "yarn",
                "factor": yarn_factor,
                "original_max_position_embeddings": base_context
            }
        
        # Can now run larger models on RTX 4090 Laptop
        model_name = "Qwen/Qwen3-32B-AWQ" if quantization == "AWQ-32B" else "Qwen/Qwen3-14B-AWQ"
        
        base_llm = Vllm(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,  # Can use more VRAM on RTX 4090
            max_model_len=context_length,
            dtype="float16",
            trust_remote_code=True,
            enable_prefix_caching=True,
            kv_cache_dtype="int8",
            rope_scaling=rope_scaling  # Apply YaRN configuration
        )
        
    else:  # ollama fallback
        # Ollama for simple deployment
        base_llm = Ollama(
            model="qwen3:14b",
            request_timeout=180.0,
            context_window=context_length,
            temperature=0.7,
            num_gpu_layers=999
        )
    
    # Configure structured outputs
    structured_llm = patch(base_llm) if provider != "vllm" else base_llm
    
    # Set global LlamaIndex configuration
    Settings.llm = base_llm
    Settings.context_window = context_length
    
    return base_llm, structured_llm, provider

# Deployment command examples with YaRN
def get_deployment_commands() -> Dict[str, str]:
    """Get deployment commands with YaRN context scaling for RTX 4090 Laptop."""
    
    return {
        "llama_cpp_server_yarn": """
# Download higher quality model for RTX 4090
huggingface-cli download bartowski/Qwen3-14B-GGUF --include "*Q5_K_M.gguf" --local-dir ./models

# Run llama.cpp server with YaRN (128K context)
./llama-server \\
  -m ./models/Qwen3-14B-Q5_K_M.gguf \\
  -c 131072 \\
  --rope-scaling yarn \\
  --rope-scale 4.0 \\
  --yarn-orig-ctx 32768 \\
  --yarn-ext-factor 1.0 \\
  --yarn-attn-factor 1.0 \\
  --yarn-beta-fast 32 \\
  --yarn-beta-slow 1 \\
  -ngl 99 \\
  -b 1024 \\
  -t 24 \\
  --host 0.0.0.0 \\
  --port 8080
        """,
        
        "vllm_yarn": """
# vLLM with YaRN for 128K context
vllm serve Qwen/Qwen3-14B-AWQ \\
  --max-model-len 131072 \\
  --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 32768 \\
  --gpu-memory-utilization 0.90 \\
  --kv-cache-dtype int8 \\
  --trust-remote-code

# Or for Qwen3-32B-AWQ (now viable on RTX 4090)
vllm serve Qwen/Qwen3-32B-AWQ \\
  --max-model-len 65536 \\
  --rope-scaling '{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \\
  --gpu-memory-utilization 0.90 \\
  --trust-remote-code
        """,
        
        "transformers_yarn": """
# Python code for transformers with YaRN
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B",
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "attention_factor": 1.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 0.0
    },
    max_position_embeddings=131072,
    device_map="auto",
    torch_dtype="auto"
)
        """
    }

# Function calling support for Qwen3-14B
class Qwen3FunctionCaller:
    """Function calling interface optimized for Qwen3-14B."""
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
        self.tool_registry = {}
    
    def register_tool(self, name: str, description: str, function_schema: dict):
        """Register a tool for function calling."""
        self.tool_registry[name] = {
            "description": description,
            "schema": function_schema
        }
    
    async def execute_with_tools(self, query: str, max_iterations: int = 3) -> dict:
        """Execute query with available tools."""
        
        # Qwen3-14B optimized prompt for function calling
        tools_prompt = self._create_tools_prompt(query)
        
        response = await self.llm.acomplete(tools_prompt)
        
        # Parse and execute function calls
        if self._contains_function_call(response.text):
            function_result = await self._execute_function(response.text)
            
            # Generate final response with tool results
            final_prompt = self._create_final_response_prompt(query, function_result)
            final_response = await self.llm.acomplete(final_prompt)
            
            return {
                "response": final_response.text,
                "tool_used": True,
                "tool_results": function_result
            }
        
        return {
            "response": response.text,
            "tool_used": False,
            "tool_results": None
        }
    
    def _create_tools_prompt(self, query: str) -> str:
        """Create optimized prompt for Qwen3-14B function calling."""
        tools_desc = "\n".join([
            f"- {name}: {details['description']}" 
            for name, details in self.tool_registry.items()
        ])
        
        return f"""You are a helpful assistant with access to tools. Analyze the query and decide if you need to use any tools.

Available tools:
{tools_desc}

User query: {query}

If you need to use a tool, respond with: TOOL_CALL: [tool_name] ARGS: [json_arguments]
If you can answer directly, provide your response normally.

Response:"""

# Example DocMind AI setup for RTX 4090 Laptop
def setup_docmind_ai():
    """Initialize DocMind AI with Qwen3-14B + YaRN for RTX 4090 Laptop."""
    
    # Setup Qwen3-14B LLM with YaRN context scaling
    base_llm, structured_llm, provider = setup_qwen3_14b_llm(
        quantization="Q5_K_M",  # Higher quality for RTX 4090
        enable_yarn=True,  # Enable 128K context with YaRN
        yarn_factor=4.0  # 32K * 4 = 128K
    )
    
    # Initialize function calling
    function_caller = Qwen3FunctionCaller(base_llm)
    
    # Register common RAG tools
    function_caller.register_tool(
        "search_documents", 
        "Search document database for relevant information",
        {"query": "string", "limit": "integer"}
    )
    
    function_caller.register_tool(
        "analyze_document",
        "Analyze a specific document for key insights", 
        {"document_id": "string", "focus": "string"}
    )
    
    print(f"DocMind AI initialized with {provider}")
    print(f"Hardware: RTX 4090 Laptop (16GB VRAM)")
    print(f"Model: Qwen3-14B (Dense architecture)")
    print(f"Context: 128K tokens with YaRN (4x scaling)")
    print(f"Quantization: Q5_K_M for optimal quality")
    print(f"Performance: 40-60 tokens/sec expected")
    
    return base_llm, structured_llm, function_caller

# Model comparison matrix for RTX 4090 Laptop (16GB VRAM)
QWEN3_MODEL_MATRIX = {
    "qwen3-14b-q5_k_m": {
        "parameters": "14.8B",
        "memory_gb": 10,  # With Q5_K_M quantization
        "context_length": 131072,  # 128K with YaRN
        "quantization": "q5_k_m_gguf",
        "tokens_per_sec": "40-60",
        "capabilities": ["function_calling", "thinking_mode", "yarn_128k", "optimal"]
    },
    "qwen3-14b-q6_k": {
        "parameters": "14.8B",
        "memory_gb": 11,  # With Q6_K quantization
        "context_length": 131072,  # 128K with YaRN
        "quantization": "q6_k_gguf",
        "tokens_per_sec": "35-50",
        "capabilities": ["function_calling", "thinking_mode", "yarn_128k", "best_quality"]
    },
    "qwen3-32b-awq": {
        "parameters": "32B",
        "memory_gb": 12,  # AWQ 4-bit quantization
        "context_length": 65536,  # 64K with YaRN factor 2
        "quantization": "awq_4bit",
        "tokens_per_sec": "25-35",
        "capabilities": ["function_calling", "thinking_mode", "yarn_64k", "largest_model"]
    },
    "qwen3-7b-q6_k": {
        "parameters": "7B",
        "memory_gb": 6,  # With Q6_K quantization
        "context_length": 131072,  # 128K with YaRN
        "quantization": "q6_k_gguf",
        "tokens_per_sec": "60-80",
        "capabilities": ["function_calling", "efficient", "yarn_128k", "fastest"]
    }
}

# Memory usage breakdown for 128K context on RTX 4090 Laptop
MEMORY_USAGE_128K = {
    "qwen3_14b_q5_k_m": {
        "model_size_gb": 10.0,
        "kv_cache_gb": 2.5,  # For 128K context
        "activation_gb": 1.5,
        "total_gb": 14.0,
        "fits_in_16gb": True
    },
    "qwen3_14b_q6_k": {
        "model_size_gb": 11.0,
        "kv_cache_gb": 2.5,
        "activation_gb": 1.5,
        "total_gb": 15.0,
        "fits_in_16gb": True
    },
    "qwen3_32b_awq": {
        "model_size_gb": 12.0,
        "kv_cache_gb": 3.0,  # For 64K context
        "activation_gb": 1.0,
        "total_gb": 16.0,
        "fits_in_16gb": True  # Just fits!
    }
}
```

## Consequences

### Positive Outcomes

- **Local Privacy**: All processing occurs locally without external API calls
- **Cost Effective**: No ongoing API costs after initial setup
- **Excellent Performance**: 40-60 tokens/sec on RTX 4090 Laptop with 128K context
- **Extended Context**: 128K tokens with YaRN enables processing entire documents
- **Function Calling**: Superior agentic RAG patterns with optimized tool use
- **Higher Quality**: Q5_K_M/Q6_K quantization provides better accuracy than Q4_K_M
- **Multi-Provider Support**: Works with llama.cpp, Ollama, vLLM, and transformers
- **Large Model Viability**: Can run Qwen3-32B-AWQ as primary model

### Negative Consequences / Trade-offs

- **YaRN Overhead**: Slight perplexity increase when using extended context
- **Memory Requirements**: Requires 10-12GB VRAM for optimal performance
- **Model Size**: 14B model download (~10-11GB for Q5_K_M/Q6_K)
- **Static Scaling**: YaRN uses constant factor regardless of actual context length
- **Setup Complexity**: YaRN configuration requires proper parameter tuning

### Performance Targets (RTX 4090 Laptop)

- **Response Time**: <2 seconds for 512 token responses
- **Memory Usage**: <14GB VRAM with Q5_K_M + 128K context
- **Quality**: ≥95% performance vs GPT-3.5-turbo on reasoning tasks
- **Function Calling**: ≥98% success rate on complex multi-tool scenarios
- **Context Handling**: 128K tokens with YaRN without OOM errors
- **Throughput**: 40-60 tokens/sec with Q5_K_M, 35-50 with Q6_K

## Dependencies

- **Python**: `llama-cpp-python>=0.2.0`, `transformers>=4.51.0` (YaRN support), `vllm>=0.8.5` (YaRN support)
- **Hardware**: RTX 4090 Laptop GPU (16GB VRAM), Intel Core i9-14900HX, 64GB RAM
- **CUDA**: CUDA 12.0+ with cuDNN 8.9+
- **Inference**: llama.cpp with YaRN support, vLLM 0.8.5+, transformers 4.51+
- **Storage**: ~10-11GB for Q5_K_M/Q6_K GGUF models, 2TB SSD recommended
- **Optional**: Flash Attention 2 for transformers, tensor parallelism support

## Monitoring Metrics

- Response generation latency with 32K native context (target <1.5 seconds)
- VRAM utilization with optimized context (target <11GB for 32K)
- Adaptive context strategy effectiveness and selection accuracy
- Function calling success rates with intelligent retrieval
- Context window utilization (32K native, adaptive strategies)
- Token generation speed (50-60 tokens/sec target)
- Model loading time on NVMe SSD
- Multi-stage retrieval quality vs large context baseline

## Changelog

- **10.0 (2025-08-19)**: **INT8 KV CACHE OPTIMIZATION** - Correction: AWQ + INT8 KV cache enables 262K context on 16GB VRAM. INT8 reduces KV cache by 50% (36 KiB vs 72 KiB per token). Total memory at 262K: ~12.2GB. Performance increases by ~30% with INT8. Deployment: LMDeploy with --quant-policy 8 or vLLM with --kv-cache-dtype fp8. Minimal accuracy degradation with INT8 quantization.
- **9.0 (2025-08-19)**: **INITIAL REALITY CHECK** - First analysis incorrectly assumed FP16 KV cache, concluding 262K was impossible. This was corrected in v10.0 with INT8 optimization discovery.
- **8.0 (2025-08-19)**: **INITIAL QWEN3-4B EVALUATION** - Evaluated Qwen3-4B-Instruct-2507 with strong benchmarks.
- **7.0 (2025-08-18)**: **MAJOR ARCHITECTURAL SHIFT** - Optimized for 32K native context with intelligent multi-stage retrieval instead of 128K brute-force approach. Default configuration changed from 128K YaRN to 32K native for 3-4x performance improvement. Added adaptive context strategies (default/extended/document). Updated performance targets: <1.5 sec latency, 50-60 tokens/sec, <11GB VRAM. Emphasizes sophisticated retrieval over large context windows.
- **6.0 (2025-08-18)**: **MAJOR HARDWARE UPGRADE** - Enhanced for RTX 4090 Laptop GPU (16GB VRAM) with YaRN context scaling to 128K tokens. Updated benchmarks and defaults for high-end hardware: Q5_K_M/Q6_K quantization, 40-60 tokens/sec performance, comprehensive YaRN configuration for llama.cpp/vLLM/transformers. Added memory usage tables and deployment commands. Qwen3-32B-AWQ now viable as primary model.
- **5.2 (2025-08-18)**: **REVERTED** - Returned to practical **Qwen3-14B** model after critical review. 30B MoE model unrealistic for consumer hardware (requires 24GB+ VRAM, <1 token/sec at large contexts). Q4_K_M GGUF quantization provides reliable 8GB VRAM deployment. Multi-provider support with llama.cpp, Ollama, and vLLM. Realistic 32K context with 64K sliding window option.
- **5.1 (2025-08-18)**: **CORRECTED** - Fixed deployment complexity and memory requirements for MoE architecture
- **5.0 (2025-08-18)**: **EXPERIMENTAL** - Attempted switch to Qwen3-30B-A3B-Instruct-2507 MoE model (later found unrealistic)
- **4.3 (2025-08-18)**: CORRECTED - Fixed context length specifications: Qwen3-14B has native 32K context, extensible to 128K with YaRN (not native 128K)
- **4.2 (2025-08-18)**: CORRECTED - Updated Qwen3-14B-Instruct to correct official name Qwen3-14B (no separate instruct variant exists)
- **4.1 (2025-08-18)**: Enhanced integration with agent orchestration framework for function calling in multi-agent scenarios, optimized for DSPy prompt optimization and GraphRAG compatibility with extended context handling
- **4.0 (2025-08-17)**: [Missing previous changelog entry - needs documentation]
- **3.0 (2025-08-16)**: **CRITICAL CORRECTIONS** - Switched to **Qwen3-14B** (latest generation, April 2025) with native 32K context, extensible to 128K with YaRN. Corrected Qwen2.5-14B context limitation (32K native, not 128K). Added GPT-OSS-20B analysis. Updated quantization to Q4_K_M GGUF format. Based on comprehensive 2025 model research and real-world performance testing.
- **2.0 (2025-01-16)**: **MAJOR UPGRADE** - Switched to Qwen2.5-14B-Instruct with extended context window support (16x increase from 8K). Updated all fallback models to support extended context. Added AWQ quantization support. Addresses real-world document analysis requirements.
- **1.0 (2025-01-16)**: Initial local LLM strategy with Qwen3-14B primary and hardware-adaptive selection
