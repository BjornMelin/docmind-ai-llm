# ADR-004-NEW: Local-First LLM Strategy

## Title

Local LLM Selection and Optimization for Consumer Hardware

## Version/Date

1.0 / 2025-01-16

## Status

Proposed

## Description

Establishes a local-first LLM strategy optimized for consumer hardware (RTX 3060-4090), focusing on Qwen3-14B as the primary model with Mistral-7B and Phi-3-Mini as alternatives. The strategy emphasizes function calling capabilities, quantization optimization, and efficient memory management while maintaining competitive performance for agentic RAG operations.

## Context

Current architecture lacks a defined local LLM strategy, relying on external APIs or unoptimized local models. Modern local LLMs have achieved significant capabilities in reasoning, function calling, and instruction following while being deployable on consumer hardware. Key requirements:

1. **Local-First Operation**: No external API dependencies for core functionality
2. **Function Calling**: Support for agentic RAG patterns requiring tool use
3. **Consumer Hardware**: Efficient operation on RTX 3060-4090 GPUs
4. **Quality**: Competitive performance with GPT-3.5 level capabilities

Research indicates Qwen3-14B, Mistral-7B, and Phi-3 provide optimal balance of capability and efficiency for local deployment.

## Related Requirements

### Functional Requirements

- **FR-1:** Support function calling for agentic RAG operations
- **FR-2:** Handle context lengths up to 8192 tokens for document analysis
- **FR-3:** Provide reasoning capabilities for query routing and result validation
- **FR-4:** Support multiple conversation turns with context retention

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Response generation <3 seconds on RTX 4060
- **NFR-2:** **(Memory)** Model memory usage <12GB VRAM for inference
- **NFR-3:** **(Quality)** Performance ≥85% of GPT-3.5-turbo on reasoning tasks
- **NFR-4:** **(Local-First)** Zero external API dependencies for core operations

## Alternatives

### 1. Cloud API Dependencies (OpenAI/Claude)

- **Benefits**: High quality, no local resource requirements
- **Issues**: Violates local-first principle, privacy concerns, ongoing costs, latency
- **Score**: 3/10 (quality: 9, local-first: 0, privacy: 2)

### 2. Smaller Local Models (Phi-3-Mini, Llama3-8B)

- **Benefits**: Lower resource requirements, faster inference
- **Issues**: Limited reasoning, weaker function calling, reduced quality
- **Score**: 6/10 (performance: 8, quality: 5, capability: 4)

### 3. Qwen3-14B Primary with Fallbacks (Selected)

- **Benefits**: Strong reasoning, excellent function calling, optimal resource balance
- **Issues**: Higher memory requirements than smaller models
- **Score**: 9/10 (quality: 9, capability: 9, performance: 8)

### 4. Large Local Models (Qwen3-32B, Mixtral-8x7B)

- **Benefits**: Highest quality, best reasoning capabilities
- **Issues**: Requires high-end hardware (RTX 4090+), slower inference
- **Score**: 7/10 (quality: 10, performance: 4, accessibility: 6)

## Decision

We will adopt **Qwen3-14B as primary with tiered fallbacks**:

### Primary Model: **Qwen3-14B-Instruct**

- **Parameters**: 14.7B parameters
- **Memory**: ~9GB VRAM with 4-bit quantization
- **Context**: 32K tokens (use 8K for efficiency)
- **Capabilities**: Excellent function calling, strong reasoning, multilingual

### Fallback Models (Hardware Adaptive)

- **Qwen3-7B-Instruct**: For RTX 3060-4060 (8GB VRAM)
- **Phi-3-Mini-4K**: For systems with limited VRAM (<6GB)
- **Mistral-7B-Instruct**: Alternative for different use cases

### Quantization Strategy

- **Primary**: 4-bit GPTQ/AWQ quantization for balance
- **Fallback**: 8-bit quantization if 4-bit quality issues
- **Memory Critical**: Dynamic quantization based on available VRAM

## Related Decisions

- **ADR-001-NEW** (Modern Agentic RAG): Provides LLM for agent decision-making
- **ADR-003-NEW** (Adaptive Retrieval Pipeline): Uses LLM for query routing and evaluation
- **ADR-010-NEW** (Performance Optimization Strategy): Implements quantization and caching
- **ADR-011-NEW** (Agent Orchestration Framework): Integrates function calling capabilities

## Design

### Hardware-Adaptive Model Selection

```python
import torch
import psutil
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for local LLM deployment."""
    name: str
    parameters: str
    memory_gb: float
    context_length: int
    quantization: str
    capabilities: List[str]

class HardwareAdaptiveModelSelector:
    """Selects optimal model based on available hardware."""
    
    MODEL_CONFIGS = {
        "qwen3-14b": ModelConfig(
            name="Qwen/Qwen3-14B-Instruct",
            parameters="14.7B",
            memory_gb=9.0,
            context_length=32768,
            quantization="4bit",
            capabilities=["function_calling", "reasoning", "multilingual"]
        ),
        "qwen3-7b": ModelConfig(
            name="Qwen/Qwen3-7B-Instruct", 
            parameters="7.6B",
            memory_gb=5.5,
            context_length=32768,
            quantization="4bit",
            capabilities=["function_calling", "reasoning"]
        ),
        "phi3-mini": ModelConfig(
            name="microsoft/Phi-3-mini-4k-instruct",
            parameters="3.8B", 
            memory_gb=3.5,
            context_length=4096,
            quantization="8bit",
            capabilities=["reasoning", "lightweight"]
        ),
        "mistral-7b": ModelConfig(
            name="mistralai/Mistral-7B-Instruct-v0.3",
            parameters="7.2B",
            memory_gb=5.0,
            context_length=8192,
            quantization="4bit",
            capabilities=["function_calling", "reasoning"]
        )
    }
    
    def select_model(self) -> ModelConfig:
        """Select optimal model based on available hardware."""
        vram_gb = self._get_available_vram()
        ram_gb = self._get_available_ram()
        
        # Model selection logic
        if vram_gb >= 12:
            return self.MODEL_CONFIGS["qwen3-14b"]
        elif vram_gb >= 8:
            return self.MODEL_CONFIGS["qwen3-7b"]
        elif vram_gb >= 6:
            return self.MODEL_CONFIGS["mistral-7b"]
        else:
            return self.MODEL_CONFIGS["phi3-mini"]
    
    def _get_available_vram(self) -> float:
        """Get available GPU VRAM in GB."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            return total_memory / (1024**3)  # Convert to GB
        return 0.0
    
    def _get_available_ram(self) -> float:
        """Get available system RAM in GB."""
        return psutil.virtual_memory().available / (1024**3)

class OptimizedLocalLLM:
    """Optimized local LLM with quantization and caching."""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self._response_cache = {}
        self.max_cache_size = 100
    
    def load_model(self):
        """Load model with optimal quantization settings."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        # Quantization configuration
        if self.config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if self._supports_flash_attention() else None
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_cache: bool = True
    ) -> str:
        """Generate response with caching and optimization."""
        
        # Check cache first
        cache_key = hash(f"{prompt}_{max_tokens}_{temperature}")
        if use_cache and cache_key in self._response_cache:
            return self._response_cache[cache_key]
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_length - max_tokens,
            padding=True
        ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Cache response
        if use_cache:
            self._cache_response(cache_key, response)
        
        return response
    
    def _cache_response(self, key: int, response: str):
        """Cache response with LRU eviction."""
        if len(self._response_cache) >= self.max_cache_size:
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[key] = response
    
    def _supports_flash_attention(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

# Function calling implementation for Qwen3
class QwenFunctionCaller:
    """Function calling interface for Qwen3 models."""
    
    def __init__(self, llm: OptimizedLocalLLM):
        self.llm = llm
    
    def call_function(
        self,
        query: str,
        available_functions: Dict[str, Dict],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Execute function calling workflow."""
        
        for iteration in range(max_iterations):
            # Generate function call decision
            function_prompt = self._create_function_prompt(query, available_functions)
            response = self.llm.generate_response(function_prompt, temperature=0.1)
            
            # Parse function call
            function_call = self._parse_function_call(response)
            
            if function_call["action"] == "call_function":
                # Execute function
                result = self._execute_function(function_call, available_functions)
                
                # Generate final response
                final_prompt = self._create_final_prompt(query, function_call, result)
                final_response = self.llm.generate_response(final_prompt)
                
                return {
                    "response": final_response,
                    "function_called": function_call["function_name"],
                    "function_result": result
                }
            
            elif function_call["action"] == "direct_answer":
                return {
                    "response": function_call["answer"],
                    "function_called": None,
                    "function_result": None
                }
        
        # Fallback if no clear action
        return {
            "response": "I need more information to answer your question.",
            "function_called": None,
            "function_result": None
        }
    
    def _create_function_prompt(self, query: str, functions: Dict) -> str:
        """Create prompt for function calling decision."""
        function_descriptions = []
        for name, details in functions.items():
            function_descriptions.append(f"- {name}: {details['description']}")
        
        return f"""
        You are a helpful assistant that can call functions when needed.
        
        User query: {query}
        
        Available functions:
        {chr(10).join(function_descriptions)}
        
        Analyze the query and decide:
        1. If you can answer directly without functions, respond with: DIRECT_ANSWER: [your answer]
        2. If you need to call a function, respond with: CALL_FUNCTION: [function_name] ARGS: [arguments as JSON]
        
        Decision:
        """
    
    def _parse_function_call(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for function calls."""
        response = response.strip()
        
        if response.startswith("DIRECT_ANSWER:"):
            return {
                "action": "direct_answer",
                "answer": response[14:].strip()
            }
        elif response.startswith("CALL_FUNCTION:"):
            # Parse function name and arguments
            parts = response.split("ARGS:")
            function_name = parts[0][14:].strip()
            
            try:
                import json
                args = json.loads(parts[1].strip()) if len(parts) > 1 else {}
            except:
                args = {}
            
            return {
                "action": "call_function",
                "function_name": function_name,
                "arguments": args
            }
        
        return {"action": "unclear", "raw_response": response}
```

### Integration with LlamaIndex

```python
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core import Settings

class LlamaIndexLocalLLM(CustomLLM):
    """LlamaIndex integration for optimized local LLM."""
    
    def __init__(self, local_llm: OptimizedLocalLLM):
        self.local_llm = local_llm
        super().__init__()
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.local_llm.config.context_length,
            num_output=512,
            model_name=self.local_llm.config.name,
            is_chat_model=True,
            is_function_calling_model="function_calling" in self.local_llm.config.capabilities
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = self.local_llm.generate_response(
            prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        return CompletionResponse(text=response)
    
    def stream_complete(self, prompt: str, **kwargs):
        # Implement streaming if needed
        response = self.complete(prompt, **kwargs)
        yield response

# Setup function
def setup_local_llm():
    """Initialize and configure local LLM for the application."""
    selector = HardwareAdaptiveModelSelector()
    model_config = selector.select_model()
    
    print(f"Selected model: {model_config.name} ({model_config.parameters})")
    print(f"Memory requirement: {model_config.memory_gb}GB")
    
    # Initialize optimized LLM
    local_llm = OptimizedLocalLLM(model_config)
    local_llm.load_model()
    
    # Create LlamaIndex interface
    llamaindex_llm = LlamaIndexLocalLLM(local_llm)
    
    # Set global configuration
    Settings.llm = llamaindex_llm
    
    return local_llm, llamaindex_llm
```

## Consequences

### Positive Outcomes

- **Local Privacy**: All processing occurs locally without external API calls
- **Cost Effective**: No ongoing API costs after initial setup
- **Low Latency**: Local inference faster than API calls for most use cases
- **Hardware Adaptive**: Automatically selects optimal model for available resources
- **Function Calling**: Supports agentic RAG patterns with tool use capabilities
- **Quantization Benefits**: 50-70% memory reduction with minimal quality loss

### Negative Consequences / Trade-offs

- **Hardware Requirements**: Requires GPU with ≥6GB VRAM for good performance
- **Setup Complexity**: Model downloading and quantization setup more complex than APIs
- **Quality Variability**: Local models may have quality gaps vs latest cloud models
- **Resource Usage**: Significant VRAM and compute requirements during inference
- **Model Updates**: Manual process to update to newer model versions

### Performance Targets

- **Response Time**: <3 seconds for 512 token responses on RTX 4060
- **Memory Usage**: <12GB VRAM for largest supported model (Qwen3-14B)
- **Quality**: ≥85% performance vs GPT-3.5-turbo on reasoning benchmarks
- **Function Calling**: ≥90% success rate on simple function calling tasks
- **Context Handling**: Support 8K tokens efficiently, 32K maximum

## Dependencies

- **Python**: `transformers>=4.40.0`, `torch>=2.0.0`, `bitsandbytes>=0.41.0`
- **Hardware**: NVIDIA GPU with ≥6GB VRAM, CUDA 11.8+
- **Optional**: `flash-attn>=2.0.0` for attention optimization
- **Storage**: 10-20GB for model weights and cache

## Monitoring Metrics

- Model selection frequency by hardware configuration
- Response generation latency across different model sizes
- Memory utilization during inference
- Function calling success rates and accuracy
- Cache hit rates and effectiveness
- Quality metrics vs baseline cloud models

## Changelog

- **1.0 (2025-01-16)**: Initial local LLM strategy with Qwen3-14B primary and hardware-adaptive selection
