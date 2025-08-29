# DocMind AI - Implementation Guide

## Quick Start

### Installation & Setup

```bash
# Clone repository
git clone <repo>
cd docmind-ai-llm

# Install dependencies with uv
uv pip install -e .

# Download models
python scripts/download_models.py

# Run with Docker (recommended)
docker-compose up

# Or run locally
streamlit run app.py
```

### Environment Configuration

```env
# .env file - minimal configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
MODEL_CACHE_DIR=./models
LMDEPLOY_HOST=http://localhost:23333  # LMDeploy for INT8 KV cache (recommended)

# Optional features (disabled by default)
ENABLE_GRAPHRAG=false
ENABLE_DSPY=false
ENABLE_SEMANTIC_CACHE=true
```

## Core Implementation Patterns

### 1. Library-First Approach

**Always use existing libraries over custom code**:

```python
# ❌ DON'T: Custom wrapper
class MyEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-m3')
    def embed(self, text):
        return self.model.encode(text)

# ✅ DO: Direct usage
model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode(texts)
```

### 2. LlamaIndex Integration

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Configure globally - no custom abstractions
Settings.embed_model = embed_model
Settings.llm = llm_model

# Use RouterQueryEngine for automatic routing
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, keyword_tool]
)

# Direct usage, no wrappers
response = query_engine.query("What is RAG?")
```

### 3. BGE-M3 Unified Embeddings

```python
from FlagEmbedding import BGEM3FlagModel

# BGE-M3 for unified dense+sparse embeddings
embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
results = embed_model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    max_length=8192
)

# Dense vectors for similarity search
dense_vectors = results['dense_vecs']

# Sparse vectors for keyword matching
sparse_vectors = results['lexical_weights']
```

### 4. Structured Outputs with Instructor

```python
from instructor import patch
from pydantic import BaseModel
import ollama

# Patch Ollama client for structured outputs
structured_llm = patch(ollama.Client())

class QueryAnalysis(BaseModel):
    """Query analysis response structure for LLM-powered query parsing.
    
    Provides structured output for analyzing user queries to determine
    intent, complexity level, and extract relevant entities.
    """
    intent: str
    complexity: str
    entities: list[str]

# Guaranteed structured response
analysis = structured_llm.create(
    model="Qwen3-4B-Instruct-2507-AWQ",
    response_model=QueryAnalysis,
    messages=[{"role": "user", "content": query}]
)
```

### 5. 5-Agent System with LangGraph

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Create 5 specialized agents
agents = [
    routing_agent,    # Query analysis and routing
    planning_agent,   # Complex query decomposition
    retrieval_agent,  # Document retrieval with DSPy
    synthesis_agent,  # Multi-source combination
    validation_agent  # Response validation
]

workflow = create_supervisor(agents, llm_model, system_prompt)
app = workflow.compile()

# Execute with built-in error handling
result = app.invoke({"messages": [user_message]})
```

### 6. Streamlit with Streaming

```python
import streamlit as st

# Native streaming support
def stream_response(query):
    for token in llm.stream(query):
        yield token

# Use st.write_stream for real-time display
with st.chat_message("assistant"):
    response = st.write_stream(stream_response(query))

# Session memory integration
@st.cache_resource
def get_memory_store():
    from langgraph.store import InMemoryStore
    return InMemoryStore()
```

## Advanced Features

### DSPy Query Optimization (Optional)

```python
import dspy

# Configure with local model (262K context)
dspy.settings.configure(lm=dspy.LM("lmdeploy/Qwen3-4B-Instruct-2507-AWQ", max_tokens=262144))

class QueryOptimizer(dspy.Module):
    def __init__(self):
        self.expand = dspy.ChainOfThought("query -> expanded_queries")
    
    def forward(self, query):
        return self.expand(query=query)

# Automatic optimization
optimizer = dspy.MIPROv2(metric=retrieval_quality)
optimized = optimizer.compile(QueryOptimizer(), trainset=examples)
```

### Semantic Caching with GPTCache

```python
from gptcache import Cache
from gptcache.manager import get_data_manager

cache = Cache()
cache.init(
    embedding_func=lambda x: embed_model.encode(x),
    data_manager=get_data_manager("sqlite", "qdrant"),
    similarity_evaluation=SearchDistanceEvaluation()
)

# Check cache before generation
if cached := cache.get(query):
    return cached
else:
    response = generate_response(query)
    cache.set(query, response)
    return response
```

### GraphRAG Integration (Optional)

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore

# Zero additional infrastructure
graph_store = SimplePropertyGraphStore()
property_graph_index = PropertyGraphIndex.from_documents(
    documents,
    property_graph_store=graph_store,
    embed_model=embed_model,
    llm=llm
)

# Hybrid retrieval: vector + graph
if enable_graphrag:
    results = property_graph_index.query(query)
else:
    results = vector_index.query(query)
```

## Document Processing

### Multi-Format Support

```python
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title

# Parse ANY document format with one function
elements = partition(filename="document.pdf")

# Intelligent chunking
chunks = chunk_by_title(
    elements,
    max_characters=1500,
    overlap=100
)
```

## Performance Optimization

### Model Loading & Caching

```python
# Load models once at startup
@st.cache_resource
def load_models():
    return {
        "embed": SentenceTransformer('BAAI/bge-m3'),
        "rerank": CrossEncoder('BAAI/bge-reranker-v2-m3'),
        "llm": Ollama(model="qwen3:14b", num_gpu=1)
    }
```

### Batch Processing

```python
from more_itertools import chunked

# Process documents in batches
for batch in chunked(documents, 32):
    embeddings = model.encode(batch, batch_size=32)
    client.upsert(collection="docs", points=embeddings)
```

### Hardware Adaptation

```python
def select_model_size(available_vram_gb):
    if available_vram_gb >= 14:
        return "qwen3:14b"
    elif available_vram_gb >= 8:
        return "qwen3:7b"
    else:
        return "qwen3:4b"

def configure_quantization(vram_gb):
    if vram_gb < 12:
        return {"quantization": "4bit", "load_in_4bit": True}
    else:
        return {"quantization": "8bit", "load_in_8bit": True}
```

### Multi-Provider LLM Support

DocMind AI supports multiple local LLM providers with automatic hardware-based selection:

| Provider | Tokens/sec | VRAM Usage | Setup | Best For |
|----------|------------|------------|-------|----------|
| **LMDeploy** | 40-60 (+30% INT8) | 12.2GB | Simple | RECOMMENDED: INT8 KV cache, 262K context |
| **vLLM** | 40-60 (+30% FP8) | 12.2GB | Moderate | FP8 KV cache alternative, 262K context |
| **llama.cpp** | 30-45 | 12.2GB | Complex | GGUF fallback, INT8 KV cache |
| **Ollama** | 25-40 | 12.2GB | Simplest | Easy setup, INT8 KV cache support |

```python
# Automatic provider selection
def select_provider(hardware):
    if hardware["gpu_memory_gb"] >= 16 and hardware["supports_int8_kv"]:
        return "lmdeploy"  # RECOMMENDED: INT8 KV cache, 262K context
    elif hardware["gpu_memory_gb"] >= 16 and hardware["supports_fp8_kv"]:
        return "vllm"  # Alternative: FP8 KV cache, 262K context
    elif has_gguf_model:
        return "llamacpp"  # GGUF fallback with INT8 support
    else:
        return "ollama"  # Default: easiest setup

# Environment configuration
export DOCMIND_LLM_PROVIDER=lmdeploy  # lmdeploy, vllm, llamacpp, ollama
export LLAMA_FLASH_ATTN=1             # Enable flash attention
export CUDA_VISIBLE_DEVICES=0,1       # For multi-GPU vLLM
```

## Implementation Roadmap

### Week 1: Foundation

#### **Days 1-2: Core Setup**

- BGE-M3 embedding pipeline (unified dense+sparse)
- Qdrant with hybrid search support
- Qwen3-4B-Instruct-2507 with Instructor for structured outputs (262K context)

#### **Days 3-4: Basic RAG**

- LlamaIndex query engine with router
- Streamlit UI with chat interface
- Document upload and processing

#### **Days 5-7: Agent System**

- Implement 5-agent system with langgraph-supervisor
- Basic routing and validation agents
- Memory integration with LangGraph

### Week 2: Enhancement

#### **Days 8-9: Streaming & Memory**

- Streamlit streaming with st.write_stream()
- Session memory with InMemoryStore
- Progress indicators for long operations

#### **Days 10-11: Caching & Performance**

- GPTCache semantic caching
- Model quantization and optimization
- Batch processing pipelines

#### **Days 12-14: Advanced Features**

- DSPy query optimization (feature flag)
- GraphRAG integration (optional)
- Multi-stage reranking pipeline

### Week 3: Production

#### **Days 15-16: Testing & Evaluation**

- DeepEval + Ragas metrics
- End-to-end testing with pytest
- Performance benchmarking

#### **Days 17-18: Deployment**

- Docker containerization
- Environment configuration
- Health monitoring

#### **Days 19-21: Polish & Documentation**

- User documentation
- Error handling improvements
- Final testing and validation

## Testing Strategy

### Unit Testing

```python
import pytest
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

def test_query_processing():
    metrics = [AnswerRelevancyMetric()]
    results = evaluate(test_cases, metrics)
    assert results.aggregate_score > 0.8
```

### Integration Testing

```python
def test_end_to_end_rag():
    # Upload document
    documents = load_test_documents()
    index = create_index(documents)
    
    # Query and validate
    response = index.query("Test question")
    assert response.confidence > 0.7
    assert len(response.source_nodes) > 0
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Use INT8 KV cache: `LMDEPLOY_QUANT_POLICY=8` |
| Slow embeddings | Enable GPU: `device='cuda'` |
| Docker fails | Check GPU passthrough: `--gpus all` |
| Models not loading | Verify with: `lmdeploy list` |
| Qdrant connection | Ensure port 6333 is open |
| Context too large | Enable 262K: `DOCMIND_CONTEXT_LENGTH=262144` |

### Performance Tuning

```python
# Memory optimization
import torch
torch.cuda.empty_cache()

# Model optimization
model = model.half()  # Use FP16
model = torch.jit.script(model)  # JIT compilation

# Batch size tuning
optimal_batch_size = find_optimal_batch_size(model, input_size)
```

## Minimal Working Example

Complete RAG application in <100 lines:

```python
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from unstructured.partition.auto import partition

# Initialize models (cached)
@st.cache_resource
def init_models():
    Settings.llm = Ollama(model="qwen3-4b-instruct-2507")  # 262K context
    Settings.embed_model = SentenceTransformer('BAAI/bge-m3')
    return CrossEncoder('BAAI/bge-reranker-v2-m3')

# Document processing
def process_docs(files):
    docs = []
    for file in files:
        elements = partition(file)
        docs.extend([e.text for e in elements])
    return docs

# Main app
st.title("DocMind AI - Local RAG")
reranker = init_models()

# File upload
if files := st.file_uploader("Upload", accept_multiple_files=True):
    docs = process_docs(files)
    index = VectorStoreIndex.from_documents(docs)
    st.success(f"Indexed {len(docs)} documents")

# Chat interface
if prompt := st.chat_input():
    if 'index' in locals():
        query_engine = index.as_query_engine(similarity_top_k=10)
        response = query_engine.query(prompt)
        st.write(response)
```

## Configuration Reference

### Model Configuration

```python
# models.yaml
models:
  llm:
    name: "qwen3-4b-instruct-2507"
    quantization: "awq"
    context_length: 262144
  
  embedding:
    name: "BAAI/bge-m3"
    dimension: 1024
    context_length: 8192
  
  reranker:
    name: "BAAI/bge-reranker-v2-m3"
    top_k: 10
```

### Feature Flags

```python
# features.yaml
features:
  graphrag:
    enabled: false
    model: "simple_property_graph"
  
  dspy:
    enabled: false
    optimizer: "MIPROv2"
  
  semantic_cache:
    enabled: true
    similarity_threshold: 0.1
```

## Production Considerations

### Security

- **Local-only operation**: No external API calls
- **Data isolation**: Documents isolated per session
- **Memory safety**: Automatic cleanup of sensitive data

### Monitoring

```python
# Simple logging for production
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Track key metrics
metrics = {
    "query_latency": response_time,
    "cache_hit_rate": hits / total_queries,
    "memory_usage": psutil.Process().memory_info().rss
}
```

### Scaling

- **Multi-GPU**: Use vLLM for 200-300% performance improvement
- **Distributed**: Scale with multiple Qdrant nodes
- **Load balancing**: Use nginx for multiple Streamlit instances

This implementation guide provides everything needed to build DocMind AI using a library-first approach with minimal custom code while achieving production-ready performance with 262K context capability.
