# Implementation Guide - DocMind AI

## Updated: 2025-08-17 - Post Expert Review

## Quick Start Setup

### Installation

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

```bash
# .env file
QDRANT_HOST=localhost
QDRANT_PORT=6333
MODEL_CACHE_DIR=./models
OLLAMA_HOST=http://localhost:11434
```

## Library Usage Guide

### 1. LlamaIndex - Direct RAG Implementation

```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
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

### 2. Sentence-Transformers - Embeddings & Reranking

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from FlagEmbedding import BGEM3FlagModel

# BGE-M3 for embeddings (dense + sparse)
embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
results = embed_model.encode(
    texts,
    return_dense=True,
    return_sparse=True,
    max_length=8192
)

# Direct reranking - no custom wrappers
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
scores = reranker.predict(query_doc_pairs)
```

### 3. LangGraph Supervisor - 5-Agent Orchestration

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

### 4. Unstructured.io - Document Processing

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

### 5. Critical New Libraries

#### Instructor - Structured Outputs

```python
from instructor import patch
from pydantic import BaseModel

# Patch LLM for structured outputs
structured_llm = patch(ollama_llm)

class QueryAnalysis(BaseModel):
    intent: str
    complexity: str
    entities: list[str]

# Guaranteed structured response
analysis = structured_llm.create(
    model="qwen3:14b",
    response_model=QueryAnalysis,
    messages=[{"role": "user", "content": query}]
)
```

#### DSPy - Query Optimization

```python
import dspy

# Configure with local model
dspy.settings.configure(lm=dspy.LM("ollama/qwen3:14b"))

class QueryOptimizer(dspy.Module):
    def __init__(self):
        self.expand = dspy.ChainOfThought("query -> expanded_queries")
    
    def forward(self, query):
        return self.expand(query=query)

# Automatic optimization
optimizer = dspy.MIPROv2(metric=retrieval_quality)
optimized = optimizer.compile(QueryOptimizer(), trainset=examples)
```

#### GPTCache - Semantic Caching

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

### 6. Streamlit with Streaming & Memory

```python
import streamlit as st
from langgraph.store import InMemoryStore

# Native streaming support
def stream_response(query):
    for token in llm.stream(query):
        yield token

# Use st.write_stream for real-time display
with st.chat_message("assistant"):
    response = st.write_stream(stream_response(query))

# LangGraph memory integration
@st.cache_resource
def get_memory_store():
    return InMemoryStore()  # Or RedisStore for persistence

# Session memory with LangGraph
from langchain.memory import ChatMemoryBuffer
memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
```

## Revised 3-Week Implementation Roadmap

### Week 1: Foundation + Critical Features

#### **Days 1-2: Core Setup**

- BGE-M3 embedding pipeline (unified dense+sparse)
- Qdrant with hybrid search support
- Qwen3-14B with Instructor for structured outputs

#### **Days 3-4: DSPy Integration**

- Query rewriting and optimization
- Bootstrap with initial examples
- Feature flag for experimental rollout

#### **Days 5-7: 5-Agent System**

- Implement all 5 agents with langgraph-supervisor
- Planning agent for query decomposition
- Synthesis agent for multi-source combination

### Week 2: Memory, Streaming, Caching

#### **Days 8-9: Memory Integration**

- LangGraph InMemoryStore setup
- LlamaIndex ChatMemoryBuffer
- Optional Redis backend preparation

#### **Days 10-11: Streaming Implementation**

- Streamlit st.write_stream() integration
- Async streaming pipeline
- Progress indicators for long operations

#### **Days 12-14: Semantic Caching**

- GPTCache integration
- Cache key strategy with doc IDs
- Invalidation on document updates

### Week 3: Optional Features + Polish

#### **Days 15-16: GraphRAG (Optional)**

- Microsoft GraphRAG as feature flag
- Basic entity/relationship extraction
- Hybrid routing logic

#### **Days 17-18: Evaluation & Testing**

- DeepEval + Ragas metrics
- End-to-end testing with pytest
- Performance benchmarking

#### **Days 19-21: Deployment & Documentation**

- Single Docker container
- Environment configuration
- User documentation

## Old 3-Week Implementation Roadmap (DEPRECATED)

### Week 1: Core Foundation

**Goal**: Basic RAG pipeline working end-to-end

#### Day 1-2: Setup & Models

```python
# Download and verify models
models = {
    "llm": "ollama pull qwen3:14b",
    "embed": SentenceTransformer('BAAI/bge-m3'),
    "rerank": CrossEncoder('BAAI/bge-reranker-v2-m3')
}
```

#### Day 3-4: Storage Setup

```python
# Qdrant for vectors
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)

# SQLModel for persistence
from sqlmodel import SQLModel, create_engine
engine = create_engine("sqlite:///docmind.db")
```

#### Day 5: Basic RAG Pipeline

```python
# Minimal working pipeline
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
```

### Week 2: Agent Integration & UI

**Goal**: Add agents and Streamlit interface

#### Day 6-7: Agent Setup

```python
# Three simple agents via langgraph
from langgraph_supervisor import create_supervisor

routing_agent = create_tool("route", route_query)
retrieval_agent = create_tool("retrieve", retrieve_docs)
validation_agent = create_tool("validate", validate_response)

app = create_supervisor([routing_agent, retrieval_agent, validation_agent])
```

#### Day 8-9: Streamlit UI

```python
# Basic chat interface
st.title("DocMind AI")

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input():
    response = app.invoke({"messages": [prompt]})
    st.chat_message("assistant").write(response)
```

#### Day 10: Document Upload

```python
# File upload with Unstructured
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

for file in uploaded_files:
    elements = partition(file)
    chunks = chunk_by_title(elements)
    index.insert(chunks)
```

### Week 3: Polish & Deploy

**Goal**: Production-ready deployment

#### Day 11-12: Evaluation Integration

```python
# Add DeepEval metrics
from deepeval import evaluate

metrics = [AnswerRelevancyMetric(), FaithfulnessMetric()]
results = evaluate(test_cases, metrics)
st.metric("Quality Score", results.aggregate_score)
```

#### Day 13-14: Docker Deployment

```dockerfile
# Simple single-stage Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["streamlit", "run", "app.py"]
```

#### Day 15: Final Testing & Launch

- Test with real documents
- Verify local-only operation
- Single command deployment: `docker-compose up`

## Code Examples

### Complete Minimal RAG Application (< 100 lines)

```python
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from unstructured.partition.auto import partition
from qdrant_client import QdrantClient

# Initialize models (cached)
@st.cache_resource
def init_models():
    Settings.llm = Ollama(model="qwen3:14b")
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
        # Retrieve
        query_engine = index.as_query_engine(similarity_top_k=10)
        candidates = query_engine.retrieve(prompt)
        
        # Rerank
        pairs = [[prompt, c.text] for c in candidates]
        scores = reranker.predict(pairs)
        best_docs = [c for c, s in sorted(zip(candidates, scores), 
                                         key=lambda x: x[1], reverse=True)[:3]]
        
        # Generate
        response = query_engine.synthesize(prompt, best_docs)
        st.write(response)
```

## Key Implementation Patterns

### Pattern 1: Direct Library Usage

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

### Pattern 2: Library Features Over Custom

```python
# ❌ DON'T: Custom routing
def route_query(query):
    if "technical" in query:
        return technical_engine
    else:
        return general_engine

# ✅ DO: LlamaIndex RouterQueryEngine
from llama_index.core.query_engine import RouterQueryEngine
router = RouterQueryEngine(selector=LLMSingleSelector())
```

### Pattern 3: Native Caching

```python
# ❌ DON'T: Custom cache
cache = {}
def get_embeddings(text):
    if text not in cache:
        cache[text] = embed(text)
    return cache[text]

# ✅ DO: Streamlit native
@st.cache_data
def get_embeddings(text):
    return model.encode(text)
```

## Performance Optimization

### Model Loading

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
# Process documents in batches
from more_itertools import chunked

for batch in chunked(documents, 32):
    embeddings = model.encode(batch, batch_size=32)
    client.upsert(collection="docs", points=embeddings)
```

### Async Operations

```python
# Use async for I/O operations
import asyncio
from llama_index.core import AsyncQueryEngine

async def process_queries(queries):
    engine = AsyncQueryEngine.from_defaults()
    tasks = [engine.aquery(q) for q in queries]
    return await asyncio.gather(*tasks)
```

## Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of memory | Use Qwen3-7B instead of 14B |
| Slow embeddings | Enable GPU: `device='cuda'` |
| Docker fails | Check GPU passthrough: `--gpus all` |
| Models not loading | Verify with: `ollama list` |
| Qdrant connection | Ensure port 6333 is open |

## Summary

This implementation guide provides everything needed to build DocMind AI in 3 weeks using a library-first approach:

- **92% less code** through library usage
- **100% local operation** with no API dependencies
- **Simple deployment** with single Docker command
- **Production ready** with evaluation and monitoring

Remember: If a library does it, use the library!
