# ADR-005-DELETED: Framework Abstraction Layer

## Title

DELETED - Use LlamaIndex Directly Without Abstraction

## Version/Date

2.0 / 2025-08-17

## Status

DELETED (Unnecessary Abstraction)

## Decision

**DELETE THIS ADR** - We will use LlamaIndex directly without any abstraction layer.

## Rationale for Deletion

The original ADR-005 proposed creating abstraction interfaces for:

- LLMInterface
- EmbeddingInterface  
- RetrieverInterface
- IndexInterface
- RerankerInterface

This is classic over-engineering and violates YAGNI (You Aren't Gonna Need It):

1. **LlamaIndex is Stable**: Major API changes are rare and well-documented
2. **No Current Need**: We're not switching frameworks
3. **Added Complexity**: Abstractions make code harder to understand
4. **Maintenance Burden**: Another layer to maintain and test
5. **Performance Overhead**: Even 5% overhead is unnecessary

## What to Do Instead

Use LlamaIndex directly and clearly:

```python
# GOOD: Direct, clear, simple
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure directly
Settings.llm = Ollama(model="qwen2.5:14b", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# Use directly
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
```

NOT this:

```python
# BAD: Unnecessary abstraction
from abstractions import LLMInterface, get_llm, component_factory

llm = component_factory.create_llm()  # Why?
response = llm.complete(prompt)  # Just hiding LlamaIndex
```

## If We Ever Need to Switch Frameworks

**IF** (big if) we ever need to switch from LlamaIndex:

1. Use search and replace (it's that simple)
2. LlamaIndex → NewFramework takes <1 day of work
3. Creating abstractions now wastes more time than future migration

## The Right Time for Abstractions

Only create abstractions when:

1. You're ACTIVELY using multiple implementations
2. You've ALREADY switched frameworks once
3. There's a PROVEN need, not hypothetical

## What This Means for the Codebase

- Delete all abstraction interfaces
- Delete ComponentFactory
- Delete all Interface classes
- Use LlamaIndex types directly
- Use LlamaIndex methods directly

## Benefits of Direct Usage

- **Clarity**: Code shows exactly what's happening
- **Searchability**: Can search LlamaIndex docs directly
- **Debugging**: Stack traces point to real code
- **Performance**: No abstraction overhead
- **Simplicity**: One less concept to understand

## Example: Direct LlamaIndex Usage

```python
# services/rag_service.py
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
import qdrant_client

class RAGService:
    def __init__(self):
        # Direct Qdrant setup
        self.client = qdrant_client.QdrantClient(path="./data/qdrant")
        self.vector_store = QdrantVectorStore(client=self.client, collection_name="documents")
        
        # Direct index creation
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        
        # Direct reranker setup
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=5
        )
    
    def query(self, question: str):
        # Direct query engine usage
        query_engine = self.index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[self.reranker]
        )
        
        response = query_engine.query(question)
        return response.response
```

## Changelog

- **2.0 (2025-08-17)**: DELETED - Unnecessary abstraction, use LlamaIndex directly
- **1.0 (2025-01-16)**: Original abstraction layer proposal
