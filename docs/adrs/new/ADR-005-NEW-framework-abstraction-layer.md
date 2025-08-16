# ADR-005-NEW: Framework Abstraction Layer

## Title

Lightweight Framework Abstraction to Reduce Vendor Lock-in

## Version/Date

1.0 / 2025-01-16

## Status

Proposed

## Description

Implements a lightweight abstraction layer over LlamaIndex to reduce vendor lock-in while maintaining simplicity and performance. The abstraction provides standard interfaces for core RAG components (LLM, embeddings, retrieval, indexing) without the complexity of supporting multiple frameworks simultaneously. This enables framework switching without complete rewrites while avoiding over-engineering.

## Context

Current architecture is tightly coupled to LlamaIndex, creating vendor lock-in risks:

1. **Breaking Changes**: LlamaIndex API changes require extensive refactoring
2. **Framework Evolution**: Inability to adopt newer/better frameworks easily
3. **Component Flexibility**: Difficulty integrating non-LlamaIndex components
4. **Testing Complexity**: Hard to mock/test individual components

However, research shows many "framework abstraction" attempts fail due to over-engineering and complexity. Our approach focuses on minimal, focused abstractions that provide flexibility without sacrificing simplicity or performance.

## Related Requirements

### Functional Requirements

- **FR-1:** Abstract core RAG components (LLM, embeddings, retrieval, indexing)
- **FR-2:** Maintain full compatibility with existing LlamaIndex-based code
- **FR-3:** Enable component-level testing and mocking
- **FR-4:** Support gradual migration between framework versions

### Non-Functional Requirements

- **NFR-1:** **(Simplicity)** Abstraction overhead <5% performance impact
- **NFR-2:** **(Maintainability)** Reduce framework coupling by >60%
- **NFR-3:** **(Testability)** Enable unit testing of business logic without framework
- **NFR-4:** **(Future-Proofing)** Support framework migration with <30% code changes

## Alternatives

### 1. No Abstraction (Current)

- **Description**: Direct LlamaIndex usage throughout codebase
- **Issues**: High coupling, difficult testing, framework lock-in, update fragility
- **Score**: 3/10 (simplicity: 8, flexibility: 1, maintainability: 1)

### 2. Heavy Multi-Framework Support

- **Description**: Support LlamaIndex + LangChain + custom implementations
- **Issues**: Over-engineered, complex maintenance, performance overhead
- **Score**: 4/10 (flexibility: 8, simplicity: 2, performance: 2)

### 3. Lightweight Single-Framework Abstraction (Selected)

- **Description**: Thin abstraction over LlamaIndex with standard interfaces
- **Benefits**: Reduced coupling, testable, migration-ready, minimal overhead
- **Score**: 8/10 (flexibility: 7, simplicity: 8, maintainability: 9)

## Decision

We will implement a **lightweight framework abstraction layer** with these principles:

1. **Minimal Interface**: Abstract only core components, not every framework feature
2. **LlamaIndex Primary**: Current implementation remains LlamaIndex-based
3. **Focused Abstractions**: Target high-value abstractions (LLM, embeddings, retrieval)
4. **Gradual Adoption**: Introduce abstractions incrementally without breaking changes
5. **Test-Friendly**: Enable mocking and unit testing of business logic

## Related Decisions

- **ADR-001-NEW** (Modern Agentic RAG): Uses abstracted components for agent operations
- **ADR-002-NEW** (Unified Embedding Strategy): Implements through abstraction layer
- **ADR-003-NEW** (Adaptive Retrieval Pipeline): Benefits from testable retrieval interfaces
- **ADR-012-NEW** (Evaluation and Quality Assurance): Enables component-level testing

## Design

### Core Abstraction Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class Document:
    """Universal document representation."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None

@dataclass
class QueryResult:
    """Universal query result representation."""
    documents: List[Document]
    scores: List[float]
    query: str
    metadata: Dict[str, Any]

class LLMInterface(ABC):
    """Abstract interface for Language Models."""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for given prompt."""
        pass
    
    @abstractmethod
    def complete_with_functions(
        self, 
        prompt: str, 
        functions: List[Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion with function calling capability."""
        pass
    
    @abstractmethod
    def get_context_length(self) -> int:
        """Return maximum context length."""
        pass

class EmbeddingInterface(ABC):
    """Abstract interface for embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        pass

class RetrieverInterface(ABC):
    """Abstract interface for document retrieval."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> QueryResult:
        """Retrieve relevant documents for query."""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to retrieval index."""
        pass

class IndexInterface(ABC):
    """Abstract interface for document indexing."""
    
    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """Build searchable index from documents."""
        pass
    
    @abstractmethod
    def update_index(self, documents: List[Document]) -> None:
        """Update index with new documents."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        pass

class RerankerInterface(ABC):
    """Abstract interface for result reranking."""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents by relevance to query."""
        pass
```

### LlamaIndex Implementation

```python
from llama_index.core import Settings, VectorStoreIndex, Document as LlamaDocument
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
import logging

logger = logging.getLogger(__name__)

class LlamaIndexLLM(LLMInterface):
    """LlamaIndex implementation of LLM interface."""
    
    def __init__(self, llm=None):
        self.llm = llm or Settings.llm
    
    def complete(self, prompt: str, **kwargs) -> str:
        response = self.llm.complete(prompt, **kwargs)
        return response.text
    
    def complete_with_functions(
        self, 
        prompt: str, 
        functions: List[Dict], 
        **kwargs
    ) -> Dict[str, Any]:
        # Implementation depends on specific LLM function calling support
        # This would integrate with our QwenFunctionCaller or similar
        from .function_calling import QwenFunctionCaller
        
        function_caller = QwenFunctionCaller(self.llm)
        return function_caller.call_function(prompt, functions, **kwargs)
    
    def get_context_length(self) -> int:
        return getattr(self.llm.metadata, 'context_window', 8192)

class LlamaIndexEmbedding(EmbeddingInterface):
    """LlamaIndex implementation of embedding interface."""
    
    def __init__(self, embed_model=None):
        self.embed_model = embed_model or Settings.embed_model
    
    def embed_text(self, text: str) -> np.ndarray:
        return np.array(self.embed_model.get_text_embedding(text))
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        embeddings = self.embed_model.get_text_embedding_batch(texts)
        return [np.array(emb) for emb in embeddings]
    
    def get_dimension(self) -> int:
        # Get dimension from model or test with sample text
        test_embedding = self.embed_text("test")
        return test_embedding.shape[0]

class LlamaIndexRetriever(RetrieverInterface):
    """LlamaIndex implementation of retriever interface."""
    
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.retriever = VectorIndexRetriever(index=index)
    
    def retrieve(self, query: str, top_k: int = 10) -> QueryResult:
        # Update retriever settings
        self.retriever.similarity_top_k = top_k
        
        # Perform retrieval
        nodes = self.retriever.retrieve(query)
        
        # Convert to universal format
        documents = []
        scores = []
        
        for node in nodes:
            doc = Document(
                content=node.text,
                metadata=node.metadata,
                doc_id=node.node_id
            )
            documents.append(doc)
            scores.append(getattr(node, 'score', 1.0))
        
        return QueryResult(
            documents=documents,
            scores=scores,
            query=query,
            metadata={'retriever_type': 'vector', 'total_nodes': len(nodes)}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        # Convert to LlamaIndex format
        llama_docs = []
        for doc in documents:
            llama_doc = LlamaDocument(
                text=doc.content,
                metadata=doc.metadata,
                doc_id=doc.doc_id
            )
            llama_docs.append(llama_doc)
        
        # Add to index
        self.index.insert(llama_docs)

class LlamaIndexReranker(RerankerInterface):
    """LlamaIndex implementation of reranker interface."""
    
    def __init__(self, reranker=None):
        self.reranker = reranker or Settings.reranker or SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3", top_n=10
        )
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        # Convert to LlamaIndex nodes
        from llama_index.core import QueryBundle
        from llama_index.core.schema import NodeWithScore
        
        nodes = []
        for doc in documents:
            node = NodeWithScore(
                node=LlamaDocument(text=doc.content, metadata=doc.metadata),
                score=1.0
            )
            nodes.append(node)
        
        # Rerank
        query_bundle = QueryBundle(query_str=query)
        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        
        # Convert back to universal format
        reranked_docs = []
        for node in reranked_nodes:
            doc = Document(
                content=node.node.text,
                metadata=node.node.metadata,
                doc_id=getattr(node.node, 'node_id', None)
            )
            reranked_docs.append(doc)
        
        return reranked_docs
```

### Component Factory and Configuration

```python
from typing import Type, Dict, Callable
from enum import Enum

class FrameworkType(Enum):
    LLAMAINDEX = "llamaindex"
    LANGCHAIN = "langchain"  # Future support
    CUSTOM = "custom"

class ComponentFactory:
    """Factory for creating framework-specific component implementations."""
    
    def __init__(self, framework: FrameworkType = FrameworkType.LLAMAINDEX):
        self.framework = framework
        self._llm_implementations = {
            FrameworkType.LLAMAINDEX: LlamaIndexLLM
        }
        self._embedding_implementations = {
            FrameworkType.LLAMAINDEX: LlamaIndexEmbedding
        }
        self._retriever_implementations = {
            FrameworkType.LLAMAINDEX: LlamaIndexRetriever
        }
        self._reranker_implementations = {
            FrameworkType.LLAMAINDEX: LlamaIndexReranker
        }
    
    def create_llm(self, **kwargs) -> LLMInterface:
        """Create LLM implementation for current framework."""
        impl_class = self._llm_implementations[self.framework]
        return impl_class(**kwargs)
    
    def create_embedding(self, **kwargs) -> EmbeddingInterface:
        """Create embedding implementation for current framework."""
        impl_class = self._embedding_implementations[self.framework]
        return impl_class(**kwargs)
    
    def create_retriever(self, **kwargs) -> RetrieverInterface:
        """Create retriever implementation for current framework."""
        impl_class = self._retriever_implementations[self.framework]
        return impl_class(**kwargs)
    
    def create_reranker(self, **kwargs) -> RerankerInterface:
        """Create reranker implementation for current framework."""
        impl_class = self._reranker_implementations[self.framework]
        return impl_class(**kwargs)

# Global factory instance
component_factory = ComponentFactory()

def get_llm() -> LLMInterface:
    """Get configured LLM instance."""
    return component_factory.create_llm()

def get_embedding() -> EmbeddingInterface:
    """Get configured embedding instance."""
    return component_factory.create_embedding()

def get_retriever(index) -> RetrieverInterface:
    """Get configured retriever instance."""
    return component_factory.create_retriever(index=index)

def get_reranker() -> RerankerInterface:
    """Get configured reranker instance."""
    return component_factory.create_reranker()
```

### Business Logic Layer

```python
class RAGOrchestrator:
    """Framework-agnostic RAG orchestration logic."""
    
    def __init__(
        self,
        llm: LLMInterface,
        retriever: RetrieverInterface,
        reranker: Optional[RerankerInterface] = None
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
    
    def query(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """Execute complete RAG query pipeline."""
        
        # Step 1: Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(question, top_k=top_k)
        
        # Step 2: Rerank if available
        if self.reranker:
            reranked_docs = self.reranker.rerank(question, retrieval_result.documents)
            retrieval_result.documents = reranked_docs
        
        # Step 3: Generate response
        context = self._format_context(retrieval_result.documents)
        prompt = self._create_prompt(question, context)
        
        response = self.llm.complete(prompt, max_tokens=512, temperature=0.7)
        
        return {
            'answer': response,
            'sources': retrieval_result.documents,
            'retrieval_metadata': retrieval_result.metadata
        }
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Limit context
            context_parts.append(f"[{i+1}] {doc.content}")
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for response generation."""
        return f"""
        Context information:
        {context}
        
        Question: {question}
        
        Based on the context information provided, please answer the question. 
        If the context doesn't contain relevant information, say so.
        
        Answer:
        """

# Usage with abstraction
def create_rag_system():
    """Create RAG system using abstracted components."""
    
    # Get components through abstraction
    llm = get_llm()
    embedding = get_embedding()
    
    # Create index (framework-specific for now)
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex([])  # Initialize empty
    
    retriever = get_retriever(index)
    reranker = get_reranker()
    
    # Create framework-agnostic orchestrator
    rag_system = RAGOrchestrator(
        llm=llm,
        retriever=retriever,
        reranker=reranker
    )
    
    return rag_system
```

### Testing Support

```python
import unittest.mock as mock

class MockLLM(LLMInterface):
    """Mock LLM for testing."""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_history = []
    
    def complete(self, prompt: str, **kwargs) -> str:
        self.call_history.append(('complete', prompt, kwargs))
        return self.responses.get(prompt, "Mock response")
    
    def complete_with_functions(self, prompt: str, functions: List[Dict], **kwargs) -> Dict[str, Any]:
        self.call_history.append(('function_call', prompt, functions, kwargs))
        return {"response": "Mock function response", "function_called": None}
    
    def get_context_length(self) -> int:
        return 8192

class MockRetriever(RetrieverInterface):
    """Mock retriever for testing."""
    
    def __init__(self, mock_documents: List[Document] = None):
        self.mock_documents = mock_documents or []
        self.call_history = []
    
    def retrieve(self, query: str, top_k: int = 10) -> QueryResult:
        self.call_history.append(('retrieve', query, top_k))
        
        return QueryResult(
            documents=self.mock_documents[:top_k],
            scores=[0.9] * len(self.mock_documents[:top_k]),
            query=query,
            metadata={'mock': True}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        self.call_history.append(('add_documents', len(documents)))
        self.mock_documents.extend(documents)

# Test example
def test_rag_orchestrator():
    """Test RAG orchestrator with mocked components."""
    
    # Create mock components
    mock_llm = MockLLM(responses={
        "Context information:\n[1] Test content\n\nQuestion: test question\n\nBased on the context information provided, please answer the question. \nIf the context doesn't contain relevant information, say so.\n\nAnswer:": "Test answer"
    })
    
    mock_docs = [Document(content="Test content", metadata={})]
    mock_retriever = MockRetriever(mock_documents=mock_docs)
    
    # Create orchestrator
    orchestrator = RAGOrchestrator(
        llm=mock_llm,
        retriever=mock_retriever
    )
    
    # Test query
    result = orchestrator.query("test question")
    
    # Verify behavior
    assert result['answer'] == "Test answer"
    assert len(result['sources']) == 1
    assert mock_retriever.call_history[0][0] == 'retrieve'
    assert mock_llm.call_history[0][0] == 'complete'
```

## Consequences

### Positive Outcomes

- **Reduced Coupling**: 60% reduction in direct framework dependencies
- **Testability**: Business logic can be unit tested with mocked components
- **Migration Readiness**: Framework changes require minimal code modifications
- **Component Flexibility**: Easy to swap individual components (LLM, embeddings, etc.)
- **Maintainability**: Cleaner separation between business logic and framework code

### Negative Consequences / Trade-offs

- **Initial Complexity**: Additional abstraction layer requires upfront design
- **Performance Overhead**: Minimal (~2-5%) but measurable abstraction cost
- **Maintenance Burden**: Abstractions need updates when frameworks evolve
- **Feature Lag**: New framework features require abstraction updates
- **Learning Curve**: Team needs to understand abstraction patterns

### Migration Strategy

1. **Incremental Adoption**: Introduce abstractions one component at a time
2. **Backward Compatibility**: Maintain existing LlamaIndex code during transition
3. **Testing First**: Use abstractions in tests before production code
4. **Documentation**: Clear examples of abstraction usage patterns

## Dependencies

- **Python**: Standard library for ABC and typing
- **Current**: LlamaIndex implementations remain primary
- **Future**: Abstractions enable additional framework support

## Performance Targets

- **Abstraction Overhead**: <5% latency increase vs direct framework usage
- **Memory Overhead**: <50MB additional memory for abstraction layer
- **Migration Effort**: <30% code changes when switching frameworks
- **Test Coverage**: >90% business logic testable without framework dependencies

## Monitoring Metrics

- Abstraction layer performance overhead
- Component interface usage patterns
- Framework coupling metrics (dependency analysis)
- Test coverage improvements
- Migration effort measurements

## Changelog

- **1.0 (2025-01-16)**: Initial framework abstraction design with LlamaIndex implementations and testing support
