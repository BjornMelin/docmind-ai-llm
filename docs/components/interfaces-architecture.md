# DocMind AI Interfaces Architecture

## Overview

DocMind AI implements a clean interfaces architecture following dependency inversion principles. The `src/interfaces/` directory contains abstract base classes that define contracts for core system components, enabling clean dependency injection, testability, and future extensibility.

## Design Philosophy

The interfaces architecture supports:

1. **Dependency Inversion**: High-level modules depend on abstractions, not concretions
2. **Clean Testing**: Mock implementations for unit testing
3. **Extensibility**: New implementations without breaking existing code
4. **Type Safety**: Clear contracts with type hints and docstrings

## Directory Structure

```
src/interfaces/
├── __init__.py          # Interface exports
└── cache.py            # Cache interface definitions
```

## Cache Interface

**Location**: `src/interfaces/cache.py`

### CacheInterface Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Any

class CacheInterface(ABC):
    """Abstract interface for cache implementations."""
    
    @abstractmethod
    async def get_document(self, path: str) -> Any | None:
        """Get cached document processing result.
        
        Args:
            path: Document path to retrieve
            
        Returns:
            Cached result or None if not found
        """
    
    @abstractmethod
    async def store_document(self, path: str, result: Any) -> bool:
        """Store document processing result.
        
        Args:
            path: Document path to store
            result: Processing result to cache
            
        Returns:
            True if stored successfully
        """
    
    @abstractmethod
    async def clear_cache(self) -> bool:
        """Clear all cached documents.
        
        Returns:
            True if cleared successfully
        """
    
    @abstractmethod
    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache metrics
        """
```

### Benefits of Interface Design

1. **Clean Contracts**: Every method has explicit type hints and documentation
2. **Async Support**: Native async/await patterns throughout
3. **Error Boundaries**: Clear return types for error handling
4. **Testability**: Easy to create mock implementations

## Implementation Examples

### Production Implementation

```python
# src/cache/simple_cache.py
from src.interfaces import CacheInterface

class SimpleCache(CacheInterface):
    """SQLite-based cache implementation."""
    
    async def get_document(self, path: str) -> Any | None:
        # Production SQLite implementation
        ...
    
    async def store_document(self, path: str, result: Any) -> bool:
        # Production storage implementation
        ...
```

### Test Implementation

```python
# tests/mocks/mock_cache.py
from src.interfaces import CacheInterface

class MockCache(CacheInterface):
    """In-memory cache for testing."""
    
    def __init__(self):
        self._cache = {}
        self._stats = {"hits": 0, "misses": 0}
    
    async def get_document(self, path: str) -> Any | None:
        if path in self._cache:
            self._stats["hits"] += 1
            return self._cache[path]
        self._stats["misses"] += 1
        return None
    
    async def store_document(self, path: str, result: Any) -> bool:
        self._cache[path] = result
        return True
```

## Usage Patterns

### Dependency Injection

```python
from src.interfaces import CacheInterface
from src.cache.simple_cache import SimpleCache

class DocumentProcessor:
    """Document processor with injected cache dependency."""
    
    def __init__(self, cache: CacheInterface = None):
        self.cache = cache or SimpleCache()
    
    async def process(self, document_path: str):
        # Check cache first
        cached = await self.cache.get_document(document_path)
        if cached:
            return cached
        
        # Process and cache
        result = await self._process_document(document_path)
        await self.cache.store_document(document_path, result)
        return result
```

### Testing with Mocks

```python
import pytest
from src.components.document_processor import DocumentProcessor
from tests.mocks.mock_cache import MockCache

@pytest.mark.asyncio
async def test_document_processor_with_cache():
    """Test document processor uses cache correctly."""
    mock_cache = MockCache()
    processor = DocumentProcessor(cache=mock_cache)
    
    # First call - cache miss
    result1 = await processor.process("test.pdf")
    stats = await mock_cache.get_cache_stats()
    assert stats["misses"] == 1
    
    # Second call - cache hit
    result2 = await processor.process("test.pdf")
    stats = await mock_cache.get_cache_stats()
    assert stats["hits"] == 1
    assert result1 == result2
```

## Interface Evolution Strategy

### Current Interfaces

- ✅ **CacheInterface**: Document processing cache abstraction

### Planned Interfaces (Future)

- 🔄 **EmbeddingInterface**: Abstraction for BGE-M3, FastEmbed, etc.
- 🔄 **VectorStoreInterface**: Abstraction for Qdrant, Pinecone, etc.
- 🔄 **LLMInterface**: Abstraction for Ollama, vLLM, etc.
- 🔄 **DocumentProcessorInterface**: Abstraction for Unstructured, PyMuPDF, etc.

### Interface Design Guidelines

When adding new interfaces:

1. **Start with Usage**: Define interface based on how it's actually used
2. **Async First**: Use async/await patterns throughout
3. **Type Safety**: Comprehensive type hints with Union types for errors
4. **Clear Contracts**: Detailed docstrings with Args/Returns
5. **Error Handling**: Explicit error return types vs exceptions

## Architecture Benefits

### Development Benefits

```python
# Before interfaces - tightly coupled
class DocumentProcessor:
    def __init__(self):
        self.cache = SimpleCache()  # Hard dependency
        
    def process(self, path):
        # Direct coupling to SimpleCache API
        cached = self.cache.get(path)  # Breaks if API changes
```

```python
# After interfaces - loosely coupled
class DocumentProcessor:
    def __init__(self, cache: CacheInterface = None):
        self.cache = cache or SimpleCache()  # Injected dependency
        
    async def process(self, path):
        # Stable interface contract
        cached = await self.cache.get_document(path)  # Always works
```

### Testing Benefits

- **Fast Tests**: Mock implementations avoid slow I/O
- **Isolated Tests**: No external dependencies (SQLite, Redis, etc.)
- **Reliable Tests**: Deterministic behavior with controlled mocks
- **Coverage**: Easy to test error scenarios with mock failures

### Maintenance Benefits

- **API Stability**: Interfaces provide stable contracts
- **Implementation Flexibility**: Can swap implementations without breaking code
- **Refactoring Safety**: Interface compliance prevents breaking changes
- **Documentation**: Interfaces serve as living documentation

## Performance Considerations

### Interface Overhead

- **Runtime Cost**: Minimal - Python ABC overhead is negligible
- **Memory Usage**: No additional memory per interface instance
- **Type Checking**: Static analysis benefits with mypy/pylance

### Design Patterns

```python
# Efficient: Interface checking at creation time
def create_processor(cache_type: str = "simple") -> DocumentProcessor:
    if cache_type == "simple":
        cache = SimpleCache()
    elif cache_type == "redis":
        cache = RedisCache()
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
    
    return DocumentProcessor(cache=cache)

# Inefficient: Runtime type checking in hot path
def process_with_cache(path: str, cache: Any):
    if isinstance(cache, CacheInterface):  # Avoid this in hot paths
        return cache.get_document(path)
```

## Integration with Configuration

### Settings Integration

```python
# src/config/settings.py
class CacheSettings(BaseModel):
    cache_type: str = "simple"
    cache_dir: str = "./cache"
    size_limit: int = 1024 * 1024 * 1024  # 1GB

# Factory pattern with interfaces
def create_cache(settings: CacheSettings) -> CacheInterface:
    if settings.cache_type == "simple":
        return SimpleCache(cache_dir=settings.cache_dir)
    elif settings.cache_type == "redis":
        return RedisCache(url=settings.redis_url)
    else:
        raise ValueError(f"Unknown cache type: {settings.cache_type}")
```

## Future Interface Expansion

### Embedding Interface (Planned)

```python
# src/interfaces/embedding.py
class EmbeddingInterface(ABC):
    """Abstract interface for embedding models."""
    
    @abstractmethod
    async def embed_texts(
        self, 
        texts: list[str], 
        return_dense: bool = True,
        return_sparse: bool = False
    ) -> EmbeddingResult:
        """Generate embeddings for input texts."""
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
```

### Vector Store Interface (Planned)

```python
# src/interfaces/vector_store.py
class VectorStoreInterface(ABC):
    """Abstract interface for vector storage systems."""
    
    @abstractmethod
    async def add_documents(
        self, 
        documents: list[Document], 
        embeddings: list[list[float]]
    ) -> bool:
        """Add documents with embeddings to vector store."""
    
    @abstractmethod
    async def similarity_search(
        self, 
        query_embedding: list[float], 
        top_k: int = 10
    ) -> list[Document]:
        """Perform similarity search."""
```

## Conclusion

The interfaces architecture provides a solid foundation for:

1. **Clean Code**: Dependency inversion and clear contracts
2. **Easy Testing**: Mock implementations and isolated tests  
3. **Future Flexibility**: New implementations without breaking changes
4. **Type Safety**: Comprehensive type hints and static analysis support

This approach demonstrates the power of interface-driven design in creating maintainable, testable, and extensible systems while maintaining the KISS principle through simple, focused interfaces.