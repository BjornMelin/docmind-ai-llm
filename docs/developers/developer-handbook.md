# DocMind AI Developer Handbook

## Overview

This comprehensive handbook provides practical implementation guidance for developers working with DocMind AI. It covers development standards, testing strategies, implementation patterns, code quality guidelines, and maintenance procedures to ensure consistent, high-quality contributions to the project.

> **Prerequisites**: Complete the [Getting Started Guide](getting-started.md) and understand the [System Architecture](system-architecture.md) before following this handbook.

## Table of Contents

1. [Development Standards](#development-standards)
2. [Implementation Patterns](#implementation-patterns)
3. [Testing Strategies](#testing-strategies)
4. [Code Quality & Maintenance](#code-quality--maintenance)
5. [Advanced Implementation Details](#advanced-implementation-details)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting & Debugging](#troubleshooting--debugging)

## Development Standards

### Core Principles

- **KISS (Keep It Simple, Stupid)**: Prefer simple, readable solutions over complex ones
- **DRY (Don't Repeat Yourself)**: Eliminate code duplication through functions and modules
- **YAGNI (You Aren't Gonna Need It)**: Don't implement features until they're actually needed
- **Library-First**: Prefer well-established libraries over custom implementations

### Python Code Style

#### Formatting & Linting

```bash
# Format and lint (run before commits)
ruff format . && ruff check . --fix

# Configuration in pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "UP", "N", "S", "B", "A", "C4", "PT", "SIM", "TID", "D"]
ignore = ["D203", "D213", "S301", "S603", "S607", "S108"]
```

#### Type Hints & Documentation

```python
# Good: Clear function with type hints and docstring
def process_documents(files: List[Path], chunk_size: int = 1000) -> List[Document]:
    """Process uploaded documents into chunks for analysis.
    
    Args:
        files: List of file paths to process
        chunk_size: Size of text chunks for processing
        
    Returns:
        List of processed document chunks
        
    Raises:
        DocumentProcessingError: If document parsing fails
    """
    documents = []
    for file in files:
        content = load_file_content(file)
        chunks = split_content(content, chunk_size)
        documents.extend(chunks)
    
    return documents
```

#### Modern Python Patterns

```python
# Use modern Python features
from pathlib import Path
from typing import List, Dict, Optional

# Walrus operator for efficiency
if result := expensive_call():
    process(result)

# F-strings for formatting
message = f"Processing {len(files)} files with chunk size {chunk_size}"

# Context managers for resources
async with aiohttp.ClientSession() as session:
    response = await session.get(url)
    
# List comprehensions when readable
processed = [process_item(item) for item in items if item.is_valid()]

# Type unions (Python 3.10+)
def get_config(key: str) -> str | None:
    return config.get(key)
```

### Project Structure Best Practices

```text
src/
├── app.py                    # Main application entry point
├── config/                   # Configuration management
│   └── settings.py           # Single source of truth for all config
├── agents/                   # Multi-agent system
│   ├── coordinator.py        # Main orchestration
│   └── tools.py             # Shared agent tools
├── utils/                    # Reusable utilities
│   ├── core.py              # Core utilities
│   ├── document.py          # Document processing
│   └── embedding.py         # Embedding operations
└── models/                   # Pydantic data models
    └── schemas.py           # API schemas
```

### Configuration Pattern (MANDATORY)

**Always use the unified configuration system:**

```python
# CORRECT: Single import pattern
from src.config import settings

# Access nested configuration
model_name = settings.vllm.model
embedding_model = settings.embedding.model_name
chunk_size = settings.processing.chunk_size
agent_timeout = settings.agents.decision_timeout

# INCORRECT: Direct environment variable access
import os
model = os.getenv("DOCMIND_LLM__MODEL")  # Don't do this
```

## Implementation Patterns

### Async-First Development

DocMind AI uses async/await throughout for optimal performance:

```python
# Agent coordination - async pattern
class MultiAgentCoordinator:
    async def arun(self, query: str) -> str:
        """Execute multi-agent coordination asynchronously."""
        
        # Route query
        routing = await self.router_agent.arun(query)
        
        # Plan if complex
        if routing.complexity > 0.7:
            plan = await self.planner_agent.arun(query, routing)
        else:
            plan = SimpleExecutionPlan(query)
        
        # Execute retrieval
        documents = await self.retrieval_agent.arun(query, plan)
        
        # Synthesize and validate in parallel
        synthesis_task = asyncio.create_task(
            self.synthesis_agent.arun(query, documents)
        )
        validation_task = asyncio.create_task(
            self.validator_agent.arun(query, documents)
        )
        
        synthesis, validation = await asyncio.gather(
            synthesis_task, validation_task
        )
        
        return self.finalize_response(synthesis, validation)
```

### Document Processing Pattern

```python
async def process_document_pipeline(
    file_path: Path, 
    settings: DocMindSettings
) -> ProcessedDocument:
    """Complete document processing pipeline."""
    
    # 1. Parse document with error handling
    try:
        raw_content = await parse_document_unstructured(file_path)
    except Exception as e:
        logger.error(f"Failed to parse {file_path}: {e}")
        raise DocumentProcessingError(f"Parse failed: {e}")
    
    # 2. NLP processing with spaCy
    nlp_doc = await process_with_spacy(raw_content)
    
    # 3. Intelligent chunking
    chunks = create_chunks(
        nlp_doc, 
        chunk_size=settings.processing.chunk_size,
        chunk_overlap=settings.processing.chunk_overlap
    )
    
    # 4. Generate embeddings
    embeddings = await generate_embeddings_batch(
        chunks, 
        settings.embedding
    )
    
    # 5. Store in vector database
    await store_in_qdrant(chunks, embeddings, settings.qdrant)
    
    return ProcessedDocument(
        file_path=file_path,
        chunks=len(chunks),
        embedding_dim=len(embeddings[0]) if embeddings else 0
    )
```

### Error Handling Patterns

```python
# Comprehensive error handling with fallbacks
class RetrievalAgent:
    async def retrieve_documents(
        self, 
        query: str, 
        strategy: str = "hybrid"
    ) -> List[Document]:
        """Retrieve documents with fallback strategies."""
        
        try:
            # Primary strategy
            if strategy == "hybrid":
                return await self._hybrid_search(query)
            elif strategy == "dense":
                return await self._dense_search(query)
            else:
                return await self._sparse_search(query)
                
        except QdrantConnectionError as e:
            logger.warning(f"Qdrant connection failed: {e}, using cache")
            return await self._fallback_cache_search(query)
            
        except EmbeddingGenerationError as e:
            logger.warning(f"Embedding generation failed: {e}, using keyword search")
            return await self._fallback_keyword_search(query)
            
        except Exception as e:
            logger.error(f"All retrieval strategies failed: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")
```

### Tool Creation Pattern

```python
# Create agent tools following LangGraph patterns
from langchain_core.tools import tool
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

@tool
def retrieve_documents(
    query: str,
    strategy: str,
    state: Annotated[dict, InjectedState]
) -> List[Document]:
    """Retrieve documents using specified strategy.
    
    Args:
        query: Search query
        strategy: Retrieval strategy (hybrid, dense, sparse, graph)
        state: Agent execution state
        
    Returns:
        List of relevant documents
    """
    retriever = get_retriever(strategy)
    
    # Execute retrieval with timeout
    with timeout_context(settings.agents.decision_timeout):
        results = retriever.retrieve(query, top_k=10)
    
    # Apply reranking if needed
    if len(results) > 5 and strategy in ["hybrid", "dense"]:
        results = rerank_documents(query, results)
    
    # Update agent state
    state["retrieved_count"] = len(results)
    state["retrieval_strategy"] = strategy
    
    return results[:5]  # Return top 5
```

## Testing Strategies

DocMind AI implements a comprehensive three-tier testing strategy:

### Testing Framework Setup

```bash
# Install test dependencies
uv sync --extra test

# Run tests by tier
pytest tests/unit/ -v                    # Tier 1: Fast unit tests (<5s each)
pytest tests/integration/ -v             # Tier 2: Cross-component tests (<30s each)
python scripts/run_tests.py --system     # Tier 3: Full system tests (<5min each)
```

### Tier 1: Unit Tests (Fast, Mocked)

```python
# Test individual functions with mocks
import pytest
from unittest.mock import Mock, patch
from src.utils.document import process_document

class TestDocumentProcessing:
    """Unit tests for document processing utilities."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock()
        settings.processing.chunk_size = 1000
        settings.processing.chunk_overlap = 100
        return settings
    
    @patch('src.utils.document.parse_document_unstructured')
    async def test_process_document_success(self, mock_parse, mock_settings):
        """Test successful document processing."""
        # Arrange
        mock_parse.return_value = "Sample document content"
        file_path = Path("test.pdf")
        
        # Act
        result = await process_document(file_path, mock_settings)
        
        # Assert
        assert result.chunks > 0
        assert result.file_path == file_path
        mock_parse.assert_called_once_with(file_path)
    
    @patch('src.utils.document.parse_document_unstructured')
    async def test_process_document_parse_error(self, mock_parse, mock_settings):
        """Test error handling in document processing."""
        # Arrange
        mock_parse.side_effect = Exception("Parse failed")
        
        # Act & Assert
        with pytest.raises(DocumentProcessingError, match="Parse failed"):
            await process_document(Path("invalid.pdf"), mock_settings)
```

### Tier 2: Integration Tests (Lightweight Models)

```python
# Test component interactions with lightweight models
import pytest
import asyncio
from src.agents.coordinator import MultiAgentCoordinator

class TestMultiAgentIntegration:
    """Integration tests for multi-agent coordination."""
    
    @pytest.fixture
    async def coordinator(self, test_settings):
        """Create coordinator with test configuration."""
        # Use lightweight test models
        test_settings.vllm.model = "microsoft/DialoGPT-small"  # Lightweight
        test_settings.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        coordinator = MultiAgentCoordinator(test_settings)
        await coordinator.initialize()
        
        yield coordinator
        
        await coordinator.cleanup()
    
    async def test_agent_coordination_pipeline(self, coordinator):
        """Test complete agent coordination flow."""
        # Arrange
        query = "What are the key findings in the uploaded documents?"
        
        # Act
        response = await coordinator.arun(query)
        
        # Assert
        assert response is not None
        assert len(response) > 0
        assert coordinator.last_execution_time < 5000  # <5s for integration test
    
    async def test_parallel_agent_execution(self, coordinator):
        """Test agents can run in parallel."""
        queries = [
            "Summarize the main points",
            "Find technical details",
            "Extract conclusions"
        ]
        
        # Execute queries in parallel
        tasks = [coordinator.arun(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(results) == len(queries)
        for result in results:
            assert result is not None
```

### Tier 3: System Tests (Real Models & GPU)

```python
# Full system tests with production models
import pytest
from src.config import settings

@pytest.mark.system
@pytest.mark.requires_gpu
class TestProductionSystem:
    """System tests with production configuration."""
    
    def test_full_document_analysis_pipeline(self, sample_documents):
        """Test complete document analysis with real models."""
        # This test requires RTX 4090 and production models
        coordinator = MultiAgentCoordinator(settings)  # Production settings
        
        # Upload and process documents
        processed = coordinator.process_documents(sample_documents)
        
        # Execute complex query
        query = "Analyze the technical implications and provide recommendations"
        response = coordinator.run(query)
        
        # Verify production performance targets
        assert coordinator.last_execution_time < 10000  # <10s total
        assert len(response) > 100  # Substantial response
        assert "recommendation" in response.lower()  # Contains requested content
    
    def test_128k_context_window(self, large_document):
        """Test handling of large context (128K tokens)."""
        # Test with document approaching context limit
        assert len(large_document.tokens) > 100000  # >100K tokens
        
        coordinator = MultiAgentCoordinator(settings)
        response = coordinator.run("Summarize this large document", large_document)
        
        # Should handle large context without truncation errors
        assert "summary" in response.lower()
        assert len(response) > 200
```

### Performance Testing

```python
# Performance validation tests
class TestPerformanceTargets:
    """Validate performance targets are met."""
    
    def test_agent_coordination_latency(self):
        """Test <200ms agent coordination target."""
        coordinator = MultiAgentCoordinator(settings)
        
        start_time = time.time()
        decision = coordinator.route_query("Simple test query")
        coordination_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert coordination_time < 200, f"Coordination took {coordination_time}ms"
    
    def test_embedding_generation_performance(self):
        """Test embedding generation speed."""
        embedder = BGEM3EmbeddingManager(settings.embedding)
        
        test_text = "Sample document text for performance testing"
        
        start_time = time.time()
        embedding = embedder.get_dense_embedding(test_text)
        generation_time = (time.time() - start_time) * 1000
        
        assert generation_time < 50, f"Embedding took {generation_time}ms"
        assert len(embedding) == 1024, "Should generate 1024D dense embedding"
```

## Implementation Experience

### Clean Test Infrastructure Implementation

Based on real implementation experience migrating DocMind AI's test architecture from legacy patterns to modern pytest + BaseSettings patterns.

#### Key Architectural Decisions

**Migration Strategy Applied**: Big Bang Migration

- All affected test files migrated simultaneously
- TestSettings Pattern: pytest fixtures with BaseSettings subclass
- ADR Compliance: 200ms timeout, BGE-M3 model references
- Zero Backward Compatibility: Complete removal of deprecated patterns

**Three-Tier Test Settings Hierarchy**:

```python
DocMindSettings (production)
├── TestDocMindSettings (unit tests)
│   └── IntegrationTestSettings (integration tests)
└── SystemTestSettings (system tests)
```

#### Test Settings Implementation

**Create Clean BaseSettings Subclasses**:

```python
# tests/fixtures/test_settings.py
from src.config.settings import DocMindSettings

class TestDocMindSettings(DocMindSettings):
    """Optimized settings for fast unit tests."""
    
    # Performance optimizations for testing
    enable_gpu_acceleration: bool = False     # Unit tests CPU-only
    agent_decision_timeout: int = 100         # 5x faster than production
    context_window_size: int = 1024           # 128x smaller than production
    chunk_size: int = 256                     # 2x smaller for test speed
    
    model_config = SettingsConfigDict(
        env_prefix="DOCMIND_TEST_",           # Isolated environment
        env_file=None                         # No .env loading
    )

class IntegrationTestSettings(TestDocMindSettings):
    """Settings for integration tests with moderate performance."""
    
    enable_gpu_acceleration: bool = True      # GPU enabled for integration
    agent_decision_timeout: int = 150         # Moderate timeout
    context_window_size: int = 4096           # Larger context for integration
    
    model_config = SettingsConfigDict(
        env_prefix="DOCMIND_INTEGRATION_"
    )

class SystemTestSettings(DocMindSettings):
    """Production settings for system tests."""
    pass  # Inherits full production configuration
```

**Update Pytest Fixtures**:

```python
# tests/conftest.py
import pytest
from tests.fixtures.test_settings import (
    TestDocMindSettings,
    IntegrationTestSettings,
    SystemTestSettings
)

@pytest.fixture(scope="session")
def test_settings():
    """Primary fixture for unit tests with temp directories."""
    return TestDocMindSettings()

@pytest.fixture(scope="session")
def integration_test_settings():
    """Moderate performance settings for component testing."""
    return IntegrationTestSettings()

@pytest.fixture(scope="session")
def system_test_settings():
    """Production defaults for end-to-end validation."""
    return SystemTestSettings()

@pytest.fixture
def settings_with_overrides(test_settings):
    """Factory fixture for custom configurations."""
    def _create_settings(**overrides):
        return test_settings.model_copy(update=overrides)
    return _create_settings
```

#### Migration Patterns and Fixes

**Before (Legacy Pattern)**:

```python
# Anti-pattern: Direct settings instantiation
def test_example():
    settings = DocMindSettings(enable_gpu_acceleration=False)
    assert settings.embedding_model == "BAAI/bge-large-en-v1.5"  # Wrong!
```

**After (Clean Pattern)**:

```python
# Modern pattern: Fixture injection
def test_example(test_settings):
    assert test_settings.bge_m3_model_name == "BAAI/bge-m3"     # ADR-compliant!
    
    # Runtime overrides when needed
    custom_settings = test_settings.model_copy(
        update={'context_window_size': 2048}
    )
    assert custom_settings.context_window_size == 2048
```

**Common Migration Fixes Applied**:

1. **Model Name Updates** (ADR-002 Compliance):

   ```python
   # Before
   settings.embedding_model == "BAAI/bge-large-en-v1.5"
   
   # After
   settings.bge_m3_model_name == "BAAI/bge-m3"
   ```

2. **Timeout Adjustments** (ADR-024 Compliance):

   ```python
   # Before
   settings.agent_decision_timeout == 300
   
   # After  
   settings.agent_decision_timeout == 200  # Production
   settings.agent_decision_timeout == 100  # Test optimized
   ```

3. **Deprecated Method Removal**:

   ```python
   # Before: Manual synchronization
   settings._sync_nested_models()  # DEPRECATED
   
   # After: Automatic synchronization
   # No manual calls needed - handled by pydantic-settings
   ```

#### Implementation Benefits Achieved

**Zero Custom Backward Compatibility Code**:

- No `_sync_nested_models()` complexity
- No dual flat/nested architecture
- Standard pydantic-settings patterns only

**Complete Test Isolation from Production**:

- Separate environment prefixes prevent contamination
- Test settings don't load `.env` file
- Production configuration remains completely clean

**Library-First Compliance**:

- pytest fixtures: Standard session/function scoped fixtures
- pydantic-settings: BaseSettings subclassing with proper inheritance
- model_copy(): Runtime overrides using official patterns

**Performance Optimizations**:

- Test settings use 90% less memory than production
- 10x faster timeouts for test execution
- Smaller batch sizes and document limits
- GPU acceleration properly disabled for unit tests

#### Critical Success Factors

**What Made This Migration Successful**:

1. **Library-First Approach**: Used proven pytest + pydantic-settings patterns
2. **Complete Separation**: Zero contamination between test and production code
3. **ADR Compliance**: All settings aligned with architectural decisions
4. **Performance Optimization**: Test-specific optimizations for speed
5. **Environment Isolation**: Clear separation via environment prefixes

**Anti-Patterns to Avoid**:

1. **Mixing Test Code in Production Classes**:

   ```python
   # WRONG: Test-specific code in production
   class DocMindSettings:
       def _sync_nested_models(self):  # Test contamination
           pass
   ```

2. **Direct Environment Variable Access**:

   ```python
   # WRONG: Bypassing configuration system
   import os
   timeout = int(os.getenv("DOCMIND_TIMEOUT", "200"))
   ```

3. **Hardcoded Test Dependencies**:

   ```python
   # WRONG: Forcing hardware assumptions
   settings = DocMindSettings(enable_gpu_acceleration=True)  # Fails on CPU-only
   ```

#### Lessons Learned

**Critical Insights for Future Development**:

1. **Local vs Server Application Context**: DocMind AI is a LOCAL USER APPLICATION, not a server application. This distinction affects all architectural decisions and requires preserving user choice.

2. **User Flexibility is Non-Negotiable**: All 5 user scenarios (Student, Developer, Researcher, Privacy User, Custom User) must be supported without forcing hardware assumptions.

3. **Library-First Success Patterns**: Standard library patterns (pytest, pydantic-settings) reduce maintenance burden and provide familiar developer experience.

4. **Test Contamination Prevention**: Complete separation between test and production code prevents subtle bugs and maintains clean architecture.

5. **Environment Isolation**: Proper environment variable prefixes ensure test runs never interfere with production configuration.

This implementation experience demonstrates that major architectural migrations can be successful when following library-first principles, maintaining clear separation of concerns, and preserving backward compatibility through proper abstraction layers.

### Test Configuration

```python
# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_settings():
    """Test-specific settings."""
    from src.config import DocMindSettings
    
    return DocMindSettings(
        debug=True,
        log_level="DEBUG",
        # Use test models for faster execution
        vllm=VLLMConfig(model="microsoft/DialoGPT-small"),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        # Reduced timeouts for testing
        agents=AgentConfig(decision_timeout=100)  # 100ms for tests
    )

@pytest.fixture
def sample_documents(tmp_path):
    """Generate sample documents for testing."""
    doc1 = tmp_path / "sample1.txt"
    doc1.write_text("This is a sample document with technical content.")
    
    doc2 = tmp_path / "sample2.txt"
    doc2.write_text("Another document with different information.")
    
    return [doc1, doc2]
```

## Code Quality & Maintenance

### Code Quality Standards

#### Automated Quality Checks

```bash
# Run comprehensive quality checks
ruff check . --fix          # Linting and auto-fix
ruff format .               # Code formatting
pytest --cov=src           # Test coverage
mypy src/                   # Type checking
```

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest tests/unit/
        language: system
        pass_filenames: false
        always_run: true
```

### Configuration Maintenance

**CRITICAL**: All configuration changes must follow the unified pattern:

```python
# Adding new configuration
class NewFeatureConfig(BaseModel):
    """Configuration for new feature."""
    enabled: bool = Field(default=False)
    parameter: int = Field(default=100, ge=1, le=1000, description="Parameter description")
    timeout: float = Field(default=1.0, ge=0.1, le=10.0)

# Update DocMindSettings
class DocMindSettings(BaseSettings):
    # Existing configs...
    new_feature: NewFeatureConfig = Field(default_factory=NewFeatureConfig)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DOCMIND_",
        env_nested_delimiter="__",  # Enable DOCMIND_NEW_FEATURE__ENABLED
        case_sensitive=False
    )
```

```bash
# Environment variable naming (MANDATORY)
DOCMIND_NEW_FEATURE__ENABLED=true           # Nested configuration
DOCMIND_NEW_FEATURE__PARAMETER=250          # Parameter with validation
DOCMIND_NEW_FEATURE__TIMEOUT=2.5            # Float parameter
```

### Performance Monitoring

```python
# Add performance monitoring to new features
import time
from src.utils.monitoring import performance_monitor, logger

@performance_monitor("document_processing")
async def process_document_with_monitoring(file_path: Path) -> ProcessedDocument:
    """Document processing with built-in performance monitoring."""
    start_time = time.time()
    
    try:
        result = await process_document_pipeline(file_path)
        
        # Log performance metrics
        processing_time = time.time() - start_time
        logger.info(f"Document processed in {processing_time:.2f}s", extra={
            "file_size": file_path.stat().st_size,
            "chunks_created": result.chunks,
            "processing_time": processing_time
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}", extra={
            "file_path": str(file_path),
            "error_type": type(e).__name__
        })
        raise
```

### Dependency Management

```bash
# Update dependencies safely
uv add package-name                      # Add new dependency
uv add --dev package-name               # Add development dependency
uv sync --all-extras                    # Update all dependencies

# Check for security vulnerabilities
uv pip-audit

# Update to latest compatible versions
uv update
```

## Advanced Implementation Details

### BGE-M3 Unified Embeddings Implementation

DocMind AI uses BGE-M3 for unified dense + sparse embeddings:

```python
class BGEM3EmbeddingManager:
    """Advanced BGE-M3 embedding manager with optimization."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._batch_size = config.batch_size
        
    async def initialize(self):
        """Initialize model with FP16 optimization."""
        from FlagEmbedding import BGEM3FlagModel
        
        self.model = BGEM3FlagModel(
            model_name_or_path=self.config.model_name,
            use_fp16=True,  # FP16 for RTX 4090 optimization
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
    async def get_unified_embeddings(
        self, 
        texts: List[str]
    ) -> Dict[str, Any]:
        """Generate both dense and sparse embeddings efficiently."""
        
        # Batch processing for efficiency
        results = {
            "dense": [],
            "sparse": [],
            "colbert": []  # For potential ColBERT usage
        }
        
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            
            # Generate all embedding types in one call
            batch_embeddings = self.model.encode(
                batch,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False  # Disabled for memory efficiency
            )
            
            results["dense"].extend(batch_embeddings["dense_vecs"])
            results["sparse"].extend(batch_embeddings["lexical_weights"])
            
        return results
    
    def _optimize_sparse_embeddings(self, sparse_embeddings: List[Dict]) -> List[Dict]:
        """Optimize sparse embeddings for storage efficiency."""
        optimized = []
        
        for embedding in sparse_embeddings:
            # Keep only top-k sparse dimensions
            sorted_items = sorted(
                embedding.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:100]  # Top 100 dimensions
            
            optimized.append(dict(sorted_items))
            
        return optimized
```

### Hybrid Search with RRF Fusion

```python
class QdrantHybridRetriever:
    """Advanced hybrid retrieval with RRF fusion."""
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,  # Dense weight
        rerank: bool = True
    ) -> List[Document]:
        """Execute hybrid search with configurable fusion."""
        
        # Generate query embeddings
        query_embeddings = await self.embedder.get_unified_embeddings([query])
        dense_query = query_embeddings["dense"][0]
        sparse_query = query_embeddings["sparse"][0]
        
        # Parallel search execution
        dense_task = asyncio.create_task(
            self._dense_search(dense_query, top_k * 2)
        )
        sparse_task = asyncio.create_task(
            self._sparse_search(sparse_query, top_k * 2)
        )
        
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )
        
        # RRF fusion
        fused_results = self._rrf_fusion(
            dense_results, sparse_results, alpha
        )
        
        # Optional reranking
        if rerank and len(fused_results) > 5:
            fused_results = await self._rerank_results(
                query, fused_results[:top_k * 2]
            )
        
        return fused_results[:top_k]
    
    def _rrf_fusion(
        self, 
        dense_results: List[Document], 
        sparse_results: List[Document], 
        alpha: float
    ) -> List[Document]:
        """Reciprocal Rank Fusion implementation."""
        
        scores = defaultdict(float)
        doc_map = {}
        
        # Dense ranking contribution
        for rank, doc in enumerate(dense_results):
            scores[doc.id] += alpha / (60 + rank + 1)  # RRF constant = 60
            doc_map[doc.id] = doc
            
        # Sparse ranking contribution  
        for rank, doc in enumerate(sparse_results):
            scores[doc.id] += (1 - alpha) / (60 + rank + 1)
            doc_map[doc.id] = doc
            
        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [doc_map[doc_id] for doc_id in sorted_ids if doc_id in doc_map]
```

### Advanced Agent Coordination

```python
class AdvancedMultiAgentCoordinator:
    """Advanced multi-agent coordination with state management."""
    
    def __init__(self, settings: DocMindSettings):
        self.settings = settings
        self.agents = {}
        self.state_manager = AgentStateManager()
        self.execution_graph = self._create_execution_graph()
    
    def _create_execution_graph(self):
        """Create LangGraph execution graph."""
        from langgraph.graph import StateGraph, MessagesState
        
        # Define agent workflow
        workflow = StateGraph(MessagesState)
        
        # Add agent nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("validator", self._validator_node)
        
        # Define conditional edges
        workflow.add_conditional_edges(
            "router",
            self._should_plan,
            {"plan": "planner", "direct": "retrieval"}
        )
        
        workflow.add_edge("planner", "retrieval")
        workflow.add_edge("retrieval", "synthesis")
        workflow.add_edge("synthesis", "validator")
        
        # Set entry and exit points
        workflow.set_entry_point("router")
        workflow.set_finish_point("validator")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    async def _router_node(self, state: MessagesState) -> Dict:
        """Router agent node with complexity analysis."""
        query = state["messages"][-1].content
        
        # Analyze query complexity
        complexity_score = self._analyze_complexity(query)
        routing_decision = RoutingDecision(
            complexity=complexity_score,
            strategy="hybrid" if complexity_score > 0.5 else "dense",
            requires_planning=complexity_score > 0.7
        )
        
        # Update state
        state["routing_decision"] = routing_decision
        state["complexity_score"] = complexity_score
        
        return state
    
    def _analyze_complexity(self, query: str) -> float:
        """Analyze query complexity for routing decisions."""
        complexity_indicators = [
            len(query.split()) > 20,           # Long query
            "compare" in query.lower(),        # Comparison request  
            "analyze" in query.lower(),        # Analysis request
            "relationship" in query.lower(),   # Relationship query
            query.count("?") > 1,              # Multiple questions
            "and" in query.lower() and "or" in query.lower()  # Complex logic
        ]
        
        return sum(complexity_indicators) / len(complexity_indicators)
```

## Performance Optimization

### GPU Memory Optimization

```python
# GPU memory management patterns
class GPUMemoryManager:
    """Manage GPU memory efficiently."""
    
    def __init__(self):
        self.memory_threshold = 0.85  # 85% usage limit
        self.cleanup_threshold = 0.95  # Force cleanup at 95%
    
    @contextmanager
    def memory_context(self):
        """Context manager for GPU memory management."""
        initial_memory = torch.cuda.memory_allocated()
        
        try:
            yield
        finally:
            current_memory = torch.cuda.memory_allocated()
            memory_increase = current_memory - initial_memory
            
            if self._should_cleanup():
                torch.cuda.empty_cache()
                logger.info(f"GPU memory cleanup: freed {memory_increase / 1e6:.1f}MB")
    
    def _should_cleanup(self) -> bool:
        """Check if GPU memory cleanup is needed."""
        if not torch.cuda.is_available():
            return False
            
        memory_used = torch.cuda.memory_allocated()
        memory_total = torch.cuda.get_device_properties(0).total_memory
        utilization = memory_used / memory_total
        
        return utilization > self.cleanup_threshold

# Usage in agent coordination
async def execute_with_memory_management(self, query: str) -> str:
    """Execute query with GPU memory management."""
    
    with self.gpu_manager.memory_context():
        # Execute agents in sequence to manage memory
        routing = await self.router_agent.arun(query)
        
        if routing.requires_planning:
            planning = await self.planner_agent.arun(query)
            
        retrieval = await self.retrieval_agent.arun(query)
        
        # Clear intermediate results to free memory
        torch.cuda.empty_cache()
        
        synthesis = await self.synthesis_agent.arun(query, retrieval)
        validation = await self.validator_agent.arun(synthesis)
        
        return validation.response
```

### Caching Strategies

```python
# Intelligent caching for expensive operations
from functools import lru_cache
from src.cache.simple_cache import SimpleCache

class IntelligentCaching:
    """Multi-level caching for performance optimization."""
    
    def __init__(self):
        self.memory_cache = {}  # In-memory for frequent access
        self.disk_cache = SimpleCache()  # Persistent disk cache
        self.embedding_cache = {}  # Specialized embedding cache
    
    @lru_cache(maxsize=1000)
    def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """Memory-cached embeddings for repeated text."""
        return self.embedding_cache.get(text_hash)
    
    async def get_or_generate_embedding(self, text: str) -> List[float]:
        """Get embedding from cache or generate new one."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Check memory cache first
        if cached := self.get_cached_embedding(text_hash):
            return cached
        
        # Check disk cache
        cache_key = f"embedding:{text_hash}"
        if cached := await self.disk_cache.get(cache_key):
            # Promote to memory cache
            self.embedding_cache[text_hash] = cached
            return cached
        
        # Generate new embedding
        embedding = await self.embedder.generate_embedding(text)
        
        # Store in both caches
        self.embedding_cache[text_hash] = embedding
        await self.disk_cache.set(cache_key, embedding, ttl=86400)  # 24h TTL
        
        return embedding
```

## Troubleshooting & Debugging

### Comprehensive Debugging Setup

```python
# Advanced logging configuration
import logging
import structlog
from src.utils.monitoring import setup_structured_logging

# Configure structured logging
setup_structured_logging(
    level=logging.DEBUG if settings.debug else logging.INFO,
    include_context=True,
    include_performance=True
)

logger = structlog.get_logger(__name__)

# Usage in agent code
async def debug_agent_execution(self, query: str) -> str:
    """Execute with comprehensive debugging."""
    
    logger.info("Starting agent execution", 
                query_length=len(query),
                agent_timeout=self.settings.agents.decision_timeout)
    
    try:
        with logger.bind(operation="agent_coordination"):
            start_time = time.time()
            
            # Execute with timing
            result = await self._execute_coordination(query)
            
            execution_time = time.time() - start_time
            
            logger.info("Agent execution completed",
                       execution_time_ms=execution_time * 1000,
                       result_length=len(result),
                       agents_used=self._get_agents_used())
            
            return result
            
    except Exception as e:
        logger.error("Agent execution failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    query_preview=query[:100])
        raise
```

### Common Issue Resolution

#### Configuration Issues

```bash
# Debug configuration loading
python -c "
from src.config import settings
import pprint
pprint.pprint(settings.model_dump())
"

# Validate specific configuration sections
python -c "
from src.config import settings
print('vLLM Config:', settings.vllm.model_dump())
print('Agent Config:', settings.agents.model_dump())
print('Embedding Config:', settings.embedding.model_dump())
"
```

#### Performance Issues

```python
# Performance profiling
import cProfile
import pstats
from src.agents.coordinator import MultiAgentCoordinator

def profile_agent_execution():
    """Profile agent execution for bottlenecks."""
    coordinator = MultiAgentCoordinator(settings)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute test query
    result = asyncio.run(coordinator.arun("Test query for profiling"))
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result

# Memory profiling
from memory_profiler import profile

@profile
def memory_profile_processing():
    """Profile memory usage during document processing."""
    documents = load_sample_documents()
    processed = process_documents_batch(documents)
    return processed
```

#### GPU Issues

```python
# GPU diagnostics
def diagnose_gpu_setup():
    """Comprehensive GPU setup diagnostics."""
    
    print("=== GPU DIAGNOSTICS ===")
    
    # CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
        
        # Memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        
        print(f"Total VRAM: {memory_total:.1f}GB")
        print(f"Allocated: {memory_allocated:.1f}GB")
        print(f"Reserved: {memory_reserved:.1f}GB")
        print(f"Available: {memory_total - memory_reserved:.1f}GB")
    
    # vLLM diagnostics
    try:
        import vllm
        print(f"vLLM Version: {vllm.__version__}")
        print(f"FlashInfer Available: {'FLASHINFER' in os.environ.get('VLLM_ATTENTION_BACKEND', '')}")
    except ImportError:
        print("vLLM not installed")
    
    # Environment variables
    print("\n=== RELEVANT ENV VARS ===")
    for key, value in os.environ.items():
        if any(prefix in key for prefix in ['VLLM_', 'CUDA_', 'TORCH_']):
            print(f"{key}: {value}")

# Run diagnostics
if __name__ == "__main__":
    diagnose_gpu_setup()
```

---

This developer handbook provides comprehensive guidance for contributing to DocMind AI. The unified architecture and clear patterns ensure consistent, maintainable, and high-performance code across all contributions.

For configuration details, see [Configuration Reference](configuration-reference.md).
For deployment procedures, see [Operations Guide](operations-guide.md).
For architectural understanding, see [System Architecture](system-architecture.md).
