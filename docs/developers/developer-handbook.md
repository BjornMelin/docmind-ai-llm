# DocMind AI Developer Handbook

## Overview

This comprehensive handbook provides practical implementation guidance for developers working with DocMind AI. It covers development standards, testing strategies, implementation patterns, code quality guidelines, and maintenance procedures to ensure consistent, high-quality contributions to the project.

> **Prerequisites**: Complete the [Getting Started Guide](getting-started.md) and understand the [System Architecture](system-architecture.md) before following this handbook.

## Table of Contents

1. [Development Standards](#development-standards)
2. [Implementation Patterns](#implementation-patterns)
3. [Configuration Architecture Implementation](#configuration-architecture-implementation)
4. [Testing Strategies](#testing-strategies)
5. [Code Quality & Maintenance](#code-quality--maintenance)
6. [Advanced Implementation Details](#advanced-implementation-details)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting & Debugging](#troubleshooting--debugging)

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
uv run ruff format . && uv run ruff check . --fix

# Configuration in pyproject.toml
[tool.ruff]
line-length = 88
target-version = "py313"
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
model = os.getenv("DOCMIND_MODEL")  # Don't do this
```

## Implementation Patterns

### Async-First Development

DocMind AI uses async/await throughout for optimal performance:

```python
# Agent coordination - public API pattern
class MultiAgentCoordinator:
    def process_query(self, query: str, context: Any | None = None) -> AgentResponse:
        """Execute multi-agent coordination synchronously."""

        # Internal workflow execution (async loop wrapped)
        result = self._run_agent_workflow(query, context)

        return self._extract_response(result, query)
```

### Document Processing Pattern

```python
"""
DocMind uses a library-first ingestion pipeline (`src/processing/ingestion_pipeline.py`)
based on LlamaIndex `IngestionPipeline`.

Optional NLP enrichment (sentences + entities) is centralized under `src/nlp/` and
wired into ingestion as a transform (`src/processing/nlp_enrichment.py`).

See: docs/specs/spec-015-nlp-enrichment-spacy.md
"""

from src.config import settings
from src.nlp.spacy_service import SpacyNlpService

if settings.spacy.enabled:
    service = SpacyNlpService(settings.spacy)
    enrichment = service.enrich_texts(["Hello world. Second sentence."])[0]

    print([s.text for s in enrichment.sentences])
    print([(e.label, e.text) for e in enrichment.entities])
else:
    print("spaCy enrichment is disabled in settings.")
```

### Error Handling Patterns

```python
# DocMind tools are fail-open: tools should return a structured payload even when
# the underlying integration fails, rather than raising and aborting the run.
#
# Example: src/agents/tools/retrieval.py returns a JSON string on both success
# and error paths.
from src.agents.tools.retrieval import retrieve_documents

payload_json = retrieve_documents("find docs about embeddings", strategy="hybrid")
print(payload_json)  # {"documents": [...], ...} or {"documents": [], "error": "...", ...}
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

## Configuration Architecture Implementation

### Overview -  Configuration

This section provides comprehensive guidance for implementing clean configuration architecture in DocMind AI, based on lessons learned from successfully eliminating 127 lines of test contamination and achieving 95% complexity reduction. The patterns documented here follow library-first principles using pytest + pydantic-settings.

### Clean Production Configuration Architecture

#### Core Configuration Principles

1. **Zero Test Contamination**: Production configuration must never contain test-specific code
2. **Single Source of Truth**: All configuration through unified `DocMindSettings` class
3. **Library-First Patterns**: Use standard Pydantic BaseSettings and pytest fixtures
4. **ADR Compliance**: Maintain alignment with architectural decisions

#### Production Settings Structure

```python
from pydantic_settings import SettingsConfigDict

# src/config/settings.py (authoritative)
SETTINGS_MODEL_CONFIG = SettingsConfigDict(
    # Dotenv is loaded explicitly at startup via `bootstrap_settings()` to avoid
    # import-time filesystem reads and to keep tests hermetic by default.
    env_file=None,
    env_prefix="DOCMIND_",
    env_nested_delimiter="__",
    case_sensitive=False,
    # Allow non-DOCMIND keys in `.env` without failing settings load.
    extra="ignore",
    populate_by_name=True,
)

# Recommended import pattern everywhere in the codebase:
from src.config import settings
from src.config import bootstrap_settings

# Entrypoints should opt into dotenv exactly once:
bootstrap_settings()
```

#### Key Anti-Patterns to Avoid

**❌ Test Code in Production Classes**:

```python
# ANTI-PATTERN: Never do this
class DocMindSettings(BaseSettings):
    if "pytest" in sys.modules:
        default_data_dir = "/tmp/docmind_test"
    else:
        default_data_dir = "./data"
```

**❌ Duplicate Field Definitions**:

```python
# ANTI-PATTERN: Conflicting field definitions
class DocMindSettings(BaseSettings):
    llm_backend: str = Field(default="vllm")    # Line 132
    # ... 50 lines later ...
    llm_backend: str = Field(default="ollama")  # Line 185 - CONFLICT!
```

**❌ Complex Synchronization Logic**:

```python
# ANTI-PATTERN: Custom synchronization instead of Pydantic patterns
def _sync_nested_models(self) -> None:
    """60+ lines of complex synchronization logic"""
    # Use Pydantic computed fields or validators instead
```

### Test Configuration Architecture

#### Test Settings Hierarchy

Prefer fixtures that:

- run tests in a temp working directory (so a developer `.env` is not accidentally loaded), and
- create isolated `DocMindSettings()` instances for pure code paths, or reset the global singleton in-place for UI/runtime code paths.

```python
# tests/conftest.py (patterns)
import importlib
from collections.abc import Iterator

import pytest

from src.config.settings import DocMindSettings


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch) -> DocMindSettings:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCMIND_LLM_BACKEND", "ollama")
    return DocMindSettings()


@pytest.fixture
def reset_global_settings() -> Iterator[None]:
    def _reset_in_place() -> None:
        settings_mod = importlib.import_module("src.config.settings")
        current = settings_mod.settings
        fresh = settings_mod.DocMindSettings()
        for field in settings_mod.DocMindSettings.model_fields:
            setattr(current, field, getattr(fresh, field))

    _reset_in_place()
    yield
    _reset_in_place()
```

### Configuration Migration Patterns

#### Migrating from Test-Contaminated Configuration

**Before: Contaminated Production Code**:

```python
# BEFORE: Production code with test contamination
class DocMindSettings(BaseSettings):
    # 127 lines of test compatibility code mixed with production logic
    
    # === FLAT ATTRIBUTES FOR TEST COMPATIBILITY ===
    model_name: str = Field(default="BAAI/bge-m3")  # Correct: BGE-M3 unified embedding model
    agent_decision_timeout: int = Field(default=300)  # Wrong - should be 200ms
    
    def _sync_nested_models(self) -> None:
        """60+ lines of complex synchronization for test support"""
        # Complex custom logic instead of Pydantic patterns
```

**After: Clean Separation**:

```python
# AFTER: Production configuration stays pure; tests control config via fixtures/env.
from src.config.settings import DocMindSettings

def make_settings_for_test(tmp_path, monkeypatch) -> DocMindSettings:
    monkeypatch.chdir(tmp_path)  # avoid reading a developer .env
    monkeypatch.setenv("DOCMIND_LLM_BACKEND", "ollama")
    return DocMindSettings()
```

#### Test Migration Examples

**Pattern 1: Basic Settings Usage**:

```python
# Before (uses backward compatibility)
def test_settings_default_values():
    settings = DocMindSettings()
    assert settings.embedding.model_name == "BAAI/bge-m3"  # Correct BGE-M3 model
    assert settings.agents.decision_timeout == 200  # Correct timeout

# After (uses proper test settings)
def test_settings_default_values(isolated_settings):
    assert isolated_settings.embedding.model_name == "BAAI/bge-m3"
    assert isolated_settings.agents.decision_timeout == 200
```

**Pattern 2: Environment Variable Testing**:

```python
# Before
@patch.dict(os.environ, {"DOCMIND_EMBEDDING_MODEL": "custom-model"})
def test_environment_override():
    settings = DocMindSettings() 
    assert settings.embedding.model_name == "custom-model"

# After (using modern nested pattern)
@patch.dict(os.environ, {"DOCMIND_EMBEDDING__MODEL_NAME": "custom-bge-m3"})
def test_environment_override():
    settings = DocMindSettings()
    assert settings.embedding.model_name == "custom-bge-m3"
```

### Configuration Reference & SSOT

For the exhaustive list of all 100+ environment variables, hardware optimization profiles, and programmatic mapping details, please refer to the:

**[Canonical Configuration Reference](configuration.md)**

#### Implementation Guidelines

1. **Zero Test Contamination**: Production configuration must never contain test-specific code. Use inheritance for test settings.
2. **Convention-Over-Configuration**: Use the `DOCMIND_` prefix and `__` delimiter for all environment overrides.
3. **Lazy Initialization**: Access `settings` instance at runtime; avoid module-level side effects during import if possible.
4. **Validation First**: Rely on Pydantic's built-in validation for range and type checks.

## Testing Strategies

See ADR‑029 for the boundary‑first strategy and ADR‑014 for CI quality gates and validation.

DocMind AI implements a comprehensive three-tier testing strategy:

### Testing Framework Setup

Testing dependencies are managed as dependency groups (`[dependency-groups]`, PEP 735).
Optional runtime features stay in `project.optional-dependencies` (PEP 621) and are installed
with `--extra` when needed.

```bash
# Install test dependencies
uv sync --group test

# Run tests by tier
pytest tests/unit/ -v                    # Tier 1: Fast unit tests (<5s each)
pytest tests/integration/ -v             # Tier 2: Cross-component tests (<30s each)
python scripts/run_tests.py --system     # Tier 3: Full system tests (<5min each)
```

### Tier 1: Unit Tests (Fast, Mocked)

```python
# Test individual functions with mocks
import pytest
import sys
from pathlib import Path
from types import ModuleType

from src.processing import ingestion_api

pytestmark = pytest.mark.unit


def test_collect_paths_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"nope")

    paths = ingestion_api.collect_paths(tmp_path, recursive=False)
    assert [p.name for p in paths] == ["a.txt", "b.md"]


@pytest.mark.asyncio
async def test_load_documents_falls_back_when_unstructured_missing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "llama_index.readers.file", ModuleType("x"))

    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")

    docs = await ingestion_api.load_documents([p])
    assert docs
    assert docs[0].metadata.get("source_filename") == "a.txt"
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
        response = coordinator.process_query(query)
        
        # Assert
        assert response.content is not None
        assert len(response.content) > 0
        assert response.processing_time < 5.0  # <5s for integration test
    
    async def test_parallel_agent_execution(self, coordinator):
        """Test agents can run in parallel."""
        queries = [
            "Summarize the main points",
            "Find technical details",
            "Extract conclusions"
        ]
        
        # Execute queries (synchronous calls)
        results = [coordinator.process_query(query) for query in queries]
        
        # Verify all succeeded
        assert len(results) == len(queries)
        for result in results:
            assert result.content is not None
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
        response = coordinator.process_query(query)
        
        # Verify production performance targets
        assert response.processing_time < 10.0  # <10s total
        assert len(response.content) > 100  # Substantial response
        assert "recommendation" in response.content.lower()  # Contains requested content
    
    def test_128k_context_window(self, large_document):
        """Test handling of large context (128K tokens)."""
        # Test with document approaching context limit
        assert len(large_document.tokens) > 100000  # >100K tokens
        
        coordinator = MultiAgentCoordinator(settings)
        response = coordinator.process_query("Summarize this large document", context=large_document)
        
        # Should handle large context without truncation errors
        assert "summary" in response.content.lower()
        assert len(response.content) > 200
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
        response = coordinator.process_query("Simple test query")
        coordination_time = response.optimization_metrics.get("coordination_overhead_ms", 0)
        
        assert coordination_time < 200, f"Coordination took {coordination_time}ms"
    
    def test_embedding_generation_performance(self):
        """Test embedding generation speed (LI MockEmbedding example)."""
        from llama_index.core.embeddings import MockEmbedding

        embedder = MockEmbedding(embed_dim=1024)

        test_text = "Sample document text for performance testing"

        start_time = time.time()
        embedding = embedder.get_text_embedding(test_text)
        generation_time = (time.time() - start_time) * 1000

        assert generation_time < 50, f"Embedding took {generation_time}ms"
        assert len(embedding) == 1024, "Should generate 1024D dense embedding"
```

### Further Resources

For further implementation patterns, see the [Configuration Guide](configuration.md) and the [System Architecture](system-architecture.md) deep dives.

```python
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
    assert settings.embedding.model_name == "BAAI/bge-m3"  # Correct!
```

**After (Clean Pattern)**:

```python
# Modern pattern: Fixture injection
def test_example(test_settings):
    assert test_settings.embedding.model_name == "BAAI/bge-m3"     # ADR-compliant!
    
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
   settings.embedding.model_name == "BAAI/bge-m3"  # CORRECT PATTERN
   
   # After
   settings.embedding.model_name == "BAAI/bge-m3"
   ```

2. **Timeout Adjustments** (ADR-024 Compliance):

   ```python
   # Before
   settings.agents.decision_timeout == 300
   
   # After  
   settings.agents.decision_timeout == 200  # Production
   settings.agents.decision_timeout == 100  # Test optimized
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
    from src.config.settings import DocMindSettings
    
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
        env_file=None,
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
# Add performance monitoring to new features (JSONL + optional OTEL)
from pathlib import Path

from src.utils.monitoring import async_performance_timer, logger


async def process_document_with_monitoring(file_path: Path) -> ProcessedDocument:
    """Document processing with built-in performance telemetry."""
    async with async_performance_timer(
        "document_processing",
        file_size_bytes=file_path.stat().st_size,
    ) as metrics:
        result = await process_document_pipeline(file_path)
        metrics["chunks_created"] = result.chunks
        logger.bind(file_path=file_path.name).info("Document processed")
        return result
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

### Hybrid Retrieval (Library-First, Server‑Side)

DocMind AI uses Qdrant's Query API server‑side hybrid search with named vectors:

```python
from qdrant_client import QdrantClient, models
from llama_index.core import Settings
from src.retrieval.sparse_query import encode_to_qdrant

client = QdrantClient(url="http://localhost:6333")

query = "unified embeddings"
dense = Settings.embed_model.get_query_embedding(query)  # BGE-M3 via LI
sparse = encode_to_qdrant(query)  # FastEmbed BM42→BM25

prefetch = [
    models.Prefetch(query=models.VectorInput(vector=dense), using="text-dense", limit=200),
]
if sparse is not None:
    prefetch.append(models.Prefetch(query=sparse, using="text-sparse", limit=400))

res = client.query_points(
    collection_name="docmind_docs",
    prefetch=prefetch,
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

```python
def get_unified_embeddings(
    self,
    texts: List[str]
) -> Dict[str, Any]:
    """Generate both dense and sparse embeddings efficiently."""
    
    # Batch processing for efficiency
    results = {"dense": [], "sparse": []}
    
    for i in range(0, len(texts), self._batch_size):
        batch = texts[i:i + self._batch_size]
        
        # Generate all embedding types in one call
        batch_embeddings = self.model.encode(
            batch,
            return_dense=True,
            return_sparse=True,
            # no colbert vectors in final architecture
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

For document-processing cache, use LlamaIndex IngestionCache with DuckDBKVStore directly. Avoid custom cache wrappers in production. See the cache implementation guide for wiring, configuration, operations, and troubleshooting: [cache.md](cache.md).

## Troubleshooting & Debugging

### Comprehensive Debugging Setup

```python
# Logging configuration (Loguru + JSONL telemetry)
from src.config import settings
from src.utils.monitoring import async_performance_timer, logger, setup_logging

setup_logging(log_level="DEBUG" if settings.debug else "INFO")

# Usage in agent code
async def debug_agent_execution(self, query: str) -> str:
    """Execute with comprehensive debugging."""
    
    logger.bind(
        query_length=len(query),
        agent_timeout=self.settings.agents.decision_timeout,
    ).info("Starting agent execution")
    
    try:
        async with async_performance_timer("agent_coordination") as metrics:
            result = await self._execute_coordination(query)
            metrics["result_length"] = len(result)
            metrics["agents_used"] = self._get_agents_used()
            logger.info("Agent execution completed")
            return result
            
    except Exception as e:
        logger.bind(
            error_type=type(e).__name__,
            query_preview=query[:100],
        ).exception("Agent execution failed")
        raise
```

### Common Issue Resolution

#### Configuration Issues

```bash
# Debug configuration loading
uv run python -c "
from src.config import settings
import pprint
pprint.pprint(settings.model_dump())
"

# Validate specific configuration sections
uv run python -c "
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
    
    # vLLM diagnostics (external server)
    base_url = os.environ.get("DOCMIND_OPENAI__BASE_URL") or os.environ.get("DOCMIND_VLLM__VLLM_BASE_URL")
    if base_url:
        print(f"vLLM Base URL (configured): {base_url}")
        print("Tip: validate the server with: curl --fail --silent \"$DOCMIND_OPENAI__BASE_URL/models\"")
    else:
        print("vLLM Base URL not configured")
    
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

For configuration details, see the [Configuration Guide](configuration.md).
For deployment procedures, see [Operations Guide](operations-guide.md).
For architectural understanding, see [System Architecture](system-architecture.md).
