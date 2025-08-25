# DocMind AI Test Framework

## Overview

This test framework implements ML testing best practices with a tiered strategy based on AI research recommendations. It uses proper LlamaIndex mocking components and lightweight models for efficient, reliable testing.

## Testing Strategy (Two-Tier + Optional GPU Smoke Tests)

### 🔹 Unit Tests (Fast, Mocked)

- **Speed**: <5 seconds each
- **Dependencies**: None (CPU-only)
- **Components**: `MockEmbedding`, `MockLLM` from LlamaIndex
- **Purpose**: Test business logic, algorithms, validation
- **Run with**: `pytest -m unit`

```python
@pytest.mark.unit
def test_embedding_logic(mock_settings):
    # Uses MockEmbedding (1024-dim) for deterministic testing
    Settings.embed_model = MockEmbedding(embed_dim=1024)
    embeddings = Settings.embed_model.get_text_embedding("test")
    assert len(embeddings) == 1024
```

### 🔸 Integration Tests (Moderate Speed, Lightweight)

- **Speed**: 10-30 seconds each
- **Dependencies**: Lightweight models (all-MiniLM-L6-v2: 80MB)
- **Components**: Real model integration, mocked external services
- **Purpose**: Test component interaction, data flow
- **Run with**: `pytest -m integration`

```python
@pytest.mark.integration
def test_real_embedding_pipeline(integration_settings, lightweight_embedding_model):
    # Uses all-MiniLM-L6-v2 (80MB) instead of BGE-M3 (1GB)
    if lightweight_embedding_model:
        embeddings = lightweight_embedding_model.encode(["test document"])
        assert embeddings.shape[1] == 384  # MiniLM dimensions
```

### 🔶 GPU Smoke Tests (Optional Manual Validation)

- **Speed**: Manual execution outside CI
- **Dependencies**: Production models, GPU, external services  
- **Components**: Real hardware validation scripts
- **Purpose**: Pre-release validation on target hardware
- **Run with**: Manual scripts or `pytest -m requires_gpu`

```bash
# Manual GPU smoke test (outside CI)
python scripts/gpu_validation.py

# Optional GPU tests via pytest
pytest -m "requires_gpu" --timeout=600
```

## Test Configuration Framework

### Settings Fixtures

The framework provides three settings fixtures for different test levels:

#### `mock_settings` (Unit Tests)

```python
@pytest.fixture(scope="session")
def mock_settings() -> AppSettings:
    """Configure MockLLM and MockEmbedding for unit tests."""
    Settings.llm = MockLLM(max_tokens=256, temperature=0.0)
    Settings.embed_model = MockEmbedding(embed_dim=1024)  # Match BGE-M3
    return AppSettings(backend="mock", ...)
```

#### `integration_settings` (Integration Tests)

```python
@pytest.fixture(scope="session") 
def integration_settings() -> AppSettings:
    """Use lightweight models for integration tests."""
    return AppSettings(
        dense_embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # 80MB
        enable_reranking=False,  # Disable expensive operations
        ...
    )
```

#### `gpu_smoke_config` (Optional GPU Validation)

```python
# For manual GPU smoke tests (outside CI)
def gpu_smoke_config() -> dict:
    """Configuration for manual GPU validation."""
    return {
        "model_name": "BAAI/bge-large-en-v1.5",  # 1GB
        "enable_reranking": True,  # Full feature testing
        "gpu_memory_limit": 14.0,  # RTX 4090 target
        ...
    }
```

### Core Test Fixtures

#### Document Fixtures

- `test_documents`: Small, consistent set (5 docs) for unit/integration
- `large_document_set`: Performance testing (100 docs)

#### Model Fixtures  

- `lightweight_embedding_model`: all-MiniLM-L6-v2 (80MB) for integration
- `in_memory_graph_store`: SimplePropertyGraphStore for testing
- `mock_qdrant_client`: Comprehensive async/sync Qdrant mock

#### Infrastructure

- `cleanup_test_artifacts`: Session cleanup for test isolation
- `temp_vector_store`: Temporary directories for testing

## Test Markers

### Core Categories

- `@pytest.mark.unit`: Fast unit tests with mocks
- `@pytest.mark.integration`: Integration tests with lightweight models
- `@pytest.mark.requires_gpu`: Optional GPU tests (manual execution)

### Resource Requirements

- `@pytest.mark.requires_gpu`: Tests requiring GPU acceleration
- `@pytest.mark.requires_network`: Tests requiring network access
- `@pytest.mark.requires_ollama`: Tests requiring Ollama server

### Feature Areas

- `@pytest.mark.agents`: Multi-agent coordination tests
- `@pytest.mark.retrieval`: Retrieval and search system tests
- `@pytest.mark.embeddings`: Embedding model tests
- `@pytest.mark.multimodal`: CLIP and multimodal tests

### Performance

- `@pytest.mark.performance`: Performance benchmarks
- `@pytest.mark.slow`: Long-running tests

## Usage Examples

### Running Test Categories

```bash
# Fast unit tests only (CI/local development)
pytest -m unit

# Integration tests with lightweight models  
pytest -m integration

# Optional GPU tests (manual execution)
pytest -m "requires_gpu" --timeout=600

# Skip GPU tests on CPU-only machines
pytest -m "not requires_gpu"

# Test specific features
pytest -m "retrieval and unit"
pytest -m "multimodal and not slow"
```

### Test Development Patterns

#### Unit Test Pattern

```python
@pytest.mark.unit
@pytest.mark.embeddings
def test_embedding_dimension_validation(mock_settings):
    """Test embedding dimension validation logic."""
    # Use MockEmbedding for deterministic testing
    embedding_model = MockEmbedding(embed_dim=1024)
    text_embedding = embedding_model.get_text_embedding("test")
    
    # Test validation logic
    assert len(text_embedding) == 1024
    assert all(isinstance(x, float) for x in text_embedding)
```

#### Integration Test Pattern  

```python
@pytest.mark.integration
@pytest.mark.embeddings  
async def test_embedding_pipeline_with_lightweight_model(
    integration_settings, lightweight_embedding_model
):
    """Test embedding pipeline with real lightweight model."""
    if not lightweight_embedding_model:
        pytest.skip("Lightweight model not available")
    
    # Use real model but lightweight (80MB vs 1GB)
    documents = ["AI system design", "Machine learning pipeline"]
    embeddings = lightweight_embedding_model.encode(documents)
    
    # Validate real model behavior
    assert embeddings.shape == (2, 384)  # all-MiniLM-L6-v2 dims
    assert embeddings.dtype == np.float32
```

#### GPU Smoke Test Pattern

```python
@pytest.mark.requires_gpu
@pytest.mark.timeout(300)
def test_gpu_smoke_validation():
    """Manual GPU smoke test for pre-release validation."""
    # Run outside CI - manual hardware validation
    if not torch.cuda.is_available():
        pytest.skip("GPU not available for smoke tests")
    
    # Basic GPU functionality check
    # ... hardware validation with timeout
```

## Benefits of This Approach

### 1. **Speed & Efficiency**

- Unit tests run in <5s each using MockEmbedding
- Integration tests use 80MB models vs 1GB production models
- GPU smoke tests run manually outside CI for targeted validation

### 2. **Reliability**

- MockEmbedding provides deterministic results
- No flaky tests due to model loading failures
- Proper test isolation with cleanup fixtures

### 3. **Developer Experience**

- Fast feedback loop during development
- Clear test categories for different purposes
- Easy to run subset of tests based on need

### 4. **CI/CD Friendly**

- Unit tests run on any machine (CPU-only)
- Integration tests work without GPU
- GPU smoke tests run manually outside CI pipeline

### 5. **Maintainability**

- Uses LlamaIndex built-in mocks (no custom mocking)
- Library-first approach reduces maintenance burden
- Clear separation of concerns by test level

## Migration from Old Tests

### Before (Over-mocking)

```python
@patch("src.utils.embedding.create_embedding_model")
@patch("src.core.llm.LLMManager")
def test_with_excessive_patching(mock_llm, mock_embedding):
    # Too much manual mocking
    mock_embedding.return_value.embed.return_value = [0.1] * 1024
    # ... complex mock setup
```

### After (Proper Boundaries)

```python
@pytest.mark.unit
def test_with_proper_mocking(mock_settings):
    # Use LlamaIndex MockEmbedding
    embedding_model = Settings.embed_model  # Already configured as Mock
    result = embedding_model.get_text_embedding("test")
    # Clean, maintainable test
```

## Performance Targets

- **Unit Tests**: <5 seconds each, 90%+ should be <1s
- **Integration Tests**: <30 seconds each, use <200MB memory
- **GPU Smoke Tests**: Manual execution, hardware-specific validation

## Best Practices

1. **Start with Unit Tests**: Test logic before integration
2. **Mock at Boundaries**: External services, not internal logic  
3. **Use Proper Fixtures**: Leverage session-scoped fixtures for expensive setup
4. **Clear Markers**: Always mark tests appropriately
5. **Cleanup**: Use cleanup fixtures for test isolation
6. **Document Edge Cases**: Test boundary conditions explicitly
7. **Performance Awareness**: Use lightweight models for speed

This framework ensures fast, reliable, maintainable tests that provide confidence in the DocMind AI system while minimizing development friction.
