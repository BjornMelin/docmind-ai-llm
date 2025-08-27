# Multi-Agent Coordination System Tests

This directory contains comprehensive pytest tests for the Multi-Agent Coordination System (FEAT-001) based on the Gherkin scenarios from the specification.

## Test Overview

The test suite provides comprehensive coverage of the multi-agent coordination system with:

- **11 test classes** covering all system components
- **33+ test methods** validating functionality
- **Mock implementations** for deterministic testing
- **Performance benchmarks** for latency requirements
- **Error handling** and recovery scenarios
- **Async operation** support with proper timeout handling

## Test Structure

### Unit Tests

#### `TestRouterAgent`

- Query complexity classification (simple vs complex)
- Routing strategy selection (vector vs hybrid)
- Classification pattern validation across query types

#### `TestPlannerAgent`

- Complex query decomposition into sub-tasks
- Task structure validation and ordering
- Planning bypass for simple queries

#### `TestRetrievalAgent`

- Vector search strategy for simple queries
- Hybrid search strategy for complex queries  
- DSPy optimization with latency constraints
- Concurrent sub-task processing

#### `TestSynthesisAgent`

- Multi-source result combination
- Processing metadata tracking
- Synthesis bypass for simple queries

#### `TestValidationAgent`

- Response quality validation
- Scoring and confidence assessment
- Hallucination detection capabilities

### Integration Tests

#### `TestMultiAgentIntegration`

Tests all 5 Gherkin scenarios from the specification:

1. **Simple Query Processing** - Router → Vector Search → Response (< 1.5s)
2. **Complex Query Decomposition** - Router → Planner → Retrieval → Synthesis → Validation
3. **Fallback on Agent Failure** - Graceful degradation to basic RAG (< 3s)
4. **Context Preservation** - Multi-turn conversation with 65K token limit
5. **DSPy Optimization** - Query rewriting with < 100ms latency overhead

### Performance Tests

#### `TestPerformanceRequirements`

- Agent coordination overhead measurement (< 300ms target)
- Concurrent query processing validation
- Memory usage constraint verification

### Resilience Tests

#### `TestErrorHandlingAndRecovery`

- Agent timeout handling with graceful fallbacks
- Invalid input validation and sanitization
- Partial agent failure recovery mechanisms
- Context buffer overflow management

#### `TestContextManagement`

- Multi-turn conversation continuity
- Token limit enforcement and truncation
- Context preservation across agent pipeline

#### `TestAsyncOperations`

- Streaming response generation
- Concurrent agent coordination
- Timeout protection for long-running operations

### Compliance Tests

#### `TestSpecificationCompliance`

- Gherkin scenario coverage verification
- Performance requirement compliance
- Complete agent pipeline validation

## Mock Implementation

The test suite uses sophisticated mock classes that simulate real agent behavior:

### `MockMultiAgentCoordinator`

- Realistic query routing logic
- Simulated planning and decomposition
- Mock retrieval with multiple strategies
- Synthesis result combination
- Validation scoring and quality assessment

### `MockAgentResponse`

- Complete response structure with content, sources, metadata
- Processing time and validation score tracking
- Source attribution and relevance scoring

## Running the Tests

### Prerequisites

```bash
# Install dependencies
uv add pytest pytest-asyncio pytest-benchmark pytest-cov

# Ensure test markers are configured in pytest.ini
```

### Execute Tests

```bash
# Run all multi-agent tests
pytest tests/test_agents/test_multi_agent_coordination_spec.py -v

# Run specific test categories
pytest tests/test_agents/test_multi_agent_coordination_spec.py::TestRouterAgent -v
pytest tests/test_agents/test_multi_agent_coordination_spec.py::TestMultiAgentIntegration -v

# Run tests by marker
pytest -m "spec and FEAT-001" -v
pytest -m "integration" -v  
pytest -m "performance" -v
pytest -m "asyncio" -v

# Run with coverage
pytest tests/test_agents/test_multi_agent_coordination_spec.py --cov=src/agents --cov-report=html

# Run performance benchmarks
pytest -m performance --benchmark-only -v
```

### Test Markers

The test suite uses pytest markers for organization:

- `@pytest.mark.spec("FEAT-001")` - Specification compliance tests
- `@pytest.mark.integration` - Full pipeline integration tests
- `@pytest.mark.performance` - Performance and latency tests
- `@pytest.mark.asyncio` - Asynchronous operation tests

## Test Configuration

### Fixtures

The test suite provides comprehensive fixtures:

- `mock_coordinator` - Fully functional mock multi-agent coordinator
- `sample_context` - Pre-populated conversation context for testing
- `complex_query` / `simple_query` - Representative test queries
- `dspy_settings` - DSPy optimization configuration
- `spec_test_cases` - Mapping of Gherkin scenarios to test parameters

### Performance Monitoring

Tests include built-in performance monitoring:

- Processing time measurement and validation
- Memory usage tracking and constraints
- Latency requirement verification
- Throughput and concurrency testing

## Validation

Run the validation script to verify test completeness:

```bash
python3 test_validation_simple.py
```

This validates:

- All required test classes are present
- Gherkin scenarios are covered
- Pytest markers are properly used
- Mock implementations are complete
- Async support is implemented
- Specification features are tested

## Expected Outcomes

When run successfully, the tests validate:

### Functional Requirements

- ✅ Router correctly classifies query complexity
- ✅ Planner decomposes complex queries into 3+ sub-tasks
- ✅ Retrieval uses appropriate search strategies
- ✅ Synthesis combines multi-source results
- ✅ Validation ensures response quality

### Performance Requirements

- ✅ Simple queries process under 1.5 seconds
- ✅ Agent coordination overhead under 300ms
- ✅ Fallback responses under 3 seconds
- ✅ DSPy optimization under 100ms latency
- ✅ Context management within 65K tokens

### Quality Requirements

- ✅ 90%+ validation scores for quality responses
- ✅ Graceful error handling and recovery
- ✅ Memory usage within constraints
- ✅ Deterministic and reproducible results

## Integration with CI/CD

The test suite is designed for automated testing:

```yaml
# Example GitHub Actions workflow
- name: Run Multi-Agent Tests
  run: |
    pytest tests/test_agents/test_multi_agent_coordination_spec.py \
      --junitxml=test-results.xml \
      --cov=src/agents \
      --cov-report=xml
```

## Future Enhancements

Planned test improvements:

- Real LLM integration tests (optional)
- End-to-end system integration
- Load testing with high concurrency
- Security and prompt injection testing
- Multi-language query testing
