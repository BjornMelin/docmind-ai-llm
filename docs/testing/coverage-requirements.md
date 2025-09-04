# Test Coverage: How-To and Expectations

## Coverage Commands

### Development coverage check

```bash
# Quick coverage check for development
uv run python -m pytest tests/unit/config/test_settings.py --cov=src --cov-report=term-missing

# Coverage for specific module
uv run python -m pytest tests/unit/test_agents/ --cov=src/agents --cov-report=term-missing

# HTML coverage report for detailed analysis
uv run python -m pytest tests/unit/ --cov=src --cov-report=html
```

### CI coverage example

```bash
# Coverage with minimum threshold enforcement
uv run python -m pytest tests/unit/ tests/integration/ \
    --cov=src \
    --cov-fail-under=35 \
    --cov-report=term-missing \
    --cov-report=xml

# Module-specific coverage requirements
uv run python -m pytest tests/unit/test_agents/ \
    --cov=src/agents \
    --cov-fail-under=40
```

### .coveragerc example

```ini
[run]
source = src/
omit = 
    */tests/*
    */venv/*
    */env/*
    */__pycache__/*
    */migrations/*
    */settings/local.py
    */manage.py
    */wsgi.py
    */asgi.py

[report]
# Regexes for lines to exclude from consideration
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:
    class .*\bProtocol\):
    @(abc\.)?abstractmethod

precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
title = DocMind AI Coverage Report

[xml]
output = coverage.xml
```

### Coverage Quality Gates

#### Minimum Requirements for PR Approval

```yaml
# Quality Gates for CI/CD
coverage_requirements:
  overall_minimum: 35.0%      # Increased from current 3.51%
  
  module_requirements:
    critical_modules:
      minimum: 40.0%
      modules:
        - src/agents/
        - src/processing/ 
        - src/core/
    
    core_modules:
      minimum: 35.0%
      modules:
        - src/retrieval/
        - src/utils/
    
    models:
      minimum: 80.0%
      modules:
        - src/models/
    
    configuration:
      minimum: 60.0%
      modules:
        - src/config/
    
    infrastructure:
      minimum: 25.0%
      modules:
        - src/core/infrastructure/
```

#### Coverage Trend Monitoring

```bash
# Track coverage over time
echo "$(date): $(coverage report --format=total)" >> coverage_history.txt

# Coverage trend analysis
python scripts/analyze_coverage_trends.py coverage_history.txt
```

## Coverage Roadmap and Milestones

### Phase 1: Foundation (Months 1-2)

**Target**: 15% overall coverage

**Priority Modules**:

1. **Models and Schemas** (80% target)
   - High value, low complexity
   - Foundation for other tests
   - Quick wins for coverage percentage

2. **Configuration** (60% target)  
   - Build on existing 53.70% - 98.67% coverage
   - Critical for system reliability
   - Relatively straightforward testing

**Deliverables**:

- Complete model validation testing
- Configuration edge case coverage
- Basic CI/CD coverage enforcement

### Phase 2: Core Functionality (Months 3-4)  

**Target**: 25% overall coverage

**Priority Modules**:

1. **Agent System** (40% target)
   - Critical business functionality
   - Multi-agent coordination testing
   - Error handling and recovery

2. **Document Processing** (40% target)
   - Core data pipeline
   - Document parsing and chunking
   - Embedding generation consistency

**Deliverables**:

- Agent coordination test suite
- Document processing pipeline tests
- Integration test framework

### Phase 3: System Integration (Months 5-6)

**Target**: 35% overall coverage

**Priority Modules**:

1. **Storage and Retrieval** (35% target)
   - Data persistence integrity
   - Vector store operations
   - Query processing accuracy

2. **Utility Functions** (35% target)
   - Supporting functionality
   - System monitoring
   - Performance utilities

**Deliverables**:

- Complete storage test coverage
- Utility function test suite
- Performance regression tests

### Phase 4: Production Readiness (Months 7+)

**Target**: 50%+ overall coverage

**Focus Areas**:

- Infrastructure monitoring (25% target)
- Advanced error scenarios
- Performance benchmarking
- Security boundary testing

## Test Strategy by Coverage Target

### High Coverage Modules (80%+): Models and Configuration

**Strategy**: Comprehensive validation testing

```python
@pytest.mark.unit
@pytest.mark.models
@pytest.mark.parametrize("invalid_input,expected_error", [
    ("", ValidationError),
    (None, ValidationError), 
    (123, TypeError),
    ({"invalid": "structure"}, ValidationError)
])
def test_model_validation_comprehensive(invalid_input, expected_error):
    """Comprehensive model validation testing."""
    with pytest.raises(expected_error):
        DocumentModel.validate(invalid_input)

# Property-based testing for edge cases
@given(st.text(min_size=0, max_size=10000))
def test_model_text_field_boundaries(text_input):
    """Property-based testing for text field validation."""
    model = DocumentModel(content=text_input)
    assert model.content == text_input
    assert len(model.content) <= 10000
```

### Medium Coverage Modules (35-40%): Core Business Logic

**Strategy**: Business logic and integration testing

```python
@pytest.mark.integration
@pytest.mark.agents
async def test_agent_coordination_workflow(ai_stack_boundary, temp_settings):
    """Test complete agent coordination workflow."""
    coordinator = MultiAgentCoordinator(settings=temp_settings)
    
    # Test successful coordination
    result = await coordinator.process_query(
        query="Analyze the document for key insights",
        document="Sample document content for analysis"
    )
    
    assert result.status == "completed"
    assert result.confidence > 0.7
    assert len(result.insights) > 0
    
    # Test error recovery
    with patch('src.agents.retrieval.RetrievalAgent.process') as mock_retrieval:
        mock_retrieval.side_effect = Exception("Retrieval service unavailable")
        
        result = await coordinator.process_query_with_fallback("test query")
        assert result.status == "completed"
        assert result.fallback_used == True
```

### Low Coverage Modules (25%): Infrastructure and Utilities

**Strategy**: Critical path and error boundary testing

```python
@pytest.mark.unit
@pytest.mark.infrastructure
def test_gpu_monitor_critical_paths(system_resource_boundary):
    """Test GPU monitoring critical functionality only."""
    monitor = GPUMonitor()
    
    # Test basic monitoring functionality
    status = monitor.get_gpu_status()
    assert status.available in [True, False]
    
    # Test error handling for GPU not available
    with patch('torch.cuda.is_available', return_value=False):
        status = monitor.get_gpu_status()
        assert status.available == False
        assert status.memory_used == 0
        assert "GPU not available" in status.messages
```

## Coverage Analysis and Quality Metrics

### Coverage Quality Assessment

#### Line Coverage vs Branch Coverage

```bash
# Measure both line and branch coverage
uv run python -m pytest --cov=src --cov-branch --cov-report=term-missing

# Branch coverage is typically 10-15% lower than line coverage
# Target: Line coverage 35%+, Branch coverage 25%+
```

#### Function Coverage Analysis

```python
# Custom coverage analysis for function coverage
def analyze_function_coverage(coverage_data):
    """Analyze function-level coverage for quality assessment."""
    functions_tested = 0
    total_functions = 0
    
    for module in coverage_data:
        for function in module.functions:
            total_functions += 1
            if function.lines_covered > 0:
                functions_tested += 1
    
    function_coverage = (functions_tested / total_functions) * 100
    return function_coverage

# Target: 30%+ function coverage alongside line coverage
```

### Coverage Quality Indicators

#### High-Quality Coverage Characteristics

- **Business Logic Focus**: Tests cover actual business rules, not just getters/setters
- **Error Path Coverage**: Exception handling and edge cases tested
- **Integration Points**: Module boundaries and interfaces tested
- **Realistic Scenarios**: Tests use production-like data and workflows

#### Coverage Anti-Patterns to Avoid

- **Vanity Coverage**: High percentage but only testing trivial code paths
- **Mock-Heavy Tests**: Tests that primarily verify mock interactions
- **Brittle Tests**: Tests that break with minor refactoring
- **Slow Tests**: High coverage achieved through slow, complex tests

### Coverage Reporting and Monitoring

#### Automated Coverage Reports

```python
# Custom coverage report generator
def generate_coverage_report():
    """Generate detailed coverage report with actionable insights."""
    
    coverage_data = load_coverage_data()
    
    report = {
        "overall": {
            "current": coverage_data.overall_percentage,
            "target": 35.0,
            "status": "✅" if coverage_data.overall_percentage >= 35.0 else "❌"
        },
        "modules": {},
        "critical_gaps": [],
        "recommendations": []
    }
    
    # Analyze module-level coverage
    for module, data in coverage_data.modules.items():
        target = get_module_target_coverage(module)
        
        report["modules"][module] = {
            "current": data.percentage,
            "target": target,
            "lines_missing": data.lines_missing,
            "status": "✅" if data.percentage >= target else "❌"
        }
        
        # Identify critical gaps
        if data.percentage < target and is_critical_module(module):
            report["critical_gaps"].append({
                "module": module,
                "gap": target - data.percentage,
                "priority": get_module_priority(module)
            })
    
    # Generate recommendations
    report["recommendations"] = generate_coverage_recommendations(report["critical_gaps"])
    
    return report
```

#### Coverage Dashboard Integration

```yaml
# Coverage dashboard configuration
coverage_dashboard:
  metrics:
    - overall_percentage
    - module_coverage_by_priority
    - critical_gaps_count
    - coverage_trend_7_days
    - test_execution_time
  
  alerts:
    - threshold: 35.0
      condition: below
      severity: warning
      message: "Overall coverage below minimum threshold"
    
    - module: src/agents/
      threshold: 40.0
      condition: below
      severity: critical
      message: "Critical module coverage insufficient"
  
  reports:
    - type: weekly_summary
    - type: critical_gaps
    - type: coverage_trends
```

## Implementation Guidelines

### Test Writing Priorities

1. **Start with Models** (80% target)
   - Quick wins for coverage metrics
   - Foundation for other tests
   - High value, low complexity

2. **Focus on Critical Business Logic** (40% target)
   - Agent coordination workflows
   - Document processing pipelines
   - Error handling and recovery

3. **Cover Integration Points** (35% target)
   - Module boundaries
   - External service interfaces
   - Data persistence operations

4. **Add Infrastructure Coverage** (25% target)
   - System monitoring
   - Resource management
   - Performance utilities

### Coverage Improvement Process

#### Monthly Coverage Reviews

1. **Measure Current Coverage**

   ```bash
   uv run python -m pytest --cov=src --cov-report=html
   ```

2. **Identify Priority Gaps**
   - Critical modules below 40%
   - Core modules below 35%
   - Models below 80%

3. **Plan Coverage Improvements**
   - Select 2-3 modules for focus
   - Estimate effort and timeline
   - Assign development resources

4. **Implement and Validate**
   - Write tests following established patterns
   - Validate coverage increases
   - Update quality gates

#### Quality Assurance Checklist

- [ ] New code includes corresponding tests
- [ ] Coverage percentage maintained or improved
- [ ] Critical modules meet minimum thresholds
- [ ] Tests follow established patterns (boundary testing, library-first)
- [ ] CI/CD pipeline enforces coverage requirements
- [ ] Coverage reports generated and reviewed

## Conclusion

The coverage requirements defined in this document provide a practical roadmap for improving test coverage from the current **3.51%** to production-ready levels of **35%+ overall** with module-specific targets based on criticality and business risk.

**Success Metrics**:

- **Phase 1**: 15% overall coverage (Models and Configuration)
- **Phase 2**: 25% overall coverage (+ Agent System and Document Processing)  
- **Phase 3**: 35% overall coverage (+ Storage, Retrieval, and Utilities)
- **Phase 4**: 50%+ overall coverage (+ Infrastructure and Advanced Testing)

**Key Principles**:

- **Risk-Based Coverage**: Higher requirements for critical business logic
- **Practical Targets**: Achievable milestones based on module complexity
- **Quality Focus**: Emphasis on valuable tests, not just percentage coverage
- **Continuous Improvement**: Monthly reviews and iterative enhancement

This approach ensures steady progress toward production-ready test coverage while maintaining development velocity and focusing effort on the most critical system components.
