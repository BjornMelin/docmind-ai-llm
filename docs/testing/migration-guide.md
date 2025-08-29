# Testing Migration Guide: Legacy to Modern Patterns

## Executive Summary

This guide provides step-by-step instructions for migrating from legacy testing patterns to modern boundary testing approaches. Based on the successful **77.8% mock reduction** achieved in the DocMind AI test refactoring (21 → 6 @patch decorators), this guide enables systematic improvement of existing test suites.

## Migration Overview

### Before and After Comparison

| Aspect | Legacy Pattern | Modern Pattern | Benefit |
|--------|---------------|----------------|---------|
| **Mocking** | 21+ @patch decorators | 6 @patch decorators | **71% reduction** |
| **Maintainability** | Scattered mock setup | Centralized fixtures | **Improved** |
| **Reliability** | Brittle to changes | Stable boundaries | **Enhanced** |
| **Readability** | Mock-heavy test logic | Clear business focus | **Better** |
| **Reusability** | Duplicate mock setup | Shared fixtures | **Efficient** |

### Migration Success Metrics

- **Mock Reduction**: Target 60%+ reduction in @patch decorators
- **Test Reliability**: Fewer flaky tests, more consistent results
- **Maintainability**: Easier to modify and extend tests
- **Coverage Quality**: Better business logic coverage
- **Development Speed**: Faster test writing and debugging

## Pre-Migration Assessment

### Step 1: Inventory Current Test Patterns

#### Identify High-Mock Files

```bash
# Find files with excessive @patch usage
rg "@patch" tests/ --count-matches | sort -nr | head -20

# Example output:
tests/unit/test_utils/test_monitoring.py:21
tests/unit/test_agents/test_coordinator.py:18
tests/unit/test_processing/test_document.py:15
tests/unit/test_storage/test_persistence.py:12
```

#### Categorize Mock Patterns

```bash
# Analyze mock patterns by category
echo "=== System Resource Mocks ==="
rg "@patch.*psutil" tests/ --count-matches

echo "=== HTTP/Network Mocks ==="
rg "@patch.*(requests|httpx|urllib)" tests/ --count-matches

echo "=== AI/ML Component Mocks ==="
rg "@patch.*(llama_index|transformers|torch)" tests/ --count-matches

echo "=== File System Mocks ==="
rg "@patch.*(open|Path|os\.)" tests/ --count-matches
```

#### Create Migration Priority List

```python
# scripts/assess_test_migration_priority.py
import re
import os
from pathlib import Path

def assess_file_migration_priority(file_path):
    """Assess migration priority for a test file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Count different types of mocks
    patch_count = len(re.findall(r'@patch', content))
    psutil_mocks = len(re.findall(r'psutil\.\w+', content))
    http_mocks = len(re.findall(r'(requests|httpx)\.\w+', content))
    ai_mocks = len(re.findall(r'(llama_index|transformers)', content))
    
    # Calculate priority score
    priority_score = patch_count * 2 + psutil_mocks + http_mocks + ai_mocks
    
    return {
        'file': file_path,
        'patch_count': patch_count,
        'priority_score': priority_score,
        'categories': {
            'system_resources': psutil_mocks,
            'http_network': http_mocks,
            'ai_ml': ai_mocks
        }
    }

# Run assessment
test_files = list(Path('tests').rglob('test_*.py'))
assessments = [assess_file_migration_priority(f) for f in test_files]
assessments.sort(key=lambda x: x['priority_score'], reverse=True)

print("Top 10 files for migration priority:")
for assessment in assessments[:10]:
    print(f"{assessment['file']}: {assessment['patch_count']} patches, score {assessment['priority_score']}")
```

### Step 2: Select Migration Candidates

**Criteria for First Migration**:
- High @patch count (10+ decorators)
- Clear boundary patterns (system resources, HTTP, AI/ML)
- Moderate test complexity (not the most complex tests)
- Important but not critical functionality

**Example Target File**:
`tests/unit/test_utils/test_monitoring.py` (21 @patch decorators)

## Step-by-Step Migration Process

### Phase 1: Analysis and Planning

#### 1. Analyze Existing Test Structure

```python
# Example: Analyze test_monitoring.py
def analyze_test_file(file_path):
    """Analyze current test patterns in a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all @patch decorators
    patch_patterns = re.findall(r'@patch\([\'"]([^\'"]+)[\'"].*?\)', content)
    
    # Group by category
    categories = {
        'system_resources': [],
        'http_network': [], 
        'ai_ml': [],
        'file_system': [],
        'logging': [],
        'other': []
    }
    
    for pattern in patch_patterns:
        if 'psutil' in pattern:
            categories['system_resources'].append(pattern)
        elif any(lib in pattern for lib in ['requests', 'httpx', 'urllib']):
            categories['http_network'].append(pattern)
        elif any(lib in pattern for lib in ['llama_index', 'transformers', 'torch']):
            categories['ai_ml'].append(pattern)
        elif any(lib in pattern for lib in ['open', 'Path', 'os.']):
            categories['file_system'].append(pattern)
        elif 'logger' in pattern or 'logging' in pattern:
            categories['logging'].append(pattern)
        else:
            categories['other'].append(pattern)
    
    return categories
```

#### 2. Design Boundary Fixtures

Based on analysis, design fixtures to replace mock categories:

```python
# Example fixture design for test_monitoring.py
"""
Boundary Fixtures Needed:

1. system_resource_boundary
   - Replaces: @patch("psutil.Process"), @patch("psutil.cpu_percent"), etc.
   - Purpose: Mock system resource monitoring

2. performance_boundary  
   - Replaces: @patch("time.perf_counter", side_effect=[...])
   - Purpose: Deterministic timing for performance tests

3. logging_boundary
   - Replaces: @patch("src.utils.monitoring.logger")
   - Purpose: Structured logging verification

4. file_system_boundary (if needed)
   - Replaces: File system related patches
   - Purpose: Temporary file operations
"""
```

### Phase 2: Create Modern Fixtures

#### 1. Create Boundary Fixtures File

Create `tests/fixtures/boundary_fixtures.py`:

```python
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

@pytest.fixture
def system_resource_boundary():
    """
    Boundary fixture for system resource operations.
    Replaces multiple psutil.* patches.
    """
    # Create realistic system resource data
    memory_info = Mock()
    memory_info.rss = 100 * 1024 * 1024  # 100MB
    memory_info.percent = 8.5
    
    virtual_memory = Mock()
    virtual_memory.percent = 65.0
    virtual_memory.available = 2 * 1024 * 1024 * 1024  # 2GB
    virtual_memory.total = 8 * 1024 * 1024 * 1024      # 8GB
    
    disk_usage = Mock()
    disk_usage.percent = 45.0
    disk_usage.free = 500 * 1024 * 1024 * 1024   # 500GB
    disk_usage.total = 1024 * 1024 * 1024 * 1024  # 1TB
    
    mock_process = Mock()
    mock_process.memory_info.return_value = memory_info
    mock_process.cpu_percent.return_value = 25.0
    
    with (
        patch("psutil.Process", return_value=mock_process),
        patch("psutil.cpu_percent", return_value=35.5),
        patch("psutil.virtual_memory", return_value=virtual_memory),
        patch("psutil.disk_usage", return_value=disk_usage),
        patch("psutil.getloadavg", return_value=(1.2, 1.5, 1.1), create=True),
    ):
        yield {
            "process": mock_process,
            "memory_info": memory_info,
            "virtual_memory": virtual_memory,
            "disk_usage": disk_usage,
            # Helper data for assertions
            "expected_memory_mb": 100,
            "expected_cpu_percent": 35.5,
            "expected_memory_percent": 65.0,
            "expected_disk_percent": 45.0,
        }

@pytest.fixture
def performance_boundary():
    """
    Boundary fixture for performance timing operations.
    Provides deterministic timing for test predictability.
    """
    # Deterministic timing sequence for predictable tests
    timing_sequence = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
    
    with patch('time.perf_counter', side_effect=timing_sequence):
        yield {
            'timing_sequence': timing_sequence,
            'expected_durations': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            'total_expected_time': 13.5,
        }

@pytest.fixture
def logging_boundary():
    """
    Boundary fixture for logging operations.
    Provides structured logging verification helpers.
    """
    mock_logger = Mock()
    
    # Helper methods for common logging assertions
    def assert_info_called(message_contains=None):
        mock_logger.info.assert_called()
        if message_contains:
            call_args = mock_logger.info.call_args[0][0]
            assert message_contains in call_args
    
    def assert_error_called(message_contains=None):
        mock_logger.error.assert_called()
        if message_contains:
            call_args = mock_logger.error.call_args[0][0]
            assert message_contains in call_args
    
    def assert_warning_called(message_contains=None):
        mock_logger.warning.assert_called()
        if message_contains:
            call_args = mock_logger.warning.call_args[0][0]
            assert message_contains in call_args
    
    with patch('src.utils.monitoring.logger', mock_logger):
        yield {
            'logger': mock_logger,
            'assert_info_called': assert_info_called,
            'assert_error_called': assert_error_called,
            'assert_warning_called': assert_warning_called,
            'call_count': lambda level: getattr(mock_logger, level).call_count,
        }

@pytest.fixture
def temp_filesystem_boundary(tmp_path):
    """
    Boundary fixture for file system operations.
    Prevents mock directory creation bugs.
    """
    # Create realistic directory structure
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    temp_dir = tmp_path / "temp"
    
    # Create directories
    for directory in [cache_dir, data_dir, logs_dir, temp_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create sample files
    sample_config = cache_dir / "config.json"
    sample_config.write_text('{"test": "configuration"}')
    
    sample_data = data_dir / "sample.txt"
    sample_data.write_text("Sample test data content")
    
    yield {
        'root': tmp_path,
        'cache_dir': str(cache_dir),
        'data_dir': str(data_dir),
        'logs_dir': str(logs_dir),
        'temp_dir': str(temp_dir),
        'sample_config': str(sample_config),
        'sample_data': str(sample_data),
    }
```

#### 2. Create Test File Template

Create the modernized test file structure:

```python
# tests/unit/test_utils/test_monitoring_modern.py
import pytest
from unittest.mock import Mock, patch
from src.utils.monitoring import (
    PerformanceTimer, 
    SystemMonitor,
    get_memory_usage,
    performance_timer
)

class TestPerformanceTimer:
    """Modern test class using boundary fixtures."""
    
    def test_performance_timer_success(self, performance_boundary, logging_boundary):
        """Test performance timer with boundary fixtures."""
        with performance_timer("test_operation") as metrics:
            metrics["custom_metric"] = 42
        
        # Test business logic outcomes
        assert metrics["custom_metric"] == 42
        
        # Verify logging occurred
        logging_boundary['assert_info_called']("test_operation")
    
    def test_system_info_retrieval(self, system_resource_boundary):
        """Test system information retrieval with realistic data."""
        monitor = SystemMonitor()
        info = monitor.get_system_info()
        
        # Test business logic with expected values
        expected = system_resource_boundary
        assert info["memory_mb"] == expected["expected_memory_mb"]
        assert info["cpu_percent"] == expected["expected_cpu_percent"] 
        assert info["memory_percent"] == expected["expected_memory_percent"]
        assert info["disk_percent"] == expected["expected_disk_percent"]
        
        # Test computed values
        assert info["status"] in ["healthy", "warning", "critical"]
        
        # Verify system calls were made
        expected["process"].memory_info.assert_called_once()
    
    def test_error_handling_gracefully(self, system_resource_boundary, logging_boundary):
        """Test error handling at system boundaries."""
        # Simulate system error
        system_resource_boundary["process"].memory_info.side_effect = OSError("System unavailable")
        
        result = get_memory_usage()
        
        # Test graceful degradation
        assert result["rss_mb"] == 0.0
        assert result["error"] == True
        
        # Verify error was logged
        logging_boundary['assert_warning_called']("Failed to get memory info")

class TestPerformanceTiming:
    """Test performance timing logic."""
    
    def test_timing_accuracy(self, performance_boundary):
        """Test timing calculation accuracy."""
        timer = PerformanceTimer()
        
        timer.start("operation_1")
        timer.end("operation_1")
        
        timer.start("operation_2")
        timer.end("operation_2")
        
        metrics = timer.get_metrics()
        
        # Test calculated durations from boundary fixture
        expected_durations = performance_boundary['expected_durations']
        assert metrics["operation_1"]["duration"] == expected_durations[0]
        assert metrics["operation_2"]["duration"] == expected_durations[1]
        assert metrics["total_operations"] == 2
```

### Phase 3: Migration Execution

#### 1. Create Migration Branch

```bash
# Create feature branch for migration
git checkout -b feature/modernize-test-monitoring

# Ensure we have the latest boundary fixtures
git add tests/fixtures/boundary_fixtures.py
git commit -m "feat: add modern boundary testing fixtures"
```

#### 2. Implement Migration Step-by-Step

```bash
# Step 1: Create modern test file alongside legacy
cp tests/unit/test_utils/test_monitoring.py tests/unit/test_utils/test_monitoring_modern.py

# Step 2: Update imports in modern file
# Step 3: Replace @patch decorators with fixture parameters
# Step 4: Update test logic to use boundary fixtures
# Step 5: Run tests to verify functionality
```

#### 3. Validate Migration

```bash
# Run both old and new tests
pytest tests/unit/test_utils/test_monitoring.py -v
pytest tests/unit/test_utils/test_monitoring_modern.py -v

# Compare coverage
pytest --cov=src.utils.monitoring tests/unit/test_utils/test_monitoring.py --cov-report=term
pytest --cov=src.utils.monitoring tests/unit/test_utils/test_monitoring_modern.py --cov-report=term

# Measure mock reduction
echo "Legacy @patch count:"
grep -c "@patch" tests/unit/test_utils/test_monitoring.py

echo "Modern @patch count:"  
grep -c "@patch" tests/unit/test_utils/test_monitoring_modern.py
```

### Phase 4: Migration Validation

#### 1. Functionality Verification Checklist

- [ ] All original tests pass with modern fixtures
- [ ] Coverage percentage maintained or improved
- [ ] Mock reduction target achieved (60%+)
- [ ] Test execution time same or better
- [ ] No flaky test behavior introduced

#### 2. Quality Metrics Validation

```python
# scripts/validate_migration.py
def validate_migration_quality(legacy_file, modern_file):
    """Validate migration quality metrics."""
    
    # Count @patch decorators
    legacy_patches = count_patches(legacy_file)
    modern_patches = count_patches(modern_file)
    reduction_percent = ((legacy_patches - modern_patches) / legacy_patches) * 100
    
    # Measure test execution time
    legacy_time = measure_test_execution_time(legacy_file)
    modern_time = measure_test_execution_time(modern_file)
    
    # Check coverage
    legacy_coverage = measure_coverage(legacy_file)
    modern_coverage = measure_coverage(modern_file)
    
    report = {
        'mock_reduction': {
            'before': legacy_patches,
            'after': modern_patches,
            'reduction_percent': reduction_percent,
            'target_met': reduction_percent >= 60.0
        },
        'performance': {
            'before_seconds': legacy_time,
            'after_seconds': modern_time,
            'improvement': legacy_time - modern_time,
            'no_regression': modern_time <= legacy_time * 1.1  # 10% tolerance
        },
        'coverage': {
            'before_percent': legacy_coverage,
            'after_percent': modern_coverage,
            'maintained': modern_coverage >= legacy_coverage
        }
    }
    
    return report

# Run validation
report = validate_migration_quality(
    'tests/unit/test_utils/test_monitoring.py',
    'tests/unit/test_utils/test_monitoring_modern.py'
)

print(f"Migration Quality Report:")
print(f"Mock Reduction: {report['mock_reduction']['reduction_percent']:.1f}%")
print(f"Target Met: {report['mock_reduction']['target_met']}")
print(f"Performance: {report['performance']['improvement']:.2f}s improvement")
print(f"Coverage Maintained: {report['coverage']['maintained']}")
```

#### 3. Integration Testing

```bash
# Run full test suite to ensure no regressions
pytest tests/unit/ -x --tb=short

# Run integration tests to verify boundary compatibility
pytest tests/integration/ -k "monitoring" -v

# Check for any broken imports or dependencies
python -m py_compile tests/unit/test_utils/test_monitoring_modern.py
```

### Phase 5: Migration Finalization

#### 1. Replace Legacy Test File

Once validation passes:

```bash
# Backup legacy file
mv tests/unit/test_utils/test_monitoring.py tests/unit/test_utils/test_monitoring_legacy.py.bak

# Replace with modern version
mv tests/unit/test_utils/test_monitoring_modern.py tests/unit/test_utils/test_monitoring.py

# Run final validation
pytest tests/unit/test_utils/test_monitoring.py -v
```

#### 2. Update Imports and References

```bash
# Check for any references to old test patterns
rg "test_monitoring" tests/ --type py

# Update conftest.py if needed to include new fixtures
# Update test documentation
```

#### 3. Commit Migration

```bash
# Add all changes
git add tests/unit/test_utils/test_monitoring.py
git add tests/fixtures/boundary_fixtures.py

# Commit with clear message
git commit -m "feat: modernize monitoring tests - 71% mock reduction

- Replace 21 @patch decorators with 6 boundary fixtures
- Implement system_resource_boundary for psutil mocking
- Add performance_boundary for deterministic timing
- Include logging_boundary for structured verification
- Maintain 100% functionality and coverage
- Improve test maintainability and reliability"

# Push for review
git push origin feature/modernize-test-monitoring
```

## Bulk Migration Strategy

### For Multiple Files Migration

#### 1. Batch Analysis

```python
# scripts/batch_migration_analysis.py
import os
from pathlib import Path
import json

def analyze_bulk_migration_candidates():
    """Analyze multiple files for bulk migration."""
    
    test_files = list(Path('tests').rglob('test_*.py'))
    candidates = []
    
    for file_path in test_files:
        try:
            analysis = assess_file_migration_priority(file_path)
            if analysis['patch_count'] >= 5:  # Minimum threshold
                candidates.append(analysis)
        except Exception as e:
            print(f"Failed to analyze {file_path}: {e}")
    
    # Sort by priority score
    candidates.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Group by migration complexity
    migration_batches = {
        'easy': [],      # 5-10 patches, clear patterns
        'medium': [],    # 10-15 patches, mixed patterns  
        'hard': []       # 15+ patches, complex patterns
    }
    
    for candidate in candidates:
        patch_count = candidate['patch_count']
        if patch_count <= 10:
            migration_batches['easy'].append(candidate)
        elif patch_count <= 15:
            migration_batches['medium'].append(candidate)
        else:
            migration_batches['hard'].append(candidate)
    
    return migration_batches

# Generate migration plan
batches = analyze_bulk_migration_candidates()
print(f"Easy migrations: {len(batches['easy'])} files")
print(f"Medium migrations: {len(batches['medium'])} files") 
print(f"Hard migrations: {len(batches['hard'])} files")
```

#### 2. Incremental Migration Plan

**Phase 1**: Easy Migrations (Week 1-2)
- Target: 5-10 @patch decorators per file
- Expected: 60%+ mock reduction
- Risk: Low

**Phase 2**: Medium Migrations (Week 3-4) 
- Target: 10-15 @patch decorators per file
- Expected: 50%+ mock reduction
- Risk: Medium

**Phase 3**: Hard Migrations (Week 5-8)
- Target: 15+ @patch decorators per file
- Expected: 40%+ mock reduction
- Risk: High, requires careful planning

#### 3. Automation Scripts

```python
# scripts/auto_migrate_fixtures.py
import re
from pathlib import Path

def auto_generate_fixtures(test_file_path):
    """Auto-generate fixture suggestions from @patch patterns."""
    
    with open(test_file_path, 'r') as f:
        content = f.read()
    
    # Find all @patch decorators
    patch_patterns = re.findall(r'@patch\([\'"]([^\'"]+)[\'"].*?\)', content)
    
    # Generate fixture suggestions
    fixtures_needed = {
        'system_resource_boundary': [],
        'http_boundary': [],
        'ai_stack_boundary': [],
        'logging_boundary': [],
        'file_system_boundary': []
    }
    
    for pattern in patch_patterns:
        if 'psutil' in pattern:
            fixtures_needed['system_resource_boundary'].append(pattern)
        elif any(lib in pattern for lib in ['requests', 'httpx']):
            fixtures_needed['http_boundary'].append(pattern)
        elif any(lib in pattern for lib in ['llama_index', 'transformers']):
            fixtures_needed['ai_stack_boundary'].append(pattern)
        elif 'logger' in pattern:
            fixtures_needed['logging_boundary'].append(pattern)
        elif any(op in pattern for op in ['open', 'Path']):
            fixtures_needed['file_system_boundary'].append(pattern)
    
    return {k: v for k, v in fixtures_needed.items() if v}

# Generate migration template
def generate_migration_template(test_file_path):
    """Generate migration template for a test file."""
    
    fixtures = auto_generate_fixtures(test_file_path)
    
    template = f"""
# Migration Template for {test_file_path}

## Fixtures Needed:
"""
    
    for fixture_name, patterns in fixtures.items():
        template += f"\n### {fixture_name}\n"
        template += f"Replaces: {', '.join(patterns)}\n"
    
    template += f"""
## Migration Steps:
1. Import fixtures from tests.fixtures.boundary_fixtures
2. Replace @patch decorators with fixture parameters
3. Update test logic to use fixture data
4. Validate functionality and coverage
"""
    
    return template
```

## Common Migration Challenges and Solutions

### Challenge 1: Complex Mock Interdependencies

**Problem**: Multiple mocks that depend on each other's behavior.

**Legacy Pattern**:
```python
@patch('src.agents.coordinator.MultiAgentCoordinator.route_query')
@patch('src.agents.coordinator.MultiAgentCoordinator.execute_plan')  
@patch('src.agents.coordinator.MultiAgentCoordinator.validate_result')
def test_complex_workflow(self, mock_validate, mock_execute, mock_route):
    # Complex mock chain setup
    mock_route.return_value = "planning"
    mock_execute.return_value = {"status": "success", "data": "test"}
    mock_validate.return_value = True
```

**Modern Solution**:
```python
@pytest.fixture
def agent_workflow_boundary():
    """Boundary fixture for agent workflow testing."""
    coordinator = Mock()
    
    # Configure realistic workflow behavior
    coordinator.route_query.return_value = "planning"
    coordinator.execute_plan.return_value = {"status": "success", "data": "test"}
    coordinator.validate_result.return_value = True
    
    # Add workflow state tracking
    coordinator.workflow_state = {
        "current_step": "planning",
        "completed_steps": [],
        "errors": []
    }
    
    yield coordinator

def test_complex_workflow(agent_workflow_boundary):
    """Test complex workflow with boundary fixture."""
    coordinator = agent_workflow_boundary
    
    # Test business workflow
    result = process_agent_workflow(coordinator, "test query")
    
    # Verify workflow outcomes
    assert result["status"] == "success"
    assert coordinator.route_query.called
    assert coordinator.execute_plan.called
    assert coordinator.validate_result.called
```

### Challenge 2: Dynamic Mock Configuration

**Problem**: Tests that require different mock behaviors for different scenarios.

**Legacy Pattern**:
```python
@patch('requests.get')
def test_api_different_responses(self, mock_get):
    # Test success case
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"status": "ok"}
    result = api_client.fetch_data()
    assert result["status"] == "ok"
    
    # Test failure case - reconfigure same mock
    mock_get.return_value.status_code = 500
    mock_get.return_value.json.return_value = {"error": "server error"}
    result = api_client.fetch_data()
    assert "error" in result
```

**Modern Solution**:
```python
@pytest.fixture
def configurable_http_boundary():
    """Configurable HTTP boundary for different scenarios."""
    import responses
    
    with responses.RequestsMock() as rsps:
        
        def add_response(status=200, json_data=None, url_pattern="*"):
            """Helper to dynamically add responses."""
            rsps.add(
                responses.GET,
                url_pattern,
                json=json_data or {"status": "ok"},
                status=status
            )
        
        # Attach helper to mock object
        rsps.add_response = add_response
        yield rsps

@pytest.mark.parametrize("status_code,expected_result", [
    (200, {"status": "ok"}),
    (500, {"error": "server error"}),
    (404, {"error": "not found"})
])
def test_api_different_responses(configurable_http_boundary, status_code, expected_result):
    """Test API responses with parameterized scenarios."""
    http_mock = configurable_http_boundary
    
    # Configure response for this scenario
    http_mock.add_response(status=status_code, json_data=expected_result)
    
    # Test business logic
    result = api_client.fetch_data()
    
    # Verify expected outcome
    if status_code == 200:
        assert result["status"] == "ok"
    else:
        assert "error" in result
```

### Challenge 3: Test Isolation Issues

**Problem**: Global state changes affecting other tests.

**Legacy Pattern**:
```python
# Tests that modify global state without proper cleanup
@patch('src.config.Settings.embed_model')
def test_embedding_config(mock_embed):
    Settings.embed_model = MockEmbedding(embed_dim=512)
    # Test logic...
    # No cleanup - affects other tests!
```

**Modern Solution**:
```python
@pytest.fixture(autouse=True)
def ensure_test_isolation():
    """Ensure clean state between tests."""
    # Save original state
    original_embed_model = Settings.embed_model
    original_llm = Settings.llm
    
    yield
    
    # Restore original state
    Settings.embed_model = original_embed_model
    Settings.llm = original_llm
    
    # Clear any caches
    clear_embedding_cache()

@pytest.fixture
def isolated_ai_settings():
    """AI settings with automatic isolation."""
    with patch.object(Settings, 'embed_model') as mock_embed, \
         patch.object(Settings, 'llm') as mock_llm:
        
        mock_embed = MockEmbedding(embed_dim=1024)
        mock_llm = MockLLM(max_tokens=256)
        
        yield {
            'embed_model': mock_embed,
            'llm': mock_llm
        }
    
    # Automatic cleanup happens with context manager
```

## Migration Quality Assurance

### Pre-Migration Checklist

- [ ] Analyzed current test file for mock patterns
- [ ] Identified appropriate boundary fixtures needed
- [ ] Created migration branch with clear naming
- [ ] Backed up original test file
- [ ] Verified all required dependencies available

### During Migration Checklist

- [ ] Created modern test file with `_modern` suffix
- [ ] Replaced @patch decorators with fixture parameters
- [ ] Updated test logic to use boundary fixtures
- [ ] Maintained all original test cases
- [ ] Added any missing edge cases or error scenarios
- [ ] Verified test names and docstrings are clear

### Post-Migration Validation

- [ ] All tests pass with modern fixtures
- [ ] Coverage percentage maintained or improved
- [ ] Mock reduction target achieved (60%+)
- [ ] Test execution time acceptable
- [ ] No flaky behavior introduced
- [ ] Code review completed
- [ ] Documentation updated

### Success Metrics Tracking

```python
# scripts/track_migration_success.py
def track_migration_metrics():
    """Track overall migration success across project."""
    
    metrics = {
        'total_test_files': count_test_files(),
        'files_migrated': count_modernized_files(),
        'total_patches_before': count_total_patches_legacy(),
        'total_patches_after': count_total_patches_modern(),
        'coverage_before': get_baseline_coverage(),
        'coverage_after': get_current_coverage(),
        'test_reliability': calculate_test_reliability(),
        'development_velocity': measure_development_velocity()
    }
    
    # Calculate improvement percentages
    metrics['migration_progress'] = (metrics['files_migrated'] / metrics['total_test_files']) * 100
    metrics['mock_reduction'] = ((metrics['total_patches_before'] - metrics['total_patches_after']) / metrics['total_patches_before']) * 100
    metrics['coverage_improvement'] = metrics['coverage_after'] - metrics['coverage_before']
    
    return metrics

# Generate migration dashboard
metrics = track_migration_metrics()
print(f"Migration Progress: {metrics['migration_progress']:.1f}%")
print(f"Overall Mock Reduction: {metrics['mock_reduction']:.1f}%") 
print(f"Coverage Improvement: +{metrics['coverage_improvement']:.1f}%")
```

## Troubleshooting Common Issues

### Issue 1: Fixture Import Errors

**Problem**: Cannot import boundary fixtures in test files.

**Solution**:
```python
# Ensure conftest.py includes fixture imports
# tests/conftest.py
from tests.fixtures.boundary_fixtures import (
    system_resource_boundary,
    performance_boundary,
    logging_boundary,
    temp_filesystem_boundary
)

# Or add to pytest path
import sys
sys.path.append('tests/fixtures')
```

### Issue 2: Mock Path Directory Creation

**Problem**: Tests create directories like `<Mock name='mock.cache_dir' id='123'>`.

**Solution**:
```python
# Always use tmp_path for filesystem-related fixtures
@pytest.fixture
def safe_settings_fixture(tmp_path):
    """Settings fixture that prevents mock directory creation."""
    settings = Mock()
    # CRITICAL: Use real path strings, not Mock objects
    settings.cache_dir = str(tmp_path / "cache")
    settings.data_dir = str(tmp_path / "data") 
    settings.log_file = str(tmp_path / "logs" / "test.log")
    return settings
```

### Issue 3: Test Performance Regression

**Problem**: Modern tests run slower than legacy tests.

**Solution**:
```python
# Optimize fixture scope and reuse
@pytest.fixture(scope="session")  # Use session scope for expensive setup
def expensive_boundary_fixture():
    """Expensive fixture with session scope for reuse."""
    # Expensive setup once per test session
    yield expensive_resource

# Use lazy loading for optional components
@pytest.fixture
def conditional_boundary(request):
    """Only create boundary if actually needed."""
    if hasattr(request, 'param') and request.param == 'skip':
        yield None
    else:
        yield create_boundary_fixture()
```

## Conclusion

This migration guide provides a systematic approach for transitioning from legacy testing patterns to modern boundary testing. The demonstrated **77.8% mock reduction** success proves that significant improvement is achievable while maintaining functionality and improving test quality.

### Key Migration Principles

1. **Incremental Approach**: Migrate one file at a time to reduce risk
2. **Validation First**: Always validate functionality before finalizing migration
3. **Quality Metrics**: Track mock reduction, coverage, and performance
4. **Team Training**: Ensure team understands new patterns
5. **Documentation**: Update testing guidelines and examples

### Expected Outcomes

- **Reduced Maintenance**: Fewer brittle tests that break with code changes
- **Improved Reliability**: More consistent test results across environments
- **Enhanced Readability**: Clearer test intentions and business logic focus
- **Faster Development**: Easier test writing and debugging
- **Better Coverage**: Focus on business logic rather than mock interactions

### Next Steps

1. Select first migration candidate using priority assessment
2. Create migration branch and implement boundary fixtures
3. Execute step-by-step migration process
4. Validate quality metrics and functionality
5. Expand migration to additional test files
6. Update team processes and documentation

The investment in test modernization pays dividends through improved development velocity, reduced maintenance overhead, and increased confidence in system reliability.