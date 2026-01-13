# DocMind AI Quality Gates Infrastructure

This directory contains comprehensive test quality gates and monitoring infrastructure to maintain code quality and prevent regression.

## Quality Gate Scripts

### 1. Coverage Threshold Checker (`check_coverage.py`)

Target gates: 80% line and 75% branch coverage with detailed reporting. Override thresholds on-demand with `--threshold`/`--branch-threshold`.

```bash
# Basic usage
uv run python scripts/check_coverage.py --threshold 80 --fail-under

# Generate detailed report with HTML output
uv run python scripts/check_coverage.py --collect --report --html

# Check only new/modified code
uv run python scripts/check_coverage.py --new-code-only --diff-from main
```

**Features:**

- Line and branch coverage validation (targets)  
- New code coverage tracking
- Detailed HTML reports
- CI/CD integration support
- File-by-file analysis

### 2. Performance Regression Detector (`performance_monitor.py`)

Monitors test execution times and detects >20% performance degradations.

```bash
# Run performance monitoring with regression detection
uv run python scripts/performance_monitor.py --run-tests --check-regressions

# Establish performance baseline
uv run python scripts/performance_monitor.py --run-tests --baseline

# Quick collection time check
uv run python scripts/performance_monitor.py --collection-only
```

**Features:**

- Test execution time monitoring
- Performance baseline tracking  
- Regression alerts (>20% degradation)
- Trend analysis and reporting
- Integration with existing performance tests

Regression checks compare durations and memory (CPU peak, GPU VRAM peak when available) against the recorded baseline.

### 3. Test Suite Health Monitor (`test_health.py`)

Monitors flaky test patterns, anti-patterns, and test execution stability.

```bash
# Comprehensive health analysis
uv run python scripts/test_health.py --analyze --runs 10

# Check for code anti-patterns
uv run python scripts/test_health.py --patterns --test-dirs tests/

# Monitor test stability
uv run python scripts/test_health.py --stability --days 7
```

**Features:**

- Flaky test detection (80% pass rate threshold)
- Anti-pattern identification (sleep usage, hardcoded paths, etc.)
- Test execution stability tracking
- Health reports with recommendations
- CI/CD pipeline integration

### 4. Unified Quality Gates Runner (`run_quality_gates.py`)

Orchestrates all quality gates with comprehensive reporting.

```bash
# Run all quality gates
uv run python scripts/run_quality_gates.py --all

# Quick validation (coverage only)
uv run python scripts/run_quality_gates.py --quick

# CI/CD pipeline (coverage + performance)
uv run python scripts/run_quality_gates.py --ci

# Individual gates
uv run python scripts/run_quality_gates.py --coverage --performance --health

# Include pre-commit hooks
uv run python scripts/run_quality_gates.py --all --pre-commit
```

## Configuration Files

### pytest.ini

Updated with three-tier testing strategy and quality gate markers:

```ini
# Three-Tier Testing Strategy
unit: Fast unit tests with mocked dependencies (<5s each)
integration: Component interaction tests (<30s each)  
system: Full system tests with real models and GPU (<5min each)

# Quality Gate Markers
performance: Performance benchmarks and regression tests
benchmark: Benchmark tests for performance tracking
flaky: Tests known to be flaky (for health monitoring)
slow: Tests taking longer than expected
quality_gate: Tests that enforce quality standards
coverage_critical: Tests critical for coverage thresholds
```

### pyproject.toml

Enhanced with comprehensive coverage configuration and quality dependencies:

```toml
[tool.coverage.run]
source = ["src"]
branch = true
fail_under = 80

[tool.coverage.report]
show_missing = true
skip_covered = false
sort = "Cover"

[dependency-groups.quality]
pre-commit = ">=4.0.1"
coverage = ">=7.6.0" 
pytest-xdist = ">=3.6.0"  # Parallel execution
pytest-timeout = ">=2.3.1"  # Timeout enforcement
```

### .pre-commit-config.yaml  

Comprehensive pre-commit hooks including custom quality gates:

- **Code Quality**: ruff, black, isort, mypy
- **Security**: bandit scanning  
- **Documentation**: docstring validation
- **Custom Gates**: coverage checks, performance monitoring, test health
- **Integration**: test discovery validation

## Usage Examples

### Development Workflow

```bash
# Before committing
uv run python scripts/run_quality_gates.py --quick
pre-commit run --all-files

# Before pushing  
uv run python scripts/run_quality_gates.py --ci --report

# Full validation
uv run python scripts/run_quality_gates.py --all --pre-commit --report
```

### CI/CD Integration

```bash
# Fast CI validation
uv run python scripts/run_quality_gates.py --ci --continue-on-failure

# Staging validation
uv run python scripts/run_quality_gates.py --all --report

# Performance baseline update
uv run python scripts/performance_monitor.py --run-tests --baseline --save
```

### Troubleshooting

```bash
# Identify flaky tests
uv run python scripts/test_health.py --flakiness --runs 10 --report

# Debug coverage issues  
uv run python scripts/check_coverage.py --collect --report --new-code-only

# Monitor performance trends
uv run python scripts/performance_monitor.py --report --days 30
```

## Success Criteria

✅ **Coverage**: >80% line coverage, >75% branch coverage  
✅ **Performance**: <20% regression in test execution  
✅ **Health**: <5 flaky tests, minimal anti-patterns  
✅ **Integration**: All pre-commit hooks pass  
✅ **Stability**: >95% test pass rate consistently  

## Integration Points

- **Pre-commit hooks**: Automatic quality validation on commit
- **CI/CD pipelines**: Enforces quality gates before merge
- **Performance tracking**: Continuous monitoring and alerting  
- **Health monitoring**: Proactive test suite maintenance
- **Coverage reporting**: Detailed HTML reports with trend analysis

This infrastructure ensures ongoing test quality maintenance and prevents future regression in the modernized test suite.
