# DocMind AI Maintenance Procedures

## Overview

This document provides operational maintenance procedures for the DocMind AI unified architecture. Follow these procedures to maintain system health, perform updates, and troubleshoot issues while preserving the architectural integrity achieved through the configuration refactoring.

## Table of Contents

1. [Configuration Management](#configuration-management)
2. [Code Quality Maintenance](#code-quality-maintenance)
3. [Performance Monitoring](#performance-monitoring)
4. [Dependency Management](#dependency-management)
5. [Backup and Recovery](#backup-and-recovery)
6. [Security Maintenance](#security-maintenance)
7. [Troubleshooting Procedures](#troubleshooting-procedures)

## Configuration Management

### Configuration Update Process

**CRITICAL**: All configuration changes MUST go through the unified settings architecture.

#### Adding New Configuration

```python
# 1. Add to appropriate nested model in src/config/settings.py
class ProcessingConfig(BaseModel):
    """Document processing configuration."""
    
    # Existing fields...
    chunk_size: int = Field(default=1500, ge=100, le=10000)
    
    # NEW FIELD - Add with proper validation
    new_feature_enabled: bool = Field(default=False)
    new_parameter: int = Field(default=100, ge=1, le=1000)
```

```bash
# 2. Add corresponding environment variable
echo "DOCMIND_PROCESSING__NEW_FEATURE_ENABLED=true" >> .env
echo "DOCMIND_PROCESSING__NEW_PARAMETER=250" >> .env
```

```python  
# 3. Update tests
def test_new_configuration():
    """Test new configuration parameter."""
    assert settings.processing.new_feature_enabled is not None
    assert 1 <= settings.processing.new_parameter <= 1000
```

#### Environment Variable Naming Convention

**MANDATORY**: Follow DOCMIND_* pattern with nested delimiter:

```bash
# Correct patterns
DOCMIND_DEBUG=true                           # Top-level setting
DOCMIND_VLLM__MODEL=custom-model            # Nested: vllm.model  
DOCMIND_EMBEDDING__MODEL_NAME=custom-embed  # Nested: embedding.model_name
DOCMIND_AGENTS__DECISION_TIMEOUT=300        # Nested: agents.decision_timeout

# NEVER use these patterns
VLLM_MODEL=custom-model                      # Missing DOCMIND_ prefix
DOCMIND_VLLM_MODEL=custom-model             # Wrong delimiter (use __)
docmind_debug=true                           # Wrong case (use UPPER)
```

#### Configuration Validation Checklist

Before deploying configuration changes:

```bash
# 1. Validate configuration loading
python -c "from src.config import settings; print('✅ Configuration loads successfully')"

# 2. Check ADR compliance
python scripts/validate_adr_compliance.py

# 3. Run configuration tests
uv run python -m pytest tests/unit/test_config_validation.py -v

# 4. Verify environment variable mapping  
python -c "
from src.config import settings
print(f'Agent timeout: {settings.agents.decision_timeout}ms')
print(f'BGE-M3 model: {settings.embedding.model_name}')
print(f'vLLM model: {settings.vllm.model}')
"
```

### Configuration Rollback Procedure

If configuration changes cause issues:

```bash
# 1. Identify problematic environment variables
env | grep DOCMIND_ | sort

# 2. Remove problematic variables
unset DOCMIND_PROBLEMATIC_SETTING

# 3. Restart application with defaults
uv run python -c "from src.config import settings; settings = DocMindSettings()"

# 4. Verify system health
python scripts/performance_validation.py
```

## Code Quality Maintenance

### Pre-Commit Quality Gates

**MANDATORY**: Run before every commit:

```bash
# 1. Format code
ruff format src tests

# 2. Fix linting issues
ruff check src tests --fix

# 3. Validate no remaining issues
ruff check src tests
# Must show: "All checks passed!"

# 4. Verify import patterns
python -c "
import ast
import glob

def check_config_imports():
    for file in glob.glob('src/**/*.py', recursive=True):
        with open(file) as f:
            try:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and 'config' in node.module:
                            print(f'{file}: {ast.unparse(node)}')
            except:
                pass

check_config_imports()
"
```

### Quality Score Maintenance

**TARGET**: Maintain Pylint score ≥9.5/10

```bash
# Monthly quality assessment
pylint src/**/*.py --score=y

# Address quality issues by priority:
# 1. Fix any score drops below 9.5
# 2. Eliminate any newly introduced warnings
# 3. Refactor complex functions (>15 lines)
# 4. Improve docstring coverage
```

### Import Pattern Enforcement

**MANDATORY**: Always use unified import pattern:

```python
# CORRECT - Single source of truth
from src.config import settings

# Access nested configuration
model_name = settings.vllm.model
embedding_model = settings.embedding.model_name
chunk_size = settings.processing.chunk_size

# NEVER import nested models directly
from src.config.settings import VLLMConfig  # ❌ Wrong
from src.config import VLLMConfig          # ❌ Wrong  
```

### Code Review Checklist

Before merging any code changes:

- [ ] All imports use `from src.config import settings`
- [ ] No backwards compatibility code exists
- [ ] Ruff shows zero issues
- [ ] Pylint score remains ≥9.5/10
- [ ] All tests pass with new changes
- [ ] Configuration patterns follow ADR-024
- [ ] No over-engineering or unnecessary complexity

## Performance Monitoring

### System Health Checks

**Daily monitoring routine:**

```bash
# 1. GPU utilization and memory
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# 2. Application performance
uv run python scripts/performance_validation.py

# 3. Agent decision latency
python -c "
import time
from src.agents.coordinator import get_agent_system
from src.config import settings

# Simple latency test
start = time.time()
# Simulate agent decision
latency = (time.time() - start) * 1000
print(f'Agent decision latency: {latency:.1f}ms')
assert latency < settings.agents.decision_timeout
"

# 4. Memory usage trends
ps aux | grep python | awk '{print $6}' | head -1
```

### Performance Baseline Validation

**Expected performance metrics (RTX 4090):**

```bash
# Validate against baselines
python scripts/performance_validation.py --validate-baselines

# Expected outputs:
# ✅ Token generation: 120-180 tok/s decode
# ✅ Prefill throughput: 900-1400 tok/s  
# ✅ VRAM usage: 12-14GB total
# ✅ Agent decisions: <200ms
# ✅ End-to-end query: 2-5 seconds
```

### Performance Degradation Response

If performance drops below baselines:

```bash
# 1. Check GPU health
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv

# 2. Validate configuration
python -c "
from src.config import settings
print(f'GPU memory util: {settings.vllm.gpu_memory_utilization}')
print(f'KV cache dtype: {settings.vllm.kv_cache_dtype}')
print(f'Attention backend: {settings.vllm.attention_backend}')
"

# 3. Clear caches if needed
rm -rf cache/ qdrant_storage/
python -c "from src.cache.simple_cache import SimpleCache; cache = SimpleCache(); cache.clear()"

# 4. Restart services
docker-compose restart qdrant
# Restart Ollama if using
```

## Dependency Management

### Monthly Dependency Updates

**SAFE UPDATE PROCEDURE:**

```bash
# 1. Create backup branch
git checkout -b dependency-update-$(date +%Y%m%d)

# 2. Update dependencies
uv sync --upgrade

# 3. Validate core functionality
uv run python -m pytest tests/unit/ --maxfail=5
uv run python -m pytest tests/integration/ --maxfail=3

# 4. Performance regression testing
uv run python scripts/performance_validation.py

# 5. If all tests pass, merge
git add uv.lock pyproject.toml
git commit -m "deps: update dependencies $(date +%Y-%m-%d)"
```

### Critical Dependency Monitoring

Monitor these core dependencies for security updates:

- `pydantic>=2.0.0` - Configuration validation
- `llama-index-core>=0.12.0` - Core RAG functionality  
- `qdrant-client>=1.7.0` - Vector database
- `torch>=2.0.0` - ML framework
- `transformers>=4.35.0` - Model loading
- `sentence-transformers>=2.2.0` - BGE-M3 support

### Dependency Rollback

If dependency updates cause issues:

```bash
# 1. Revert to previous lock file
git checkout HEAD~1 uv.lock

# 2. Reinstall previous versions
uv sync

# 3. Validate functionality
uv run python -c "from src.config import settings; print('✅ Configuration OK')"

# 4. Create issue for investigation
git log --oneline -5  # Note problematic commit
```

## Backup and Recovery

### Configuration Backup

**Weekly backup procedure:**

```bash
# 1. Backup configuration files
tar -czf backups/config-$(date +%Y%m%d).tar.gz \
    src/config/ \
    .env \
    pyproject.toml \
    uv.lock

# 2. Backup critical data
tar -czf backups/data-$(date +%Y%m%d).tar.gz \
    data/ \
    cache/ \
    qdrant_storage/ \
    --exclude="*.tmp" \
    --exclude="*.log"

# 3. Verify backup integrity
tar -tzf backups/config-$(date +%Y%m%d).tar.gz | head -5
```

### Configuration Recovery

To restore from backup:

```bash
# 1. Stop application
pkill -f "streamlit run"
pkill -f "python.*app.py"

# 2. Restore configuration
cd $(mktemp -d)
tar -xzf /path/to/backup/config-YYYYMMDD.tar.gz
cp -r src/config/ /path/to/docmind/src/
cp .env /path/to/docmind/

# 3. Validate restoration
cd /path/to/docmind
python -c "from src.config import settings; print('✅ Config restored')"

# 4. Restart application
streamlit run src/app.py
```

### Database Recovery

For Qdrant vector database recovery:

```bash
# 1. Stop Qdrant
docker-compose stop qdrant

# 2. Restore data
rm -rf qdrant_storage/
tar -xzf backups/data-YYYYMMDD.tar.gz

# 3. Restart and validate
docker-compose start qdrant
curl http://localhost:6333/collections
```

## Security Maintenance

### Security Update Procedure

**Monthly security assessment:**

```bash
# 1. Check for CVEs in dependencies
uv audit

# 2. Update security-critical packages
uv add "package_name>=secure_version"

# 3. Validate no functionality regression
uv run python -m pytest tests/unit/test_config_validation.py

# 4. Update production deployments immediately
```

### Environment Variable Security

**Secure configuration management:**

```bash
# 1. Audit environment variables
env | grep DOCMIND_ | grep -v "PASSWORD\|SECRET\|KEY"

# 2. Ensure no secrets in version control
git log --grep="password\|secret\|key" --oneline

# 3. Validate file permissions
ls -la .env*
# Should be: -rw------- (600) for .env files
```

### Local Security Hardening

```bash
# 1. Ensure Qdrant is local-only
netstat -tlnp | grep 6333
# Should show: 127.0.0.1:6333 or localhost:6333

# 2. Validate Ollama binding
netstat -tlnp | grep 11434  
# Should show: 127.0.0.1:11434 or localhost:11434

# 3. Check for unnecessary network services
ss -tlnp | grep python
```

## Troubleshooting Procedures

### Configuration Issues

**Problem**: Configuration not loading

```bash
# Diagnosis
python -c "
try:
    from src.config import settings
    print(f'✅ Config loaded: {settings.app_name}')
except Exception as e:
    print(f'❌ Config error: {e}')
    import traceback
    traceback.print_exc()
"

# Common fixes:
# 1. Check .env file syntax
grep -n "=" .env | grep -v "^#" | head -5

# 2. Validate environment variable format
env | grep DOCMIND_ | grep -E "(^DOCMIND_[A-Z_]*=|__)"

# 3. Reset to defaults
mv .env .env.backup
cp .env.example .env
```

**Problem**: ADR compliance violations

```bash
# Check specific ADR requirements
python -c "
from src.config import settings
tests = [
    ('BGE-M3 model', settings.embedding.model_name == 'BAAI/bge-m3'),
    ('Agent timeout ≤200ms', settings.agents.decision_timeout <= 200),
    ('FP8 optimization', settings.vllm.kv_cache_dtype == 'fp8_e5m2'),
    ('FlashInfer backend', settings.vllm.attention_backend == 'FLASHINFER'),
]

for name, test in tests:
    status = '✅' if test else '❌'
    print(f'{status} {name}: {test}')
"
```

### Performance Issues

**Problem**: High memory usage

```bash
# Diagnosis
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
free -h | grep Mem

# Fixes:
# 1. Reduce GPU memory utilization
export DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.8

# 2. Clear caches
rm -rf cache/docmind.db
python -c "from src.cache.simple_cache import SimpleCache; SimpleCache().clear()"

# 3. Restart with clean state
pkill -f python
streamlit run src/app.py
```

**Problem**: Slow agent decisions

```bash
# Check agent performance
python -c "
import time
from src.config import settings
print(f'Configured timeout: {settings.agents.decision_timeout}ms')

# Test actual decision time
start = time.time()
# Simulate decision logic
time.sleep(0.05)  # 50ms simulated decision
actual = (time.time() - start) * 1000
print(f'Actual decision time: {actual:.1f}ms')
print(f'Within limit: {actual < settings.agents.decision_timeout}')
"

# If too slow, increase timeout temporarily:
export DOCMIND_AGENTS__DECISION_TIMEOUT=500
```

### Application Errors

**Problem**: Import errors after updates

```bash
# Diagnosis
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from config import settings
    print('✅ Import successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Check PYTHONPATH and working directory')
"

# Fix import paths
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**Problem**: Model loading failures

```bash
# Check model availability
python -c "
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-m3')
    print('✅ BGE-M3 model loaded')
except Exception as e:
    print(f'❌ Model error: {e}')
    print('Run: pip install sentence-transformers')
"

# Download models if missing
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('BAAI/bge-m3')
AutoTokenizer.from_pretrained('BAAI/bge-m3')
print('✅ Models downloaded')
"
```

### Emergency Recovery

**Complete system recovery procedure:**

```bash
# 1. Stop all processes
pkill -f streamlit
pkill -f python.*app
docker-compose down

# 2. Reset to known good state
git stash
git checkout main
git pull origin main

# 3. Clean rebuild
uv sync
python -c "from src.config import settings; print('✅ Config OK')"

# 4. Clear all caches and data
rm -rf cache/ qdrant_storage/ logs/
mkdir -p cache logs

# 5. Restart services
docker-compose up -d qdrant
sleep 5

# 6. Validate system
python scripts/performance_validation.py
streamlit run src/app.py
```

---

## Maintenance Schedule

### Daily
- [ ] Check system health metrics
- [ ] Monitor error logs
- [ ] Validate GPU utilization

### Weekly  
- [ ] Run full test suite
- [ ] Backup configuration and data
- [ ] Review performance trends

### Monthly
- [ ] Update dependencies (safely)
- [ ] Security audit
- [ ] Performance baseline validation
- [ ] Code quality assessment

### Quarterly
- [ ] Architecture review
- [ ] Documentation updates
- [ ] ADR compliance audit
- [ ] Disaster recovery testing

This maintenance guide ensures the unified architecture remains stable, performant, and secure while preserving the architectural integrity achieved through the refactoring process.