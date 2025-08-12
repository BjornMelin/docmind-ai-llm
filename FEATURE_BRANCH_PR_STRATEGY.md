# Feature Branch & Pull Request Strategy

## **Pure LlamaIndex Stack Implementation - 85% Code Reduction**

---

## 🎯 Branch Naming Convention

```text
feat/<pr-number>-<component>-<action>
```

**Examples:**

- `feat/01-infra-pytorch-gpu`

- `feat/02-deps-cleanup-remove`

- `feat/03-agent-single-react`

- `feat/04-search-qdrant-hybrid`

- `feat/05-kg-native-integration`

---

## 📋 Pull Request Sequence

### PR #1: Infrastructure - PyTorch Native GPU & spaCy Optimization

**Branch:** `feat/01-infra-pytorch-gpu`

**Effort:** 4-5 hours

**Dependencies:** None (can start immediately)

#### Commits - PR #1

```bash
# Remove pynvml dependencies
git commit -m "refactor: remove pynvml and nvidia-ml-py3 dependencies

BREAKING CHANGE: Replace pynvml with PyTorch native GPU monitoring

- Remove 3 unnecessary dependencies (pynvml, nvidia-ml-py3, gpustat)

- Reduces installation complexity and maintenance burden"

# Add PyTorch native GPU monitoring
git commit -m "feat: implement PyTorch native GPU monitoring

- Add GPUMetrics dataclass for structured metrics

- Implement async context manager for performance monitoring

- Use torch.cuda native APIs for all GPU operations

- 79% code reduction (120 lines → 25 lines)"

# Optimize spaCy with native features
git commit -m "perf: optimize spaCy with 3.8+ native memory zones

- Implement ADRCompliantSpacyManager with native APIs

- Add memory_zone() for 40% performance improvement

- Remove custom model downloading logic

- 70% code reduction (250 lines → 75 lines)"

# Add tests
git commit -m "test: add comprehensive infrastructure tests

- Unit tests for GPU monitoring

- Integration tests for spaCy optimization

- Performance benchmarks for memory usage"
```

#### Files to Modify/Delete

```text
DELETE:

- src/utils/nvidia_utils.py (entire file - 120 lines)

- src/utils/spacy_downloader.py (entire file - 185 lines)

CREATE:

- src/core/infrastructure/gpu_monitor.py (25 lines)

- src/core/infrastructure/spacy_manager.py (35 lines)

- tests/unit/test_gpu_monitor.py

- tests/unit/test_spacy_manager.py

MODIFY:

- pyproject.toml (remove pynvml, nvidia-ml-py3, gpustat)

- src/config/settings.py (update GPU config)
```

---

### PR #2: Dependency Cleanup & Optimization

**Branch:** `feat/02-deps-cleanup-remove`  

**Effort:** 2-3 hours  

**Dependencies:** PR #1 (for GPU changes)

#### Commits - PR #2

```bash

# Remove ragatouille and dependencies
git commit -m "refactor: remove ragatouille and 20 sub-dependencies

BREAKING CHANGE: Replace ragatouille with llama-index-postprocessor-colbert-rerank

- Remove 20 unnecessary transitive dependencies

- Use native LlamaIndex ColBERT integration

- Simplifies dependency tree significantly"

# Clean up unused libraries
git commit -m "chore: remove unused and deprecated dependencies

- Remove langchain remnants (not used)

- Remove phoenix observability (move to dev deps)

- Remove duplicate embedding libraries

- Total: 15 packages removed"

# Update dependency management
git commit -m "build: optimize pyproject.toml with minimal dependencies

- Consolidate to core libraries only

- Pin versions for reproducibility

- Add dependency groups (core, dev, test)

- 33% reduction in total dependencies (45 → 30)"
```

#### Files to Modify/Delete - PR #2

```text
DELETE:

- src/legacy/ (entire directory if exists)

- src/utils/ragatouille_wrapper.py

MODIFY:

- pyproject.toml (major cleanup)

- requirements.txt (if exists - DELETE)

- setup.py (if exists - DELETE)
```

---

### PR #3: Single ReActAgent Implementation

**Branch:** `feat/03-agent-single-react`  

**Effort:** 6-8 hours  

**Dependencies:** PR #2 (clean dependencies first)

#### Commits - PR #3

```bash

# Remove multi-agent orchestration
git commit -m "refactor!: remove multi-agent LangGraph supervisor

BREAKING CHANGE: Replace multi-agent with single ReActAgent

- Delete 450+ lines of orchestration code

- Remove complex routing and coordination logic

- Simplifies agent architecture dramatically"

# Implement single ReActAgent
git commit -m "feat: implement Pure LlamaIndex ReActAgent

- Single agent with full agentic capabilities

- Native tool calling and query planning

- Chain-of-thought reasoning built-in

- 82% code reduction (450 → 80 lines)"

# Add agent tools
git commit -m "feat: add comprehensive agent tools

- QueryEngineTool for semantic search

- SummaryTool for document summarization

- KnowledgeGraphTool for entity relationships

- All tools leverage existing indices"

# Add streaming support
git commit -m "feat: add native streaming for ReActAgent

- Implement async streaming responses

- Add progress tracking for long operations

- Improve user experience with real-time feedback"

# Add tests and benchmarks
git commit -m "test: add agent tests and performance benchmarks

- Unit tests for ReActAgent functionality

- Integration tests for tool calling

- Performance: verify <2s response time

- Success rate: validate 82.5% accuracy"
```

#### Files to Modify/Delete - PR #3

```text
DELETE:

- src/agents/multi_agent_system.py (450+ lines)

- src/agents/supervisor.py

- src/agents/routing.py

- src/agents/coordination.py

CREATE:

- src/agents/react_agent.py (80 lines)

- src/agents/tools.py (50 lines)

- tests/integration/test_react_agent.py

MODIFY:

- src/app.py (update agent initialization)

- src/config/settings.py (simplify agent config)
```

---

### PR #4: Qdrant Native Hybrid Search

**Branch:** `feat/04-search-qdrant-hybrid`  

**Effort:** 5-6 hours  

**Dependencies:** PR #3 (agent needs search tools)

#### Commits - PR #4

```bash

# Remove custom hybrid search logic
git commit -m "refactor: remove custom hybrid search implementation

BREAKING CHANGE: Replace with Qdrant native hybrid search

- Delete 300 lines of manual fusion logic

- Remove custom sparse/dense combination code

- Leverage Qdrant's built-in hybrid support"

# Implement Qdrant native hybrid
git commit -m "feat: implement Qdrant native hybrid search

- Enable native hybrid mode in Qdrant

- Use FastEmbed for sparse BM25 encoding

- Configure RRF fusion with alpha=0.7

- 80% code reduction (300 → 60 lines)"

# Add ColBERT reranking
git commit -m "feat: integrate ColBERT reranking

- Add llama-index-postprocessor-colbert-rerank

- Configure for top-10 reranking

- Improve accuracy while maintaining <2s latency"

# Optimize async pipeline
git commit -m "perf: implement async QueryPipeline

- Enable parallel retrieval and reranking

- Add streaming support for large results

- 40% latency improvement via async execution"

# Add comprehensive tests
git commit -m "test: add hybrid search tests and benchmarks

- Test dense, sparse, and hybrid modes

- Validate RRF fusion accuracy

- Benchmark: <2s query latency

- Test ColBERT reranking quality"
```

#### Files to Modify/Delete - PR #4

```text
DELETE:

- src/search/hybrid_fusion.py (200+ lines)

- src/search/custom_retriever.py (100+ lines)

CREATE:

- src/search/qdrant_hybrid.py (60 lines)

- src/search/pipeline.py (40 lines)

- tests/integration/test_hybrid_search.py

MODIFY:

- src/config/settings.py (Qdrant config)

- docker-compose.yml (update Qdrant version)
```

---

### PR #5: Knowledge Graph Native Integration

**Branch:** `feat/05-kg-native-integration`  

**Effort:** 4-5 hours  

**Dependencies:** PR #4 (uses search infrastructure)

#### Commits - PR #5

```bash

# Simplify KG implementation
git commit -m "refactor: simplify knowledge graph to native LlamaIndex

- Remove custom graph construction logic

- Use LlamaIndex KnowledgeGraphIndex

- Integrate with spaCy for NER

- 75% code reduction"

# Add hybrid embeddings
git commit -m "feat: enable hybrid embeddings for KG

- Add dense + sparse embeddings to KG

- Improve entity relationship extraction

- Enable semantic + keyword search on graph"

# Optimize KG querying
git commit -m "perf: optimize KG query engine

- Add async query support

- Implement caching for common queries

- Use tree_summarize for better responses"

# Add tests
git commit -m "test: add knowledge graph tests

- Test entity extraction accuracy

- Validate relationship discovery

- Benchmark query performance"
```

#### Files to Modify/Delete - PR #5

```text
DELETE:

- src/kg/custom_graph.py (if exists)

CREATE:

- src/kg/knowledge_graph.py (50 lines)

- tests/integration/test_knowledge_graph.py

MODIFY:

- src/agents/tools.py (add KG tool)
```

---

### PR #6: Performance & Production Hardening

**Branch:** `feat/06-perf-production`  

**Effort:** 3-4 hours  

**Dependencies:** PRs #1-5 (final integration)

#### Commits

```bash

# Add comprehensive monitoring
git commit -m "feat: add production monitoring

- Add metrics collection for all components

- Implement performance dashboards

- Add alerting for degradation"

# Add resilience patterns
git commit -m "feat: implement resilience patterns

- Add circuit breakers for external calls

- Implement retry with exponential backoff

- Add graceful degradation"

# Optimize caching
git commit -m "perf: implement multi-level caching

- Add Redis for embedding cache

- SQLite for document cache

- Memory cache for hot queries"

# Final cleanup
git commit -m "chore: final cleanup and optimization

- Remove all deprecated code

- Update documentation

- Add production deployment guide"

# Add e2e tests
git commit -m "test: add end-to-end test suite

- Complete user journey tests

- Performance regression tests

- Load testing for production readiness"
```

#### Files to Create/Modify - PR #6

```text
CREATE:

- src/monitoring/metrics.py

- src/resilience/circuit_breaker.py

- deployment/production.md

- tests/e2e/test_complete_flow.py

MODIFY:

- src/app.py (final integration)

- README.md (update with new architecture)
```

---

## 🚀 Implementation Schedule

```text
Week 1:
  Mon-Tue: PR #1 (Infrastructure)
  Wed-Thu: PR #2 (Dependencies) + PR #3 start
  Fri: PR #3 completion (Agent)

Week 2:
  Mon-Tue: PR #4 (Hybrid Search)
  Wed-Thu: PR #5 (Knowledge Graph)
  Fri: PR #6 (Production)

Week 3:
  Mon-Tue: Integration testing
  Wed: Performance validation
  Thu-Fri: Documentation & deployment
```

---

## ✅ Testing Strategy per PR

### Unit Tests (per PR)

- Minimum 85% code coverage

- Mock external dependencies

- Test error conditions

### Integration Tests (per PR)

- Test with real Qdrant instance

- Validate agent tool interactions

- End-to-end query flows

### Performance Tests (PR #6)

- Query latency <2s

- Memory usage <300MB

- 100 concurrent queries

### Acceptance Criteria

- All tests passing

- No regressions in functionality

- Performance targets met

- Documentation updated

---

## 🎯 Success Metrics

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| Code Lines | ~2000 | ~300 | 85% reduction ✓ |
| Dependencies | 45 | 30 | 33% reduction ✓ |
| Query Latency | ~5s | <2s | Performance tests |
| Memory Usage | 500MB | <300MB | Monitoring |
| Test Coverage | 60% | >85% | Coverage reports |
| Agent Success | 37% | 82.5% | Benchmarks |

---

## 🔄 Rollback Strategy

Each PR can be reverted independently:

```bash

# Revert specific PR
git revert <merge-commit>

# Or reset to previous state
git reset --hard <previous-commit>
```

Feature flags for gradual rollout:

```python
if settings.use_new_agent:
    agent = ReActAgent(...)
else:
    agent = LegacyAgent(...)  # Until fully validated
```

---

## 📝 Conventional Commit Types Used

- `feat`: New features (agent, search, KG)

- `refactor`: Code removal/simplification

- `perf`: Performance improvements

- `test`: Test additions

- `chore`: Maintenance tasks

- `build`: Dependency changes

- `BREAKING CHANGE`: Backwards incompatible changes

---

## Next Steps

1. **Review this strategy** with the team
2. **Create first branch**: `git checkout -b feat/01-infra-pytorch-gpu`
3. **Begin PR #1** implementation with subagents
4. **Monitor progress** against targets
5. **Adjust as needed** based on findings

Ready to begin implementation with PR #1!
