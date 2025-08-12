# Master Implementation Prompts for DocMind AI Refactoring

## Context & Background

DocMind AI is undergoing a major refactoring to implement the **Pure LlamaIndex Stack** approach, achieving an 8.6/10 architecture score while adhering to KISS, DRY, and YAGNI principles. Research has validated that a single LlamaIndex ReActAgent provides full agentic capabilities without multi-agent complexity.

### Completed PRs:

- **PR #1**: Infrastructure - PyTorch Native GPU & spaCy Optimization ✅

- **PR #2**: Dependency Cleanup & Optimization ✅

### Core Decision Framework (Apply to ALL Decisions)

When deciding whether to add complexity to address violations, apply this framework:

```

## 📊 Decision Framework (Internal Use)

- **Solution Leverage (35%)** – Library-first, proven patterns, SOTA/community-maintained solutions  

- **Application Value (30%)** – Maximize real-user impact and feature completeness  

- **Maintenance & Cognitive Load (25%)** – Minimize support burden and developer friction  

- **Architectural Adaptability (10%)** – Favor modular, future-proof design when justified  
```

### Core Engineering Principles (MANDATORY)

```

## Core Software Engineering Principles
Ensure EVERY change strictly adheres to:  

- **KISS** – Keep It Simple, Stupid  

- **DRY** – Don't Repeat Yourself  

- **YAGNI** – You Aren't Gonna Need It  

Prioritize leveraging **modern libraries and frameworks**—favoring robust, 
pre-built features over custom code—by conducting deep research into the 
latest ecosystem capabilities.
```

### Toolchain & Research Workflow

For any step requiring research, code validation, or decision analysis:

1. **context7** – Retrieve up-to-date library documentation
2. **exa** – Search for real-world usage and examples
3. **firecrawl** – Scrape and extract documentation
4. **clear-thought** – Structured problem-solving and trade-off analysis
5. **qdrant** – Store and retrieve insights for long-term reference
6. **uv** – Use exclusively for all Python package management (NEVER use pip)

---

## PR #3: Single ReActAgent Implementation

### Master Prompt for PR #3

```markdown

# Implement PR #3: Single ReActAgent Implementation

## Objective
Replace the complex multi-agent LangGraph supervisor system with a single LlamaIndex ReActAgent, achieving 82% code reduction while maintaining full agentic capabilities.

## Research Context

- Research shows single-agent success rate: 82.5% vs multi-agent: 37%

- Pure LlamaIndex Stack scored 8.6/10 vs Multi-agent LangGraph 2.89/10

- Target: 450+ lines → 80 lines (82% reduction)

## Implementation Steps

### Step 1: Setup Branch
```bash
git checkout main && git pull origin main
git checkout -b feat/03-agent-single-react
```

### Step 2: Coordinate Subagents

Use these specialized agents in this sequence:

1. **@agent-implementation-executor**: Remove multi-agent code
   - Delete src/agents/multi_agent_system.py (450+ lines)
   - Delete src/agents/supervisor.py
   - Delete src/agents/routing.py
   - Delete src/agents/coordination.py

2. **@agent-implementation-executor**: Implement single ReActAgent
   ```python
   from llama_index.core.agent import ReActAgent
   from llama_index.core.tools import QueryEngineTool
   
   def create_agentic_rag_system(documents):
       """Single ReActAgent with full agentic capabilities."""
       vector_index = VectorStoreIndex.from_documents(documents)
       summary_index = SummaryIndex.from_documents(documents)
       
       vector_tool = QueryEngineTool.from_defaults(
           vector_index.as_query_engine(),
           name="semantic_search",
           description="Dense vector search for semantic similarity"
       )
       
       summary_tool = QueryEngineTool.from_defaults(
           summary_index.as_query_engine(),
           name="keyword_search",
           description="Sparse keyword-based search"
       )
       
       agent = ReActAgent.from_tools(
           [vector_tool, summary_tool],
           verbose=True,
           max_iterations=3
       )
       return agent
   ```

3. **@agent-pytest-test-generator**: Generate comprehensive tests
   - Test ReActAgent functionality
   - Test tool calling
   - Verify <2s response time
   - Validate 82.5% accuracy

4. **@agent-pytest-qa-agent**: Run QA validation
   - Execute all tests
   - Verify performance benchmarks
   - Check code coverage

5. **python-code-review-workflow**: Review code quality
   - Verify KISS compliance (80 lines target)
   - Check DRY principle adherence
   - Validate library-first approach

6. **@agent-merge-ready-pr-reviewer**: Final validation
   - Confirm all tests passing
   - Verify no breaking changes
   - Check performance requirements met

### Key Requirements

- Maintain ALL agentic capabilities:
  ✅ Chain-of-thought reasoning
  ✅ Dynamic tool selection
  ✅ Query decomposition
  ✅ Adaptive retrieval

- Code reduction: 450 lines → 80 lines (82% reduction)

- Response time: <2s

- Success rate: Target 82.5%

### Conventional Commits
```bash
git commit -m "refactor!: remove multi-agent LangGraph supervisor

BREAKING CHANGE: Replace multi-agent with single ReActAgent

- Delete 450+ lines of orchestration code

- Simplifies agent architecture dramatically"

git commit -m "feat: implement Pure LlamaIndex ReActAgent

- Single agent with full agentic capabilities

- Native tool calling and query planning

- 82% code reduction (450 → 80 lines)"
```

### Decision Framework Application

- Solution Leverage (35%): ✅ Pure LlamaIndex native patterns

- Application Value (30%): ✅ All agentic features preserved

- Maintenance (25%): ✅ 82% code reduction

- Adaptability (10%): ✅ Modular tool system

### Research with Tools

- Use context7 to get latest ReActAgent documentation

- Use exa to find real-world ReActAgent examples

- Use clear-thought for architecture decisions

- Store learnings in qdrant for future reference
```

---

## PR #4: Qdrant Native Hybrid Search

### Master Prompt for PR #4

```markdown

# Implement PR #4: Qdrant Native Hybrid Search

## Objective
Replace custom hybrid search implementation with Qdrant's native hybrid search capabilities, achieving 80% code reduction while improving performance.

## Research Context

- Qdrant scored 9.2/10 for native BM25 integration

- Target: 300 lines → 60 lines (80% reduction)

- Performance target: <2s query latency

## Implementation Steps

### Step 1: Setup Branch
```bash
git checkout main && git pull origin main
git checkout -b feat/04-search-qdrant-hybrid
```

### Step 2: Coordinate Subagents

Use these specialized agents in this sequence:

1. **@agent-implementation-executor**: Remove custom hybrid search
   - Delete src/search/hybrid_fusion.py (200+ lines)
   - Delete src/search/custom_retriever.py (100+ lines)

2. **@agent-implementation-executor**: Implement Qdrant native hybrid
   ```python
   from llama_index.core.retrievers import QueryFusionRetriever
   from llama_index.vector_stores.qdrant import QdrantVectorStore
   from llama_index.postprocessor.colbert_rerank import ColbertRerank
   
   async def setup_hybrid_search(documents):
       """Qdrant with native hybrid search support."""
       vector_store = QdrantVectorStore(
           collection_name="docmind",
           enable_hybrid=True,
           fastembed_sparse_model="Qdrant/bm25",
           hybrid_fusion="rrf",
           alpha=0.7  # ADR-013 compliant
       )
       
       storage_context = StorageContext.from_defaults(
           vector_store=vector_store
       )
       
       index = VectorStoreIndex.from_documents(
           documents,
           storage_context=storage_context,
           show_progress=True
       )
       
       retriever = QueryFusionRetriever(
           [index.as_retriever()],
           similarity_top_k=5,
           num_queries=3
       )
       
       # Add ColBERT reranking
       reranker = ColbertRerank(
           top_n=5,
           keep_retrieval_score=True
       )
       
       return retriever, reranker
   ```

3. **database-optimization-agent**: Optimize Qdrant configuration
   - Configure collection settings
   - Set up indexing parameters
   - Optimize for <2s latency

4. **@agent-pytest-test-generator**: Generate search tests
   - Test dense, sparse, and hybrid modes
   - Validate RRF fusion accuracy
   - Benchmark query latency
   - Test ColBERT reranking quality

5. **@agent-pytest-qa-agent**: Run performance validation
   - Verify <2s query latency
   - Test with various query types
   - Validate accuracy improvements

6. **@agent-merge-ready-pr-reviewer**: Final validation
   - Check all search functionality preserved
   - Verify performance improvements
   - Confirm code reduction achieved

### Key Requirements

- Native hybrid search with Qdrant

- RRF fusion with alpha=0.7

- ColBERT reranking integration

- Query latency <2s

- 80% code reduction

### Conventional Commits
```bash
git commit -m "refactor: remove custom hybrid search implementation

BREAKING CHANGE: Replace with Qdrant native hybrid search

- Delete 300 lines of manual fusion logic

- Leverage Qdrant's built-in hybrid support"

git commit -m "feat: implement Qdrant native hybrid search

- Enable native hybrid mode in Qdrant

- Use FastEmbed for sparse BM25 encoding

- Configure RRF fusion with alpha=0.7

- 80% code reduction (300 → 60 lines)"

git commit -m "feat: integrate ColBERT reranking

- Add llama-index-postprocessor-colbert-rerank

- Configure for top-10 reranking

- Improve accuracy while maintaining <2s latency"
```

### Decision Framework Application

- Solution Leverage (35%): ✅ Qdrant native features

- Application Value (30%): ✅ Better search accuracy

- Maintenance (25%): ✅ 80% code reduction

- Adaptability (10%): ✅ Standard Qdrant API

### Research with Tools

- Use context7 for Qdrant hybrid search documentation

- Use exa to find Qdrant hybrid search examples

- Use firecrawl to extract Qdrant best practices

- Use database-optimization-agent for configuration
```

---

## PR #5: Knowledge Graph Native Integration

### Master Prompt for PR #5

```markdown

# Implement PR #5: Knowledge Graph Native Integration

## Objective
Simplify knowledge graph implementation using LlamaIndex native KG with spaCy NER, achieving 75% code reduction.

## Research Context

- LlamaIndex KnowledgeGraphIndex with hybrid embeddings

- Target: 75% code reduction

- Integration with existing spaCy infrastructure from PR #1

## Implementation Steps

### Step 1: Setup Branch
```bash
git checkout main && git pull origin main
git checkout -b feat/05-kg-native-integration
```

### Step 2: Coordinate Subagents

Use these specialized agents in this sequence:

1. **@agent-implementation-executor**: Simplify KG implementation
   ```python
   from llama_index.core import KnowledgeGraphIndex
   from llama_index.core.node_parser import SentenceSplitter
   import spacy
   
   def create_knowledge_graph(documents):
       """Native KG with entity extraction."""
       nlp = spacy.load("en_core_web_sm")
       
       parser = SentenceSplitter(
           chunk_size=512,
           chunk_overlap=50
       )
       
       kg_index = KnowledgeGraphIndex.from_documents(
           documents,
           max_triplets_per_chunk=2,
           include_embeddings=True,
           kg_triple_extract_fn=lambda text: extract_triplets(text, nlp)
       )
       
       query_engine = kg_index.as_query_engine(
           include_text=True,
           response_mode="tree_summarize",
           embedding_mode="hybrid"
       )
       
       return kg_index, query_engine
   
   def extract_triplets(text, nlp):
       """Extract entity relationships."""
       doc = nlp(text)
       triplets = []
       for ent in doc.ents:
           if ent.label_ in ["PERSON", "ORG", "GPE"]:
               triplets.append((ent.text, "mentioned_in", text[:50]))
       return triplets
   ```

2. **@agent-pytest-test-generator**: Generate KG tests
   - Test entity extraction accuracy
   - Validate relationship discovery
   - Benchmark query performance

3. **@agent-pytest-qa-agent**: Validate integration
   - Test with existing spaCy manager
   - Verify memory optimization
   - Check performance metrics

4. **python-code-review-workflow**: Review implementation
   - Verify KISS compliance
   - Check integration with PR #1 infrastructure
   - Validate library-first approach

5. **@agent-merge-ready-pr-reviewer**: Final validation

### Key Requirements

- Native LlamaIndex KnowledgeGraphIndex

- spaCy NER integration

- Hybrid embeddings

- 75% code reduction

- Tree summarize response mode

### Conventional Commits
```bash
git commit -m "refactor: simplify knowledge graph to native LlamaIndex

- Remove custom graph construction logic

- Use LlamaIndex KnowledgeGraphIndex

- Integrate with spaCy for NER

- 75% code reduction"

git commit -m "feat: enable hybrid embeddings for KG

- Add dense + sparse embeddings to KG

- Improve entity relationship extraction

- Enable semantic + keyword search on graph"
```

### Decision Framework Application

- Solution Leverage (35%): ✅ Native LlamaIndex KG

- Application Value (30%): ✅ Entity relationship extraction

- Maintenance (25%): ✅ 75% code reduction

- Adaptability (10%): ✅ Standard KG patterns

### Research with Tools

- Use context7 for KnowledgeGraphIndex documentation

- Use exa for KG implementation examples

- Use clear-thought for entity extraction strategy
```

---

## PR #6: Performance & Production Hardening

### Master Prompt for PR #6

```markdown

# Implement PR #6: Performance & Production Hardening

## Objective
Add production monitoring, resilience patterns, and performance optimizations to complete the refactoring.

## Implementation Steps

### Step 1: Setup Branch
```bash
git checkout main && git pull origin main
git checkout -b feat/06-perf-production
```

### Step 2: Coordinate Subagents

Use these specialized agents in parallel where possible:

1. **@agent-implementation-executor**: Add monitoring
   ```python
   from contextlib import asynccontextmanager
   import time
   from loguru import logger
   
   class PerformanceMonitor:
       """Production performance monitoring."""
       
       @asynccontextmanager
       async def monitor_operation(self, operation: str):
           start = time.perf_counter()
           try:
               yield
               duration = time.perf_counter() - start
               logger.info(f"{operation}: {duration:.2f}s")
               if duration > 2.0:
                   logger.warning(f"{operation} exceeded 2s target: {duration:.2f}s")
           except Exception as e:
               logger.error(f"{operation} failed: {e}")
               raise
   ```

2. **@agent-implementation-executor**: Add resilience patterns
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   import stamina
   
   @retry(
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10)
   )
   async def resilient_query(query: str, agent):
       """Query with retry and backoff."""
       return await agent.aquery(query)
   ```

3. **@agent-implementation-executor**: Optimize caching
   ```python
   from functools import lru_cache
   import redis
   import diskcache
   
   # Multi-level caching
   memory_cache = {}
   disk_cache = diskcache.Cache("./cache")
   redis_client = redis.Redis(host='localhost', port=6379)
   
   async def cached_query(query: str):
       # Check memory cache
       if query in memory_cache:
           return memory_cache[query]
       
       # Check disk cache
       if query in disk_cache:
           result = disk_cache[query]
           memory_cache[query] = result
           return result
       
       # Check Redis
       result = redis_client.get(query)
       if result:
           memory_cache[query] = result
           disk_cache[query] = result
           return result
       
       # Compute and cache
       result = await compute_result(query)
       cache_result(query, result)
       return result
   ```

4. **@agent-pytest-test-generator**: Generate e2e tests
   - Complete user journey tests
   - Performance regression tests
   - Load testing for production readiness

5. **@agent-pytest-qa-agent**: Run comprehensive validation
   - Verify <2s query latency
   - Test memory usage <300MB
   - Validate 100 concurrent queries

6. **@agent-merge-ready-pr-reviewer**: Final production check
   - All performance targets met
   - Monitoring working correctly
   - Resilience patterns effective

### Key Requirements

- Query latency <2s with monitoring

- Memory usage <300MB

- Support 100 concurrent queries

- Comprehensive monitoring

- Resilience patterns

- Multi-level caching

### Conventional Commits
```bash
git commit -m "feat: add production monitoring

- Add metrics collection for all components

- Implement performance dashboards

- Add alerting for degradation"

git commit -m "feat: implement resilience patterns

- Add circuit breakers for external calls

- Implement retry with exponential backoff

- Add graceful degradation"

git commit -m "perf: implement multi-level caching

- Add Redis for embedding cache

- SQLite for document cache

- Memory cache for hot queries"
```

### Decision Framework Application

- Solution Leverage (35%): ✅ Standard monitoring/caching patterns

- Application Value (30%): ✅ Production readiness

- Maintenance (25%): ✅ Observable, maintainable system

- Adaptability (10%): ✅ Extensible monitoring

### Research with Tools

- Use context7 for production best practices

- Use exa for monitoring pattern examples

- Use clear-thought for caching strategy

- Store production patterns in qdrant
```

---

## Universal Subagent Workflow

For ALL PRs, follow this subagent coordination pattern:

### Parallel Execution (When Possible)
```
Run in parallel:

- @agent-implementation-executor (main implementation)

- @agent-pytest-test-generator (test creation)

- debug-architect-specialist (if issues arise)
```

### Sequential Execution (When Dependencies Exist)
```
1. @agent-implementation-executor (remove old code)
2. @agent-implementation-executor (add new code)
3. @agent-pytest-test-generator (create tests)
4. @agent-pytest-qa-agent (run tests)
5. python-code-review-workflow (code review)
6. @agent-merge-ready-pr-reviewer (final check)
```

### Quality Gates
Each PR must pass:

- ✅ All tests passing (>85% coverage on new code)

- ✅ Performance targets met (<2s latency)

- ✅ Code reduction achieved (per PR targets)

- ✅ KISS/DRY/YAGNI compliance validated

- ✅ Library-first approach confirmed

- ✅ No breaking changes (unless documented)

- ✅ Conventional commits used

### Final PR Creation
```bash

# Push branch
git push -u origin <branch-name>

# Create PR with gh CLI
gh pr create --title "<conventional-commit-title>" \
  --body "<comprehensive-description>" \
  --base main
```

## Important Notes

1. **ALWAYS use uv for Python package management** (never pip)
2. **Apply decision framework to EVERY complexity addition**
3. **Research with tools before implementing**
4. **Store learnings in qdrant for future reference**
5. **Follow conventional commits strictly**
6. **Maintain backward compatibility unless BREAKING CHANGE noted**
7. **Target code reduction percentages are MANDATORY**
8. **Performance targets (<2s latency) are MANDATORY**

Each PR builds on the previous ones, so complete them in order. The infrastructure from PR #1 (GPU monitoring, spaCy) and dependency cleanup from PR #2 are prerequisites for the remaining work.
