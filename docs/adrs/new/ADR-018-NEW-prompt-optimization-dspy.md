# ADR-018-NEW: Automatic Prompt Optimization with DSPy

## Title

DSPy-Based Automatic Prompt Optimization and Query Rewriting

## Version/Date

1.0 / 2025-08-17

## Status

Accepted (Experimental)

## Description

Implements automatic prompt optimization and query rewriting using DSPy (Declarative Self-Improving Python) integrated with LlamaIndex. This replaces manual prompt engineering with data-driven optimization, improving retrieval quality through automatic query expansion and refinement. DSPy provides compile-time optimization of prompts based on training examples, significantly improving RAG performance without manual tuning.

## Context

Current manual prompt engineering has limitations:

- Time-consuming trial and error process
- Inconsistent results across different query types
- No systematic way to improve prompts based on feedback
- Poor query formulation leads to suboptimal retrieval

DSPy offers a paradigm shift from "prompting" to "programming" LLMs, providing:

- Automatic prompt optimization through compilation
- Data-driven improvements using training examples
- Systematic query rewriting and expansion
- Measurable performance improvements

## Related Requirements

### Functional Requirements

- **FR-1:** Automatically optimize prompts based on performance metrics
- **FR-2:** Rewrite user queries for improved retrieval
- **FR-3:** Expand queries with synonyms and related terms
- **FR-4:** Learn from user feedback and improve over time
- **FR-5:** Maintain prompt version history for rollback

### Non-Functional Requirements

- **NFR-1:** **(Performance)** Query optimization <200ms overhead
- **NFR-2:** **(Quality)** ≥20% improvement in retrieval metrics
- **NFR-3:** **(Local-First)** All optimization runs locally
- **NFR-4:** **(Maintainability)** Clear optimization pipeline

## Alternatives

### 1. Manual Prompt Engineering (Current)

- **Description**: Hand-craft and manually tune prompts
- **Issues**: Time-consuming, inconsistent, no systematic improvement
- **Score**: 3/10 (quality: variable, scalability: 2, maintainability: 3)

### 2. DSPy with LlamaIndex (Selected)

- **Description**: Automatic optimization with DSPy integrated into LlamaIndex
- **Benefits**: Data-driven, systematic improvement, measurable results
- **Score**: 9/10 (quality: 9, automation: 10, integration: 8)

### 3. LangChain LCEL with Few-Shot

- **Description**: LangChain expression language with few-shot examples
- **Issues**: Still manual, no automatic optimization
- **Score**: 5/10 (quality: 6, automation: 3, complexity: 6)

## Decision

We will implement **DSPy-based automatic prompt optimization** with:

1. **Query Rewriting Module**: Automatic query expansion and refinement
2. **Prompt Compilation**: MIPROv2 optimizer for prompt tuning
3. **Bootstrapping**: Self-improvement from unlabeled data
4. **Integration**: Seamless integration with LlamaIndex pipeline
5. **Feature Flag**: Experimental feature behind flag initially

## Related Decisions

- **ADR-003-NEW** (Adaptive Retrieval): Benefits from optimized queries
- **ADR-004-NEW** (Local LLM): Provides model for DSPy optimization
- **ADR-011-NEW** (Agent Orchestration): Query agent uses DSPy
- **ADR-012-NEW** (Evaluation): Provides metrics for optimization

## Design

### DSPy Query Optimization Module

```python
import dspy
from typing import List, Dict, Any
from llama_index.core import QueryBundle
from pydantic import BaseModel, Field

# Configure DSPy with local model
dspy.settings.configure(
    lm=dspy.LM("ollama/qwen3:14b"),
    rm=dspy.ColBERTv2(url="http://localhost:8893/api/search")
)

class QueryExpansion(dspy.Signature):
    """Expand a query into multiple search variants for better retrieval."""
    
    original_query: str = dspy.InputField(
        desc="Original user query"
    )
    expanded_queries: List[str] = dspy.OutputField(
        desc="3-5 semantically similar query variants"
    )
    key_concepts: List[str] = dspy.OutputField(
        desc="Key concepts and entities to search for"
    )

class QueryRewriter(dspy.Module):
    """DSPy module for automatic query rewriting and expansion."""
    
    def __init__(self):
        super().__init__()
        self.expand = dspy.ChainOfThought(QueryExpansion)
        self.refine = dspy.Predict("query, concepts -> refined_query")
    
    def forward(self, query: str) -> Dict[str, Any]:
        # Expand query into variants
        expansion = self.expand(original_query=query)
        
        # Refine based on key concepts
        refined = self.refine(
            query=query,
            concepts=", ".join(expansion.key_concepts)
        )
        
        return {
            "original": query,
            "refined": refined.refined_query,
            "variants": expansion.expanded_queries,
            "concepts": expansion.key_concepts
        }

class OptimizedRetriever(dspy.Module):
    """Retriever with DSPy-optimized query processing."""
    
    def __init__(self, base_retriever):
        super().__init__()
        self.query_rewriter = QueryRewriter()
        self.base_retriever = base_retriever
        self.retrieve = dspy.Retrieve(k=20)
    
    def forward(self, query: str) -> List[Document]:
        # Rewrite query using DSPy
        rewritten = self.query_rewriter(query)
        
        # Retrieve with all variants
        all_docs = []
        for variant in [rewritten["refined"]] + rewritten["variants"]:
            docs = self.retrieve(variant)
            all_docs.extend(docs)
        
        # Deduplicate and rerank
        unique_docs = self._deduplicate(all_docs)
        return self._rerank(unique_docs, query)[:10]
```

### Automatic Prompt Optimization

```python
from dspy.teleprompt import MIPROv2, BootstrapFewShot
from typing import List, Tuple

class DSPyOptimizer:
    """Automatic prompt optimization using DSPy."""
    
    def __init__(self, retriever: OptimizedRetriever):
        self.retriever = retriever
        self.optimizer = None
        self.compiled_retriever = None
    
    def create_training_set(
        self, 
        queries: List[str], 
        relevant_docs: List[List[str]]
    ) -> List[dspy.Example]:
        """Create training examples from queries and relevant documents."""
        examples = []
        for query, docs in zip(queries, relevant_docs):
            examples.append(dspy.Example(
                query=query,
                relevant_docs=docs
            ).with_inputs("query"))
        return examples
    
    def optimize_with_examples(
        self, 
        train_examples: List[dspy.Example],
        metric: callable = None
    ):
        """Optimize prompts using labeled examples."""
        if metric is None:
            metric = self.retrieval_recall_metric
        
        # Use MIPROv2 for optimization
        self.optimizer = MIPROv2(
            metric=metric,
            prompt_model=dspy.settings.lm,
            task_model=dspy.settings.lm,
            num_candidates=10,
            init_temperature=0.7
        )
        
        # Compile the retriever
        self.compiled_retriever = self.optimizer.compile(
            self.retriever,
            trainset=train_examples,
            num_trials=20,
            max_bootstrapped_demos=4,
            max_labeled_demos=8
        )
        
        return self.compiled_retriever
    
    def bootstrap_from_unlabeled(
        self, 
        queries: List[str],
        min_examples: int = 10
    ):
        """Bootstrap optimization from unlabeled queries."""
        bootstrapper = BootstrapFewShot(
            metric=self.relevance_metric,
            max_bootstrapped_demos=min_examples,
            max_rounds=3
        )
        
        # Create pseudo-examples
        unlabeled = [
            dspy.Example(query=q).with_inputs("query") 
            for q in queries
        ]
        
        # Bootstrap and compile
        self.compiled_retriever = bootstrapper.compile(
            self.retriever,
            trainset=unlabeled
        )
        
        return self.compiled_retriever
    
    @staticmethod
    def retrieval_recall_metric(example, prediction, trace=None):
        """Metric for retrieval recall."""
        relevant_ids = set(doc.id for doc in example.relevant_docs)
        retrieved_ids = set(doc.id for doc in prediction.docs)
        
        if not relevant_ids:
            return 0.0
        
        recall = len(relevant_ids & retrieved_ids) / len(relevant_ids)
        return recall
    
    @staticmethod
    def relevance_metric(example, prediction, trace=None):
        """Metric for relevance using LLM judge."""
        judge = dspy.Predict("query, document -> relevance_score")
        
        scores = []
        for doc in prediction.docs[:5]:  # Check top 5
            result = judge(
                query=example.query,
                document=doc.content
            )
            scores.append(float(result.relevance_score))
        
        return sum(scores) / len(scores) if scores else 0.0
```

### Integration with LlamaIndex

```python
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.retrievers import BaseRetriever
import os

class DSPyLlamaIndexRetriever(BaseRetriever):
    """LlamaIndex retriever with DSPy optimization."""
    
    def __init__(
        self, 
        index: VectorStoreIndex,
        enable_dspy: bool = False,
        dspy_model_path: str = None
    ):
        self.index = index
        self.base_retriever = index.as_retriever(similarity_top_k=20)
        self.enable_dspy = enable_dspy
        
        if enable_dspy:
            self._init_dspy(dspy_model_path)
        
        super().__init__()
    
    def _init_dspy(self, model_path: str = None):
        """Initialize DSPy components."""
        self.query_rewriter = QueryRewriter()
        
        # Load compiled model if exists
        if model_path and os.path.exists(model_path):
            self.query_rewriter.load(model_path)
        else:
            # Use default or bootstrap
            self.optimizer = DSPyOptimizer(self)
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not self.enable_dspy:
            return self.base_retriever.retrieve(query_bundle)
        
        # Use DSPy for query optimization
        rewritten = self.query_rewriter(query_bundle.query_str)
        
        # Retrieve with all variants
        all_results = []
        for variant in [rewritten["refined"]] + rewritten["variants"][:3]:
            variant_bundle = QueryBundle(query_str=variant)
            results = self.base_retriever.retrieve(variant_bundle)
            all_results.extend(results)
        
        # Deduplicate and rerank
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.node.node_id not in seen_ids:
                seen_ids.add(result.node.node_id)
                unique_results.append(result)
        
        # Sort by score and return top k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:10]
    
    def optimize_offline(
        self, 
        training_queries: List[Tuple[str, List[str]]]
    ):
        """Offline optimization with training data."""
        if not self.enable_dspy:
            raise ValueError("DSPy not enabled")
        
        examples = []
        for query, relevant_docs in training_queries:
            examples.append(dspy.Example(
                query=query,
                relevant_docs=relevant_docs
            ).with_inputs("query"))
        
        # Optimize
        self.optimizer.optimize_with_examples(examples)
        
        # Save optimized model
        self.query_rewriter.save("models/dspy_optimized_retriever.pkl")
```

### Configuration and Usage

```python
# Configuration
DSPY_CONFIG = {
    "enabled": os.getenv("ENABLE_DSPY", "false").lower() == "true",
    "model": "qwen3:14b",
    "optimization_trials": 20,
    "bootstrap_examples": 10,
    "cache_dir": "cache/dspy",
    "model_save_path": "models/dspy_optimized.pkl"
}

# Usage in main pipeline
def create_retriever(index: VectorStoreIndex) -> BaseRetriever:
    """Create retriever with optional DSPy optimization."""
    
    if DSPY_CONFIG["enabled"]:
        retriever = DSPyLlamaIndexRetriever(
            index=index,
            enable_dspy=True,
            dspy_model_path=DSPY_CONFIG["model_save_path"]
        )
        
        # Bootstrap if no saved model
        if not os.path.exists(DSPY_CONFIG["model_save_path"]):
            print("Bootstrapping DSPy optimization...")
            sample_queries = load_sample_queries()  # Load from logs
            retriever.optimizer.bootstrap_from_unlabeled(
                sample_queries,
                min_examples=DSPY_CONFIG["bootstrap_examples"]
            )
    else:
        retriever = index.as_retriever(similarity_top_k=10)
    
    return retriever
```

## Consequences

### Positive Outcomes

- **Automatic Improvement**: Prompts improve automatically with usage
- **Query Quality**: 20-30% improvement in retrieval metrics
- **Reduced Manual Work**: Eliminates manual prompt engineering
- **Systematic Optimization**: Data-driven approach with metrics
- **Version Control**: Can save and load optimized models

### Negative Consequences / Trade-offs

- **Initial Overhead**: Requires training examples or bootstrapping
- **Complexity**: Adds another framework to learn
- **Optimization Time**: Initial optimization can take minutes
- **Experimental**: DSPy is relatively new, may have edge cases

### Migration Strategy

1. **Feature Flag**: Start with DSPy disabled by default
2. **Data Collection**: Log queries and relevance feedback
3. **Offline Training**: Optimize prompts offline initially
4. **A/B Testing**: Compare DSPy vs baseline performance
5. **Gradual Rollout**: Enable for power users first

## Performance Targets

- **Query Optimization**: <200ms overhead per query
- **Retrieval Quality**: ≥20% improvement in Recall@10
- **Bootstrap Time**: <5 minutes for 100 queries
- **Model Size**: <100MB for saved optimized model

## Dependencies

- **Python**: `dspy-ai>=2.4.0`
- **Integration**: Works alongside LlamaIndex
- **Models**: Requires local LLM for optimization
- **Storage**: Cache directory for compiled models

## Monitoring Metrics

- Query rewriting latency
- Retrieval quality improvements (before/after)
- Cache hit rates for optimized prompts
- User satisfaction metrics
- Optimization convergence rates

## Future Enhancements

- Online learning from user feedback
- Multi-objective optimization (quality + latency)
- Prompt versioning and A/B testing
- Integration with evaluation framework
- Automatic retraining triggers

## Changelog

- **1.0 (2025-08-17)**: Initial DSPy integration design with query rewriting and automatic optimization
