# ADR-012: Evaluation with DeepEval

## Title

Leverage DeepEval for All Quality Assurance

## Version/Date

4.0 / 2025-08-19

## Status

Accepted

## Description

Use DeepEval library for all evaluation needs leveraging Qwen3-4B-Instruct-2507's 262K context capability for evaluation scenarios. DeepEval provides RAG evaluation with large context processing, including metrics for the 5-agent system utilizing 262K context windows, multi-stage retrieval quality assessment, DSPy optimization effectiveness measurement, and document evaluation within single context spans without chunking limitations.

## Context

The original ADR-012 had 1000+ lines of custom evaluation code including:

- Custom retrieval evaluators
- Custom generation evaluators  
- Custom metrics calculations
- Custom quality tracking

This is over-engineering when DeepEval already provides these features with simple function calls.

**Enhanced Requirements:**

- **Multi-Agent Evaluation** (ADR-011): Assess the effectiveness of the 5-agent coordination system
- **DSPy Evaluation** (ADR-018): Measure prompt optimization effectiveness and query rewriting quality
- **GraphRAG Assessment** (ADR-019): Evaluate PropertyGraphIndex performance for relationship queries
- **Performance Metrics** (ADR-010): Evaluate cache effectiveness and overall system performance

## Decision

We will use **DeepEval directly** for all evaluation:

### Simple Evaluation Setup

```python
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    LatencyMetric
)
from deepeval.test_case import LLMTestCase

class DocMindEvaluator:
    """Simple wrapper around DeepEval for DocMind evaluation."""
    
    def __init__(self):
        # Initialize metrics with local LLM (Qwen3-14B)
        self.metrics = [
            AnswerRelevancyMetric(threshold=0.7, model="Qwen/Qwen3-14B"),
            FaithfulnessMetric(threshold=0.7, model="Qwen/Qwen3-14B"),
            ContextualPrecisionMetric(threshold=0.7, model="Qwen/Qwen3-14B"),
            ContextualRecallMetric(threshold=0.7, model="Qwen/Qwen3-14B"),
            HallucinationMetric(threshold=0.1, model="Qwen/Qwen3-14B"),
            LatencyMetric(max_latency=3.0),
        ]
        
        # DSPy-specific evaluation metrics (ADR-018)
        self.dspy_metrics = {
            "query_optimization_improvement": 0.0,  # Before/after retrieval quality
            "optimization_time": 0.0,  # Time to optimize queries
            "cache_hit_rate": 0.0,  # Optimized query cache hits
        }
        
        # GraphRAG-specific evaluation metrics (ADR-019)
        self.graphrag_metrics = {
            "relationship_extraction_quality": 0.0,  # Entity/relationship accuracy
            "multi_hop_reasoning_success": 0.0,  # Complex query success rate
            "graph_construction_time": 0.0,  # Time to build PropertyGraphIndex
        }
        
        # Multi-agent coordination metrics (ADR-011)
        self.agent_metrics = {
            "query_router_accuracy": 0.0,  # Routing decision effectiveness
            "query_planner_success": 0.0,  # Planning decomposition quality
            "retrieval_expert_performance": 0.0,  # Retrieval quality with optimizations
            "result_synthesizer_coherence": 0.0,  # Multi-source combination quality
            "response_validator_precision": 0.0,  # Validation accuracy
            "agent_coordination_latency": 0.0,  # Overall orchestration overhead
        }
    
    def evaluate_response(self, query: str, response: str, contexts: List[str], latency: float):
        """Evaluate a single RAG response."""
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=contexts,
            latency=latency
        )
        
        # Run evaluation
        results = evaluate(
            test_cases=[test_case],
            metrics=self.metrics,
            print_results=False,
            use_cache=True
        )
        
        return {
            'passed': results.test_passed,
            'scores': results.scores,
            'reasons': results.reasons
        }
    
    def batch_evaluate(self, test_cases: List[Dict]):
        """Evaluate multiple responses."""
        
        cases = [
            LLMTestCase(
                input=tc['query'],
                actual_output=tc['response'],
                retrieval_context=tc['contexts'],
                latency=tc.get('latency', 0)
            )
            for tc in test_cases
        ]
        
        return evaluate(
            test_cases=cases,
            metrics=self.metrics,
            print_results=True
        )

# Usage in Streamlit
evaluator = DocMindEvaluator()

# After generating response
result = evaluator.evaluate_response(
    query=user_query,
    response=generated_answer,
    contexts=retrieved_docs,
    latency=response_time
)

# Display in UI
if result['passed']:
    st.success(f"✅ Quality Check Passed (Score: {result['scores']['average']:.2f})")
else:
    st.warning(f"⚠️ Quality Issues Detected: {result['reasons']}")
```

### Continuous Monitoring with DeepEval

```python
from deepeval import track

# Track metrics over time
@track(metrics=["answer_relevancy", "faithfulness", "latency"])
def generate_response(query: str) -> str:
    # Your RAG pipeline here
    response = rag_pipeline.query(query)
    return response

# DeepEval automatically tracks and stores metrics
```

### Testing with DeepEval + Pytest

```python
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

def test_rag_quality():
    """Test RAG response quality."""
    
    metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieval_context=["France is a country in Europe. Its capital is Paris."]
    )
    
    assert_test(test_case, [metric])
```

## What DeepEval Provides Out-of-the-Box

1. **RAG Metrics**: All standard RAG evaluation metrics pre-implemented
2. **Local LLM Support**: Works with Ollama, LlamaCpp, any local model
3. **Caching**: Automatic caching of evaluation results
4. **Visualization**: Built-in dashboard and reporting
5. **CI/CD Integration**: Works with pytest, GitHub Actions
6. **Benchmarking**: Compare against standard datasets
7. **Synthetic Data**: Generate test cases automatically
8. **Observability**: Integration with LangSmith, Weights & Biases

## Benefits of Using DeepEval

- **Zero Custom Code**: All evaluation logic is in the library
- **Battle-Tested**: Used by 100+ companies in production
- **Maintained**: Active development and community
- **Comprehensive**: Covers all aspects of RAG evaluation
- **Fast Setup**: 5 minutes to full evaluation suite

## What We Removed

- ❌ 1000+ lines of custom evaluation code
- ❌ Custom metric calculations
- ❌ Custom quality tracking database
- ❌ Custom dashboard code
- ❌ Custom test harnesses

## Alternative: RAGAS

If DeepEval doesn't meet needs, RAGAS is another good choice:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

result = evaluate(
    dataset=your_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
)
```

## Dependencies

```toml
[project.dependencies]
deepeval = "^1.0.0"
# or
ragas = "^0.2.0"
```

## Related Decisions

- **ADR-001** (Modern Agentic RAG Architecture): Provides the 5-agent system architecture being evaluated
- **ADR-010** (Performance Optimization Strategy): Performance metrics complement DeepEval quality metrics
- **ADR-011** (Agent Orchestration Framework): Defines the 5-agent coordination patterns requiring evaluation

## Monitoring Metrics

DeepEval automatically tracks:

- Response relevancy and faithfulness
- Retrieval precision and recall
- Hallucination rates
- Latency percentiles
- Error rates and types

## Changelog

- **3.3 (2025-08-18)**: **REVERTED** - Confirmed Qwen3-14B as evaluation model after rejecting impractical 30B MoE model
- **3.2 (2025-08-18)**: CORRECTED - Updated Qwen3-14B-Instruct to correct official name Qwen3-14B (no separate instruct variant exists)
- **3.1 (2025-08-18)**: Added DSPy-specific evaluation metrics for query optimization effectiveness and GraphRAG evaluation criteria for relationship extraction quality and multi-hop reasoning assessment
- **3.0 (2025-08-17)**: FINALIZED - Updated with Qwen3-14B model selection, accepted status
- **2.0 (2025-08-17)**: SIMPLIFIED - Use DeepEval library instead of custom code
- **1.0 (2025-01-16)**: Original version with 1000+ lines of custom evaluation
