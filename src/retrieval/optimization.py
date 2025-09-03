"""DSPy Progressive Optimization Module.

Implements zero-shot, few-shot, and production optimization strategies
for query improvement with >20% quality improvement target.
"""

import asyncio
import time
from enum import Enum
from typing import Any

import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from loguru import logger
from pydantic import BaseModel, Field

try:
    import dspy
except ImportError:
    logger.warning(
        "DSPy not available or has compatibility issues. "
        "DSPy optimization features will be disabled."
    )
    dspy = None


class OptimizationMode(str, Enum):
    """Optimization modes for DSPy."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    PRODUCTION = "production"


class QueryStrategy(str, Enum):
    """Query strategy types."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"


class DSPyConfig(BaseModel):
    """Configuration for DSPy optimization."""

    mode: OptimizationMode = Field(default=OptimizationMode.ZERO_SHOT)
    max_bootstrapped_demos: int = Field(default=4)
    max_labeled_demos: int = Field(default=4)
    num_threads: int = Field(default=4)
    optimization_metric: str = Field(default="ndcg@10")
    target_improvement: float = Field(default=0.20)
    lm_model: str = Field(default="openai/gpt-4")
    lm_api_base: str = Field(default="http://localhost:11434/v1")
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600)
    num_query_variants: int = Field(default=5)
    min_quality_score: float = Field(default=0.7)
    enable_a_b_testing: bool = Field(default=True)
    a_b_test_sample_size: int = Field(default=100)


class DocMindRAG:
    """DSPy RAG module for DocMind.

    Note: Falls back to basic functionality when DSPy is not available.
    """

    def __init__(
        self,
        index: VectorStoreIndex | None = None,
        retriever: BaseRetriever | None = None,
    ):
        """Initialize DocMindRAG module.

        Args:
            index: LlamaIndex VectorStoreIndex for retrieval
            retriever: Alternative retriever to use instead of index
        """
        if dspy is not None:
            super().__init__()

        self.index = index
        self.retriever = retriever or (
            index.as_retriever(similarity_top_k=5) if index else None
        )

        # Define DSPy signatures if available
        if dspy is not None:
            self.generate_answer = dspy.ChainOfThought("context, question -> answer")
            self.rewrite_query = dspy.ChainOfThought("question -> rewritten_question")
            self.classify_query = dspy.Predict("question -> strategy")
        else:
            self.generate_answer = None
            self.rewrite_query = None
            self.classify_query = None

    def forward(self, question: str) -> dict[str, Any]:
        """Forward pass for RAG.

        Args:
            question: User query

        Returns:
            Dictionary with answer and metadata (compatible with DSPy when available)
        """
        if not self.retriever:
            # Fallback for testing without real retriever
            return {
                "answer": "No retriever configured",
                "context": "",
                "question": question,
                "strategy": "none",
            }

        # Classify query strategy
        if dspy is not None and self.classify_query is not None:
            strategy_pred = self.classify_query(question=question)
            strategy = (
                strategy_pred.strategy
                if hasattr(strategy_pred, "strategy")
                else "factual"
            )
        else:
            # Fallback strategy classification without DSPy
            strategy = self._classify_query_fallback(question)

        # Optionally rewrite query based on strategy
        search_query = question
        if strategy in ["analytical", "comparison"]:
            if dspy is not None and self.rewrite_query is not None:
                rewrite_pred = self.rewrite_query(question=question)
                search_query = (
                    rewrite_pred.rewritten_question
                    if hasattr(rewrite_pred, "rewritten_question")
                    else question
                )
            else:
                # Fallback query rewriting without DSPy
                search_query = self._rewrite_query_fallback(question, strategy)

        # Retrieve context
        nodes = self.retriever.retrieve(search_query)
        context = "\n".join([n.text for n in nodes]) if nodes else ""

        # Generate answer
        if dspy is not None and self.generate_answer is not None:
            pred = self.generate_answer(context=context, question=question)
            answer = pred.answer if hasattr(pred, "answer") else str(pred)
        else:
            # Fallback answer generation without DSPy
            answer = self._generate_answer_fallback(context, question)

        result = {
            "answer": answer,
            "context": context,
            "question": question,
            "strategy": strategy,
            "search_query": search_query,
        }

        # Return DSPy Prediction if available, otherwise return dict
        if dspy is not None:
            pred = dspy.Prediction(**result)
            return pred
        else:
            return result

    def _classify_query_fallback(self, question: str) -> str:
        """Fallback query strategy classification without DSPy.

        Args:
            question: User query

        Returns:
            Strategy classification
        """
        query_lower = question.lower()

        # Pattern matching for strategy classification
        if any(
            word in query_lower
            for word in ["compare", "contrast", "difference", "versus", "vs"]
        ):
            return "comparison"

        if any(
            word in query_lower
            for word in ["analyze", "explain", "why", "how does", "evaluate"]
        ):
            return "analytical"

        if any(
            word in query_lower
            for word in ["relationship", "connect", "relate", "link", "between"]
        ):
            return "relationship"

        # Default to factual
        return "factual"

    def _rewrite_query_fallback(self, question: str, strategy: str) -> str:
        """Fallback query rewriting without DSPy.

        Args:
            question: Original question
            strategy: Query strategy

        Returns:
            Rewritten query
        """
        strategy_templates = {
            "comparison": f"Compare and contrast {question}",
            "analytical": f"Analyze the key aspects of {question}",
            "relationship": f"How does {question} relate to other concepts",
        }
        return strategy_templates.get(strategy, question)

    def _generate_answer_fallback(self, context: str, question: str) -> str:
        """Fallback answer generation without DSPy.

        Args:
            context: Retrieved context
            question: User question

        Returns:
            Basic answer based on context
        """
        if not context:
            return "I don't have enough information to answer this question."

        # Basic answer generation - just return the most relevant context
        context_parts = context.split("\n")
        relevant_parts = [
            part
            for part in context_parts
            if any(
                word.lower() in part.lower() for word in question.lower().split()[:3]
            )
        ]

        if relevant_parts:
            return f"Based on the available information: {relevant_parts[0][:500]}..."
        else:
            return f"Here's what I found: {context[:500]}..."


class DSPyOptimizer:
    """Progressive DSPy optimizer."""

    def __init__(self, config: DSPyConfig | None = None):
        """Initialize optimizer.

        Args:
            config: DSPy configuration
        """
        self.config = config or DSPyConfig()
        self.optimized_module = None
        self.optimization_history = []

        if dspy is not None:
            self._configure_dspy()
        else:
            logger.warning("DSPy not available - optimization features will be limited")

    def _configure_dspy(self) -> None:
        """Configure DSPy settings."""
        if dspy is not None:
            dspy.settings.configure(
                lm=dspy.LM(model=self.config.lm_model, api_base=self.config.lm_api_base)
            )

    def zero_shot_optimize(self, module: DocMindRAG) -> DocMindRAG:
        """Zero-shot optimization without training data.

        Args:
            module: RAG module to optimize

        Returns:
            Optimized module
        """
        if dspy is None:
            logger.warning("DSPy not available - returning module without optimization")
            self.optimized_module = module
            return module

        from dspy.teleprompt import MIPROv2

        # Define simple quality metric
        def quality_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
            """Simple quality metric for optimization."""
            if not prediction or not hasattr(prediction, "answer"):
                return 0.0

            answer = prediction.answer
            # Basic quality checks
            if len(answer) < 20:
                return 0.0
            if len(answer) > 2000:
                return 0.5

            # Check for structure
            has_structure = any(
                word in answer.lower()
                for word in ["first", "second", "additionally", "moreover"]
            )
            return 1.0 if has_structure else 0.7

        # Zero-shot optimization
        optimizer = MIPROv2(
            metric=quality_metric, auto="light", num_threads=self.config.num_threads
        )

        # Compile with empty training set
        optimized = optimizer.compile(
            module, trainset=[], max_bootstrapped_demos=0, max_labeled_demos=0
        )

        self.optimized_module = optimized
        self.optimization_history.append(
            {
                "mode": "zero_shot",
                "timestamp": time.time(),
                "config": self.config.model_dump(),
            }
        )

        return optimized

    def few_shot_optimize(
        self, module: DocMindRAG, examples: list[Any] | None = None
    ) -> DocMindRAG:
        """Few-shot optimization with minimal examples.

        Args:
            module: RAG module to optimize
            examples: Few training examples (5-10) - can be dspy.Example or dict

        Returns:
            Optimized module
        """
        if dspy is None:
            logger.warning("DSPy not available - returning module without optimization")
            self.optimized_module = module
            return module

        from dspy.teleprompt import BootstrapFewShot

        # Define evaluation metric
        def answer_quality_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
            """Evaluate answer quality."""
            if not prediction or not hasattr(prediction, "answer"):
                return 0.0

            # Check if answer addresses the question
            answer = prediction.answer.lower()
            question_words = example.question.lower().split()

            # Simple relevance check
            relevance_score = sum(1 for word in question_words if word in answer) / len(
                question_words
            )

            # Length check
            length_score = 1.0 if 50 < len(answer) < 1000 else 0.5

            return (relevance_score + length_score) / 2

        # Few-shot optimization
        optimizer = BootstrapFewShot(
            metric=answer_quality_metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
        )

        # Split examples for train/val
        train_size = max(1, len(examples) - 2)
        trainset = examples[:train_size]
        valset = examples[train_size:] if len(examples) > train_size else trainset

        # Compile with few examples
        optimized = optimizer.compile(module, trainset=trainset, valset=valset)

        self.optimized_module = optimized
        self.optimization_history.append(
            {
                "mode": "few_shot",
                "timestamp": time.time(),
                "num_examples": len(examples),
                "config": self.config.model_dump(),
            }
        )

        return optimized

    def production_optimize(
        self, module: DocMindRAG, usage_data: list[dict[str, Any]]
    ) -> DocMindRAG:
        """Production optimization with accumulated usage data.

        Args:
            module: RAG module to optimize
            usage_data: Accumulated user interaction data

        Returns:
            Optimized module
        """
        if dspy is None:
            logger.warning("DSPy not available - returning module without optimization")
            self.optimized_module = module
            return module

        # Convert usage data to examples
        examples = []
        for item in usage_data:
            if "question" in item and "answer" in item:
                ex = dspy.Example(
                    question=item["question"], answer=item["answer"]
                ).with_inputs("question")
                examples.append(ex)

        if len(examples) < 20:
            # Not enough data for production optimization, use few-shot
            return self.few_shot_optimize(module, examples)

        from dspy.teleprompt import BootstrapFewShotWithRandomSearch

        # Production metric with more sophisticated evaluation
        def production_metric(example: Any, prediction: Any, trace: Any | None = None) -> float:
            """Production-level quality metric."""
            if not prediction or not hasattr(prediction, "answer"):
                return 0.0

            answer = prediction.answer

            # Multiple quality dimensions
            scores = []

            # Relevance
            question_terms = set(example.question.lower().split())
            answer_terms = set(answer.lower().split())
            overlap = (
                len(question_terms & answer_terms) / len(question_terms)
                if question_terms
                else 0
            )
            scores.append(overlap)

            # Completeness
            completeness = 1.0 if 100 < len(answer) < 1500 else 0.5
            scores.append(completeness)

            # Structure
            structure = (
                1.0
                if any(marker in answer for marker in ["1.", "2.", "•", "-"])
                else 0.7
            )
            scores.append(structure)

            return np.mean(scores)

        # Use random search for robust optimization
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=production_metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
            num_candidate_programs=10,
            num_threads=self.config.num_threads,
        )

        # Split data
        train_size = int(0.8 * len(examples))
        trainset = examples[:train_size]
        valset = examples[train_size:]

        # Compile with production data
        optimized = optimizer.compile(module, trainset=trainset, valset=valset)

        self.optimized_module = optimized
        self.optimization_history.append(
            {
                "mode": "production",
                "timestamp": time.time(),
                "num_examples": len(examples),
                "config": self.config.model_dump(),
            }
        )

        return optimized

    def generate_query_variants(self, query: str) -> list[dict[str, Any]]:
        """Generate optimized query variants.

        Args:
            query: Original query

        Returns:
            List of query variants with scores
        """
        module = DocMindRAG() if not self.optimized_module else self.optimized_module

        variants = []

        # Classify query strategy
        strategy_pred = module.classify_query(question=query)
        base_strategy = (
            strategy_pred.strategy if hasattr(strategy_pred, "strategy") else "factual"
        )

        # Generate variants based on strategy
        strategies = [
            ("comparison", f"Compare and contrast {query}"),
            ("analytical", f"Analyze the key aspects of {query}"),
            ("factual", f"What are the facts about {query}"),
            ("relationship", f"How does {query} relate to other concepts"),
            ("specific", f"Provide specific details about {query}"),
        ]

        for strategy_name, variant_query in strategies:
            score = (
                0.9
                if strategy_name == base_strategy
                else 0.7 + np.random.random() * 0.2
            )

            variants.append(
                {
                    "query": variant_query,
                    "score": score,
                    "strategy": strategy_name,
                    "reasoning": f"{strategy_name} approach",
                }
            )

        # Sort by score and limit to configured number
        variants.sort(key=lambda x: x["score"], reverse=True)
        return variants[: self.config.num_query_variants]


class DSPyABTest:
    """A/B testing framework for DSPy optimization."""

    def __init__(
        self,
        baseline: DocMindRAG,
        optimized: DocMindRAG,
        config: DSPyConfig | None = None,
    ):
        """Initialize A/B test.

        Args:
            baseline: Baseline RAG module
            optimized: Optimized RAG module
            config: DSPy configuration
        """
        self.baseline = baseline
        self.optimized = optimized
        self.config = config or DSPyConfig()
        self.metrics = {"baseline": [], "optimized": []}
        self.results = {}

    def run_comparison(self, test_queries: list[str]) -> dict[str, Any]:
        """Run A/B comparison on test queries.

        Args:
            test_queries: List of test queries

        Returns:
            Comparison results with metrics
        """
        for query in test_queries:
            # Run baseline
            start_time = time.time()
            baseline_result = self.baseline(question=query)
            baseline_latency = time.time() - start_time

            # Run optimized
            start_time = time.time()
            optimized_result = self.optimized(question=query)
            optimized_latency = time.time() - start_time

            # Store metrics
            self.metrics["baseline"].append(
                {
                    "query": query,
                    "answer": baseline_result.answer
                    if hasattr(baseline_result, "answer")
                    else "",
                    "latency": baseline_latency,
                }
            )

            self.metrics["optimized"].append(
                {
                    "query": query,
                    "answer": optimized_result.answer
                    if hasattr(optimized_result, "answer")
                    else "",
                    "latency": optimized_latency,
                    "strategy": optimized_result.strategy
                    if hasattr(optimized_result, "strategy")
                    else None,
                }
            )

        # Calculate aggregate metrics
        self.results = self._calculate_metrics()
        return self.results

    def _calculate_metrics(self) -> dict[str, Any]:
        """Calculate comparison metrics.

        Returns:
            Metrics dictionary
        """
        baseline_latencies = [m["latency"] for m in self.metrics["baseline"]]
        optimized_latencies = [m["latency"] for m in self.metrics["optimized"]]

        baseline_answer_lengths = [len(m["answer"]) for m in self.metrics["baseline"]]
        optimized_answer_lengths = [len(m["answer"]) for m in self.metrics["optimized"]]

        # Calculate quality improvement (simplified)
        baseline_quality = np.mean(
            [1.0 if len(m["answer"]) > 50 else 0.5 for m in self.metrics["baseline"]]
        )
        optimized_quality = np.mean(
            [1.0 if len(m["answer"]) > 50 else 0.5 for m in self.metrics["optimized"]]
        )

        quality_improvement = (
            (optimized_quality - baseline_quality) / baseline_quality
            if baseline_quality > 0
            else 0
        )

        return {
            "baseline_avg_latency": np.mean(baseline_latencies),
            "optimized_avg_latency": np.mean(optimized_latencies),
            "baseline_avg_answer_length": np.mean(baseline_answer_lengths),
            "optimized_avg_answer_length": np.mean(optimized_answer_lengths),
            "quality_improvement": quality_improvement,
            "latency_reduction": (
                np.mean(baseline_latencies) - np.mean(optimized_latencies)
            )
            / np.mean(baseline_latencies)
            if baseline_latencies
            else 0,
            "num_queries_tested": len(self.metrics["baseline"]),
            "improvement_achieved": quality_improvement
            >= self.config.target_improvement,
        }

    async def run_async_comparison(self, test_queries: list[str]) -> dict[str, Any]:
        """Run asynchronous A/B comparison.

        Args:
            test_queries: List of test queries

        Returns:
            Comparison results
        """
        # Convert to async execution
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_comparison, test_queries)


def measure_quality_improvement(
    baseline_results: list[dict], optimized_results: list[dict], metric: str = "ndcg@10"
) -> float:
    """Measure quality improvement between baseline and optimized results.

    Args:
        baseline_results: Baseline retrieval results
        optimized_results: Optimized retrieval results
        metric: Metric to use for comparison

    Returns:
        Quality improvement percentage
    """
    if not baseline_results or not optimized_results:
        return 0.0

    # Simple NDCG approximation based on result relevance
    def calculate_ndcg(results: list[dict], k: int = 10) -> float:
        """Calculate simplified NDCG@k."""
        scores = []
        for i, result in enumerate(results[:k]):
            # Use score if available, otherwise use position-based relevance
            relevance = result["score"] if "score" in result else 1.0 / (i + 1)

            # DCG calculation
            if i == 0:
                scores.append(relevance)
            else:
                scores.append(relevance / np.log2(i + 2))

        return sum(scores) / k if scores else 0.0

    # Calculate metrics
    if metric == "ndcg@10":
        baseline_score = np.mean(
            [
                calculate_ndcg(r.get("results", []))
                for r in baseline_results
                if "results" in r
            ]
        )
        optimized_score = np.mean(
            [
                calculate_ndcg(r.get("results", []))
                for r in optimized_results
                if "results" in r
            ]
        )
    else:
        # Fallback to simple accuracy
        baseline_score = np.mean(
            [1.0 if r.get("answer") else 0.0 for r in baseline_results]
        )
        optimized_score = np.mean(
            [1.0 if r.get("answer") else 0.0 for r in optimized_results]
        )

    # Calculate improvement
    if baseline_score > 0:
        improvement = (optimized_score - baseline_score) / baseline_score
    else:
        improvement = 0.0

    return improvement


def classify_query_strategy(query: str) -> QueryStrategy:
    """Classify query into strategy type.

    Args:
        query: User query

    Returns:
        Query strategy classification
    """
    query_lower = query.lower()

    # Pattern matching for strategy classification
    if any(
        word in query_lower
        for word in ["compare", "contrast", "difference", "versus", "vs"]
    ):
        return QueryStrategy.COMPARISON

    if any(
        word in query_lower
        for word in ["analyze", "explain", "why", "how does", "evaluate"]
    ):
        return QueryStrategy.ANALYTICAL

    if any(
        word in query_lower
        for word in ["relationship", "connect", "relate", "link", "between"]
    ):
        return QueryStrategy.RELATIONSHIP

    # Default to factual
    return QueryStrategy.FACTUAL


async def progressive_optimization_pipeline(
    index: VectorStoreIndex,
    queries: list[str] | None = None,
    config: DSPyConfig | None = None,
) -> tuple[DocMindRAG, dict[str, Any]]:
    """Run progressive optimization pipeline.

    Args:
        index: Vector store index for retrieval
        queries: Optional training queries
        config: DSPy configuration

    Returns:
        Optimized module and metrics
    """
    config = config or DSPyConfig()
    optimizer = DSPyOptimizer(config)

    # Create base module
    base_module = DocMindRAG(index)

    # Phase 1: Zero-shot optimization
    optimized_module = optimizer.zero_shot_optimize(base_module)

    # Phase 2: Few-shot if examples available
    if queries and len(queries) >= 5 and dspy is not None:
        examples = [
            dspy.Example(question=q, answer="").with_inputs("question")
            for q in queries[:10]
        ]
        optimized_module = optimizer.few_shot_optimize(optimized_module, examples)
    elif queries and len(queries) >= 5:
        # Fallback when DSPy not available - just return the module
        logger.info("DSPy not available - skipping few-shot optimization")

    # Phase 3: A/B testing if enabled
    metrics = {}
    if config.enable_a_b_testing and queries and dspy is not None:
        ab_test = DSPyABTest(base_module, optimized_module, config)
        metrics = ab_test.run_comparison(queries[: config.a_b_test_sample_size])
    elif config.enable_a_b_testing and queries:
        logger.info("DSPy not available - skipping A/B testing")
        metrics = {"dspy_available": False}

    return optimized_module, metrics
