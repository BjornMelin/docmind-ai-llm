"""Shared tool functions for multi-agent coordination system.

This module provides shared @tool functions that agents use for coordinated
document retrieval and analysis. Tools are designed for use with LangGraph
agents and include proper type hints and documentation.

Features:
- Query routing and complexity analysis
- Query planning and decomposition
- Multi-strategy document retrieval
- Result synthesis and combination
- Response validation and quality scoring
- Context-aware processing with conversation memory
- Fallback mechanisms and error handling

Example:
    Using tools in an agent::

        from agents.tools import route_query, retrieve_documents
        from langgraph.prebuilt import create_react_agent

        tools = [route_query, retrieve_documents]
        agent = create_react_agent(llm, tools)
"""

import json
import time
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from llama_index.core import Document
from loguru import logger

from src.agents.tool_factory import ToolFactory
from src.models.core import settings


@tool
def route_query(
    query: str,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """Analyze query and determine optimal processing strategy.

    Evaluates query complexity, intent, and requirements to route to the most
    appropriate retrieval strategy. Uses pattern matching and heuristics to
    classify queries as simple (direct lookup), medium (multi-step), or
    complex (requiring decomposition).

    Args:
        query: User query to analyze and route
        state: LangGraph state containing context and configuration

    Returns:
        JSON string containing routing decision with strategy, complexity,
        confidence score, and planning requirements

    Example:
        >>> result = route_query("What is the capital of France?")
        >>> decision = json.loads(result)
        >>> print(decision["strategy"])  # "vector"
        >>> print(decision["complexity"])  # "simple"
    """
    try:
        start_time = time.perf_counter()

        # Extract context from state if available
        context = state.get("context") if state else None
        previous_queries = []
        if context and hasattr(context, "chat_history"):
            previous_queries = [msg.content for msg in context.chat_history[-3:]]

        # Analyze query characteristics
        query_lower = query.lower().strip()
        word_count = len(query.split())

        # Complex query indicators
        complex_patterns = [
            "compare",
            "contrast",
            "difference",
            "vs",
            "versus",
            "analyze",
            "breakdown",
            "explain how",
            "step by step",
            "process of",
            "relationship between",
            "impact of",
            "cause and effect",
        ]

        # Medium complexity indicators
        medium_patterns = [
            "find",
            "search",
            "look for",
            "tell me about",
            "what is",
            "how does",
            "why",
            "when",
            "where",
            "list",
            "show me",
        ]

        # Simple query indicators
        simple_patterns = ["define", "what is", "who is", "when was", "where is"]

        # Determine complexity
        complexity = "simple"
        strategy = "vector"
        needs_planning = False
        confidence = 0.8

        if (
            any(pattern in query_lower for pattern in complex_patterns)
            or word_count > 20
        ):
            complexity = "complex"
            strategy = "hybrid"
            needs_planning = True
            confidence = 0.9
        elif (
            any(pattern in query_lower for pattern in medium_patterns)
            or word_count > 10
        ):
            complexity = "medium"
            strategy = "hybrid"
            needs_planning = False
            confidence = 0.85
        elif any(pattern in query_lower for pattern in simple_patterns):
            complexity = "simple"
            strategy = "vector"
            needs_planning = False
            confidence = 0.95

        # Check for context dependencies
        context_indicators = ["this", "that", "it", "they", "them", "above", "previous"]
        if (
            any(indicator in query_lower for indicator in context_indicators)
            and not previous_queries
        ):
            # Contextual query without previous context
            confidence *= 0.8  # Lower confidence without context

        # GraphRAG strategy for relationship queries
        if any(
            pattern in query_lower
            for pattern in ["connect", "relationship", "network", "link"]
        ):
            strategy = "graphrag"

        processing_time = time.perf_counter() - start_time

        decision = {
            "strategy": strategy,
            "complexity": complexity,
            "needs_planning": needs_planning,
            "confidence": confidence,
            "processing_time_ms": round(processing_time * 1000, 2),
            "word_count": word_count,
            "context_dependent": bool(previous_queries),
        }

        logger.info(f"Query routed: {complexity} complexity, {strategy} strategy")
        return json.dumps(decision)

    except Exception as e:
        logger.error(f"Query routing failed: {e}")
        # Fallback decision
        fallback = {
            "strategy": "vector",
            "complexity": "simple",
            "needs_planning": False,
            "confidence": 0.5,
            "error": str(e),
        }
        return json.dumps(fallback)


@tool
def plan_query(
    query: str,
    complexity: str,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """Decompose complex queries into structured sub-tasks.

    Breaks down complex or multi-part queries into manageable sub-tasks that
    can be processed independently or sequentially. Uses query analysis to
    identify key components and create an execution plan.

    Args:
        query: Original user query to decompose
        complexity: Query complexity level (simple/medium/complex)
        state: LangGraph state containing context and configuration

    Returns:
        JSON string containing planning output with sub-tasks, execution order,
        and estimated complexity for each sub-task

    Example:
        >>> result = plan_query("Compare AI vs ML performance", "complex")
        >>> plan = json.loads(result)
        >>> print(plan["sub_tasks"])  # ["Define AI", "Define ML",
        >>> # "Compare performance"]
    """
    try:
        start_time = time.perf_counter()

        if complexity == "simple":
            # Simple queries don't need decomposition
            plan = {
                "original_query": query,
                "sub_tasks": [query],
                "execution_order": "sequential",
                "estimated_complexity": "low",
            }
            return json.dumps(plan)

        # Extract context from state (for future use)
        # context = state.get("context") if state else None

        # Decomposition patterns for different query types
        query_lower = query.lower()
        sub_tasks = []
        execution_order = "sequential"

        # Comparison queries
        if any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            # Extract entities to compare
            parts = query.replace(" vs ", " versus ").split(" versus ")
            if len(parts) == 1:
                parts = query.replace(" and ", " | ").split(" | ")

            if len(parts) >= 2:
                entity1, entity2 = parts[0].strip(), parts[1].strip()
                sub_tasks = [
                    f"Find information about {entity1}",
                    f"Find information about {entity2}",
                    f"Compare {entity1} and {entity2}",
                    f"Summarize key differences between {entity1} and {entity2}",
                ]
                execution_order = "parallel"
            else:
                # Fallback for unclear comparison
                sub_tasks = [
                    f"Identify key concepts in: {query}",
                    "Research each concept separately",
                    "Compare and contrast the concepts",
                ]

        # Analysis queries
        elif any(word in query_lower for word in ["analyze", "analysis", "breakdown"]):
            sub_tasks = [
                f"Identify key components of: {query}",
                "Research background information",
                "Analyze relationships and patterns",
                "Synthesize findings and insights",
            ]

        # Process/explanation queries
        elif any(word in query_lower for word in ["how", "process", "step", "explain"]):
            sub_tasks = [
                "Find definition and overview of the topic",
                "Identify key steps or components",
                "Research detailed explanations",
                "Organize information in logical sequence",
            ]

        # List/enumeration queries
        elif any(
            word in query_lower for word in ["list", "enumerate", "examples", "types"]
        ):
            sub_tasks = [
                f"Find comprehensive information about: {query}",
                "Extract and categorize relevant items",
                "Organize findings into structured list",
            ]

        # Default decomposition for complex queries
        else:
            # Split on common connectors
            connectors = [" and ", " or ", " also ", " additionally ", " furthermore "]
            parts = [query]

            for connector in connectors:
                new_parts = []
                for part in parts:
                    new_parts.extend(part.split(connector))
                parts = new_parts

            if len(parts) > 1:
                sub_tasks = [part.strip() for part in parts if part.strip()]
                sub_tasks.append("Synthesize information from all parts")
            else:
                # Single complex query - break into research phases
                sub_tasks = [
                    f"Research background information for: {query}",
                    "Find detailed analysis and examples",
                    "Synthesize comprehensive response",
                ]

        # Ensure we have at least one sub-task
        if not sub_tasks:
            sub_tasks = [query]

        processing_time = time.perf_counter() - start_time

        plan = {
            "original_query": query,
            "sub_tasks": sub_tasks,
            "execution_order": execution_order,
            "estimated_complexity": "high" if len(sub_tasks) > 3 else "medium",
            "processing_time_ms": round(processing_time * 1000, 2),
            "task_count": len(sub_tasks),
        }

        logger.info(
            f"Query planned: {len(sub_tasks)} sub-tasks, {execution_order} execution"
        )
        return json.dumps(plan)

    except Exception as e:
        logger.error(f"Query planning failed: {e}")
        # Fallback plan
        fallback = {
            "original_query": query,
            "sub_tasks": [query],
            "execution_order": "sequential",
            "estimated_complexity": "medium",
            "error": str(e),
        }
        return json.dumps(fallback)


@tool
def retrieve_documents(
    query: str,
    strategy: str = "hybrid",
    use_dspy: bool = True,
    use_graphrag: bool = False,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """Execute document retrieval using specified strategy and optimizations.

    Retrieves relevant documents using vector search, hybrid search, or graph RAG
    based on the specified strategy. Includes DSPy optimization for query rewriting
    and supports multiple retrieval backends.

    Args:
        query: Search query for document retrieval
        strategy: Retrieval strategy ("vector", "hybrid", "graphrag")
        use_dspy: Whether to use DSPy query optimization
        use_graphrag: Whether to use GraphRAG for entity relationships
        state: LangGraph state containing indexes and configuration

    Returns:
        JSON string containing retrieved documents with metadata including
        relevance scores, source information, and retrieval timing

    Example:
        >>> result = retrieve_documents("machine learning", "hybrid")
        >>> docs = json.loads(result)
        >>> print(f"Found {len(docs['documents'])} documents")
    """
    try:
        start_time = time.perf_counter()

        # Get retrieval tools from state
        tools_data = state.get("tools_data") if state else None
        if not tools_data:
            logger.warning("No tools data available in state, using fallback")
            return json.dumps(
                {
                    "documents": [],
                    "error": "No retrieval tools available",
                    "strategy_used": strategy,
                    "query_optimized": query,
                }
            )

        # Extract index components
        vector_index = tools_data.get("vector")
        kg_index = tools_data.get("kg")
        retriever = tools_data.get("retriever")

        # DSPy query optimization (mock implementation)
        optimized_query = query
        if use_dspy:
            # In a real implementation, this would use DSPy for query rewriting
            if len(query.split()) < 3:
                optimized_query = f"Find documents about {query}"
            logger.debug(f"DSPy optimization: '{query}' -> '{optimized_query}'")

        # Select retrieval strategy
        documents = []
        strategy_used = strategy

        if strategy == "graphrag" and use_graphrag and kg_index:
            # Use knowledge graph retrieval
            tool = ToolFactory.create_kg_search_tool(kg_index)
            if tool:
                try:
                    result = tool.call(optimized_query)
                    # Parse result to extract documents
                    documents = _parse_tool_result(result)
                    logger.info(f"GraphRAG retrieved {len(documents)} documents")
                except Exception as e:
                    logger.warning(f"GraphRAG failed, falling back to hybrid: {e}")
                    strategy_used = "hybrid"

        if strategy in ["hybrid", "vector"] or (
            strategy == "graphrag" and not documents
        ):
            # Use hybrid or vector retrieval
            if strategy == "hybrid" and retriever:
                tool = ToolFactory.create_hybrid_search_tool(retriever)
                strategy_used = "hybrid_fusion"
            elif vector_index:
                if strategy == "hybrid":
                    tool = ToolFactory.create_hybrid_vector_tool(vector_index)
                    strategy_used = "hybrid_vector"
                else:
                    tool = ToolFactory.create_vector_search_tool(vector_index)
                    strategy_used = "vector"
            else:
                logger.error("No vector index available for retrieval")
                return json.dumps(
                    {
                        "documents": [],
                        "error": "No vector index available",
                        "strategy_used": strategy,
                        "query_optimized": optimized_query,
                    }
                )

            try:
                result = tool.call(optimized_query)
                documents = _parse_tool_result(result)
                logger.info(f"{strategy_used} retrieved {len(documents)} documents")
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                documents = []

        processing_time = time.perf_counter() - start_time

        # Format results
        result_data = {
            "documents": documents,
            "strategy_used": strategy_used,
            "query_original": query,
            "query_optimized": optimized_query,
            "document_count": len(documents),
            "processing_time_ms": round(processing_time * 1000, 2),
            "dspy_used": use_dspy,
            "graphrag_used": use_graphrag and strategy == "graphrag",
        }

        return json.dumps(result_data, default=str)

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return json.dumps(
            {
                "documents": [],
                "error": str(e),
                "strategy_used": strategy,
                "query_optimized": query,
            }
        )


@tool
def synthesize_results(
    sub_results: str,
    original_query: str,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """Combine and synthesize results from multiple retrieval operations.

    Merges document results from multiple sub-queries or retrieval strategies,
    removes duplicates, ranks by relevance, and creates a unified result set
    optimized for the original query.

    Args:
        sub_results: JSON string containing list of retrieval results
        original_query: Original user query for context
        state: LangGraph state containing configuration

    Returns:
        JSON string containing synthesized documents with combined metadata,
        deduplication information, and synthesis strategy details

    Example:
        >>> results = synthesize_results(sub_results_json, "AI overview")
        >>> synthesis = json.loads(results)
        >>> print(f"Synthesized {synthesis['final_count']} unique documents")
    """
    try:
        start_time = time.perf_counter()

        # Parse sub-results
        try:
            results_list = (
                json.loads(sub_results) if isinstance(sub_results, str) else sub_results
            )
        except json.JSONDecodeError:
            logger.error("Invalid JSON in sub_results")
            return json.dumps(
                {
                    "documents": [],
                    "error": "Invalid input format",
                    "synthesis_metadata": {},
                }
            )

        all_documents = []
        strategies_used = set()
        total_processing_time = 0

        # Collect all documents from sub-results
        for result in results_list:
            if isinstance(result, dict) and "documents" in result:
                all_documents.extend(result["documents"])
                if "strategy_used" in result:
                    strategies_used.add(result["strategy_used"])
                if "processing_time_ms" in result:
                    total_processing_time += result["processing_time_ms"]

        logger.info(
            f"Synthesizing {len(all_documents)} documents from "
            f"{len(results_list)} sources"
        )

        # Deduplicate documents by content similarity
        unique_documents = []
        seen_content = set()

        for doc in all_documents:
            if isinstance(doc, dict):
                # Create content hash for deduplication
                content = doc.get("content", doc.get("text", ""))
                content_words = set(content.lower().split())

                # Check for substantial overlap with existing documents
                is_duplicate = False
                for seen_words in seen_content:
                    if (
                        len(content_words.intersection(seen_words))
                        / max(len(content_words), 1)
                        > 0.8
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_documents.append(doc)
                    seen_content.add(frozenset(content_words))

        # Rank documents by relevance to original query
        ranked_documents = _rank_documents_by_relevance(
            unique_documents, original_query
        )

        # Limit to top results
        max_results = getattr(settings, "synthesis_max_docs", 10)
        final_documents = ranked_documents[:max_results]

        processing_time = time.perf_counter() - start_time

        synthesis_metadata = {
            "original_count": len(all_documents),
            "after_deduplication": len(unique_documents),
            "final_count": len(final_documents),
            "strategies_used": list(strategies_used),
            "deduplication_ratio": round(
                len(unique_documents) / max(len(all_documents), 1), 2
            ),
            "processing_time_ms": round(processing_time * 1000, 2),
            "total_retrieval_time_ms": total_processing_time,
        }

        result_data = {
            "documents": final_documents,
            "synthesis_metadata": synthesis_metadata,
            "original_query": original_query,
        }

        logger.info(f"Synthesis complete: {len(final_documents)} final documents")
        return json.dumps(result_data, default=str)

    except Exception as e:
        logger.error(f"Result synthesis failed: {e}")
        return json.dumps({"documents": [], "error": str(e), "synthesis_metadata": {}})


@tool
def validate_response(
    query: str,
    response: str,
    sources: str,
    state: Annotated[dict, InjectedState] = None,
) -> str:
    """Validate response quality and accuracy against sources.

    Analyzes the generated response for hallucinations, source attribution,
    completeness, and accuracy. Provides validation scores and identifies
    potential issues that may require response regeneration.

    Args:
        query: Original user query
        response: Generated response to validate
        sources: JSON string containing source documents
        state: LangGraph state containing configuration

    Returns:
        JSON string containing validation results with quality scores,
        identified issues, and suggested actions (accept/regenerate/refine)

    Example:
        >>> result = validate_response(query, response, sources_json)
        >>> validation = json.loads(result)
        >>> print(f"Confidence: {validation['confidence']}")
        >>> print(f"Action: {validation['suggested_action']}")
    """
    try:
        start_time = time.perf_counter()

        # Parse sources
        try:
            source_docs = json.loads(sources) if isinstance(sources, str) else sources
            if isinstance(source_docs, dict) and "documents" in source_docs:
                source_docs = source_docs["documents"]
        except json.JSONDecodeError:
            source_docs = []

        # Validation checks
        issues = []
        confidence = 1.0

        # Check 1: Response length and completeness
        if len(response.strip()) < 50:
            issues.append(
                {
                    "type": "incomplete_response",
                    "severity": "medium",
                    "description": (
                        "Response appears too brief for the query complexity"
                    ),
                }
            )
            confidence *= 0.8

        # Check 2: Source attribution
        source_mentioned = False
        if source_docs:
            # Look for source references in response
            for doc in source_docs[:3]:  # Check first 3 sources
                if isinstance(doc, dict):
                    doc_content = doc.get("content", doc.get("text", ""))
                    # Simple check for content overlap
                    doc_words = set(doc_content.lower().split())
                    response_words = set(response.lower().split())
                    if len(doc_words.intersection(response_words)) > 5:
                        source_mentioned = True
                        break

            if not source_mentioned:
                issues.append(
                    {
                        "type": "missing_source",
                        "severity": "medium",
                        "description": (
                            "Response does not appear to reference provided sources"
                        ),
                    }
                )
                confidence *= 0.7
        else:
            issues.append(
                {
                    "type": "no_sources",
                    "severity": "high",
                    "description": "No source documents provided for validation",
                }
            )
            confidence *= 0.6

        # Check 3: Hallucination detection (basic checks)
        hallucination_indicators = [
            "I cannot find",
            "No information available",
            "According to my knowledge",
            "I don't have access",
            "Based on my training",
        ]

        if any(indicator in response for indicator in hallucination_indicators):
            # These phrases suggest the model is not using provided sources
            issues.append(
                {
                    "type": "potential_hallucination",
                    "severity": "high",
                    "description": (
                        "Response contains phrases suggesting use of training data "
                        "over sources"
                    ),
                }
            )
            confidence *= 0.5

        # Check 4: Query relevance
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance_overlap = len(query_words.intersection(response_words)) / len(
            query_words
        )

        if relevance_overlap < 0.3:
            issues.append(
                {
                    "type": "low_relevance",
                    "severity": "medium",
                    "description": "Response may not adequately address the query",
                }
            )
            confidence *= 0.8

        # Check 5: Response coherence (basic structure check)
        sentences = response.split(". ")
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(
                sentences
            )
            if avg_sentence_length < 3 or avg_sentence_length > 50:
                issues.append(
                    {
                        "type": "coherence_issue",
                        "severity": "low",
                        "description": "Response may have unusual sentence structure",
                    }
                )
                confidence *= 0.9

        # Determine suggested action
        if confidence >= 0.8 and len(issues) <= 1:
            suggested_action = "accept"
        elif confidence >= 0.6:
            suggested_action = "refine"
        else:
            suggested_action = "regenerate"

        processing_time = time.perf_counter() - start_time

        validation_result = {
            "valid": confidence >= 0.6,
            "confidence": round(confidence, 2),
            "issues": issues,
            "suggested_action": suggested_action,
            "processing_time_ms": round(processing_time * 1000, 2),
            "source_count": len(source_docs),
            "response_length": len(response),
            "issue_count": len(issues),
        }

        logger.info(
            f"Response validation: {confidence:.2f} confidence, "
            f"{suggested_action} action"
        )
        return json.dumps(validation_result)

    except Exception as e:
        logger.error(f"Response validation failed: {e}")
        return json.dumps(
            {
                "valid": False,
                "confidence": 0.0,
                "issues": [
                    {
                        "type": "validation_error",
                        "severity": "high",
                        "description": str(e),
                    }
                ],
                "suggested_action": "regenerate",
                "error": str(e),
            }
        )


# Helper functions


def _parse_tool_result(result: Any) -> list[dict[str, Any]]:
    """Parse tool result to extract document list."""
    if isinstance(result, str):
        # Tool returned text response - create mock document
        return [
            {"content": result, "metadata": {"source": "tool_response"}, "score": 1.0}
        ]
    elif hasattr(result, "response"):
        # LlamaIndex response object
        return [
            {
                "content": result.response,
                "metadata": getattr(result, "metadata", {}),
                "score": 1.0,
            }
        ]
    elif isinstance(result, list):
        # List of documents
        documents = []
        for item in result:
            if isinstance(item, Document):
                documents.append(
                    {
                        "content": item.text,
                        "metadata": item.metadata,
                        "score": getattr(item, "score", 1.0),
                    }
                )
            elif isinstance(item, dict):
                documents.append(item)
        return documents
    else:
        # Fallback - convert to string
        return [
            {"content": str(result), "metadata": {"source": "unknown"}, "score": 1.0}
        ]


def _rank_documents_by_relevance(documents: list[dict], query: str) -> list[dict]:
    """Rank documents by relevance to query using simple scoring."""
    query_words = set(query.lower().split())

    scored_docs = []
    for doc in documents:
        content = doc.get("content", doc.get("text", ""))
        content_words = set(content.lower().split())

        # Simple relevance scoring
        word_overlap = len(query_words.intersection(content_words))
        total_words = len(content_words)

        # Calculate relevance score
        relevance_score = word_overlap / max(len(query_words), 1)
        if total_words > 0:
            relevance_score += (word_overlap / total_words) * 0.5

        # Boost score if existing score is available
        existing_score = doc.get("score", 1.0)
        final_score = relevance_score * existing_score

        doc_copy = doc.copy()
        doc_copy["relevance_score"] = final_score
        scored_docs.append(doc_copy)

    # Sort by relevance score descending
    scored_docs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return scored_docs
