"""Result synthesis agent for multi-agent coordination system.

This module implements the SynthesisAgent that combines and merges results
from multiple retrieval operations. The agent deduplicates documents, ranks
by relevance, and creates unified result sets optimized for query responses.

Features:
- Multi-source result combination and merging
- Intelligent deduplication by content similarity
- Relevance-based ranking and scoring
- Cross-strategy result fusion
- Performance monitoring under 100ms
- Metadata preservation and aggregation

Example:
    Using the synthesis agent::

        from agents.synthesis import SynthesisAgent

        synthesis_agent = SynthesisAgent(llm)
        result = synthesis_agent.synthesize_results(
            sub_results_list,
            "Compare AI vs ML performance"
        )
        print(f"Synthesized {result.final_count} documents")
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.tools import synthesize_results
from src.config.settings import settings


class SynthesisResult(BaseModel):
    """Result synthesis output with metadata."""

    documents: list[dict[str, Any]] = Field(
        default_factory=list, description="Final synthesized documents"
    )
    original_count: int = Field(description="Original document count before processing")
    final_count: int = Field(description="Final document count after synthesis")
    deduplication_ratio: float = Field(
        description="Ratio of documents retained after deduplication"
    )
    strategies_used: list[str] = Field(
        default_factory=list, description="Retrieval strategies combined"
    )
    processing_time_ms: float = Field(description="Time taken for synthesis")
    confidence_score: float = Field(
        default=0.0, description="Confidence in synthesis quality"
    )
    reasoning: str = Field(default="", description="Explanation of synthesis decisions")
    synthesis_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Detailed synthesis metrics"
    )


class SynthesisAgent:
    """Specialized agent for multi-source result combination and synthesis.

    Combines document results from multiple retrieval operations, removes
    duplicates, ranks by relevance, and creates unified result sets. Preserves
    metadata and provides comprehensive synthesis metrics.

    Synthesis Operations:
    - Cross-source document merging
    - Content-based deduplication with similarity thresholds
    - Relevance ranking based on query alignment
    - Strategy-aware result fusion
    - Metadata aggregation and preservation
    """

    def __init__(self, llm: Any):
        """Initialize synthesis agent.

        Args:
            llm: Language model for synthesis decisions
        """
        self.llm = llm
        self.total_syntheses = 0
        self.synthesis_times = []
        self.deduplication_stats = []

        # Create LangGraph agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=[synthesize_results],
        )

        logger.info("SynthesisAgent initialized")

    def synthesize_results(
        self,
        sub_results: list[dict[str, Any]],
        original_query: str,
        _context: ChatMemoryBuffer | None = None,
        max_documents: int = 10,
        **_kwargs,
    ) -> SynthesisResult:
        """Combine and synthesize results from multiple retrieval operations.

        Merges document results from multiple sources, removes duplicates based
        on content similarity, ranks by relevance to the original query, and
        creates a unified result set with comprehensive metadata.

        Args:
            sub_results: List of retrieval results to combine
            original_query: Original user query for relevance ranking
            context: Optional conversation context
            max_documents: Maximum number of documents in final result
            **kwargs: Additional parameters for synthesis

        Returns:
            SynthesisResult with combined documents and detailed metadata

        Example:
            >>> result = agent.synthesize_results(retrieval_results, "AI ethics")
            >>> print(f"Combined {result.original_count} → {result.final_count}")
        """
        start_time = time.perf_counter()
        self.total_syntheses += 1

        try:
            # Handle empty or invalid input
            if not sub_results or not isinstance(sub_results, list):
                logger.warning("No valid sub-results provided for synthesis")
                return self._empty_synthesis_result(original_query, 0.0)

            # Prepare agent input
            messages = [
                HumanMessage(content=f"Synthesize results for query: {original_query}")
            ]

            # Include context and configuration
            # Note: state not used directly but kept for future context handling

            # Execute synthesis through agent
            result = self.agent.invoke(
                {"messages": messages},
                {"recursion_limit": 3},  # Limit iterations for performance
            )

            # Parse agent response
            synthesis_data = self._parse_agent_response(
                result, sub_results, original_query
            )

            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            self.synthesis_times.append(processing_time)

            # Build comprehensive result
            result_data = self._build_synthesis_result(
                synthesis_data,
                sub_results,
                original_query,
                max_documents,
                processing_time,
            )

            synthesis_result = SynthesisResult(**result_data)

            # Update statistics
            if synthesis_result.original_count > 0:
                dedup_ratio = synthesis_result.deduplication_ratio
                self.deduplication_stats.append(dedup_ratio)

            logger.info(
                f"Synthesized {synthesis_result.original_count} → "
                f"{synthesis_result.final_count} documents "
                f"({processing_time * 1000:.1f}ms)"
            )

            return synthesis_result

        except (ValueError, TypeError, RuntimeError, KeyError) as e:
            logger.error(f"Result synthesis failed: {e}")

            # Fallback synthesis
            processing_time = time.perf_counter() - start_time
            return self._fallback_synthesis(
                sub_results, original_query, processing_time, str(e)
            )

    def _parse_agent_response(
        self, result: dict, sub_results: list, original_query: str
    ) -> dict[str, Any]:
        """Parse agent response to extract synthesis data."""
        try:
            # Get the final message from agent
            messages = result.get("messages", [])
            if not messages:
                raise ValueError("No messages in agent response")

            last_message = messages[-1]
            content = getattr(last_message, "content", str(last_message))

            # Try to parse JSON from agent response
            try:
                synthesis_data = json.loads(content)
                if isinstance(synthesis_data, dict) and "documents" in synthesis_data:
                    return synthesis_data
            except json.JSONDecodeError:
                pass

            # If JSON parsing failed, use fallback synthesis
            logger.warning("Could not parse JSON from agent, using fallback synthesis")
            return self._execute_fallback_synthesis(sub_results, original_query)

        except Exception as e:
            logger.error(f"Failed to parse agent response: {e}")
            return self._execute_fallback_synthesis(sub_results, original_query)

    def _execute_fallback_synthesis(
        self, sub_results: list, original_query: str
    ) -> dict[str, Any]:
        """Execute fallback synthesis when agent fails."""
        try:
            all_documents = []
            strategies_used = set()
            total_processing_time = 0

            # Collect all documents from sub-results
            for result in sub_results:
                if isinstance(result, dict):
                    if "documents" in result:
                        all_documents.extend(result["documents"])
                    if "strategy_used" in result:
                        strategies_used.add(result["strategy_used"])
                    if "processing_time_ms" in result:
                        total_processing_time += result["processing_time_ms"]

            # Deduplicate documents
            unique_documents = self._deduplicate_documents(all_documents)

            # Rank by relevance
            ranked_documents = self._rank_by_relevance(unique_documents, original_query)

            # Limit results
            max_results = getattr(settings, "synthesis_max_docs", 10)
            final_documents = ranked_documents[:max_results]

            return {
                "documents": final_documents,
                "synthesis_metadata": {
                    "original_count": len(all_documents),
                    "after_deduplication": len(unique_documents),
                    "final_count": len(final_documents),
                    "strategies_used": list(strategies_used),
                    "deduplication_ratio": len(unique_documents)
                    / max(len(all_documents), 1),
                    "total_retrieval_time_ms": total_processing_time,
                },
                "original_query": original_query,
            }

        except Exception as e:
            logger.error(f"Fallback synthesis also failed: {e}")
            return {
                "documents": [],
                "synthesis_metadata": {
                    "original_count": 0,
                    "after_deduplication": 0,
                    "final_count": 0,
                    "strategies_used": [],
                    "error": str(e),
                },
                "original_query": original_query,
            }

    def _deduplicate_documents(self, documents: list[dict]) -> list[dict]:
        """Remove duplicate documents based on content similarity."""
        if not documents:
            return []

        unique_documents = []
        seen_content = set()
        similarity_threshold = 0.8  # 80% content overlap threshold

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            # Extract content for comparison
            content = doc.get("content", doc.get("text", ""))
            if not content:
                continue

            content_words = set(content.lower().split())
            if not content_words:
                continue

            # Check for substantial overlap with existing documents
            is_duplicate = False
            for seen_words in seen_content:
                if len(content_words) == 0:
                    break

                overlap_ratio = len(content_words.intersection(seen_words)) / len(
                    content_words
                )
                if overlap_ratio > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_documents.append(doc)
                seen_content.add(frozenset(content_words))

        return unique_documents

    def _rank_by_relevance(self, documents: list[dict], query: str) -> list[dict]:
        """Rank documents by relevance to query."""
        if not documents or not query:
            return documents

        query_words = set(query.lower().split())
        scored_documents = []

        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            content_words = set(content.lower().split())

            # Calculate relevance score
            if not content_words:
                relevance_score = 0.0
            else:
                # Word overlap score
                word_overlap = len(query_words.intersection(content_words))
                overlap_score = word_overlap / max(len(query_words), 1)

                # Content coverage score
                coverage_score = (
                    word_overlap / len(content_words) if content_words else 0
                )

                # Combined relevance score
                relevance_score = overlap_score + (coverage_score * 0.5)

            # Boost existing score if available
            existing_score = doc.get("score", doc.get("relevance_score", 1.0))
            final_score = relevance_score * existing_score

            # Create scored document
            scored_doc = doc.copy()
            scored_doc["relevance_score"] = final_score
            scored_documents.append(scored_doc)

        # Sort by relevance score descending
        scored_documents.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return scored_documents

    def _build_synthesis_result(
        self,
        synthesis_data: dict,
        sub_results: list,
        original_query: str,
        max_documents: int,
        processing_time: float,
    ) -> dict[str, Any]:
        """Build comprehensive synthesis result."""
        documents = synthesis_data.get("documents", [])
        metadata = synthesis_data.get("synthesis_metadata", {})

        # Extract metrics
        original_count = metadata.get("original_count", len(documents))
        final_count = min(len(documents), max_documents)

        # Apply document limit
        final_documents = documents[:max_documents] if documents else []

        # Calculate confidence
        confidence = self._calculate_confidence_score(
            original_count, final_count, metadata, sub_results
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            metadata, sub_results, original_count, final_count
        )

        return {
            "documents": final_documents,
            "original_count": original_count,
            "final_count": final_count,
            "deduplication_ratio": metadata.get("deduplication_ratio", 1.0),
            "strategies_used": metadata.get("strategies_used", []),
            "processing_time_ms": round(processing_time * 1000, 2),
            "confidence_score": confidence,
            "reasoning": reasoning,
            "synthesis_metadata": metadata,
        }

    def _calculate_confidence_score(
        self, original_count: int, final_count: int, metadata: dict, sub_results: list
    ) -> float:
        """Calculate confidence in synthesis quality."""
        confidence = 0.5  # Base confidence

        # Document availability factor
        if final_count >= 5:
            confidence += 0.2
        elif final_count >= 2:
            confidence += 0.1
        elif final_count >= 1:
            confidence += 0.05

        # Deduplication effectiveness
        dedup_ratio = metadata.get("deduplication_ratio", 1.0)
        if 0.5 <= dedup_ratio <= 0.9:  # Good deduplication
            confidence += 0.1
        elif dedup_ratio > 0.9:  # Little duplication found
            confidence += 0.05

        # Source diversity
        strategies = metadata.get("strategies_used", [])
        if len(strategies) > 1:
            confidence += 0.1

        # Error handling
        if "error" not in metadata:
            confidence += 0.1

        # Source result quality
        if len(sub_results) >= 2:
            confidence += 0.05

        return min(confidence, 1.0)

    def _generate_reasoning(
        self, metadata: dict, sub_results: list, original_count: int, final_count: int
    ) -> str:
        """Generate human-readable reasoning for synthesis decisions."""
        reasons = []

        # Source combination
        source_count = len(sub_results)
        if source_count > 1:
            reasons.append(f"Combined results from {source_count} retrieval sources")
        else:
            reasons.append("Single source synthesis")

        # Deduplication
        dedup_ratio = metadata.get("deduplication_ratio", 1.0)
        if dedup_ratio < 0.9:
            removed = original_count - int(original_count * dedup_ratio)
            reasons.append(f"Removed {removed} duplicate documents")

        # Strategy diversity
        strategies = metadata.get("strategies_used", [])
        if strategies:
            strategy_text = ", ".join(strategies)
            reasons.append(f"Strategies: {strategy_text}")

        # Final results
        if final_count > 0:
            reasons.append(f"Finalized {final_count} most relevant documents")
        else:
            reasons.append("No documents met synthesis criteria")

        # Error handling
        if "error" in metadata:
            reasons.append(f"Handled synthesis error: {metadata['error']}")

        return "; ".join(reasons)

    def _empty_synthesis_result(
        self, query: str, processing_time: float
    ) -> SynthesisResult:
        """Create empty synthesis result."""
        return SynthesisResult(
            documents=[],
            original_count=0,
            final_count=0,
            deduplication_ratio=1.0,
            strategies_used=[],
            processing_time_ms=round(processing_time * 1000, 2),
            confidence_score=0.0,
            reasoning="No input results provided for synthesis",
            synthesis_metadata={"error": "No input results"},
        )

    def _fallback_synthesis(
        self, sub_results: list, original_query: str, processing_time: float, error: str
    ) -> SynthesisResult:
        """Create fallback synthesis result when synthesis fails."""
        # Attempt basic combination
        try:
            all_docs = []
            for result in sub_results:
                if isinstance(result, dict) and "documents" in result:
                    all_docs.extend(result["documents"])

            return SynthesisResult(
                documents=all_docs[:5],  # Limit to first 5
                original_count=len(all_docs),
                final_count=min(len(all_docs), 5),
                deduplication_ratio=1.0,  # No deduplication
                strategies_used=["fallback"],
                processing_time_ms=round(processing_time * 1000, 2),
                confidence_score=0.3,  # Low confidence
                reasoning=f"Fallback synthesis due to error: {error}",
                synthesis_metadata={"error": error, "fallback": True},
            )
        except Exception:
            return self._empty_synthesis_result(original_query, processing_time)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get synthesis performance statistics.

        Returns:
            Dictionary with synthesis performance metrics
        """
        if not self.synthesis_times:
            return {
                "total_syntheses": self.total_syntheses,
                "avg_synthesis_time_ms": 0.0,
                "max_synthesis_time_ms": 0.0,
                "min_synthesis_time_ms": 0.0,
                "avg_deduplication_ratio": 0.0,
            }

        avg_time = sum(self.synthesis_times) / len(self.synthesis_times)
        max_time = max(self.synthesis_times)
        min_time = min(self.synthesis_times)

        avg_dedup = (
            sum(self.deduplication_stats) / len(self.deduplication_stats)
            if self.deduplication_stats
            else 1.0
        )

        return {
            "total_syntheses": self.total_syntheses,
            "avg_synthesis_time_ms": round(avg_time * 1000, 2),
            "max_synthesis_time_ms": round(max_time * 1000, 2),
            "min_synthesis_time_ms": round(min_time * 1000, 2),
            "performance_target_met": avg_time < 0.1,  # 100ms target
            "avg_deduplication_ratio": round(avg_dedup, 3),
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_syntheses = 0
        self.synthesis_times = []
        self.deduplication_stats = []
        logger.info("Synthesis performance stats reset")


# Factory function for backward compatibility
def create_synthesis_agent(llm: Any) -> SynthesisAgent:
    """Create synthesis agent instance.

    Args:
        llm: Language model for synthesis decisions

    Returns:
        Configured SynthesisAgent instance
    """
    return SynthesisAgent(llm)


# Synthesis utilities
def calculate_content_similarity(doc1: dict, doc2: dict) -> float:
    """Calculate content similarity between two documents.

    Args:
        doc1: First document
        doc2: Second document

    Returns:
        Similarity score between 0 and 1
    """
    content1 = doc1.get("content", doc1.get("text", ""))
    content2 = doc2.get("content", doc2.get("text", ""))

    if not content1 or not content2:
        return 0.0

    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


def merge_document_metadata(documents: list[dict]) -> dict[str, Any]:
    """Merge metadata from multiple documents.

    Args:
        documents: List of documents to merge metadata from

    Returns:
        Combined metadata dictionary
    """
    merged_metadata = {}

    for doc in documents:
        if isinstance(doc, dict) and "metadata" in doc:
            doc_metadata = doc["metadata"]
            if isinstance(doc_metadata, dict):
                for key, value in doc_metadata.items():
                    if key not in merged_metadata:
                        merged_metadata[key] = value
                    elif isinstance(value, list):
                        if isinstance(merged_metadata[key], list):
                            merged_metadata[key].extend(value)
                        else:
                            merged_metadata[key] = [merged_metadata[key]] + value

    return merged_metadata
