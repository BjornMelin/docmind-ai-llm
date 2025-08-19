"""Response validation agent for multi-agent coordination system.

This module implements the ValidationAgent that analyzes generated responses
for quality, accuracy, and completeness. The agent performs hallucination
detection, source attribution verification, and provides quality scores.

Features:
- Hallucination detection and prevention
- Source attribution verification
- Response completeness analysis
- Quality scoring and confidence assessment
- Issue identification and classification
- Suggested actions (accept/refine/regenerate)
- Performance monitoring under 75ms

Example:
    Using the validation agent::

        from agents.validator import ValidationAgent

        validator = ValidationAgent(llm)
        result = validator.validate_response(
            "What is machine learning?",
            "Machine learning is...",
            source_documents
        )
        print(f"Validation confidence: {result.confidence}")
        print(f"Suggested action: {result.suggested_action}")
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.tools import validate_response


class ValidationIssue(BaseModel):
    """Individual validation issue."""

    type: str = Field(
        description="Issue type (hallucination/missing_source/inaccuracy)"
    )
    severity: str = Field(description="Issue severity (low/medium/high)")
    description: str = Field(description="Human-readable description of the issue")


class ValidationResult(BaseModel):
    """Response validation result with detailed analysis."""

    valid: bool = Field(description="Whether response passes validation")
    confidence: float = Field(description="Validation confidence score (0-1)")
    issues: list[ValidationIssue] = Field(
        default_factory=list, description="Identified issues"
    )
    suggested_action: str = Field(
        description="Suggested action (accept/refine/regenerate)"
    )
    processing_time_ms: float = Field(description="Time taken for validation")
    source_count: int = Field(description="Number of source documents analyzed")
    response_length: int = Field(description="Length of response analyzed")
    quality_scores: dict[str, float] = Field(
        default_factory=dict, description="Detailed quality metrics"
    )
    reasoning: str = Field(
        default="", description="Explanation of validation decisions"
    )


class ValidationAgent:
    """Specialized agent for response quality validation and verification.

    Analyzes generated responses for accuracy, completeness, and quality.
    Performs comprehensive validation including hallucination detection,
    source attribution verification, and relevance assessment.

    Validation Checks:
    - Hallucination detection using source comparison
    - Source attribution and reference verification
    - Response completeness relative to query complexity
    - Content relevance and query alignment
    - Structural coherence and readability
    """

    def __init__(self, llm: Any):
        """Initialize validation agent.

        Args:
            llm: Language model for validation decisions
        """
        self.llm = llm
        self.total_validations = 0
        self.validation_times = []
        self.confidence_scores = []
        self.issue_counts = {
            "hallucination": 0,
            "missing_source": 0,
            "inaccuracy": 0,
            "low_relevance": 0,
        }

        # Create LangGraph agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=[validate_response],
        )

        logger.info("ValidationAgent initialized")

    def validate_response(
        self,
        query: str,
        response: str,
        sources: list[dict[str, Any]],
        context: ChatMemoryBuffer | None = None,
        **kwargs,
    ) -> ValidationResult:
        """Validate response quality and accuracy against sources.

        Performs comprehensive validation of the generated response including
        hallucination detection, source verification, completeness analysis,
        and quality assessment.

        Args:
            query: Original user query
            response: Generated response to validate
            sources: Source documents used for response generation
            context: Optional conversation context
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation scores and detailed analysis

        Example:
            >>> result = validator.validate_response(query, response, sources)
            >>> if result.suggested_action == "accept":
            >>>     print("Response validated successfully")
        """
        start_time = time.perf_counter()
        self.total_validations += 1

        try:
            # Prepare agent input
            messages = [HumanMessage(content=f"Validate response for query: {query}")]

            # Include context and validation data
            # Context will be integrated into agent state for multi-turn conversations

            # Execute validation through agent
            result = self.agent.invoke(
                {"messages": messages},
                {"recursion_limit": 3},  # Limit iterations for performance
            )

            # Parse agent response
            validation_data = self._parse_agent_response(
                result, query, response, sources
            )

            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            self.validation_times.append(processing_time)

            # Build comprehensive result
            result_data = self._build_validation_result(
                validation_data, query, response, sources, processing_time
            )

            validation_result = ValidationResult(**result_data)

            # Update statistics
            self.confidence_scores.append(validation_result.confidence)
            for issue in validation_result.issues:
                if issue.type in self.issue_counts:
                    self.issue_counts[issue.type] += 1

            logger.info(
                f"Response validated: {validation_result.confidence:.2f} confidence, "
                f"{validation_result.suggested_action} action "
                f"({processing_time * 1000:.1f}ms)"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Response validation failed: {e}")

            # Fallback validation
            processing_time = time.perf_counter() - start_time
            return self._fallback_validation(
                query, response, sources, processing_time, str(e)
            )

    def _parse_agent_response(
        self, result: dict, query: str, response: str, sources: list
    ) -> dict[str, Any]:
        """Parse agent response to extract validation data."""
        try:
            # Get the final message from agent
            messages = result.get("messages", [])
            if not messages:
                raise ValueError("No messages in agent response")

            last_message = messages[-1]
            content = getattr(last_message, "content", str(last_message))

            # Try to parse JSON from agent response
            try:
                validation_data = json.loads(content)
                if isinstance(validation_data, dict) and "valid" in validation_data:
                    return validation_data
            except json.JSONDecodeError:
                pass

            # If JSON parsing failed, use fallback validation
            logger.warning("Could not parse JSON from agent, using fallback validation")
            return self._execute_fallback_validation(query, response, sources)

        except Exception as e:
            logger.error(f"Failed to parse agent response: {e}")
            return self._execute_fallback_validation(query, response, sources)

    def _execute_fallback_validation(
        self, query: str, response: str, sources: list
    ) -> dict[str, Any]:
        """Execute fallback validation when agent fails."""
        try:
            issues = []
            confidence = 1.0

            # Basic validation checks

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
            if sources:
                # Look for source references in response
                for doc in sources[:3]:  # Check first 3 sources
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

            # Check 3: Hallucination detection (basic)
            hallucination_indicators = [
                "I cannot find",
                "No information available",
                "According to my knowledge",
                "I don't have access",
                "Based on my training",
            ]

            if any(indicator in response for indicator in hallucination_indicators):
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

            # Check 5: Response coherence
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
                            "description": (
                                "Response may have unusual sentence structure"
                            ),
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

            return {
                "valid": confidence >= 0.6,
                "confidence": confidence,
                "issues": issues,
                "suggested_action": suggested_action,
                "source_count": len(sources),
                "response_length": len(response),
                "issue_count": len(issues),
            }

        except Exception as e:
            logger.error(f"Fallback validation also failed: {e}")
            return {
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
                "source_count": len(sources) if sources else 0,
                "response_length": len(response) if response else 0,
                "error": str(e),
            }

    def _build_validation_result(
        self,
        validation_data: dict,
        query: str,
        response: str,
        sources: list,
        processing_time: float,
    ) -> dict[str, Any]:
        """Build comprehensive validation result."""
        issues_data = validation_data.get("issues", [])
        issues = [
            ValidationIssue(**issue) for issue in issues_data if isinstance(issue, dict)
        ]

        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(
            validation_data, query, response, sources
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            validation_data, query, response, sources, issues
        )

        return {
            "valid": validation_data.get("valid", False),
            "confidence": round(validation_data.get("confidence", 0.0), 2),
            "issues": issues,
            "suggested_action": validation_data.get("suggested_action", "regenerate"),
            "processing_time_ms": round(processing_time * 1000, 2),
            "source_count": validation_data.get("source_count", len(sources)),
            "response_length": validation_data.get("response_length", len(response)),
            "quality_scores": quality_scores,
            "reasoning": reasoning,
        }

    def _calculate_quality_scores(
        self, validation_data: dict, query: str, response: str, sources: list
    ) -> dict[str, float]:
        """Calculate detailed quality metrics."""
        scores = {}

        # Completeness score
        response_length = len(response.split())
        query_complexity = len(query.split())
        expected_length = max(
            50, query_complexity * 5
        )  # Minimum 50 words, scale with complexity
        completeness = min(response_length / expected_length, 1.0)
        scores["completeness"] = round(completeness, 2)

        # Relevance score
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance = len(query_words.intersection(response_words)) / max(
            len(query_words), 1
        )
        scores["relevance"] = round(relevance, 2)

        # Source attribution score
        if sources and response:
            attribution_score = 0.0
            for doc in sources[:3]:
                if isinstance(doc, dict):
                    doc_content = doc.get("content", doc.get("text", ""))
                    doc_words = set(doc_content.lower().split())
                    overlap = len(response_words.intersection(doc_words))
                    if overlap > 5:
                        attribution_score = min(attribution_score + 0.4, 1.0)
            scores["source_attribution"] = round(attribution_score, 2)
        else:
            scores["source_attribution"] = 0.0

        # Coherence score (basic sentence structure analysis)
        sentences = response.split(". ")
        if len(sentences) > 1:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            coherence = 1.0 if 5 <= avg_length <= 25 else 0.7
            scores["coherence"] = coherence
        else:
            scores["coherence"] = 0.8  # Single sentence responses

        # Overall quality score
        weights = {
            "completeness": 0.3,
            "relevance": 0.3,
            "source_attribution": 0.25,
            "coherence": 0.15,
        }
        overall = sum(scores[metric] * weight for metric, weight in weights.items())
        scores["overall"] = round(overall, 2)

        return scores

    def _generate_reasoning(
        self,
        validation_data: dict,
        query: str,
        response: str,
        sources: list,
        issues: list[ValidationIssue],
    ) -> str:
        """Generate human-readable reasoning for validation decisions."""
        reasons = []

        # Confidence reasoning
        confidence = validation_data.get("confidence", 0.0)
        if confidence >= 0.8:
            reasons.append("High confidence validation")
        elif confidence >= 0.6:
            reasons.append("Medium confidence validation")
        else:
            reasons.append("Low confidence validation")

        # Issues reasoning
        if issues:
            issue_types = [issue.type for issue in issues]
            issue_text = ", ".join(set(issue_types))
            reasons.append(f"Issues found: {issue_text}")
        else:
            reasons.append("No significant issues detected")

        # Source reasoning
        source_count = len(sources)
        if source_count > 0:
            reasons.append(f"Validated against {source_count} source documents")
        else:
            reasons.append("No source documents available for verification")

        # Length reasoning
        response_length = len(response.split())
        if response_length < 20:
            reasons.append("Response may be too brief")
        elif response_length > 200:
            reasons.append("Comprehensive response provided")

        # Action reasoning
        action = validation_data.get("suggested_action", "regenerate")
        action_explanations = {
            "accept": "Response meets quality standards",
            "refine": "Response needs minor improvements",
            "regenerate": "Response requires regeneration",
        }
        if action in action_explanations:
            reasons.append(action_explanations[action])

        return "; ".join(reasons)

    def _fallback_validation(
        self,
        query: str,
        response: str,
        sources: list,
        processing_time: float,
        error: str,
    ) -> ValidationResult:
        """Create fallback validation result when validation fails."""
        return ValidationResult(
            valid=False,
            confidence=0.0,
            issues=[
                ValidationIssue(
                    type="validation_error",
                    severity="high",
                    description=f"Validation system error: {error}",
                )
            ],
            suggested_action="regenerate",
            processing_time_ms=round(processing_time * 1000, 2),
            source_count=len(sources) if sources else 0,
            response_length=len(response) if response else 0,
            quality_scores={"overall": 0.0},
            reasoning=f"Validation failed due to system error: {error}",
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get validation performance statistics.

        Returns:
            Dictionary with validation performance metrics
        """
        if not self.validation_times:
            return {
                "total_validations": self.total_validations,
                "avg_validation_time_ms": 0.0,
                "max_validation_time_ms": 0.0,
                "min_validation_time_ms": 0.0,
                "avg_confidence": 0.0,
                "issue_counts": self.issue_counts,
            }

        avg_time = sum(self.validation_times) / len(self.validation_times)
        max_time = max(self.validation_times)
        min_time = min(self.validation_times)
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)

        return {
            "total_validations": self.total_validations,
            "avg_validation_time_ms": round(avg_time * 1000, 2),
            "max_validation_time_ms": round(max_time * 1000, 2),
            "min_validation_time_ms": round(min_time * 1000, 2),
            "performance_target_met": avg_time < 0.075,  # 75ms target
            "avg_confidence": round(avg_confidence, 3),
            "issue_counts": self.issue_counts,
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_validations = 0
        self.validation_times = []
        self.confidence_scores = []
        self.issue_counts = {
            "hallucination": 0,
            "missing_source": 0,
            "inaccuracy": 0,
            "low_relevance": 0,
        }
        logger.info("Validation performance stats reset")


# Factory function for backward compatibility
def create_validation_agent(llm: Any) -> ValidationAgent:
    """Create validation agent instance.

    Args:
        llm: Language model for validation decisions

    Returns:
        Configured ValidationAgent instance
    """
    return ValidationAgent(llm)


# Validation utilities
def detect_hallucinations(response: str, sources: list[dict]) -> list[str]:
    """Detect potential hallucinations in response.

    Args:
        response: Response text to analyze
        sources: Source documents for verification

    Returns:
        List of potential hallucination indicators
    """
    hallucinations = []

    # Check for training data indicators
    training_indicators = [
        "according to my knowledge",
        "based on my training",
        "i was trained",
        "as an ai",
        "i don't have access",
        "i cannot browse",
    ]

    response_lower = response.lower()
    for indicator in training_indicators:
        if indicator in response_lower:
            hallucinations.append(f"Training data reference: '{indicator}'")

    # Check for unsupported claims
    if sources:
        # Extract factual claims (simplified)
        sentences = response.split(". ")
        for sentence in sentences:
            # Look for definitive statements
            if any(
                word in sentence.lower()
                for word in ["definitely", "certainly", "always", "never"]
            ):
                # Check if supported by sources
                supported = False
                for doc in sources:
                    doc_content = doc.get("content", doc.get("text", ""))
                    if any(
                        word in doc_content.lower() for word in sentence.lower().split()
                    ):
                        supported = True
                        break

                if not supported:
                    hallucinations.append(f"Unsupported claim: '{sentence[:50]}...'")

    return hallucinations


def calculate_source_coverage(response: str, sources: list[dict]) -> float:
    """Calculate how well sources cover the response content.

    Args:
        response: Response text to analyze
        sources: Source documents

    Returns:
        Coverage ratio between 0 and 1
    """
    if not sources or not response:
        return 0.0

    response_words = set(response.lower().split())
    covered_words = set()

    for doc in sources:
        doc_content = doc.get("content", doc.get("text", ""))
        doc_words = set(doc_content.lower().split())
        covered_words.update(response_words.intersection(doc_words))

    return len(covered_words) / len(response_words) if response_words else 0.0


def assess_response_completeness(query: str, response: str) -> float:
    """Assess response completeness relative to query.

    Args:
        query: Original query
        response: Generated response

    Returns:
        Completeness score between 0 and 1
    """
    query_words = len(query.split())
    response_words = len(response.split())

    # Expected response length based on query complexity
    if "compare" in query.lower() or "analyze" in query.lower():
        expected_min = query_words * 8  # Complex queries need longer responses
    elif any(word in query.lower() for word in ["what", "how", "why"]):
        expected_min = query_words * 5  # Explanatory queries
    else:
        expected_min = query_words * 3  # Simple factual queries

    return min(response_words / max(expected_min, 50), 1.0)
