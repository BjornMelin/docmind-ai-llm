"""Lean retrieval agent built on LangGraph tooling.

This module intentionally mirrors the lightweight patterns recommended in the
official LangGraph tutorials and the ``langgraph-supervisor-py`` reference
implementation. The agent simply orchestrates the shared
``retrieve_documents`` tool through ``create_react_agent`` and only contains
small amounts of glue code for metrics and result normalisation.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.tools.retrieval import retrieve_documents

RECURSION_LIMIT = 3
BASE_CONFIDENCE = 0.4
HIGH_DOC_BONUS = 0.3
MEDIUM_DOC_BONUS = 0.2
LOW_DOC_BONUS = 0.1
STRATEGY_BONUS = 0.1
DSPY_BONUS = 0.05
TARGET_LATENCY_S = 0.150


class RetrievalPayload(BaseModel):
    """Internal representation of the tool payload."""

    model_config = ConfigDict(extra="ignore")

    documents: list[dict[str, Any]] = Field(default_factory=list)
    strategy_used: str | None = Field(default=None)
    query_original: str | None = Field(default=None)
    query_optimized: str | None = Field(default=None)
    document_count: int = 0
    processing_time_ms: float = 0.0
    dspy_used: bool = False
    graphrag_used: bool = False
    error: str | None = None


class RetrievalResult(BaseModel):
    """Document retrieval result exposed to callers."""

    documents: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved documents"
    )
    strategy_used: str = Field(description="Actual retrieval strategy used")
    query_original: str = Field(description="Original search query")
    query_optimized: str = Field(description="Optimised query after DSPy")
    document_count: int = Field(description="Number of documents retrieved")
    processing_time_ms: float = Field(description="Time taken for retrieval")
    dspy_used: bool = Field(
        default=False, description="Whether DSPy optimisation was used"
    )
    graphrag_used: bool = Field(default=False, description="Whether GraphRAG was used")
    confidence_score: float = Field(
        default=0.0, description="Confidence in retrieval quality"
    )
    reasoning: str = Field(default="", description="Explanation of retrieval decisions")


class RetrievalAgent:
    """Thin wrapper around ``retrieve_documents`` using LangGraph."""

    def __init__(self, llm: Any, tools_data: dict[str, Any]):
        """Initialise the agent with its language model and shared tool state.

        Args:
            llm: Large language model provided to ``create_react_agent``.
            tools_data: Shared retrieval configuration injected into the tool.
        """
        self.llm = llm
        self.tools_data = tools_data
        self.total_retrievals = 0
        self._durations: list[float] = []
        self.strategy_usage = {
            "vector": 0,
            "hybrid": 0,
            "graphrag": 0,
            "fallback": 0,
        }
        self.agent = create_react_agent(
            model=self.llm,
            tools=[retrieve_documents],
            name="retrieval_agent",
        )
        self._tool_func = getattr(retrieve_documents, "func", retrieve_documents)
        logger.info("RetrievalAgent initialised")

    def retrieve_documents(
        self,
        query: str,
        strategy: str = "hybrid",
        *,
        use_dspy: bool = True,
        use_graphrag: bool = False,
        context: ChatMemoryBuffer | None = None,
    ) -> RetrievalResult:
        """Execute retrieval through the shared tool and post-process the payload.

        Args:
            query: Natural-language query describing the information need.
            strategy: Preferred retrieval strategy requested by the caller.
            use_dspy: Whether DSPy optimisation should be applied when supported.
            use_graphrag: Whether GraphRAG augmentation should be requested.
            context: Optional chat history forwarded to the agent for grounding.

        Returns:
            RetrievalResult: Normalised retrieval output for downstream agents.
        """
        start_time = time.perf_counter()
        self.total_retrievals += 1

        agent_failed = False
        try:
            payload = self._invoke_agent(query, strategy, context)
        except (
            OSError,
            RuntimeError,
            ValueError,
            AttributeError,
            ValidationError,
        ) as exc:
            logger.warning("Agent retrieval failed; falling back to tool: %s", exc)
            payload = self._call_tool_directly(
                query,
                strategy,
                use_dspy=use_dspy,
                use_graphrag=use_graphrag,
            )
            agent_failed = True

        duration = time.perf_counter() - start_time
        self._durations.append(duration)

        result = self._build_retrieval_result(
            payload,
            query,
            requested_strategy=strategy,
            use_dspy=use_dspy,
            use_graphrag=use_graphrag,
            processing_time=duration,
        )
        self._record_strategy(payload, agent_failed)
        logger.info(
            "Retrieved %d documents via %s (%.1fms)",
            result.document_count,
            result.strategy_used,
            result.processing_time_ms,
        )
        return result

    def _invoke_agent(
        self,
        query: str,
        strategy: str,
        context: ChatMemoryBuffer | None,
    ) -> RetrievalPayload:
        """Invoke the LangGraph agent and extract the retrieval payload.

        Args:
            query: User search query to forward to the agent.
            strategy: Retrieval strategy requested by the caller.
            context: Optional chat memory shared with the agent.

        Returns:
            RetrievalPayload: Structured payload produced by the agent.

        Raises:
            ValueError: If the agent response cannot be parsed into a payload.
        """
        payload: dict[str, Any] = {
            "messages": [
                HumanMessage(
                    content=f"Retrieve documents for: {query} using {strategy} strategy"
                )
            ],
            "tools_data": self.tools_data,
        }
        if context is not None:
            payload["context"] = context

        response = self.agent.invoke(
            payload,
            config={"recursion_limit": RECURSION_LIMIT},
        )
        return self._payload_from_response(response)

    def _payload_from_response(self, response: Any) -> RetrievalPayload:
        """Convert the LangGraph response into a retrieval payload.

        Args:
            response: Raw result returned by ``create_react_agent``.

        Returns:
            RetrievalPayload: Structured payload parsed from the agent output.

        Raises:
            ValueError: If the response is not a dictionary or lacks messages.
        """
        if not isinstance(response, dict):
            raise ValueError("Agent response must be a dictionary")
        messages = response.get("messages", [])
        content = self._extract_message_content(messages)
        return self._payload_from_raw(content)

    def _extract_message_content(self, messages: list[BaseMessage] | list[Any]) -> str:
        """Return the textual payload from the final agent message.

        Args:
            messages: Sequence of messages returned by the agent call.

        Returns:
            str: Textual body holding the retrieval payload.

        Raises:
            ValueError: If the agent emitted no messages.
        """
        if not messages:
            raise ValueError("Agent produced no messages")
        last_message = messages[-1]
        content = getattr(last_message, "content", last_message)
        if isinstance(content, list):
            parts: list[str] = []
            for chunk in content:
                if isinstance(chunk, dict) and "text" in chunk:
                    parts.append(str(chunk["text"]))
                else:
                    parts.append(str(chunk))
            content = "".join(parts)
        return str(content)

    def _payload_from_raw(self, raw: Any) -> RetrievalPayload:
        """Parse the raw payload emitted by the tool into structured data.

        Args:
            raw: Raw payload, typically JSON or a dictionary.

        Returns:
            RetrievalPayload: Structured payload matching ``RetrievalPayload``.

        Raises:
            ValueError: If the payload cannot be deserialised or validated.
        """
        if isinstance(raw, dict):
            return RetrievalPayload(**raw)
        serialised = raw if isinstance(raw, str) else str(raw)
        try:
            return RetrievalPayload.model_validate_json(serialised)
        except ValidationError as exc:
            logger.debug("Malformed retrieval payload", exc_info=False)
            raise ValueError("Malformed tool payload") from exc

    def _call_tool_directly(
        self,
        query: str,
        strategy: str,
        *,
        use_dspy: bool,
        use_graphrag: bool,
    ) -> RetrievalPayload:
        """Invoke the retrieval tool directly as a fallback path.

        Args:
            query: Natural-language query issued by the caller.
            strategy: Retrieval strategy originally requested.
            use_dspy: Whether DSPy optimisation should be used when supported.
            use_graphrag: Whether GraphRAG augmentation should be requested.

        Returns:
            RetrievalPayload: Structured payload returned by the fallback tool.
        """
        raw = self._tool_func(
            query=query,
            strategy=strategy,
            use_dspy=use_dspy,
            use_graphrag=use_graphrag,
            state={"tools_data": self.tools_data},
        )
        try:
            return self._payload_from_raw(raw)
        except ValueError:
            logger.error("Direct tool returned malformed payload", exc_info=False)
            return RetrievalPayload(
                documents=[],
                query_original=query,
                query_optimized=query,
                strategy_used=f"{strategy}_failed",
                error="Malformed tool payload",
            )

    def _build_retrieval_result(
        self,
        payload: RetrievalPayload,
        original_query: str,
        *,
        requested_strategy: str,
        use_dspy: bool,
        use_graphrag: bool,
        processing_time: float,
    ) -> RetrievalResult:
        """Transform a payload into the externally-consumed result model.

        Args:
            payload: Structured payload returned by the retrieval tool.
            original_query: Query string issued by the user.
            requested_strategy: Strategy that the caller initially requested.
            use_dspy: Whether DSPy optimisation was requested.
            use_graphrag: Whether GraphRAG augmentation was requested.
            processing_time: Runtime of the full retrieval call in seconds.

        Returns:
            RetrievalResult: Normalised result object for downstream consumers.
        """
        documents = payload.documents
        strategy_used = payload.strategy_used or requested_strategy
        query_original = payload.query_original or original_query
        query_optimized = payload.query_optimized or original_query
        processing_time_ms = (
            payload.processing_time_ms
            if payload.processing_time_ms > 0
            else round(processing_time * 1000, 2)
        )

        confidence = self._calculate_confidence_score(payload, len(documents))
        reasoning = self._generate_reasoning(
            payload,
            requested_strategy,
            use_dspy,
            use_graphrag,
        )

        return RetrievalResult(
            documents=documents,
            strategy_used=strategy_used,
            query_original=query_original,
            query_optimized=query_optimized,
            document_count=len(documents),
            processing_time_ms=processing_time_ms,
            dspy_used=use_dspy and payload.dspy_used,
            graphrag_used=use_graphrag and payload.graphrag_used,
            confidence_score=confidence,
            reasoning=reasoning,
        )

    def _calculate_confidence_score(
        self, payload: RetrievalPayload, doc_count: int
    ) -> float:
        """Compute a heuristic confidence score for the retrieval output.

        Args:
            payload: Structured payload returned by the retrieval tool.
            doc_count: Number of documents contained in the payload.

        Returns:
            float: Confidence value in the range ``[0.0, 1.0]``.
        """
        if payload.error:
            return 0.0

        confidence = BASE_CONFIDENCE
        if doc_count >= 5:
            confidence += HIGH_DOC_BONUS
        elif doc_count >= 2:
            confidence += MEDIUM_DOC_BONUS
        elif doc_count >= 1:
            confidence += LOW_DOC_BONUS

        if payload.strategy_used in {"hybrid", "graphrag"}:
            confidence += STRATEGY_BONUS
        if payload.dspy_used:
            confidence += DSPY_BONUS
        return min(confidence, 1.0)

    def _generate_reasoning(
        self,
        payload: RetrievalPayload,
        requested_strategy: str,
        use_dspy: bool,
        use_graphrag: bool,
    ) -> str:
        """Generate a concise reasoning summary for the retrieval outcome.

        Args:
            payload: Structured payload returned by the retrieval tool.
            requested_strategy: Retrieval strategy the caller requested.
            use_dspy: Whether DSPy optimisation was requested by the caller.
            use_graphrag: Whether GraphRAG augmentation was requested.

        Returns:
            str: Human-readable explanation of the retrieval path taken.
        """
        if payload.error:
            return f"Retrieval failed: {payload.error}"

        steps: list[str] = []
        strategy_used = payload.strategy_used or requested_strategy
        if strategy_used == requested_strategy:
            steps.append(f"Used requested {requested_strategy} strategy")
        else:
            steps.append(
                f"Used {strategy_used} strategy after {requested_strategy} request"
            )

        steps.append(f"Retrieved {len(payload.documents)} documents")

        if use_dspy and payload.dspy_used:
            steps.append("Applied DSPy optimisation")
        if use_graphrag and payload.graphrag_used:
            steps.append("Included GraphRAG context")

        return "; ".join(steps)

    def _record_strategy(self, payload: RetrievalPayload, agent_failed: bool) -> None:
        """Track strategy usage metrics for observability.

        Args:
            payload: Structured payload returned by the retrieval tool.
            agent_failed: Whether the agent invocation failed and triggered fallback.

        Returns:
            None: This helper only updates local counters.
        """
        strategy = payload.strategy_used or ""
        if agent_failed or "fallback" in strategy or "failed" in strategy:
            self.strategy_usage["fallback"] += 1
            return
        normalised = strategy.split("_", 1)[0]
        if normalised in self.strategy_usage:
            self.strategy_usage[normalised] += 1
        else:
            self.strategy_usage["fallback"] += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Summarise latency samples and strategy usage for observability.

        Returns:
            dict[str, Any]: Aggregated latency and strategy metrics.
        """
        if not self._durations:
            return {
                "total_retrievals": self.total_retrievals,
                "avg_retrieval_time_ms": 0.0,
                "max_retrieval_time_ms": 0.0,
                "min_retrieval_time_ms": 0.0,
                "performance_target_met": True,
                "strategy_usage": self.strategy_usage,
            }

        avg_time = sum(self._durations) / len(self._durations)
        return {
            "total_retrievals": self.total_retrievals,
            "avg_retrieval_time_ms": round(avg_time * 1000, 2),
            "max_retrieval_time_ms": round(max(self._durations) * 1000, 2),
            "min_retrieval_time_ms": round(min(self._durations) * 1000, 2),
            "performance_target_met": avg_time <= TARGET_LATENCY_S,
            "strategy_usage": self.strategy_usage,
        }

    def reset_stats(self) -> None:
        """Reset metrics to support deterministic tests and benchmarks.

        Returns:
            None: This method only clears internal statistics.
        """
        self.total_retrievals = 0
        self._durations.clear()
        self.strategy_usage = {
            "vector": 0,
            "hybrid": 0,
            "graphrag": 0,
            "fallback": 0,
        }
        logger.info("Retrieval performance stats reset")
