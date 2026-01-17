"""Tool registry abstraction used by the LangGraph supervisor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from src.config.settings import DocMindSettings, settings


def _is_jsonable(value: Any, *, _depth: int = 0) -> bool:
    if _depth > 6:
        return False
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list | tuple):
        return all(_is_jsonable(v, _depth=_depth + 1) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_jsonable(v, _depth=_depth + 1)
            for k, v in value.items()
        )
    return False


class ToolRegistry(Protocol):
    """Protocol describing registry capabilities for agent tool wiring."""

    def build_tools_data(
        self, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return the ``tools_data`` payload injected into agent state."""
        ...

    def get_router_tools(self) -> Sequence[Any]:
        """Return tool callables for the router agent."""
        ...

    def get_planner_tools(self) -> Sequence[Any]:
        """Return tool callables for the planner agent."""
        ...

    def get_retrieval_tools(self) -> Sequence[Any]:
        """Return tool callables for the retrieval agent."""
        ...

    def get_synthesis_tools(self) -> Sequence[Any]:
        """Return tool callables for the synthesis agent."""
        ...

    def get_validation_tools(self) -> Sequence[Any]:
        """Return tool callables for the validation agent."""
        ...


@dataclass(slots=True)
class DefaultToolRegistry:
    """Concrete registry that wraps the existing tool modules.

    The default registry preserves the previous ad-hoc wiring while centralising
    configuration defaults from :mod:`src.config.settings`. During Phase 1 the
    registry focusses on deduplicating tool construction; subsequent phases will
    extend it with policy controls, telemetry hooks, and cooperative
    cancellation.
    """

    app_settings: DocMindSettings = field(default_factory=lambda: settings)

    def build_tools_data(
        self, overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return ``tools_data`` merged with runtime overrides.

        The defaults mirror the previous behaviour inside
        :meth:`MultiAgentCoordinator.process_query`. Overrides supplied by the
        caller (e.g., router engine instances or snapshot indexes) take
        precedence.
        """
        defaults: dict[str, Any] = {
            "enable_dspy": bool(
                getattr(self.app_settings, "enable_dspy_optimization", False)
            ),
            "enable_graphrag": self._resolve_graphrag_flag(),
            "enable_multimodal": bool(
                getattr(self.app_settings, "enable_multimodal", False)
            ),
            "reranker_normalize_scores": self._resolve_reranker_normalize(),
            "reranking_top_k": self._resolve_reranking_top_k(),
        }

        combined = defaults.copy()
        if overrides:
            for key, value in overrides.items():
                if _is_jsonable(value):
                    combined[key] = value
        return combined

    def get_router_tools(self) -> Sequence[Any]:
        """Return the router planning tool callable."""
        from src.agents.tools.planning import route_query

        return [route_query]

    def get_planner_tools(self) -> Sequence[Any]:
        """Return the planner tool callable."""
        from src.agents.tools.planning import plan_query

        return [plan_query]

    def get_retrieval_tools(self) -> Sequence[Any]:
        """Return the retrieval tool callable."""
        from src.agents.tools.memory import forget_memory, recall_memories, remember
        from src.agents.tools.ollama_web_tools import get_langchain_web_tools
        from src.agents.tools.retrieval import retrieve_documents

        tools: list[Any] = [
            retrieve_documents,
            recall_memories,
            remember,
            forget_memory,
        ]
        tools.extend(get_langchain_web_tools(self.app_settings))
        return tools

    def get_synthesis_tools(self) -> Sequence[Any]:
        """Return the synthesis tool callable."""
        from src.agents.tools.synthesis import synthesize_results

        return [synthesize_results]

    def get_validation_tools(self) -> Sequence[Any]:
        """Return the validation tool callable."""
        from src.agents.tools.validation import validate_response

        return [validate_response]

    def _resolve_graphrag_flag(self) -> bool:
        """Derive the GraphRAG enablement flag with defensive guards."""
        fn = getattr(self.app_settings, "is_graphrag_enabled", None)
        if callable(fn):
            try:
                return bool(fn())
            except (AttributeError, TypeError, ValueError) as exc:
                from src.utils.log_safety import build_pii_log_entry

                redaction = build_pii_log_entry(
                    str(exc), key_id="tool_registry.is_graphrag_enabled"
                )
                logger.debug(
                    "is_graphrag_enabled() raised; GraphRAG disabled (fail-closed) "
                    "(error_type=%s error=%s)",
                    type(exc).__name__,
                    redaction.redacted,
                )
                return False
        try:
            base_flag = bool(getattr(self.app_settings, "enable_graphrag", False))
            cfg = getattr(self.app_settings, "graphrag_cfg", None)
            enabled = bool(getattr(cfg, "enabled", True)) if cfg is not None else True
            return base_flag and enabled
        except (AttributeError, TypeError, ValueError):
            return False

    def _resolve_reranker_normalize(self) -> bool:
        try:
            return bool(self.app_settings.retrieval.reranker_normalize_scores)
        except (AttributeError, TypeError, ValueError):
            return False

    def _resolve_reranking_top_k(self) -> int:
        try:
            return int(self.app_settings.retrieval.reranking_top_k)
        except (AttributeError, TypeError, ValueError):
            return 0
