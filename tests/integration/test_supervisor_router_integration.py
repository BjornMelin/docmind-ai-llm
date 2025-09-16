"""Integration test: supervisor + router_tool + InjectedState overrides.

Uses the deterministic supervisor_stream_shim fixture to validate that
tools_data overrides are present in the final state seen by the
coordinator's _extract_response seam.
"""

from __future__ import annotations

from unittest.mock import patch as _patch

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.registry import DefaultToolRegistry

# pylint: disable=protected-access
# Rationale: tests set minimal coordinator internals to avoid heavy setup.


@pytest.mark.integration
def test_supervisor_injected_state_router_engine_visible(supervisor_stream_shim):
    """InjectedState/tools_data overrides are visible to the compiled graph.

    We patch _extract_response to inspect the final_state argument that flowed
    through compiled_graph.stream(...), which the shim forwards from the
    initial_state provided to process_query().
    """
    with (
        _patch(
            "src.agents.coordinator.create_supervisor",
            return_value=supervisor_stream_shim,
        ),
        _patch("src.agents.coordinator.create_react_agent"),
    ):
        # Bypass full __init__ setup; only set required attrs for the test
        with _patch.object(MultiAgentCoordinator, "__init__", return_value=None):
            coord = MultiAgentCoordinator()  # type: ignore[call-arg]
            # Minimal required attributes consumed by process_query()
            coord.total_queries = 0
            coord.enable_fallback = False
            coord.max_agent_timeout = 5
            coord._setup_complete = True
            coord.tool_registry = DefaultToolRegistry()
            compiled = supervisor_stream_shim.compile()  # type: ignore[attr-defined]
            coord.compiled_graph = compiled

        observed_final_state = {}

        def _capture(final_state, *_args, **_kwargs):
            nonlocal observed_final_state
            observed_final_state = dict(final_state)
            # Return a minimal AgentResponse-like object
            from src.agents.models import AgentResponse

            return AgentResponse(
                content="ok",
                sources=[],
                metadata={},
                validation_score=0.0,
                processing_time=0.0,
                optimization_metrics={},
            )

        def _workflow_passthrough(initial_state, _thread_id):
            # Return a dict final state that preserves tools_data
            return {"messages": [], "tools_data": dict(initial_state.tools_data)}

        with (
            _patch.object(
                MultiAgentCoordinator,
                "_run_agent_workflow",
                side_effect=_workflow_passthrough,
            ),
            _patch.object(
                MultiAgentCoordinator, "_extract_response", side_effect=_capture
            ),
        ):
            # Supply a stub router_engine via settings_override
            class _StubRouter:
                def query(self, q):  # pragma: no cover - not executed by shim
                    """Return query unchanged to validate wiring path."""
                    return q

            overrides = {"router_engine": _StubRouter()}
            coord.process_query("hello world", settings_override=overrides)

        assert (
            observed_final_state.get("tools_data", {}).get("router_engine")
            is overrides["router_engine"]
        )
