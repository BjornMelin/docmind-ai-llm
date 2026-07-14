"""Integration test for coordinator runtime-context overrides.

Uses the deterministic supervisor_stream_shim fixture (repo-local supervisor graph shim)
to validate that transient objects reach workflow execution without entering state.
"""

from __future__ import annotations

from unittest.mock import patch as _patch

import pytest

from src.agents.coordinator import MultiAgentCoordinator
from tests.integration.coordinator_helpers import patch_supervisor_and_react

# Rationale: tests set minimal coordinator internals to avoid heavy setup.


@pytest.mark.integration
def test_coordinator_runtime_router_engine_visible(supervisor_stream_shim):
    """Runtime overrides reach workflow execution without entering state.

    We patch workflow execution to inspect both initial state and runtime context.
    """
    with patch_supervisor_and_react(supervisor_stream_shim):
        coord = MultiAgentCoordinator(max_agent_timeout=5)
        coord._setup_complete = True
        compiled = supervisor_stream_shim.compile()  # type: ignore[attr-defined]
        coord.compiled_graph = compiled

        observed_runtime_context = {}

        def _capture(final_state, *_args, **_kwargs):
            assert "tools_data" not in final_state
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

        def _workflow_passthrough(initial_state, *_a, **_k):
            nonlocal observed_runtime_context
            observed_runtime_context = dict(_k.get("runtime_context") or {})
            assert "tools_data" not in initial_state
            return {"messages": []}

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
            observed_runtime_context.get("router_engine") is overrides["router_engine"]
        )
