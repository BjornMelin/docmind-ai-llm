"""Programmatic multipage app entrypoint (SPEC-008).

All UI pages live under `src/pages/`. In addition to the Streamlit navigation,
this module exposes a small set of utility functions and patch points used by
unit tests and lightweight integrations. These helpers keep business logic
testable while the UI remains thin. KISS/DRY/YAGNI: minimal, explicit exports.
"""

from __future__ import annotations

from typing import Any

# Patch points imported at module scope for tests to stub
import ollama  # type: ignore
import streamlit as st
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.coordinator import create_multi_agent_coordinator
from src.agents.tool_factory import ToolFactory
from src.config import settings
from src.prompting.registry import render_prompt
from src.utils.telemetry import log_jsonl

# Hardware/backend availability flags (tests patch these directly)
LLAMACPP_AVAILABLE = True


def main() -> None:  # pragma: no cover - Streamlit entrypoint
    """Initialize and run the DocMind AI Streamlit application.

    This function sets up the Streamlit page configuration and defines the
    navigation structure for the multipage application. It configures the
    main pages including chat, documents, analytics, and settings.

    The function does not take any parameters and does not return any values.
    It runs the Streamlit navigation loop to handle page routing.
    """
    app_title = getattr(settings, "app_name", "DocMind AI")
    st.set_page_config(page_title=app_title, page_icon="🧠", layout="wide")

    chat = st.Page(
        "src/pages/01_chat.py",
        title="Chat",
        icon=":material/chat:",
        default=True,
    )
    docs = st.Page(
        "src/pages/02_documents.py",
        title="Documents",
        icon=":material/description:",
    )
    analytics = st.Page(
        "src/pages/03_analytics.py",
        title="Analytics",
        icon=":material/insights:",
    )
    settings_page = st.Page(
        "src/pages/04_settings.py",
        title="Settings",
        icon=":material/settings:",
    )

    st.navigation([chat, docs, analytics, settings_page]).run()


# ---------
# Utilities
# ---------


async def get_ollama_models() -> dict[str, Any]:
    """Return available Ollama models via Python client.

    Kept async to align with UI flow; call is synchronous under the hood.
    """
    return ollama.list()


async def pull_ollama_model(model: str) -> dict[str, Any]:
    """Pull an Ollama model by name and return the client response."""
    return ollama.pull(model)


def create_tools_from_index(index: Any | None) -> list[Any]:
    """Create basic tools from a single vector index.

    Tests patch `ToolFactory.create_basic_tools`; keep call surface simple.
    """
    return ToolFactory.create_basic_tools({"vector": index})


def get_multi_agent_coordinator():
    """Factory indirection for dependency injection in tests."""
    return create_multi_agent_coordinator()


def get_agent_system(
    tools: list[Any] | None,
    llm: Any | None,
    memory: ChatMemoryBuffer | None,
    *,
    multi_agent_coordinator: Any | None = None,
) -> tuple[Any, str]:
    """Return agent system and selected mode.

    Currently always returns the multi-agent coordinator and the mode string.
    """
    coordinator = multi_agent_coordinator or get_multi_agent_coordinator()
    return coordinator, "multi_agent"


def process_query_with_agent_system(
    agent_system: Any,
    query: str,
    mode: str,
    memory: ChatMemoryBuffer | None,
):
    """Process a query with the provided agent system, handling basic fallbacks."""
    if mode == "multi_agent":
        return agent_system.process_query(query, context=memory)

    # Minimal fallback object with `content` attribute
    class _Resp:
        def __init__(self, content: str) -> None:
            self.content = content

    return _Resp("Processing error")


def _build_prompt_context_and_log_telemetry(
    *,
    template_id: str,
    tone_selection: str,
    role_selection: str,
    resources: dict[str, Any],
) -> str:
    """Build prompt context, render with registry, and emit telemetry."""
    tones = resources.get("tones", {})
    roles = resources.get("roles", {})
    templates = resources.get("templates", [])

    # Resolve selected preset metadata
    tone = tones.get(tone_selection, {})
    role = roles.get(role_selection, {})
    name = None
    version = None
    for t in templates:
        if getattr(t, "id", None) == template_id:
            name = getattr(t, "name", None)
            version = getattr(t, "version", None)
            break

    ctx = {
        "context": "Docs indexed",  # caller injects real context
        "tone": tone,
        "role": role,
    }

    try:
        out = render_prompt(template_id, ctx)
    except KeyError as exc:  # Surface nicely to UI callers
        st.error(f"Template rendering failed: {exc}")
        raise RuntimeError(f"Template rendering failed: {exc}") from exc

    # Lightweight local-first telemetry
    log_jsonl(
        {
            "event": "prompt.render",
            "prompt.template_id": template_id,
            "prompt.name": name,
            "prompt.version": version,
        }
    )
    return out


def is_llamacpp_available() -> bool:
    """Return whether LlamaCPP backend is available (tests patch this flag)."""
    return bool(LLAMACPP_AVAILABLE)


if __name__ == "__main__":  # pragma: no cover
    main()
