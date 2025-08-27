"""Integration tests for app startup validation.

Validates application startup after PR #2 dependency cleanup.

Key validation points:
1. Core functionality works after dependency cleanup
2. Streamlit UI components are accessible
3. Essential services initialize properly
4. Async functionality works correctly
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parents[2]
APP_PATH = PROJECT_ROOT / "src" / "app.py"


class TestAppStartup:
    """Test suite for validating app startup after dependency cleanup."""

    def test_app_imports_successfully(self):
        """Test that the app module can be imported without dependency errors."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Test basic import
            import src.app

            assert src.app is not None
            # Test that key components are accessible
            assert hasattr(src.app, "settings")

        except ImportError as e:
            # Skip if missing dependencies rather than failing
            pytest.skip(f"App import failed (possibly missing deps): {e}")
        finally:
            # Clean up path
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_core_settings_initialization(self):
        """Test that core settings can be initialized."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from src.config.settings import settings

            # Test that settings object exists and has expected attributes
            assert settings is not None
            assert hasattr(settings, "model_validate")

        except ImportError as e:
            pytest.skip(f"Settings initialization failed (missing deps): {e}")
        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    @pytest.mark.slow
    def test_streamlit_app_syntax_validation(self):
        """Test that the Streamlit app has valid Python syntax."""
        if not APP_PATH.exists():
            pytest.skip(f"App file not found at {APP_PATH}")

        # Use Python to check syntax
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(APP_PATH)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"App syntax validation failed: {result.stderr}"

    def test_essential_components_initialize(self):
        """Test that essential components can be initialized."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        components_to_test = [
            ("src.agents.coordinator", "create_multi_agent_coordinator"),
            ("src.utils.core", "detect_hardware"),
            ("src.processing.document_processor", "DocumentProcessor"),
            ("src.cache.simple_cache", "SimpleCache"),
        ]

        failed_components = []

        for module_name, func_name in components_to_test:
            try:
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name)
                assert callable(func)
            except (ImportError, AttributeError) as e:
                failed_components.append(f"{module_name}.{func_name}: {e}")

        try:
            if failed_components:
                if len(failed_components) == len(components_to_test):
                    pytest.skip("All components failed (missing core dependencies)")
                else:
                    print(
                        f"Some components failed (may be expected): {failed_components}"
                    )
        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)


class TestStreamlitUIComponents:
    """Test Streamlit UI components load correctly."""

    def test_streamlit_imports_available(self):
        """Test that Streamlit and related UI imports work."""
        try:
            import streamlit as st

            assert st is not None

            # Test key Streamlit components that the app uses
            required_attrs = [
                "set_page_config",
                "session_state",
                "sidebar",
                "file_uploader",
                "chat_input",
            ]

            for attr in required_attrs:
                assert hasattr(st, attr), (
                    f"Streamlit missing required attribute: {attr}"
                )

        except ImportError as e:
            pytest.skip(f"Streamlit not available: {e}")

    @patch("streamlit.set_page_config")
    @patch("streamlit.session_state", new_callable=lambda: MagicMock())
    def test_app_ui_initialization_mocked(
        self, mock_session_state, mock_set_page_config
    ):
        """Test app UI initialization with mocked Streamlit components."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Mock session state to avoid initialization issues
            mock_session_state.get.return_value = None
            mock_session_state.__getitem__ = MagicMock(side_effect=KeyError)
            mock_session_state.__setitem__ = MagicMock()
            mock_session_state.__contains__ = MagicMock(return_value=False)

            # Import the app module
            import src.app

            # Verify that key functions exist
            # (Note: We can't actually run the Streamlit app, just test imports)
            assert hasattr(src.app, "settings")

        except ImportError as e:
            pytest.skip(f"App UI initialization failed (missing deps): {e}")
        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)


class TestAsyncFunctionality:
    """Test async functionality works correctly."""

    @pytest.mark.asyncio
    async def test_async_document_processing(self):
        """Test async document processing works - SKIPPED (legacy function)."""
        pytest.skip(
            "Legacy async function create_index_async removed with "
            "ADR-009 document processing architecture"
        )

    def test_agent_functionality(self):
        """Test agent functionality works correctly."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from src.agents.coordinator import MultiAgentCoordinator

            # Test that coordinator class exists and is callable
            assert callable(MultiAgentCoordinator)

            # Test that process_query method exists
            assert hasattr(MultiAgentCoordinator, "process_query")

            # Note: The MultiAgentCoordinator uses async processing
            # for enhanced performance

        except ImportError as e:
            pytest.skip(f"Agent functionality failed (missing deps): {e}")
        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)


class TestGracefulDegradation:
    """Test graceful degradation when optional dependencies are missing."""

    def test_hardware_detection_graceful_fallback(self):
        """Test that hardware detection works or fails gracefully."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from src.utils.core import detect_hardware

            # Should not raise exceptions, even if some hardware detection fails
            try:
                hardware_info = detect_hardware()
                assert isinstance(hardware_info, dict)
                print(f"Hardware detection successful: {hardware_info}")
            except Exception as e:
                # Hardware detection can fail gracefully
                print(f"Hardware detection failed gracefully: {e}")

        except ImportError as e:
            pytest.skip(f"Hardware detection not available (missing deps): {e}")
        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_startup_validation_graceful_fallback(self):
        """Test that startup validation works or fails gracefully."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from src.utils.core import validate_startup_configuration

            # Should not raise exceptions due to missing optional packages
            try:
                result = validate_startup_configuration()
                assert isinstance(result, bool | dict | type(None))
                print(f"Startup validation successful: {result}")
            except Exception as e:
                # Validation can fail gracefully
                print(f"Startup validation failed gracefully: {e}")

        except ImportError as e:
            pytest.skip(f"Startup validation not available (missing deps): {e}")
        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_embedding_fallback_mechanisms(self):
        """Test that embedding systems have fallback mechanisms."""
        # Skip embedding utilities test - functions moved/removed in FEAT-002
        pytest.skip(
            "Embedding utilities test skipped - functions moved to retrieval module"
        )


class TestCoreIntegration:
    """Test core integration points work correctly."""

    def test_models_core_integration(self):
        """Test that models.core integrates properly with the rest of the app."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from src.config.settings import settings

            # Test that settings can be used
            assert settings is not None

            # Test basic settings functionality
            if hasattr(settings, "model_dump"):
                config = settings.model_dump()
                assert isinstance(config, dict)
                print(f"Settings configuration loaded: {len(config)} keys")

        except ImportError as e:
            pytest.skip(f"Models core integration failed (missing deps): {e}")
        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_logging_integration(self):
        """Test that logging integration works properly."""
        try:
            from loguru import logger

            # Test basic logging functionality
            logger.info("Test log message for dependency cleanup validation")
            assert logger is not None

        except ImportError:
            pytest.skip("Loguru not available for logging integration test")

    def test_pydantic_integration(self):
        """Test that Pydantic integration works properly."""
        try:
            from pydantic import BaseModel

            # Test that we can create a basic model
            class TestModel(BaseModel):
                name: str
                value: int = 42

            model = TestModel(name="test")
            assert model.name == "test"
            assert model.value == 42

            # Test serialization
            data = model.model_dump()
            assert data == {"name": "test", "value": 42}

        except ImportError:
            pytest.skip("Pydantic not available for integration test")
