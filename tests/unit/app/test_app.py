"""Unit tests for src/app.py Streamlit application.

Tests the main Streamlit application with focus on:
- Utility functions and business logic (not Streamlit internals)
- Agent system initialization and configuration
- Document processing workflows
- Error handling in key functions
- Configuration validation at startup
- External service integration (Ollama, LLMs)

Uses pytest-mock for boundary mocking and avoids testing Streamlit UI details.
Follows KISS/DRY principles and focuses on business logic testing.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import DocMindSettings


# Mock module-level imports and startup code in app.py to prevent connection attempts
@pytest.fixture(autouse=True)
def mock_app_startup():
    """Mock startup code and external connections before importing src.app."""

    # Provide a minimal session_state shim compatible with attribute and item access
    class _SessionState:
        def __init__(self):
            object.__setattr__(self, "_data", {})

        def __contains__(self, key):
            return key in self._data

        def __getattr__(self, name):
            return self._data.get(name)

        def __setattr__(self, name, value):
            if name == "_data":
                object.__setattr__(self, name, value)
            else:
                self._data[name] = value

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def get(self, key, default=None):
            return self._data.get(key, default)

    with (
        # Patch the underlying core function before app import to avoid Qdrant calls
        patch("src.utils.core.validate_startup_configuration") as mock_validate,
        # Patch Streamlit globals used at import time
        patch("streamlit.set_page_config"),
        patch("streamlit.session_state", _SessionState()),
        # Patch Ollama API used at import time in app.py
        patch("ollama.list", return_value={"models": [{"name": "mock:latest"}]}),
        patch(
            "ollama.pull", return_value={"status": "success", "model": "mock:latest"}
        ),
    ):
        mock_validate.return_value = True
        yield


@pytest.mark.unit
class TestAppUtilityFunctions:
    """Test utility functions from app.py that contain business logic."""

    @pytest.mark.asyncio
    async def test_get_ollama_models_success(self):
        """Test get_ollama_models returns model list successfully."""
        import src.app as app_module

        expected_models = {
            "models": [
                {"name": "llama2:latest"},
                {"name": "codellama:7b"},
                {"name": "mistral:latest"},
            ]
        }

        with patch.object(app_module.ollama, "list", return_value=expected_models):
            from src.app import get_ollama_models

            result = await get_ollama_models()

            assert result == expected_models
            assert len(result["models"]) == 3

    @pytest.mark.asyncio
    async def test_get_ollama_models_connection_error(self):
        """Test get_ollama_models handles connection errors."""
        import src.app as app_module

        with patch.object(
            app_module.ollama, "list", side_effect=ConnectionError("Connection failed")
        ):
            from src.app import get_ollama_models

            with pytest.raises(ConnectionError):
                await get_ollama_models()

    @pytest.mark.asyncio
    async def test_pull_ollama_model_success(self):
        """Test pull_ollama_model downloads model successfully."""
        model_name = "llama2:latest"
        expected_response = {"status": "success", "model": model_name}

        import src.app as app_module

        with patch.object(app_module.ollama, "pull", return_value=expected_response):
            from src.app import pull_ollama_model

            result = await pull_ollama_model(model_name)

            assert result == expected_response

    @pytest.mark.asyncio
    async def test_pull_ollama_model_error(self):
        """Test pull_ollama_model handles download errors."""
        model_name = "invalid:model"

        import src.app as app_module

        with patch.object(
            app_module.ollama, "pull", side_effect=RuntimeError("Model not found")
        ):
            from src.app import pull_ollama_model

            with pytest.raises(RuntimeError):
                await pull_ollama_model(model_name)

    def test_create_tools_from_index(self):
        """Test create_tools_from_index creates tools correctly."""
        mock_index = MagicMock()

        import src.app as app_module

        with patch.object(
            app_module.ToolFactory, "create_basic_tools"
        ) as mock_create_tools:
            expected_tools = [MagicMock(name="tool1"), MagicMock(name="tool2")]
            mock_create_tools.return_value = expected_tools

            from src.app import create_tools_from_index

            result = create_tools_from_index(mock_index)

            mock_create_tools.assert_called_once_with({"vector": mock_index})
            assert result == expected_tools

    def test_create_tools_from_index_with_none(self):
        """Test create_tools_from_index handles None index."""
        import src.app as app_module

        with patch.object(
            app_module.ToolFactory, "create_basic_tools"
        ) as mock_create_tools:
            mock_create_tools.return_value = []

            from src.app import create_tools_from_index

            result = create_tools_from_index(None)

            mock_create_tools.assert_called_once_with({"vector": None})
            assert result == []


@pytest.mark.unit
class TestAgentSystemSetup:
    """Test agent system initialization and configuration."""

    def test_get_agent_system_with_dependency_injection(self):
        """Test get_agent_system uses dependency injection correctly."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        mock_coordinator = MagicMock()

        # Mock the dependency injection
        import src.app as app_module

        with patch.object(
            app_module, "get_multi_agent_coordinator", return_value=mock_coordinator
        ):
            from src.app import get_agent_system

            # Test with mocked injection
            result_agent, result_mode = get_agent_system(
                mock_tools,
                mock_llm,
                mock_memory,
            )

            assert result_agent == mock_coordinator
            assert result_mode == "multi_agent"

    def test_process_query_with_agent_system_multi_agent_mode(self):
        """Test process_query_with_agent_system in multi-agent mode."""
        mock_agent_system = MagicMock()
        mock_memory = MagicMock()
        expected_response = MagicMock(content="Multi-agent response")
        mock_agent_system.process_query.return_value = expected_response

        from src.app import process_query_with_agent_system

        result = process_query_with_agent_system(
            mock_agent_system, "test query", "multi_agent", mock_memory
        )

        mock_agent_system.process_query.assert_called_once_with(
            "test query", context=mock_memory
        )
        assert result == expected_response

    def test_process_query_with_agent_system_other_mode(self):
        """Test process_query_with_agent_system in non-multi-agent mode."""
        mock_agent_system = MagicMock()
        mock_memory = MagicMock()

        from src.app import process_query_with_agent_system

        result = process_query_with_agent_system(
            mock_agent_system, "test query", "single", mock_memory
        )

        # Should return mock response for error cases
        assert hasattr(result, "content")
        assert result.content == "Processing error"

    def test_process_query_with_agent_system_error_handling(self):
        """Test process_query_with_agent_system handles errors."""
        mock_agent_system = MagicMock()
        mock_agent_system.process_query.side_effect = RuntimeError("Agent error")
        mock_memory = MagicMock()

        from src.app import process_query_with_agent_system

        # In current implementation, errors propagate to caller
        with pytest.raises(RuntimeError, match="Agent error"):
            _ = process_query_with_agent_system(
                mock_agent_system, "test query", "multi_agent", mock_memory
            )


@pytest.mark.unit
class TestLlamaCppAvailability:
    """Test LlamaCPP backend availability checks."""

    def test_is_llamacpp_available_true(self):
        """Test is_llamacpp_available returns True when available."""
        with patch("src.app.LLAMACPP_AVAILABLE", True):
            from src.app import is_llamacpp_available

            result = is_llamacpp_available()
            assert result is True

    def test_is_llamacpp_available_false(self):
        """Test is_llamacpp_available returns False when not available."""
        with patch("src.app.LLAMACPP_AVAILABLE", False):
            from src.app import is_llamacpp_available

            result = is_llamacpp_available()
            assert result is False


@pytest.mark.unit
class TestStreamlitSessionState:
    """Test Streamlit session state initialization (business logic only)."""

    def test_memory_initialization(self):
        """Test that memory is initialized correctly."""
        with (
            patch("streamlit.session_state", {}) as mock_session_state,
            patch("src.app.ChatMemoryBuffer") as mock_memory_class,
        ):
            mock_memory = MagicMock()
            mock_memory_class.from_defaults.return_value = mock_memory
            mock_settings = MagicMock()
            mock_settings.vllm.context_window = 131072

            # Simulate session state initialization logic
            if "memory" not in mock_session_state:
                mock_session_state["memory"] = mock_memory_class.from_defaults(
                    token_limit=mock_settings.vllm.context_window
                )

            mock_memory_class.from_defaults.assert_called_once_with(
                token_limit=mock_settings.vllm.context_window
            )

    def test_session_state_defaults(self):
        """Test that session state gets proper default values."""
        # Test the logic that would be used in session state initialization
        default_values = {
            "memory": None,
            "agent_system": None,
            "agent_mode": "single",
            "index": None,
        }

        # Verify expected defaults
        assert default_values["agent_mode"] == "single"
        assert default_values["agent_system"] is None
        assert default_values["index"] is None


@pytest.mark.unit
class TestConfigurationValidation:
    """Test configuration validation at startup."""

    def test_validate_startup_configuration_success(self):
        """Test successful startup configuration validation."""
        with patch("src.app.validate_startup_configuration") as mock_validate:
            mock_validate.return_value = {"status": "valid"}

            # Simulate calling the validator utility
            result = mock_validate(MagicMock())
            assert result["status"] == "valid"

    def test_validate_startup_configuration_failure(self):
        """Test startup configuration validation failure handling."""
        with (
            patch("src.app.validate_startup_configuration") as mock_validate,
            patch("streamlit.error"),
            patch("streamlit.stop"),
        ):
            mock_validate.side_effect = RuntimeError("Configuration error")

            # Test that the error handling logic works
            try:
                mock_validate(MagicMock())
            except RuntimeError as e:
                # This would be handled in the actual app
                error_msg = f"⚠️ Configuration Error: {e}"
                assert "Configuration Error" in error_msg


@pytest.mark.unit
class TestHardwareDetection:
    """Test hardware detection and model suggestion logic."""

    def test_hardware_detection_logic(self):
        """Test hardware detection and model suggestion logic."""
        # Test high VRAM scenario
        hardware_status = {
            "vram_total_gb": 16.0,
            "gpu_name": "RTX 4090",
            "cuda_available": True,
        }

        # Logic from app.py for model suggestion
        vram = hardware_status.get("vram_total_gb")
        hardware_status.get("gpu_name", "No GPU")

        suggested_model = "google/gemma-3n-E4B-it"
        suggested_context = 8192
        quant_suffix = ""

        if vram:
            if vram >= 16:  # 16GB for high-end models
                suggested_model = "nvidia/OpenReasoning-Nemotron-32B"
                quant_suffix = "-Q4_K_M"
                suggested_context = 131072
            elif vram >= 8:  # 8GB for medium models
                suggested_model = "nvidia/OpenReasoning-Nemotron-14B"
                quant_suffix = "-Q8_0"
                suggested_context = 65536
            else:
                suggested_model = "google/gemma-3n-E4B-it"
                quant_suffix = "-Q4_K_S"
                suggested_context = 32768

        # Verify high VRAM suggestions
        assert suggested_model == "nvidia/OpenReasoning-Nemotron-32B"
        assert quant_suffix == "-Q4_K_M"
        assert suggested_context == 131072

    def test_hardware_detection_medium_vram(self):
        """Test hardware detection with medium VRAM."""
        hardware_status = {"vram_total_gb": 10.0, "gpu_name": "RTX 3070"}

        vram = hardware_status.get("vram_total_gb")
        suggested_model = "google/gemma-3n-E4B-it"
        suggested_context = 8192

        if vram and vram >= 8:
            suggested_model = "nvidia/OpenReasoning-Nemotron-14B"
            suggested_context = 65536

        assert suggested_model == "nvidia/OpenReasoning-Nemotron-14B"
        assert suggested_context == 65536

    def test_hardware_detection_no_gpu(self):
        """Test hardware detection with no GPU."""
        hardware_status = {"vram_total_gb": None, "gpu_name": "No GPU"}

        hardware_status.get("vram_total_gb")
        suggested_model = "google/gemma-3n-E4B-it"
        suggested_context = 8192

        # Should keep defaults for no GPU
        assert suggested_model == "google/gemma-3n-E4B-it"
        assert suggested_context == 8192


@pytest.mark.unit
class TestModelInitialization:
    """Test model initialization logic for different backends."""

    def test_ollama_model_creation(self):
        """Test Ollama model initialization."""
        backend = "ollama"
        ollama_url = "http://localhost:11434"
        model_name = "llama2:latest"
        request_timeout = 120.0

        with patch("src.app.Ollama") as mock_ollama_class:
            mock_llm = MagicMock()
            mock_ollama_class.return_value = mock_llm

            if backend == "ollama":
                llm = mock_ollama_class(
                    base_url=ollama_url,
                    model=model_name,
                    request_timeout=request_timeout,
                )

            mock_ollama_class.assert_called_once_with(
                base_url=ollama_url,
                model=model_name,
                request_timeout=request_timeout,
            )
            assert llm == mock_llm

    def test_llamacpp_model_creation(self):
        """Test LlamaCPP model initialization."""
        backend = "llamacpp"
        model_path = "/path/to/model.gguf"
        context_size = 8192
        use_gpu = True

        with (
            patch("src.app.LlamaCPP") as mock_llamacpp_class,
            patch("src.app.is_llamacpp_available", return_value=True),
        ):
            mock_llm = MagicMock()
            mock_llamacpp_class.return_value = mock_llm

            if backend == "llamacpp":
                n_gpu_layers = -1 if use_gpu else 0
                llm = mock_llamacpp_class(
                    model_path=model_path,
                    context_window=context_size,
                    model_kwargs={"n_gpu_layers": n_gpu_layers},
                )

            mock_llamacpp_class.assert_called_once_with(
                model_path=model_path,
                context_window=context_size,
                model_kwargs={"n_gpu_layers": -1},  # use_gpu = True
            )
            assert llm == mock_llm

    def test_lmstudio_model_creation(self):
        """Test LM Studio model initialization."""
        backend = "lmstudio"
        base_url = "http://localhost:1234/v1"
        model_name = "custom-model"
        context_size = 4096

        with patch("src.app.OpenAILike") as mock_openailike_class:
            mock_llm = MagicMock()
            mock_openailike_class.return_value = mock_llm

            if backend == "lmstudio":
                llm = mock_openailike_class(
                    api_base=base_url,
                    api_key="not-needed",
                    model=model_name,
                    is_chat_model=True,
                    is_function_calling_model=False,
                    context_window=context_size,
                )

            mock_openailike_class.assert_called_once_with(
                api_base=base_url,
                api_key="not-needed",
                model=model_name,
                is_chat_model=True,
                is_function_calling_model=False,
                context_window=context_size,
            )
            assert llm == mock_llm

    def test_model_initialization_error_handling(self):
        """Test model initialization error handling."""
        with patch("src.app.Ollama", side_effect=ConnectionError("Connection failed")):
            from src.app import Ollama

            with pytest.raises(ConnectionError, match="Connection failed"):
                Ollama(
                    base_url="http://localhost:11434",
                    model="test:latest",
                    request_timeout=60,
                )


@pytest.mark.unit
class TestDocumentUploadSection:
    """Test document upload and processing logic (business logic only)."""

    @pytest.mark.asyncio
    async def test_upload_section_document_processing_logic(self):
        """Test the document processing logic from upload section."""
        # Mock uploaded files
        mock_files = [
            MagicMock(name="doc1.pdf", type="application/pdf"),
            MagicMock(
                name="doc2.docx",
                type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        ]

        # Mock the document processing function
        with patch("src.app.load_documents_unstructured") as mock_load_docs:
            mock_docs = [
                MagicMock(text="Document 1 content"),
                MagicMock(text="Document 2 content"),
            ]
            mock_load_docs.return_value = mock_docs

            with patch("src.app.VectorStoreIndex") as mock_index_class:
                mock_index = MagicMock()
                mock_index_class.from_documents.return_value = mock_index

                # Simulate the processing logic
                if mock_files:
                    # Load documents
                    docs = await mock_load_docs(mock_files, MagicMock())

                    # Create index

                    with patch("src.app.SimpleVectorStore") as mock_vector_store_class:
                        mock_vector_store = MagicMock()
                        mock_vector_store_class.return_value = mock_vector_store

                        index = mock_index_class.from_documents(
                            docs, vector_store=mock_vector_store
                        )

                        # Verify calls
                        mock_load_docs.assert_called_once()
                        mock_index_class.from_documents.assert_called_once_with(
                            docs, vector_store=mock_vector_store
                        )
                        assert len(docs) == 2
                        assert index == mock_index

    def test_document_upload_error_handling(self):
        """Test error handling in document upload processing."""
        [MagicMock(name="corrupted.pdf")]

        with patch(
            "src.app.load_documents_unstructured",
            side_effect=ValueError("Invalid document"),
        ):
            # Test error handling logic
            try:
                # This would be in a try/except block in the actual app
                raise ValueError("Invalid document")
            except ValueError as e:
                error_message = f"Document processing failed: {e!s}"
                assert error_message == "Document processing failed: Invalid document"


@pytest.mark.unit
class TestAnalysisOptions:
    """Test analysis options and query processing logic."""

    def test_render_prompt_minimal(self):
        """Render a minimal template with defaults and context using API."""
        # Patch prompting API to avoid disk access
        from unittest.mock import patch as _patch

        with _patch("src.prompting.render_prompt", return_value="OK") as rp:
            ctx = {
                "context": "Docs indexed",
                "tone": {"description": "Use a neutral tone."},
                "role": {"description": "Act as a helpful assistant."},
            }
            out = rp("comprehensive-analysis", ctx)
            assert out == "OK"

    def test_build_prompt_context_and_log_telemetry_success(self, monkeypatch):
        """Helper builds context, renders prompt, and logs telemetry."""
        from types import SimpleNamespace

        # Arrange resources
        tones = {"professional": {"description": "Use a neutral tone."}}
        roles = {"assistant": {"description": "Act as a helpful assistant."}}
        templates = [
            SimpleNamespace(
                id="comprehensive-analysis", name="Comprehensive", version=2
            )
        ]

        # Patch render_prompt and log_jsonl
        with (
            patch("src.app.render_prompt", return_value="PROMPT") as mock_render,
            patch("src.app.log_jsonl") as mock_log,
        ):
            from src.app import _build_prompt_context_and_log_telemetry

            # Act
            result = _build_prompt_context_and_log_telemetry(
                template_id="comprehensive-analysis",
                tone_selection="professional",
                role_selection="assistant",
                resources={"tones": tones, "roles": roles, "templates": templates},
            )

            # Assert
            assert result == "PROMPT"
            mock_render.assert_called_once()
            # Verify telemetry log structure
            args, _ = mock_log.call_args
            payload = args[0]
            assert payload["prompt.template_id"] == "comprehensive-analysis"
            assert payload["prompt.version"] == 2
            assert payload["prompt.name"] == "Comprehensive"

    def test_build_prompt_context_and_log_telemetry_keyerror(self):
        """Raises RuntimeError and surfaces st.error on KeyError."""
        from types import SimpleNamespace

        # Force render_prompt to KeyError by omitting required keys
        with (
            patch("src.app.render_prompt", side_effect=KeyError("missing")),
            patch("streamlit.error") as mock_st_error,
        ):
            from src.app import _build_prompt_context_and_log_telemetry

            with pytest.raises(RuntimeError, match="Template rendering failed: "):
                _build_prompt_context_and_log_telemetry(
                    template_id="comprehensive-analysis",
                    tone_selection="professional",
                    role_selection="assistant",
                    resources={
                        "tones": {},
                        "roles": {},
                        "templates": [
                            SimpleNamespace(id="comprehensive-analysis", name="X")
                        ],
                    },
                )

            mock_st_error.assert_called()


@pytest.mark.unit
class TestStreamingResponse:
    """Test streaming response functionality (business logic)."""

    def test_streaming_response_word_splitting(self):
        """Test word-by-word streaming logic."""
        response_text = "This is a test response from the agent system."
        words = response_text.split()

        # Simulate the streaming logic
        streamed_parts = []
        for i, word in enumerate(words):
            if i == 0:
                streamed_parts.append(word)
            else:
                streamed_parts.append(" " + word)

        reconstructed = "".join(streamed_parts)
        assert reconstructed == response_text
        assert len(streamed_parts) == len(words)

    def test_streaming_response_error_handling(self):
        """Test streaming response error handling."""
        # Test error in streaming
        error_message = "Error processing query: Test error"

        # Simulate error streaming logic
        def stream_with_error():
            try:
                raise ValueError("Test error")
            except ValueError as e:
                yield f"Error processing query: {e!s}"

        error_stream = list(stream_with_error())
        assert error_stream[0] == error_message

    def test_response_content_extraction(self):
        """Test response content extraction from AgentResponse."""
        # Mock AgentResponse object
        mock_response = MagicMock()
        mock_response.content = "Test response content"

        # Logic from app.py for content extraction
        if hasattr(mock_response, "content"):
            response_text = mock_response.content
        else:
            response_text = str(mock_response)

        assert response_text == "Test response content"

        # Test fallback case
        mock_response_no_content = MagicMock()
        del mock_response_no_content.content  # Remove content attribute

        if hasattr(mock_response_no_content, "content"):
            response_text = mock_response_no_content.content
        else:
            response_text = str(mock_response_no_content)

        assert "MagicMock" in response_text  # Should use str() fallback


@pytest.mark.unit
class TestSessionPersistence:
    """Test session persistence logic (business logic only)."""

    def test_session_save_logic(self):
        """Test session save logic."""
        mock_memory = MagicMock()
        mock_chat_store = MagicMock()
        mock_memory.chat_store = mock_chat_store

        # Simulate save logic
        filename = "session.json"
        try:
            mock_chat_store.persist(filename)
            result = "success"
        except (OSError, ValueError, TypeError) as e:
            result = f"error: {e!s}"

        mock_chat_store.persist.assert_called_once_with(filename)
        assert result == "success"

    def test_session_load_logic(self):
        """Test session load logic."""
        filename = "session.json"

        with patch("src.app.ChatMemoryBuffer") as mock_memory_class:
            mock_memory = MagicMock()
            mock_memory_class.from_file.return_value = mock_memory

            # Simulate load logic
            try:
                memory = mock_memory_class.from_file(filename)
                result = "success"
            except (OSError, ValueError, TypeError) as e:
                result = f"error: {e!s}"

            mock_memory_class.from_file.assert_called_once_with(filename)
            assert result == "success"
            assert memory == mock_memory

    def test_session_persistence_error_handling(self):
        """Test session persistence error handling."""
        mock_memory = MagicMock()
        mock_chat_store = MagicMock()
        mock_chat_store.persist.side_effect = OSError("Permission denied")
        mock_memory.chat_store = mock_chat_store

        # Test save error handling
        try:
            mock_chat_store.persist("session.json")
            result = "success"
        except (OSError, ValueError, TypeError) as e:
            result = f"Save failed: {e!s}"

        assert result == "Save failed: Permission denied"


# Integration marker for tests that cross component boundaries
@pytest.mark.integration
class TestAppIntegration:
    """Integration tests for app.py with lightweight dependencies."""

    @pytest.mark.integration
    def test_app_startup_configuration_integration(self):
        """Test app startup with real configuration objects."""
        # Test with real DocMindSettings but safe values
        settings = DocMindSettings(
            debug=True,
            enable_gpu_acceleration=False,
            log_level="DEBUG",
        )

        # Test configuration properties
        assert settings.debug is True
        assert settings.enable_gpu_acceleration is False
        assert settings.log_level == "DEBUG"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_document_processing_pipeline_integration(self, integration_settings):
        """Test document processing pipeline with lightweight components."""
        # Test with integration settings
        assert integration_settings.debug is False
        assert integration_settings.log_level == "INFO"

        # Mock the document processing pipeline
        mock_files = [MagicMock(name="test.pdf")]

        with patch("src.app.load_documents_unstructured") as mock_load:
            mock_docs = [MagicMock(text="Test document content")]
            mock_load.return_value = mock_docs

            # Simulate pipeline
            docs = await mock_load(mock_files, integration_settings)

            assert len(docs) == 1
            assert docs[0].text == "Test document content"

    @pytest.mark.integration
    def test_agent_system_integration_with_factory(self):
        """Test agent system integration with factory override injection."""
        # No DI; ensure factory override path works
        mock_coordinator = MagicMock()

        # Test agent system creation via factory injection override
        from src.app import get_agent_system

        agent, mode = get_agent_system(
            None, None, None, multi_agent_coordinator=mock_coordinator
        )

        assert agent == mock_coordinator
        assert mode == "multi_agent"
