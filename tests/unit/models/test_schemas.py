"""Unit tests for Pydantic models in schemas.py.

Focus on field validation, constraints, custom methods, and business logic.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    AgentDecision,
    AnalysisOutput,
    ConversationContext,
    ConversationTurn,
    Document,
    ErrorResponse,
    PerformanceMetrics,
    QueryRequest,
    ValidationResult,
)

# --- merged from test_models_schemas_coverage.py ---


@pytest.mark.unit
class TestConversationContextCoverage:
    """Additional coverage for ConversationContext methods (merged)."""

    def test_add_turn_and_token_counting(self):
        """Test adding conversation turns and token counting."""
        ctx = ConversationContext(session_id="s")
        t = ConversationTurn(id="t1", role="user", content="one two three four")
        ctx.add_turn(t)
        assert len(ctx.turns) == 1
        assert ctx.total_tokens == len(t.content.split()) * 2

    def test_get_context_window_exact_limit(self):
        """Test context window handling at exact limit."""
        ctx = ConversationContext(session_id="s2")
        t1 = ConversationTurn(id="a", role="user", content="Two words")  # 4
        t2 = ConversationTurn(id="b", role="assistant", content="Three words here")  # 6
        ctx.add_turn(t1)
        ctx.add_turn(t2)
        win = ctx.get_context_window(max_tokens=10)
        assert win == [t1, t2]


class TestDocument:
    """Test suite for Document model."""

    @pytest.mark.unit
    def test_document_creation_valid(self):
        """Test Document creation with valid data."""
        doc = Document(
            id="doc-123",
            text="Test document content",
            metadata={"source": "test.pdf", "page": 1},
            score=0.85,
        )

        assert doc.id == "doc-123"
        assert doc.text == "Test document content"
        assert doc.metadata == {"source": "test.pdf", "page": 1}
        assert doc.score == 0.85

    @pytest.mark.unit
    def test_document_minimal_creation(self):
        """Test Document creation with minimal required fields."""
        doc = Document(id="doc-456", text="Minimal content")

        assert doc.id == "doc-456"
        assert doc.text == "Minimal content"
        assert doc.metadata == {}
        assert doc.score is None

    @pytest.mark.unit
    def test_document_empty_text_allowed(self):
        """Test Document allows empty text content."""
        doc = Document(id="doc-789", text="")
        assert doc.text == ""

    @pytest.mark.unit
    def test_document_serialization(self):
        """Test Document JSON serialization and deserialization."""
        original = Document(
            id="doc-serialization",
            text="Content for serialization",
            metadata={"type": "test"},
            score=0.95,
        )

        # Test serialization
        json_data = original.model_dump()
        assert json_data["id"] == "doc-serialization"
        assert json_data["score"] == 0.95

        # Test deserialization
        restored = Document.model_validate(json_data)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata
        assert restored.score == original.score

    @pytest.mark.unit
    def test_document_validation_errors(self):
        """Test Document validation with invalid data types."""
        with pytest.raises(ValidationError) as exc_info:
            Document(id=123, text="content")  # Invalid id type

        errors = exc_info.value.errors()
        assert any(error["type"] == "string_type" for error in errors)

        with pytest.raises(ValidationError):
            Document(id="doc-123", text=None)  # Invalid text type


class TestQueryRequest:
    """Test suite for QueryRequest model."""

    @pytest.mark.unit
    def test_query_request_creation_valid(self):
        """Test QueryRequest creation with valid data."""
        req = QueryRequest(
            query="What is machine learning?",
            context={"session_id": "sess-123"},
            use_multi_agent=True,
            retrieval_strategy="hybrid",
            top_k=15,
        )

        assert req.query == "What is machine learning?"
        assert req.context == {"session_id": "sess-123"}
        assert req.use_multi_agent is True
        assert req.retrieval_strategy == "hybrid"
        assert req.top_k == 15

    @pytest.mark.unit
    def test_query_request_minimal(self):
        """Test QueryRequest with only required field."""
        req = QueryRequest(query="Simple query")

        assert req.query == "Simple query"
        assert req.context is None
        assert req.use_multi_agent is None
        assert req.retrieval_strategy is None
        assert req.top_k is None

    @pytest.mark.unit
    @pytest.mark.parametrize("strategy", ["vector", "hybrid", "graphrag"])
    def test_query_request_valid_retrieval_strategies(self, strategy):
        """Test QueryRequest with valid retrieval strategies."""
        req = QueryRequest(query="test", retrieval_strategy=strategy)
        assert req.retrieval_strategy == strategy

    @pytest.mark.unit
    def test_query_request_invalid_retrieval_strategy(self):
        """Test QueryRequest with invalid retrieval strategy."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", retrieval_strategy="invalid")

        errors = exc_info.value.errors()
        assert any("literal_error" in str(error) for error in errors)

    @pytest.mark.unit
    @pytest.mark.parametrize("top_k_value", [1, 25, 50])
    def test_query_request_valid_top_k_boundary(self, top_k_value):
        """Test QueryRequest top_k boundary values."""
        req = QueryRequest(query="test", top_k=top_k_value)
        assert req.top_k == top_k_value

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_top_k", [0, -1, 51, 100])
    def test_query_request_invalid_top_k(self, invalid_top_k):
        """Test QueryRequest with invalid top_k values."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", top_k=invalid_top_k)

        errors = exc_info.value.errors()
        assert any(
            error["type"] in ("greater_than_equal", "less_than_equal")
            for error in errors
        )

    @pytest.mark.unit
    def test_query_request_empty_query_allowed(self):
        """Test QueryRequest allows empty query string."""
        req = QueryRequest(query="")
        assert req.query == ""


class TestAgentDecision:
    """Test suite for AgentDecision model."""

    @pytest.mark.unit
    def test_agent_decision_creation_valid(self):
        """Test AgentDecision creation with valid data."""
        decision = AgentDecision(
            agent="router",
            decision_type="route_selection",
            confidence=0.85,
            reasoning="High confidence based on keyword analysis",
            metadata={"keywords": ["machine", "learning"]},
        )

        assert decision.agent == "router"
        assert decision.decision_type == "route_selection"
        assert decision.confidence == 0.85
        assert decision.reasoning == "High confidence based on keyword analysis"
        assert decision.metadata["keywords"] == ["machine", "learning"]

    @pytest.mark.unit
    def test_agent_decision_minimal(self):
        """Test AgentDecision with minimal required fields."""
        decision = AgentDecision(
            agent="planner", decision_type="task_creation", confidence=0.7
        )

        assert decision.agent == "planner"
        assert decision.decision_type == "task_creation"
        assert decision.confidence == 0.7
        assert decision.reasoning is None
        assert decision.metadata == {}

    @pytest.mark.unit
    @pytest.mark.parametrize("confidence_value", [0.0, 0.5, 1.0])
    def test_agent_decision_valid_confidence_boundary(self, confidence_value):
        """Test AgentDecision confidence boundary values."""
        decision = AgentDecision(
            agent="test", decision_type="test", confidence=confidence_value
        )
        assert decision.confidence == confidence_value

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_confidence", [-0.1, 1.1, 2.0])
    def test_agent_decision_invalid_confidence(self, invalid_confidence):
        """Test AgentDecision with invalid confidence values."""
        with pytest.raises(ValidationError) as exc_info:
            AgentDecision(
                agent="test", decision_type="test", confidence=invalid_confidence
            )

        errors = exc_info.value.errors()
        assert any(
            error["type"] in ("greater_than_equal", "less_than_equal")
            for error in errors
        )

    @pytest.mark.unit
    def test_agent_decision_serialization(self):
        """Test AgentDecision serialization and deserialization."""
        original = AgentDecision(
            agent="synthesis",
            decision_type="quality_check",
            confidence=0.92,
            reasoning="All validation checks passed",
            metadata={"checks": ["grammar", "relevance", "completeness"]},
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = AgentDecision.model_validate(json_data)

        assert restored.agent == original.agent
        assert restored.confidence == original.confidence
        assert restored.metadata == original.metadata


class TestConversationTurn:
    """Test suite for ConversationTurn model."""

    @pytest.mark.unit
    def test_conversation_turn_creation_valid(self):
        """Test ConversationTurn creation with valid data."""
        timestamp = datetime.now(UTC)
        turn = ConversationTurn(
            id="turn-123",
            timestamp=timestamp,
            role="user",
            content="What is the capital of France?",
            metadata={"session": "sess-456"},
        )

        assert turn.id == "turn-123"
        assert turn.timestamp == timestamp
        assert turn.role == "user"
        assert turn.content == "What is the capital of France?"
        assert turn.metadata == {"session": "sess-456"}

    @pytest.mark.unit
    def test_conversation_turn_default_timestamp(self):
        """Test ConversationTurn creates default timestamp."""
        before_creation = datetime.now()
        turn = ConversationTurn(
            id="turn-auto-time", role="assistant", content="Test response"
        )
        after_creation = datetime.now()

        assert before_creation <= turn.timestamp <= after_creation

    @pytest.mark.unit
    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    def test_conversation_turn_valid_roles(self, role):
        """Test ConversationTurn with valid role values."""
        turn = ConversationTurn(id="turn-role", role=role, content="Test content")
        assert turn.role == role

    @pytest.mark.unit
    def test_conversation_turn_invalid_role(self):
        """Test ConversationTurn with invalid role."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationTurn(id="turn-bad", role="invalid", content="Test")

        errors = exc_info.value.errors()
        assert any("literal_error" in str(error) for error in errors)

    @pytest.mark.unit
    def test_conversation_turn_empty_content_allowed(self):
        """Test ConversationTurn allows empty content."""
        turn = ConversationTurn(id="turn-empty", role="user", content="")
        assert turn.content == ""


class TestConversationContext:
    """Test suite for ConversationContext model."""

    @pytest.mark.unit
    def test_conversation_context_creation(self):
        """Test ConversationContext creation and initialization."""
        context = ConversationContext(
            session_id="sess-789",
            metadata={"user_id": "user-123", "language": "en"},
        )

        assert context.session_id == "sess-789"
        assert context.turns == []
        assert context.total_tokens == 0
        assert context.metadata == {"user_id": "user-123", "language": "en"}

    @pytest.mark.unit
    def test_conversation_context_add_turn(self):
        """Test ConversationContext.add_turn method."""
        context = ConversationContext(session_id="sess-add-turn")
        turn = ConversationTurn(
            id="turn-1", role="user", content="Hello world this is a test"
        )

        context.add_turn(turn)

        assert len(context.turns) == 1
        assert context.turns[0] == turn
        # Simple token counting: 6 words * 2 = 12 tokens
        assert context.total_tokens == 12

    @pytest.mark.unit
    def test_conversation_context_add_multiple_turns(self):
        """Test ConversationContext with multiple turns."""
        context = ConversationContext(session_id="sess-multi")

        turn1 = ConversationTurn(
            id="t1", role="user", content="Hello world"
        )  # 4 tokens
        turn2 = ConversationTurn(
            id="t2", role="assistant", content="Hi there friend"
        )  # 6 tokens
        turn3 = ConversationTurn(
            id="t3", role="user", content="How are you"
        )  # 6 tokens

        context.add_turn(turn1)
        context.add_turn(turn2)
        context.add_turn(turn3)

        assert len(context.turns) == 3
        assert context.total_tokens == 16  # (2 + 3 + 3) * 2

    @pytest.mark.unit
    def test_conversation_context_get_context_window_all_fits(self):
        """Test get_context_window when all turns fit."""
        context = ConversationContext(session_id="sess-window")

        turns = [
            ConversationTurn(id="t1", role="user", content="Short"),  # 2 tokens
            ConversationTurn(id="t2", role="assistant", content="Response"),  # 2 tokens
            ConversationTurn(id="t3", role="user", content="Another short"),  # 4 tokens
        ]

        for turn in turns:
            context.add_turn(turn)

        window = context.get_context_window(max_tokens=20)
        assert len(window) == 3
        assert window == turns

    @pytest.mark.unit
    def test_conversation_context_get_context_window_truncates(self):
        """Test get_context_window truncates older turns."""
        context = ConversationContext(session_id="sess-truncate")

        turns = [
            ConversationTurn(
                id="t1", role="user", content="Very long message with many words here"
            ),  # 18 tokens
            ConversationTurn(
                id="t2", role="assistant", content="Short reply"
            ),  # 4 tokens
            ConversationTurn(id="t3", role="user", content="Final message"),  # 4 tokens
        ]

        for turn in turns:
            context.add_turn(turn)

        # Only allow 10 tokens - should get last 2 turns (8 tokens total)
        window = context.get_context_window(max_tokens=10)
        assert len(window) == 2
        assert window[0] == turns[1]  # "Short reply"
        assert window[1] == turns[2]  # "Final message"

    @pytest.mark.unit
    def test_conversation_context_get_context_window_empty(self):
        """Test get_context_window with no turns."""
        context = ConversationContext(session_id="sess-empty")
        window = context.get_context_window()
        assert window == []

    @pytest.mark.unit
    def test_conversation_context_get_context_window_single_turn_too_large(self):
        """Test get_context_window when single turn exceeds limit."""
        context = ConversationContext(session_id="sess-large")

        large_turn = ConversationTurn(
            id="large",
            role="user",
            content=" ".join(["word"] * 50),  # 100 tokens
        )
        context.add_turn(large_turn)

        window = context.get_context_window(max_tokens=50)
        assert window == []  # Nothing fits


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics model."""

    @pytest.mark.unit
    def test_performance_metrics_creation_valid(self):
        """Test PerformanceMetrics creation with valid data."""
        metrics = PerformanceMetrics(
            query_latency_ms=150.5,
            agent_overhead_ms=25.2,
            retrieval_latency_ms=75.8,
            llm_latency_ms=89.3,
            memory_usage_mb=1024.5,
            vram_usage_mb=8192.0,
            tokens_per_second=125.7,
            cache_hit_rate=85.5,
        )

        assert metrics.query_latency_ms == 150.5
        assert metrics.agent_overhead_ms == 25.2
        assert metrics.retrieval_latency_ms == 75.8
        assert metrics.llm_latency_ms == 89.3
        assert metrics.memory_usage_mb == 1024.5
        assert metrics.vram_usage_mb == 8192.0
        assert metrics.tokens_per_second == 125.7
        assert metrics.cache_hit_rate == 85.5

    @pytest.mark.unit
    @pytest.mark.parametrize("cache_hit_rate", [0.0, 50.0, 100.0])
    def test_performance_metrics_valid_cache_hit_rate_boundary(self, cache_hit_rate):
        """Test PerformanceMetrics cache_hit_rate boundary values."""
        metrics = PerformanceMetrics(
            query_latency_ms=100.0,
            agent_overhead_ms=10.0,
            retrieval_latency_ms=50.0,
            llm_latency_ms=40.0,
            memory_usage_mb=512.0,
            vram_usage_mb=4096.0,
            tokens_per_second=100.0,
            cache_hit_rate=cache_hit_rate,
        )
        assert metrics.cache_hit_rate == cache_hit_rate

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_cache_rate", [-0.1, 100.1, 150.0])
    def test_performance_metrics_invalid_cache_hit_rate(self, invalid_cache_rate):
        """Test PerformanceMetrics with invalid cache_hit_rate values."""
        with pytest.raises(ValidationError) as exc_info:
            PerformanceMetrics(
                query_latency_ms=100.0,
                agent_overhead_ms=10.0,
                retrieval_latency_ms=50.0,
                llm_latency_ms=40.0,
                memory_usage_mb=512.0,
                vram_usage_mb=4096.0,
                tokens_per_second=100.0,
                cache_hit_rate=invalid_cache_rate,
            )

        errors = exc_info.value.errors()
        assert any(
            error["type"] in ("greater_than_equal", "less_than_equal")
            for error in errors
        )

    @pytest.mark.unit
    def test_performance_metrics_zero_values_allowed(self):
        """Test PerformanceMetrics allows zero values."""
        metrics = PerformanceMetrics(
            query_latency_ms=0.0,
            agent_overhead_ms=0.0,
            retrieval_latency_ms=0.0,
            llm_latency_ms=0.0,
            memory_usage_mb=0.0,
            vram_usage_mb=0.0,
            tokens_per_second=0.0,
            cache_hit_rate=0.0,
        )

        assert metrics.query_latency_ms == 0.0
        assert metrics.tokens_per_second == 0.0
        assert metrics.cache_hit_rate == 0.0

    @pytest.mark.unit
    def test_performance_metrics_serialization(self):
        """Test PerformanceMetrics serialization and deserialization."""
        original = PerformanceMetrics(
            query_latency_ms=200.7,
            agent_overhead_ms=30.1,
            retrieval_latency_ms=80.5,
            llm_latency_ms=90.1,
            memory_usage_mb=2048.0,
            vram_usage_mb=12288.0,
            tokens_per_second=150.3,
            cache_hit_rate=92.5,
        )

        json_data = original.model_dump()
        restored = PerformanceMetrics.model_validate(json_data)

        assert restored.query_latency_ms == original.query_latency_ms
        assert restored.cache_hit_rate == original.cache_hit_rate
        assert restored.tokens_per_second == original.tokens_per_second


class TestValidationResult:
    """Test suite for ValidationResult model."""

    @pytest.mark.unit
    def test_validation_result_creation_valid(self):
        """Test ValidationResult creation with valid data."""
        result = ValidationResult(
            valid=True,
            confidence=0.95,
            issues=[
                {"type": "grammar", "severity": "low", "message": "Minor typo"},
                {"type": "relevance", "severity": "medium", "message": "Off-topic"},
            ],
            suggested_action="refine",
        )

        assert result.valid is True
        assert result.confidence == 0.95
        assert len(result.issues) == 2
        assert result.suggested_action == "refine"

    @pytest.mark.unit
    def test_validation_result_minimal(self):
        """Test ValidationResult with minimal data."""
        result = ValidationResult(valid=False, confidence=0.3)

        assert result.valid is False
        assert result.confidence == 0.3
        assert result.issues == []
        assert result.suggested_action == "accept"  # default

    @pytest.mark.unit
    @pytest.mark.parametrize("confidence_value", [0.0, 0.5, 1.0])
    def test_validation_result_valid_confidence_boundary(self, confidence_value):
        """Test ValidationResult confidence boundary values."""
        result = ValidationResult(valid=True, confidence=confidence_value)
        assert result.confidence == confidence_value

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_confidence", [-0.1, 1.1, 2.0])
    def test_validation_result_invalid_confidence(self, invalid_confidence):
        """Test ValidationResult with invalid confidence values."""
        with pytest.raises(ValidationError) as exc_info:
            ValidationResult(valid=True, confidence=invalid_confidence)

        errors = exc_info.value.errors()
        assert any(
            error["type"] in ("greater_than_equal", "less_than_equal")
            for error in errors
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("action", ["accept", "regenerate", "refine"])
    def test_validation_result_valid_suggested_actions(self, action):
        """Test ValidationResult with valid suggested actions."""
        result = ValidationResult(valid=True, confidence=0.8, suggested_action=action)
        assert result.suggested_action == action

    @pytest.mark.unit
    def test_validation_result_invalid_suggested_action(self):
        """Test ValidationResult with invalid suggested action."""
        with pytest.raises(ValidationError) as exc_info:
            ValidationResult(
                valid=True, confidence=0.8, suggested_action="invalid_action"
            )

        errors = exc_info.value.errors()
        assert any("literal_error" in str(error) for error in errors)


class TestErrorResponse:
    """Test suite for ErrorResponse model."""

    @pytest.mark.unit
    def test_error_response_creation_full(self):
        """Test ErrorResponse creation with all fields."""
        error = ErrorResponse(
            error="Connection timeout",
            error_type="NetworkError",
            details={"timeout_seconds": 30, "url": "http://example.com"},
            traceback="Traceback (most recent call last):\\n  ...",
            suggestion="Check network connection and retry",
        )

        assert error.error == "Connection timeout"
        assert error.error_type == "NetworkError"
        assert error.details["timeout_seconds"] == 30
        assert error.traceback.startswith("Traceback")
        assert error.suggestion == "Check network connection and retry"

    @pytest.mark.unit
    def test_error_response_minimal(self):
        """Test ErrorResponse with minimal required fields."""
        error = ErrorResponse(error="Something went wrong", error_type="GenericError")

        assert error.error == "Something went wrong"
        assert error.error_type == "GenericError"
        assert error.details is None
        assert error.traceback is None
        assert error.suggestion is None

    @pytest.mark.unit
    def test_error_response_empty_strings_allowed(self):
        """Test ErrorResponse allows empty strings."""
        error = ErrorResponse(error="", error_type="")
        assert error.error == ""
        assert error.error_type == ""

    @pytest.mark.unit
    def test_error_response_serialization(self):
        """Test ErrorResponse serialization and deserialization."""
        original = ErrorResponse(
            error="Validation failed",
            error_type="ValidationError",
            details={"field": "username", "value": "invalid"},
            suggestion="Use alphanumeric characters only",
        )

        json_data = original.model_dump()
        restored = ErrorResponse.model_validate(json_data)

        assert restored.error == original.error
        assert restored.error_type == original.error_type
        assert restored.details == original.details
        assert restored.suggestion == original.suggestion


class TestAnalysisOutput:
    """Additional comprehensive tests for AnalysisOutput model."""

    @pytest.mark.unit
    def test_analysis_output_empty_lists_allowed(self):
        """Test AnalysisOutput allows empty lists."""
        output = AnalysisOutput(
            summary="Brief summary",
            key_insights=[],
            action_items=[],
            open_questions=[],
        )

        assert output.summary == "Brief summary"
        assert output.key_insights == []
        assert output.action_items == []
        assert output.open_questions == []

    @pytest.mark.unit
    def test_analysis_output_large_lists(self):
        """Test AnalysisOutput with large lists."""
        insights = [f"Insight {i}" for i in range(100)]
        actions = [f"Action {i}" for i in range(50)]
        questions = [f"Question {i}" for i in range(25)]

        output = AnalysisOutput(
            summary="Complex analysis",
            key_insights=insights,
            action_items=actions,
            open_questions=questions,
        )

        assert len(output.key_insights) == 100
        assert len(output.action_items) == 50
        assert len(output.open_questions) == 25

    @pytest.mark.unit
    def test_analysis_output_unicode_content(self):
        """Test AnalysisOutput with unicode and special characters."""
        output = AnalysisOutput(
            summary="Résumé with émojis 🚀 and symbols: α, β, γ",
            key_insights=["Insight with 中文", "Insight with العربية"],
            action_items=["Action mit Deutsch ü", "Acción en Español ñ"],
            open_questions=["Question with Русский?", "Vraag in Nederlands?"],
        )

        assert "🚀" in output.summary
        assert "中文" in output.key_insights[0]
        assert "ñ" in output.action_items[1]

    @pytest.mark.unit
    def test_analysis_output_validation_invalid_types(self):
        """Test AnalysisOutput validation with invalid field types."""
        with pytest.raises(ValidationError):
            AnalysisOutput(
                summary=123,  # Should be string
                key_insights=["valid"],
                action_items=["valid"],
                open_questions=["valid"],
            )

        with pytest.raises(ValidationError):
            AnalysisOutput(
                summary="valid",
                key_insights="not a list",  # Should be list
                action_items=["valid"],
                open_questions=["valid"],
            )

        with pytest.raises(ValidationError):
            AnalysisOutput(
                summary="valid",
                key_insights=["valid"],
                action_items=[123, 456],  # List items should be strings
                open_questions=["valid"],
            )
