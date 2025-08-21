# Feature Specification: User Interface System

## Metadata

- **Feature ID**: FEAT-005
- **Version**: 1.0.0
- **Status**: Implemented
- **Created**: 2025-08-19
- **Validated At**: 2025-08-20
- **Completion Percentage**: 15% (ADR-Validated - significant gap correction)
- **Requirements Covered**: REQ-0071 to REQ-0080, REQ-0091 to REQ-0096
- **ADR Dependencies**: [ADR-013, ADR-016, ADR-020, ADR-021, ADR-022, ADR-023]
- **Implementation Status**: ADR-Validated Ready
- **Validation Timestamp**: 2025-08-21
- **Code Replacement Plan**: See Implementation Instructions section

## 1. Objective

The User Interface System provides an intuitive Streamlit-based web interface for document analysis, enabling users to upload documents, interact with the multi-agent system through natural language queries, configure system settings, view retrieved sources with attribution, maintain conversation history, and export analysis results - all while operating entirely offline with real-time status updates.

## 2. Scope

### In Scope

- Streamlit web application framework
- Document upload interface (drag-and-drop)
- Chat interface with streaming responses
- Settings panel for configuration toggles
- Source attribution and document viewer
- Session state management
- Chat history persistence
- Export functionality (Markdown, JSON)
- Real-time processing status
- Error handling and user feedback

### Out of Scope

- Mobile native applications
- Desktop native applications
- Multi-user collaboration
- Real-time synchronization
- Custom UI themes beyond Streamlit defaults

## 3. Inputs and Outputs

### Inputs

- **User Queries**: Natural language text input
- **Document Uploads**: Files via drag-and-drop or file selector
- **Configuration Settings**: UI toggles and preferences
- **Session Actions**: Clear, export, new session commands

### Outputs

- **Agent Responses**: Formatted text with sources
- **Processing Status**: Progress bars and status messages
- **Retrieved Documents**: Source snippets with metadata
- **Export Files**: Markdown or JSON formatted results
- **Error Messages**: User-friendly error descriptions

## 4. Interfaces

### Main Application Interface

```python
class StreamlitApp:
    """Main Streamlit application controller."""
    
    def __init__(self):
        """Initialize app with session state."""
        self.init_session_state()
        self.load_configuration()
    
    def render_sidebar(self) -> None:
        """Render sidebar with settings and upload."""
        pass
    
    def render_chat_interface(self) -> None:
        """Render main chat interface."""
        pass
    
    def render_source_viewer(self) -> None:
        """Render source documents panel."""
        pass
```

### Session State Manager

```python
class SessionStateManager:
    """Manages Streamlit session state."""
    
    def initialize_state(self) -> None:
        """Initialize session variables."""
        st.session_state.messages = []
        st.session_state.documents = []
        st.session_state.settings = {}
        st.session_state.processing = False
    
    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[List[Document]] = None
    ) -> None:
        """Add message to chat history."""
        pass
    
    def persist_session(self) -> None:
        """Save session to database."""
        pass
```

### Upload Handler

```python
class DocumentUploadHandler:
    """Handles document upload and processing."""
    
    async def process_upload(
        self,
        uploaded_file: UploadedFile
    ) -> ProcessingResult:
        """Process uploaded document."""
        pass
    
    def validate_file(
        self,
        file: UploadedFile
    ) -> Tuple[bool, str]:
        """Validate file type and size."""
        pass
```

## 5. Data Contracts

### Chat Message Schema

```json
{
  "message_id": "msg_uuid",
  "timestamp": "2025-08-19T10:00:00Z",
  "role": "user|assistant|system",
  "content": "Message text",
  "sources": [
    {
      "document_id": "doc_uuid",
      "chunk_id": "chunk_001",
      "content": "Source snippet",
      "page": 5,
      "score": 0.85
    }
  ],
  "metadata": {
    "processing_time": 1.5,
    "tokens_used": 512,
    "strategy_used": "hybrid"
  }
}
```

### UI Settings Schema

```json
{
  "general": {
    "theme": "light|dark|auto",
    "language": "en",
    "auto_scroll": true
  },
  "backend": {
    "llm_backend": "ollama|llamacpp|vllm",
    "enable_gpu": true,
    "enable_multi_agent": true
  },
  "retrieval": {
    "search_strategy": "hybrid|vector|graphrag",
    "enable_reranking": true,
    "top_k": 10
  },
  "advanced": {
    "enable_dspy": false,
    "enable_graphrag": false,
    "analysis_mode": "detailed|summary|comparison",
    "chunk_size": 512,
    "chunk_overlap": 50
  }
}
```

### Export Format Schema

```json
{
  "export_id": "export_uuid",
  "created_at": "2025-08-19T10:00:00Z",
  "session_id": "session_uuid",
  "format": "markdown|json",
  "content": {
    "messages": [...],
    "documents": [...],
    "settings": {...},
    "statistics": {
      "total_messages": 20,
      "total_documents": 5,
      "session_duration": 1800
    }
  }
}
```

## 6. Change Plan

### New Files

- `src/ui/streamlit_app.py` - Main Streamlit application
- `src/ui/components/sidebar.py` - Sidebar component
- `src/ui/components/chat.py` - Chat interface component
- `src/ui/components/upload.py` - Upload handler component
- `src/ui/components/sources.py` - Source viewer component
- `src/ui/components/settings.py` - Settings panel component
- `src/ui/session/state_manager.py` - Session state management
- `src/ui/session/persistence.py` - Session persistence
- `src/ui/export/formatter.py` - Export formatting
- `tests/test_ui/` - UI test suite

### Modified Files

- `.streamlit/config.toml` - Streamlit configuration
- `src/main.py` - Launch Streamlit app
- `src/config/ui_config.py` - UI configuration

### Streamlit Configuration

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 100

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"
```

## 7. Acceptance Criteria

### Scenario 1: Document Upload Flow

```gherkin
Given a user on the main interface
When they drag and drop a PDF document
Then the upload progress is displayed
And the document is processed asynchronously
And a success notification appears
And the document appears in the sidebar
And the UI remains responsive during processing
```

### Scenario 2: Chat Interaction

```gherkin
Given a user with uploaded documents
When they type a query and press Enter
Then the query appears in the chat history
And a typing indicator shows while processing
And the response streams in real-time
And source documents are displayed below the response
And the response includes clickable source citations
```

### Scenario 3: Settings Configuration

```gherkin
Given a user in the settings panel
When they toggle GPU acceleration off
Then the setting persists across sessions
And the system switches to CPU mode
And a confirmation message appears
And performance metrics update accordingly
```

### Scenario 4: Session Persistence

```gherkin
Given a user with an active chat session
When they refresh the browser page
Then the chat history is restored
And uploaded documents remain available
And settings are preserved
And the conversation can continue seamlessly
```

### Scenario 5: Export Functionality

```gherkin
Given a user with chat history and documents
When they click "Export to Markdown"
Then a formatted markdown file is generated
And it includes all messages with timestamps
And source citations are properly formatted
And the file downloads automatically
And includes session metadata
```

### Scenario 6: Comprehensive Error Handling

```gherkin
Given a user uploads an invalid file or experiences system errors
When document processing fails
Or when the 5-agent system encounters coordination issues
Or when memory management hits 128K context limits
Then st.status containers show error state with details
And user-friendly error messages appear with recovery suggestions
And technical details are available in expandable sections
And the system gracefully degrades to basic RAG if agents fail
And errors are logged with full context for debugging
And the UI remains responsive and functional
And retry mechanisms are available for recoverable errors
```

### Scenario 7: Analysis Mode Selection (ADR-023)

```gherkin
Given a user with multiple uploaded documents
When they select "Analyze each document separately" mode
And they submit a query about comparing documents
Then the DocumentAnalysisModeManager activates parallel processing
And individual analysis results are generated using the 5-agent system
And cross-document patterns are identified through ResultAggregator
And comparative analysis is provided with processing metrics
And UI displays results in tabs for each document
And synthesized insights show patterns across all documents
And performance monitoring tracks 3-5x speedup achievements
```

### Scenario 8: Dynamic Prompt Template Configuration (ADR-020)

```gherkin
Given a user wants to customize their analysis approach
When they access the prompt configuration panel
And they select role: "researcher", tone: "academic", detail: "comprehensive"
And they choose prompt type: "comprehensive_analysis"
Then the PromptTemplateManager compiles the template from 1,600+ combinations
And DSPy optimization enhances the prompt effectiveness
And the template is cached for future use
And the chat interface adapts to the selected configuration
And responses reflect the academic researcher perspective
And the export system uses matching academic templates
```

## 8. Tests

### Unit Tests (ADR-Compliant)

**Core UI Components (ADR-013):**

- Multipage navigation with st.navigation testing
- Native streaming with st.write_stream() validation
- Component rendering with AgGrid and Plotly integration
- Status container behavior with st.status testing

**State Management (ADR-016):**

- Native Streamlit session state operations
- LangGraph memory integration testing
- Chat memory buffer operations (128K context)
- Session persistence with SQLite backend

**Prompt Template System (ADR-020):**

- Template compilation for 1,600+ combinations
- Jinja2 rendering with role/tone/detail variations
- DSPy optimization integration testing
- Cache performance and hit rate validation

**Export System (ADR-022):**

- Type-safe Pydantic model serialization
- Multi-format export (JSON, Markdown, Rich console)
- Template rendering for all export templates
- Export metadata accuracy validation

**Analysis Modes (ADR-023):**

- Separate vs combined mode selection logic
- Parallel processing with ThreadPoolExecutor
- Result aggregation and synthesis testing
- Cross-document pattern detection

### Integration Tests

**End-to-End Workflows:**

- Complete upload-to-analysis-to-export pipeline
- 5-agent coordination through UI interactions
- Memory management across conversation sessions
- Settings changes propagating to all subsystems

**ADR System Integration:**

- Agent coordination logs displayed in UI
- Memory condensation triggers and UI updates
- Prompt template changes affecting chat responses
- Analysis mode switching with result format changes
- Export system integration with all analysis modes

**Cross-Component Communication:**

- Chat interface with memory manager coordination
- Document upload triggering vector database updates
- Settings panel affecting all dependent systems
- Performance monitoring integration across components

### UI/UX Tests

**Modern Interface (ADR-013):**

- Responsive design with wide layout configuration
- Native Streamlit component accessibility
- Keyboard navigation across multipage structure
- Loading state indicators with st.status containers

**User Experience Validation:**

- Prompt configuration discoverability and usability
- Analysis mode selection clarity and guidance
- Export interface intuitive operation
- Error message clarity and actionability

**Performance UX:**

- Real-time streaming response experience
- Progress feedback during document processing
- Memory management transparency to users
- System resource utilization visibility

### Performance Tests

**UI Responsiveness (ADR-013):**

- Page load times <2 seconds with st.navigation
- Component render times <100ms
- State management operations <50ms
- Real-time streaming performance validation

**Memory Management (ADR-021):**

- 128K context handling without UI degradation
- Chat history with extended conversations (500+ messages)
- Memory condensation without user interruption
- Session restoration performance validation

**Processing Performance (ADR-023):**

- Parallel document processing UI responsiveness
- Analysis mode switching performance
- Cross-document aggregation speed
- Export generation performance across all formats

**System Integration Performance:**

- Agent coordination visualization without lag
- Template compilation speed across all combinations
- Export system performance with large datasets
- Error handling and recovery speed

## 9. Security Considerations

- Input sanitization for chat messages
- File type validation for uploads
- XSS prevention in rendered content
- CSRF protection for forms
- Secure session storage
- No sensitive data in browser storage

## 10. Quality Gates

### Performance Gates

- Page load time: <2 seconds
- Upload processing start: <500ms
- Chat response streaming: Real-time
- Settings toggle effect: <100ms
- Export generation: <2 seconds

### Usability Gates

- Mobile responsive: Yes (tablet+)
- Keyboard accessible: 100%
- Error recovery: Graceful
- Loading indicators: All async operations
- Help documentation: Context-sensitive

### Reliability Gates

- Session recovery: 100% success
- Upload success rate: >95%
- Settings persistence: 100%
- Export accuracy: 100%
- Browser compatibility: Chrome, Firefox, Safari, Edge

## 11. Requirements Covered

- **REQ-0071**: Streamlit web interface ✓
- **REQ-0072**: GPU and backend toggles ✓
- **REQ-0073**: Real-time status display ✓
- **REQ-0074**: Session state persistence ✓
- **REQ-0075**: Multi-format file upload ✓
- **REQ-0076**: Source attribution display ✓
- **REQ-0077**: Chat history persistence ✓
- **REQ-0078**: Export functionality ✓
- **REQ-0079**: Context window indicators ✓
- **REQ-0080**: Error message handling ✓
- **REQ-0093**: Customizable prompts (settings) ✓
- **REQ-0094-v2**: Chat memory management ✓
- **REQ-0095**: Analysis mode selection ✓
- **REQ-0096**: Export format compliance ✓

## 11. ADR Dependencies

### ADR-013 (User Interface Architecture)

**Native Streamlit Multipage Navigation with Modern Components:**

- **Required**: st.navigation() multipage architecture (replaces single-page app.py)
- **Required**: AgGrid integration for document tables and data display
- **Required**: Plotly integration for analytics dashboards and visualizations
- **Required**: Native streaming with st.write_stream() for all LLM responses
- **Required**: .streamlit/config.toml theming (removes unsafe HTML usage)
- **Gap**: Current src/app.py is monolithic single-page - needs complete restructure

### ADR-016 (UI State Management)

**Native Streamlit + LangGraph Memory Integration:**

- **Required**: Native st.session_state usage throughout application
- **Required**: StreamlitChatMemory bridge class for LangGraph integration
- **Required**: InMemoryStore for long-term conversation persistence
- **Required**: Session persistence across application restarts
- **Gap**: Current ChatMemoryBuffer usage lacks LangGraph integration

### ADR-020 (Prompt Template System)

**1,600+ Template Combinations with DSPy Optimization:**

- **Required**: PromptTemplateManager class (replaces PREDEFINED_PROMPTS dict)
- **Required**: TemplateRegistry with 1,600+ combinations (10 roles × 10 tones × 16 detail levels)
- **Required**: Jinja2 template rendering system for dynamic prompts
- **Required**: DSPy optimization integration for prompt efficiency
- **Required**: UI components for template selection and customization
- **Gap**: Current prompts.py has only 4 basic templates vs 1,600+ required

### ADR-021 (Chat Memory & Context Management)

**128K Context Window with FP8 Optimization:**

- **Required**: 128K context window support (upgrade from 32K limit)
- **Required**: FP8 KV cache optimization integration for memory efficiency
- **Required**: Context trimming strategies with user feedback
- **Required**: Conversation condensation for long sessions
- **Required**: Memory usage indicators in UI
- **Gap**: Current ChatMemoryBuffer.from_defaults(token_limit=32768) vs 128K required

### ADR-022 (Export & Output Formatting)

**Type-Safe Multi-Format Export System:**

- **Required**: Type-safe export with Pydantic model validation
- **Required**: JSON, Markdown, PDF, and Rich console format support
- **Required**: Jinja2 template system for customizable output formatting
- **Required**: Export progress indicators and batch processing
- **Required**: Template customization UI for different output styles
- **Gap**: No export functionality currently implemented

### ADR-023 (Analysis Mode Strategy)

**Parallel Document Processing with Mode Selection:**

- **Required**: Analysis mode selection UI (separate/combined/auto)
- **Required**: DocumentAnalysisModeManager for parallel processing
- **Required**: Parallel document processing with 3-5x speedup
- **Required**: Result aggregation and cross-document comparison
- **Required**: Processing status indicators for multiple documents
- **Gap**: Current single processing mode vs parallel/combined modes required

## 12. Dependencies

### Technical Dependencies

**Current Dependencies**:

- `streamlit>=1.47.1`
- `streamlit-chat>=0.1.0`
- `streamlit-aggrid>=0.3.0`
- `streamlit-option-menu>=0.3.0`
- `markdown>=3.4.0`
- `pygments>=2.15.0`

**Required Additions for ADR Compliance**:

- `plotly>=5.0.0` (ADR-013: Analytics dashboards)
- `jinja2>=3.1.0` (ADR-020: Template rendering, ADR-022: Export formatting)
- `rich>=13.0.0` (ADR-022: Rich console export)
- `fpdf2>=2.7.0` (ADR-022: PDF export)
- `sqlalchemy>=2.0.0` (ADR-021: SQLite conversation persistence)
- `dspy-ai>=2.5.0` (ADR-020: Prompt optimization - already in pyproject.toml)
- `langgraph>=0.2.74` (ADR-016: Memory integration - already in pyproject.toml)

### ADR Dependencies (ALL MANDATORY)

**Critical ADR Requirements**:

- **ADR-013**: User Interface Architecture - **BLOCKING** multipage navigation requirement
- **ADR-016**: UI State Management - **BLOCKING** LangGraph memory integration requirement  
- **ADR-020**: Prompt Template System - **BLOCKING** 1,600+ template combinations requirement
- **ADR-021**: Chat Memory & Context Management - **BLOCKING** 128K FP8 context requirement
- **ADR-022**: Export & Output Formatting - **MISSING** type-safe export system requirement
- **ADR-023**: Analysis Mode Strategy - **MISSING** parallel processing requirement

**⚠️ CRITICAL**: All six ADR dependencies must be fully implemented. Current implementation violates 4 ADRs and is missing 2 entirely.

### Feature Dependencies

- Multi-Agent (FEAT-001) for query processing
- Retrieval (FEAT-002) for source display  
- Document Processing (FEAT-003) for uploads
- Infrastructure (FEAT-004) for settings

### Browser Requirements

- Modern browser with JavaScript enabled
- WebSocket support for streaming
- Local storage for preferences
- File API for uploads

## 13. Implementation Instructions

> **CRITICAL: Complete Architectural Overhaul Required - Delete All Existing UI Code**

This implementation requires **COMPLETE DELETION** of the existing monolithic UI architecture and replacement with the ADR-designed multipage system. The current `src/app.py` (411 lines) fundamentally violates ADR-013, ADR-016, ADR-020, ADR-021, ADR-022, and ADR-023 requirements and must be entirely replaced.

### STEP 1: Delete All Existing UI Files (MANDATORY)

**DELETE these files completely** - they violate ADR architecture:

```bash
# DELETE monolithic UI that violates ADR-013 multipage architecture
rm src/app.py  # 411 lines of monolithic single-page - VIOLATES ADR-013

# DELETE simple prompts that violate ADR-020 template system
rm src/prompts.py  # Only 4 prompts vs 1,600+ required - VIOLATES ADR-020
```

**⚠️ CRITICAL**: Any attempt to preserve existing `src/app.py` or `src/prompts.py` will result in ADR violations. These files implement fundamentally incompatible architectures and must be completely replaced.

### STEP 2: ADR-Compliant Architecture Implementation

### Files to CREATE for ADR Compliance

**Primary UI Architecture (ADR-013) - COMPLETE NEW IMPLEMENTATION**:

- **CREATE** `src/ui/streamlit_app.py` - New main application entry point with st.navigation()
- **CREATE** `src/ui/pages/chat.py` - Chat interface with native streaming via st.write_stream()
- **CREATE** `src/ui/pages/documents.py` - Document management with AgGrid tables
- **CREATE** `src/ui/pages/analytics.py` - Performance metrics with Plotly dashboards
- **CREATE** `src/ui/pages/settings.py` - Configuration interface with model selection
- **CREATE** `src/ui/components/` - Reusable UI components library
- **CREATE** `.streamlit/config.toml` - Theming configuration (removes unsafe HTML)

**State Management (ADR-016)**:

- Replace custom SessionStateManager usage with native st.session_state + LangGraph integration
- Implement StreamlitChatMemory bridge class for memory persistence
- Add InMemoryStore integration for long-term conversation storage

**Prompt System (ADR-020) - COMPLETE REPLACEMENT**:

- **CREATE** `src/prompts/manager.py` - PromptTemplateManager class (replaces simple PREDEFINED_PROMPTS dict)
- **CREATE** `src/prompts/registry.py` - TemplateRegistry with 1,600+ combinations (10 roles × 10 tones × 16 detail levels)
- **CREATE** `src/prompts/optimizer.py` - DSPy integration for prompt efficiency
- **CREATE** `src/prompts/templates/` - Jinja2 template files for dynamic rendering
- **CREATE** `src/ui/components/prompt_selector.py` - Template selection UI

**⚠️ CRITICAL**: The existing `src/prompts.py` contains only 4 basic prompts but ADR-020 requires 1,600+ combinations. This is a **fundamental architecture change**, not an enhancement.

**Memory Management (ADR-021) - BLOCKING UPGRADE REQUIRED**:

- **REPLACE** `ChatMemoryBuffer.from_defaults(token_limit=32768)` with 128K context:

  ```python
  # CURRENT (VIOLATES ADR-021):
  ChatMemoryBuffer.from_defaults(token_limit=32768)  # Only 32K - INADEQUATE
  
  # REQUIRED (ADR-021 COMPLIANT):
  ChatMemoryBuffer.from_defaults(token_limit=131072)  # 128K with FP8 optimization
  ```

- **CREATE** `src/ui/session/memory_bridge.py` - StreamlitChatMemory for LangGraph integration
- **CREATE** `src/ui/session/persistence.py` - SQLite conversation storage
- **IMPLEMENT** FP8 KV cache optimization integration for Qwen/Qwen3-4B-Instruct-2507-FP8

**Export System (ADR-022) - COMPLETE NEW IMPLEMENTATION**:

- **CREATE** `src/ui/export/manager.py` - ExportManager class with type-safe Pydantic models
- **CREATE** `src/ui/export/models.py` - ExportableAnalysis, BatchExportResult, ExportConfiguration
- **CREATE** `src/ui/export/templates/` - Jinja2 templates for Markdown formatting
- **CREATE** `src/ui/pages/export.py` - Export functionality page
- **CREATE** `src/ui/components/export_options.py` - Export configuration UI
- **IMPLEMENT** JSON, Markdown, PDF, and Rich console format support

**⚠️ CRITICAL**: Export functionality is completely missing from current implementation. This is not an enhancement but a fundamental requirement per ADR-022.

**Analysis Modes (ADR-023) - COMPLETE NEW IMPLEMENTATION**:

- **CREATE** `src/analysis/mode_manager.py` - DocumentAnalysisModeManager class
- **CREATE** `src/analysis/processors.py` - Parallel processing with ThreadPoolExecutor
- **CREATE** `src/analysis/aggregator.py` - ResultAggregator for cross-document synthesis
- **CREATE** `src/ui/components/mode_selector.py` - Analysis mode selection UI
- **CREATE** `src/ui/components/processing_status.py` - Progress indicators for parallel processing
- **INTEGRATE** with existing MultiAgentCoordinator for both individual and combined analysis

**⚠️ CRITICAL**: Analysis mode functionality is completely missing. Current implementation only supports single processing mode, violating ADR-023 requirements for 3-5x speedup.

### Functions to Deprecate

**Custom State Management (violates ADR-016)**:

- Any custom Streamlit state abstraction layers
- Remove custom session persistence implementations
- Replace with native st.session_state + LangGraph memory integration

**Simple Prompt Templates (violates ADR-020)**:

- PREDEFINED_PROMPTS dictionary approach
- Simple string-based template system
- Replace with PromptTemplateManager and 1,600+ combinations

**Basic Export Functions (violates ADR-022)**:

- Any basic file save functionality
- Simple text output methods
- Replace with type-safe multi-format export system

**Single Processing Mode (violates ADR-023)**:

- Single document processing workflows
- Basic analysis without mode selection
- Replace with parallel/combined analysis strategy

### Dead Code Removal - MANDATORY DELETIONS

**Monolithic UI Architecture (ADR-013 Violations)**:

- **DELETE** `src/app.py` entirely - 411 lines of monolithic single-page architecture
- **DELETE** any single-page Streamlit implementations
- **DELETE** custom theming with unsafe_allow_html (security risk, violates ADR-013)

**Simple Prompt System (ADR-020 Violations)**:

- **DELETE** `src/prompts.py` entirely - only 4 prompts vs 1,600+ required
- **DELETE** PREDEFINED_PROMPTS, TONES, INSTRUCTIONS, LENGTHS dictionaries
- **DELETE** any static prompt approaches

**Basic Memory System (ADR-021 Violations)**:

- **DELETE** 32K token limit configurations everywhere
- **DELETE** `ChatMemoryBuffer.from_defaults(token_limit=32768)` usage
- **DELETE** basic conversation storage without LangGraph integration

**Missing Export/Analysis Systems (ADR-022/023 Violations)**:

- **MISSING ENTIRELY**: No export functionality exists - complete ADR-022 violation
- **MISSING ENTIRELY**: No analysis mode selection - complete ADR-023 violation
- **MISSING ENTIRELY**: No parallel processing capabilities
- **MISSING ENTIRELY**: No result aggregation or cross-document comparison

**⚠️ CRITICAL**: These are not code deletions but missing fundamental features required by ADRs

### Migration Strategy

**Phase 1 - UI Architecture Foundation**:

1. Implement pure st.navigation multipage architecture per ADR-013
2. Create component library with AgGrid and Plotly integration
3. Add native streaming with st.write_stream() for all responses
4. Implement .streamlit/config.toml for theming (remove unsafe HTML)

**Phase 2 - Memory & Template Systems**:

1. Upgrade to 128K context with FP8 optimization per ADR-021
2. Deploy PromptTemplateManager with 1,600+ combinations per ADR-020
3. Integrate native Streamlit state with LangGraph memory per ADR-016
4. Add conversation persistence with SQLite backend

**Phase 3 - Export & Analysis Features**:

1. Implement type-safe export system with Pydantic models per ADR-022
2. Add DocumentAnalysisModeManager for parallel processing per ADR-023
3. Create comprehensive result aggregation and comparison
4. Integrate all systems with existing MultiAgentCoordinator

**Phase 4 - Integration & Testing**:

1. End-to-end testing of all ADR features
2. Performance validation of 128K context and parallel processing
3. User experience testing of multipage navigation
4. Security validation of input handling and export functionality

### Required Dependency Additions

**Add to pyproject.toml dependencies**:

```toml
# UI enhancements for ADR-013
"streamlit-aggrid>=1.0.0",
"plotly>=5.0.0",

# Template system for ADR-020
"jinja2>=3.1.0",

# Export formatting for ADR-022
"rich>=13.0.0",
"fpdf2>=2.7.0",  # PDF export

# Additional memory optimization for ADR-021
"sqlalchemy>=2.0.0",  # SQLite persistence
```

### Directory Structure Changes

**New Structure Required**:

```text
src/
├── ui/
│   ├── streamlit_app.py          # Main app with st.navigation
│   ├── pages/                     # Navigation pages
│   │   ├── upload.py             # Document upload
│   │   ├── analysis.py           # Analysis mode selection
│   │   ├── chat.py               # Chat interface
│   │   ├── export.py             # Export functionality
│   │   └── settings.py           # Configuration
│   ├── components/                # Reusable components
│   │   ├── document_table.py     # AgGrid document display
│   │   ├── analytics.py          # Plotly dashboards
│   │   ├── prompt_selector.py    # Template selection
│   │   ├── memory_status.py      # Context indicators
│   │   └── export_options.py     # Export configuration
│   ├── session/                   # State management
│   │   ├── memory_bridge.py      # LangGraph integration
│   │   └── persistence.py        # Conversation storage
│   └── export/                    # Export system
│       ├── manager.py            # ExportManager class
│       ├── templates/            # Jinja2 templates
│       └── models.py             # Pydantic export models
├── prompts/                       # Enhanced prompt system
│   ├── manager.py                # PromptTemplateManager
│   ├── registry.py               # Template combinations
│   └── optimizer.py              # DSPy integration
└── analysis/                      # Analysis mode system
    ├── mode_manager.py           # DocumentAnalysisModeManager
    └── processors.py             # Parallel processing
```

### Success Criteria - ADR Compliance Validation

**ADR-013 Compliance (User Interface Architecture)**:

- [ ] **BLOCKING**: `src/app.py` completely deleted and replaced with st.navigation multipage architecture
- [ ] **BLOCKING**: AgGrid tables operational for document display
- [ ] **BLOCKING**: Plotly analytics dashboards functional
- [ ] **BLOCKING**: st.write_stream() implemented for all streaming responses
- [ ] **BLOCKING**: .streamlit/config.toml theming configured (no unsafe HTML)
- [ ] **VALIDATION**: Multipage navigation <2s load time

**ADR-016 Compliance**:

- [ ] Native st.session_state used throughout
- [ ] LangGraph memory integration via StreamlitChatMemory
- [ ] Conversation persistence across sessions
- [ ] InMemoryStore for long-term storage

**ADR-020 Compliance (Prompt Template System)**:

- [ ] **BLOCKING**: `src/prompts.py` completely deleted and replaced with PromptTemplateManager
- [ ] **BLOCKING**: TemplateRegistry with 1,600+ combinations (10 roles × 10 tones × 16 detail levels)
- [ ] **BLOCKING**: Jinja2 template rendering system operational
- [ ] **BLOCKING**: DSPy optimization integration functional
- [ ] **BLOCKING**: Template selection UI components working
- [ ] **VALIDATION**: Template compilation <50ms for any combination

**ADR-021 Compliance (Chat Memory & Context Management)**:

- [ ] **BLOCKING**: 32K memory limit completely replaced with 128K context (131,072 tokens)
- [ ] **BLOCKING**: FP8 KV cache optimization integration for Qwen3-4B-Instruct-2507-FP8
- [ ] **BLOCKING**: StreamlitChatMemory bridge for LangGraph integration
- [ ] **BLOCKING**: Context trimming strategies with user feedback
- [ ] **BLOCKING**: SQLite conversation persistence operational
- [ ] **VALIDATION**: 128K context conversations under 2s response time

**ADR-022 Compliance (Export & Output Formatting)**:

- [ ] **MISSING**: Type-safe export system with Pydantic models (currently not implemented)
- [ ] **MISSING**: JSON, Markdown, PDF, and Rich console format support
- [ ] **MISSING**: Jinja2 template system for customizable output formatting
- [ ] **MISSING**: Export progress indicators and batch processing
- [ ] **MISSING**: Template customization UI for different output styles
- [ ] **VALIDATION**: Export generation <5s for typical documents

**ADR-023 Compliance (Analysis Mode Strategy)**:

- [ ] **MISSING**: Analysis mode selection UI (separate/combined/auto) - currently not implemented
- [ ] **MISSING**: DocumentAnalysisModeManager for parallel processing
- [ ] **MISSING**: Parallel document processing with 3-5x speedup
- [ ] **MISSING**: Result aggregation and cross-document comparison
- [ ] **MISSING**: Processing status indicators for multiple documents
- [ ] **VALIDATION**: 3-5x speedup for multiple documents in parallel mode

**Performance Targets (ADR-MANDATED)**:

- [ ] **Page Load**: <2s for multipage navigation (ADR-013)
- [ ] **Template Speed**: <50ms for any of 1,600+ combinations (ADR-020)
- [ ] **Context Handling**: <100ms for 128K memory operations (ADR-021)
- [ ] **Export Speed**: <5s for document exports (ADR-022)
- [ ] **Parallel Speedup**: 3-5x improvement for multiple documents (ADR-023)

**Architecture Validation**:

- [ ] **Verify deletion**: Confirm `src/app.py` and `src/prompts.py` completely removed
- [ ] **Verify creation**: Confirm all ADR-mandated components implemented
- [ ] **Verify integration**: Test LangGraph memory bridge, agent coordination
- [ ] **Verify compliance**: All 6 ADR requirements fully satisfied

## 14. Traceability

### Source Documents

- ADR-013: User Interface Architecture (multipage navigation, AgGrid, Plotly)
- ADR-016: UI State Management (native Streamlit + LangGraph integration)
- ADR-020: Prompt Template System (1,600+ combinations, DSPy optimization)
- ADR-021: Chat Memory & Context Management (128K context, FP8 optimization)
- ADR-022: Export & Output Formatting (type-safe multi-format export)
- ADR-023: Analysis Mode Strategy (parallel processing, mode selection)
- PRD FR-12: Configurable UI requirement

### Related Specifications

- 001-multi-agent-coordination.spec.md
- 003-document-processing.spec.md
- 004-infrastructure-performance.spec.md
