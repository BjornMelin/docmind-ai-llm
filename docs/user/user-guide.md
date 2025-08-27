# DocMind AI User Guide

Master DocMind AI's powerful document analysis capabilities with this comprehensive guide to daily workflows and core features.

## Overview

DocMind AI transforms how you work with documents through:

- **🤖 Multi-Agent Intelligence**: 5 specialized AI agents coordinate to handle complex queries
- **🔍 Advanced Search**: Semantic + keyword hybrid search in 100+ languages  
- **📄 Universal Format Support**: PDF, DOCX, TXT, XLSX, and 15+ other formats
- **🧠 128K Context Window**: Analyze entire documents without chunking
- **⚡ Local-First**: Complete privacy with offline operation

## Document Management

### Supported File Formats

DocMind AI processes a wide range of document types:

**Primary Formats:**

- **PDF**: Full text extraction with image recognition
- **Microsoft Office**: DOCX, XLSX, PPTX with complete formatting preservation
- **Text Files**: TXT, MD, RTF with encoding auto-detection
- **Web Formats**: HTML, XML, JSON with structure preservation

**Additional Formats:**

- **Archives**: ZIP files (extracts and processes contents)
- **Email**: MSG files with attachment processing
- **eBooks**: EPUB with chapter navigation
- **Code**: PY, JS, Java, and 20+ programming languages
- **Specialized**: CSV, TSV with data structure recognition

### Document Upload Workflow

#### Basic Upload Process

1. **Access Upload Interface**
   - Click "Upload Documents" in the sidebar
   - Or drag and drop files directly onto the main interface

2. **Select Files**

   ```text
   ✅ Single files up to 100MB
   ✅ Multiple files in batch
   ✅ Mixed file types simultaneously
   ✅ Folder contents (via ZIP)
   ```

3. **Monitor Processing**
   - Watch real-time progress bar
   - View processing status for each file
   - Check for any format-specific warnings

4. **Verify Success**
   - Confirm files appear in document list
   - Check document preview availability
   - Validate text extraction quality

#### Advanced Upload Options

**Batch Processing Settings:**

```bash
# Configure in .env for optimal batch processing
DOCMIND_CHUNK_SIZE=512              # Optimal for most documents
DOCMIND_CHUNK_OVERLAP=50            # Maintains context continuity
DOCMIND_MAX_DOCUMENT_SIZE_MB=100    # Prevents memory issues
DOCMIND_PARALLEL_PROCESSING=true    # Faster multi-file uploads
```

**Document Preview Controls:**

- **Enable Image Preview**: View PDF pages and embedded images
- **Text Snippet Preview**: See extracted text samples
- **Metadata Display**: File size, format, processing time
- **Structure Analysis**: Headings, sections, tables detected

### Document Organization

#### Document Lists and Management

**Document Status Indicators:**

- 🟢 **Processed**: Ready for querying
- 🟡 **Processing**: Currently being analyzed
- 🔴 **Error**: Processing failed (check logs)
- 📄 **Preview**: Text preview available

**Document Actions:**

- **View**: See extracted text and structure
- **Reprocess**: Re-analyze if initial processing failed
- **Remove**: Delete from current session
- **Export**: Save processed text or metadata

#### Session Management

**Save and Load Sessions:**

```python
# Sessions automatically preserve:
# - Uploaded documents and processing state
# - Conversation history and context
# - Configuration settings
# - Agent coordination preferences
```

**Session Best Practices:**

- Save sessions before major configuration changes
- Load previous sessions to continue complex analyses
- Export session data for backup and sharing
- Clear sessions periodically to free memory

## Query Interface and Interaction

### Understanding the Multi-Agent System

DocMind AI uses 5 specialized agents that coordinate automatically:

#### Agent Roles and Specializations

**🎯 Query Router Agent**:

- **Purpose**: Analyzes query complexity and determines optimal processing strategy
- **When Active**: Every query (acts as entry point)
- **Decision Making**: Routes to appropriate specialized agents
- **Performance**: <100ms response time

**📋 Query Planner Agent**:

- **Purpose**: Breaks complex questions into manageable sub-tasks
- **When Active**: Multi-part questions, comparison queries, analytical requests
- **Capabilities**: Sequential task planning, dependency management
- **Example**: "Compare X and Y, then recommend Z" → 3 sequential sub-tasks

**🔍 Retrieval Expert Agent**:

- **Purpose**: Finds most relevant content using advanced search strategies
- **When Active**: All document search operations
- **Search Types**: Dense semantic, hybrid, multi-query, graph-based
- **Performance**: Processes 8K token contexts in <500ms

**🔧 Result Synthesizer Agent**:

- **Purpose**: Combines information from multiple sources into coherent responses
- **When Active**: Complex queries requiring information fusion
- **Capabilities**: Cross-document synthesis, conflict resolution, source attribution
- **Quality**: Maintains consistent tone and comprehensive coverage

**✅ Response Validator Agent**:

- **Purpose**: Ensures response quality, accuracy, and completeness
- **When Active**: All responses before delivery
- **Validation Checks**: Factual consistency, source accuracy, completeness scoring
- **Quality Threshold**: Minimum 0.7 validation score (configurable)

#### Agent Coordination Examples

**Simple Query Flow:**

```text
Query: "What is the project budget?"

Router → Retrieval → Validator → Response
Time: ~1 second
```

**Complex Query Flow:**

```text
Query: "Compare Q1 vs Q2 performance, identify trends, and recommend actions"

Router → Planner → [Retrieve Q1, Retrieve Q2, Analyze Trends] → Synthesizer → Validator → Response  
Time: ~3 seconds
```

### Query Types and Best Practices

#### Effective Query Formulation

**✅ Specific Information Requests**:

*Good Examples:*

```text
What are the key findings from the 2023 financial audit?
List the project milestones mentioned in the requirements document.
Who are the stakeholders identified in the project charter?
```

*Why These Work:*

- Clear, specific scope
- Identifiable information type
- Context-appropriate language

**✅ Comparative Analysis Queries**:

*Good Examples:*

```text
Compare the marketing strategies proposed in documents A and B.
How do the Q1 sales figures differ from Q2 results?
What are the pros and cons of each implementation approach?
```

*Why These Work:*

- Clear comparison subjects
- Specific comparison criteria
- Expected analytical output

**✅ Multi-Step Analytical Questions**:

*Good Examples:*

```text
What risks are identified in the project plan, how severe are they, and what mitigation strategies are proposed?
Analyze the customer feedback themes, prioritize by frequency, and suggest improvements.
```

*Why These Work:*

- Logical step progression
- Clear analytical framework
- Actionable output expected

#### Query Optimization Tips

**🎯 Be Specific About Context**:

```text
❌ "What do the documents say about revenue?"
✅ "What was the total revenue for Q3 2023 according to the financial report?"
```

**🎯 Use Document-Specific Language**:

```text
❌ "Tell me about the thing in section 2"
✅ "Explain the risk assessment methodology described in section 2 of the security audit"
```

**🎯 Frame Complex Questions Clearly**:

```text
❌ "Compare everything and tell me what's best"
✅ "Compare the three proposed architectures based on cost, performance, and maintainability, then recommend the optimal choice"
```

### Response Interpretation

#### Understanding Response Components

**Main Response Structure:**

- **Primary Answer**: Direct response to your question
- **Supporting Evidence**: Quotes and references from source documents
- **Source Attribution**: Specific document and section references
- **Confidence Metadata**: Quality scores and processing information

**Quality Indicators:**

**Validation Score (0.0 - 1.0):**

- **0.9-1.0**: Excellent - High confidence, well-supported answer
- **0.8-0.9**: Good - Solid answer with minor limitations
- **0.7-0.8**: Acceptable - Adequate answer, some uncertainty
- **<0.7**: Review needed - May lack sufficient evidence

**Source Count and Distribution:**

- **Multiple Sources**: More comprehensive analysis
- **Diverse Sections**: Broader coverage of topic
- **Recent References**: Up-to-date information prioritized

**Processing Time Indicators:**

- **<1s**: Simple retrieval, direct answers
- **1-3s**: Standard multi-agent coordination
- **3-5s**: Complex analysis or large document processing
- **>5s**: May indicate system load or very complex queries

#### Agent Contribution Analysis

Monitor which agents contributed to understand query complexity:

**Agent Usage Patterns:**

```text
Simple Query: Router + Retrieval + Validator
Complex Query: Router + Planner + Multiple Retrieval + Synthesizer + Validator
Comparative Query: Router + Planner + Parallel Retrieval + Synthesizer + Validator
```

**Performance Optimization:**

- Queries consistently using all agents may benefit from simplification
- Fast router-only responses indicate well-matched simple queries
- Frequent fallback modes suggest configuration optimization needed

## Advanced Query Techniques

### Multi-Document Analysis

#### Cross-Document Queries

**Relationship Analysis:**

```text
How do the requirements in document A relate to the implementation in document B?
Which solutions mentioned in the technical spec address the problems identified in the analysis report?
```

**Consistency Checking:**

```text
Are there any contradictions between the financial projections and the business plan?
Do the test results align with the expected outcomes in the design document?
```

**Gap Analysis:**

```text
What requirements from the specification are not addressed in the implementation guide?
Which risks identified in the audit are not covered by the mitigation plan?
```

#### Sequential Document Processing

**Time-Series Analysis:**

```text
How have the project milestones evolved from the initial plan to the latest update?
Track the changes in budget allocations across the quarterly reports.
```

**Version Comparison:**

```text
What are the key differences between version 1.0 and 2.0 of the product specification?
Compare the original proposal with the final approved version.
```

### Conversation Context Management

#### Building on Previous Queries

**Contextual Follow-ups:**

```text
Initial: "What are the main project risks?"
Follow-up: "Which of these risks have the highest impact?"
Follow-up: "What mitigation strategies are proposed for the high-impact risks?"
```

**Reference Previous Results:**

```text
Based on the budget analysis you just provided, which departments are over their allocation?
Using the performance metrics from my last query, what trends do you see over time?
```

#### Context Optimization

**Managing Context Windows:**

- DocMind AI automatically manages 128K token context
- Long conversations may trigger context summarization
- Start new sessions for unrelated topics
- Use specific document references to maintain focus

**Memory Best Practices:**

```bash
# Configure context management
MAX_CONTEXT_TOKENS=65000          # Preserves context while maintaining performance
CONTEXT_PRESERVATION=true         # Maintains conversation history
ENABLE_CONTEXT_SUMMARIZATION=true # Automatically summarizes when needed
```

## Configuration and Customization

### User Interface Customization

#### Sidebar Configuration

**Model and Backend Settings:**

- **Model Selection**: Choose from available models
- **Backend Configuration**: Ollama, vLLM, or OpenAI API
- **Context Window**: Adjust based on document size and complexity
- **GPU Toggle**: Enable/disable GPU acceleration

**Analysis Options:**

- **Multi-Agent Mode**: Toggle 5-agent coordination
- **Search Strategy**: Auto, Dense, Hybrid, or Graph
- **Response Length**: Concise, Standard, or Detailed
- **Language Preference**: Set for multilingual documents

#### Theme and Display Settings

**Interface Customization:**

```bash
# Available themes
STREAMLIT_THEME=dark              # Dark mode for extended use
STREAMLIT_THEME=light             # Light mode for better readability
```

**Display Options:**

- **Document Preview**: Toggle image and text previews
- **Agent Status**: Show/hide agent coordination details  
- **Performance Metrics**: Display processing times and system stats
- **Advanced Logging**: Enable detailed operation logging

### Performance Configuration

#### Memory and Processing Settings

**GPU Memory Management:**

```bash
# Adjust based on your hardware
DOCMIND_GPU_MEMORY_UTILIZATION=0.85  # RTX 4090 (16GB) optimal
DOCMIND_GPU_MEMORY_UTILIZATION=0.90  # RTX 4090 (24GB) optimal
DOCMIND_GPU_MEMORY_UTILIZATION=0.75  # Conservative setting
```

**Processing Optimization:**

```bash
# Agent performance tuning
AGENT_CONCURRENCY_LIMIT=5         # Parallel agent operations
AGENT_TIMEOUT_SECONDS=30          # Agent response timeout  
RETRY_ATTEMPTS=3                  # Failed operation retries
CACHE_TTL_SECONDS=300             # Result caching duration
```

#### Quality and Accuracy Settings

**Response Quality Controls:**

```bash
# Validation settings
MIN_VALIDATION_SCORE=0.7          # Minimum acceptable response quality
ENABLE_HALLUCINATION_CHECK=true   # Verify information accuracy
SOURCE_ATTRIBUTION_REQUIRED=true  # Ensure proper source citation
```

**Search Precision:**

```bash
# Retrieval settings
RETRIEVAL_TOP_K=20                # Number of candidate results
RERANKING_ENABLED=true            # Use advanced result reranking
RRF_ALPHA=0.7                     # Dense vs sparse search balance
```

## Best Practices and Workflows

### Daily Usage Patterns

#### Document Analysis Workflow

**1. Preparation Phase:**

- Upload related documents in batches
- Verify processing success for all files
- Check document preview quality
- Organize files by topic or project

**2. Exploration Phase:**

- Start with broad overview questions
- Identify key topics and themes
- Note important sections and references
- Build understanding of document structure

**3. Analysis Phase:**

- Ask specific, targeted questions
- Use comparative analysis for multiple documents
- Drill down into details with follow-up queries
- Extract actionable insights and recommendations

**4. Documentation Phase:**

- Export relevant findings and responses
- Save session for future reference
- Document key insights and decisions
- Prepare summaries for stakeholders

#### Collaborative Workflows

**Team Document Review:**

```bash
# Share session configurations
# Export findings in consistent formats
# Maintain audit trail of analysis decisions
# Enable reproducible analysis results
```

**Client Presentation Preparation:**

```bash
# Use high validation score responses (>0.9)
# Include clear source attributions
# Prepare supporting evidence from multiple documents
# Test queries in advance to ensure reliability
```

### Performance Optimization

#### Query Optimization Strategies

**Start Simple, Then Elaborate:**

```text
1. "What are the main topics in this document?"
2. "Tell me more about the budget section"
3. "Compare the budget allocation between Q1 and Q2"
4. "What factors contributed to the budget variance?"
```

**Use Context Effectively:**

```text
✅ Build on previous responses
✅ Reference specific documents by name
✅ Maintain conversation threads for related topics
❌ Jump between unrelated topics in same session
```

**Batch Related Queries:**

```text
✅ Ask follow-up questions immediately
✅ Explore related topics in sequence
✅ Use conversation context for efficiency
❌ Start new sessions for each related question
```

#### System Resource Management

**Monitor Performance Indicators:**

- **Response Time**: Target <3 seconds for complex queries
- **GPU Utilization**: Should stay <90% during processing
- **Memory Usage**: Monitor system RAM consumption
- **Cache Hit Rate**: Higher rates indicate efficient operation

**Optimize Based on Usage Patterns:**

```bash
# For frequent simple queries
PERFORMANCE_MODE=fast
AGENT_CONCURRENCY_LIMIT=3

# For complex analytical work
PERFORMANCE_MODE=thorough
ENABLE_RESULT_CACHING=true
```

## Export and Integration

### Response Export Options

#### Export Formats

**Markdown Export:**

- Clean formatting with proper headers
- Source citations as clickable links
- Code blocks and tables preserved
- Suitable for documentation systems

**JSON Export:**

- Structured data with metadata
- Agent contribution details
- Source attribution with locations
- Machine-readable for further processing

**PDF Export:**

- Professional formatted output
- Complete with source references
- Suitable for reports and presentations
- Maintains formatting and structure

#### Export Workflow

1. **Complete Analysis**: Finish your document analysis session
2. **Review Results**: Ensure response quality meets requirements
3. **Select Format**: Choose appropriate export format
4. **Configure Options**: Set export parameters and filters
5. **Generate Export**: Create and download export file
6. **Verify Quality**: Check exported content accuracy

### API and Programmatic Access

#### Basic Programmatic Usage

```python
# Example programmatic interface
from src.config import settings
from src.utils.document import load_documents_unstructured
from src.utils.embedding import create_index_async
from src.agents.coordinator import get_agent_system

# Load and process documents
documents = await load_documents_unstructured(file_paths, settings)
index = await create_index_async(documents, settings)

# Create agent system
agent_system = get_agent_system(index, settings)

# Query documents
response = await agent_system.arun("Analyze the key findings in these documents")
```

## Getting Help and Support

### Built-in Help Features

#### System Status and Health Monitoring

**Health Indicators:**

- 🟢 **All Systems Operational**: Normal operation
- 🟡 **Performance Degraded**: Some slowdown expected  
- 🔴 **Issues Detected**: Check logs and configuration

**Performance Monitoring:**

```bash
# Check system health
tail -f logs/app.log | grep "ERROR\|WARNING"

# Monitor resource usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1
```

#### Log Analysis

**Key Log Patterns:**

```bash
# Agent coordination issues
grep "agent_timeout\|coordination_failed" logs/app.log

# Performance problems
grep "slow_response\|memory_warning" logs/app.log

# Quality issues
grep "validation_score.*0\.[0-6]" logs/app.log
```

### Community Resources

**Documentation:**

- **Advanced Features Guide**: [advanced-features.md](advanced-features.md)
- **Troubleshooting Reference**: [troubleshooting-reference.md](troubleshooting-reference.md)
- **Developer Documentation**: [../developers/](../developers/)

**Support Channels:**

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community Q&A and best practices
- **Documentation Updates**: Contribute improvements and fixes

---

**Ready for advanced features?** Continue to [Advanced Features](advanced-features.md) to unlock power-user capabilities and complex configurations.
