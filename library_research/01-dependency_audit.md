# Dependency Hygiene Analysis Report

**Date:** 2025-01-12  

**Project:** DocMind AI LLM  

**Analyzer:** @dep-audit subagent  

## Executive Summary

Performed comprehensive dependency analysis on pyproject.toml with 66 main dependencies and 331 total resolved packages. Found several optimization opportunities including unused dependencies, missing explicit declarations, and potential dev/main misplacement.

**Key Findings:**

- 8 potentially unused dependencies identified

- 3 dependencies that should be explicit rather than transitive

- 2 dependencies that could be moved to dev group

- Several dependencies are legitimately used but only in specific contexts

## Analysis Methodology

1. **Static Import Analysis**: Scanned all Python files in `src/`, `tests/`, and legacy `utils/` directories
2. **Dependency Tree Analysis**: Used `uv tree` to understand transitive dependencies
3. **Code Pattern Matching**: Searched for indirect usage patterns (dynamic imports, entry points, etc.)
4. **Context Analysis**: Reviewed ADRs, documentation, and configuration files

## Current Dependency Breakdown

### Main Dependencies (66 packages)

- **LLM/AI Core**: 26 packages (LlamaIndex ecosystem, OpenAI, Ollama, etc.)

- **Document Processing**: 8 packages (Unstructured, PyMuPDF, python-docx, etc.)

- **ML/Torch**: 7 packages (torch, transformers, etc.)

- **Vector/DB**: 4 packages (Qdrant, fastembed, etc.)

- **Utility**: 15 packages (streamlit, loguru, pydantic, etc.)

- **Media Processing**: 6 packages (moviepy, whisper, etc.)

## Detailed Findings

### Category 1: Potentially Unused Dependencies ⚠️

#### 1.1 Polars (polars==1.31.0)

- **Status**: Not found in any import statements

- **Risk**: LOW - Only used for DataFrame operations

- **Evidence**: No direct imports found in codebase

- **Recommendation**: REMOVE unless used for specific data processing

#### 1.2 PyArrow (pyarrow<21.0.0) 

- **Status**: Only found as transitive dependency

- **Risk**: MEDIUM - Required by multiple packages

- **Evidence**: Used by streamlit, phoenix, pandas

- **Recommendation**: KEEP - Transitive dependency, removing could break functionality

#### 1.3 MoviePy (moviepy==2.2.1)

- **Status**: Only found in test mocks

- **Risk**: MEDIUM - Video processing capability

- **Evidence**: `tests/unit/test_resource_management.py` mocks VideoFileClip

- **Recommendation**: EVALUATE - Keep if video processing is planned feature

#### 1.4 Numba (numba==0.61.2)

- **Status**: Only found as transitive dependency

- **Risk**: LOW - Required by whisper

- **Evidence**: Required by openai-whisper package

- **Recommendation**: KEEP - Transitive dependency for whisper

#### 1.5 Ragatouille (ragatouille==0.0.9.post2)

- **Status**: Not found in current imports

- **Risk**: HIGH - Large dependency tree

- **Evidence**: No imports found in active codebase

- **Recommendation**: REMOVE - Adds 20+ transitive dependencies

#### 1.6 TorchVision (torchvision==0.22.1)

- **Status**: Not found in imports, large package

- **Risk**: MEDIUM - Computer vision capability

- **Evidence**: No direct usage found

- **Recommendation**: EVALUATE - Remove if not needed for image processing

### Category 2: Missing Explicit Dependencies 🔍

#### 2.1 PSUtil

- **Status**: Used but not explicitly declared

- **Risk**: HIGH - System monitoring functionality

- **Evidence**: Used in `src/utils/monitoring.py`

- **Current**: Transitive via phoenix, fastembed

- **Recommendation**: ADD as explicit dependency

#### 2.2 Requests

- **Status**: Used but not explicitly declared  

- **Risk**: MEDIUM - HTTP client functionality

- **Evidence**: Likely used for API calls

- **Current**: Transitive via multiple packages

- **Recommendation**: ADD as explicit dependency if directly used

### Category 3: Dev Dependencies Misplacement 📦

#### 3.1 OpenInference Instrumentation (openinference-instrumentation-llama-index==4.3.2)

- **Status**: Development/monitoring tool

- **Risk**: LOW - Only for observability

- **Evidence**: Only used when Phoenix observability is enabled

- **Recommendation**: MOVE to dev group

#### 3.2 Arize Phoenix (arize-phoenix==11.13.2)

- **Status**: Development/observability tool

- **Risk**: LOW - Large dependency tree for dev tool

- **Evidence**: Only used optionally via checkbox in UI

- **Recommendation**: MOVE to dev group

### Category 4: Legitimately Used Dependencies ✅

#### 4.1 Media Processing Stack

- **OpenAI Whisper**: Used for audio transcription (found in test mocks)

- **Pillow**: Required for image processing (transitive via multiple packages)

- **PyMuPDF**: Used for PDF processing (found in test comments)

#### 4.2 LlamaIndex Ecosystem
All llama-index-* packages are legitimately used:

- Core functionality in `src/agents/` and `src/utils/`

- Vector stores, embeddings, LLMs properly integrated

- Multimodal capabilities actively used

#### 4.3 Core Infrastructure

- **Streamlit**: Main UI framework

- **Pydantic**: Data validation throughout codebase

- **Loguru**: Logging system

- **DisCache**: Caching system  

- **Qdrant-client**: Vector database client

- **Torch**: ML backend

- **Transformers**: Model loading and inference

### Category 5: Dependency Tree Concerns 🌳

#### 5.1 Heavy Transitive Dependencies

- **unstructured[all-docs]**: 25+ extra packages for document processing

- **ragatouille**: 20+ packages including faiss, langchain

- **arize-phoenix**: 30+ packages for observability

- **transformers**: Large ML framework

#### 5.2 Version Conflicts Risk

- Multiple packages depend on different versions of:
  - `numpy` (shared across many packages)
  - `torch` (ML ecosystem dependencies)
  - `pydantic` (API serialization)

## Safety Considerations

### Dependencies to NEVER Remove
1. **Entry Point Dependencies**: Packages that provide entry points even without imports
2. **Runtime Plugin Systems**: LlamaIndex extensions loaded dynamically
3. **Streamlit Extensions**: UI components loaded at runtime
4. **Framework Extensions**: Packages that extend functionality through registration

### False Positive Risks
1. **Dynamic Imports**: Code using `importlib` or `__import__`
2. **Configuration-Based Loading**: Models loaded based on settings
3. **Conditional Imports**: Dependencies used only in specific environments
4. **Test-Only Dependencies**: Packages only used in test scenarios

## Environment Health Analysis

### Current Status: ✅ GOOD

- **Resolution**: All 331 packages resolve without conflicts

- **Version Constraints**: Appropriately specified

- **Python Compatibility**: Supports 3.10-3.12 range

- **Platform Support**: Linux/MacOS/Windows compatible

### Risk Areas
1. **Large Dependency Count**: 331 resolved packages is substantial
2. **Version Pinning**: Several exact pins may need updates
3. **Transitive Complexity**: Some packages have deep dependency trees

## Optimization Opportunities

### Immediate (Low Risk)
1. Remove unused polars if data processing not needed
2. Move phoenix/observability to dev dependencies
3. Add explicit psutil dependency

### Evaluation Needed (Medium Risk)
1. Assess moviepy need for video processing features
2. Evaluate ragatouille usage for ColBERT reranking
3. Consider torchvision necessity for vision tasks

### Long-term (High Value)
1. Regular dependency auditing process
2. Implement dependency update automation
3. Consider dependency constraints for security

## Recommendations Summary

**High Priority:**

- Remove ragatouille if not actively used (-20 packages)

- Move phoenix/observability to dev group

- Add explicit psutil dependency

**Medium Priority:**  

- Evaluate moviepy and torchvision necessity

- Remove polars if unused

- Regular audit process implementation

**Low Priority:**

- Monitor for version conflicts

- Consider dependency constraints

- Optimize for specific deployment scenarios
