---
allowed-tools: Task, TodoWrite, Read, LS, Glob, Bash(git status), Bash(git log:*), Bash(rg:*), Bash(fd:*)
argument-hint: [project-type] (optional: ai-system, web-app, api, mobile)
description: Complete ADR-to-specification alignment workflow with drift audit and spec rewrites
model: claude-3-5-sonnet-20241022
---

# ADR-Specification Alignment Workflow

## Overview

This workflow performs a comprehensive alignment between Architecture Decision Records (ADRs) and specification documents, ensuring all specs are implementation-ready and ADR-compliant. Based on proven methodology from DocMind AI project achieving 66% → 100% ADR compliance.

## Project Context

**Project Type**: $ARGUMENTS
**Working Directory**: !`pwd`
**Git Status**: !`git status --porcelain`
**Recent Commits**: !`git log --oneline -5`

## Phase 1: Comprehensive Drift Audit (15-20 minutes)

### Step 1.1: Create Task Tracking

Launch TodoWrite to establish comprehensive task tracking for the entire workflow:

```
TodoWrite with tasks:
1. "Launch adr-drift-auditor for comprehensive documentation audit" (in_progress)
2. "Audit ALL ADRs for consistency and implementation alignment" (pending)
3. "Review all spec files against ADR requirements" (pending) 
4. "Cross-validate specifications with implementation in src/" (pending)
5. "Generate comprehensive drift analysis and correction recommendations" (pending)
6. "Launch technical-docs-architect to rewrite specification files" (pending)
7. "Update all spec files to be ADR-compliant and implementation-ready" (pending)
8. "Validate final alignment and create implementation roadmap" (pending)
```

### Step 1.2: Launch ADR Drift Auditor

Execute comprehensive drift audit using the adr-drift-auditor subagent:

**Audit Scope:**

- ALL Architecture Decision Records in @docs/adrs/
- ALL specification files in @docs/specs/
- Current implementation validation against @src/ directory
- Requirements tracking and compliance matrices

**Key Focus Areas:**

- ADR compliance gaps and violations
- Specification-to-implementation misalignments  
- Missing requirements and documentation drift
- Architecture consistency across all documents

**Agent Invocation:**

```
Task(subagent_type="adr-drift-auditor", description="Comprehensive ADR-spec-implementation drift audit", prompt="...")
```

**Expected Deliverables:**

- Complete drift analysis report in `docs/reports/adrs/`
- Priority-ranked misalignment identification
- Specific correction recommendations
- ADR compliance percentage and gap analysis

## Phase 2: Specification Rewrite for ADR Compliance (30-45 minutes)

### Step 2.1: Launch Technical Documentation Architect

Execute specification rewrites using technical-docs-architect subagent:

**Rewrite Scope:**

- Transform all specification files to be ADR-compliant
- Fill identified gaps with implementable content
- Create step-by-step implementation guides
- Ensure technical precision with code examples

**Critical Requirements:**

- REWRITE specification content (don't just document gaps)
- Make specs implementation-ready with detailed instructions
- Include specific library versions, configuration examples
- Provide ADR traceability for every requirement

**Agent Invocation:**

```
Task(subagent_type="technical-docs-architect", description="Rewrite spec files with ADR-compliant content", prompt="...")
```

### Step 2.2: Focus Areas for Specification Updates

**Document Processing Specifications:**

- Update from generic approaches to specific ADR-required implementations
- Include direct library integration patterns (e.g., Unstructured.io vs wrappers)
- Add performance benchmarks and validation criteria

**User Interface Specifications:**  

- Transform from monolithic to modern architecture patterns
- Add multi-page navigation, state management, export systems
- Include component specifications and integration details

**Infrastructure Specifications:**

- Reflect current successful implementations
- Update performance optimization achievements
- Include configuration management patterns

**Requirements Tracking:**

- Add atomic requirements for all non-compliant ADRs
- Update completion percentages based on audit findings
- Create complete traceability matrices

## Phase 3: Implementation Roadmap Creation (10-15 minutes)

### Step 3.1: Generate Implementation Plan

Create comprehensive implementation roadmap based on updated specifications:

**Roadmap Components:**

- Phase-by-phase implementation plan
- Priority-ranked feature development
- Dependencies and critical path analysis
- Success criteria and validation metrics

### Step 3.2: Final Validation and Todo Completion

Systematically complete all workflow tasks and validate results:

**Validation Checklist:**

- [ ] All specification files are ADR-compliant
- [ ] Every spec contains implementation-ready instructions
- [ ] Developers can follow specs to implement features
- [ ] Complete traceability from ADRs to requirements to implementation
- [ ] Clear prioritization for remaining work

## Expected Outcomes

### Immediate Results

- **Drift Audit Report**: Comprehensive analysis of ADR compliance gaps
- **Updated Specifications**: Fully rewritten, implementation-ready spec files
- **Requirements Matrix**: Complete traceability with priority rankings
- **Implementation Roadmap**: Phase-by-phase development plan

### Quality Standards

- **ADR Compliance**: All specifications align with architectural decisions
- **Implementation Ready**: Step-by-step instructions with code examples
- **Developer Focused**: Clear, actionable guidance for feature development
- **Validation Criteria**: Measurable success metrics for each requirement

## Workflow Customization by Project Type

### AI System Projects (like DocMind AI)

- Focus on multi-agent coordination, LLM integration, vector databases
- Emphasize performance optimization, GPU utilization, memory management
- Include model serving, embedding strategies, retrieval architectures

### Web Applications

- Focus on frontend frameworks, API design, database schemas
- Emphasize responsive design, state management, authentication
- Include deployment strategies, monitoring, performance optimization

### API Projects

- Focus on OpenAPI specifications, endpoint design, data models
- Emphasize authentication, rate limiting, error handling
- Include testing strategies, documentation, versioning

### Mobile Applications

- Focus on platform-specific requirements, UI/UX guidelines
- Emphasize performance, offline capabilities, native integrations
- Include app store requirements, testing, distribution

## Success Metrics

### Process Metrics

- **Audit Completeness**: 100% of ADRs and specs reviewed
- **Gap Identification**: All misalignments documented with evidence
- **Specification Quality**: Implementation-ready with code examples
- **Traceability**: Complete ADR → requirement → implementation mapping

### Outcome Metrics

- **ADR Compliance**: Target 90%+ compliance after implementation
- **Implementation Velocity**: Specs enable faster, more accurate development
- **Quality Improvement**: Reduced rework and architectural inconsistencies
- **Team Alignment**: Clear, shared understanding of system architecture

## Troubleshooting

### Common Issues

**"Specifications still show gaps instead of solutions"**

- Ensure technical-docs-architect agent received explicit rewrite instructions
- Verify agent was told to fill gaps with implementable content, not document them
- Re-run with emphasis on "REWRITE content to be ADR-compliant"

**"ADR compliance percentage seems low"**  

- Normal for complex systems - focus on critical architectural decisions first
- Prioritize blocking and high-impact ADRs before comprehensive compliance
- Consider updating ADR requirements if implementation proves superior approach

**"Implementation instructions too abstract"**

- Request specific code examples, configuration snippets, library versions
- Ask for step-by-step procedures with validation criteria
- Include error handling, troubleshooting, and testing guidance

### Quality Checkpoints

1. **Post-Audit**: Drift report identifies specific, actionable gaps
2. **Post-Rewrite**: Specifications contain implementable instructions with examples
3. **Post-Validation**: Development team can begin implementation immediately
4. **Post-Implementation**: Features match architectural decisions and specifications

---

## Usage Examples

```bash
# Basic usage for AI system
/adr-spec-alignment ai-system

# For web application project  
/adr-spec-alignment web-app

# For API-focused project
/adr-spec-alignment api

# Without specific project type (auto-detects)
/adr-spec-alignment
```

This workflow transforms documentation from aspirational to implementable, ensuring architectural decisions are reflected in specifications that developers can follow to build compliant, high-quality systems.
