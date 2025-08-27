# Delta Specification Format Template

This template defines the high-quality structure and format for creating delta specifications based on the successful format demonstrated in the DocMind AI project's delta specifications (001.1, 002.1).

## Template Structure

### 1. YAML Metadata Section (Required)

```yaml
---
spec_id: [SPEC_ID]-[VERSION]-[CHANGE_TYPE]
parent_spec: [PARENT_SPEC_ID]
implementation_status: [implemented|in_progress|ready|planned]
change_type: [enhancement|feature|bugfix|architectural|performance]
supersedes: [list of superseded specs]
implements: [REQ-XXXX-v2, REQ-YYYY-v2, ...]
created_at: YYYY-MM-DD
validated_at: YYYY-MM-DD
---
```

**Alternative Metadata Format** (structured section):

```markdown
## Metadata

- **Feature ID**: [FEAT-XXX]
- **Parent Spec**: [Parent specification reference]  
- **Version**: [X.Y.Z]
- **Status**: ✅ [COMPLETED|IN_PROGRESS|READY] - [Brief status description]
- **Created**: YYYY-MM-DD
- **Updated**: YYYY-MM-DD ([Update description])
- **Type**: Delta Specification ([Description of change])
- **ADR Dependencies**: [ADR-XXX, ADR-YYY, ADR-ZZZ]
- **Implementation Status**: ✅ [Percentage]% COMPLETE
- **Requirements Covered**: [REQ-XXXX, REQ-YYYY] ([completion status])
- **Test Coverage Target**: [XX]%+ for [target modules]
```

### 2. Executive Summary (Required)

```markdown
# Delta Specification: [Title]

## Change Summary

This delta specification documents the **[STATUS]** [brief description of changes]. The implementation successfully [key achievements with **VALIDATED** markers for completed items].

## Current State vs Target State

### Current State (from [PARENT-SPEC])
- **[Component A]**: ✅ [Status] - [Brief description]
- **[Component B]**: ⚠️ [Percentage]% - [Brief description] 
- **[Component C]**: ❌ Not implemented - [Brief description]

### Target State (after [THIS-SPEC])
- **[Component A]**: ✅ 100% - [Enhanced description with achievements]
- **[Component B]**: ✅ 100% - [Complete implementation description]
- **[Component C]**: ✅ 100% - [New implementation description]
```

### 3. Updated Requirements (Required)

```markdown
## Updated Requirements

### REQ-XXXX-v2: [Requirement Title]

- **Previous**: [Previous requirement or limitation]
- **Updated**: [New requirement with specific details]
- **✅ VALIDATED**: [Validation status and measurements]
- **Impact**:
  - [Specific impact on code/architecture]
  - [Performance improvements with metrics]
  - [Configuration changes needed]
  - [Integration implications]

### REQ-YYYY-v2: [Another Requirement]

[Follow same pattern for each requirement]
```

### 4. Technical Implementation Details (Required)

```markdown
## Technical Details

### [Component Name] Implementation

```[language]
# PRODUCTION CONFIGURATION - VALIDATED AND OPERATIONAL
class [ComponentName]:
    """[Description with validation status]."""
    
    # [Configuration with specific values - VALIDATED]
    [CONFIG_NAME] = "[value]"  # [Description - VALIDATED]
    
    def __init__(self):
        """Initialize [component] with [specific features]."""
        # [Implementation details with validation markers]
        pass
    
    async def [method_name](
        self,
        [parameters with types]
    ) -> [ReturnType]:
        """[Method description with validation status].
        
        Args:
            [parameter]: [Description with constraints]
            
        Returns:
            [Return description with performance metrics]
        """
        # [Implementation with performance validation]
        pass
```

### [Integration Points]

- **[System A]**: [Integration description with validation status]
- **[System B]**: [Integration description with performance metrics]
- **[System C]**: [Integration description with compliance status]
```

### 5. Acceptance Criteria (Required)

```markdown
## Acceptance Criteria

### Scenario 1: [Scenario Name]

```gherkin
Given [initial conditions with specific details]
When [action taken with parameters]
Then [expected outcome with measurable criteria]
And [additional validations with performance targets]
And [system state verification with metrics]
```

### Scenario 2: [Performance Validation]

```gherkin
Given [system configuration]
When [performance test executed]
Then [performance criteria] should be [specific measurement]
And [resource usage] should be [specific limits]
And [response time] should be [specific target]
```

### Scenario 3: [Integration Testing]

```gherkin
Given [integrated system state]
When [cross-component interaction occurs]
Then [behavior verification with success criteria]
And [data consistency validation]
And [error handling verification]
```
```

### 6. Implementation Phases (Required)

```markdown
## Implementation Plan

### Phase 1: [Foundation Phase] ([Duration])

1. [Specific task with deliverable]
2. [Configuration update with target]
3. [Integration point with validation criteria]

### Phase 2: [Enhancement Phase] ([Duration])

1. [Feature implementation with performance target]
2. [Integration testing with success criteria]
3. [Performance validation with metrics]

### Phase 3: [Optimization Phase] ([Duration])

1. [Performance tuning with specific improvements]
2. [Quality assurance with coverage targets]
3. [Documentation completion]

### Phase 4: [Validation Phase] ([Duration])

1. [End-to-end testing with scenarios]
2. [Performance benchmarking with targets]
3. [Production readiness assessment]
```

### 7. Validation and Testing (Required)

```markdown
## Tests

### Unit Tests (New/Enhanced)

- `test_[component]_[functionality]` - [Test description with validation]
- `test_[performance]_[criteria]` - [Performance test with targets]
- `test_[integration]_[scenario]` - [Integration test with success criteria]

### Integration Tests (New/Enhanced)

- `test_[end_to_end]_[workflow]` - [Workflow test with full validation]
- `test_[cross_component]_[interaction]` - [Component interaction test]
- `test_[performance]_[integration]` - [Performance integration test]

### Performance Tests (New/Enhanced)

- `test_[latency]_[requirements]` - [Latency validation with targets]
- `test_[throughput]_[requirements]` - [Throughput validation with metrics]
- `test_[resource]_[utilization]` - [Resource usage validation]

### Coverage Requirements

- New modules: [XX]%+ coverage
- Modified modules: Maintain existing coverage
- Overall [system]: Achieve [XX]%+ coverage
```

### 8. Dependencies and Traceability (Required)

```markdown
## Dependencies

### Technical Dependencies ([Change Type])

```toml
# [Description of dependency changes]
[dependency-name] = ">=[version]"  # [Reason for dependency]
[optional-dependency] = {version = ">=[version]", extras = ["[extras]"]}
```

### Infrastructure Dependencies ([ADR References])

- [Infrastructure requirement with ADR reference]
- [System requirement with validation criteria]
- [Hardware requirement with specifications]

### Feature Dependencies ([Integration Requirements])

- **[FEAT-XXX]**: [Dependency description with integration points]
- **[FEAT-YYY]**: [Dependency description with data flow]
- **[FEAT-ZZZ]**: [Dependency description with shared components]

## Traceability

### Parent Documents ([ADR Compliance Status])

- **[ADR-XXX]**: [ADR title] ([Implementation status with validation])
- **[ADR-YYY]**: [ADR title] ([Compliance verification with results])
- **[ADR-ZZZ]**: [ADR title] ([Integration status with performance])

### Related Specifications

- **[FEAT-XXX]**: [Brief relationship description]
- **[FEAT-YYY]**: [Integration dependency description]  
- **[FEAT-ZZZ]**: [Shared component description]

### Validation Criteria

- All [parent spec] requirements achieve [XX]% completion
- All ADR requirements fully implemented (no stubs)
- Test coverage reaches [XX]%+ target
- Performance gates met on [target hardware]
```

### 9. Success Metrics (Required)

```markdown
## Success Metrics

### Completion Metrics

- [REQ-XXXX]: [Component] [status] with [specific measurement]
- [REQ-YYYY]: [Feature] [status] with [performance metric]
- [REQ-ZZZZ]: [Integration] [status] with [validation criteria]
- Test Coverage: >[XX]% for [target modules]

### Performance Metrics

- [Performance aspect]: <[target] [units] ([validation status])
- [Efficiency metric]: [target range] ([measurement status])  
- [Quality metric]: >[target]% [criteria] ([achievement status])
- [Resource utilization]: <[limit] [units] ([compliance status])

### Quality Metrics

- [Implementation approach]: [percentage]% [improvement description]
- All existing functionality preserved and enhanced
- Full ADR compliance with [specific validation]
- Production-ready [status] with [quality indicators]
```

### 10. Risk Mitigation (Required)

```markdown
## Risk Mitigation

### Technical Risks ([Mitigation Status])

#### Risk: [Risk Description] [✅ RESOLVED | ⚠️ MONITORED | ❌ ACTIVE]

- **[✅ VALIDATED]**: [Risk resolution with validation]
- **Result**: [Outcome description with measurements]
- **Fallback**: [Fallback strategy if needed]

#### Risk: [Another Risk] [Status]

- **[Mitigation Status]**: [Mitigation description with results]
- **[Implementation Status]**: [Implementation description with validation]
- **Result**: [Outcome with success metrics]

### Mitigation Strategies

- [Strategy description with implementation status]
- [Validation approach with success criteria]
- [Monitoring approach with metrics]
- [Recovery procedures with tested scenarios]

## [Final Status Section] - [STATUS ACHIEVED]

**Status**: [Final implementation status with validation]

### Achievement Summary

- **[REQ-XXXX]**: ✅ [Achievement description] - `[implementation location]`
- **[REQ-YYYY]**: ✅ [Achievement description] - `[implementation location]`
- **[ADR-ZZZ]**: ✅ [Compliance description] - `[validation location]`

### Validation Results

- **Test Coverage**: [XX] comprehensive tests across all modules
- **ADR Compliance**: [XX]/100 score (Production Ready)
- **[Implementation Approach]**: [XX]% [improvement description]
- **Performance**: All targets achieved ([specific measurements])

**Document Status**: ✅ **SPECIFICATION COMPLETE** - All [SPEC-ID] requirements successfully implemented with production-ready quality.
```

## Template Usage Guidelines

### When to Use Delta Specification Format

1. **Complex multi-step implementations** requiring structured planning
2. **ADR compliance updates** with specific validation requirements  
3. **Performance improvements** with measurable targets and validation
4. **System enhancements** building on existing specifications
5. **Production readiness** documentation with comprehensive validation

### Quality Standards

1. **All code examples must be runnable** with proper syntax highlighting
2. **All performance targets must be measurable** with specific metrics
3. **All validation claims must be verifiable** with test evidence  
4. **All ADR references must be specific** with compliance verification
5. **All success criteria must be actionable** with clear measurement

### Validation Checklist

- [ ] **Metadata Complete**: All required fields populated with accurate information
- [ ] **Executive Summary**: Clear change description with validation status
- [ ] **Requirements Updated**: All requirements use REQ-XXXX-v2 format with validation
- [ ] **Technical Details**: Complete implementation with working code examples
- [ ] **Acceptance Criteria**: Gherkin scenarios cover all functionality with measurable outcomes
- [ ] **Implementation Plan**: Phased approach with realistic timelines and deliverables
- [ ] **Testing Coverage**: Comprehensive test strategy with coverage targets
- [ ] **Dependencies Clear**: All technical and ADR dependencies identified
- [ ] **Success Metrics**: Measurable completion and performance criteria
- [ ] **Risk Mitigation**: All technical risks addressed with validation

### Template Customization

This template should be customized for specific project contexts while maintaining the core structure. The key elements that make delta specifications effective are:

1. **Validation-Driven**: Every claim includes validation status
2. **Implementation-Ready**: Complete code examples and configuration  
3. **Performance-Focused**: Specific metrics and measurement criteria
4. **ADR-Compliant**: Full traceability to architectural decisions
5. **Quality-Assured**: Comprehensive testing and coverage requirements

Use this template to create specifications that serve as complete implementation guides rather than high-level descriptions.