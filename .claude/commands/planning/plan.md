---
description: Create implementation plan for a task, following extracted conventions
model: opus
---

# Plan Implementation

You are tasked with creating detailed, actionable implementation plans that follow the project's established conventions and patterns.

## Output Location

Plans are stored in `.claude/plans/`:
```
.claude/plans/
├── YYYY-MM-DD-short-description.md
└── archive/                          # Completed plans
```

## Initial Response

When invoked:

1. **If task description provided** (e.g., `/plan add user authentication`):
   - Acknowledge the task
   - Begin research immediately

2. **If ticket reference provided** (e.g., `/plan PROJ-1234`):
   - Ask user to paste ticket content

3. **If no parameters**:
   ```
   What would you like to plan? You can:
   - Describe the task: `/plan add caching layer to API`
   - Reference a ticket: `/plan PROJ-1234`
   - Paste requirements directly
   ```

## Planning Process

### Phase 1: Context Gathering

1. **Load project conventions** (if they exist):
   ```
   Read .claude/docs/conventions.md
   Read .claude/docs/patterns.md
   Read .claude/docs/architecture.md
   Read .claude/docs/dependencies.md
   ```

   If `.claude/docs/` doesn't exist, suggest running `/extract_conventions` first, but continue anyway.

2. **Understand the task**:
   - Parse requirements from task description or ticket
   - Identify affected components/modules
   - List explicit and implicit requirements

3. **Research codebase** (parallel sub-agents):

   **Agent 1: Find similar implementations**
   ```
   Search for similar features or patterns already implemented.
   Find code that can serve as templates.
   Identify relevant test examples.
   ```

   **Agent 2: Map affected files**
   ```
   Identify all files that will need changes.
   Find integration points.
   Locate relevant configuration.
   ```

   **Agent 3: Check constraints**
   ```
   Find related tests that must pass.
   Identify API contracts that must be maintained.
   Check for deprecation warnings or TODOs.
   ```

### Phase 2: Plan Design

1. **Break down into phases**:
   - Each phase should be independently testable
   - Order by dependencies (what must exist first)
   - Keep phases small (1-2 hours of work max)

2. **For each phase, specify**:
   - Goal (one sentence)
   - Files to create/modify (with line numbers if modifying)
   - Code approach (reference existing patterns)
   - Validation steps

3. **Identify risks**:
   - Breaking changes
   - Performance implications
   - Security considerations
   - Migration needs

### Phase 3: Write Plan

Create `.claude/plans/YYYY-MM-DD-short-description.md`:

```markdown
# Plan: [Short Title]

> Created: YYYY-MM-DD
> Task: [Original task description or ticket reference]
> Status: Draft | Approved | In Progress | Complete

## Summary
[2-3 sentence overview of what this plan accomplishes]

## Requirements
- [ ] [Explicit requirement 1]
- [ ] [Explicit requirement 2]
- [ ] [Implicit requirement - inferred from context]

## Affected Components
| Component | Change Type | Risk |
|-----------|-------------|------|
| `src/api/users.ts` | Modify | Low |
| `src/services/auth.ts` | Create | Medium |
| ... | ... | ... |

## Dependencies
- Requires: [other systems, libraries, or completed work]
- Blocks: [what depends on this being done]

---

## Phase 1: [Name]

**Goal**: [One sentence]

**Changes**:

### `path/to/file.ts`
```typescript
// At line X, add:
[code snippet following project patterns]
```

### `path/to/another.ts`
[Similar structure]

**Validation**:
- [ ] Unit test: `npm test -- --grep "feature"`
- [ ] Manual: [specific check to perform]

---

## Phase 2: [Name]

[Same structure]

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| [Risk 1] | High/Med/Low | [How to address] |

## Rollback Plan
[How to undo if something goes wrong]

## Success Criteria
- [ ] [Measurable outcome 1]
- [ ] [Measurable outcome 2]
- [ ] All existing tests pass
- [ ] New tests cover added functionality

---

## References
- Pattern source: `src/existing/similar.ts`
- Convention: `.claude/docs/patterns.md#section`
- Ticket: [PROJ-XXXX](link)
```

### Phase 4: Review & Iterate

After writing the plan:

1. **Self-review checklist**:
   - Does each phase have clear validation?
   - Are code snippets following documented conventions?
   - Are all affected files identified?
   - Is the order logical (dependencies first)?

2. **Present to user**:
   ```
   ## Plan Ready: [Title]

   **Scope**: X phases, ~Y files affected

   **Phases**:
   1. [Phase 1 name] - [brief description]
   2. [Phase 2 name] - [brief description]
   ...

   **Key Risks**: [Top 1-2 risks]

   Plan saved to: `.claude/plans/YYYY-MM-DD-description.md`

   Would you like me to:
   - Walk through any phase in detail?
   - Adjust scope or approach?
   - Start implementation?
   ```

3. **Handle feedback**:
   - Update plan based on user input
   - Re-validate after changes
   - Keep plan file updated

## Implementation Handoff

When user approves and wants to implement:

```
Starting implementation of: [Plan Title]

I'll work through each phase, validating as I go.
Progress will be tracked in the plan file with [x] markers.

Beginning Phase 1...
```

Track progress by updating the plan file:
- Mark completed items with `[x]`
- Add notes about deviations
- Update timestamps

## Tips

- **For large tasks**: Suggest breaking into multiple plans
- **For unclear requirements**: Ask clarifying questions before planning
- **For risky changes**: Emphasize rollback plan and testing
- **For cross-repo changes**: Create linked plans in each repo
