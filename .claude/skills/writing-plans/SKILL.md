---
name: writing-plans
description: Create detailed implementation plans with bite-sized tasks before coding. Use for complex features requiring multiple steps.
---

# Writing Implementation Plans

## When to Write a Plan

Write a plan when:
- Feature requires 3+ files to change
- Implementation path isn't immediately obvious
- Multiple components need coordination

## Plan Structure

### Header

```markdown
# Plan: [Feature Name]

**Goal**: One sentence describing what success looks like.

**Architecture**: 2-3 sentences on the approach.

**Estimated Tasks**: X tasks, ~Y minutes each
```

### Task Breakdown

Break into **bite-sized tasks (2-5 minutes each)**:

```markdown
## Task 1: [Descriptive Name]

**Files**:
- Create: `src/module/new_file.py`
- Modify: `src/module/existing.py` (lines 45-60)
- Test: `tests/test_module.py`

**Steps**:
1. Write failing test for [specific behavior]
2. Run test, verify it fails for the right reason
3. Implement [specific function] in [specific file]
4. Run test, verify it passes
5. Commit: `feat(module): add [feature]`

**Expected Output**:
```bash
$ uv run pytest tests/test_module.py::test_new_feature -v
PASSED
```
```

## Task Requirements

Every task MUST include:

| Element | Required |
|---------|----------|
| Exact file paths | Yes |
| Line numbers (for edits) | Yes |
| Complete code | Yes (not "add validation") |
| Test command | Yes |
| Expected output | Yes |
| Commit message | Yes |

## Precision Standards

### Wrong (Vague)

```markdown
## Task 1: Add validation
Add input validation to the parser.
```

### Right (Precise)

```markdown
## Task 1: Add altitude validation

**Files**: `src/core/state.py` (lines 23-35), `tests/test_state.py`

**Steps**:
1. Write test `test_altitude_rejects_negative`
2. Add validation: `if altitude < 0: raise ValueError(...)`
3. Commit: `feat(core): add altitude validation`
```

---

Create a detailed implementation plan for: $ARGUMENTS
