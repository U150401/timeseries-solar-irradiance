---
name: systematic-debugging
description: Debug issues systematically with reproducing tests. Never debug without a test that captures the bug.
---

# Systematic Debugging

## The Iron Law

**Never fix a bug without a test that reproduces it.**

If you can't write a failing test, you don't understand the bug.

## The Debugging Workflow

```
┌──────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  1. REPRODUCE │ ──► │  2. ISOLATE  │ ──► │  3. FIX     │ ──► │  4. VERIFY   │
│  Write test   │     │  Narrow scope│     │  Minimal    │     │  All tests   │
└──────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

### Step 1: REPRODUCE - Write Failing Test

```python
def test_bug_issue_123(self) -> None:
    """Reproduce bug #123: description."""
    with pytest.raises(ValueError, match="expected"):
        buggy_function(trigger_input)
```

**Verify it fails for the RIGHT reason.**

### Step 2: ISOLATE - Narrow Scope

- Simplify inputs to minimal reproduction
- Add debug output
- Binary search to find failing section

### Step 3: FIX - Minimal Change

Make the **smallest possible fix**. Do NOT:
- Refactor unrelated code
- Add extra features
- "Clean up while you're there"

### Step 4: VERIFY - All Tests Pass

```bash
# Bug test passes
uv run pytest tests/test_x.py::test_bug_issue_123 -v

# ALL tests pass
uv run pytest tests/ -v

# Linting clean
uv run ruff check src/ && uv run mypy src/
```

## Debugging Checklist

- [ ] Test reproduces the bug
- [ ] Test failed before fix
- [ ] Test passes after fix
- [ ] All other tests pass
- [ ] Commit references issue: `fix(module): description (#123)`

## Anti-Patterns

| Anti-Pattern | Why It's Wrong |
|--------------|----------------|
| "I'll just fix it and see" | You won't know if it's really fixed |
| "The test is too hard to write" | Then you don't understand the bug |
| "It works when I try manually" | Manual testing isn't repeatable |

---

Debug $ARGUMENTS using this systematic approach. Start with a failing test.
