---
name: tdd
description: Test-driven development with strict RED-GREEN-REFACTOR enforcement. Non-negotiable for features and bug fixes.
---

# Test-Driven Development

## The Iron Law

**Write code before the test? Delete it. Start over.**

Not "keep as reference." Not "adapt slightly." Delete it.

This is non-negotiable. No exceptions without explicit human approval.

## When TDD Applies

| Situation | TDD Required? |
|-----------|---------------|
| New feature | **Yes** |
| Bug fix | **Yes** - write reproducing test first |
| Refactoring | **Yes** - ensure tests exist first |
| Config files | Ask first |
| Prototype/spike | Ask first, then delete and redo properly |

## The RED-GREEN-REFACTOR Cycle

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   ┌─────┐      ┌───────┐      ┌──────────┐        │
│   │ RED │ ───► │ GREEN │ ───► │ REFACTOR │ ──┐    │
│   └─────┘      └───────┘      └──────────┘   │    │
│      ▲                                        │    │
│      └────────────────────────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1. RED: Write Failing Test

```python
def test_feature_does_something(self) -> None:
    """Test that feature does X when given Y."""
    result = feature(input_value)
    assert result == expected_value
```

**Then verify it fails:** `uv run pytest tests/test_module.py -v`

**MANDATORY**: Confirm the test fails for the RIGHT reason.

### 2. GREEN: Minimal Implementation

Write the **smallest possible code** to make the test pass.
- No extra features
- No premature optimization
- Ugly is fine. Green is the goal.

**Verify it passes:** `uv run pytest tests/test_module.py -v`

### 3. REFACTOR: Clean Up

Now and ONLY now, improve the code. After EVERY change, verify tests still pass.

## Verification is Mandatory

See `.claude/rules/verification.md` for details.

## Common Rationalizations (All Wrong)

| Rationalization | Why It's Wrong |
|-----------------|----------------|
| "I'll add tests after" | You won't. And the design will be worse. |
| "It's just a small change" | Small changes cause big bugs. |
| "I know this works" | Prove it. Write the test. |

## Bug Fix Workflow

1. **Write a test that reproduces the bug** (must fail)
2. **Verify it fails** for the right reason
3. **Fix the bug** with minimal code
4. **Verify the test passes**
5. **Verify all other tests still pass**

Never fix a bug without a reproducing test. Period.

---

Implement $ARGUMENTS using TDD. Start with the failing test.
