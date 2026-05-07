# Verification Before Completion

> GENERIC: This file can be reused across any project.

## The Iron Law

**Never claim success without empirical proof.**

Evidence before claims, always. No exceptions.

## Required Verification Steps

Before ANY success claim, complete ALL five gates:

1. **Identify** - What command proves your assertion?
2. **Execute** - Run it freshly (not from memory or cache)
3. **Read** - Check full output AND exit status
4. **Verify** - Does output actually support your claim?
5. **Report** - State claim WITH the evidence

Skip any step = violation.

## Red Flags (Immediate Stop)

| Red Flag | Why It's Wrong |
|----------|----------------|
| "should work" | Assumption, not evidence |
| "probably fixed" | Hedging = uncertainty = unverified |
| "seems good" | Feeling, not proof |
| "Perfect!" before running tests | Premature celebration |
| "Done!" without showing output | Claim without evidence |
| Trusting previous test runs | Stale evidence |

## What Counts as Verification

| Claim | Required Verification |
|-------|----------------------|
| "Tests pass" | Show pytest output with exit code 0 |
| "Linting clean" | Show ruff/mypy output |
| "Build succeeds" | Show build command output |
| "Bug fixed" | Show test that reproduces bug now passes |
| "Feature works" | Show test or demo output |

## Partial Proves Nothing

- 5/10 tests passing ≠ "mostly works"
- Linter clean ≠ tests pass
- Tests pass ≠ type checker clean

Each check proves only what it checks. Nothing more.

## The Cost of Skipping

Verification takes seconds. Recovery takes hours.
