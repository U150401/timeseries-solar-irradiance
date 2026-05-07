# GitHub Copilot Instructions

## Project Overview

**my-project**: A short description of your project.

## Key References

- **Project-Specific**: See [AGENTS.md](../AGENTS.md) for domain knowledge and patterns
- **Workflow**: See [WORKFLOW.md](../WORKFLOW.md) for complete project workflow
- **File-Specific**: See `.github/instructions/` for file-scoped rules

## Code Standards

### Python Style

- Python 3.11+
- NumPy-style docstrings
- Type hints on all public functions
- Pydantic for data validation

### Testing

- pytest with fixtures
- Test files: `tests/test_*.py`
- Minimum 80% coverage
- Use Hypothesis for property-based testing

### Git

- Conventional commits: `type(scope): description`
- Types: feat, fix, docs, test, refactor, perf, chore

## Quick Commands

```bash
make check      # Run all checks (lint + test)
make test       # Run tests only
make lint       # Run linting only
make format     # Auto-format code
```

## File-Specific Instructions

Copilot uses instruction files from `.github/instructions/`:

| File | Applies To |
|------|------------|
| `python.instructions.md` | `*.py` files |
| `testing.instructions.md` | `tests/**/*.py` files |
| `academic-writing.instructions.md` | `paper/**/*.md`, `presentation/**/*.md` |
