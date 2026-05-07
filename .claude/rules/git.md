# Git Standards

> GENERIC: This file can be reused across any project.

## Conventional Commits

```
type(scope): description

feat(api): add user authentication endpoint
fix(parser): handle empty input correctly
docs(readme): update installation instructions
test(auth): add integration tests
refactor(core): simplify processing logic
perf(db): optimize query performance
chore(deps): update dependencies

# Breaking change
feat(api)!: change response format

# With body
fix(auth): prevent session fixation

The session ID is now regenerated after login.

Closes #123
```

## Commit Types

| Type | When | Changelog Section |
|------|------|-------------------|
| `feat` | New feature | Features |
| `fix` | Bug fix | Bug Fixes |
| `docs` | Documentation only | Documentation |
| `test` | Add/update tests | Testing |
| `refactor` | Code change (no feature/fix) | Refactoring |
| `perf` | Performance improvement | Performance |
| `chore` | Maintenance | Miscellaneous |
| `ci` | CI/CD changes | CI/CD |

## Boundaries

### Always Do
- Use conventional commit format
- Write clear, descriptive messages
- Reference issues when relevant

### Ask First
- Force push to shared branches
- Rebase published history
- Delete remote branches

### Never Do
- Push to main/master directly
- Commit secrets or credentials
- Use generic messages ("fix", "update", "wip")
