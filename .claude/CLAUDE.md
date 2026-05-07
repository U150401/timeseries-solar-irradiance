# Claude Code Instructions

See [AGENTS.md](../AGENTS.md) for project-specific instructions.
See [WORKFLOW.md](../WORKFLOW.md) for complete project workflow.

## Quick Commands

```bash
uv run pytest tests/ -v              # Run tests
uv run ruff check . && uv run mypy src/  # Quality check
make check                           # Full check (lint + test)
```

## Memory Structure

| File | Purpose |
|------|---------|
| `AGENTS.md` | Project-specific patterns |
| `.claude/rules/` | Generic Python standards (reusable) |
| `WORKFLOW.md` | Complete project workflow |

## Available Rules

| Rule | Purpose |
|------|---------|
| `python.md` | Python code standards (type hints, NumPy docstrings) |
| `testing.md` | Testing standards (pytest, Hypothesis) |
| `git.md` | Git conventions (conventional commits) |
| `libraries.md` | Modern library choices (Pydantic, httpx, etc.) |
| `verification.md` | **Never claim success without proof** |

## External Skills Setup

For academic writing features, install these skills (one-time):

```bash
# Create skills directory
mkdir -p ~/.claude/skills

# Scientific writing (papers, citations, literature)
git clone https://github.com/K-Dense-AI/claude-scientific-writer.git ~/.claude/skills/claude-scientific-writer

# Link individual skills
cd ~/.claude/skills
for skill in claude-scientific-writer/skills/*/; do
  ln -sf "claude-scientific-writer/skills/$(basename $skill)" "$(basename $skill)"
done

# Text humanization
git clone https://github.com/blader/humanizer.git ~/.claude/skills/humanizer
```

## Available Skills

### Project Skills (in `.claude/skills/`)

| Skill | Purpose |
|-------|---------|
| `/tdd` | Test-driven development (strict RED-GREEN-REFACTOR) |
| `/writing-plans` | Create detailed implementation plans |
| `/systematic-debugging` | Debug with reproducing tests |

### External Skills (after setup above)

| Skill | Purpose |
|-------|---------|
| `/scientific-writing` | Write papers (IMRaD) |
| `/scientific-slides` | Create presentations |
| `/literature-review` | Research papers |
| `/citation-management` | BibTeX references |
| `/peer-review` | Get feedback |
| `/humanizer` | Remove AI patterns |

## Claude Code Tips

- Use `/init` to bootstrap new projects
- Use `#` key to add notes to CLAUDE.md
- Use `/memory` to edit memory files
- Use `/skill-name` to invoke skills
