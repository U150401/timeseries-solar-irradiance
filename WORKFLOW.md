# Project Workflow

Complete guide from setup to publication.

## Phase 1: Setup

```bash
# Clone and install
git clone https://github.com/your-username/my-project.git
cd my-project
uv sync --all-groups
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Verify
make check
```

## Phase 2: Development (TDD)

### For New Features

1. **Plan** (if complex): `/writing-plans "feature description"`
2. **Write failing test** first
3. **Implement** minimal code
4. **Verify** tests pass
5. **Refactor** (keep tests green)
6. **Commit**: `feat(scope): description`

### For Bug Fixes

1. **Write test** that reproduces bug
2. **Verify** it fails
3. **Fix** with minimal change
4. **Verify** all tests pass
5. **Commit**: `fix(scope): description (#issue)`

### Quick Commands

```bash
make test       # Run tests
make lint       # Run linting
make check      # Run all checks
make format     # Auto-format
```

## Phase 3: Experiments

```bash
# Run experiments
uv run python experiments/scripts/run_experiment.py

# Analyze results
uv run python experiments/scripts/analyze_results.py

# Generate figures
uv run python experiments/scripts/generate_figures.py --output paper/figures/
```

## Phase 4: Academic Writing

### Paper

```bash
# Research literature
/literature-review "your topic"

# Write paper
/scientific-writing "Write paper about X using data from experiments/results/"

# Remove AI patterns
/humanizer paper/paper.md

# Build PDF
make paper
```

### Presentation

```bash
# Generate slides
/scientific-slides "Create 20-min presentation from paper/paper.md"

# Build PDF
make slides
```

## Phase 5: Release

```bash
# Create release
make release VERSION=1.0.0

# Or manually
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

## Before Going Public

Hide private files from git:

```bash
make hide-private   # Hide AI config + academic files
git commit -m "chore: prepare for public release"
```

To restore later:

```bash
make show-private   # Restore tracking
```
