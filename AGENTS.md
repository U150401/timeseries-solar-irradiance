# AI Agent Instructions

<!--
This file contains PROJECT-SPECIFIC instructions for AI agents.
Customize this for YOUR project's domain, architecture, and goals.

Generic rules (Python, testing, git) are in .claude/rules/
Generic skills (TDD, planning, debugging) are in .claude/skills/
-->

## Project Overview

**my-project**: A short description of your project.

<!-- Example:
**AetherX**: A modular, differentiable aircraft performance modeling framework
supporting BADA3, BADA4, Poll-Schumann, and OpenAP. Designed for trajectory
optimization using CasADi symbolic computation.
-->

## Goals

<!-- What should this project achieve? Be specific. -->

- Goal 1: ...
- Goal 2: ...
- Goal 3: ...

<!-- Example:
- Unified ICAO interface for all aircraft models
- Full CasADi compatibility for symbolic differentiation
- Accurate fuel flow, thrust, and drag calculations
-->

## Architecture

```
src/my_project/
├── __init__.py      # Package exports
├── core/            # Core abstractions and interfaces
├── models/          # Data models (Pydantic)
└── utils/           # Utilities and helpers
```

<!--
Describe your architecture. What are the main components?
How do they interact? What are the key abstractions?

Example:
src/aetherx/
├── core/           # Abstract interfaces (AircraftPerformanceModel)
├── models/
│   ├── bada3/      # BADA3 implementation
│   ├── bada4/      # BADA4 implementation
│   ├── openap/     # OpenAP wrapper
│   └── ps/         # Poll-Schumann model
├── physics/        # ISA, aero calculations
└── numpy/          # CasADi-compatible NumPy ops
-->

## Domain Knowledge

<!--
Add domain-specific knowledge that AI agents need to understand.
This is the most important section for project-specific context.

Include:
- Key concepts and terminology
- Important equations or algorithms
- External dependencies and their purpose
- Data formats and sources
-->

### Key Concepts

<!-- Example:
### Total Energy Model (TEM)

The fundamental equation governing aircraft performance:

```
(T - D) · V = mg · dh/dt + mV · dV/dt
```

Where:
- T = Thrust [N]
- D = Drag [N]
- V = True airspeed [m/s]
- m = Aircraft mass [kg]
- h = Altitude [m]
-->

### External Dependencies

<!-- Example:
### CasADi Compatibility

ALL physics functions must work with both:
- `numpy.ndarray` (for simulation)
- `casadi.SX/MX` (for optimization)

Use `aetherx.numpy` instead of `numpy` directly:

```python
import aetherx.numpy as anp  # CasADi-compatible
result = anp.sqrt(x)  # Works with both NumPy and CasADi
```
-->

## Key Patterns

<!-- Show code patterns that should be followed in this project -->

### Pattern 1: Data Validation

```python
from pydantic import BaseModel, Field

class Config(BaseModel):
    """Configuration with validation."""
    name: str = Field(..., min_length=1)
    value: float = Field(..., ge=0)
```

### Pattern 2: Error Handling

```python
# Specific exceptions with context
raise ValueError(f"Invalid input: expected positive, got {value}")
```

<!-- Example project-specific pattern:
### Pattern: Model Initialization

All models use ICAO aircraft type codes:

```python
from aetherx.models.ps import PSModel

model = PSModel("A320")  # Initialize with ICAO code
print(model.mtom)        # Access aircraft properties
```
-->

## Boundaries

### Always Do

- Write tests first (TDD)
- Use type hints on all public functions
- Validate external inputs with Pydantic
- Follow conventional commits

<!-- Add project-specific requirements:
- Ensure CasADi compatibility for all physics functions
- Use aetherx.numpy instead of raw numpy
-->

### Ask First

- Add new dependencies
- Change public APIs
- Modify CI/CD workflows
- Refactor core abstractions

### Never Do

- Skip tests
- Use bare `except:`
- Commit secrets or credentials
- Push directly to main

<!-- Add project-specific prohibitions:
- Use raw numpy in physics calculations (breaks CasADi)
- Hardcode aircraft parameters (use data files)
- Break symbolic differentiation
-->

## References

<!-- Link to relevant documentation, papers, or resources -->

- [Project Documentation](docs/)
- [WORKFLOW.md](WORKFLOW.md) - Development workflow

<!-- Example:
- [BADA User Manual](https://www.eurocontrol.int/model/bada)
- [CasADi Documentation](https://web.casadi.org/docs/)
- [Poll-Schumann Paper](https://doi.org/...)
-->
