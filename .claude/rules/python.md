# Python Code Standards

> GENERIC: This file can be reused across any project.

## Type Hints

All public functions must have complete type hints:

```python
def process_data(items: list[str], threshold: float = 0.5) -> dict[str, int]:
    """Process items and return counts."""
    ...
```

## Docstrings (NumPy Style)

```python
def function_name(param1: int, param2: str) -> bool:
    """Short description of function.

    Longer description if needed.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2.

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        When param1 is negative.

    Examples
    --------
    >>> function_name(1, "test")
    True
    """
```

## Imports Order

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import json
from pathlib import Path
from typing import TYPE_CHECKING

# 3. Third-party
import numpy as np
from pydantic import BaseModel

# 4. Local imports
from my_project.core import utils

# 5. Type-checking only imports
if TYPE_CHECKING:
    from my_project.models import MyModel
```

## Error Handling

```python
# Specific exceptions with context
raise ValueError(f"Invalid input: expected positive, got {value}")

# Never bare except
try:
    result = risky_operation()
except SpecificError as e:
    logger.error("Operation failed", error=str(e))
    raise
```

## Validation

Use Pydantic for data validation:

```python
from pydantic import BaseModel, Field

class Config(BaseModel):
    name: str = Field(..., min_length=1)
    value: float = Field(..., ge=0)
```

## Boundaries

### Always Do
- Use type hints on all public functions
- Write NumPy-style docstrings
- Use Pydantic for external data
- Use pathlib for file paths

### Never Do
- Use bare `except:`
- Use mutable default arguments
- Ignore type checker errors with `# type: ignore` without explanation
