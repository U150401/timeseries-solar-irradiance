---
applyTo: "**/*.py"
---

# Python Code Standards

## Type Hints

All public functions must have type hints:

```python
def calculate_value(input_data: np.ndarray, scale: float = 1.0) -> float:
    """Calculate scaled value from input data."""
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

## Imports

```python
# Standard library
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party
import numpy as np
from pydantic import BaseModel

# Local
from my_project.core import utils

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
