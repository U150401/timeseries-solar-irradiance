---
applyTo: "tests/**/*.py"
---

# Testing Standards

## Test Structure

```python
"""Tests for module_name."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from my_project.module import function_to_test


class TestFunctionName:
    """Tests for function_name."""

    def test_basic_case(self) -> None:
        """Test basic functionality."""
        result = function_to_test(1, 2)
        assert result == 3

    def test_edge_case(self) -> None:
        """Test edge case with empty input."""
        result = function_to_test(0, 0)
        assert result == 0

    def test_error_handling(self) -> None:
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            function_to_test(-1, 2)
```

## Fixtures

```python
@pytest.fixture
def sample_data() -> dict:
    """Create sample data for testing."""
    return {"key": "value", "number": 42}

@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """Create a temporary file for testing."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    return file_path
```

## Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=1000))
def test_function_with_any_positive_int(value: int) -> None:
    """Test function works with any positive integer."""
    result = function_to_test(value)
    assert result >= 0
```

## Parametrized Tests

```python
@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        (1, 2),
        (2, 4),
        (0, 0),
    ],
)
def test_double(input_val: int, expected: int) -> None:
    """Test doubling function with various inputs."""
    assert double(input_val) == expected
```
