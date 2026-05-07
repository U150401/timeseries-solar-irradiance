# Testing Standards

> GENERIC: This file can be reused across any project.

## Test File Structure

```python
"""Tests for module_name."""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from my_project.module import function_to_test


class TestFunctionName:
    """Tests for function_name."""

    def test_happy_path(self) -> None:
        """Test normal operation."""
        result = function_to_test(valid_input)
        assert result == expected

    def test_edge_case_empty(self) -> None:
        """Test with empty input."""
        result = function_to_test("")
        assert result == empty_expected

    def test_raises_on_invalid(self) -> None:
        """Test error handling."""
        with pytest.raises(ValueError, match="specific message"):
            function_to_test(invalid_input)

    @given(st.text())
    def test_property_never_crashes(self, text: str) -> None:
        """Property: handles any input without crashing."""
        function_to_test(text)  # Should not raise
```

## Fixtures

```python
@pytest.fixture
def sample_data() -> dict:
    """Create sample test data."""
    return {"key": "value"}

@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create temporary config file."""
    config = tmp_path / "config.json"
    config.write_text('{"setting": true}')
    return config
```

## Test Coverage

Minimum 80% coverage. Test:
- Happy paths
- Edge cases (empty, None, boundary values)
- Error conditions
- Property-based tests for complex logic

## Boundaries

### Always Do
- Write tests before implementation (TDD)
- Use descriptive test names
- Test one thing per test
- Use fixtures for shared setup

### Never Do
- Test implementation details
- Skip edge cases
- Use `time.sleep()` in tests
- Leave flaky tests unfixed
