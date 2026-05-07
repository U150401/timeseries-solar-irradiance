# Modern Python Libraries

> GENERIC: This file can be reused across any Python project.

## Core Principle

**ALWAYS use existing, well-tested libraries instead of custom implementations.**

## Preferred Libraries (2025+)

### Data & Validation
| Use | Not | Why |
|-----|-----|-----|
| **Pydantic** | dataclasses | Validation, serialization, settings |
| **Polars** | pandas | 10-100x faster, better memory |
| **msgspec** | json/pickle | Fast serialization |

### Web & HTTP
| Use | Not | Why |
|-----|-----|-----|
| **httpx** | requests | Async, HTTP/2, modern |
| **aiofiles** | open() | Async file I/O |

### CLI & Config
| Use | Not | Why |
|-----|-----|-----|
| **Cyclopts** | argparse/click | Type hints to CLI |
| **Rich** | print | Beautiful output |
| **structlog** | logging | Structured logs |

### Testing
| Use | Not | Why |
|-----|-----|-----|
| **Hypothesis** | manual edge cases | Property-based |
| **Faker** | manual test data | Realistic fakes |
| **respx** | responses | Mock httpx |

### Utilities
| Use | Not | Why |
|-----|-----|-----|
| **pathlib** | os.path | OOP paths (stdlib) |
| **tenacity** | manual retries | Retry with backoff |

## Never Do

- Write custom validation → **Pydantic**
- Write manual retries → **tenacity**
- Write CLI parsing → **Cyclopts**
- Write HTTP client → **httpx**
- Use `os.path` → **pathlib**
