# Contributing to BehaviorCI

First off, thank you for considering contributing to BehaviorCI! It's people like you that make this tool better for everyone shipping LLM applications to production.

This document provides guidelines for contributing to the project. Following these guidelines helps maintainers review and merge your contributions efficiently.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)
- [Questions?](#questions)

---

## Code of Conduct

This project and everyone participating in it is governed by our commitment to:

- **Be respectful**: Treat everyone with respect. Healthy debate is encouraged, but harassment is not tolerated.
- **Be constructive**: Criticize ideas, not people. Provide actionable feedback.
- **Be inclusive**: Welcome newcomers and help them learn.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Fork and Clone

1. **Fork the repository** on GitHub:
   - Visit https://github.com/0-uddeshya-0/BehaviorCI
   - Click the "Fork" button

2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BehaviorCI.git
   cd BehaviorCI
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/0-uddeshya-0/BehaviorCI.git
   ```

---

## Development Environment

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

### Pre-commit Hooks (Recommended)

We use pre-commit to ensure code quality:

```bash
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Our pre-commit hooks include:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

---

## How to Contribute

### Reporting Bugs

Before creating a bug report, please:

1. **Check existing issues** to avoid duplicates
2. **Update to the latest version** to verify the bug still exists

When reporting a bug, include:

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Install BehaviorCI version X.Y.Z
2. Run command '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened. Include full error messages and stack traces.

**Environment**
- OS: [e.g. macOS 14, Ubuntu 22.04]
- Python version: [e.g. 3.11.4]
- BehaviorCI version: [e.g. 0.1.0]
- pytest version: [e.g. 7.4.0]

**Additional Context**
Any other context about the problem (e.g., CI system, parallel execution settings).
```

**File bug reports at**: https://github.com/0-uddeshya-0/BehaviorCI/issues/new?template=bug_report.md

### Suggesting Features

We welcome feature suggestions! Before proposing:

1. **Check if the feature aligns** with our core philosophy: local-first, pytest-native, regression testing
2. **Search existing issues** for similar requests

Feature request template:

```markdown
**Feature Description**
What feature would you like to see?

**Problem It Solves**
What problem does this feature solve? Be specific.

**Proposed Solution**
How should this feature work? Include API examples if relevant.

**Alternatives Considered**
What alternative solutions have you considered?

**Additional Context**
Any other context, mockups, or examples.
```

**File feature requests at**: https://github.com/0-uddeshya-0/BehaviorCI/issues/new?template=feature_request.md

### Pull Requests

#### Before You Start

1. **Open an issue first** for significant changes to discuss approach
2. **For bug fixes**: Reference the issue number in your PR
3. **For features**: Ensure the feature is approved before investing time

#### PR Process

1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our [Coding Standards](#coding-standards)

3. **Add tests** for new functionality (see [Testing](#testing))

4. **Update documentation** if needed (README, docstrings, comments)

5. **Run the full test suite**:
   ```bash
   pytest tests/ -v
   pytest tests/ -n 4  # Parallel execution test
   mypy src/behaviorci/
   black src/ tests/ --check
   ```

6. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add support for custom embedding models"
   
   # or for fixes:
   git commit -m "fix: handle WAL file cleanup on Windows"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `chore:` Build/tooling changes

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request** on GitHub:
   - Fill out the PR template completely
   - Link related issues with `Fixes #123` or `Relates to #456`
   - Request review from maintainers

#### PR Review Criteria

Your PR will be reviewed for:

- ✅ **Correctness**: Does it solve the stated problem?
- ✅ **Tests**: Are there adequate tests? Do they pass?
- ✅ **Code Quality**: Follows style guidelines, clean code principles
- ✅ **Documentation**: Docstrings, comments, README updates if needed
- ✅ **Performance**: No significant regressions
- ✅ **Backwards Compatibility**: Breaking changes must be justified

---

## Development Workflow

### Branch Naming

```
feature/description    # New features
fix/description        # Bug fixes
docs/description       # Documentation
refactor/description   # Code refactoring
test/description       # Test improvements
```

### Keeping Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Rebasing

If your branch is behind `main`:

```bash
git checkout feature/your-feature
git rebase main
# Resolve any conflicts
git push origin feature/your-feature --force-with-lease
```

---

## Coding Standards

### Python Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with these specifics:

#### Formatting

- **Black** for code formatting (line length: 100)
- **isort** for import sorting (profile: black)
- Run `black src/ tests/` before committing

#### Type Hints

- All public functions must have type hints
- Use `Optional[Type]` for nullable parameters
- Use `from __future__ import annotations` for Python 3.8 compatibility

```python
from typing import Optional, List, Dict
from pathlib import Path

def save_snapshot(
    behavior_id: str,
    input_json: str,
    output_text: str,
    embedding: np.ndarray,
    model_name: str,
    git_commit: Optional[str] = None
) -> str:
    """Save a behavioral snapshot.
    
    Args:
        behavior_id: Logical behavior identifier
        input_json: JSON-serialized input arguments
        output_text: LLM output text
        embedding: Normalized embedding vector
        model_name: Name of embedding model used
        git_commit: Optional git commit hash
        
    Returns:
        snapshot_id: The computed snapshot ID
        
    Raises:
        StorageError: If database operation fails
    """
```

#### Docstrings

- **Google style** docstrings
- All public classes and methods must be documented
- Include `Args`, `Returns`, `Raises` sections

#### Comments

- Explain **why**, not **what** (code should be self-documenting)
- For bug fixes, include bug reference:
  ```python
  # BUG-003: Use singleton to prevent connection exhaustion
  # with 1000+ tests
  ```

### Code Organization

```
src/behaviorci/
├── __init__.py          # Public API exports
├── api.py               # @behavior decorator
├── storage.py           # Database operations
├── embedder.py          # Embedding computation
├── comparator.py        # Comparison logic
├── plugin.py            # pytest integration
├── cli.py               # Command-line interface
├── models.py            # Pydantic models
└── exceptions.py        # Error classes
```

### Error Handling

- Use custom exception hierarchy from `exceptions.py`
- Provide actionable error messages
- Never silently swallow exceptions

```python
from .exceptions import StorageError, SerializationError

try:
    result = json.dumps(data)
except TypeError as e:
    raise SerializationError(type(e.obj).__name__, e) from e
```

---

## Testing

### Test Structure

```
tests/
├── test_behaviorci.py          # Core functionality
├── test_bug_001_*.py           # Regression tests for specific bugs
├── test_fix_004_*.py           # Tests for specific fixes
├── examples/
│   ├── fake_llm.py             # Mock LLM for testing
│   ├── mock_embedder.py        # Fast mock embedder
│   ├── conftest.py             # pytest fixtures
│   ├── test_app.py             # Example tests
│   └── test_regression.py      # Regression demos
└── conftest.py                 # Global fixtures
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_storage.py -v

# With coverage
pytest tests/ --cov=behaviorci --cov-report=html

# Parallel execution (tests concurrency)
pytest tests/ -n 4

# Specific test
pytest tests/test_behaviorci.py::test_snapshot_creation -v
```

### Writing Tests

#### Unit Tests

Test individual components in isolation:

```python
def test_snapshot_id_computation():
    """Test that snapshot ID is deterministic."""
    from behaviorci.storage import compute_snapshot_id
    
    id1 = compute_snapshot_id("behavior", '{"x": 1}')
    id2 = compute_snapshot_id("behavior", '{"x": 1}')
    
    assert id1 == id2
    assert len(id1) == 64  # SHA-256 hex
```

#### Integration Tests

Test component interaction:

```python
def test_record_and_compare():
    """Test full record/compare cycle."""
    from behaviorci.storage import get_storage, reset_all_storage
    from behaviorci.comparator import Comparator
    from behaviorci.embedder import MockEmbedder
    
    reset_all_storage()
    storage = get_storage(":memory:")
    comparator = Comparator(storage, MockEmbedder())
    
    # Record
    comparator.record_snapshot("test", '{}', "output")
    
    # Compare same output
    result = comparator.compare("test", '{}', "output", 0.85)
    assert result.passed
    assert result.similarity == 1.0
```

#### Regression Tests

For bug fixes, include a test that would have failed before the fix:

```python
def test_concurrent_writes_no_lock_errors():
    """BUG-002: Verify WAL mode prevents database locking."""
    # This test would fail without WAL mode
    ...
```

### Test Requirements

- All bug fixes must include a regression test
- New features must include unit and integration tests
- Tests must pass in parallel (`pytest -n 4`)
- Tests must be deterministic (no random failures)

### Mocking

Use mocks for external dependencies:

```python
# tests/mock_embedder.py
import numpy as np

class MockEmbedder:
    """Fast mock embedder for testing (no model download)."""
    
    def embed_single(self, text: str) -> np.ndarray:
        # Deterministic "embedding" based on text hash
        hash_val = hash(text) % 10000
        vec = np.zeros(384, dtype=np.float32)
        vec[0] = hash_val / 10000.0
        vec[1] = 1.0 - (hash_val / 10000.0)
        # Normalize
        vec = vec / np.linalg.norm(vec)
        return vec
    
    @property
    def model_name(self):
        return "mock-embedder"
```

---

## Documentation

### Code Documentation

- All public APIs must have docstrings
- Complex algorithms must have inline comments explaining the approach
- Bug fixes must reference the bug number and explain the fix

### README Updates

If your change affects user-facing behavior, update the README:

- Add new features to "Quick Start" or "Advanced Usage"
- Update CLI command documentation
- Add troubleshooting entries for new error modes

### Changelog

We maintain a changelog in `CHANGELOG.md`. For user-facing changes, add an entry:

```markdown
## [Unreleased]

### Added
- New feature description (#123)

### Fixed
- Bug fix description (#456)

### Changed
- Breaking change description (#789)
```

---

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag -a v0.2.0 -m "Release version 0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions automatically builds and publishes to PyPI

---

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/0-uddeshya-0/BehaviorCI/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/0-uddeshya-0/BehaviorCI/issues)

---

## Recognition

Contributors will be recognized in our `README.md` and release notes. Thank you for helping make BehaviorCI better!

---

**Happy contributing!**
