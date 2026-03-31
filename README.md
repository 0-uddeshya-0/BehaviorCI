# BehaviorCI

Pytest-native behavioral regression testing for LLM applications.

## Overview

BehaviorCI enables developers to:

1. **Write tests using normal pytest functions**
2. **Capture LLM outputs as behavioral snapshots**
3. **Compare future outputs using semantic similarity**
4. **Fail CI when behavior regresses**

Think: "Jest snapshot testing for LLM outputs, using embeddings instead of string equality."

## Quick Start

### Installation

```bash
pip install behaviorci
```

### Write Your First Behavior Test

```python
from behaviorci import behavior

@behavior("refund_classifier", threshold=0.85)
def test_refund_classification():
    result = classify("I want a refund")
    return result  # RETURN VALUE CAPTURED
```

### Record Baseline

```bash
pytest --behaviorci-record
```

### Check for Regressions

```bash
pytest --behaviorci
```

### Update on Intentional Changes

```bash
pytest --behaviorci-update
```

## Core Concepts

### The `@behavior` Decorator

```python
@behavior(
    behavior_id="unique_id",      # Logical behavior identifier
    threshold=0.85,                # Minimum similarity (0-1)
    must_contain=["required"],     # Required substrings
    must_not_contain=["forbidden"] # Forbidden substrings
)
def my_test():
    return llm.generate("prompt")  # Must return string
```

### How It Works

1. **Decorator captures return value** from test function
2. **Input is serialized** to JSON (strict - fails on non-serializable types)
3. **Snapshot ID** = SHA256(behavior_id + input_json)
4. **Embedding** computed using local sentence-transformers model
5. **Comparison** uses cosine similarity with variance-aware thresholds

### Storage

- SQLite database at `.behaviorci/behaviorci.db`
- Embeddings stored as float32 BLOBs
- Per-snapshot similarity history for variance tracking

## CLI Commands

```bash
# Record new snapshots
behaviorci record

# Check for regressions (CI mode)
behaviorci check

# Update failing snapshots
behaviorci update

# View statistics
behaviorci stats

# Clear all snapshots
behaviorci clear --force
```

## Configuration Options

### Pytest Options

```bash
pytest --behaviorci          # Enable regression testing
pytest --behaviorci-record   # Record/update snapshots
pytest --behaviorci-update   # Update failing snapshots
pytest --behaviorci-db PATH  # Custom database path
```

### Decorator Options

```python
@behavior(
    "my_behavior",
    threshold=0.85,                    # Similarity threshold
    must_contain=["helpful", "safe"],  # Required content
    must_not_contain=["harmful"]       # Forbidden content
)
```

## Architecture

```
behaviorci/
├── api.py        # @behavior decorator + return capture
├── plugin.py     # pytest hooks (core engine)
├── storage.py    # SQLite + BLOB serialization
├── embedder.py   # sentence-transformers wrapper
├── comparator.py # Layer 0 (lexical) + Layer 1 (semantic)
├── cli.py        # Typer CLI wrapper
├── models.py     # Pydantic models
└── exceptions.py # Error classes
```

### Evaluation Pipeline

**Layer 0: Lexical (fast fail)**
- `must_contain`: All substrings must exist
- `must_not_contain`: Safety violations
- If failed: Immediate failure, skip embedding

**Layer 1: Semantic (core)**
1. Load snapshot by ID
2. Compute embedding of current output
3. Cosine similarity vs stored embedding
4. Variance-aware threshold adjustment

### Variance-Aware Thresholds

```python
history = storage.get_similarity_history(snapshot_id, limit=5)
if len(history) >= 3:
    mean = np.mean(history)
    std = np.std(history)
    effective_threshold = max(base_threshold, mean - 2*std)
```

Different inputs under the same behavior_id have separate histories.

## Design Principles

1. **Pytest-native**: No separate workflow, runs via `pytest`
2. **Local-first**: Uses local embedding model (sentence-transformers)
3. **Deterministic-enough**: Handles embedding noise via tolerance
4. **Adoption-first**: 1-line integration per test
5. **Strict constraints**: JSON-serializable inputs only

## Requirements

- Python 3.8+
- pytest 7.0+
- sentence-transformers 2.0+
- numpy 1.20+
- pydantic 2.0+

## Testing

```bash
# Run BehaviorCI self-tests
pytest tests/test_behaviorci.py -v

# Run example tests
pytest tests/examples/test_app.py --behaviorci-record
pytest tests/examples/test_app.py --behaviorci
```

## License

MIT License