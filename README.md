<div align="center">
  <h1>🤖 BehaviorCI</h1>

  <p>
    <strong>pytest for LLM behavior</strong><br>
    Catch prompt regressions before they reach production.
  </p>

  <p>
    <a href="https://pypi.org/project/behaviorci/"><img src="https://img.shields.io/pypi/v/behaviorci.svg?style=for-the-badge&color=blue" alt="PyPI"></a>
    <a href="https://github.com/0-uddeshya-0/BehaviorCI/actions"><img src="https://img.shields.io/github/actions/workflow/status/0-uddeshya-0/BehaviorCI/ci.yml?style=for-the-badge&logo=github" alt="CI"></a>
    <a href="https://codecov.io/gh/0-uddeshya-0/BehaviorCI"><img src="https://img.shields.io/codecov/c/gh/0-uddeshya-0/BehaviorCI?style=for-the-badge&logo=codecov" alt="Coverage"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License"></a>
  </p>

  <p>
    <em>Record once. Check forever. Fail on semantic drift.</em>
  </p>
</div>

<hr/>

## 📑 Table of Contents
- The Problem
- The Solution
- Why BehaviorCI?
- Quick Start
- How It Works
- Core Concepts
- Installation & Setup
- CLI Reference
- Advanced Usage
- Architecture
- Alternatives
- Troubleshooting

---

## 🚨 The Problem

You shipped a prompt update. It looked fine in testing. Then production metrics dropped.

**What happened?** Your "improved" prompt changed output format. The downstream parser broke. Users saw errors. Revenue dipped.

Traditional testing doesn't catch this. Unit tests pass. Integration tests pass. But **behavior changed** in subtle ways that matter.

```python
# Before: "The refund will be processed in 3-5 business days."
# After:  "Refund approved. Processing time: 3-5 days."
# Result: Similar meaning, different format. Broken parser.
```

## 💡 The Solution

BehaviorCI captures your LLM outputs as **behavioral snapshots** and detects **semantic drift** using embeddings. Not string matching—**meaning comparison**.

```python
from behaviorci import behavior

@behavior("refund_classifier", threshold=0.85, must_contain=["days"])
def test_refund_processing():
    result = classify_support_ticket("How long for refund?")
    return result  # RETURN VALUE CAPTURED
```

---

## ✨ Why BehaviorCI?

| ❌ Without BehaviorCI | ✅ With BehaviorCI |
|-------------------|-----------------|
| "Looks good to me" code reviews | Automated regression detection |
| Production surprises | CI fails before merge |
| Manual output inspection | Semantic similarity scoring |
| "It worked yesterday" mysteries | Per-input variance tracking |
| Silent prompt degradation | Quantified behavior drift |

> **Used by teams shipping LLM applications to production.**

---

## 🚀 Quick Start (60 seconds)

### 1. Install

```bash
pip install behaviorci
```
> **Note**: The first run downloads the embedding model (~22MB). Subsequent runs are instant and offline.

### 2. Write Your First Behavior Test

```python
# test_support.py
from behaviorci import behavior

@behavior("support_tone_check", threshold=0.90, must_contain=["help"])
def test_support_response_tone():
    """Verify support responses maintain helpful tone."""
    response = generate_support_response("I'm frustrated with billing")
    return response  # Must return string
```

### 3. Record Baseline

```bash
pytest test_support.py --behaviorci-record
```
```text
.behaviorci/behaviorci.db created
1 snapshot recorded
```

### 4. Check for Regressions

Change your prompt, then run the check:

```bash
pytest test_support.py --behaviorci
```
```text
FAILED test_support.py::test_support_response_tone
BehaviorCI: Similarity 0.72 < threshold 0.90
--- STORED OUTPUT ---
I'm sorry to hear about your billing frustration. Let me help resolve this...
--- CURRENT OUTPUT ---
Billing issues are handled by our finance team. Contact finance@example.com.
```
*Regression caught in CI, not production!*

### 5. Update When Intentional

```bash
pytest test_support.py --behaviorci-update
```

---

## ⚙️ How It Works

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Test Function │────▶│  @behavior       │────▶│  Capture Output │
│   (returns str) │     │  (decorator)     │     │  + Input Args   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
                              ┌─────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  SQLite Storage  │
                    │  (.behaviorci/)  │
                    │                  │
                    │  • Input hash    │
                    │  • Output text   │
                    │  • Embedding     │◄── sentence-transformers
                    │  • History       │    (all-MiniLM-L6-v2)
                    └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │  Layer 0:    │    │  Layer 1:    │
            │  Lexical     │    │  Semantic    │
            │  (must_contain)   │  (embedding  │
            │              │    │  similarity) │
            └──────────────┘    └──────────────┘
```

* **Layer 0: Lexical Guards** (Fast Fail)
  * `must_contain`: Required substrings
  * `must_not_contain`: Forbidden patterns
  * Zero embedding cost
* **Layer 1: Semantic Comparison**
  * Embedding similarity via cosine distance
  * Variance-aware thresholds (adapts to your data)
  * Model mismatch detection

---

## 🧠 Core Concepts

### The `@behavior` Decorator

```python
@behavior(
    behavior_id="unique_identifier",      # Logical name (validated for uniqueness)
    threshold=0.85,                       # Minimum similarity (0-1)
    must_contain=["required", "words"],   # Required substrings
    must_not_contain=["forbidden"]        # Forbidden patterns
)
def test_your_llm_function():
    result = your_llm_function("input")
    return result  # RETURN VALUE IS CAPTURED
```
> **Critical**: The test function must return a string. This return value is the behavior being tracked.

### Variance-Aware Thresholds

BehaviorCI learns your normal variance:
* **Runs 1 & 2**: `threshold = 0.85` (your baseline setting).
* **Runs 3+**: `threshold = max(0.85, mean(history) - 2*std)`.
* If your outputs naturally vary (e.g., creative writing), the threshold adapts to avoid false positives. If outputs should be identical (e.g., structured JSON), the threshold stays high.

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.8+
- pytest 7.0+
- ~22MB disk space (embedding model)

### CI/CD Configuration

BehaviorCI is built for automated pipelines. Use `--behaviorci-record-missing` to automatically approve new tests while strictly verifying existing ones.

**GitHub Actions** (`.github/workflows/behavior.yml`):
```yaml
name: Behavioral Regression Tests
on: [push, pull_request]

jobs:
  behavior:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
          
      - uses: actions/cache@v4
        with:
          path: ~/.cache/torch/sentence_transformers
          key: ${{ runner.os }}-st-model-${{ hashFiles('**/requirements.txt') }}
          
      - run: pip install behaviorci pytest
      
      # Check existing behaviors, auto-record new ones
      - run: pytest --behaviorci-record-missing
      
      # Commit updated snapshots if on main
      - uses: stefanzweifel/git-auto-commit-action@v5
        if: github.ref == 'refs/heads/main'
        with:
          commit_message: "chore: update behavioral snapshots"
          file_pattern: ".behaviorci/"
```

---

## 💻 CLI Reference

```bash
# Record snapshots (create or overwrite)
behaviorci record [test_directory]

# Check for regressions (CI mode)
behaviorci check [test_directory]

# Update failing snapshots (accept new behavior)
behaviorci update [test_directory]

# Record only missing snapshots (CI workflow)
behaviorci record-missing [test_directory]

# View database statistics
behaviorci stats

# Clear all snapshots (destructive)
behaviorci clear --force
```
> All commands support `--db PATH` for custom database locations.

---

## 📈 Advanced Usage

### Handling Non-Deterministic Outputs
Some outputs contain timestamps, random IDs, or dates. Handle these by normalizing inputs or mocking:

```python
from unittest.mock import patch
from datetime import datetime

@behavior("daily_summary")
def test_daily_summary():
    with patch('myapp.get_today', return_value=datetime(2024, 1, 15)):
        return generate_daily_summary()  # Now deterministic
```

### Parallel Execution (pytest-xdist)
BehaviorCI's WAL-mode SQLite database natively supports concurrent execution without database locking.
```bash
pip install pytest-xdist
pytest --behaviorci -n 4  # Safe 4-worker execution
```

---

## 🏛️ Architecture

Designed for robustness and speed:
1. **Local-first**: No API keys, no cloud calls, works completely offline.
2. **pytest-native**: No separate workflow; it hooks directly into your existing test suites. Safe against double-execution bugs.
3. **Deterministic-enough**: Handles FP32 embedding variance via tolerance bands.
4. **Production-hardened**: WAL mode, singleton storage, and thread-safe validations ensure duplicate `behavior_id` constraints are strictly enforced before runtime.

---

## ⚖️ Comparison with Alternatives

| Tool | Approach | CI Integration | Environment | Best For |
|------|----------|----------------|-------------|----------|
| **BehaviorCI** | Snapshot + embeddings | pytest-native | **Local** | Regression testing, CI safety |
| *Promptfoo* | Prompt A/B testing | CLI-first | Cloud option | Prompt iteration, evals |
| *DeepEval* | Metrics-based | pytest plugin | Cloud | LLM evaluation, scoring |
| *LangSmith* | Observability | Separate UI | Cloud-only | Debugging, tracing |

> **BehaviorCI is the only tool that treats LLM outputs like Jest snapshots**: record once, compare forever, fail on drift.

---

## 🚑 Troubleshooting

<details>
<summary><strong>"database is locked" errors</strong></summary>
<br>
<strong>Cause</strong>: Multiple processes writing without WAL mode.<br>
<strong>Fix</strong>: Upgrade to BehaviorCI ≥0.1.1 (WAL mode enabled by default).
</details>

<details>
<summary><strong>"No snapshot found" in CI</strong></summary>
<br>
<strong>Cause</strong>: New test added, snapshot not committed.<br>
<strong>Fix</strong>: Use the <code>--behaviorci-record-missing</code> flag in CI.
</details>

<details>
<summary><strong>High similarity but test fails</strong></summary>
<br>
<strong>Cause</strong>: <code>must_contain</code> or <code>must_not_contain</code> lexical checks.<br>
<strong>Fix</strong>: Check error message for "Missing required" or "Found forbidden".
</details>

<details>
<summary><strong>Embedding model download is slow</strong></summary>
<br>
<strong>Cause</strong>: First run downloads 22MB from HuggingFace.<br>
<strong>Fix</strong>: Cache <code>~/.cache/torch/sentence_transformers</code> in your CI pipeline.
</details>

<details>
<summary><strong>"Duplicate behavior_id" error on collection</strong></summary>
<br>
<strong>Cause</strong>: Two or more tests are sharing the same behavior ID string.<br>
<strong>Fix</strong>: Update the <code>@behavior("your_id")</code> decorator to be strictly unique across the suite.
</details>

---

## 🤝 Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

**Development setup**:
```bash
git clone https://github.com/0-uddeshya-0/BehaviorCI.git
cd BehaviorCI
pip install -e ".[dev]"
pytest tests/ -v
```

## 📝 License

Distributed under the MIT License. See LICENSE for more information.

---

<div align="center">
  <strong>Star ⭐ if this saved your production deployment.</strong><br>
  <a href="https://github.com/0-uddeshya-0/BehaviorCI/issues">Report issues</a> • 
  <a href="https://github.com/0-uddeshya-0/BehaviorCI/discussions">Discussions</a>
</div>

---

## Why BehaviorCI?

| Without BehaviorCI | With BehaviorCI |
|-------------------|-----------------|
| "Looks good to me" code reviews | Automated regression detection |
| Production surprises | CI fails before merge |
| Manual output inspection | Semantic similarity scoring |
| "It worked yesterday" mysteries | Per-input variance tracking |
| Silent prompt degradation | Quantified behavior drift |

**Used by teams shipping LLM applications to production.**

---

## Quick Start (60 seconds)

### 1. Install

```bash
pip install behaviorci
```

**First run downloads the embedding model (~22MB).** Subsequent runs are instant and offline.

### 2. Write Your First Behavior Test

```python
# test_support.py
from behaviorci import behavior

@behavior("support_tone_check", threshold=0.90, must_contain=["help"])
def test_support_response_tone():
    """Verify support responses maintain helpful tone."""
    response = generate_support_response("I'm frustrated with billing")
    return response  # Must return string
```

### 3. Record Baseline

```bash
pytest test_support.py --behaviorci-record
```

```
.behaviorci/behaviorci.db created
1 snapshot recorded
```

### 4. Check for Regressions

Change your prompt. Run check:

```bash
pytest test_support.py --behaviorci
```

```
FAILED test_support.py::test_support_response_tone
BehaviorCI: Similarity 0.72 < threshold 0.90
--- STORED OUTPUT ---
I'm sorry to hear about your billing frustration. Let me help resolve this...
--- CURRENT OUTPUT ---
Billing issues are handled by our finance team. Contact finance@example.com.
```

**Regression caught in CI, not production.**

### 5. Update When Intentional

```bash
pytest test_support.py --behaviorci-update
```

---

## How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Test Function │────▶│  @behavior       │────▶│  Capture Output │
│   (returns str) │     │  (decorator)     │     │  + Input Args   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
                              ┌─────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  SQLite Storage  │
                    │  (.behaviorci/)  │
                    │                  │
                    │  • Input hash    │
                    │  • Output text   │
                    │  • Embedding     │◄── sentence-transformers
                    │  • History       │    (all-MiniLM-L6-v2)
                    └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │  Layer 0:    │    │  Layer 1:    │
            │  Lexical     │    │  Semantic    │
            │  (must_contain)   │  (embedding  │
            │              │    │  similarity) │
            └──────────────┘    └──────────────┘
```

**Layer 0: Lexical Guards** (Fast Fail)
- `must_contain`: Required substrings
- `must_not_contain`: Forbidden patterns
- Zero embedding cost

**Layer 1: Semantic Comparison**
- Embedding similarity via cosine distance
- Variance-aware thresholds (adapts to your data)
- Model mismatch detection

---

## Core Concepts

### The `@behavior` Decorator

```python
@behavior(
    behavior_id="unique_identifier",    # Logical name
    threshold=0.85,                       # Minimum similarity (0-1)
    must_contain=["required", "words"],   # Required substrings
    must_not_contain=["forbidden"]        # Forbidden patterns
)
def test_your_llm_function():
    # Execute your LLM call
    result = your_llm_function("input")
    return result  # RETURN VALUE IS CAPTURED
```

**Critical**: The test function must return a string. This return value is the behavior being tracked.

### Variance-Aware Thresholds

BehaviorCI learns your normal variance:

```python
# First 2 runs: threshold = 0.85 (your setting)
# Run 3-5: threshold = max(0.85, mean(history) - 2*std)
# 
# If your outputs naturally vary (creative writing),
# threshold adapts to avoid false positives.
# 
# If your outputs should be identical (structured data),
# threshold stays high.
```

### Storage & CI

```bash
# Local development
pytest --behaviorci-record    # Create/update snapshots

# CI pipeline (GitHub Actions, GitLab, etc.)
pytest --behaviorci           # Fail on regression

# CI with auto-record for new tests
pytest --behaviorci-record-missing  # Check existing, record new
```

**Database**: SQLite at `.behaviorci/behaviorci.db`
- Commit the `.db` file to version control
- WAL mode enabled for parallel execution
- Embeddings stored as compressed BLOBs

---

## Installation & Setup

### Requirements

- Python 3.8+
- pytest 7.0+
- ~22MB disk space (embedding model)

### pip install

```bash
pip install behaviorci
```

### Development install

```bash
git clone https://github.com/0-uddeshya-0/BehaviorCI.git
cd BehaviorCI
pip install -e ".[dev]"
```

### CI Configuration

**GitHub Actions** (`.github/workflows/behavior.yml`):

```yaml
name: Behavioral Regression Tests

on: [push, pull_request]

jobs:
  behavior:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - uses: actions/cache@v4
        with:
          path: ~/.cache/torch/sentence_transformers
          key: ${{ runner.os }}-sentence-transformers-${{ hashFiles('**/requirements.txt') }}
          
      - run: pip install behaviorci pytest
      
      # Check existing behaviors, auto-record new ones
      - run: pytest --behaviorci-record-missing
      
      # Commit updated snapshots if on main
      - uses: stefanzweifel/git-auto-commit-action@v5
        if: github.ref == 'refs/heads/main'
        with:
          commit_message: "Update behavioral snapshots"
          file_pattern: ".behaviorci/"
```

**GitLab CI** (`.gitlab-ci.yml`):

```yaml
behavior_tests:
  image: python:3.11
  cache:
    paths:
      - .behaviorci/
      - ~/.cache/torch/sentence_transformers
  script:
    - pip install behaviorci pytest
    - pytest --behaviorci-record-missing
  artifacts:
    paths:
      - .behaviorci/
    expire_in: 1 week
```

---

## CLI Commands

```bash
# Record snapshots (create or overwrite)
behaviorci record [test_directory]

# Check for regressions (CI mode)
behaviorci check [test_directory]

# Update failing snapshots (accept new behavior)
behaviorci update [test_directory]

# Record only missing snapshots (CI workflow)
behaviorci record-missing [test_directory]

# View database statistics
behaviorci stats

# Clear all snapshots (destructive)
behaviorci clear --force
```

All commands support `--db PATH` for custom database locations.

---

## Advanced Usage

### Handling Non-Deterministic Outputs

Some outputs contain timestamps, random IDs, or dates. Handle these by normalizing inputs or mocking:

```python
from unittest.mock import patch
from datetime import datetime

@behavior("daily_summary")
def test_daily_summary():
    with patch('myapp.get_today', return_value=datetime(2024, 1, 15)):
        return generate_daily_summary()  # Now deterministic
```

### Custom Thresholds Per Test

```python
# High variance allowed (creative writing)
@behavior("poem_generator", threshold=0.75)
def test_poem():
    return generate_poem("love")

# Strict consistency (structured data)
@behavior("json_formatter", threshold=0.95, must_contain=["{", "}"])
def test_json():
    return format_as_json(data)
```

### Parallel Execution

BehaviorCI supports `pytest-xdist`:

```bash
pip install pytest-xdist
pytest --behaviorci -n 4  # 4 parallel workers
```

WAL mode ensures no database locking.

---

## Architecture

```
behaviorci/
├── api.py           # @behavior decorator, input serialization
├── plugin.py        # pytest hooks (collection, execution, reporting)
├── storage.py       # SQLite with WAL, singleton pattern, thread-local
├── embedder.py      # sentence-transformers wrapper (local, offline)
├── comparator.py    # Layer 0 (lexical) + Layer 1 (semantic)
├── cli.py           # Typer-based CLI
├── models.py        # Pydantic models (Snapshot, ComparisonResult)
└── exceptions.py    # Structured error hierarchy
```

**Design Principles**:
1. **Local-first**: No API keys, no cloud calls, works offline
2. **pytest-native**: No separate workflow, integrates with existing tests
3. **Deterministic-enough**: Handles FP32 embedding variance via tolerance bands
4. **Production-hardened**: WAL mode, singleton storage, thread-safe

---

## Comparison with Alternatives

| Tool | Approach | CI Integration | Local/Cloud | Best For |
|------|----------|----------------|-------------|----------|
| **BehaviorCI** | Snapshot + embeddings | pytest-native | Local | Regression testing, CI safety |
| Promptfoo | Prompt A/B testing | CLI-first | Cloud option | Prompt iteration, evals |
| DeepEval | Metrics-based | pytest plugin | Cloud | LLM evaluation, scoring |
| LangSmith | Observability | Separate UI | Cloud-only | Debugging, tracing |

**BehaviorCI is the only tool that treats LLM outputs like Jest snapshots**: record once, compare forever, fail on drift.

---

## Troubleshooting

### "database is locked" errors

**Cause**: Multiple processes writing without WAL mode.  
**Fix**: Upgrade to BehaviorCI ≥0.1.1 (WAL mode enabled by default).

### "No snapshot found" in CI

**Cause**: New test added, snapshot not committed.  
**Fix**: Use `--behaviorci-record-missing` flag in CI.

### High similarity but test fails

**Cause**: `must_contain` or `must_not_contain` lexical checks.  
**Fix**: Check error message for "Missing required" or "Found forbidden".

### Embedding model download slow

**Cause**: First run downloads 22MB from HuggingFace.  
**Fix**: Cache `~/.cache/torch/sentence_transformers` in CI.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Development setup**:

```bash
git clone https://github.com/0-uddeshya-0/BehaviorCI.git
cd BehaviorCI
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT License. See [LICENSE](LICENSE).

---

<p align="center">
  <strong>Star ⭐ if this saved your production deployment.</strong><br>
  <a href="https://github.com/0-uddeshya-0/BehaviorCI/issues">Report issues</a> • 
  <a href="https://github.com/0-uddeshya-0/BehaviorCI/discussions">Discussions</a>
</p>
