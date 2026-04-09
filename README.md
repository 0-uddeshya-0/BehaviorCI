<div align="center">
  <h1>🤖 BehaviorCI</h1>
  
  <p>
    <strong>pytest for LLM behavior</strong><br>
    Track semantic drift and prompt regressions during development.
  </p>

  <p>
    <a href="https://pypi.org/project/behaviorci/"><img src="https://img.shields.io/pypi/v/behaviorci.svg?style=for-the-badge&color=blue" alt="PyPI"></a>
    <a href="https://codecov.io/gh/0-uddeshya-0/BehaviorCI"><img src="https://img.shields.io/codecov/c/gh/0-uddeshya-0/BehaviorCI?style=for-the-badge&logo=codecov" alt="Coverage"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License"></a>
  </p>

  <br>

  <a href="https://github.com/0-uddeshya-0/BehaviorCI"><img src="https://img.shields.io/badge/Status-Alpha-orange.svg?style=for-the-badge" alt="Status: Alpha"></a>
</div>

> ⚠️ **Project Status: Alpha** — BehaviorCI is actively used for local development and CI pipelines, but the internal API and storage mechanisms are currently stabilizing. Breaking changes may occur before v1.0. We strongly recommend pinning your dependency version (e.g., `behaviorci==0.1.0`).

<hr/>

## 📑 Table of Contents
- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Installation Options](#-installation-options)
- [CLI Reference](#-cli-reference)
- [Advanced: API Embedders](#-advanced-api-embedders)
- [Known Limitations & Workflows](#️-known-limitations--team-workflows)
- [Architecture](#️-architecture)

---

## 🚨 The Problem

You shipped a prompt update. It looked fine in standard unit tests, but downstream metrics dropped.

**What happened?** Your prompt tweak changed the output format. The downstream parser broke. 

Traditional exact-match string testing (`assert result == "Success"`) is brittle for generative AI. Unit tests pass, but the **underlying semantic behavior** has drifted.

```python
# Baseline Output:  "The refund will be processed in 3-5 business days."
# Modified Prompt:  "Refund approved. Processing time: 3-5 days."
# Result:           Similar semantic meaning, completely different string.
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

> **Built for teams shipping LLM Applications to production. Early adopters welcome - feedback shapes the roadmap.**

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
    return response  # Must return the generative string
```

### 3. Record Baseline

```bash
pytest test_support.py --behaviorci-record
```
```text
✅ Recorded snapshot: support_tone_check
```
Note: Always review your terminal output on the first record to ensure you aren't capturing a hallucination as your ground truth.

### 4. Check for Regressions

Change your prompt or switch underlying models, then run the check:

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

If the behavior change was intentional, accept it:

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
    behavior_id="unique_identifier",      # Logical name (must be unique per suite)
    threshold=0.85,                       # Minimum similarity (0-1)
    must_contain=["required", "words"],   # Lexical fast-fail (optional)
    must_not_contain=["forbidden"]        # Lexical fast-fail (optional)
)
def test_your_llm_function():
    result = your_llm_function("input")
    return result  # RETURN VALUE IS CAPTURED
```
> **Critical**: The test function must return a string. This return value is the behavior being tracked.

### Variance-Aware Thresholds

BehaviorCI attempts to learn your prompt's normal variance:
* **Runs 1 & 2**: `threshold = 0.85` (your baseline setting).
* **Runs 3+**: `threshold = max(0.85, mean(history) - 2*std)`.
* If your outputs naturally fluctuate (e.g., highly creative prompts), the system will dynamically relax the similarity threshold to avoid false positive test failures. If the output is highly rigid (e.g., JSON), the threshold remains strict.
---

## 🛠️ Installation Options

BehaviorCI offers two installation paths to keep your CI pipelines as lean as possible.

**Option A: Lightweight (API-Based)**
Installs only core dependencies (a few MBs). Requires you to inject an API-based embedder (like OpenAI or Cohere) in your tests.
```bash
pip install behaviorci
```

**Option B: Offline (Local Models)**
Installs `sentence-transformers` and PyTorch (~1GB). Runs everything locally with zero API calls.
```bash
pip install "behaviorci[local]"
```

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
          
      - run: pip install "behaviorci[local]" pytest
      
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

## 🔌 Advanced: API Embedders

If you chose the Lightweight installation, you can easily inject any API client you already use (OpenAI, Gemini, Cohere, etc.) to handle the embeddings without adding PyTorch to your system.

Simply define and inject your embedder in your `conftest.py`:

```python
import pytest
import numpy as np
from behaviorci.embedder import Embedder, set_embedder
from openai import OpenAI

class OpenAIEmbedder(Embedder):
    def __init__(self):
        super().__init__(model_name="text-embedding-3-small")
        self.client = OpenAI()

    def embed_single(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=text, 
            model=self.model_name
        )
        # BehaviorCI requires a normalized float32 numpy array
        return np.array(response.data[0].embedding, dtype=np.float32)

# Inject globally before test collection
set_embedder(OpenAIEmbedder())
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

## ⚠️ Known Limitations & Team Workflows

### 1. The Git + SQLite Merge Conflict
Currently, BehaviorCI stores all snapshots in a binary SQLite database (`.behaviorci/behaviorci.db`). Because it is a binary file, **Git cannot automatically resolve merge conflicts** if multiple developers record new snapshots on different branches simultaneously.

**Recommended Workflow for Teams:**
* **Read-Only in CI:** Running `pytest --behaviorci` in your GitHub Actions/CI pipeline for Pull Requests works perfectly and safely.
* **Record on Main:** To avoid binary merge conflicts, only run `pytest --behaviorci-record` or `--behaviorci-update` locally on the `main` branch. Alternatively, designate a single maintainer to handle all snapshot updates.
* *(Note: We are actively exploring Git-friendly JSON file snapshots for v0.2.0 to fully resolve this).*

### 2. Handling Non-Determinism (Creative Outputs)
If your LLM prompt has high variance (e.g., high temperature, creative writing), do not force a high threshold. BehaviorCI features **Variance-Aware Thresholds**—if a test historically shows high variance, the tool automatically lowers the effective threshold. 
* For strict formatting (JSON, low temperature), use `@behavior(threshold=0.95)`
* For creative text, use `@behavior(threshold=0.60)`

  
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
