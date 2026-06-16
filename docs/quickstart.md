# Quick start

## Install

```bash
# Lightweight core (bring your own embedding API)
pip install "git+https://github.com/0-uddeshya-0/BehaviorCI.git"

# With the local model (sentence-transformers + torch)
pip install "behaviorci[local] @ git+https://github.com/0-uddeshya-0/BehaviorCI.git"
```

Requires Python 3.10+.

!!! note "About the name"
    The `behaviorci` name on PyPI currently belongs to an unrelated project, so
    install from this repository for now.

## 1. Write a behavior test

A behavior test is an ordinary pytest function that **returns the string** you
want to track.

```python
# test_support.py
from behaviorci import behavior
from myapp import assistant

@behavior("support_tone", threshold=0.88, must_contain=["help"])
def test_support_tone():
    return assistant("I'm frustrated with my bill")
```

## 2. Record the baseline

```console
$ pytest test_support.py --behaviorci-record
```

BehaviorCI prints the captured output. **Read it** — it becomes the ground truth,
so make sure it isn't a hallucination before you commit it.

```text
Recorded snapshot: support_tone
Snapshot ID: 9f1c2a7b4e6d8a03...

Review the captured output below to make sure it is correct --
it becomes the baseline for future runs.
==================================================
I'm sorry you're dealing with this. I can help sort the bill out right now…
==================================================
```

## 3. Check for regressions

Change a prompt or swap a model, then run the check:

```console
$ pytest test_support.py --behaviorci
```
```text
FAILED test_support.py::test_support_tone
BehaviorCI: Similarity 0.7100 < threshold 0.8800

--- STORED OUTPUT (Primary Sample) ---
I'm sorry you're dealing with this. I can help sort the bill out right now…

--- CURRENT OUTPUT (Primary Sample) ---
Billing is handled by the finance team. Email finance@example.com.
```

## 4. Accept intentional changes

If the new behavior is what you want, update the baseline:

```console
$ pytest test_support.py --behaviorci-update
```

That's the whole loop — **record → check → update**. Snapshots live in a local
SQLite database under `.behaviorci/`.

Next: read the [Guide](guide.md) for thresholds, guardrails, and centroid
baselines.
