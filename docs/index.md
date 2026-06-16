# BehaviorCI

<img src="assets/banner.svg" alt="BehaviorCI" style="width:100%;border-radius:12px;">

BehaviorCI is a **pytest plugin** that snapshots what your prompt-based functions
say and fails the build when the meaning drifts. Write a normal test, return the
generated string, and BehaviorCI records a baseline on the first run. Later runs
compare the new output to the baseline **by semantic similarity** — so a
reworded-but-equivalent answer passes, and a genuine regression fails.

```python
from behaviorci import behavior

@behavior("refund_reply", threshold=0.85, must_contain=["business days"])
def test_refund_reply():
    return assistant("How long does a refund take?")
```

```console
$ pytest --behaviorci-record   # save the baseline
$ pytest --behaviorci          # fail on drift
$ pytest --behaviorci-update   # accept an intentional change
```

## Why

`assert reply == "..."` can't test generative output — the wording is never
byte-stable. BehaviorCI compares **meaning** instead:

```text
Baseline : "Your refund will be processed in 3–5 business days."
New build : "Refund approved. Processing time: 3–5 days."
            same meaning, different string → exact-match testing is useless
```

It works like a snapshot test (think Jest), but the comparison is an embedding
cosine similarity rather than string equality. **Record once, compare forever,
fail on drift.**

## What you get

- Semantic comparison with a per-test `threshold`.
- `must_contain` / `must_not_contain` lexical guardrails that fail fast.
- Variance-aware thresholds that adapt to noisy prompts.
- Centroid baselines for high-temperature, creative output.
- Local-first and offline after a one-time model download — or inject your own
  embedding API.
- CI-ready: record-missing mode, a JSON report, parallel-safe storage, and
  async/parametrized test support.

## Where next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quick start](quickstart.md)** — record and check in about a minute.
- :material-book-open-variant: **[Guide](guide.md)** — thresholds, guardrails, centroids, async.
- :material-console: **[CLI reference](cli.md)** — every command and flag.
- :material-infinity: **[CI & teams](ci.md)** — pipelines and the shared database.
- :material-puzzle: **[Custom embedders](embedders.md)** — plug in OpenAI, Cohere, Gemini.

</div>
