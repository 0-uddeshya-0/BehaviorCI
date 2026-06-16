# FAQ & troubleshooting

### "No snapshot found" in CI

A new test has no committed baseline. Run with `--behaviorci-record-missing` so
CI records it instead of failing, then review and commit the database. See
[CI & teams](ci.md).

### A test fails even though the outputs look similar

A lexical guardrail fired. Check the failure for `Missing required` or
`Found forbidden` — `must_contain` / `must_not_contain` are enforced regardless
of the similarity score.

### `ModelMismatchError`

The baseline was recorded with a different embedding model. Re-record with
`--behaviorci-update`, or point at the original model via `--behaviorci-model`.
Embeddings from different models aren't comparable, so BehaviorCI refuses to
guess.

### The first run is slow

The local model (~80 MB) downloads once from Hugging Face. Cache
`~/.cache/huggingface` in CI; every later run is offline.

### Can it work completely offline?

Yes. After the one-time model download (or if you inject your own embedder),
nothing leaves the machine and no API keys are needed.

### `database is locked` under `pytest -n`

The database uses WAL mode with per-thread connections, so `pytest-xdist` is
supported. If you still see this, make sure no other process is holding the file
open, and that the database isn't on a filesystem that doesn't support locking
(some network mounts).

### How do I pick a threshold?

Start at `0.85`. Tighten toward `0.9–0.95` for structured or classification
output, loosen toward `0.6–0.75` for creative or high-temperature prompts (and
consider `samples` for those). After a few runs, the
[variance-aware threshold](guide.md#variance-aware-thresholds) adapts on its own.

### My output has timestamps or random ids

Mock or normalize them so the baseline stays stable — see
[handling non-determinism](guide.md#handling-non-determinism).

### Where are snapshots stored, and should I commit them?

In `.behaviorci/behaviorci.db` by default. Commit it to version your behavior
with your code. The `*.db-wal` / `*.db-shm` sidecars are throwaway and already
git-ignored.

### Does it run the test more than once?

No. With `samples=1` the test runs exactly once; BehaviorCI reads the captured
return value afterward. With `samples=N` it intentionally runs `N` times to build
a centroid baseline.
