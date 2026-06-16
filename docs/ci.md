# CI & teams

## The CI pattern

Check existing behaviors strictly, but **auto-record brand new ones** so a
freshly added test doesn't fail the build before anyone has reviewed its
baseline. That's exactly what `--behaviorci-record-missing` does.

```yaml
# .github/workflows/behavior.yml
name: Behavior
on: [push, pull_request]

jobs:
  behavior:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Cache the embedding model
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: ${{ runner.os }}-behaviorci-model

      - run: pip install "behaviorci[local] @ git+https://github.com/0-uddeshya-0/BehaviorCI.git" pytest

      - run: pytest --behaviorci-record-missing --behaviorci-report bci.json

      - uses: actions/upload-artifact@v4
        with:
          name: behaviorci-report
          path: bci.json
```

Caching `~/.cache/huggingface` keeps the one-time model download (~80 MB) out of
every run.

## Reading the result in automation

The `--behaviorci-report` JSON is the integration point. A follow-up step can
turn it into a PR comment, a status check, or a metrics push:

```bash
python - <<'PY'
import json
r = json.load(open("bci.json"))
s = r["summary"]
print(f"BehaviorCI: {s['passed']}/{s['total']} passed, {s['failed']} regressions")
for item in r["results"]:
    if not item["passed"]:
        print(f"  ✗ {item['behavior_id']} — similarity {item.get('similarity')}")
PY
```

## Team workflows for the binary database

Baselines live in a single SQLite file (`.behaviorci/behaviorci.db`). Commit it
to version your behavior alongside your code. Because it's binary, Git can't
merge two copies, so concurrent baseline edits on different branches can
conflict.

A workflow that avoids that:

- **CI stays read-only.** `pytest --behaviorci` on pull requests is always safe
  and never writes.
- **Record on a known branch.** Run `--behaviorci-record` / `--behaviorci-update`
  on `main`, or let a single maintainer own baseline updates.
- **Ignore the sidecars.** WAL files (`*.db-wal`, `*.db-shm`) are throwaway and
  already excluded by `.gitignore`.

## Parallel runs

The database uses WAL mode with per-thread connections, so `pytest -n auto`
(pytest-xdist) is supported out of the box — no `database is locked` errors.

!!! tip "A Git-friendly backend is on the roadmap"
    A JSON snapshot backend that diffs and merges cleanly in Git is planned;
    until then, the single-writer workflow above keeps things simple.
