# CLI reference

Everything BehaviorCI does is available through pytest flags. The `behaviorci`
command is a thin wrapper around those flags plus a couple of inspection
commands.

## Commands

| Command | What it does |
| --- | --- |
| `behaviorci record [path]` | Record/overwrite baselines |
| `behaviorci check [path]` | Fail on regressions (CI mode) |
| `behaviorci update [path]` | Accept new behavior for failing tests |
| `behaviorci record-missing [path]` | Record only missing snapshots, check the rest |
| `behaviorci stats` | Totals plus a per-behavior table |
| `behaviorci history <id>` | Similarity over time for one behavior |
| `behaviorci clear --force` | Delete all snapshots (and WAL sidecars) |

`path` defaults to `tests/`. Every command accepts `--db PATH` to target a
non-default database.

```console
$ behaviorci stats
BehaviorCI Statistics
=====================
Total Snapshots:  12
Unique Behaviors: 12
History Records:  34

Behavior                         Snapshots   Last recorded
----------------------------------------------------------------------
refund_reply                             1   2026-06-16 09:39
support_tone                             1   2026-06-16 09:39
```

```console
$ behaviorci history refund_reply
Behavior: refund_reply   (snapshot 4eb1e60f5a11)
Input:    {"args": [], "kwargs": {}}
  2026-06-16 09:39   0.9013  [######################--]
  2026-06-15 17:02   0.8456  [####################----]
```

## pytest flags

The plugin adds these options to `pytest`:

| Flag | Meaning |
| --- | --- |
| `--behaviorci` | Check mode — compare against baselines and fail on drift |
| `--behaviorci-record` | Record/overwrite baselines |
| `--behaviorci-update` | Update baselines for behaviors that would otherwise fail |
| `--behaviorci-record-missing` | Record snapshots that don't exist yet; check the rest |
| `--behaviorci-db PATH` | Database location (default `.behaviorci/behaviorci.db`) |
| `--behaviorci-model NAME` | Embedding model for the local backend |
| `--behaviorci-report PATH` | Write a JSON report of the run |

## JSON report

Add `--behaviorci-report report.json` to any run to emit a structured summary
for dashboards, PR bots, or other automation.

```json
{
  "schema": "behaviorci/report/v1",
  "generated_at": "2026-06-16T09:39:12.481+00:00",
  "mode": "check",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "summary": { "total": 12, "passed": 11, "failed": 1, "recorded": 0, "checked": 12 },
  "results": [
    {
      "behavior_id": "refund_reply",
      "snapshot_id": "4eb1e60f5a11…",
      "action": "checked",
      "passed": false,
      "similarity": 0.71,
      "base_threshold": 0.85,
      "effective_threshold": 0.85,
      "model_mismatch": false,
      "samples": 1,
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "nodeid": "tests/test_support.py::test_refund_reply"
    }
  ]
}
```

`action` is one of `recorded`, `recorded_missing`, `checked`, or `error`. The
top-level `model` reflects the embedder actually used (including an injected
one), not just the configured default.
