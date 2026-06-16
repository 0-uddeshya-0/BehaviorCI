# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-16

### Added

- **Custom embedders.** Implement `BaseEmbedder` and register it with
  `set_embedder()` to back BehaviorCI with any embedding API (OpenAI, Cohere,
  Gemini, …) — no local ML stack required.
- **Centroid baselines** via `samples=N`: the test runs N times and is compared
  against the averaged embedding, which tames high-temperature prompts.
- **Async test support** for `async def` behavior tests.
- **`behaviorci history <id>`** shows a behavior's similarity over time as a
  small ASCII meter.
- **Machine-readable JSON report** via `--behaviorci-report PATH` for CI
  dashboards and automation.
- **Richer `behaviorci stats`** with a per-behavior table (count and last
  recorded time).
- **Documentation site** built with MkDocs Material, plus an SVG project banner.

### Changed

- The local embedding model is now an optional `[local]` extra; the core install
  no longer pulls in PyTorch.
- Minimum supported Python is now 3.10.
- Recording a snapshot prints the captured output so it can be reviewed before
  it becomes the baseline.
- Reports and summaries now show the embedding model actually used, including an
  injected one, rather than the configured default.

### Fixed

- WAL mode with per-thread connections resolves `database is locked` errors
  under `pytest-xdist`.
- The variance-aware threshold now uses the configured threshold as a ceiling:
  high-variance prompts loosen toward their observed floor while low-variance
  prompts stay strict.
- Comparing a baseline against a different embedding model raises a clear
  `ModelMismatchError` instead of scoring incompatible vectors.
- Storage connections are closed on reset and on `clear`, so handles aren't
  leaked and the database can be deleted on Windows.

## [0.1.0]

- Initial release: the `@behavior` decorator and pytest plugin, semantic
  similarity via `sentence-transformers`, and baseline recording in SQLite.
