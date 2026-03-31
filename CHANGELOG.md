# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-01

### Added
- **TASK 1**: Model mismatch hard fail - `ModelMismatchError` raised when comparing embeddings from different models
- **TASK 2**: Database index on `similarity_history(timestamp DESC)` for efficient history queries
- **TASK 3**: Verified `@pytest.mark.parametrize` compatibility with `@behavior` decorator
- **TASK 4**: GitHub Actions CI workflow with Python 3.9-3.12 matrix testing
- **TASK 5**: `--behaviorci-model` flag for custom embedding models (e.g., `--behaviorci-model sentence-transformers/all-mpnet-base-v2`)
- `--model` flag added to all CLI commands (`record`, `check`, `update`, `record-missing`)
- `get_similarity_history_with_timestamps()` method in Storage for history command

### Changed
- Model mismatch is now a hard error instead of a warning (mathematically invalid to compare different embedding spaces)
- CLI `stats` command now uses `storage.get_behavior_summary()` instead of raw SQLite
- Terminal summary now shows active model name when `--behaviorci-model` is used

### Fixed
- N/A (no bug fixes in this release, only features)

## [0.1.0] - 2026-03-31

### Added
- Initial release of BehaviorCI
- pytest-native behavioral regression testing
- SQLite storage with WAL mode for concurrency
- Variance-aware thresholds
- @behavior decorator with return value capture
- CLI commands: record, check, update, stats, clear

### Fixed
- CRITICAL-001: Double test execution in pytest hook
- CRITICAL-002: Backwards variance threshold logic
- BUG-001: datetime import location
- BUG-002: SQLite concurrency with WAL mode
- BUG-003: Connection leak with Storage singleton
