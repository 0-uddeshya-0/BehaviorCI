# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-31

### Added
- Initial release of BehaviorCI
- pytest-native behavioral regression testing for LLM applications
- SQLite storage with WAL mode for concurrent writes
- Variance-aware thresholds (high variance → lower threshold)
- `@behavior` decorator with return value capture
- CLI commands: `record`, `check`, `update`, `record-missing`, `stats`, `clear`
- Thread-safe singleton patterns for Storage and Embedder
- Duplicate behavior_id validation at collection time
- Automatic snapshot recording for missing tests (`--behaviorci-record-missing`)

### Fixed
- CRITICAL-001: Double test execution in pytest hook (now uses hookwrapper)
- CRITICAL-002: Backwards variance threshold logic (now uses min() correctly)
- BUG-001: datetime import location (moved to top of file)
- BUG-002: SQLite concurrency with WAL mode
- BUG-003: Connection leak with Storage singleton

### Security
- Thread-safe singleton implementations for Storage and Embedder
- Proper connection management to prevent resource exhaustion
