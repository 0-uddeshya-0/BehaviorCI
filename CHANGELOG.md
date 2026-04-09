# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-alpha] - Unreleased

This release focuses on enterprise scalability, stripping out heavy dependencies, and adding robust support for highly creative, non-deterministic LLM prompts.

### Added
- **API Embedder Injection:** Introduced the `Embedder` Abstract Base Class and `set_embedder()` interface. You can now inject lightweight APIs (OpenAI, Gemini, Cohere) directly via `conftest.py` without installing local ML models.
- **Centroid Baselines (`samples` parameter):** Added multi-run sampling to the `@behavior` decorator. BehaviorCI can now execute highly creative prompts multiple times, compute the average embedding (Centroid), and evaluate drift against the true mathematical center of the outputs.
- **Async Test Support:** The `@behavior` decorator now natively supports `async def` test functions (e.g., LangChain or raw `asyncio` LLM calls).

### Changed
- **Massive Dependency Reduction:** `sentence-transformers` and PyTorch have been removed from the core installation, shrinking the baseline CI footprint by ~1GB. Local models are now an optional extra: `pip install behaviorci[local]`.
- **Mandatory Output Review:** To prevent the accidental commitment of LLM hallucinations as ground truth, `--behaviorci-record` now forces the primary generated text into the terminal stdout for explicit developer review.

### Fixed
- **Concurrent Execution Lockups:** Hardened the SQLite storage engine with Write-Ahead Logging (WAL) and thread-local connection pooling, fully resolving `database is locked` timeouts when using `pytest-xdist`.
- **Variance Calculation:** Refactored the variance-aware threshold formula to strictly enforce the user's base threshold as a ceiling, preventing the system from becoming overly permissive on low-variance structural outputs.
- **Model Mismatch Evaluation:** Cross-model comparisons (e.g., evaluating an `all-MiniLM-L6-v2` snapshot against an `OpenAI` embedding) now immediately raise a strict `ModelMismatchError` rather than failing silently on mathematically invalid vector comparisons.

## [0.1.0] - Initial Release
- Initial MVP release.
- Core `@behavior` decorator and Pytest plugin integration.
- Semantic similarity engine using `sentence-transformers`.
- Baseline recording and updating mechanism via SQLite.
