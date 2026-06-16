# Contributing to BehaviorCI

Thanks for taking the time to contribute. This guide covers how to get set up,
the standards the codebase follows, and how to get a change merged.

## Getting set up

You'll need Python 3.10+ and Git.

```bash
git clone https://github.com/0-uddeshya-0/BehaviorCI.git
cd BehaviorCI

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev,local]"      # dev tools + the local embedding model
pytest                             # should be green
```

The `[local]` extra installs `sentence-transformers` (and PyTorch). It's only
needed for the handful of tests that exercise the real model — everything else
runs against a deterministic mock and is fast.

### Pre-commit hooks (recommended)

```bash
pip install pre-commit
pre-commit install
```

The hooks run Black, isort, flake8, and mypy.

## Project layout

```
src/behaviorci/
├── __init__.py      # public API
├── api.py           # the @behavior decorator and input serialization
├── plugin.py        # pytest integration (record / compare / report)
├── comparator.py    # lexical guards, similarity, variance thresholds
├── embedder.py      # local model wrapper + injectable BaseEmbedder
├── storage.py       # SQLite snapshots and history
├── models.py        # pydantic models
├── exceptions.py    # error types
└── cli.py           # the `behaviorci` command

tests/
├── support.py       # MockEmbedder used across the suite
├── test_api.py      # decorator + serialization
├── test_storage.py  # storage, singletons, concurrency
├── test_embedder.py # embedder selection, injection, local model
├── test_comparator.py
├── test_plugin.py   # end-to-end plugin behavior via pytester
├── test_cli.py
└── examples/        # runnable example tests (also dogfood the plugin)
```

## Standards

### Formatting and types

- **Black** (line length 100) and **isort** (`profile = black`).
- **flake8** must pass (config in `.flake8`).
- **mypy** runs in strict mode (`disallow_untyped_defs`); all public functions
  are typed.

```bash
black src/ tests/ conftest.py
isort src/ tests/ conftest.py
flake8 src/ tests/ conftest.py
mypy src/behaviorci/ --ignore-missing-imports
```

### Comments and docstrings

Explain *why*, not *what* — the code already says what it does. Keep docstrings
practical: a one-line summary, then `Args`/`Returns`/`Raises` only where they add
something. Avoid referencing internal ticket numbers in code; describe the
behavior instead.

### Tests

- New behavior needs tests. Bug fixes should come with a test that fails before
  the fix.
- Tests must be deterministic and pass under `pytest -n auto`.
- Use `tests.support.MockEmbedder` for fast, offline comparison tests:

    ```python
    from behaviorci.comparator import Comparator
    from behaviorci.storage import Storage
    from tests.support import MockEmbedder

    def test_identical_output_passes(tmp_path):
        comparator = Comparator(Storage(str(tmp_path / "c.db")), MockEmbedder())
        comparator.record_snapshot("greet", "{}", "hello there")
        assert comparator.compare("greet", "{}", "hello there", 0.85).passed
    ```

- Plugin behavior is tested end-to-end with pytest's `pytester` fixture (see
  `tests/test_plugin.py`) — that's the right place for anything touching the
  record/compare/report flow.

## Pull requests

1. Branch from `main` (`feature/…`, `fix/…`, `docs/…`).
2. Make the change, add tests, update docs (`README.md`, `docs/`, `CHANGELOG.md`)
   where user-facing behavior changes.
3. Run the full check locally:

    ```bash
    pytest
    black --check src/ tests/ conftest.py && isort --check-only src/ tests/ conftest.py
    flake8 src/ tests/ conftest.py && mypy src/behaviorci/ --ignore-missing-imports
    ```

4. Use [Conventional Commits](https://www.conventionalcommits.org/) for messages
   (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
5. Open the PR and link any related issue.

## Reporting bugs

Open an issue with the BehaviorCI version, Python version, OS, a minimal
reproduction, and the full error output. For behavior-comparison surprises,
include the `--behaviorci-report` JSON if you can — it captures the scores and
thresholds involved.

## Code of conduct

Be respectful, be constructive, and assume good faith. Critique ideas, not
people.

Thanks again — every fix, test, and doc improvement helps.
