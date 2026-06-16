"""End-to-end tests for the pytest plugin.

These run real pytest sessions in a temporary directory (via the ``pytester``
fixture) with a deterministic embedder injected, so they exercise the actual
record/compare/report flow without downloading a model.
"""

import json

from behaviorci.storage import Storage

# Injected into each temporary project so behavior tests run offline and
# deterministically: identical text always yields an identical embedding.
MOCK_CONFTEST = """
import hashlib
import numpy as np
import pytest
from behaviorci.embedder import set_embedder, reset_embedder


class _Mock:
    model_name = "mock-model"

    def embed_single(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.zeros(384, dtype=np.float32)
        for i in range(384):
            vec[i] = (digest[i % len(digest)] / 128.0) - 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def compute_similarity(self, a, b):
        return max(-1.0, min(1.0, float(np.dot(a, b))))


@pytest.fixture(autouse=True)
def _use_mock_embedder():
    set_embedder(_Mock())
    yield
    reset_embedder()
"""


def _snapshot_count(db_path) -> int:
    store = Storage(str(db_path))
    try:
        return store.get_stats()["snapshots"]
    finally:
        store.close()


class TestRecordAndCheck:
    def test_record_creates_snapshot(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("greet", threshold=0.85)
            def test_greet():
                return "Hello! How can I help you today?"
            """)
        db = pytester.path / "b.db"
        result = pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=1)
        assert _snapshot_count(db) == 1

    def test_check_passes_when_stable(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("greet", threshold=0.85)
            def test_greet():
                return "Hello! How can I help you today?"
            """)
        db = pytester.path / "b.db"
        pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        result = pytester.runpytest("--behaviorci", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=1)

    def test_check_fails_on_semantic_drift(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("answer", threshold=0.85)
            def test_answer():
                return "Your refund will arrive in 3-5 business days."
            """)
        pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))

        # Rewrite the test so it returns something unrelated, then check.
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("answer", threshold=0.85)
            def test_answer():
                return "Contact the finance department by fax."
            """)
        result = pytester.runpytest("--behaviorci", "--behaviorci-db", str(db))
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*BEHAVIORAL REGRESSION DETECTED*"])

    def test_update_overwrites_baseline(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("answer", threshold=0.85)
            def test_answer():
                return "Version one of the answer."
            """)
        pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("answer", threshold=0.85)
            def test_answer():
                return "A completely rewritten second version."
            """)
        result = pytester.runpytest("--behaviorci-update", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=1)
        # After update, checking the new output passes.
        result = pytester.runpytest("--behaviorci", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=1)


class TestGuardrails:
    def test_must_contain_failure(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("policy", threshold=0.85, must_contain=["refund"])
            def test_policy():
                return "We cannot process that request."
            """)
        pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        result = pytester.runpytest("--behaviorci", "--behaviorci-db", str(db))
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*Missing required*"])

    def test_must_not_contain_failure(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("safety", threshold=0.85, must_not_contain=["password"])
            def test_safety():
                return "Here is your password: hunter2"
            """)
        pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        result = pytester.runpytest("--behaviorci", "--behaviorci-db", str(db))
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*Found forbidden*"])


class TestCollection:
    def test_duplicate_behavior_id_is_rejected(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile(
            test_a="""
            from behaviorci import behavior

            @behavior("dup", threshold=0.85)
            def test_a():
                return "one"
            """,
            test_b="""
            from behaviorci import behavior

            @behavior("dup", threshold=0.85)
            def test_b():
                return "two"
            """,
        )
        result = pytester.runpytest("--behaviorci", "--behaviorci-db", str(pytester.path / "b.db"))
        assert result.ret != 0
        result.stdout.fnmatch_lines(["*Duplicate behavior_id*"])

    def test_behavior_runs_exactly_once(self, pytester):
        # A previous design re-ran each test in a custom hook; guard against that.
        pytester.makeconftest(MOCK_CONFTEST)
        marker = pytester.path / "calls.txt"
        db = pytester.path / "b.db"
        pytester.makepyfile(
            "from behaviorci import behavior\n"
            "@behavior('once', threshold=0.85)\n"
            "def test_once():\n"
            f"    open({str(marker)!r}, 'a').write('x')\n"
            "    return 'hello'\n"
        )
        result = pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=1)
        assert marker.read_text() == "x"


class TestCiWorkflow:
    def test_record_missing_autorecords(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("fresh", threshold=0.85)
            def test_fresh():
                return "A brand new behavior with no baseline yet."
            """)
        # No snapshot exists; record-missing should create it instead of failing.
        result = pytester.runpytest("--behaviorci-record-missing", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=1)
        assert _snapshot_count(db) == 1

    def test_parametrize_creates_distinct_snapshots(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        pytester.makepyfile("""
            import pytest
            from behaviorci import behavior

            @pytest.mark.parametrize("word", ["alpha", "beta", "gamma"])
            @behavior("param", threshold=0.85)
            def test_param(word):
                return f"output for {word}"
            """)
        result = pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=3)
        assert _snapshot_count(db) == 3
        result = pytester.runpytest("--behaviorci", "--behaviorci-db", str(db))
        result.assert_outcomes(passed=3)


class TestJsonReport:
    def test_report_is_written_on_record(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        report = pytester.path / "report.json"
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("greet", threshold=0.85)
            def test_greet():
                return "Hello there, friend."
            """)
        result = pytester.runpytest(
            "--behaviorci-record",
            "--behaviorci-db",
            str(db),
            "--behaviorci-report",
            str(report),
        )
        result.assert_outcomes(passed=1)

        data = json.loads(report.read_text())
        assert data["schema"] == "behaviorci/report/v1"
        assert data["mode"] == "record"
        assert data["summary"]["recorded"] == 1
        assert data["results"][0]["behavior_id"] == "greet"

    def test_report_captures_similarity_on_check(self, pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        db = pytester.path / "b.db"
        report = pytester.path / "report.json"
        pytester.makepyfile("""
            from behaviorci import behavior

            @behavior("greet", threshold=0.85)
            def test_greet():
                return "Hello there, friend."
            """)
        pytester.runpytest("--behaviorci-record", "--behaviorci-db", str(db))
        result = pytester.runpytest(
            "--behaviorci", "--behaviorci-db", str(db), "--behaviorci-report", str(report)
        )
        result.assert_outcomes(passed=1)

        data = json.loads(report.read_text())
        assert data["mode"] == "check"
        entry = data["results"][0]
        assert entry["action"] == "checked"
        assert entry["passed"] is True
        assert entry["similarity"] > 0.99
