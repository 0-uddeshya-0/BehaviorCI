"""Tests for the ``behaviorci`` command-line interface."""

import numpy as np
from typer.testing import CliRunner

from behaviorci.cli import app
from behaviorci.storage import get_storage

runner = CliRunner()


def _seed(db_path: str):
    store = get_storage(db_path)
    store.save_snapshot(
        behavior_id="greet",
        input_json="{}",
        output_text="hello",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        model_name="m",
    )
    return store


class TestInfoCommands:
    def test_help_lists_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ["record", "check", "update", "record-missing", "stats", "clear", "history"]:
            assert command in result.stdout

    def test_stats(self, tmp_path):
        db = str(tmp_path / "s.db")
        _seed(db)
        result = runner.invoke(app, ["stats", "--db", db])
        assert result.exit_code == 0
        assert "Total Snapshots:" in result.stdout
        assert "greet" in result.stdout  # per-behavior table

    def test_history_missing_behavior(self, tmp_path):
        db = str(tmp_path / "s.db")
        get_storage(db)  # create empty database
        result = runner.invoke(app, ["history", "ghost", "--db", db])
        assert result.exit_code == 1
        assert "No snapshots found" in result.stdout

    def test_history_shows_scores(self, tmp_path):
        db = str(tmp_path / "s.db")
        store = _seed(db)
        snapshot = store.get_all_snapshots_for_behavior("greet")[0]
        store.record_similarity(snapshot.id, 0.9100)
        store.record_similarity(snapshot.id, 0.8800)
        result = runner.invoke(app, ["history", "greet", "--db", db])
        assert result.exit_code == 0
        assert "greet" in result.stdout
        assert "0.9100" in result.stdout

    def test_clear_force_removes_database(self, tmp_path):
        db = str(tmp_path / "s.db")
        _seed(db)
        assert (tmp_path / "s.db").exists()
        result = runner.invoke(app, ["clear", "--force", "--db", db])
        assert result.exit_code == 0
        assert not (tmp_path / "s.db").exists()


class TestPytestWrappers:
    """The record/check/update commands shell out to pytest; verify the args."""

    def _capture(self, monkeypatch):
        captured = {}

        def fake_run(cmd, check=True):
            captured["cmd"] = cmd

            class Result:
                returncode = 0

            return Result()

        monkeypatch.setattr("behaviorci.cli.subprocess.run", fake_run)
        return captured

    def test_record(self, monkeypatch):
        captured = self._capture(monkeypatch)
        result = runner.invoke(app, ["record", "tests/", "--db", "x.db"])
        assert result.exit_code == 0
        assert "--behaviorci-record" in captured["cmd"]
        assert "tests/" in captured["cmd"]
        assert "--behaviorci-db" in captured["cmd"] and "x.db" in captured["cmd"]

    def test_check(self, monkeypatch):
        captured = self._capture(monkeypatch)
        result = runner.invoke(app, ["check", "tests/"])
        assert result.exit_code == 0
        assert "--behaviorci" in captured["cmd"]

    def test_update(self, monkeypatch):
        captured = self._capture(monkeypatch)
        result = runner.invoke(app, ["update", "tests/"])
        assert result.exit_code == 0
        assert "--behaviorci-update" in captured["cmd"]

    def test_record_missing(self, monkeypatch):
        captured = self._capture(monkeypatch)
        result = runner.invoke(app, ["record-missing", "tests/"])
        assert result.exit_code == 0
        assert "--behaviorci-record-missing" in captured["cmd"]
