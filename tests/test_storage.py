"""Tests for the SQLite storage layer."""

import multiprocessing
import sqlite3
import threading

import numpy as np
import pytest

from behaviorci.storage import (
    Storage,
    compute_snapshot_id,
    get_storage,
    reset_all_storage,
    reset_storage,
)


@pytest.fixture
def storage(tmp_path):
    store = Storage(str(tmp_path / "test.db"))
    yield store
    store.close()


class TestSnapshotId:
    def test_deterministic(self):
        a = compute_snapshot_id("behavior", '{"args": [], "kwargs": {}}')
        b = compute_snapshot_id("behavior", '{"args": [], "kwargs": {}}')
        assert a == b
        assert len(a) == 64  # SHA-256 hex

    def test_input_changes_id(self):
        a = compute_snapshot_id("behavior", '{"args": [1]}')
        b = compute_snapshot_id("behavior", '{"args": [2]}')
        assert a != b


class TestSnapshotCrud:
    def test_save_and_get(self, storage):
        snapshot_id = storage.save_snapshot(
            behavior_id="greet",
            input_json='{"args": [], "kwargs": {}}',
            output_text="hello",
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            model_name="test-model",
        )
        snapshot = storage.get_snapshot(snapshot_id)
        assert snapshot.behavior_id == "greet"
        assert snapshot.output_text == "hello"
        assert snapshot.model_name == "test-model"
        assert snapshot.created_at > 0

    def test_embedding_is_normalized(self, storage):
        snapshot_id = storage.save_snapshot(
            behavior_id="b",
            input_json="{}",
            output_text="t",
            embedding=np.array([3.0, 4.0], dtype=np.float32),  # norm 5
            model_name="m",
        )
        stored = storage.get_snapshot(snapshot_id).get_embedding_array()
        assert abs(np.linalg.norm(stored) - 1.0) < 1e-5

    def test_save_overwrites_existing(self, storage):
        input_json = '{"args": [], "kwargs": {}}'
        id1 = storage.save_snapshot(
            behavior_id="b",
            input_json=input_json,
            output_text="original",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="m",
        )
        id2 = storage.save_snapshot(
            behavior_id="b",
            input_json=input_json,
            output_text="updated",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            model_name="m",
        )
        assert id1 == id2
        assert storage.get_snapshot(id1).output_text == "updated"

    def test_find_snapshot_missing_returns_none(self, storage):
        assert storage.find_snapshot("nope", "{}") is None

    def test_delete_snapshot(self, storage):
        snapshot_id = storage.save_snapshot(
            behavior_id="b",
            input_json="{}",
            output_text="t",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="m",
        )
        assert storage.delete_snapshot(snapshot_id) is True
        assert storage.delete_snapshot(snapshot_id) is False


class TestHistory:
    def test_record_and_read(self, storage):
        snapshot_id = storage.save_snapshot(
            behavior_id="b",
            input_json="{}",
            output_text="t",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="m",
        )
        for sim in (0.95, 0.92, 0.94):
            storage.record_similarity(snapshot_id, sim)
        history = storage.get_similarity_history(snapshot_id)
        assert sorted(history) == [0.92, 0.94, 0.95]

    def test_history_isolated_per_snapshot(self, storage):
        id1 = storage.save_snapshot(
            behavior_id="same",
            input_json='{"args": [1]}',
            output_text="o1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="m",
        )
        id2 = storage.save_snapshot(
            behavior_id="same",
            input_json='{"args": [2]}',
            output_text="o2",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            model_name="m",
        )
        storage.record_similarity(id1, 0.99)
        storage.record_similarity(id2, 0.50)
        assert storage.get_similarity_history(id1) == [0.99]
        assert storage.get_similarity_history(id2) == [0.50]

    def test_history_with_timestamps(self, storage):
        snapshot_id = storage.save_snapshot(
            behavior_id="b",
            input_json="{}",
            output_text="t",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="m",
        )
        storage.record_similarity(snapshot_id, 0.9)
        rows = storage.get_similarity_history_with_timestamps(snapshot_id)
        assert len(rows) == 1
        similarity, timestamp = rows[0]
        assert similarity == 0.9
        assert timestamp > 0


class TestStats:
    def test_stats_and_summary(self, storage):
        empty = storage.get_stats()
        assert empty["snapshots"] == 0 and empty["behaviors"] == 0

        for behavior_id, idx in [("b1", 1), ("b1", 2), ("b2", 1)]:
            storage.save_snapshot(
                behavior_id=behavior_id,
                input_json=f'{{"args": [{idx}]}}',
                output_text="o",
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                model_name="m",
            )

        stats = storage.get_stats()
        assert stats["snapshots"] == 3
        assert stats["behaviors"] == 2

        summary = dict((row[0], row[1]) for row in storage.get_behavior_summary())
        assert summary == {"b1": 2, "b2": 1}


class TestSingleton:
    def test_same_path_returns_same_instance(self, tmp_path):
        db = str(tmp_path / "s.db")
        assert get_storage(db) is get_storage(db)

    def test_different_paths_differ(self, tmp_path):
        assert get_storage(str(tmp_path / "a.db")) is not get_storage(str(tmp_path / "b.db"))

    def test_default_path_is_singleton(self):
        assert get_storage(None) is get_storage(None)

    def test_reset_storage(self, tmp_path):
        db = str(tmp_path / "s.db")
        first = get_storage(db)
        reset_storage(db)
        assert get_storage(db) is not first

    def test_reset_all_storage(self, tmp_path):
        a, b = str(tmp_path / "a.db"), str(tmp_path / "b.db")
        first_a, first_b = get_storage(a), get_storage(b)
        reset_all_storage()
        assert get_storage(a) is not first_a
        assert get_storage(b) is not first_b

    def test_concurrent_get_storage_is_consistent(self, tmp_path):
        db = str(tmp_path / "s.db")
        seen = []

        def grab():
            seen.append(get_storage(db))

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(s is seen[0] for s in seen)


class TestConcurrency:
    def test_wal_mode_enabled(self, tmp_path):
        db = str(tmp_path / "s.db")
        Storage(db)
        conn = sqlite3.connect(db)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode.upper() == "WAL"

    def test_busy_timeout_set(self, tmp_path):
        db = str(tmp_path / "s.db")
        Storage(db)
        conn = sqlite3.connect(db)
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.close()
        assert timeout >= 5000

    def test_synchronous_normal_on_connections(self, storage):
        storage.save_snapshot(
            behavior_id="b",
            input_json="{}",
            output_text="t",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="m",
        )
        conn = storage._get_connection()
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL

    def test_thread_local_writes(self, tmp_path):
        store = get_storage(str(tmp_path / "s.db"))
        done = []

        def write(thread_id):
            for i in range(5):
                store.save_snapshot(
                    behavior_id=f"thread_{thread_id}",
                    input_json=f'{{"idx": {i}}}',
                    output_text="o",
                    embedding=np.array([0.1, 0.2], dtype=np.float32),
                    model_name="m",
                )
            store.close()  # release this thread's connection before it exits
            done.append(thread_id)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(done) == 4
        assert store.get_stats()["snapshots"] == 20

    @pytest.mark.slow
    def test_concurrent_writes_no_lock_errors(self, tmp_path):
        db = str(tmp_path / "s.db")
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(4) as pool:
            results = pool.starmap(_write_worker, [(db, f"proc_{i}") for i in range(4)])
        assert all(results)
        store = Storage(db)
        assert store.get_stats()["snapshots"] == 40
        store.close()


class TestInMemory:
    def test_memory_db_does_not_create_files(self, tmp_path, monkeypatch):
        # Run inside an empty cwd so a stray ":memory:" file would be obvious.
        monkeypatch.chdir(tmp_path)
        store = Storage(":memory:")
        store.save_snapshot(
            behavior_id="b",
            input_json="{}",
            output_text="t",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="m",
        )
        assert store.get_stats()["snapshots"] == 1
        assert not any(p.name.startswith(":memory:") for p in tmp_path.iterdir())
        store.close()


def _write_worker(db_path: str, behavior_id: str, num_writes: int = 10) -> bool:
    """Worker for the multiprocess concurrency test (must be importable for spawn)."""
    store = Storage(db_path)
    for i in range(num_writes):
        store.save_snapshot(
            behavior_id=behavior_id,
            input_json=f'{{"idx": {i}}}',
            output_text=f"output {i}",
            embedding=np.random.randn(384).astype(np.float32),
            model_name="m",
        )
    return True
