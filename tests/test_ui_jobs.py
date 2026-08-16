"""The job supervisor, driven by scripted fake workers.

The worker command is injectable so these tests are deterministic and instant: a scripted
child emits an exact event sequence, or misbehaves in an exact way, without sampling a
single shot. `tests/test_ui_worker.py` covers the real worker separately.

Most of these assert the same property from different angles — **a job always reaches a
terminal state**. A run stuck on "running" forever is the worst failure this component
has, because polling it looks identical to waiting.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from qecgen.run import GenerateSpec, SweepSpec
from qecgen.ui.jobs import JobRecord, JobStatus, JobStore

TERMINAL = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


def spec(tmp_path: Path) -> GenerateSpec:
    return GenerateSpec(
        distance=3, p=0.01, shots=200, seed=1, out=tmp_path / "x.h5", chunk_size=100
    )


def scripted(*, body: str) -> tuple[str, ...]:
    """A worker command that runs ``body`` instead of generating anything."""
    return (sys.executable, "-c", body)


def settle(store: JobStore, job_id: str, timeout: float = 30.0) -> JobRecord:
    """Wait for a terminal state, failing the test rather than hanging the suite."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get(job_id)
        assert record is not None
        if record.status in TERMINAL:
            return record
        time.sleep(0.02)
    record = store.get(job_id)
    assert record is not None
    pytest.fail(f"job never reached a terminal state; stuck on {record.status}")


DONE = 'print(\'{"event": "done", "files": []}\')'


class TestHappyPath:
    def test_events_accumulate_into_the_record(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body=(
                    'print(\'{"event": "started", "total_units": 200, "unit": "shots"}\')\n'
                    'print(\'{"event": "phase", "phase": "sampling"}\')\n'
                    'print(\'{"event": "progress", "completed": 100}\')\n'
                    'print(\'{"event": "progress", "completed": 200}\')\n'
                    'print(\'{"event": "done", "files": [{"path": "x.h5", "shots": 200,'
                    ' "content_hash": "abc", "drift_condition": "not_applicable",'
                    ' "structure_source_environment_id": null}]}\')'
                )
            ),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.SUCCEEDED
        assert record.completed_units == 200
        assert record.progress_unit == "shots"
        assert record.phase == "sampling"
        assert record.files[0]["content_hash"] == "abc"

    def test_events_are_replayable_with_increasing_ids(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        events = store.events_since(job_id, 0)
        assert [event.id for event in events] == sorted(event.id for event in events)
        assert store.events_since(job_id, events[-1].id) == []
        # Reconnecting from a cursor must not replay what the client already has.
        assert all(event.id > 1 for event in store.events_since(job_id, 1))

    def test_warnings_are_kept(self, tmp_path: Path) -> None:
        # JSONLExporter warns above 100k shots. On a terminal that lands in front of the
        # user; through a pipe it would vanish unless the record keeps it.
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body='print(\'{"event": "warning", "message": "this file will be large"}\')\n'
                + DONE
            ),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.warnings == ["this file will be large"]


class TestAlwaysTerminal:
    """Every way a worker can misbehave still has to end the job."""

    def test_silent_exit(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body="pass"))
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.FAILED
        assert "without reporting a result" in (record.error or "")

    def test_crash_reports_its_stderr(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="import sys; sys.stderr.write('boom\\n'); sys.exit(3)"),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.FAILED
        assert "boom" in (record.error or "")

    def test_garbage_on_stdout_does_not_wedge_the_run(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="print('not json'); print('{broken')"),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.FAILED

    def test_garbage_before_a_valid_result_is_only_a_diagnostic(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="print('noise')\n" + DONE),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.SUCCEEDED

    def test_a_flood_on_stderr_cannot_block_the_child(self, tmp_path: Path) -> None:
        # An unread stderr pipe fills at 64 KB and the child blocks forever on its next
        # write, which presents as a run frozen mid-progress. 1.6 MB here.
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body=(
                    "import sys\nfor _ in range(20000): sys.stderr.write('x' * 80 + '\\n')\n" + DONE
                )
            ),
        )
        record = settle(store, store.submit(spec(tmp_path)).id, timeout=60)
        assert record.status is JobStatus.SUCCEEDED

    def test_a_flood_on_stdout_cannot_block_the_child(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body=(
                    "for i in range(20000):\n"
                    '    print(\'{"event": "progress", "completed": %d}\' % i)\n' + DONE
                )
            ),
        )
        record = settle(store, store.submit(spec(tmp_path)).id, timeout=60)
        assert record.status is JobStatus.SUCCEEDED

    def test_a_worker_that_cannot_start(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs", worker_command=("this-executable-does-not-exist-anywhere",)
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.FAILED
        assert "could not start worker" in (record.error or "")


class TestCancellation:
    def test_a_queued_job_cancels_without_starting(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="import time\nwhile True: time.sleep(1)"),
            max_concurrent=1,
            kill_grace_seconds=1.0,
        )
        first = store.submit(spec(tmp_path))
        queued = store.submit(spec(tmp_path))
        assert store.get(queued.id) is not None
        assert store.cancel(queued.id) is True
        assert settle(store, queued.id).status is JobStatus.CANCELLED
        store.cancel(first.id)
        settle(store, first.id)

    def test_a_worker_that_ignores_cancellation_is_killed(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="import time\nwhile True: time.sleep(1)"),
            kill_grace_seconds=1.0,
        )
        job_id = store.submit(spec(tmp_path)).id
        time.sleep(0.5)
        assert store.cancel(job_id) is True
        assert settle(store, job_id).status is JobStatus.CANCELLED

    def test_cancelling_a_finished_job_is_refused(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        assert store.cancel(job_id) is False

    def test_cancelling_an_unknown_job_is_false(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        assert store.cancel("nope") is False


class TestQueueing:
    def test_only_one_runs_at_a_time_by_default(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="import time; time.sleep(1.2)\n" + DONE),
            max_concurrent=1,
        )
        first = store.submit(spec(tmp_path))
        second = store.submit(spec(tmp_path))
        time.sleep(0.4)
        assert store.get(first.id).status is JobStatus.RUNNING  # type: ignore[union-attr]
        assert store.get(second.id).status is JobStatus.QUEUED  # type: ignore[union-attr]
        settle(store, second.id, timeout=60)


class TestProgressUnits:
    """A sweep counts tasks; a dataset run counts shots. The record has to say which.

    Nothing here samples anything — the scripted child emits the exact event sequence a
    real sweep worker would.
    """

    def sweep(self, tmp_path: Path) -> SweepSpec:
        return SweepSpec(
            distances=(3, 5),
            error_rates=(0.005, 0.01, 0.02),
            out=tmp_path / "sweeps" / "s.csv",
        )

    def test_a_sweep_is_denominated_in_tasks_before_it_starts(self, tmp_path: Path) -> None:
        # 2 distances x 3 rates x 1 decoder, known at submit time. `total_shots` has no
        # answer for a sweep, which is why the field is not called that.
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        record = store.submit(self.sweep(tmp_path))
        assert record.mode == "sweep"
        assert record.total_units == 6
        assert record.progress_unit == "tasks"

    def test_sweep_progress_carries_shots_and_sinter_status(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body=(
                    'print(\'{"event": "started", "total_units": 6, "unit": "tasks"}\')\n'
                    'print(\'{"event": "phase", "phase": "collecting"}\')\n'
                    'print(\'{"event": "progress", "completed": 4,'
                    ' "shots_collected": 91234, "detail": "2 tasks left"}\')\n'
                    'print(\'{"event": "done", "files": [{"kind": "sweep_results",'
                    ' "path": "s.csv"}, {"kind": "sweep_plot", "path": "s.png"}]}\')'
                )
            ),
        )
        record = settle(store, store.submit(self.sweep(tmp_path)).id)
        assert record.status is JobStatus.SUCCEEDED
        assert record.progress_unit == "tasks"
        assert record.shots_collected == 91234
        assert record.detail == "2 tasks left"
        assert [entry["kind"] for entry in record.files] == ["sweep_results", "sweep_plot"]

    def test_a_dataset_progress_event_does_not_blank_the_sweep_readouts(
        self, tmp_path: Path
    ) -> None:
        # A progress message without the sweep-only keys must leave the previous values
        # alone. Treating "absent" as "zero" would make the readout flicker to nothing
        # every time a message arrived without them.
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body=(
                    'print(\'{"event": "progress", "completed": 1,'
                    ' "shots_collected": 500, "detail": "half way"}\')\n'
                    'print(\'{"event": "progress", "completed": 2}\')\n' + DONE
                )
            ),
        )
        record = settle(store, store.submit(self.sweep(tmp_path)).id)
        assert record.shots_collected == 500
        assert record.detail == "half way"


class TestPersistence:
    def test_records_survive_a_restart(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        store = JobStore(runs, worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)

        revived = JobStore(runs, worker_command=scripted(body=DONE))
        revived.load_history()
        record = revived.get(job_id)
        assert record is not None
        assert record.status is JobStatus.SUCCEEDED
        assert record.spec["distance"] == 3

    def test_a_record_written_before_units_existed_still_loads(self, tmp_path: Path) -> None:
        """Progress fields were once named `total_shots`/`completed_shots`.

        A restart is exactly when someone goes looking for what already ran, so a record
        from before the rename has to survive it rather than vanish from history.
        """
        runs = tmp_path / "runs"
        runs.mkdir(parents=True)
        (runs / "old.json").write_text(
            '{"id": "old", "mode": "generate", "spec": {}, "status": "succeeded",'
            ' "total_shots": 5000, "completed_shots": 5000, "files": [], "warnings": []}',
            encoding="utf-8",
        )
        store = JobStore(runs, worker_command=scripted(body=DONE))
        store.load_history()
        record = store.get("old")
        assert record is not None
        assert record.status is JobStatus.SUCCEEDED
        assert record.total_units == 5000
        assert record.completed_units == 5000
        # Those old records could only ever have counted shots.
        assert record.progress_unit == "shots"

    def test_a_run_interrupted_by_a_restart_is_not_still_running(self, tmp_path: Path) -> None:
        # Its process died with the server; leaving the record on "running" would mean a
        # job that can never finish and can never be cancelled.
        runs = tmp_path / "runs"
        runs.mkdir(parents=True)
        (runs / "ghost.json").write_text(
            '{"id": "ghost", "mode": "generate", "spec": {}, "status": "running",'
            ' "total_shots": 10, "completed_shots": 5, "files": [], "warnings": []}',
            encoding="utf-8",
        )
        store = JobStore(runs, worker_command=scripted(body=DONE))
        store.load_history()
        record = store.get("ghost")
        assert record is not None
        assert record.status is JobStatus.FAILED
        assert "server stopped" in (record.error or "")
