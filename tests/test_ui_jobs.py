"""The job supervisor, driven by scripted fake workers.

The worker command is injectable so these tests are deterministic and instant: a scripted
child emits an exact event sequence, or misbehaves in an exact way, without sampling a
single shot. `tests/test_ui_worker.py` covers the real worker separately.

Most of these assert the same property from different angles — **a job always reaches a
terminal state**. A run stuck on "running" forever is the worst failure this component
has, because polling it looks identical to waiting.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qecgen.run import GenerateSpec
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
                    'print(\'{"event": "started", "total_shots": 200}\')\n'
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
        assert record.completed_shots == 200
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


class TestProcessTree:
    """Killing a worker must kill what the worker started.

    Measured before this was fixed: a sweep worker hands its grid to sinter, which forces
    multiprocessing 'spawn' and runs its own pool. `Popen.kill()` reached exactly one
    process and left all three children alive, saturating a core each, indefinitely --
    because on Windows they are not in a job object and TerminateProcess does not walk
    a tree. A cancelled sweep would have leaked N busy processes every time.
    """

    def test_a_grandchild_does_not_survive_the_kill(self, tmp_path: Path) -> None:
        marker = tmp_path / "grandchild.pid"
        # A worker that spawns a long-lived child, records its pid, then hangs. Only the
        # tree kill reaches that child; `process.kill()` alone leaves it running.
        body = (
            "import subprocess, sys, time, pathlib\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time\\n"
            "while True: time.sleep(1)'])\n"
            f"pathlib.Path(r'{marker}').write_text(str(child.pid))\n"
            'print(\'{"event": "started", "total_shots": 1}\', flush=True)\n'
            "while True: time.sleep(1)\n"
        )
        store = JobStore(
            tmp_path / "runs", worker_command=scripted(body=body), kill_grace_seconds=1.0
        )
        job_id = store.submit(spec(tmp_path)).id

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.exists(), "the scripted worker never started its child"
        grandchild = int(marker.read_text())

        store.cancel(job_id)
        settle(store, job_id)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and _process_alive(grandchild):
            time.sleep(0.2)
        assert not _process_alive(grandchild), (
            f"pid {grandchild} outlived its worker; the kill did not walk the tree"
        )


if sys.platform == "win32":

    def _process_alive(pid: int) -> bool:
        """Whether ``pid`` is still running, without importing psutil."""
        found = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return str(pid) in found

else:

    def _process_alive(pid: int) -> bool:
        """Whether ``pid`` is still running, without importing psutil."""
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True


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


class TestResultPayload:
    """A job can produce a summary and non-dataset files, not only datasets."""

    def test_a_result_and_artifacts_reach_the_record(self, tmp_path: Path) -> None:
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body=(
                    'print(\'{"event": "done", "files": [],'
                    ' "artifacts": [{"path": "s.png", "kind": "plot", "size_bytes": 42}],'
                    ' "result": {"crossing_p": 0.008}}\')'
                )
            ),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.SUCCEEDED
        assert record.result == {"crossing_p": 0.008}
        assert record.artifacts == [{"path": "s.png", "kind": "plot", "size_bytes": 42}]
        assert record.files == [], "an artifact must never be filed as a dataset"

    def test_a_dataset_run_still_reports_no_result(self, tmp_path: Path) -> None:
        """The absent-field path: every existing worker emits `done` with `files` alone."""
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.result is None
        assert record.artifacts == []

    def test_a_non_object_result_is_dropped_rather_than_stored(self, tmp_path: Path) -> None:
        """`record.result` is typed as an object. A worker sending a string or a list is
        misbehaving, and storing it would push the type error out to whichever front end
        rendered it."""
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body='print(\'{"event": "done", "files": [], "result": "not an object"}\')'
            ),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.SUCCEEDED
        assert record.result is None

    def test_a_non_finite_number_in_a_result_never_reaches_the_record(self, tmp_path: Path) -> None:
        """`encode_line` refuses to *emit* Infinity, but `json.loads` happily *accepts*
        it, so the parent cannot assume the wire was clean. A record holding one serves
        invalid JSON to the browser and writes invalid JSON to its own durable record."""
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(
                body=(
                    'print(\'{"event": "done", "files": [],'
                    ' "result": {"lambda": Infinity, "nested": [NaN, 1.5]}}\')'
                )
            ),
        )
        record = settle(store, store.submit(spec(tmp_path)).id)
        assert record.status is JobStatus.SUCCEEDED
        assert record.result == {"lambda": None, "nested": [None, 1.5]}
        # The durable record must be readable by something other than Python.
        import json

        stored = json.loads((tmp_path / "runs" / f"{record.id}.json").read_text(encoding="utf-8"))
        assert stored["result"] == {"lambda": None, "nested": [None, 1.5]}

    def test_a_progress_unit_is_known_before_the_worker_starts(self, tmp_path: Path) -> None:
        """Set at submit, not on `started`. A queued job is visible in the browser before
        its worker exists, and a total with no unit is a number counting nothing."""
        store = JobStore(
            tmp_path / "runs", worker_command=scripted(body="import time; time.sleep(5)")
        )
        record = store.submit(spec(tmp_path))
        assert record.status is JobStatus.QUEUED or record.status is JobStatus.RUNNING
        assert record.progress_unit == "shots"
        assert record.total_shots == 200
        store.cancel(record.id)
        settle(store, record.id)


class TestPersistence:
    def test_a_record_written_before_the_new_fields_still_loads(self, tmp_path: Path) -> None:
        """`progress_unit`, `artifacts` and `result` were added to a record that is already
        on disk in every user's data root. Requiring them would make the first restart
        after an upgrade drop the entire run history."""
        runs = tmp_path / "runs"
        runs.mkdir(parents=True)
        (runs / "old.json").write_text(
            '{"id": "old", "mode": "generate", "spec": {"distance": 3},'
            ' "status": "succeeded", "total_shots": 10, "completed_shots": 10,'
            ' "files": [], "warnings": []}',
            encoding="utf-8",
        )
        store = JobStore(runs, worker_command=scripted(body=DONE))
        store.load_history()
        record = store.get("old")
        assert record is not None
        assert record.status is JobStatus.SUCCEEDED
        assert record.progress_unit == "shots", "an old record was counting shots"
        assert record.artifacts == []
        assert record.result is None

    def test_a_hand_edited_result_costs_its_own_entry_only(self, tmp_path: Path) -> None:
        """The lesson `JobStatus()` already taught here: a coercion outside the guard took
        the whole server down with the FastAPI lifespan."""
        runs = tmp_path / "runs"
        runs.mkdir(parents=True)
        (runs / "weird.json").write_text(
            '{"id": "weird", "mode": "generate", "spec": {}, "status": "succeeded",'
            ' "total_shots": 1, "completed_shots": 1, "files": [], "warnings": [],'
            ' "result": ["not", "an", "object"]}',
            encoding="utf-8",
        )
        store = JobStore(runs, worker_command=scripted(body=DONE))
        store.load_history()
        record = store.get("weird")
        assert record is not None
        assert record.result is None

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
