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

from qecgen import deletion
from qecgen.run import GenerateSpec, SweepSpec
from qecgen.ui.jobs import (
    JobRecord,
    JobStatus,
    JobStore,
    RunNotFinishedError,
    run_input_paths,
    run_output_paths,
)

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
            'print(\'{"event": "started", "total_units": 1}\', flush=True)\n'
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
        assert record.total_units == 200
        store.cancel(record.id)
        settle(store, record.id)


@pytest.fixture
def trashed(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record what would have gone to the recycle bin, and unlink it instead.

    Named rather than autouse: a fixture that silently disables the real path everywhere is
    how the real path stops being covered at all.
    """
    seen: list[Path] = []

    def fake(path: Path) -> None:
        seen.append(Path(path))
        Path(path).unlink()

    monkeypatch.setattr(deletion, "send_to_trash", fake)
    return seen


class TestDiscard:
    """Forgetting a finished run, in memory and on disk."""

    def test_a_finished_run_is_forgotten_in_memory_and_on_disk(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        record_path = tmp_path / "runs" / f"{job_id}.json"
        assert record_path.is_file()

        outcome = store.discard(job_id)
        assert outcome is not None
        assert outcome.removed is True
        assert outcome.problem is None
        assert store.get(job_id) is None
        assert store.records() == []
        assert not record_path.exists()

    def test_a_discarded_run_does_not_come_back_on_restart(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        """The only assertion that can see the real bug.

        `store.get(job_id) is None` on the *first* store passes against an implementation
        that pops `_jobs` and leaves the JSON behind -- and `load_history` re-adopts it on
        the next `qecgen ui`, so the user deletes a run, restarts, and it is back. A second
        store reading the same directory is what catches that.
        """
        runs = tmp_path / "runs"
        store = JobStore(runs, worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        store.discard(job_id)

        restarted = JobStore(runs, worker_command=scripted(body=DONE))
        restarted.load_history()
        assert restarted.records() == []

    def test_the_order_list_is_kept_in_step(self, tmp_path: Path, trashed: list[Path]) -> None:
        """A `del self._jobs[id]` without `_order.remove` makes `records()` raise KeyError.

        A test that only calls `get()` never notices, and the real symptom is `_pump` dying
        on a daemon thread so the queue never starts another job.
        """
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        first = store.submit(spec(tmp_path)).id
        settle(store, first)
        second = store.submit(spec(tmp_path)).id
        settle(store, second)
        third = store.submit(spec(tmp_path)).id
        settle(store, third)

        store.discard(second)
        assert [record.id for record in store.records()] == [third, first]

    def test_an_unfinished_run_is_refused(self, tmp_path: Path) -> None:
        """The plausible bug is a partial discard that pops memory and *then* raises, so
        asserting only the raise is not enough."""
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="import time\nwhile True: time.sleep(1)"),
        )
        job_id = store.submit(spec(tmp_path)).id
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            record = store.get(job_id)
            assert record is not None
            if record.status is JobStatus.RUNNING:
                break
            time.sleep(0.02)

        with pytest.raises(RunNotFinishedError):
            store.discard(job_id)
        assert store.get(job_id) is not None
        assert (tmp_path / "runs" / f"{job_id}.json").is_file()
        store.cancel(job_id)
        settle(store, job_id)
        store.shutdown()

    def test_a_queued_run_is_refused_too(self, tmp_path: Path) -> None:
        """Covers the non-terminal branch that is not RUNNING."""
        store = JobStore(
            tmp_path / "runs",
            worker_command=scripted(body="import time\nwhile True: time.sleep(1)"),
            max_concurrent=1,
        )
        first = store.submit(spec(tmp_path)).id
        second = store.submit(spec(tmp_path)).id
        queued = store.get(second)
        assert queued is not None
        assert queued.status is JobStatus.QUEUED

        with pytest.raises(RunNotFinishedError):
            store.discard(second)
        store.cancel(first)
        store.cancel(second)
        store.shutdown()

    def test_an_unknown_id_returns_none(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        assert store.discard("nosuchrun") is None

    def test_a_record_file_already_gone_is_not_a_problem(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        (tmp_path / "runs" / f"{job_id}.json").unlink()

        outcome = store.discard(job_id)
        assert outcome is not None
        assert outcome.removed is True
        assert outcome.problem is None

    def test_a_record_that_could_not_be_removed_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`load_history` re-adopts a surviving record, so reporting a clean delete here
        would be a well-formed claim about something that comes back next restart."""
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)

        def refuse(path: Path) -> None:
            raise PermissionError(13, "in use", str(path))

        monkeypatch.setattr(deletion, "send_to_trash", refuse)
        outcome = store.discard(job_id)
        assert outcome is not None
        assert outcome.removed is False
        assert outcome.problem is not None
        assert store.get(job_id) is None


class TestRunOutputs:
    """Which paths belong to a run, and which emphatically do not."""

    def test_only_produced_files_are_listed(self) -> None:
        """The catastrophic case, named at the function that owns the rule.

        A `score` run produces nothing and names the user's own training set in its spec.
        A deletion that walked `spec` would bin the input of every analysis run in history.
        """
        record = JobRecord(
            id="a" * 12,
            mode="score",
            spec={"dataset": "/x/train.h5", "correction": "/x/c.npz"},
            total_units=0,
        )
        assert run_output_paths(record) == []
        assert [p.name for p in run_input_paths(record)] == ["train.h5", "c.npz"]

    def test_a_sweeps_three_artifacts_are_all_returned(self) -> None:
        record = JobRecord(
            id="b" * 12,
            mode="sweep",
            spec={"out": "/x/s.csv"},
            total_units=0,
            artifacts=[
                {"path": "/x/s.csv", "kind": "results table", "size_bytes": 1},
                {"path": "/x/s.png", "kind": "plot", "size_bytes": 1},
                {"path": "/x/s.threshold.json", "kind": "summary", "size_bytes": 1},
            ],
        )
        assert [p.name for p in run_output_paths(record)] == [
            "s.csv",
            "s.png",
            "s.threshold.json",
        ]

    def test_a_legacy_record_with_null_lists_is_tolerated(self) -> None:
        """Records written before the analysis-job layer have no `artifacts` key at all, and
        a hand-edited or foreign one can carry an explicit `null`. Neither may raise on the
        path of a delete: `list(None)` is a TypeError, and this runs while the user is
        looking at a confirmation dialog."""
        record = JobRecord(id="c" * 12, mode="generate", spec={}, total_units=0)
        record.artifacts = None  # type: ignore[assignment]
        record.files = None  # type: ignore[assignment]
        assert run_output_paths(record) == []


class TestOrphanedRuns:
    """Which run records a dataset deletion leaves describing nothing."""

    def test_a_run_whose_only_output_is_deleted_is_orphaned(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        record = store.get(job_id)
        assert record is not None
        target = tmp_path / "x.h5"
        target.write_text("x", encoding="utf-8")
        record.files = [
            {
                "path": str(target),
                "shots": 200,
                "content_hash": "abc",
                "drift_condition": "not_applicable",
                "structure_source_environment_id": None,
            }
        ]
        assert [r.id for r in store.orphaned_runs([target])] == [job_id]

    def test_an_analysis_run_is_never_orphaned_by_a_dataset_delete(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        """A score result is a true statement about a run that happened. The file it read
        going away does not make it false, and removing the record would delete history the
        user never named."""
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        record = store.get(job_id)
        assert record is not None
        record.mode = "score"
        record.spec = {"dataset": str(tmp_path / "gone.h5")}
        record.files = []
        record.artifacts = []
        assert store.orphaned_runs([tmp_path / "gone.h5"]) == []

    def test_a_run_with_a_surviving_output_is_not_orphaned(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        """Partial coverage must not orphan: a drift run whose training file went but whose
        test files remain still describes files that are there."""
        store = JobStore(tmp_path / "runs", worker_command=scripted(body=DONE))
        job_id = store.submit(spec(tmp_path)).id
        settle(store, job_id)
        record = store.get(job_id)
        assert record is not None
        kept = tmp_path / "kept.h5"
        kept.write_text("x", encoding="utf-8")
        removed = tmp_path / "removed.h5"
        record.files = [
            {
                "path": str(removed),
                "shots": 1,
                "content_hash": None,
                "drift_condition": "not_applicable",
                "structure_source_environment_id": None,
            },
            {
                "path": str(kept),
                "shots": 1,
                "content_hash": None,
                "drift_condition": "not_applicable",
                "structure_source_environment_id": None,
            },
        ]
        assert store.orphaned_runs([removed]) == []
