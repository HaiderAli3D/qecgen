"""The real worker, spawned as a real subprocess.

`tests/test_ui_jobs.py` covers the supervisor with scripted children; this file covers the
child itself, because the protocol has two sides and a scripted fake can agree with a
supervisor that both misread the contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import get_args

import pytest

from qecgen.dataset import DriftCondition
from qecgen.exporters import get_exporter
from qecgen.run import (
    PARTIAL_PREFIX,
    DriftSpec,
    GenerateSpec,
    JobSpec,
    MultiEnvSpec,
    QaSpec,
    ScoreSpec,
    SweepSpec,
    preload,
)
from qecgen.ui.protocol import MODES, encode_line, spec_from_json, spec_to_json
from qecgen.ui.worker import LineReader
from qecgen.validate import validate_dataset

WORKER = (sys.executable, "-m", "qecgen.ui.worker")


def drive(
    spec: JobSpec, cancel_after_events: int | None = None
) -> tuple[list[dict[str, object]], int]:
    """Run the worker over pipes and collect its events.

    Three guards keep a worker regression a test *failure* rather than a suite hang:
    stderr is drained on a thread (an unread pipe fills — 64 KB on Windows — and the
    child then blocks forever on its next write); a watchdog kills the child if the
    stdout loop outlives its deadline (killing closes the pipe, which ends the loop);
    and the ``finally`` always reaps the process. The earlier version had none of
    these, so a worker that flooded stderr or fell silent deadlocked the fast suite
    indefinitely instead of failing it.
    """
    process = subprocess.Popen(
        list(WORKER),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    assert process.stderr is not None
    stderr_tail: list[str] = []
    drain = threading.Thread(target=lambda: stderr_tail.extend(process.stderr or ()), daemon=True)
    drain.start()
    watchdog = threading.Timer(120.0, process.kill)
    watchdog.daemon = True
    watchdog.start()

    events: list[dict[str, object]] = []
    try:
        process.stdin.write(encode_line(spec_to_json(spec)))
        process.stdin.flush()
        sent = False
        for line in process.stdout:
            if line.strip():
                events.append(json.loads(line))
            if cancel_after_events is not None and not sent and len(events) >= cancel_after_events:
                process.stdin.write(encode_line({"cancel": True}))
                process.stdin.flush()
                sent = True
        code = process.wait(timeout=30)
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdin.close()
    return events, code


class TestSpecRoundTrip:
    """Every member of `JobSpec` must survive the wire, and be checked automatically.

    `spec_from_json` is an if-chain ending in a raise, so mypy cannot check it for
    exhaustiveness the way it checks `mode_of` and `spec_to_json`. Driving the cases from
    `get_args(JobSpec)` covers a new member the day it is added, which is stronger than
    the type checker can be here — a spec that encodes but does not decode reaches the
    user as "could not read spec" from inside a subprocess.
    """

    @staticmethod
    def _example(spec_type: type) -> object:
        examples: dict[str, object] = {
            "GenerateSpec": GenerateSpec(
                distance=3, p=0.01, shots=10, seed=1, out=Path("a.h5"), chunk_size=10
            ),
            "MultiEnvSpec": MultiEnvSpec(
                distance=3,
                axis_values=(0.01, 0.02),
                shots_per_env=5,
                seed=1,
                out=Path("b.h5"),
                shuffle=False,
            ),
            "DriftSpec": DriftSpec(
                distance=3,
                train_p=0.005,
                test_values=(0.01,),
                shots=5,
                seed=1,
                condition=DriftCondition.FROZEN_PRIOR,
                out=Path("d"),
                emit_mechanisms=True,
            ),
            "ScoreSpec": ScoreSpec(
                dataset=Path("a.h5"), correction=Path("c.npz"), unpacked=True, alpha=0.01
            ),
            "QaSpec": QaSpec(dataset=Path("a.h5"), max_shots=1_000, target_errors=10),
            "SweepSpec": SweepSpec(
                distances=(3, 5),
                error_rates=(0.005, 0.01),
                out=Path("sweeps/s.csv"),
                decoders=("pymatching",),
                workers=2,
            ),
        }
        example = examples.get(spec_type.__name__)
        assert example is not None, (
            f"{spec_type.__name__} joined JobSpec with no round-trip example here"
        )
        return example

    def test_every_job_spec_survives_the_wire(self) -> None:
        members = get_args(JobSpec)
        assert len(members) >= 4
        for spec_type in members:
            original = self._example(spec_type)
            payload = spec_to_json(original)  # type: ignore[arg-type]
            assert payload["mode"] in MODES
            assert spec_from_json(json.loads(json.dumps(payload))) == original

    def test_an_unknown_mode_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="unknown mode"):
            spec_from_json({"mode": "telepathy"})


class TestLineReader:
    def test_reads_lines_and_reports_end_of_input(self, tmp_path: Path) -> None:
        path = tmp_path / "in.txt"
        path.write_text("one\ntwo\n", encoding="utf-8")
        with path.open("rb") as handle:
            reader = LineReader(handle.fileno())
            assert reader.readline() == "one"
            assert reader.readline() == "two"
            assert reader.readline() is None

    def test_a_final_line_without_a_newline_is_still_returned(self, tmp_path: Path) -> None:
        path = tmp_path / "in.txt"
        path.write_text("only", encoding="utf-8")
        with path.open("rb") as handle:
            assert LineReader(handle.fileno()).readline() == "only"


class TestHeavyImportsHappenBeforeTheWatcherParks:
    """A worker must finish its imports before it starts watching stdin for a cancel.

    Measured on Windows: a process with **any** thread blocked reading stdin cannot
    afterwards complete a large DLL-loading import on the main thread. `import
    scipy.linalg` never returns, while `import decimal` and `import xml.dom.minidom` are
    unaffected, and raw `os.read` and `sys.stdin.readline` deadlock identically. Closing
    stdin lets the same import finish in 1.5 s.

    Generation never hit it because `qecgen.run` imports everything it needs at module
    scope. Scoring imported `qecgen.correction` lazily inside `analyse`, i.e. after the
    watcher had parked, and the job hung with **no output at all** -- no `started`, no
    error, no exit. The supervisor can only report that as a run that never finished, and
    a 10-second force-kill does not apply because nothing was ever cancelled.

    `drive` holds stdin open for the whole run, which is precisely the trigger.
    """

    def test_every_job_spec_preloads_what_it_needs(self) -> None:
        """Structural half: `preload` must answer for every member of the union.

        The failure has no symptom other than silence, so an exhaustive match is the
        only thing standing between a new analysis kind and a hang.
        """
        for spec_type in get_args(JobSpec):
            preload(TestSpecRoundTrip._example(spec_type))  # type: ignore[arg-type]

    def test_a_sweep_completes_with_stdin_held_open(self, tmp_path: Path) -> None:
        """The second face of the same hazard, and the one with no import in sight.

        A sweep hands its grid to sinter, which runs a multiprocessing pool with start
        method 'spawn'. On Windows a spawned child inherits the standard handles --
        including the pipe this worker's cancel watcher is blocked reading -- and the
        children then never finish starting. Measured: the sweep stops at "Starting 2
        workers..." forever, while the identical sweep with stdin closed finishes in
        three seconds.

        `_private_stdin` is the fix: the watcher keeps a non-inheritable duplicate and
        fd 0 becomes the null device, so no child inherits the pipe. `drive` holds stdin
        open for the whole run, which is the trigger.
        """
        events, code = drive(
            SweepSpec(
                distances=(3,),
                error_rates=(0.01,),
                out=tmp_path / "sweep" / "s.csv",
                max_errors=5,
                max_shots=500,
                workers=2,
            )
        )
        assert code == 0, events
        assert events[-1]["event"] == "done", events
        result = events[-1]["result"]
        assert isinstance(result, dict)
        assert result["kind"] == "sweep"

        # All three files, committed together, and none of them filed as a dataset.
        assert events[-1]["files"] == []
        artifacts = events[-1]["artifacts"]
        assert isinstance(artifacts, list)
        assert {entry["kind"] for entry in artifacts} == {"results table", "plot", "summary"}
        for entry in artifacts:
            assert Path(entry["path"]).is_file()

    def test_a_score_job_completes_with_stdin_held_open(self, tmp_path: Path) -> None:
        """Behavioural half: the exact shape that used to hang, end to end."""
        import numpy as np

        dataset = tmp_path / "scored.h5"
        events, code = drive(
            GenerateSpec(distance=3, p=0.02, shots=64, seed=3, out=dataset, chunk_size=64)
        )
        assert code == 0, events

        correction = tmp_path / "identity.npz"
        np.savez(
            correction,
            correction_x=np.zeros((64, 2), dtype=np.uint8),
            correction_z=np.zeros((64, 2), dtype=np.uint8),
        )

        events, code = drive(ScoreSpec(dataset=dataset, correction=correction))
        assert code == 0, events
        assert events[-1]["event"] == "done", events
        result = events[-1]["result"]
        assert isinstance(result, dict)
        assert result["kind"] == "score"
        assert result["shots"] == 64
        assert result["n_data_qubits"] == 9


class TestRealWorker:
    def test_progress_accumulates_to_the_exact_total(self, tmp_path: Path) -> None:
        """The worker time-coalesces progress events, so a per-chunk cadence is NOT
        part of the contract — only that the last progress value is exact. The old name
        promised one-per-chunk, which the body never asserted."""
        spec = GenerateSpec(
            distance=3, p=0.01, shots=400, seed=1, out=tmp_path / "w.h5", chunk_size=100
        )
        events, code = drive(spec)
        kinds = [event["event"] for event in events]

        assert code == 0
        assert kinds[0] == "started"
        assert kinds[-1] == "done"
        assert events[0]["total_shots"] == 400
        # Progress carries the accumulated total, so coalescing can drop intermediate
        # messages without losing anything -- but the last one must be exact.
        progress = [event for event in events if event["event"] == "progress"]
        assert progress[-1]["completed"] == 400

        written = tmp_path / "w.h5"
        assert written.is_file()
        report = validate_dataset(get_exporter("hdf5").read(written))
        assert report.ok, [c.name for c in report.failures]

    def test_end_of_input_is_not_a_cancellation(self, tmp_path: Path) -> None:
        # Feeding a spec from a file closes stdin immediately. Treating that as "the
        # parent is gone" made the worker cancel itself before sampling a single shot.
        spec = GenerateSpec(
            distance=3, p=0.01, shots=200, seed=1, out=tmp_path / "f.h5", chunk_size=100
        )
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(encode_line(spec_to_json(spec)), encoding="utf-8")
        with spec_file.open("rb") as handle:
            result = subprocess.run(
                list(WORKER), stdin=handle, capture_output=True, text=True, timeout=120
            )
        assert result.returncode == 0, result.stdout + result.stderr
        assert '"event": "done"' in result.stdout
        assert (tmp_path / "f.h5").is_file()

    def test_bad_input_is_reported_as_input_not_a_crash(self, tmp_path: Path) -> None:
        spec = GenerateSpec(
            distance=3, p=1.5, shots=100, seed=1, out=tmp_path / "bad.h5", chunk_size=100
        )
        events, code = drive(spec)
        assert code == 1
        assert events[-1]["event"] == "error"
        assert events[-1]["kind"] == "input"
        assert not (tmp_path / "bad.h5").exists()

    def test_cancellation_leaves_no_file_and_no_staging(self, tmp_path: Path) -> None:
        spec = GenerateSpec(
            distance=5,
            p=0.01,
            shots=2_000_000,
            seed=1,
            out=tmp_path / "big.h5",
            chunk_size=25_000,
        )
        started = time.monotonic()
        events, code = drive(spec, cancel_after_events=3)
        assert code == 2
        assert events[-1]["event"] == "cancelled"
        assert not (tmp_path / "big.h5").exists()
        assert list(tmp_path.glob(f"{PARTIAL_PREFIX}*")) == []
        # Cancellation is observed between chunks, so it must not take the whole run.
        assert time.monotonic() - started < 60
