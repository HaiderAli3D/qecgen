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

from qecgen.exporters import get_exporter
from qecgen.run import PARTIAL_PREFIX, GenerateSpec
from qecgen.ui.protocol import encode_line, spec_to_json
from qecgen.ui.worker import LineReader
from qecgen.validate import validate_dataset

WORKER = (sys.executable, "-m", "qecgen.ui.worker")


def drive(
    spec: GenerateSpec, cancel_after_events: int | None = None
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
