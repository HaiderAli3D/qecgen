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
from typing import cast

from qecgen.exporters import get_exporter
from qecgen.run import PARTIAL_PREFIX, GenerateSpec, JobSpec, SweepSpec
from qecgen.ui.protocol import encode_line, spec_to_json
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


class TestControlChannel:
    """The fd-0 rule, asserted directly rather than inferred from a hang.

    `TestSweepWorker` catches a regression here only by timing out after two minutes, with
    a failure message that names neither fd 0 nor the control channel. These pin the three
    documented properties in under a second each.
    """

    PROBE = (
        "import json, os, sys\n"
        "from qecgen.ui.worker import _detach_control_channel\n"
        "fd = _detach_control_channel()\n"
        "print(json.dumps({\n"
        "    'from_control': os.read(fd, 5).decode(),\n"
        "    'from_fd0': os.read(0, 5).decode(),\n"
        "    'control_inheritable': os.get_inheritable(fd),\n"
        "    'control_fd': fd,\n"
        "}), flush=True)\n"
    )

    def _probe(self) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-c", self.PROBE],
            input="hello",
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0, result.stderr
        return cast("dict[str, object]", json.loads(result.stdout))

    def test_the_parents_bytes_arrive_on_the_returned_descriptor(self) -> None:
        assert self._probe()["from_control"] == "hello"

    def test_descriptor_zero_is_left_pointing_at_nothing(self) -> None:
        """Not the pipe. A thread blocked reading fd 0 is what wedges a sweep: on Windows
        `multiprocessing`'s spawn handshake never completes and sinter's pool never
        starts. Devnull reads empty immediately."""
        assert self._probe()["from_fd0"] == ""
        assert self._probe()["control_fd"] != 0

    def test_the_control_channel_is_not_inheritable(self) -> None:
        """A sinter grandchild inheriting it could consume the `{"cancel": true}` line,
        producing a cancel that silently never arrives. `os.dup` is non-inheritable by
        PEP 446 -- this asserts that rather than assuming it."""
        assert self._probe()["control_inheritable"] is False


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
        assert events[0]["total_units"] == 400
        assert events[0]["unit"] == "shots"
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


def _sweep(out: Path, **overrides: object) -> SweepSpec:
    base: dict[str, object] = {
        "distances": (3,),
        "error_rates": (0.02, 0.05),
        "out": out,
        "max_errors": 6,
        "max_shots": 300,
        "workers": 2,
    }
    return SweepSpec(**{**base, **overrides})  # type: ignore[arg-type]


def _descendants(pid: int) -> set[int]:
    """Every live process descended from ``pid``.

    Used to prove a cancelled sweep leaves nothing behind. sinter runs a pool of its own,
    so "the worker exited" is not the same claim as "the machine is idle again".

    Windows-only, hence the skip on its caller: this shells out to `Get-CimInstance`.
    """
    listing = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId "
            "| ConvertTo-Json -Compress",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    rows = json.loads(listing) if listing.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    by_parent: dict[int, list[int]] = {}
    for row in rows:
        by_parent.setdefault(int(row["ParentProcessId"]), []).append(int(row["ProcessId"]))
    found: set[int] = set()
    queue = [pid]
    while queue:
        for child in by_parent.get(queue.pop(), []):
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


class TestSweepWorker:
    """The sweep path through a real worker process.

    Deliberately NOT marked slow, despite collecting for real. These two are the only
    thing standing between the tree and a permanent Windows hang in `sinter.collect`, and
    `pytest -m "not slow"` is the command in the docs and the one people actually run --
    a guard that lives only in the suite nobody runs by default is not a guard. The sweeps
    are sized so the pair costs a handful of seconds: one distance, two rates, and a
    stopping rule of a few dozen errors.
    """

    def test_a_sweep_reports_tasks_and_writes_all_three_artifacts(self, tmp_path: Path) -> None:
        spec = _sweep(tmp_path / "sweeps" / "s.csv")
        events, code = drive(spec)
        kinds = [event["event"] for event in events]

        assert code == 0, events
        assert events[0] == {"event": "started", "total_units": 2, "unit": "tasks"}
        assert kinds[-1] == "done"

        # The denominator is tasks, not shots: max_errors stops a sweep and max_shots is
        # only a ceiling, so a shot total is not knowable before the run.
        progress = [event for event in events if event["event"] == "progress"]
        assert progress[-1]["completed"] == 2
        # The last report must carry the real shot count. Coalescing drops intermediate
        # messages, and the dropped one used to be the only one that had the total.
        shots = progress[-1]["shots_collected"]
        assert isinstance(shots, int) and shots > 0

        # Each kind must name the RIGHT file, not merely appear in the right order.
        # `run_sweep_job` matches artifacts by name because `Staging` commits in sorted
        # order; under these filenames sorted order happens to agree, so asserting the
        # sequence alone would still pass with `sweep_results` pointing at the .png.
        files = cast("list[dict[str, str]]", events[-1]["files"])
        by_kind = {entry["kind"]: entry["path"] for entry in files}
        assert set(by_kind) == {"sweep_results", "sweep_plot", "sweep_summary"}
        assert by_kind["sweep_results"].endswith(".csv")
        assert by_kind["sweep_plot"].endswith(".png")
        assert by_kind["sweep_summary"].endswith(".threshold.json")
        for path in by_kind.values():
            assert Path(path).is_file()
        assert list((tmp_path / "sweeps").glob(f"{PARTIAL_PREFIX}*")) == []

    def test_a_cancelled_sweep_leaves_no_files_and_no_stray_processes(self, tmp_path: Path) -> None:
        """Cancellation has to tear down sinter's pool, not just this process.

        Raising from the progress hook propagates out of sinter's consumption loop and
        closes the ``iter_collect`` generator, whose ``CollectionManager.__exit__`` calls
        ``hard_stop()`` — which kills and joins every worker. Asserted rather than assumed:
        a cancel that leaves four sampling processes behind looks identical to a clean one
        from the browser.
        """
        spec = _sweep(
            tmp_path / "sweeps" / "big.csv",
            error_rates=(0.002, 0.004, 0.006),
            distances=(5,),
            max_errors=10_000,
            max_shots=5_000_000,
        )
        process = subprocess.Popen(
            list(WORKER),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        watchdog = threading.Timer(180.0, process.kill)
        watchdog.daemon = True
        watchdog.start()
        events: list[dict[str, object]] = []
        try:
            process.stdin.write(encode_line(spec_to_json(spec)))
            process.stdin.flush()
            sent = False
            for line in process.stdout:
                if not line.strip():
                    continue
                events.append(json.loads(line))
                # Cancel once shots are genuinely being collected. sinter's FIRST progress
                # callback fires at pool startup with `shots_collected: 0` and the message
                # "Starting N workers...", so triggering on any progress event cancelled
                # during the very startup race this test claims to look past -- and the
                # "no stray processes" assertion then only covered tearing down a pool
                # that had not begun sampling.
                shots = events[-1].get("shots_collected")
                collecting = (
                    events[-1].get("event") == "progress" and isinstance(shots, int) and shots > 0
                )
                if not sent and collecting:
                    process.stdin.write(encode_line({"cancel": True}))
                    process.stdin.flush()
                    sent = True
            code = process.wait(timeout=120)
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            process.stdin.close()

        assert code == 2, events
        assert events[-1]["event"] == "cancelled"
        assert not (tmp_path / "sweeps" / "big.csv").exists()
        assert not (tmp_path / "sweeps" / "big.png").exists()
        assert not (tmp_path / "sweeps" / "big.threshold.json").exists()
        if (tmp_path / "sweeps").is_dir():
            assert list((tmp_path / "sweeps").glob(f"{PARTIAL_PREFIX}*")) == []
        if sys.platform == "win32":
            # The process-tree probe is Windows-only; on other platforms the rest of the
            # assertions still hold and this one is skipped rather than raising
            # FileNotFoundError from a missing powershell.
            assert _descendants(process.pid) == set()
