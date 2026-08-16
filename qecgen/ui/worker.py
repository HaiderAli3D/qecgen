"""One generation run, in its own process. Run as ``python -m qecgen.ui.worker``.

A separate process rather than a thread because **stim's sampler holds the GIL**.
Measured on this machine with the job on a thread and an asyncio loop ticking beside it:
median event-loop lag 102 ms at d=9 with a 100k chunk, worst case 1033 ms at a 1M chunk.
The same workload in a subprocess left the loop at 13 ms, which is Windows timer
granularity — i.e. untouched. A web server that freezes for a second at a time cannot
serve a cancel request, which is the one request that matters during a long run.

A plain ``subprocess`` child rather than ``multiprocessing``: spawn re-imports
``__main__``, and under a setuptools console-script ``.exe`` that is fragile. ``-m`` has
no such coupling, the payload on stdin is the same JSON the browser posted, and the whole
thing is runnable by hand for debugging.

Protocol
--------
stdin  — one JSON line: the spec (see :mod:`qecgen.ui.protocol`). Then the line
         ``{"cancel": true}`` at any point to request cancellation. End-of-input is not
         a cancellation, so ``python -m qecgen.ui.worker < spec.json`` runs to completion.
stdout — JSON lines: ``started``, ``phase``, ``progress``, ``warning``, then exactly one
         of ``done`` / ``error`` / ``cancelled``.
exit   — 0 done, 1 error, 2 cancelled. A non-zero exit with no terminal event means the
         worker died, which the parent reports as a crash rather than a failure.

``started`` carries ``total_units`` and ``unit``, not a shot count. A dataset run counts
shots; a sweep counts sinter tasks, because ``max_errors`` stops a sweep and ``max_shots``
is only a ceiling, so how many shots it will take is not knowable before it runs. Naming
the unit is what keeps a progress bar from claiming one thing while counting another.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qecgen.run import (
    JobSpec,
    RunCancelledError,
    RunSpec,
    SweepSpec,
    WrittenFile,
    run,
    run_sweep_job,
    sweep_tasks,
    total_shots,
)
from qecgen.ui.protocol import encode_line, spec_from_json

if TYPE_CHECKING:
    # Annotation only. `qecgen.sweep` pulls in matplotlib and sinter, and a worker running
    # an ordinary generate should not pay that import cost to describe a callback it never
    # receives.
    from qecgen.sweep import SweepProgress

__all__ = ["LineReader", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CANCELLED = 2

PROGRESS_COALESCE_SECONDS = 0.1
"""Floor on the gap between ``progress`` messages.

At d=5 with a 100k chunk, sampling runs fast enough to emit ~43 messages a second. The
browser cannot use that and the pipe should not carry it. Coalescing here rather than in
the server keeps the volume off the wire entirely; the accumulated total is carried in
every message, so dropping intermediate ones loses nothing.
"""


def _emit(payload: dict[str, Any]) -> None:
    """Write one message and flush.

    Unflushed, progress would arrive in pipe-buffer-sized bursts and the bar would jump.
    """
    sys.stdout.write(encode_line(payload))
    sys.stdout.flush()


class LineReader:
    """Newline-delimited reader over a raw file descriptor.

    Raw ``os.read`` rather than ``sys.stdin`` for two reasons, both found the hard way.
    A daemon thread blocked inside ``sys.stdin``'s buffered reader still holds that
    object's lock when the interpreter finalises it, so the process hangs on exit instead
    of returning its status. And two readers on one ``TextIOWrapper`` race over its
    read-ahead buffer, which can swallow a cancel that arrived in the same packet as the
    spec. One reader, one buffer, no wrapper.
    """

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buffer = b""

    def readline(self) -> str | None:
        """The next line without its terminator, or None at end of input.

        The trailing ``\\r`` is stripped explicitly. Reading the raw descriptor skips the
        text layer that would normally undo Windows line endings, and the parent writes
        through a text-mode pipe that adds them — so without this every message would
        arrive with a stray carriage return and only work for as long as every consumer
        happened to tolerate trailing whitespace.
        """
        while b"\n" not in self._buffer:
            try:
                chunk = os.read(self._fd, 4096)
            except OSError:
                chunk = b""
            if not chunk:
                if not self._buffer:
                    return None
                line, self._buffer = self._buffer, b""
                return line.decode("utf-8", errors="replace").rstrip("\r")
            self._buffer += chunk
        line_bytes, _, self._buffer = self._buffer.partition(b"\n")
        return line_bytes.decode("utf-8", errors="replace").rstrip("\r")


def _watch_for_cancel(reader: LineReader, cancel: threading.Event) -> None:
    """Set ``cancel`` when the parent explicitly asks for it.

    End of input is *not* a cancellation. Treating it as one would mean a worker fed a
    spec from a file cancelled itself before sampling a single shot, which is exactly
    what happened the first time this was written.
    """
    while True:
        text = reader.readline()
        if text is None:
            return
        if not text.strip():
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("cancel"):
            cancel.set()
            return


def _files_payload(files: list[WrittenFile]) -> list[dict[str, Any]]:
    """Datasets a run wrote.

    ``kind`` discriminates these from a sweep's artifacts, which share only ``path``. A
    sweep has no shot count, no content hash and no drift condition, so the two are a
    tagged union on the wire rather than one record with half its fields nulled out.
    """
    return [
        {
            "kind": "dataset",
            "path": str(file.path),
            "shots": file.shots,
            "content_hash": file.content_hash,
            "drift_condition": str(file.drift_condition),
            "structure_source_environment_id": file.structure_source_environment_id,
        }
        for file in files
    ]


@dataclass
class _ProgressState:
    """The last reported figure, shared between a progress hook and the terminal events.

    A holder rather than ``nonlocal`` because the dataset and sweep paths live in separate
    functions and :func:`main` still has to report the final count from either of them.
    """

    completed: int = 0
    last_sent: float = 0.0

    def due(self) -> bool:
        """Whether enough time has passed to send another progress message."""
        now = time.monotonic()
        if now - self.last_sent < PROGRESS_COALESCE_SECONDS:
            return False
        self.last_sent = now
        return True


def _run_dataset(
    spec: RunSpec, cancel: threading.Event, state: _ProgressState
) -> list[dict[str, Any]]:
    """Sample a dataset run, reporting progress in shots."""
    _emit({"event": "started", "total_units": total_shots(spec), "unit": "shots"})

    def on_progress(delta: int) -> None:
        if cancel.is_set():
            raise RunCancelledError("cancelled by request")
        state.completed += delta
        if state.due():
            _emit({"event": "progress", "completed": state.completed})

    def on_phase(phase: str) -> None:
        _emit({"event": "phase", "phase": phase, "completed": state.completed})

    files = _files_payload(run(spec, progress=on_progress, on_phase=on_phase))
    # The coalescing floor can drop the last increment, so the exact total is stated here
    # rather than left at whichever partial figure happened to survive.
    _emit({"event": "progress", "completed": state.completed})
    return files


def _run_sweep(
    spec: SweepSpec, cancel: threading.Event, state: _ProgressState
) -> list[dict[str, Any]]:
    """Collect a threshold sweep, reporting progress in sinter tasks.

    Cancellation is observed when sinter next reports statistics, which is the sweep's
    equivalent of a dataset run's chunk boundary. Raising from the hook is what tears the
    collection down — see :func:`qecgen.sweep.run_sweep`.
    """
    _emit({"event": "started", "total_units": sweep_tasks(spec), "unit": "tasks"})
    latest: list[SweepProgress] = []

    def on_progress(progress: SweepProgress) -> None:
        if cancel.is_set():
            raise RunCancelledError("cancelled by request")
        # Assigned, not accumulated: a sweep's progress is a count of finished tasks that
        # the adapter recomputes each time, unlike a dataset run's per-chunk increments.
        state.completed = progress.completed_tasks
        latest[:] = [progress]
        if state.due():
            _emit(
                {
                    "event": "progress",
                    "completed": state.completed,
                    "shots_collected": progress.shots_collected,
                    "detail": progress.status_message,
                }
            )

    def on_phase(phase: str) -> None:
        _emit({"event": "phase", "phase": phase, "completed": state.completed})

    result = run_sweep_job(spec, progress=on_progress, on_phase=on_phase)

    # The coalescing floor drops intermediate reports, and the last one it dropped is
    # usually the only one that had the final shot count -- so the readout would settle on
    # whichever partial figure happened to survive.
    #
    # The total is taken from the collected points rather than from the last progress
    # report, so it needs no fallback: a missing report would have had to be reported as
    # some number, and 0 is indistinguishable from "collected nothing". `detail` is simply
    # omitted when there is none; jobs._handle leaves the previous value alone for an
    # absent key, whereas sending null would land in the record as the string "None".
    final: dict[str, Any] = {
        "event": "progress",
        "completed": sweep_tasks(spec),
        "shots_collected": sum(point.shots for point in result.points),
    }
    if latest:
        final["detail"] = latest[0].status_message
    _emit(final)
    return [
        {"kind": f"sweep_{artifact.kind}", "path": str(artifact.path)}
        for artifact in result.artifacts
    ]


def _dispatch(
    spec: JobSpec, cancel: threading.Event, state: _ProgressState
) -> list[dict[str, Any]]:
    """Run whichever kind of job the spec describes."""
    if isinstance(spec, SweepSpec):
        return _run_sweep(spec, cancel, state)
    return _run_dataset(spec, cancel, state)


def _detach_control_channel() -> int:
    """Move the parent's control pipe off file descriptor 0, and return its new one.

    Two separate reasons, both found by measurement on Windows.

    **A sweep deadlocks without this.** ``sinter.collect`` spawns a worker pool, and
    ``multiprocessing``'s spawn handshake cannot complete while another thread of this
    process is parked in a blocking ``os.read`` on descriptor 0: the children reach about
    9 MB resident with a single thread and no Python frame at all, and the parent waits
    forever in ``_compute_task_ids``. Measured over four variants of the same collection —
    open pipe with no reader thread finishes in 1.2 s, open pipe *with* a reader on fd 0
    never finishes, and a reader on a duplicate with fd 0 pointed at ``os.devnull``
    finishes in 1.2 s.

    **A grandchild must not inherit the control channel anyway.** Descriptor 0 is
    inheritable; a duplicate made by :func:`os.dup` is not (PEP 446). So this also stops a
    sinter worker from holding — or worse, consuming — the pipe that carries
    ``{"cancel": true}``. That would be a cancel that silently never arrives.

    A dataset run never needed this only because it spawns nothing.
    """
    control_fd = os.dup(0)
    null_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null_fd, 0)
    finally:
        os.close(null_fd)
    return control_fd


def _preload(spec: JobSpec) -> None:
    """Import what the job needs before announcing that the run has started.

    **This is not the deadlock fix, despite how it was originally written.**
    :func:`_detach_control_channel` is, and it subsumes this: with the control pipe moved
    off descriptor 0, importing ``qecgen.sweep`` after the watcher thread has started
    completes normally. Measured both ways on the same collection — reverted to a no-op,
    the worker still finishes in 2.1 s and writes all three artifacts. The hang this
    function was written for was reproducible with the imports *already done*, which is
    what identifies fd 0 rather than the import as the cause.

    What it is still worth: a broken or partial install fails here, as an ``input`` error
    before ``started`` is emitted, rather than a minute into a collection as an
    ``internal`` one. That is a real difference to whoever reads the run record.

    It also keeps the import conditional, which preserves the rule the laziness exists for
    — a worker running ``generate`` never pays for matplotlib. Do not promote it to module
    scope for tidiness.
    """
    if isinstance(spec, SweepSpec):
        import qecgen.sweep  # noqa: F401  (imported for its side effect: loading the DLLs)


def main(argv: list[str] | None = None) -> int:
    """Read one spec from stdin, run it, report on stdout."""
    del argv
    reader = LineReader(_detach_control_channel())
    raw = reader.readline()
    if raw is None or not raw.strip():
        _emit({"event": "error", "kind": "input", "message": "no spec on stdin"})
        return EXIT_ERROR

    try:
        spec = spec_from_json(json.loads(raw))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        _emit({"event": "error", "kind": "input", "message": f"could not read spec: {exc}"})
        return EXIT_ERROR

    try:
        _preload(spec)
    except ImportError as exc:
        _emit({"event": "error", "kind": "input", "message": f"missing dependency: {exc}"})
        return EXIT_ERROR

    # Every heavy import is done by this point. Nothing below may import a native
    # extension for the first time -- see _preload.
    cancel = threading.Event()
    threading.Thread(target=_watch_for_cancel, args=(reader, cancel), daemon=True).start()

    state = _ProgressState()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            files = _dispatch(spec, cancel, state)
        for warning in caught:
            # JSONLExporter warns above 100k shots with a size estimate. On a terminal
            # that lands in front of the user; through a pipe it would vanish silently.
            _emit({"event": "warning", "message": str(warning.message)})
    except RunCancelledError:
        _emit({"event": "cancelled", "completed": state.completed})
        return EXIT_CANCELLED
    except ValueError as exc:
        # Every input problem in the package is a bare ValueError, and the messages are
        # written for humans. Passed through verbatim; the server turns this into a 400.
        _emit({"event": "error", "kind": "input", "message": str(exc)})
        return EXIT_ERROR
    # The process boundary. Nothing above this catches, so an unexpected failure has to
    # be reported as an event rather than as a traceback on a pipe nobody reads.
    except Exception as exc:
        _emit({"event": "error", "kind": "internal", "message": f"{type(exc).__name__}: {exc}"})
        return EXIT_ERROR

    # No final progress event here: each run kind emits its own, because only it knows
    # what else belongs in the last report. A generic one appended after theirs used to
    # overwrite a sweep's final shot count with nothing.
    _emit({"event": "done", "files": files})
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
