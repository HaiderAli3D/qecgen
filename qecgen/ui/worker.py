"""One job, in its own process. Run as ``python -m qecgen.ui.worker``.

A job is either a generation run (:data:`~qecgen.run.RunSpec`) or an analysis of files
that already exist (:data:`~qecgen.run.AnalysisSpec`). The spec on stdin is the worker's
entire input, so which one it is is decided by the ``mode`` in that payload and by
nothing else — there is one worker command, and every UI job test depends on that,
because they all substitute a scripted child for it.

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
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation only -- importing `qecgen.sweep` here would pull sinter and
    # matplotlib into every worker start-up, which is what `preload` exists to defer.
    from qecgen.sweep import SweepProgress

from qecgen.run import (
    AnalysisResult,
    DriftSpec,
    GenerateSpec,
    MultiEnvSpec,
    RunCancelledError,
    SweepSpec,
    WrittenFile,
    analyse,
    job_total,
    preload,
    run,
    run_threshold_sweep,
)
from qecgen.ui.protocol import encode_line, json_safe, spec_from_json

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

_ARTIFACT_KINDS = {".png": "plot", ".csv": "results table", ".json": "summary"}
"""How to label a non-dataset output, so the browser can say what a file is.

By extension rather than by the producing job, because a sweep writes three files of
three different kinds in one call and "sweep" on all three says nothing useful.
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


def _detach_control_channel() -> int:
    """Move the parent's control pipe off file descriptor 0, and return its new one.

    A sweep hands its grid to sinter, which runs a ``multiprocessing`` pool with start
    method ``spawn``. On Windows the spawn handshake cannot complete while another thread
    of this process is parked in a blocking ``os.read`` on descriptor 0: the children
    reach about 9 MB resident with a single thread and no Python frame at all, and the
    parent waits forever in ``_compute_task_ids`` — the sweep stops at "Starting 2
    workers..." and never resumes. Measured over four variants of the same collection:
    an open pipe with no reader thread finishes in 1.2 s, an open pipe *with* a reader on
    fd 0 never finishes, and a reader on a duplicate with fd 0 pointed at ``os.devnull``
    finishes in 1.2 s.

    The duplicate is what fixes it. ``os.dup`` returns a **non-inheritable** descriptor
    (PEP 446), so the watcher keeps a working pipe that no child receives, while fd 0 —
    the one children do inherit — becomes the null device. Cancellation is unaffected: the
    watcher reads the same pipe it always did, through a different descriptor.

    This is the second face of the same Windows hazard as :func:`~qecgen.run.preload`. A
    thread blocked reading stdin first broke a DLL-loading import; here it breaks process
    creation. Neither has any symptom other than silence.
    """
    private = os.dup(0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, 0)
    finally:
        os.close(devnull)
    return private


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
    return [
        {
            "path": str(file.path),
            "shots": file.shots,
            "content_hash": file.content_hash,
            "drift_condition": str(file.drift_condition),
            "structure_source_environment_id": file.structure_source_environment_id,
        }
        for file in files
    ]


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

    # Before the watcher parks, never after. A thread blocked reading stdin stops this
    # process from completing a large DLL-loading import -- see `run.preload`, which
    # documents the measurement. Cancellation is unavailable for the duration, which is
    # correct: there is nothing yet to cancel, and the wait is bounded by an import.
    try:
        preload(spec)
    except Exception as exc:  # pragma: no cover - an unimportable dependency
        _emit(
            {
                "event": "error",
                "kind": "internal",
                "message": f"could not load what this job needs: {type(exc).__name__}: {exc}",
            }
        )
        return EXIT_ERROR

    cancel = threading.Event()
    threading.Thread(target=_watch_for_cancel, args=(reader, cancel), daemon=True).start()

    completed = 0
    last_sent = 0.0
    # Only a sweep sets this. The terminal progress event has to carry it, because
    # coalescing drops intermediate messages and the dropped one is often the only
    # one that saw the final total.
    shots_seen: int | None = None
    total, unit = job_total(spec)
    _emit({"event": "started", "total_units": total, "unit": unit})

    def on_progress(delta: int) -> None:
        nonlocal completed, last_sent
        if cancel.is_set():
            raise RunCancelledError("cancelled by request")
        completed += delta
        now = time.monotonic()
        if now - last_sent >= PROGRESS_COALESCE_SECONDS:
            last_sent = now
            _emit({"event": "progress", "completed": completed})

    def on_phase(phase: str) -> None:
        _emit({"event": "phase", "phase": phase, "completed": completed})

    def on_sweep_progress(update: SweepProgress) -> None:
        """A sweep's progress, with the two extras only sinter can supply.

        `shots_collected` and `detail` ride along with the task count rather than going
        through `on_progress`, because a sweep is the only job that has them: its bar
        counts tasks, and how many shots that took is a separate number the record shows
        beside it. Absent keys must never blank an existing readout, so they are only
        ever sent, never sent as null -- see `JobStore._handle`.
        """
        nonlocal completed, last_sent, shots_seen
        if cancel.is_set():
            raise RunCancelledError("cancelled by request")
        completed = update.completed_tasks
        shots_seen = update.shots_collected
        now = time.monotonic()
        if now - last_sent >= PROGRESS_COALESCE_SECONDS:
            last_sent = now
            message: dict[str, Any] = {
                "event": "progress",
                "completed": completed,
                "shots_collected": update.shots_collected,
            }
            if update.status_message:
                message["detail"] = update.status_message
            _emit(message)

    files: list[WrittenFile] = []
    analysis: AnalysisResult | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The only branch in this module. A worker is otherwise spec-kind-agnostic,
            # and it stays that way for analysis specs too -- `analyse` dispatches among
            # them exactly as `run` does among the run kinds.
            if isinstance(spec, GenerateSpec | MultiEnvSpec | DriftSpec):
                files = run(spec, progress=on_progress, on_phase=on_phase)
            elif isinstance(spec, SweepSpec):
                # Called directly rather than through `analyse`, which flattens sinter's
                # update to an increment. The extra fields are the whole reason a sweep's
                # readout is legible while it runs.
                analysis = run_threshold_sweep(spec, progress=on_sweep_progress, on_phase=on_phase)
            else:
                analysis = analyse(spec, progress=on_progress, on_phase=on_phase)
        for warning in caught:
            # JSONLExporter warns above 100k shots with a size estimate. On a terminal
            # that lands in front of the user; through a pipe it would vanish silently.
            _emit({"event": "warning", "message": str(warning.message)})
    except RunCancelledError:
        _emit({"event": "cancelled", "completed": completed})
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

    final: dict[str, Any] = {"event": "progress", "completed": completed}
    if shots_seen is not None:
        final["shots_collected"] = shots_seen
    _emit(final)
    if analysis is None:
        _emit_done(files=_files_payload(files))
    else:
        _emit_done(
            files=[],
            artifacts=[_artifact_payload(path, analysis.kind) for path in analysis.artifacts],
            result={"kind": analysis.kind, **analysis.summary},
        )
    return EXIT_OK


def _artifact_payload(path: Path, kind: str) -> dict[str, Any]:
    """Describe one non-dataset output. ``size_bytes`` is 0 if it vanished under us."""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return {"path": str(path), "kind": _ARTIFACT_KINDS.get(path.suffix, kind), "size_bytes": size}


def _emit_done(
    *,
    files: list[dict[str, Any]],
    artifacts: list[dict[str, Any]] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    """Emit the terminal message, degrading rather than failing if it will not encode.

    This is the one message that must always get out. Everything above it has already
    happened: the work succeeded and :func:`~qecgen.run.staged` committed its files. If
    ``encode_line`` raises here, the parent sees a worker that exited without reporting a
    result and reports a *failed* run whose output is sitting on disk — the single most
    misleading state this protocol can reach.

    :func:`json_safe` should make that unreachable. The fallback exists because "should"
    is doing real work in that sentence: a summary is an arbitrary dict assembled by a
    domain module, and the cost of one unencodable value in it is not worth a false
    failure. A run that loses its summary and says so is strictly better.
    """
    payload: dict[str, Any] = {
        "event": "done",
        "files": files,
        "artifacts": artifacts or [],
        "result": json_safe(result) if result is not None else None,
    }
    try:
        _emit(payload)
    except ValueError as exc:
        _emit({"event": "warning", "message": f"result dropped, could not encode: {exc}"})
        _emit({"event": "done", "files": files, "artifacts": artifacts or [], "result": None})


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
