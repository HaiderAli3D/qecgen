"""Background generation jobs: one worker subprocess each, supervised from a thread.

Deliberately boring concurrency. Each job gets a ``subprocess.Popen`` and two reader
threads (stdout for events, stderr so a chatty child cannot fill its pipe and block), and
every mutation happens under one lock. Nothing here touches the asyncio loop: the SSE
endpoint reads the same ring buffer the supervisor writes to. Cross-thread
``call_soon_threadsafe`` plumbing would buy about 100 ms of latency for a local single
user and cost a whole class of ordering bugs.

Run records are written to disk as they change. The CLI's promise is that *"a terminal
log is a complete record of the run"* (README); a browser form that vanishes on restart
would not be, so the fully resolved config, its timestamps and its outcome are persisted
per run.
"""

from __future__ import annotations

import contextlib
import enum
import json
import queue
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from qecgen.run import JobSpec, job_total, sweep_partials
from qecgen.ui.protocol import encode_line, json_safe, mode_of, spec_to_json

__all__ = [
    "DEFAULT_WORKER_COMMAND",
    "JobEvent",
    "JobRecord",
    "JobStatus",
    "JobStore",
]

DEFAULT_WORKER_COMMAND: tuple[str, ...] = (sys.executable, "-m", "qecgen.ui.worker")
"""How to start a worker.

``-m`` rather than the ``qecgen`` console script so the child is reachable from a
source checkout, and injectable so tests can substitute a scripted fake worker and get a
deterministic event sequence without sampling anything.
"""

EVENT_BUFFER = 2000
"""Events retained per job, for replay after a browser reconnects.

Bounded because a very long run with a small chunk size can emit thousands; losing the
oldest progress ticks costs nothing, and the terminal event is what matters.
"""

KILL_GRACE_SECONDS = 10.0
"""How long a cancelled worker gets to stop politely before it is killed.

Cancellation is observed between chunks, so the honest floor is one chunk's sampling
time — measured at 593 ms for d=13 with a 100k chunk. Ten seconds covers a much larger
chunk without leaving a wedged process behind forever.
"""


class JobStatus(enum.StrEnum):
    """Lifecycle of one run."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Whether no further transition is possible."""
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass(frozen=True, slots=True)
class JobEvent:
    """One entry in a job's replayable event log."""

    id: int
    kind: str
    data: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe view, as sent over SSE."""
        return {"id": self.id, "kind": self.kind, **self.data}


@dataclass
class JobRecord:
    """Everything known about one run, including the config it was resolved from."""

    id: str
    mode: str
    spec: dict[str, Any]
    total_shots: int
    status: JobStatus = JobStatus.QUEUED
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    completed_shots: int = 0
    phase: str | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None
    progress_unit: str = "shots"
    """What ``total_shots`` and ``completed_shots`` are counting.

    The field names keep saying "shots" deliberately. Renaming them would break every run
    record already on disk for a cosmetic gain, and the durable record is the thing this
    class exists to keep readable across restarts. ``"shots"`` remains the only value a
    dataset run ever uses; it is a sweep counting tasks that needs to say so.

    Empty when the total is not knowable in advance, which the browser renders as an
    indeterminate bar rather than as a bar stuck at zero.
    """
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    """Files a run produced that are **not** datasets, as ``{path, kind, size_bytes}``.

    Separate from ``files`` because ``files`` is a list of :class:`~qecgen.run.WrittenFile`
    and means one specific thing: a dataset, with a shot count, a content hash and a drift
    condition. A sweep's plot has none of those, and forcing it through that shape would
    put an invented shot count and a ``drift_condition`` on a PNG — a well-formed record
    of something that is not true, which is the failure this codebase is organised around
    avoiding. The browser renders them as different tables because they are different things.
    """
    result: dict[str, Any] | None = None
    """The summary an analysis job produced, or ``None`` for a run that produced datasets."""

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe view. This is the durable run record and the API payload."""
        return {
            "id": self.id,
            "mode": self.mode,
            "spec": self.spec,
            "status": str(self.status),
            "total_shots": self.total_shots,
            "completed_shots": self.completed_shots,
            "progress_unit": self.progress_unit,
            "phase": self.phase,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "files": self.files,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "error": self.error,
            "error_kind": self.error_kind,
            "result": self.result,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _drain(pipe: IO[str] | None, lines: queue.Queue[str | None], is_stdout: bool) -> None:
    """Move one of the child's pipes into ``lines``, and nothing else.

    This function deliberately cannot block on a lock, touch a record or write a file.
    An unread pipe fills — 4 KB on Windows for stdout, 64 KB for stderr — and the child
    then blocks forever on its next write, which presents as a run frozen at whatever
    progress it last reported. Keeping the readers this dumb is what guarantees the
    child always has somewhere to write.

    Stderr lines are wrapped so the consumer can tell them from protocol messages
    without a second queue.
    """
    try:
        if pipe is not None:
            for raw in pipe:
                text = raw.strip()
                if not text:
                    continue
                lines.put(text if is_stdout else json.dumps({"event": "stderr", "text": text}))
    except (OSError, ValueError):  # pragma: no cover - pipe torn down mid-read
        pass
    finally:
        lines.put(None)


@dataclass
class _Live:
    """Mutable per-job state that never leaves this module."""

    record: JobRecord
    events: deque[JobEvent]
    next_event_id: int = 1
    process: subprocess.Popen[str] | None = None
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    cancel_requested: bool = False


class JobStore:
    """Submits, supervises and remembers generation jobs.

    Concurrency defaults to one. These runs saturate a core and compete for memory, so a
    queue finishes a batch sooner than a stampede does, and progress means something.
    """

    def __init__(
        self,
        runs_dir: Path,
        *,
        worker_command: tuple[str, ...] = DEFAULT_WORKER_COMMAND,
        max_concurrent: int = 1,
        kill_grace_seconds: float = KILL_GRACE_SECONDS,
    ) -> None:
        self._runs_dir = runs_dir
        self._worker_command = worker_command
        self._max_concurrent = max(1, max_concurrent)
        self._kill_grace = kill_grace_seconds
        self._lock = threading.RLock()
        self._jobs: dict[str, _Live] = {}
        self._order: list[str] = []
        runs_dir.mkdir(parents=True, exist_ok=True)

    # -- submission -----------------------------------------------------------------

    def submit(self, spec: JobSpec) -> JobRecord:
        """Queue a run. Returns immediately; nothing has been sampled yet."""
        job_id = uuid.uuid4().hex[:12]
        # Both halves at submit time, not when the worker sends `started`. A queued job
        # is visible in the browser before its worker exists, and a total without its unit
        # renders as a number counting nothing.
        total, unit = job_total(spec)
        record = JobRecord(
            id=job_id,
            mode=mode_of(spec),
            spec=spec_to_json(spec),
            total_shots=total,
            progress_unit=unit,
            created_at=_now(),
        )
        with self._lock:
            self._jobs[job_id] = _Live(record=record, events=deque(maxlen=EVENT_BUFFER))
            self._order.append(job_id)
            self._append_event(job_id, "status", {"status": str(record.status)})
            self._persist(record)
            self._pump()
        return record

    def get(self, job_id: str) -> JobRecord | None:
        """One run's record, or None if the id is unknown."""
        with self._lock:
            live = self._jobs.get(job_id)
            return live.record if live else None

    def records(self) -> list[JobRecord]:
        """Every run this process knows about, newest first.

        Not named ``list``: as a method it would shadow the builtin for every annotation
        in this class body, and ``list[Path]`` elsewhere in the file would silently mean
        "subscript this method".
        """
        with self._lock:
            return [self._jobs[job_id].record for job_id in reversed(self._order)]

    def events_since(self, job_id: str, after_id: int) -> list[JobEvent]:
        """Buffered events with an id above ``after_id``."""
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None:
                return []
            return [event for event in live.events if event.id > after_id]

    # -- cancellation ---------------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Ask a run to stop. False if the id is unknown or it already finished.

        A queued job is cancelled outright. A running one is asked politely first, so it
        can unwind through the writer's abort path rather than being killed mid-write.
        """
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None or live.record.status.terminal:
                return False
            live.cancel_requested = True
            if live.record.status is JobStatus.QUEUED:
                self._finish(job_id, JobStatus.CANCELLED)
                self._pump()
                return True
            self._set_status(job_id, JobStatus.CANCELLING)
            process = live.process
        if process is not None and process.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                process.stdin.write(encode_line({"cancel": True}))
                process.stdin.flush()
            threading.Timer(self._kill_grace, lambda: self._force_kill(job_id)).start()
        return True

    def _force_kill(self, job_id: str) -> None:
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None or live.record.status.terminal or live.process is None:
                return
            process = live.process
        with contextlib.suppress(OSError):
            process.kill()

    def shutdown(self) -> None:
        """Kill every live worker. Called when the server stops."""
        with self._lock:
            processes = [live.process for live in self._jobs.values() if live.process]
        for process in processes:
            with contextlib.suppress(OSError):
                process.kill()

    # -- internals ------------------------------------------------------------------

    def _pump(self) -> None:
        """Start queued jobs while there is a free slot. Call under the lock."""
        running = sum(
            1
            for live in self._jobs.values()
            if live.record.status in (JobStatus.RUNNING, JobStatus.CANCELLING)
        )
        for job_id in self._order:
            if running >= self._max_concurrent:
                return
            live = self._jobs[job_id]
            if live.record.status is not JobStatus.QUEUED:
                continue
            running += 1
            live.record.status = JobStatus.RUNNING
            live.record.started_at = _now()
            self._append_event(job_id, "status", {"status": str(JobStatus.RUNNING)})
            self._persist(live.record)
            threading.Thread(target=self._supervise, args=(job_id,), daemon=True).start()

    def _supervise(self, job_id: str) -> None:
        """Run one worker to completion. Owns the child for its whole lifetime.

        Structured so that draining the child's pipes is separated from interpreting what
        comes out of them. A reader thread that also parses, locks and persists is a
        reader thread that can stall or fault — and a child whose stdout stops being read
        blocks on its next write, with the run stuck at whatever progress it had reached
        and no terminal state, forever. Here the drain threads do nothing but move bytes,
        and every failure below still ends with the job in a terminal state.
        """
        with self._lock:
            live = self._jobs[job_id]
            spec_payload = live.record.spec
        try:
            process = subprocess.Popen(
                list(self._worker_command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self._fail(job_id, "internal", f"could not start worker: {exc}")
            with self._lock:
                self._pump()
            return

        with self._lock:
            live.process = process
            already_cancelled = live.cancel_requested
        if already_cancelled:
            with contextlib.suppress(OSError):
                process.kill()

        lines: queue.Queue[str | None] = queue.Queue()
        readers = [
            threading.Thread(target=_drain, args=(process.stdout, lines, True), daemon=True),
            threading.Thread(target=_drain, args=(process.stderr, lines, False), daemon=True),
        ]
        for reader in readers:
            reader.start()

        if process.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                process.stdin.write(encode_line(spec_payload))
                process.stdin.flush()

        saw_terminal = False
        try:
            saw_terminal = self._consume(job_id, lines)
        finally:
            code = process.wait()
            with contextlib.suppress(OSError, ValueError):
                if process.stdin is not None:
                    process.stdin.close()
            self._finalise(job_id, saw_terminal, code)
            with self._lock:
                self._jobs[job_id].process = None
                self._pump()

    def _consume(self, job_id: str, lines: queue.Queue[str | None]) -> bool:
        """Apply worker messages until both pipes reach end of input.

        One bad message is dropped rather than allowed to end the loop: a worker that
        prints something unexpected should cost its diagnostic, not the whole run's
        bookkeeping.
        """
        saw_terminal = False
        open_pipes = 2
        while open_pipes > 0:
            line = lines.get()
            if line is None:
                open_pipes -= 1
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._note_stderr(job_id, line[:200])
                continue
            if not isinstance(message, dict):
                self._note_stderr(job_id, line[:200])
                continue
            try:
                if self._handle(job_id, message):
                    saw_terminal = True
            except Exception as exc:  # pragma: no cover - defensive
                self._note_stderr(job_id, f"could not apply worker event: {exc!r}")
        return saw_terminal

    def _finalise(self, job_id: str, saw_terminal: bool, code: int) -> None:
        """Guarantee the job ends somewhere terminal.

        Without this a supervisor that fell over — or a child killed before it could
        report — would leave a record stuck on "running" that no amount of polling ever
        resolves.
        """
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None or live.record.status.terminal:
                return
            requested = live.cancel_requested
            tail = "\n".join(live.stderr_tail).strip()
        if requested:
            with self._lock:
                if not self._jobs[job_id].record.status.terminal:
                    self._finish(job_id, JobStatus.CANCELLED)
            return
        detail = (
            tail
            if tail and not saw_terminal
            else f"the worker exited with code {code} without reporting a result"
        )
        self._fail(job_id, "internal", detail)

    def _note_stderr(self, job_id: str, text: str) -> None:
        with self._lock:
            live = self._jobs.get(job_id)
            if live is not None:
                live.stderr_tail.append(text)

    def _handle(self, job_id: str, message: dict[str, Any]) -> bool:
        """Apply one worker message. Returns True if it was a terminal one."""
        kind = message.get("event")
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None:
                return False
            record = live.record
            if kind == "stderr":
                live.stderr_tail.append(str(message.get("text", "")))
            elif kind == "started":
                record.total_shots = int(message.get("total_shots", record.total_shots))
                record.progress_unit = str(message.get("unit", record.progress_unit))
                self._append_event(
                    job_id,
                    "started",
                    {"total_shots": record.total_shots, "unit": record.progress_unit},
                )
            elif kind == "progress":
                record.completed_shots = int(message.get("completed", record.completed_shots))
                self._append_event(job_id, "progress", {"completed": record.completed_shots})
            elif kind == "phase":
                record.phase = str(message.get("phase"))
                self._append_event(job_id, "phase", {"phase": record.phase})
            elif kind == "warning":
                text = str(message.get("message", ""))
                record.warnings.append(text)
                self._append_event(job_id, "warning", {"message": text})
            elif kind == "done":
                record.files = list(message.get("files", []))
                record.artifacts = list(message.get("artifacts", []))
                result = message.get("result")
                # Sanitised again on the way in, not only on the way out. `json.loads`
                # *accepts* the `Infinity` and `NaN` tokens that `encode_line` refuses to
                # emit, so a non-finite number can still arrive here — and a record
                # holding one serves invalid JSON to the browser and writes invalid JSON
                # to its own durable record.
                record.result = json_safe(dict(result)) if isinstance(result, dict) else None
                if record.progress_unit:
                    # Only meaningful when there was a denominator. A job whose total is
                    # unknown reports 0/0, and forcing completed to match would turn an
                    # indeterminate bar into a full one at the moment it stops mattering.
                    record.completed_shots = record.total_shots
                self._finish(job_id, JobStatus.SUCCEEDED)
                return True
            elif kind == "cancelled":
                self._finish(job_id, JobStatus.CANCELLED)
                return True
            elif kind == "error":
                record.error = str(message.get("message", "unknown error"))
                record.error_kind = str(message.get("kind", "internal"))
                self._finish(job_id, JobStatus.FAILED)
                return True
        return False

    def _fail(self, job_id: str, kind: str, message: str) -> None:
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None or live.record.status.terminal:
                return
            live.record.error = message
            live.record.error_kind = kind
            self._finish(job_id, JobStatus.FAILED)

    def _set_status(self, job_id: str, status: JobStatus) -> None:
        live = self._jobs[job_id]
        live.record.status = status
        self._append_event(job_id, "status", {"status": str(status)})
        self._persist(live.record)

    def _finish(self, job_id: str, status: JobStatus) -> None:
        live = self._jobs[job_id]
        live.record.status = status
        live.record.finished_at = _now()
        self._append_event(
            job_id,
            "status",
            {
                "status": str(status),
                "files": live.record.files,
                "artifacts": live.record.artifacts,
                "result": live.record.result,
                "error": live.record.error,
                "error_kind": live.record.error_kind,
            },
        )
        self._persist(live.record)

    def _append_event(self, job_id: str, kind: str, data: dict[str, Any]) -> None:
        live = self._jobs[job_id]
        live.events.append(JobEvent(id=live.next_event_id, kind=kind, data=data))
        live.next_event_id += 1

    def _persist(self, record: JobRecord) -> None:
        path = self._runs_dir / f"{record.id}.json"
        with contextlib.suppress(OSError):
            path.write_text(json.dumps(record.to_json_dict(), indent=2), encoding="utf-8")

    def load_history(self, limit: int = 200) -> None:
        """Adopt run records left by earlier sessions, so history survives a restart.

        Anything still marked running or queued when the file was written is now a lie —
        that process is gone — so it is adopted as failed with a reason rather than
        appearing to be in flight forever.
        """
        files = sorted(self._runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for path in files[-limit:]:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            job_id = str(raw.get("id", path.stem))
            if job_id in self._jobs:
                continue
            # Every coercion sits inside the guard. The JobStatus() call used to run
            # bare, so one hand-edited or foreign record with an unrecognised status
            # string raised ValueError out of the FastAPI lifespan and the server never
            # started — a corrupt history entry must cost that entry, not the UI.
            try:
                status = JobStatus(raw.get("status", "failed"))
                error = raw.get("error")
                error_kind = raw.get("error_kind")
                stored_result = raw.get("result")
                if not status.terminal:
                    status = JobStatus.FAILED
                    error = "the server stopped while this run was in flight"
                    error_kind = "internal"
                record = JobRecord(
                    id=job_id,
                    mode=str(raw.get("mode", "generate")),
                    spec=dict(raw.get("spec", {})),
                    total_shots=int(raw.get("total_shots", 0)),
                    status=status,
                    created_at=str(raw.get("created_at", "")),
                    started_at=raw.get("started_at"),
                    finished_at=raw.get("finished_at"),
                    completed_shots=int(raw.get("completed_shots", 0)),
                    phase=raw.get("phase"),
                    files=list(raw.get("files", [])),
                    warnings=list(raw.get("warnings", [])),
                    error=error,
                    error_kind=error_kind,
                    # Defaulted, not required: every record written before these fields
                    # existed must still load. "shots" is the right default because it is
                    # what every one of those records was counting.
                    progress_unit=str(raw.get("progress_unit", "shots")),
                    artifacts=list(raw.get("artifacts", [])),
                    result=stored_result if isinstance(stored_result, dict) else None,
                )
            except (ValueError, TypeError):
                continue
            self._jobs[job_id] = _Live(record=record, events=deque(maxlen=EVENT_BUFFER))
            self._order.append(job_id)

    def clean_partials(self, root: Path) -> list[Path]:
        """Remove staging directories orphaned by a killed worker, and say which."""
        return sweep_partials(root)
