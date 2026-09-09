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
import os
import queue
import signal
import subprocess
import sys
import threading
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from qecgen import deletion
from qecgen.deletion import DeletionRefusedError, plan_deletion
from qecgen.run import JobSpec, job_total, sweep_partials
from qecgen.ui.protocol import encode_line, json_safe, mode_of, spec_to_json

__all__ = [
    "DEFAULT_WORKER_COMMAND",
    "ActiveOutput",
    "DiscardOutcome",
    "JobEvent",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "RunNotFinishedError",
    "run_input_paths",
    "run_output_paths",
]


class RunNotFinishedError(RuntimeError):
    """A run that has not reached a terminal state cannot be forgotten.

    Refused rather than cancelled-then-deleted, and the reason is a deadlock rather than
    tidiness. ``cancel`` is not synchronous: it writes to the child's stdin and schedules a
    force kill ``KILL_GRACE_SECONDS`` later, and the worker only notices between chunks. So
    an auto-cancel would either hold the request thread for ten seconds or return before the
    child stopped -- and in the second case ``_supervise``'s ``finally`` reaches for a record
    that is no longer there, on a daemon thread, with ``_pump()`` never called again. A
    killed worker also leaves a staging directory that only ``clean_partials`` removes, at
    startup, so the file the user just "deleted" can still land after the delete returns.
    """

    def __init__(self, job_id: str, status: JobStatus) -> None:
        self.job_id = job_id
        self.status = status
        super().__init__(f"run {job_id!r} is {status}; cancel it first, then delete")


@dataclass(frozen=True, slots=True)
class DiscardOutcome:
    """What :meth:`JobStore.discard` did."""

    record: JobRecord
    record_path: Path
    removed: bool
    """Whether the durable record file is gone.

    False is reported rather than raised, and it matters: ``load_history`` re-adopts any
    ``runs/*.json`` it finds, so a record whose file survived comes back on the next restart.
    Returning a bare success for that would be a well-formed record of something untrue.
    """

    problem: str | None


def run_output_paths(record: JobRecord) -> list[Path]:
    """Every file this run **wrote**, deduplicated, in report order.

    Reads ``files`` and ``artifacts`` and nothing else. ``spec`` holds the run's *inputs* --
    ``dataset`` and ``correction`` -- and a ``score``, ``qa`` or ``benchmark`` run produces no
    files at all while naming the user's own training set there. A deletion that walked
    ``spec`` would bin the input of every analysis run in the history, so this takes the
    record rather than the spec and has no parameter that could turn ``spec`` reading on.

    That request will be made ("also clean up what it read"). It must be refused; the input
    is not the run's to take back.

    ``or []`` on both fields: records written before the analysis-job layer have no
    ``artifacts`` key, and a hand-edited or foreign one can carry an explicit ``null``.
    ``list(None)`` is a ``TypeError``, and this runs while a user is looking at a
    confirmation dialog.
    """
    seen: dict[str, Path] = {}
    for entry in [*(record.files or []), *(record.artifacts or [])]:
        raw = entry.get("path")
        if isinstance(raw, str) and raw:
            seen.setdefault(os.path.normcase(raw), Path(raw))
    return list(seen.values())


def run_input_paths(record: JobRecord) -> list[Path]:
    """Every file this run **read**, so a confirmation can promise to keep it.

    The counterpart to :func:`run_output_paths`, and read-only in the strictest sense: no
    caller may pass this to a deletion. It exists because a user deleting a ``score`` run
    sees its input dataset named all over the run detail page, and has no other way to learn
    that it is safe.
    """
    seen: dict[str, Path] = {}
    for key in ("dataset", "correction"):
        raw = record.spec.get(key)
        if isinstance(raw, str) and raw:
            seen.setdefault(os.path.normcase(raw), Path(raw))
    config = record.spec.get("config")
    if record.mode == "configured" and isinstance(config, dict):
        for section, keys in (("hardware", ("table", "circuit")), ("circuit", ("stim_file",))):
            fields = config.get(section)
            if isinstance(fields, dict):
                for key in keys:
                    raw = fields.get(key)
                    if isinstance(raw, str) and raw:
                        seen.setdefault(os.path.normcase(raw), Path(raw))
    return list(seen.values())


def _output_from_record(record: JobRecord) -> Any:
    """Only the declared output participates in live-write protection, never its inputs."""
    if record.mode == "configured":
        config = record.spec.get("config")
        if isinstance(config, dict) and isinstance(config.get("output"), dict):
            return config["output"].get("path")
        return None
    return record.spec.get("out")


@dataclass(frozen=True, slots=True)
class ActiveOutput:
    """Where one unfinished run is going to write."""

    job_id: str
    status: JobStatus
    base: Path
    members: frozenset[str]
    """Normcased paths the run's output expands to; empty when it cannot be planned yet."""

    def claims(self, candidate: Path) -> bool:
        """Whether ``candidate`` is part of what this run is about to write."""
        key = os.path.normcase(str(candidate))
        if key in self.members or key == os.path.normcase(str(self.base)):
            return True
        return self.base in candidate.parents


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
    """Everything known about one run, including the config it was resolved from.

    Progress is counted in *units*, and :attr:`progress_unit` says which. A dataset run
    counts shots; a sweep counts sinter tasks, because ``max_errors`` stops a sweep and
    ``max_shots`` is only a ceiling, so its shot total is not knowable in advance. These
    fields were once named ``total_shots``/``completed_shots``; keeping those names while
    counting tasks would have been a field that lies about its own contents, which is the
    class of bug this project exists to avoid. :meth:`JobStore.load_history` still reads
    the old names off disk.
    """

    id: str
    mode: str
    spec: dict[str, Any]
    total_units: int
    progress_unit: str = "shots"
    status: JobStatus = JobStatus.QUEUED
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    completed_units: int = 0
    phase: str | None = None
    detail: str | None = None
    """A free-form line about what is happening right now. For a sweep this is sinter's
    own status message — tasks remaining and an ETA for each — which is the only estimate
    of time to completion anything here has."""
    shots_collected: int | None = None
    """Sweep only. A dataset run's shot count *is* :attr:`completed_units`, so repeating it
    would invite the two to disagree."""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    """Files a run produced that are **not** datasets, as ``{path, kind, size_bytes}``.

    Separate from :attr:`files` because that list means one specific thing: a dataset,
    with a shot count, a content hash and a drift condition. A sweep's plot has none of
    those, and forcing it through that shape would put an invented shot count and a
    ``drift_condition`` on a PNG -- a well-formed record of something untrue, which is
    the failure this codebase is organised around avoiding.
    """
    result: dict[str, Any] | None = None
    """The summary an analysis job produced, or ``None`` for a run that made datasets."""
    files: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    error_kind: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe view. This is the durable run record and the API payload."""
        return {
            "id": self.id,
            "mode": self.mode,
            "spec": self.spec,
            "status": str(self.status),
            "total_units": self.total_units,
            "completed_units": self.completed_units,
            "progress_unit": self.progress_unit,
            "phase": self.phase,
            "detail": self.detail,
            "shots_collected": self.shots_collected,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifacts": self.artifacts,
            "result": self.result,
            "files": self.files,
            "warnings": self.warnings,
            "error": self.error,
            "error_kind": self.error_kind,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _kill_tree(process: subprocess.Popen[str]) -> None:
    """Kill a worker **and everything it spawned**.

    ``Popen.kill()`` reaches only the direct child. That was harmless while every worker
    was a lone sampling process, but a sweep owns a ``multiprocessing`` pool: killing just
    the worker leaves sinter's children finishing their in-flight batch and then blocking
    forever on a queue whose other end is gone. Measured on Windows — two pool processes
    still resident two and a half minutes after the worker exited, CPU frozen, never
    reaped. Both callers can reach that state: the force-kill after a cancel grace expires,
    and :meth:`JobStore.shutdown`, which the server lifespan calls with no grace at all, so
    a Ctrl-C during a sweep would strand one process per ``workers`` on every restart.

    On POSIX the workers are started in their own session (see ``_supervise``) so the whole
    group can be signalled without touching this process; Windows has no process groups
    that survive here, so ``taskkill /T`` walks the tree instead.
    """
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
    else:
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    # Always finish with the direct kill: taskkill fails once the pid is already gone, and
    # killpg cannot run if the child never made it into its own session.
    with contextlib.suppress(OSError):
        process.kill()


def _with_kind(entry: dict[str, Any]) -> dict[str, Any]:
    """Backfill the artifact discriminator on a record written before it existed.

    Every file a run reports now carries ``kind`` — ``dataset`` or one of the ``sweep_*``
    variants — and the browser branches on it. Records persisted by an earlier build have
    no such key, and a front end that reads it unguarded gets ``undefined`` for every file
    of every run in the existing history. Filling it in on read is the fix at the source:
    only a dataset run could have written those records, so ``dataset`` is not a guess.
    """
    if "kind" in entry:
        return entry
    return {**entry, "kind": "dataset"}


def _progress_denominator(spec: JobSpec) -> tuple[int, str]:
    """How much work the run is, and what that number counts.

    The unit travels with the number so nothing downstream has to infer it from the mode.
    A bar labelled "shots" over a task count is a well-formed display of the wrong thing,
    which is exactly the failure mode this codebase spends its docstrings on.

    Delegated to :func:`qecgen.run.job_total` rather than branched on here. That match is
    exhaustive over ``JobSpec``, so a new job kind fails there — where the author is
    already working — instead of falling through to ``total_shots`` and asking a spec
    with no shots how many it has. An empty unit means the total is not knowable in
    advance and the bar should render indeterminate.
    """
    return job_total(spec)


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
        units, unit = _progress_denominator(spec)
        record = JobRecord(
            id=job_id,
            mode=mode_of(spec),
            spec=spec_to_json(spec),
            total_units=units,
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

    # -- forgetting a run -----------------------------------------------------------

    def discard(self, job_id: str) -> DiscardOutcome | None:
        """Forget one finished run: its in-memory entry, its events, and its record file.

        ``None`` when the id is unknown, mirroring :meth:`get`. Named for ``set.discard``
        semantics -- remove if present, do not raise if absent -- and deliberately *not*
        ``delete``: this removes the record and only the record, and a name suggesting it
        also removes the run's datasets is exactly the confusion this feature has to avoid.

        The record file goes to the recycle bin like everything else rather than being
        unlinked. A run record is the only durable copy of a fully resolved config -- the
        web half of the CLI's promise that a terminal log is a complete record of the run --
        so destroying it permanently while recoverably binning the reproducible dataset
        beside it would be backwards.

        Raises:
            RunNotFinishedError: the run has not reached a terminal state. See that class
                for why this refuses rather than cancelling first.
        """
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None:
                return None
            if not live.record.status.terminal:
                raise RunNotFinishedError(job_id, live.record.status)
            record = live.record
            # Both structures, under one acquisition. `_pump` iterates `_order` and indexes
            # `_jobs`; leaving them disagreeing for even a moment makes it raise KeyError on
            # a daemon thread, and then the queue never starts another job.
            del self._jobs[job_id]
            self._order.remove(job_id)

        # Outside the lock on purpose: on Windows this goes through SHFileOperationW, which
        # can block for a visible interval, and holding the store lock across it would stall
        # every SSE poll and every listing in the process. `_drain`'s reasoning, one level up.
        path = self._runs_dir / f"{job_id}.json"
        removed, problem = self._trash_record(path)
        return DiscardOutcome(record=record, record_path=path, removed=removed, problem=problem)

    def _trash_record(self, path: Path) -> tuple[bool, str | None]:
        """Send one run record to the recycle bin. Never raises.

        A failure is reported rather than raised because the in-memory record is already gone
        by this point. ``load_history`` re-adopts any surviving ``runs/*.json``, so the caller
        has to be able to say "forgotten, but it will be back after a restart" instead of
        claiming a clean delete.
        """
        if not path.exists():
            return True, None
        try:
            # Through the module, never a direct `from ... import send_to_trash`. That
            # binding would be a second seam: patching `qecgen.deletion.send_to_trash` would
            # not reach it, so a test believing it had disabled real deletion would move the
            # developer's own run records into their recycle bin.
            deletion.send_to_trash(path)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if path.exists():
            return False, "the recycle request returned but the record file is still there"
        return True, None

    def active_outputs(self) -> list[ActiveOutput]:
        """Where every run that has not finished is going to write.

        Built only from the declared output (``out`` or ``config.output.path``).
        ``dataset`` and ``correction`` are *inputs*: a
        second run reading the same training set is ordinary, and blocking a delete because
        something is reading a file would make every dataset undeletable while a QA job ran.
        Deleting a file mid-read corrupts nothing either -- a POSIX reader keeps its fd, and
        Windows fails the recycle move with a sharing violation, which the outcome reports
        honestly.

        Expanded through :func:`~qecgen.deletion.plan_deletion` so a live sweep claims its
        whole triple rather than only the ``.csv`` it names. One planner answers both "what am
        I about to delete" and "what is a live run about to write", so the two cannot disagree
        about what a set is.

        **A courtesy, not a lock.** It sees only runs this process supervises; a
        ``qecgen generate`` in another terminal is invisible and cannot be made visible,
        because the ``.qecgen-lock`` lives inside the staging directory rather than beside the
        destination, and the destination is untouched until the atomic commit. What makes that
        acceptable is that the destructive act is a recycle-bin move. Do not build on this as
        though it were exclusive.
        """
        with self._lock:
            pending = [
                (live.record.id, live.record.status, _output_from_record(live.record))
                for live in self._jobs.values()
                if not live.record.status.terminal
            ]
        active: list[ActiveOutput] = []
        for job_id, status, raw in pending:
            if not isinstance(raw, str) or not raw:
                continue
            base = Path(raw)
            members: frozenset[str] = frozenset()
            try:
                plan = plan_deletion(base)
            except (DeletionRefusedError, OSError):
                # Nothing there yet is the normal case: staged writes mean the destination
                # does not exist until the run commits. The bare path still guards it.
                pass
            else:
                members = frozenset(os.path.normcase(str(entry.path)) for entry in plan.files)
            active.append(ActiveOutput(job_id=job_id, status=status, base=base, members=members))
        return active

    def blocking_run(self, candidate: Path) -> ActiveOutput | None:
        """The unfinished run that is about to write ``candidate``, if any."""
        for output in self.active_outputs():
            if output.claims(candidate):
                return output
        return None

    def orphaned_runs(self, removed: Iterable[Path]) -> list[JobRecord]:
        """Finished runs whose every output is in ``removed`` or already gone.

        What "clean up" means when a dataset is deleted: the record that produced it now
        describes files that do not exist, and keeping it keeps a claim nothing backs.

        A run qualifies only if it *produced* files. An analysis run has none -- its
        ``spec["dataset"]`` is an input -- so deleting a dataset never removes the ``score``
        result measured against it. That number is still a true statement about a run that
        happened, and the file it read going away does not make it false.
        """
        gone = {os.path.normcase(str(path)) for path in removed}
        orphaned: list[JobRecord] = []
        for record in self.records():
            if not record.status.terminal:
                continue
            outputs = run_output_paths(record)
            if not outputs:
                continue
            if all(os.path.normcase(str(q)) in gone or not q.exists() for q in outputs):
                orphaned.append(record)
        return orphaned

    def _force_kill(self, job_id: str) -> None:
        with self._lock:
            live = self._jobs.get(job_id)
            if live is None or live.record.status.terminal or live.process is None:
                return
            process = live.process
        _kill_tree(process)

    def shutdown(self) -> None:
        """Kill every live worker and its descendants. Called when the server stops.

        The tree matters here more than anywhere: the lifespan calls this with no grace, so
        a Ctrl-C during a sweep would otherwise strand sinter's whole pool.
        """
        with self._lock:
            processes = [live.process for live in self._jobs.values() if live.process]
        for process in processes:
            _kill_tree(process)

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
                # POSIX only: gives the worker its own session so `_kill_tree` can signal
                # the whole group -- a sweep's sinter pool included -- without also
                # signalling the server that started it. Windows walks the tree instead.
                start_new_session=sys.platform != "win32",
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
            _kill_tree(process)

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
                # `.get`, not indexing: `discard` can remove the record between the two
                # separate lock acquisitions in this function, and a KeyError raised here is
                # raised on a daemon thread -- silently, with `_pump()` never reached, so the
                # queue never starts another job again.
                surviving = self._jobs.get(job_id)
                if surviving is not None:
                    surviving.process = None
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
                # `.get` for the same reason as in `_supervise`: this reacquires the lock,
                # and the record may have been discarded in between.
                live = self._jobs.get(job_id)
                if live is not None and not live.record.status.terminal:
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
                record.total_units = int(message.get("total_units", record.total_units))
                record.progress_unit = str(message.get("unit", record.progress_unit))
                self._append_event(
                    job_id,
                    "started",
                    {"total_units": record.total_units, "unit": record.progress_unit},
                )
            elif kind == "progress":
                record.completed_units = int(message.get("completed", record.completed_units))
                # Sweep-only keys. Absent for a dataset run, and absent keys must leave the
                # previous value alone rather than blanking a readout mid-run.
                if "shots_collected" in message:
                    record.shots_collected = int(message["shots_collected"])
                if "detail" in message:
                    record.detail = str(message["detail"])
                self._append_event(
                    job_id,
                    "progress",
                    {
                        "completed": record.completed_units,
                        "shots_collected": record.shots_collected,
                        "detail": record.detail,
                    },
                )
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
                # *accepts* the `Infinity` and `NaN` tokens that `encode_line` refuses
                # to emit, so a non-finite number can still arrive here -- and a record
                # holding one serves invalid JSON to the browser and writes invalid JSON
                # to its own durable record.
                record.result = json_safe(dict(result)) if isinstance(result, dict) else None
                if record.progress_unit:
                    # Only meaningful when there was a denominator. A job whose total is
                    # unknown reports 0/0, and forcing completed to match would turn an
                    # indeterminate bar into a full one at the moment it stops mattering.
                    record.completed_units = record.total_units
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
                if not status.terminal:
                    status = JobStatus.FAILED
                    error = "the server stopped while this run was in flight"
                    error_kind = "internal"
                # Records written before progress grew a unit named the fields
                # `total_shots`/`completed_shots` and could only ever mean shots. Read
                # them under the old names rather than dropping the run from history: a
                # restart is exactly when a user goes looking for what already ran.
                record = JobRecord(
                    id=job_id,
                    mode=str(raw.get("mode", "generate")),
                    spec=dict(raw.get("spec", {})),
                    total_units=int(raw.get("total_units", raw.get("total_shots", 0))),
                    progress_unit=str(raw.get("progress_unit", "shots")),
                    status=status,
                    created_at=str(raw.get("created_at", "")),
                    started_at=raw.get("started_at"),
                    finished_at=raw.get("finished_at"),
                    completed_units=int(raw.get("completed_units", raw.get("completed_shots", 0))),
                    phase=raw.get("phase"),
                    detail=raw.get("detail"),
                    shots_collected=raw.get("shots_collected"),
                    # Defaulted, not required: records written before the analysis
                    # job layer existed have neither key, and a restart must not
                    # drop a run's history because its schema predates a feature.
                    artifacts=list(raw.get("artifacts", [])),
                    result=raw.get("result"),
                    files=[_with_kind(entry) for entry in raw.get("files", [])],
                    warnings=list(raw.get("warnings", [])),
                    error=error,
                    error_kind=error_kind,
                )
            except (ValueError, TypeError):
                continue
            self._jobs[job_id] = _Live(record=record, events=deque(maxlen=EVENT_BUFFER))
            self._order.append(job_id)

    def clean_partials(self, root: Path) -> list[Path]:
        """Remove staging directories orphaned by a killed worker, and say which."""
        return sweep_partials(root)
