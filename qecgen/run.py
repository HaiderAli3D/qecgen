"""One dataset-producing run: format routing, naming and staged writes included.

The layer between the builders and any front end. It exists because the CLI and the web
UI must not each re-derive three things: which builder a ``(format, shots, chunk_size)``
triple routes to, what the produced files are called, and — load-bearing — that no front
end ever writes to the path a user is going to read.

That last one is not hypothetical. Every exporter writes **in place**:
``StreamingHDF5Writer`` opens ``h5py.File(path, "w")`` at the destination, and the other
three truncate it on ``open``/``savez``/``write_table``. So an interrupted run destroys
whatever was already at that path and leaves something named like a dataset. Worse, JSONL
puts its manifest on line 1, so a truncated JSONL *reads cleanly* and only
``validate_dataset`` notices the shot count disagrees. Staging through
:func:`staged` and committing with :func:`os.replace` is what makes "the file is there"
mean "the run finished".

The specs here are plain dataclasses, not pydantic models: ``cli.py`` builds them from
typer arguments and ``qecgen.ui`` builds them from request bodies, so this module must
import neither.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

if sys.platform == "win32":
    import msvcrt

    def _try_lock(handle: IO[bytes]) -> bool:
        """Take a non-blocking exclusive lock on the first byte, False when held."""
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

else:
    import fcntl

    def _try_lock(handle: IO[bytes]) -> bool:
        """Take a non-blocking exclusive lock, False when held."""
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True


from qecgen.circuits import Basis, NoiseModel, default_rounds
from qecgen.dataset import DatasetMeta, DriftCondition, StructureLevel
from qecgen.environments import (
    DriftAxis,
    build_drift_environments,
    build_multi_environment,
    build_single_environment,
    drift_dataset_names,
    stream_single_environment,
)
from qecgen.exporters import get_exporter, infer_format
from qecgen.sampling import DEFAULT_CHUNK_SIZE

__all__ = [
    "DEFAULT_SWEEP_DECODERS",
    "DISPLACED_PREFIX",
    "PARTIAL_PREFIX",
    "SWEEP_BATCH_SECONDS",
    "AnalysisProgress",
    "AnalysisResult",
    "AnalysisSpec",
    "DriftSpec",
    "GenerateSpec",
    "JobSpec",
    "MultiEnvSpec",
    "PhaseHook",
    "ProgressHook",
    "QaSpec",
    "RunCancelledError",
    "RunSpec",
    "ScoreSpec",
    "Staging",
    "SweepSpec",
    "WrittenFile",
    "analyse",
    "generate_drift",
    "generate_multi",
    "generate_single",
    "job_total",
    "linear_rates",
    "materialised_datasets",
    "parse_range",
    "preload",
    "qa_report",
    "resolved_config",
    "resolved_rounds_note",
    "run",
    "run_threshold_sweep",
    "score_correction",
    "should_stream",
    "staged",
    "sweep_partials",
    "total_shots",
]

PARTIAL_PREFIX = ".qecgen-partial-"
"""Prefix for staging directories. Named so a stray one is obviously ours, and obviously
not a dataset."""

DISPLACED_PREFIX = ".qecgen-displaced-"
"""Prefix for the salvage directory a failed commit may leave behind.

Holds a previous dataset file that was displaced for overwriting and could not be
restored when the commit failed. Deliberately not matched by :func:`sweep_partials`:
deleting a user's old file during cleanup would be worse than leaving evidence.
"""

_LOCK_NAME = ".qecgen-lock"
"""Lock file inside every staging directory, held open by the creating process.

Liveness for :func:`sweep_partials`. Age is not a usable signal: a materialising run
leaves its scratch directory untouched for the entire sampling phase, which can be hours,
so any mtime threshold either deletes live staging or leaves real orphans for days. An
advisory lock held from creation to cleanup is unambiguous — if the probe can take the
lock, the creator is gone.
"""

ProgressHook = Callable[[int], None]
"""Called once per sampled chunk with that chunk's shot count. Increments, not totals —
the convention ``stream_single_environment`` established. Raising cancels the run."""

PhaseHook = Callable[[str], None]
"""Called with a short label for the stage a job has reached.

A generation run reports ``"sampling"`` then ``"writing"``. Sampling is the only phase the
builders report progress for, but it is not the only slow one: concatenation, the seeded
shuffle, the content hash and gzip export all happen after the last chunk. A front end
that stops at a full bar shows a run that looks hung.

Analysis jobs use it more heavily, because several of them have no useful numeric
progress at all — the phase label is the only honest signal they can give.
"""

SWEEP_BATCH_SECONDS = 10
"""Cap on how long a sinter worker collects before flushing a progress update.

sinter's own default is 120 s. Left there, a cancel can go unobserved for two minutes
against the supervisor's 10-second grace window -- so the supervisor gives up and kills
the worker instead of letting sinter tear down its pool cleanly. This also makes the bar
move on a sweep whose individual tasks are long.
"""

DEFAULT_SWEEP_DECODERS: tuple[str, ...] = ("pymatching",)
"""Decoders a sweep runs when none are named.

Duplicated from ``decoders.DEFAULT_DECODERS`` on purpose: importing that module here
would pull sinter into every worker's start-up, which is the cost `preload` exists to
keep deferred. ``tests/test_sweep.py`` asserts the two agree.
"""

AnalysisProgress = Callable[[int], None]
"""Progress for a job that is not sampling shots. Increments, like :data:`ProgressHook`.

A separate alias rather than a reuse, because ``ProgressHook``'s contract is stated as
"once per sampled chunk with that chunk's shot count" and analysis jobs count other
things — sweep tasks, QA environments. The arithmetic is identical; the promise is not,
and a docstring that quietly stops being true is how the next reader is misled.
"""


class RunCancelledError(Exception):
    """Raised out of a progress hook to stop a run at the next chunk boundary.

    An ``Exception`` rather than a ``BaseException`` so ordinary worker-level
    ``except Exception`` handling catches it. ``stream_single_environment`` catches
    ``BaseException`` and calls ``writer.abort()``, so it is handled correctly either
    way; the materialising builders have written nothing at all by that point.
    """


@dataclass(frozen=True, slots=True)
class GenerateSpec:
    """A single-environment run."""

    distance: int
    p: float
    shots: int
    seed: int
    out: Path
    fmt: str = "hdf5"
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL
    rounds: int | None = None
    basis: Basis = Basis.Z
    rotated: bool = True
    structure_level: StructureLevel = StructureLevel.NONE
    emit_mechanisms: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE


@dataclass(frozen=True, slots=True)
class MultiEnvSpec:
    """A pooled multi-environment run.

    ``axis_values`` are coordinates on ``axis``, which is only the physical error rate
    when ``axis`` is ``P``. Naming them ``error_rates`` (as the builder does) reads wrong
    for ``xz_bias``.
    """

    distance: int
    axis_values: tuple[float, ...]
    shots_per_env: int
    seed: int
    out: Path
    fmt: str = "hdf5"
    axis: DriftAxis = DriftAxis.P
    base_p: float | None = None
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL
    rounds: int | None = None
    basis: Basis = Basis.Z
    rotated: bool = True
    shuffle: bool = True
    structure_level: StructureLevel = StructureLevel.NONE
    emit_mechanisms: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE


@dataclass(frozen=True, slots=True)
class DriftSpec:
    """A drift study: one training file plus one test file per drifted value.

    ``out`` is a **directory**, unlike the other two specs.
    """

    distance: int
    train_p: float
    test_values: tuple[float, ...]
    shots: int
    seed: int
    condition: DriftCondition
    out: Path
    fmt: str = "hdf5"
    axis: DriftAxis = DriftAxis.P
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL
    rounds: int | None = None
    basis: Basis = Basis.Z
    rotated: bool = True
    structure_level: StructureLevel = StructureLevel.DEM
    emit_mechanisms: bool = False
    chunk_size: int = DEFAULT_CHUNK_SIZE


RunSpec = GenerateSpec | MultiEnvSpec | DriftSpec
"""A job that produces a dataset.

Kept separate from :data:`AnalysisSpec` rather than widened to cover it. ``total_shots``,
``should_stream``, ``materialised_datasets`` and the web UI's cost preview are all shaped
by "this produces a dataset"; a spec that reads one instead has no shots, no output
format and no structure level, and adding it here would leave every one of those
functions with a member it cannot answer for.
"""


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    """Score a supplied Pauli correction against a dataset's true observables.

    Reads; writes nothing. The correction is an **input**, not a target — see
    ``DATA_CONTRACT.md``, which is emphatic that this is not Contract C.
    """

    dataset: Path
    correction: Path
    fmt: str | None = None
    """Format override for ``dataset``. ``None`` infers it from the extension."""
    unpacked: bool = False
    """The correction arrays are bool ``(shots, n_data_qubits)`` rather than packed."""
    alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class QaSpec:
    """Statistical QA on a dataset: structural checks first, then the slow measurement.

    The two are one job rather than two because that is already the CLI's semantics —
    ``validate --qa`` refuses to spend minutes on statistics for a file whose shapes do
    not match its manifest, and prints "skipping --qa: structural checks failed first"
    instead. A second front end offering statistics without that gate would be offering a
    number measured against a file it had not checked.
    """

    dataset: Path
    fmt: str | None = None
    max_shots: int = 50_000
    target_errors: int = 100


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """A sinter threshold sweep. Writes three files, none of them a dataset.

    ``out`` names the **CSV**; the plot and the threshold sidecar are derived from it by
    replacing the suffix, exactly as the CLI has always derived them. That is why ``out``
    ending in ``.png`` is refused rather than accommodated: it would make the plot and the
    data the same path, and the plot is written second.
    """

    distances: tuple[int, ...]
    error_rates: tuple[float, ...]
    out: Path
    max_errors: int = 500
    max_shots: int = 100_000_000
    workers: int = 4
    decoders: tuple[str, ...] = ()
    """Empty means the default set. Resolved by ``decoders.resolve_decoders``, which is
    the only thing that knows which names exist and which backends are installed."""
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL
    basis: Basis = Basis.Z
    rotated: bool = True
    rounds: int | None = None
    alpha: float = 0.05

    @property
    def plot_path(self) -> Path:
        """Where the plot lands."""
        return self.out.with_suffix(".png")

    @property
    def summary_path(self) -> Path:
        """Where the threshold sidecar lands."""
        return self.out.with_suffix(".threshold.json")

    @property
    def n_tasks(self) -> int:
        """Grid size, before decoders multiply it."""
        return len(self.distances) * len(self.error_rates)


AnalysisSpec = ScoreSpec | QaSpec | SweepSpec
"""A job that reads existing files and reports on them, producing no dataset."""

JobSpec = RunSpec | AnalysisSpec
"""Anything a worker can be asked to do."""


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """What an analysis job produced.

    ``summary`` is the reportable content and ``artifacts`` the files, if any. Both are
    deliberately not ``WrittenFile``: that type describes a dataset, and an analysis
    either writes nothing at all or writes something (a plot, a sidecar) with no shot
    count, no content hash and no drift condition to put in one.
    """

    kind: str
    summary: dict[str, Any]
    artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class WrittenFile:
    """One file that a run actually produced.

    ``path`` is where the file landed, which is not always ``spec.out``: the NPZ exporter
    appends ``.npz`` to any path that lacks it. Reporting the requested path instead of
    the real one would print a filename that does not exist.
    """

    path: Path
    shots: int
    content_hash: str | None
    drift_condition: DriftCondition
    structure_source_environment_id: int | None

    @classmethod
    def from_meta(cls, path: Path, meta: DatasetMeta) -> WrittenFile:
        """Describe a written file from the manifest that went into it."""
        return cls(
            path=path,
            shots=meta.shots,
            content_hash=meta.content_hash,
            drift_condition=meta.drift_condition,
            structure_source_environment_id=meta.structure_source_environment_id,
        )


def total_shots(spec: RunSpec) -> int:
    """Shots the run will sample in total — the denominator for a progress bar.

    Known before any sampling starts for all three run kinds, which is why the hooks can
    report bare increments and leave the percentage to the caller.
    """
    match spec:
        case GenerateSpec():
            return spec.shots
        case MultiEnvSpec():
            return spec.shots_per_env * len(spec.axis_values)
        case DriftSpec():
            return spec.shots * (1 + len(spec.test_values))


def job_total(spec: JobSpec) -> tuple[int, str]:
    """The denominator a progress bar should use, and what it counts.

    Shots, for everything that produces a dataset. It exists as its own function because
    not every job a front end supervises counts shots — and a bare integer with the unit
    left implicit is how a bar ends up labelled "12 / 48 shots" for work measured in
    something else entirely.

    An empty unit means the total is not knowable in advance, which a front end should
    render as indeterminate rather than as a bar pinned at zero. Scoring is that case
    honestly: its cost is dominated by reading a dataset whose parser reports nothing,
    and inventing a denominator to fill a bar with would describe none of it.
    """
    match spec:
        case GenerateSpec() | MultiEnvSpec() | DriftSpec():
            return total_shots(spec), "shots"
        case ScoreSpec():
            return 0, ""
        case QaSpec():
            # An upper bound, not a target: the sampler stops early once it has seen
            # `target_errors` failures in an environment. So the bar deliberately stops
            # short of full rather than overshooting, and the phase says which
            # environment is running. Environment count needs the manifest, which is one
            # cheap read.
            from qecgen.exporters import read_manifest

            try:
                raw = read_manifest(spec.dataset, spec.fmt)
                environments = len(list(raw.get("environments") or [])) or 1
            except Exception:
                # An unreadable file is the job's problem to report, not this function's.
                environments = 1
            return spec.max_shots * environments, "shots"
        case SweepSpec():
            # Tasks, not shots. A sweep's shot count is decided by `max_errors` as it
            # runs, so there is no shot denominator to report -- but the grid size is
            # known before anything starts, and sinter reports per-task completion.
            decoders = len(spec.decoders or DEFAULT_SWEEP_DECODERS)
            return spec.n_tasks * decoders, "tasks"


def should_stream(format_name: str, shots: int, chunk_size: int) -> bool:
    """Whether a single-environment run can avoid materialising every shot.

    Only HDF5 has an incremental writer, and only above one chunk is there anything to
    stream. Below that the streaming path would add a resize per chunk for no benefit.
    """
    return get_exporter(format_name).streaming and shots > chunk_size


def parse_range(text: str) -> tuple[float, float, int]:
    """Parse a ``low:high:count`` sweep range.

    Here rather than in a front end because the refusals are the point. A count below 1
    used to be coerced to a single rate at ``low``, which runs a different sweep than the
    flag describes; a second front end re-deriving this could reintroduce that silently.

    Raises:
        ValueError: on a malformed string, ``count < 1``, or ``high < low``. Front ends
            restate it in their own vocabulary (``typer.BadParameter``, HTTP 400).
    """
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected low:high:count, got {text!r}")
    try:
        low, high, count = float(parts[0]), float(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(f"could not parse {text!r} as low:high:count") from None
    if count < 1:
        raise ValueError(f"count must be >= 1 in low:high:count, got {count}")
    if high < low:
        raise ValueError(f"low must not exceed high in low:high:count, got {text!r}")
    return low, high, count


def linear_rates(low: float, high: float, count: int) -> list[float]:
    """Expand a parsed range into the physical error rates a sweep will sample.

    **Linear, not logarithmic.** That is worth stating because log spacing is the usual
    choice for a threshold sweep and the difference is invisible in the output: both
    produce a plausible curve, and only the point density near the crossing differs.
    This is the existing behaviour and changing it would silently move every published
    sweep's resolution, so it is pinned here rather than left as an implementation
    detail of whichever front end expanded the range.
    """
    if count > 1:
        return [low + (high - low) * i / (count - 1) for i in range(count)]
    return [low]


def resolved_rounds_note(noise_model: NoiseModel, distance: int, rounds: int | None) -> str:
    """The rounds value a run will actually use, annotated when it was defaulted.

    A config table that echoes ``rounds: None`` records nothing: the reader cannot tell
    whether the run used ``distance`` rounds or the single round ``CODE_CAPACITY``
    forces. The printed record is supposed to be the resolved one.

    Raises:
        ValueError: for a combination the domain refuses, such as ``CODE_CAPACITY`` with
            ``rounds != 1``.
    """
    resolved = default_rounds(noise_model, distance, rounds)
    if rounds is not None:
        return str(resolved)
    if noise_model is NoiseModel.CODE_CAPACITY:
        return f"{resolved} (default: code capacity is single-round)"
    return f"{resolved} (default = distance)"


def _drift_condition_means(condition: DriftCondition) -> str:
    """What the condition actually does to the test files, in one line."""
    if condition is DriftCondition.FROZEN_PRIOR:
        return "test files ship the TRAINING environment's structure; the decoder must generalise"
    return (
        "each test file ships its OWN environment's structure; this measures a "
        "ceiling, not generalisation"
    )


def resolved_config(spec: JobSpec) -> dict[str, str]:
    """The fully resolved configuration of a run, as ordered ``setting -> value`` text.

    "Every command prints its fully resolved config before doing work, so a terminal log
    is a complete record of the run" is this tool's headline promise, and it is a promise
    about *resolved* values: the table carries things the spec does not hold, such as the
    rounds a ``None`` resolved to, the contract ``--emit-mechanisms`` selects, and the
    bit order every reader must assume.

    It lives here rather than in ``cli.py`` because the promise is not the CLI's. A
    browser that renders ``spec`` as raw JSON shows the request, not the record: it omits
    every derived value, which is exactly the set a reader cannot reconstruct. One
    function, so the two front ends cannot drift into recording different things about
    the same run.

    Raises:
        ValueError: from :func:`resolved_rounds_note`, for a refused noise/rounds
            combination. Callers resolve a config before printing it precisely so a
            config that cannot be honoured is never printed.
    """
    match spec:
        case GenerateSpec():
            return {
                "distance": str(spec.distance),
                "p": str(spec.p),
                "shots": f"{spec.shots:,}",
                **_common_config(spec),
                "out": str(spec.out),
            }
        case MultiEnvSpec():
            return {
                "distance": str(spec.distance),
                "environments": str(len(spec.axis_values)),
                "axis": str(spec.axis),
                "axis_values": str(list(spec.axis_values)),
                "base_p": str(spec.base_p),
                "shots_per_env": f"{spec.shots_per_env:,}",
                "total_shots": f"{total_shots(spec):,}",
                **_common_config(spec),
                "shuffle": (
                    "yes (seeded, derived from master seed)"
                    if spec.shuffle
                    # Not merely "no". Unshuffled means every shot of environment 0
                    # precedes every shot of environment 1, so an index-ordered
                    # train/test split tears along an environment boundary and the row
                    # index alone tells a model which environment a shot came from.
                    else "NO (shots stay grouped by environment, in axis order)"
                ),
                "out": str(spec.out),
            }
        case DriftSpec():
            return {
                "distance": str(spec.distance),
                "axis": str(spec.axis),
                "train_p": str(spec.train_p),
                "test_values": str(list(spec.test_values)),
                "condition": str(spec.condition),
                "condition_means": _drift_condition_means(spec.condition),
                "shots_per_file": f"{spec.shots:,}",
                **_common_config(spec),
                "out_dir": str(spec.out),
            }
        case ScoreSpec():
            return {
                "dataset": str(spec.dataset),
                "correction": str(spec.correction),
                "format": spec.fmt or "inferred from the extension",
                "correction_arrays": (
                    "bool over data qubits (packed here)"
                    if spec.unpacked
                    else "little-endian packed uint8"
                ),
                "alpha": str(spec.alpha),
                "operators_from": "noiseless circuit rebuilt from manifest parameters",
                "note": "scores a supplied correction; this is not Contract C",
            }
        case QaSpec():
            return {
                "dataset": str(spec.dataset),
                "format": spec.fmt or "inferred from the extension",
                "max_shots_per_environment": f"{spec.max_shots:,}",
                "target_errors": str(spec.target_errors),
                "order": "structural checks first; statistics only if they pass",
                "method": "re-samples each recorded environment and decodes with pymatching",
                "note": "reported as results, never asserted against a threshold",
            }
        case SweepSpec():
            return {
                "distances": str(list(spec.distances)),
                "rates": str([f"{rate:.4g}" for rate in spec.error_rates]),
                "max_errors": str(spec.max_errors),
                "max_shots": f"{spec.max_shots:,}",
                "workers": str(spec.workers),
                "noise_model": str(spec.noise_model),
                "rounds": "per task: distance (the memory-experiment default)",
                "basis": str(spec.basis),
                "rotated": str(spec.rotated),
                "decoders": ", ".join(spec.decoders or DEFAULT_SWEEP_DECODERS),
                # The sweep decoder and the QA oracle are different things, and the
                # natural wrong instinct once decoders are selectable is to make the
                # oracle configurable too. An oracle that can be set to a decoder under
                # test is not an oracle.
                "qa_oracle": "pymatching (qa.py only; the sweep decoder does not change it)",
                "dem_seen_by_decoders": (
                    "decomposed (sinter derives it with decompose_errors=True)"
                ),
                "out_csv": str(spec.out),
                "out_plot": str(spec.plot_path),
                "out_summary": str(spec.summary_path),
                "note": "sinter timing is throughput, NOT decoder latency",
            }


def _common_config(spec: RunSpec) -> dict[str, str]:
    """The settings every run kind records, in the order they have always printed.

    ``emit_mechanisms`` is in here rather than per-kind because it is a property of every
    ``RunSpec``. The drift table used to omit it — not because a drift run cannot emit
    mechanisms (``DriftSpec`` has always had the field, and the web UI has always been
    able to set it) but because the CLI had no flag for it, so the omission read as "not
    applicable" when it actually meant "whatever the caller passed, unrecorded".
    """
    return {
        "noise_model": str(spec.noise_model),
        "rounds": resolved_rounds_note(spec.noise_model, spec.distance, spec.rounds),
        "basis": str(spec.basis),
        "rotated": str(spec.rotated),
        "format": str(spec.fmt),
        "structure": str(spec.structure_level),
        "emit_mechanisms": str(spec.emit_mechanisms),
        "contract": "dem_mechanism" if spec.emit_mechanisms else "logical_frame",
        "seed": str(spec.seed),
        "chunk_size": f"{spec.chunk_size:,}",
        "bit_order": "little",
    }


@dataclass(slots=True)
class Staging:
    """A private directory that mirrors a destination directory.

    Write files into :attr:`scratch` under their final names; on a clean exit from
    :func:`staged` every one of them is moved into the destination and listed in
    :attr:`committed`. Reading ``committed`` rather than assuming the caller's intended
    filename is what makes the NPZ exporter's suffix rewriting harmless.
    """

    scratch: Path
    destination: Path
    committed: list[Path] = field(default_factory=list)


@contextlib.contextmanager
def staged(destination: Path) -> Iterator[Staging]:
    """Yield a staging directory whose contents move into ``destination`` on success.

    A sibling directory rather than the system temp directory because ``os.replace`` is
    only atomic within a volume; across volumes Windows degrades it to a copy, which is
    neither atomic nor cheap for a multi-gigabyte dataset. A sibling is the only way to
    guarantee the same volume.

    A staging *directory* rather than a ``.partial`` suffix because the NPZ exporter
    rewrites any path whose suffix is not ``.npz`` — ``dataset.npz.partial`` would land
    as ``dataset.npz.partial.npz``. Inside a directory every extension stays exact.

    No ``fsync``. The failure this guards is process death, against which the page cache
    is already enough; fsync would only buy protection against power loss, which is not
    the threat here.

    On any exception the staging directory is removed and nothing reaches
    ``destination`` — so a cancelled or crashed run cannot leave a file that looks
    finished, and cannot destroy a good file it was about to replace.

    The commit itself is two-phase because a bare ``os.replace`` loop is not
    all-or-nothing: with several staged files (a drift set), a mid-loop failure — a
    destination file held open by another process, exactly the Windows failure mode —
    left the earlier files committed while the cleanup then destroyed the not-yet-moved
    remainder, producing the mixed old/new set ``generate_drift`` exists to prevent.
    Files about to be overwritten are first displaced into a backup directory inside the
    staging area, then every staged file is moved in; on any failure the new files are
    removed and the displaced ones restored, so the destination reverts to the previous
    complete set. A displaced file whose restore *also* fails is salvaged to a
    ``.qecgen-displaced-*`` sibling rather than deleted with the scratch directory.
    """
    destination.mkdir(parents=True, exist_ok=True)
    scratch = destination / f"{PARTIAL_PREFIX}{secrets.token_hex(6)}"
    scratch.mkdir()
    # Held open (and locked) until cleanup: sweep_partials' liveness probe. See _LOCK_NAME.
    lock_handle = (scratch / _LOCK_NAME).open("wb")
    _try_lock(lock_handle)
    staging = Staging(scratch=scratch, destination=destination)
    try:
        yield staging
        _commit(staging, scratch, destination)
    finally:
        lock_handle.close()
        shutil.rmtree(scratch, ignore_errors=True)


def _commit(staging: Staging, scratch: Path, destination: Path) -> None:
    """Move every staged file into the destination, or revert to the previous set."""
    children = sorted(child for child in scratch.iterdir() if child.name != _LOCK_NAME)
    backup = scratch / "previous"
    backup.mkdir()
    displaced: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for child in children:
            final = destination / child.name
            if final.exists():
                saved = backup / child.name
                os.replace(final, saved)
                displaced.append((final, saved))
            os.replace(child, final)
            committed.append(final)
    except BaseException:
        # Revert: drop what landed, put back what was displaced. Best-effort by
        # necessity — the very lock that broke the commit can break the revert — so
        # anything still left in the backup afterwards is salvaged out of the scratch
        # directory before the caller's cleanup deletes it.
        for final in reversed(committed):
            with contextlib.suppress(OSError):
                os.remove(final)
        for final, saved in reversed(displaced):
            with contextlib.suppress(OSError):
                os.replace(saved, final)
        with contextlib.suppress(OSError):
            if any(backup.iterdir()):
                os.replace(backup, destination / f"{DISPLACED_PREFIX}{secrets.token_hex(6)}")
        raise
    staging.committed.extend(committed)


def sweep_partials(root: Path) -> list[Path]:
    """Remove staging directories left by runs that died, and say which.

    Returns the paths removed rather than cleaning up quietly: a staging directory that
    outlives its run means a worker was killed, and silently tidying that away hides it.

    A directory whose creator still holds the staging lock is skipped, not removed. The
    UI calls this at startup while a CLI run may be mid-write into the same data root;
    without the probe, that startup deleted the live run's staged output from under it.
    A pre-lock directory (no lock file at all) can only be an orphan and is removed.
    """
    if not root.is_dir():
        return []
    removed: list[Path] = []
    for candidate in sorted(root.rglob(f"{PARTIAL_PREFIX}*")):
        if candidate.is_dir() and not _scratch_is_live(candidate):
            shutil.rmtree(candidate, ignore_errors=True)
            removed.append(candidate)
    return removed


def _scratch_is_live(scratch: Path) -> bool:
    """True when the process that created ``scratch`` still holds its lock."""
    try:
        handle = (scratch / _LOCK_NAME).open("rb+")
    except OSError:
        return False
    with handle:
        return not _try_lock(handle)


def generate_single(
    spec: GenerateSpec,
    progress: ProgressHook | None = None,
    on_phase: PhaseHook | None = None,
) -> list[WrittenFile]:
    """Run a single-environment generation, streaming where that is possible."""
    exporter = get_exporter(spec.fmt)
    with staged(spec.out.parent) as staging:
        scratch = staging.scratch / spec.out.name
        if on_phase is not None:
            on_phase("sampling")
        if should_stream(spec.fmt, spec.shots, spec.chunk_size):
            meta = stream_single_environment(
                path=scratch,
                distance=spec.distance,
                p=spec.p,
                shots=spec.shots,
                seed=spec.seed,
                noise_model=spec.noise_model,
                rounds=spec.rounds,
                basis=spec.basis,
                rotated=spec.rotated,
                chunk_size=spec.chunk_size,
                emit_mechanisms=spec.emit_mechanisms,
                structure_level=spec.structure_level,
                progress=progress,
                on_phase=on_phase,
            )
        else:
            dataset = build_single_environment(
                distance=spec.distance,
                p=spec.p,
                shots=spec.shots,
                seed=spec.seed,
                noise_model=spec.noise_model,
                rounds=spec.rounds,
                basis=spec.basis,
                rotated=spec.rotated,
                chunk_size=spec.chunk_size,
                emit_mechanisms=spec.emit_mechanisms,
                structure_level=spec.structure_level,
                progress=progress,
            )
            if on_phase is not None:
                on_phase("writing")
            exporter.write(dataset, scratch, spec.structure_level)
            meta = dataset.meta
    return [WrittenFile.from_meta(staging.committed[0], meta)]


def generate_multi(
    spec: MultiEnvSpec,
    progress: ProgressHook | None = None,
    on_phase: PhaseHook | None = None,
) -> list[WrittenFile]:
    """Run a pooled multi-environment generation.

    Always materialises: the seeded shuffle needs every shot resident, so there is no
    streaming variant to route to.
    """
    exporter = get_exporter(spec.fmt)
    if on_phase is not None:
        on_phase("sampling")
    dataset = build_multi_environment(
        distance=spec.distance,
        error_rates=spec.axis_values,
        shots_per_env=spec.shots_per_env,
        seed=spec.seed,
        noise_model=spec.noise_model,
        rounds=spec.rounds,
        basis=spec.basis,
        rotated=spec.rotated,
        chunk_size=spec.chunk_size,
        emit_mechanisms=spec.emit_mechanisms,
        axis=spec.axis,
        base_p=spec.base_p,
        shuffle=spec.shuffle,
        structure_level=spec.structure_level,
        progress=progress,
    )
    if on_phase is not None:
        on_phase("writing")
    with staged(spec.out.parent) as staging:
        exporter.write(dataset, staging.scratch / spec.out.name, spec.structure_level)
    return [WrittenFile.from_meta(staging.committed[0], dataset.meta)]


def generate_drift(
    spec: DriftSpec,
    progress: ProgressHook | None = None,
    on_phase: PhaseHook | None = None,
) -> list[WrittenFile]:
    """Run a drift study, committing the whole set of files or none of it.

    A ``train`` file without its ``test_*`` siblings is a trap rather than a partial
    result, so every file is staged and they are moved into place together.

    The filename collision check runs *before* any sampling. The CLI historically ran it
    after building every dataset, which meant a long run could complete and then refuse
    to write.
    """
    exporter = get_exporter(spec.fmt)
    names = drift_dataset_names(spec.test_values)
    if on_phase is not None:
        on_phase("sampling")
    datasets = build_drift_environments(
        distance=spec.distance,
        train_p=spec.train_p,
        test_values=spec.test_values,
        shots=spec.shots,
        seed=spec.seed,
        condition=spec.condition,
        axis=spec.axis,
        noise_model=spec.noise_model,
        rounds=spec.rounds,
        basis=spec.basis,
        rotated=spec.rotated,
        chunk_size=spec.chunk_size,
        emit_mechanisms=spec.emit_mechanisms,
        structure_level=spec.structure_level,
        progress=progress,
    )
    if on_phase is not None:
        on_phase("writing")
    filenames = [f"{name}{exporter.extension}" for name in names]
    metas: list[DatasetMeta] = []
    with staged(spec.out) as staging:
        for filename, dataset in zip(filenames, datasets, strict=True):
            exporter.write(dataset, staging.scratch / filename, spec.structure_level)
            metas.append(dataset.meta)

    # `Staging` commits in sorted order, which is not build order -- "test_0.014" sorts
    # before "train". Pairing positionally would stamp each file with another file's
    # manifest, so match by name and return in build order.
    committed_by_name = {path.name: path for path in staging.committed}
    return [
        WrittenFile.from_meta(committed_by_name[filename], meta)
        for filename, meta in zip(filenames, metas, strict=True)
    ]


def run(
    spec: RunSpec,
    progress: ProgressHook | None = None,
    on_phase: PhaseHook | None = None,
) -> list[WrittenFile]:
    """Dispatch a spec to the right generator."""
    match spec:
        case GenerateSpec():
            return generate_single(spec, progress, on_phase)
        case MultiEnvSpec():
            return generate_multi(spec, progress, on_phase)
        case DriftSpec():
            return generate_drift(spec, progress, on_phase)


def score_correction(
    spec: ScoreSpec,
    progress: AnalysisProgress | None = None,
    on_phase: PhaseHook | None = None,
) -> AnalysisResult:
    """Score a supplied Pauli correction by its logical effect.

    Here rather than in a front end because the *whole* orchestration is load-bearing and
    was previously written down once, in ``cli.py``, where a second front end could only
    copy it. Two parts in particular:

    **The operators are rebuilt from a noiseless circuit** generated from the dataset's
    own decoder-visible manifest parameters. That is what makes scoring a ``frozen_prior``
    test file safe — the file's own error model is never read — and it is sound because
    the correction schema is a property of the code rather than of the noise, asserted by
    ``test_schema_digest_is_stable_and_noise_independent``. A copy of this rule that drifted
    to ``p=meta.p`` would still produce numbers, and they would still look reasonable.

    **The correction is checked against the schema before the dataset is read.** The
    manifest is read first, cheaply, so a width mismatch — the commonest mistake, and the
    one ``_require_packed`` exists to catch — is refused in milliseconds instead of after
    minutes of parsing a million-shot CSV.

    ``progress`` is called with the shot count once scoring completes; there is no
    intermediate signal to give, because the cost is the read and the scoring itself is a
    single vectorised pass.
    """
    import numpy as np

    from qecgen.circuits import build_circuit
    from qecgen.correction import (
        estimate_correction_logical_error_rate,
        extract_logical_operators,
        pack_correction,
    )
    from qecgen.exporters import read_manifest

    if on_phase is not None:
        on_phase("reading manifest")
    resolved = spec.fmt or infer_format(spec.dataset)
    meta = DatasetMeta.from_json_dict(read_manifest(spec.dataset, resolved))

    if on_phase is not None:
        on_phase("rebuilding operators")
    circuit, _ = build_circuit(
        meta.distance, 0.0, rounds=meta.rounds, basis=meta.basis, rotated=meta.rotated
    )
    operators = extract_logical_operators(circuit, strict_single_basis=True)

    if on_phase is not None:
        on_phase("reading correction")
    with np.load(spec.correction, allow_pickle=False) as payload:
        missing = [key for key in ("correction_x", "correction_z") if key not in payload]
        if missing:
            raise ValueError(
                f"{spec.correction} is missing {missing}; it must hold correction_x and "
                "correction_z arrays over the data qubits."
            )
        corr_x = np.asarray(payload["correction_x"])
        corr_z = np.asarray(payload["correction_z"])
    if spec.unpacked:
        corr_x = pack_correction(corr_x)
        corr_z = pack_correction(corr_z)

    # Both checks before the dataset read, and both name the number they expected. The
    # domain raises for either of these anyway -- from inside `estimate_...`, after the
    # whole file has been materialised.
    expected = operators.schema.packed_width
    if corr_x.ndim != 2 or corr_x.shape[1] != expected:
        raise ValueError(
            f"correction arrays are {corr_x.shape} but this dataset needs "
            f"(shots, {expected}) over {operators.schema.n_data_qubits} data qubits. "
            "Pass --unpacked if they are bool arrays over the data qubits."
        )
    if corr_x.shape[0] != meta.shots:
        raise ValueError(
            f"the correction holds {corr_x.shape[0]} shots but {spec.dataset.name} holds "
            f"{meta.shots}; every shot needs a correction."
        )

    if on_phase is not None:
        on_phase("reading dataset")
    dataset = get_exporter(resolved).read(spec.dataset)

    if on_phase is not None:
        on_phase("scoring")
    estimate = estimate_correction_logical_error_rate(
        corr_x, corr_z, dataset.observables, operators, alpha=spec.alpha
    )
    if progress is not None:
        progress(estimate.shots)

    return AnalysisResult(
        kind="score",
        summary={
            "dataset": str(spec.dataset),
            # Reported together, and neither is decoration. The digest identifies the
            # qubit ordering the score was computed under -- the same arrays under a
            # different ordering give a different, equally plausible number -- and the
            # content hash identifies the shots it was computed against.
            "schema_digest": estimate.schema_digest,
            "content_hash": meta.content_hash,
            "drift_condition": str(meta.drift_condition),
            "n_data_qubits": estimate.n_data_qubits,
            "n_observables": estimate.n_observables,
            "shots": estimate.shots,
            "failures": estimate.failures,
            "logical_error_rate": estimate.interval.point,
            "ci_low": estimate.interval.low,
            "ci_high": estimate.interval.high,
            "alpha": spec.alpha,
            "operators_from": "noiseless circuit rebuilt from manifest parameters",
            "contract": "scores a supplied correction; this is not Contract C",
        },
    )


def preload(spec: JobSpec) -> None:
    """Import everything :func:`run` or :func:`analyse` will need for ``spec``.

    Exists for one reason, and it is a Windows deadlock rather than a speed tweak.

    Measured here: a process with **any** thread blocked reading stdin cannot afterwards
    complete a large DLL-loading import on the main thread. ``import scipy.linalg`` never
    returns; ``import decimal`` and ``import xml.dom.minidom`` are unaffected, and the
    reader style makes no difference — raw ``os.read`` and ``sys.stdin.readline`` deadlock
    identically. Closing stdin lets the same import finish in 1.5 s.

    The worker parks exactly such a thread, to watch for a cancel message, and it parks it
    for the whole run. So every heavy import a spec will trigger has to happen *before*
    that thread starts. Generation was never affected because ``qecgen.run`` pulls all of
    its dependencies in at module scope; an analysis spec that imports lazily inside
    :func:`analyse` hangs with no output at all, which the supervisor can only report as a
    run that never finished.

    An exhaustive ``match`` so a new analysis kind cannot quietly reintroduce it — the
    failure has no symptom other than silence.
    """
    match spec:
        case GenerateSpec() | MultiEnvSpec() | DriftSpec():
            pass  # qecgen.run's own module-level imports already cover these.
        case ScoreSpec():
            import qecgen.correction
        case QaSpec():
            import qecgen.qa
        case SweepSpec():
            import qecgen.sweep  # noqa: F401  (sinter and matplotlib: the heaviest)


def qa_report(
    spec: QaSpec,
    progress: AnalysisProgress | None = None,
    on_phase: PhaseHook | None = None,
) -> AnalysisResult:
    """Validate a dataset structurally, then measure each environment's logical rate.

    The order is the whole design, and it is ``cli.validate --qa``'s order: statistics
    measured against a file whose arrays disagree with its manifest are a number about
    nothing, so a structural failure stops here and says so rather than spending minutes
    producing one.

    The measurement **re-samples**; it does not read the file's shots. Each environment is
    rebuilt from its recorded axis and axis value and decoded with PyMatching, which is
    why this is slow and why ``qa.py`` is separate from ``validate.py`` — a deterministic
    check must never fail for sampling reasons.
    """
    from qecgen.qa import estimate_environment_rates
    from qecgen.validate import validate_dataset

    if on_phase is not None:
        on_phase("reading dataset")
    resolved = spec.fmt or infer_format(spec.dataset)
    dataset = get_exporter(resolved).read(spec.dataset)

    if on_phase is not None:
        on_phase("structural checks")
    report = validate_dataset(dataset)
    checks = [
        {
            "name": check.name,
            "passed": check.passed,
            "detail": check.detail,
            "requirement": check.requirement,
        }
        for check in report.results
    ]
    if not report.ok:
        return AnalysisResult(
            kind="qa",
            summary={
                "dataset": str(spec.dataset),
                "ok": False,
                "checks": checks,
                "environments": [],
                "skipped": (
                    f"{len(report.failures)} structural check(s) failed, so the statistical "
                    "measurement was not run: a rate measured against a file whose arrays "
                    "disagree with its manifest describes nothing."
                ),
            },
        )

    if on_phase is not None:
        on_phase("sampling")
    measured = estimate_environment_rates(
        dataset.meta,
        max_shots=spec.max_shots,
        target_errors=spec.target_errors,
        progress=progress,
        on_phase=on_phase,
    )
    return AnalysisResult(
        kind="qa",
        summary={
            "dataset": str(spec.dataset),
            "ok": True,
            "checks": checks,
            "environments": [
                {
                    "environment_id": env.environment_id,
                    "axis": str(env.axis),
                    "axis_value": env.axis_value,
                    "p": env.p,
                    "logical_error_rate": estimate.interval.point,
                    "ci_low": estimate.interval.low,
                    "ci_high": estimate.interval.high,
                    "failures": estimate.interval.successes,
                    "shots": estimate.interval.trials,
                    "detection_event_rate": estimate.detection_event_rate,
                }
                for env, estimate in measured
            ],
            "skipped": None,
            # Carried with the numbers, not left to a front end to remember to say. The
            # commonly quoted 0.5-1% threshold depends on channel convention, rounds,
            # basis and decoder, so these are results rather than verdicts.
            "reported_not_asserted": (
                "Measured by re-sampling each environment and decoding with PyMatching. "
                "Reported as results, never asserted against a threshold."
            ),
        },
    )


def run_threshold_sweep(
    spec: SweepSpec,
    progress: AnalysisProgress | None = None,
    on_phase: PhaseHook | None = None,
) -> AnalysisResult:
    """Collect a threshold sweep and commit its three files together.

    ``qecgen.sweep`` is imported **here**, not at module scope, because it pulls in sinter
    and matplotlib — and this module is imported by every worker, including the ones that
    only generate. :func:`preload` is what makes that import safe to defer.

    The ``.png`` refusal is a domain rule, not a CLI one: it stops the plot overwriting
    the data, and it is checked before any collection starts, the way
    ``generate_drift`` checks for filename collisions before it samples anything.

    All three outputs go through one :func:`staged` block, so a sweep that fails or is
    cancelled leaves nothing rather than a CSV without its plot. Note the ordering:
    collection finishes *before* staging opens, so a cancelled sweep never had a staging
    directory at all.
    """
    from qecgen.sweep import (
        censored_points,
        plot_threshold,
        run_sweep,
        threshold_summary,
        write_csv,
        write_threshold_json,
    )

    if spec.out.suffix.lower() == ".png":
        raise ValueError(
            f"out={spec.out} would make the CSV and the plot the same path, so the plot "
            "would overwrite the data. Name a .csv path; the plot is written beside it."
        )
    if not spec.distances or not spec.error_rates:
        raise ValueError("a sweep needs at least one distance and one error rate")

    if on_phase is not None:
        on_phase("collecting")

    reported = 0

    def on_sweep_progress(started: int, message: str) -> None:
        nonlocal reported
        if on_phase is not None and message:
            on_phase(message)
        if progress is not None:
            # The hook takes increments and sinter reports a running total, so only the
            # difference is forwarded. Raising from `progress` is what cancels the sweep.
            progress(max(0, started - reported))
        reported = max(reported, started)

    points = run_sweep(
        distances=list(spec.distances),
        error_rates=list(spec.error_rates),
        max_errors=spec.max_errors,
        max_shots=spec.max_shots,
        workers=spec.workers,
        decoders=spec.decoders or DEFAULT_SWEEP_DECODERS,
        noise_model=spec.noise_model,
        rounds=spec.rounds,
        basis=spec.basis,
        rotated=spec.rotated,
        # Off, always, when a hook is supplied. sinter writes an ANSI-wrapped status line
        # to stderr, and a worker's stderr is a 50-line ring buffer the supervisor reports
        # as the failure reason -- so leaving this on makes a crashed sweep report a
        # progress bar as its cause of death.
        print_progress=progress is None and on_phase is None,
        progress=on_sweep_progress,
        # Bounded so a cancel is noticed inside the supervisor's grace window rather than
        # up to sinter's 120-second default flush period later.
        max_batch_seconds=SWEEP_BATCH_SECONDS,
    )

    if on_phase is not None:
        on_phase("writing")
    summary = threshold_summary(
        points,
        max_errors=spec.max_errors,
        max_shots=spec.max_shots,
        noise_model=str(spec.noise_model),
        basis=str(spec.basis),
        alpha=spec.alpha,
    )
    with staged(spec.out.parent) as staging:
        write_csv(points, staging.scratch / spec.out.name)
        plot_threshold(points, staging.scratch / spec.plot_path.name)
        write_threshold_json(
            points,
            staging.scratch / spec.summary_path.name,
            max_errors=spec.max_errors,
            max_shots=spec.max_shots,
            noise_model=str(spec.noise_model),
            basis=str(spec.basis),
            alpha=spec.alpha,
        )

    censored = censored_points(points, spec.max_errors, spec.max_shots)
    return AnalysisResult(
        kind="sweep",
        summary={
            **summary,
            "csv": str(spec.out),
            "plot": str(spec.plot_path),
            "n_points": len(points),
            "points": [
                {
                    "decoder": point.decoder,
                    "distance": point.distance,
                    "p": point.p,
                    "shots": point.shots,
                    "errors": point.errors,
                    "rate": point.rate,
                }
                for point in points
            ],
            "censored": [
                {
                    "decoder": point.decoder,
                    "distance": point.distance,
                    "p": point.p,
                    "shots": point.shots,
                    "errors": point.errors,
                }
                for point in censored
            ],
        },
        artifacts=tuple(staging.committed),
    )


def analyse(
    spec: AnalysisSpec,
    progress: AnalysisProgress | None = None,
    on_phase: PhaseHook | None = None,
) -> AnalysisResult:
    """Dispatch an analysis spec, the way :func:`run` dispatches a generation spec.

    Exhaustive ``match`` with no default, so a new member of :data:`AnalysisSpec` is a
    type error here rather than a silent fall-through.
    """
    match spec:
        case ScoreSpec():
            return score_correction(spec, progress, on_phase)
        case QaSpec():
            return qa_report(spec, progress, on_phase)
        case SweepSpec():
            return run_threshold_sweep(spec, progress, on_phase)


def materialised_datasets(spec: RunSpec) -> bool:
    """Whether the run holds every shot in memory at once.

    Surfaced so a front end can warn before a large run rather than after it fails.

    A ``match`` with no default rather than the ``isinstance`` chain this used to be. The
    old form fell through to ``return True``, so a new member of :data:`RunSpec` would
    have been reported as materialising with no type error anywhere — the one dispatch in
    this module mypy could not protect.
    """
    match spec:
        case GenerateSpec():
            return not should_stream(spec.fmt, spec.shots, spec.chunk_size)
        case MultiEnvSpec() | DriftSpec():
            return True
