"""One run, end to end: format routing, naming and staged writes included.

The layer between the builders and any front end. It exists because the CLI and the web
UI must not each re-derive three things: which builder a ``(format, shots, chunk_size)``
triple routes to, what the produced files are called, and — load-bearing — that no front
end ever writes to the path a user is going to read.

Most of this module is about *dataset* runs, which is what ``RunSpec`` means. A threshold
sweep is the one run kind here that produces no dataset: it writes a results table, a plot
and a summary sidecar, and :data:`JobSpec` is the union a front end actually dispatches
over. Keeping ``RunSpec`` narrow is deliberate — ``run()``, ``total_shots()`` and
``materialised_datasets()`` are exhaustive matches over dataset runs, and a fourth member
would quietly change what all three mean.

Two traps live in this file because of the sweep:

- **The ``qecgen.sweep`` import must stay inside** :func:`run_sweep_job`. That module pulls
  in matplotlib and sinter; at module scope here, every ``qecgen generate`` would pay for
  them. ``cli.py`` avoids the same import for the same reason.
- **:func:`sweep_partials` has nothing to do with a threshold sweep.** It sweeps (verb) the
  staging directories an interrupted run left behind. Both senses of the word now live in
  this module; the name is kept because ``ui/jobs.py``, the tests and the architecture docs
  all refer to it.

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
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation only. Importing `qecgen.sweep` at run time here would pull matplotlib and
    # sinter into every `qecgen generate`; see the module docstring.
    from qecgen.sweep import SweepPoint, SweepProgress

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


from qecgen.circuits import Basis, NoiseModel
from qecgen.dataset import DatasetMeta, DriftCondition, StructureLevel
from qecgen.environments import (
    DriftAxis,
    build_drift_environments,
    build_multi_environment,
    build_single_environment,
    drift_dataset_names,
    stream_single_environment,
)
from qecgen.exporters import get_exporter
from qecgen.sampling import DEFAULT_CHUNK_SIZE

__all__ = [
    "DISPLACED_PREFIX",
    "PARTIAL_PREFIX",
    "DriftSpec",
    "GenerateSpec",
    "JobSpec",
    "MultiEnvSpec",
    "PhaseHook",
    "ProgressHook",
    "RunCancelledError",
    "RunSpec",
    "Staging",
    "SweepArtifact",
    "SweepProgressHook",
    "SweepResult",
    "SweepSpec",
    "WrittenFile",
    "expand_range",
    "generate_drift",
    "generate_multi",
    "generate_single",
    "materialised_datasets",
    "run",
    "run_sweep_job",
    "should_stream",
    "staged",
    "sweep_partials",
    "sweep_tasks",
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
"""Called with ``"sampling"`` then ``"writing"``.

Sampling is the only phase the builders report progress for, but it is not the only slow
one: concatenation, the seeded shuffle, the content hash and gzip export all happen after
the last chunk. A front end that stops at a full bar shows a run that looks hung.
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
"""A run that produces a dataset. Deliberately does not include :class:`SweepSpec`."""


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """A threshold sweep: results table, plot and summary sidecar, no dataset.

    ``error_rates`` are already resolved. The CLI takes them as ``low:high:count`` and the
    web form as a range builder, but both expand through :func:`expand_range` before they
    get here, so the recorded spec says which rates actually ran rather than the notation
    someone typed.

    ``out`` names the **CSV**; the plot and the summary are derived from its stem. That is
    also why a ``.png`` target is refused — see :meth:`__post_init__`.
    """

    distances: tuple[int, ...]
    error_rates: tuple[float, ...]
    out: Path
    max_errors: int = 500
    max_shots: int = 100_000_000
    workers: int = 4
    decoders: tuple[str, ...] = ("pymatching",)
    noise_model: NoiseModel = NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL
    rounds: int | None = None
    basis: Basis = Basis.Z
    rotated: bool = True

    def __post_init__(self) -> None:
        """Structural checks only, so constructing a spec never imports sinter.

        Whether a *decoder name* is real, and whether its backend is installed, is
        :func:`qecgen.decoders.resolve_decoders`' job and happens at the top of
        :func:`run_sweep_job`. Asking it here would drag sinter into every import of this
        module through the spec.
        """
        if self.out.suffix.lower() == ".png":
            # The plot is written to the stem, so a .png target would have the plot
            # overwrite the data it is plotting. Caught here rather than in the CLI so the
            # web form inherits the same refusal instead of restating it.
            raise ValueError(
                f"out={self.out} would make the results table and the plot the same path, "
                "so the plot would overwrite the data. Name a .csv path; the plot is "
                "written beside it."
            )
        if not self.distances:
            raise ValueError("a sweep needs at least one distance")
        if any(distance < 2 for distance in self.distances):
            raise ValueError(f"every distance must be >= 2, got {list(self.distances)}")
        if not self.error_rates:
            raise ValueError("a sweep needs at least one error rate")
        if any(not 0.0 <= rate <= 1.0 for rate in self.error_rates):
            raise ValueError(f"every error rate must lie in [0, 1], got {list(self.error_rates)}")
        # Every axis is checked for duplicates, not just decoders. `build_tasks` emits one
        # task per (distance, rate) and sinter refuses an identical task outright --
        # `ValueError: Same task given twice:` followed by the *entire* stim circuit, which
        # then lands in a run record and is rendered in a browser. Caught here it is one
        # readable line, and it is caught before any circuit is built.
        for axis, values in (("distances", self.distances), ("error rates", self.error_rates)):
            if len(set(values)) != len(values):
                raise ValueError(f"{axis} must be distinct, got {list(values)}")
        if not self.decoders:
            raise ValueError("a sweep needs at least one decoder")
        if len(set(self.decoders)) != len(self.decoders):
            # Structural, so it needs no sinter import. sinter would happily collect the
            # same decoder twice and `write_csv` would emit duplicate rows for it; worse,
            # `sweep_tasks` would report a denominator larger than the grid actually run,
            # so the progress bar would stall at a fraction and snap to full at the end.
            raise ValueError(f"decoders must be distinct, got {list(self.decoders)}")
        for name, value in (
            ("max_errors", self.max_errors),
            ("max_shots", self.max_shots),
            ("workers", self.workers),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")


JobSpec = RunSpec | SweepSpec
"""Everything a worker can be asked to run. The union front ends dispatch over."""


def expand_range(low: float, high: float, count: int) -> list[float]:
    """``count`` evenly spaced rates from ``low`` to ``high`` inclusive.

    Shared rather than duplicated: the CLI parses ``low:high:count`` from a string and the
    web form collects three numbers, but the arithmetic that turns them into the rates a
    sweep actually runs has to be identical or the two front ends sweep different grids
    from the same numbers.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if high < low:
        raise ValueError(f"low must not exceed high, got {low}:{high}")
    if count == 1:
        return [low]
    if high == low:
        # Refused rather than returning `[low] * count`. Identical rates build identical
        # sinter tasks, which share a strong_id -- so the collection runs one task while
        # the grid claims `count`, and the progress bar sticks at 1/count for the whole
        # run. A range of zero width with more than one step is a typo, not a request.
        raise ValueError(
            f"a range of {count} steps needs a non-zero width, got {low}:{high}; "
            "pass a count of 1 to sweep a single rate"
        )
    return [low + (high - low) * index / (count - 1) for index in range(count)]


def sweep_tasks(spec: SweepSpec) -> int:
    """Tasks the sweep will collect — the denominator for its progress bar.

    sinter expands one task per (distance, rate, decoder), so this is the product. Unlike
    :func:`total_shots` there is no shot denominator to offer: ``max_errors`` is the real
    stopping condition and ``max_shots`` only a ceiling, so how many shots a sweep will
    take is not knowable before it runs.
    """
    return len(spec.distances) * len(spec.error_rates) * len(spec.decoders)


@dataclass(frozen=True, slots=True)
class SweepArtifact:
    """One file a sweep produced.

    A sweep's outputs are not datasets, so they carry none of :class:`WrittenFile`'s
    fields — there is no shot count, no content hash and no drift condition to report. A
    separate type rather than a widened ``WrittenFile`` full of nulls, because a record
    whose fields are meaningless for half its instances stops being checkable.
    """

    path: Path
    kind: str
    """``results`` (the CSV), ``plot`` (the PNG) or ``summary`` (the threshold JSON)."""


SweepProgressHook = Callable[["SweepProgress"], None]
"""Called as collection proceeds. Raising cancels the sweep, exactly as
:data:`ProgressHook` does for a dataset run."""


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What a finished sweep produced: the files, and the numbers behind them.

    Both, because the two front ends need different halves. A worker reports
    :attr:`artifacts` over a pipe; the CLI additionally prints the crossing and suppression
    summary, which needs the points themselves. Re-reading the sidecar it just wrote would
    work but would make the terminal report depend on the JSON schema rather than on the
    collection it is describing.
    """

    artifacts: list[SweepArtifact]
    points: list[SweepPoint]


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


def should_stream(format_name: str, shots: int, chunk_size: int) -> bool:
    """Whether a single-environment run can avoid materialising every shot.

    Only HDF5 has an incremental writer, and only above one chunk is there anything to
    stream. Below that the streaming path would add a resize per chunk for no benefit.
    """
    return get_exporter(format_name).streaming and shots > chunk_size


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


def run_sweep_job(
    spec: SweepSpec,
    progress: SweepProgressHook | None = None,
    on_phase: PhaseHook | None = None,
) -> SweepResult:
    """Collect a threshold sweep and stage its three outputs.

    Lifted out of the CLI so both front ends share one implementation. The pieces that
    used to sit inline in ``cli.sweep`` — resolving decoders before any circuit is built,
    deriving the plot and summary paths from the CSV stem, and committing all three
    together — are the parts a second front end would otherwise have got subtly different.

    All three files go through :func:`staged` for the same reason a dataset does: each of
    them truncates its target on open, so an interrupted sweep would destroy the previous
    sweep's results and leave a results table that still parses. Committing them together
    also means a plot never survives without the numbers behind it.

    Raises:
        ValueError: for an unknown decoder name or an absent decoder backend, before any
            circuit is constructed.
    """
    # Imported here, never at module scope: `qecgen.sweep` pulls in matplotlib and sinter.
    from qecgen.sweep import plot_threshold, run_sweep, write_csv, write_threshold_json

    plot = spec.out.with_suffix(".png")
    summary = spec.out.with_suffix(".threshold.json")

    if on_phase is not None:
        on_phase("collecting")
    points = run_sweep(
        distances=list(spec.distances),
        error_rates=list(spec.error_rates),
        max_errors=spec.max_errors,
        max_shots=spec.max_shots,
        workers=spec.workers,
        decoders=spec.decoders,
        noise_model=spec.noise_model,
        rounds=spec.rounds,
        basis=spec.basis,
        rotated=spec.rotated,
        # A front end reads progress through the hook; sinter's own printer would be
        # writing a redrawing terminal table into a pipe nobody renders.
        print_progress=progress is None,
        progress_callback=progress,
    )

    if on_phase is not None:
        on_phase("writing")
    with staged(spec.out.parent) as staging:
        write_csv(points, staging.scratch / spec.out.name)
        plot_threshold(points, staging.scratch / plot.name)
        write_threshold_json(
            points,
            staging.scratch / summary.name,
            max_errors=spec.max_errors,
            max_shots=spec.max_shots,
            noise_model=str(spec.noise_model),
            basis=str(spec.basis),
        )

    # Matched by name rather than positionally: `Staging` commits in sorted order, which
    # is not the order they were written -- the same trap `generate_drift` documents.
    # Returned results-first because that is the artifact the other two are derived from.
    committed_by_name = {path.name: path for path in staging.committed}
    artifacts = [
        SweepArtifact(path=committed_by_name[name], kind=kind)
        for name, kind in (
            (spec.out.name, "results"),
            (plot.name, "plot"),
            (summary.name, "summary"),
        )
    ]
    return SweepResult(artifacts=artifacts, points=points)


def materialised_datasets(spec: RunSpec) -> bool:
    """Whether the run holds every shot in memory at once.

    Surfaced so a front end can warn before a large run rather than after it fails.
    """
    if isinstance(spec, GenerateSpec):
        return not should_stream(spec.fmt, spec.shots, spec.chunk_size)
    return True
