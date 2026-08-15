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
from typing import IO

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
    "MultiEnvSpec",
    "PhaseHook",
    "ProgressHook",
    "RunCancelledError",
    "RunSpec",
    "Staging",
    "WrittenFile",
    "generate_drift",
    "generate_multi",
    "generate_single",
    "materialised_datasets",
    "run",
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


def materialised_datasets(spec: RunSpec) -> bool:
    """Whether the run holds every shot in memory at once.

    Surfaced so a front end can warn before a large run rather than after it fails.
    """
    if isinstance(spec, GenerateSpec):
        return not should_stream(spec.fmt, spec.shots, spec.chunk_size)
    return True
