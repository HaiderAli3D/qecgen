"""Removing an artifact from a data root, as the set of files it actually is.

Deletion in this tool is never "unlink the path the user clicked", for two reasons that
both end in a well-formed file describing something untrue.

**An artifact is often several files.** ``ml_csv`` writes a table plus up to three JSON
sidecars, and its ``.ml.manifest.json`` is that format's magic line: remove the table alone
and three orphan JSONs remain; remove the manifest alone and a real dataset is demoted to a
*not a qecgen dataset* row, indistinguishable from somebody else's CSV. A sweep is a
``.csv``/``.png``/``.threshold.json`` triple sharing one stem, and :mod:`qecgen.ui.sweeps`
keys its listing on the sidecar -- so removing the results table alone leaves a sweep the
browser still lists with nothing to draw. A drift study is a directory whose members only
mean anything together; :func:`qecgen.run.generate_drift` exists to guarantee that a mixed
old/new set cannot occur, and a per-file delete would manufacture one.

**A data root holds names that are not artifacts.** ``.qecgen-partial-*`` is live staging,
``.qecgen-displaced-*`` is the salvaged only-copy of a file an interrupted overwrite could
not put back, ``.qecgen-lock`` is the liveness lock the first of those is probed with.
:func:`qecgen.ui.datasets.resolve_within` admits every one of them -- it was written to
confine reads and writes, and confinement is not authorisation to destroy.

So the module is two steps, and front ends must use both: :func:`plan_deletion` names the
whole set and refuses what must not go, and :func:`execute` removes it and reports one
outcome per file. The plan exists so a confirmation can show the set *before* it is asked
for, which is this feature's version of the house rule that every command prints its
resolved config before doing work.

**Files go to the operating system's recycle bin, and that is not optional.** There is no
``except ImportError: os.remove`` fallback and there must never be one: a command that
recycles or destroys depending on what happened to be installed, with no way for its own
message to say which, is the archetypal failure this project is organised against. What the
recycle bin does *not* buy is a promise -- see :data:`RECYCLE_CAVEAT`.
"""

from __future__ import annotations

import enum
import functools
import importlib.util
import json
import os
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qecgen.exporters import (
    NotAQecgenDatasetError,
    get_exporter,
    match_extension,
    read_manifest,
)
from qecgen.run import (
    DISPLACED_PREFIX,
    LOCK_NAME,
    PARTIAL_PREFIX,
    SWEEP_PLOT_SUFFIX,
    SWEEP_SUMMARY_SUFFIX,
    scratch_is_live,
)

__all__ = [
    "RECYCLE_CAVEAT",
    "DeletionPlan",
    "DeletionRefusedError",
    "DeletionReport",
    "Disposition",
    "Outcome",
    "PlannedFile",
    "RefusalReason",
    "TargetKind",
    "deletion_support",
    "execute",
    "plan_deletion",
    "send_to_trash",
]


RECYCLE_CAVEAT = (
    "Files are handed to the operating system's recycle bin. Windows deletes a file "
    "permanently, with no prompt and no signal, when it is larger than the Recycle Bin's "
    "capacity for that volume, and a volume with no recycle bin at all -- a network share, "
    "some removable media -- behaves the same way. qecgen routinely writes multi-gigabyte "
    "datasets, so that is the ordinary case rather than the edge one. Nothing here can tell "
    "which happened, so check the recycle bin before relying on a file being recoverable."
)
"""Carried on every plan and every report, unconditionally.

Not a warning badge, and not conditional on size. Both of those are deliberate: this is a
description of what the operation *does*, so it belongs beside the operation once, the way a
docstring does -- rather than firing at a threshold. A caveat shown only above some byte
count teaches the reader that its absence means "safe", which is a promise this code cannot
keep. ``FOF_NOCONFIRMATION`` is precisely what makes Windows delete an oversize file outright
instead of prompting, and the bin's capacity is a per-volume setting that is not reliably
readable from here: the registry value is absent until a user changes it, the default is
computed by an undocumented formula that has changed between Windows versions, and group
policy can override both. A prediction built on that is wrong in both directions, and a false
"this will be deleted permanently" sends someone hunting for a backup they do not need.

The number that *does* vary is :attr:`DeletionPlan.total_bytes`. That is the signal; this
sentence is the explanation.
"""


class TargetKind(enum.StrEnum):
    """What a path turned out to belong to."""

    DATASET = "dataset"
    """A file this tool wrote, plus every sidecar its format writes beside it."""

    SWEEP = "sweep"
    """The three files ``qecgen sweep`` writes under one stem, reached from any of them."""

    DRIFT_SET = "drift_set"
    """A directory holding a train file and one test file per drifted value."""

    PLAIN_FILE = "plain_file"
    """One file with nothing else on disk depending on it.

    Includes a file carrying a registered dataset extension that this tool did not write --
    ``qecgen score`` reads a proposed correction from an ``.npz``, and a data root may hold
    any ``.parquet`` at all. Those have no companions, and inventing some from the extension
    alone would delete a user's unrelated file.
    """


class RefusalReason(enum.StrEnum):
    """Why a path will not be removed. Machine-readable so a front end need not parse prose."""

    NOT_FOUND = "not_found"
    STAGING_LIVE = "staging_live"
    STAGING_ORPHAN = "staging_orphan"
    DISPLACED_SALVAGE = "displaced_salvage"
    RESERVED_NAME = "reserved_name"
    PROTECTED_ROOT = "protected_root"
    RUN_RECORD = "run_record"
    COMPANION_OF_DATASET = "companion_of_dataset"
    UNRECOGNISED_DIRECTORY = "unrecognised_directory"
    CONTAINS_LINK = "contains_link"


class DeletionRefusedError(ValueError):
    """This tool will not remove that path, and the reason is machine-readable.

    A :class:`ValueError` subclass for the same reason
    :class:`~qecgen.exporters.base.NotAQecgenDatasetError` is one: the CLI's
    ``typer.BadParameter`` wrapper and the UI's ``except ValueError -> HTTPException``
    boundary both already handle it, so neither front end needs a new arm. ``reason`` rides
    alongside the message so the UI can pick a status code and the browser can pick wording
    without matching on strings.
    """

    def __init__(self, reason: RefusalReason, path: Path, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = path


class Outcome(enum.StrEnum):
    """What became of one path. Every member states a fact this code observed."""

    REMOVED = "removed"
    """The recycle request returned and the path is gone.

    Deliberately not ``recycled``. Windows permanently deletes a file too large for the
    volume's Recycle Bin and reports success either way -- see :data:`RECYCLE_CAVEAT` --
    so "the path is gone" is the whole of what is knowable. It is *checked* after the call
    rather than inferred from the call returning, because ``send2trash`` picks its Windows
    backend at import time and the two differ in what they raise.
    """

    ALREADY_MISSING = "already_missing"
    """Named by the plan and absent when the plan was built. Never handed to the OS.

    Filtered before the call rather than after the error: ``send2trash``'s Windows fallback
    calls ``get_short_path_name`` first, which raises for a path that does not exist, so an
    unfiltered call turns "already gone" into a traceback whose type depends on which
    backend was picked.
    """

    VANISHED = "vanished"
    """Present when the plan was built, gone before :func:`execute` reached it.

    Distinct from :attr:`ALREADY_MISSING` because it means the plan was raced -- a real
    window, since a confirmation dialog can sit open for minutes -- and because reporting it
    as :attr:`REMOVED` would claim this code did something it did not do.
    """

    SKIPPED = "skipped"
    """Not attempted, because the anchor failed. See :func:`execute`."""

    LOCKED = "locked"
    """The OS refused: another program holds the file open.

    Windows opens files without ``FILE_SHARE_DELETE`` by default, including from Python's
    own ``open``, so a spreadsheet or a stray reader is enough. The user action is to close
    it and ask again, which is why this is not folded into :attr:`FAILED`.
    """

    NO_TRASH = "no_trash"
    """The volume's trash could not be used. Nothing was deleted.

    A different user action from :attr:`LOCKED`: no amount of closing files helps.
    """

    FAILED = "failed"
    """Anything else. :attr:`Disposition.error` carries the exception verbatim.

    Same shape as ``DatasetEntry.unreadable``: the honest answer to an unclassified failure
    is the failure.
    """


@dataclass(frozen=True, slots=True)
class PlannedFile:
    """One file a deletion would remove."""

    path: Path
    size_bytes: int
    exists: bool
    role: str
    """What this file is, for a confirmation to show: ``dataset``, ``manifest sidecar``,
    ``results table``, ``plot``, ``summary``, ``drift member``, ``file``."""

    reason: str
    """One sentence saying why it is in the list."""

    def to_json_dict(self, root: Path | None = None) -> dict[str, Any]:
        """JSON-safe view, root-relative when ``root`` is given."""
        return {
            "path": _display(self.path, root),
            "name": self.path.name,
            "size_bytes": self.size_bytes,
            "exists": self.exists,
            "role": self.role,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DeletionPlan:
    """Everything that will be removed, named before anything is."""

    kind: TargetKind
    requested: Path
    """What the caller passed.

    Not always the anchor: a sweep is reachable from its plot, and the Datasets page lists a
    sweep's *results table*, so the path a user clicked is frequently not the file the set is
    identified by. Reported so a front end can say "you asked for sweep.png; this is a sweep"
    rather than silently widening.
    """

    anchor: Path
    """The file every listing keys on. Removed first -- see :func:`execute`."""

    files: tuple[PlannedFile, ...]
    """Every member, anchor first. Missing members are included, flagged ``exists=False``."""

    reason: str
    """One sentence saying why these files travel together."""

    @property
    def present(self) -> tuple[PlannedFile, ...]:
        """The members actually on disk, in the order :func:`execute` will attempt them."""
        return tuple(entry for entry in self.files if entry.exists)

    @property
    def missing(self) -> tuple[PlannedFile, ...]:
        """Named by the rule that built this plan, but not on disk.

        Reported rather than filtered away. ``ml_csv`` at ``structure_level=none`` never has
        two of its three sidecars, so a plan that hid them would be indistinguishable from a
        plan that forgot them -- and after a partial failure this is exactly the field that
        shows which half of a set is already gone.
        """
        return tuple(entry for entry in self.files if not entry.exists)

    @property
    def total_bytes(self) -> int:
        """Summed over members that exist."""
        return sum(entry.size_bytes for entry in self.files if entry.exists)

    def to_json_dict(self, root: Path | None = None) -> dict[str, Any]:
        """JSON-safe view, root-relative when ``root`` is given."""
        return {
            "kind": str(self.kind),
            "requested": _display(self.requested, root),
            "anchor": _display(self.anchor, root),
            "files": [entry.to_json_dict(root) for entry in self.files],
            "missing": [_display(entry.path, root) for entry in self.missing],
            "n_files": len(self.present),
            "total_bytes": self.total_bytes,
            "reason": self.reason,
            "caveat": RECYCLE_CAVEAT,
        }

    @staticmethod
    def merge(plans: Iterable[DeletionPlan]) -> DeletionPlan | None:
        """One plan covering several, deduplicated. ``None`` when there is nothing to do.

        What collapses a sweep run's three artifacts: planning each of ``.csv``, ``.png``
        and ``.threshold.json`` yields the same triple three times, so nine entries merge to
        three. Keyed on ``os.path.normcase`` because Windows paths differing only in case are
        the same file, and ``Path.resolve()`` does not reliably normalise case for a path
        that no longer exists.
        """
        ordered: list[DeletionPlan] = list(plans)
        if not ordered:
            return None
        first = ordered[0]
        seen: dict[str, PlannedFile] = {}
        for plan in ordered:
            for entry in plan.files:
                seen.setdefault(os.path.normcase(str(entry.path)), entry)
        kinds = {plan.kind for plan in ordered}
        kind = first.kind if len(kinds) == 1 else TargetKind.PLAIN_FILE
        reason = (
            first.reason
            if len(ordered) == 1
            else f"{len(ordered)} artifacts named together by one request"
        )
        return DeletionPlan(
            kind=kind,
            requested=first.requested,
            anchor=first.anchor,
            files=tuple(seen.values()),
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class Disposition:
    """One path's fate."""

    path: Path
    outcome: Outcome
    size_bytes: int
    error: str | None = None

    def to_json_dict(self, root: Path | None = None) -> dict[str, Any]:
        """JSON-safe view, root-relative when ``root`` is given."""
        return {
            "path": _display(self.path, root),
            "name": self.path.name,
            "outcome": str(self.outcome),
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


_GONE = frozenset({Outcome.REMOVED, Outcome.ALREADY_MISSING, Outcome.VANISHED})


@dataclass(frozen=True, slots=True)
class DeletionReport:
    """What :func:`execute` actually did, per file."""

    plan: DeletionPlan
    dispositions: tuple[Disposition, ...]

    @property
    def complete(self) -> bool:
        """True when every path the plan named is gone.

        A caller must not compute success from the absence of an exception: :func:`execute`
        raises only for a refusal, never for a file it could not remove.
        """
        return all(entry.outcome in _GONE for entry in self.dispositions)

    @property
    def bytes_removed(self) -> int:
        """Summed over the files actually removed."""
        return sum(e.size_bytes for e in self.dispositions if e.outcome is Outcome.REMOVED)

    @property
    def failures(self) -> tuple[Disposition, ...]:
        """The members that are still on disk."""
        return tuple(entry for entry in self.dispositions if entry.outcome not in _GONE)

    def to_json_dict(self, root: Path | None = None) -> dict[str, Any]:
        """JSON-safe view, root-relative when ``root`` is given."""
        return {
            "kind": str(self.plan.kind),
            "files": [entry.to_json_dict(root) for entry in self.dispositions],
            "removed": [
                _display(e.path, root) for e in self.dispositions if e.outcome is Outcome.REMOVED
            ],
            "failed": [
                entry.to_json_dict(root)
                for entry in self.dispositions
                if entry.outcome not in _GONE
            ],
            "n_removed": sum(1 for e in self.dispositions if e.outcome is Outcome.REMOVED),
            "bytes_removed": self.bytes_removed,
            "complete": self.complete,
            "caveat": RECYCLE_CAVEAT,
        }


def _display(path: Path, root: Path | None) -> str:
    """``path`` as a front end should show it: root-relative with forward slashes.

    Falls back to the absolute path when it is not under ``root`` -- a run record adopted
    from a previous ``--data-root`` names files elsewhere, and silently printing a bare
    filename for one of those would suggest it sits in the directory being browsed.
    """
    if root is not None:
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            pass
    return str(path)


def send_to_trash(path: Path) -> None:
    """Hand one path to the operating system's recycle bin.

    A module-level indirection with one job: being patchable. Without it every test that
    exercises a real deletion moves files into the developer's actual recycle bin, and a CI
    container with no trash directory fails the suite for a reason that has nothing to do
    with the code under test.

    Imported inside the function so that importing this module -- which
    :mod:`qecgen.ui.app` does at startup, and :mod:`qecgen.cli` does for one subcommand --
    never depends on the backend resolving. :func:`deletion_support` reports that separately
    without importing anything.
    """
    from send2trash import send2trash

    send2trash(os.fspath(path))


@functools.cache
def deletion_support() -> dict[str, Any]:
    """Whether deletion can work here, without importing the backend.

    ``find_spec`` rather than an import, matching how :mod:`qecgen.decoders` probes its
    optional backends: a front end asks this to decide whether to offer the control at all,
    and that question must not cost a DLL-loading import on the request path.

    Cached, so a package installed *after* the server started is not seen until it restarts.
    That is the same caveat the decoder probe carries and it is acceptable for the same
    reason: the answer changes only when the environment does.
    """
    found = importlib.util.find_spec("send2trash") is not None
    return {
        "available": found,
        "destination": "recycle bin",
        "problem": None if found else "send2trash is not installed",
    }


# A run record's filename. Twelve hex characters is `uuid.uuid4().hex[:12]`, which is what
# `JobStore.submit` names them; the JSON's own `id` is checked against it before refusing, so
# a user's unrelated `runs/` directory is not caught by the shape alone.
_RUN_RECORD_NAME = re.compile(r"^[0-9a-f]{12}\.json$")

_DRIFT_TRAIN = "train"
_DRIFT_TEST = "test_"


def _stat_size(path: Path) -> int:
    """Size in bytes, or 0 when the path is unreadable. Never raises."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _planned(path: Path, role: str, reason: str) -> PlannedFile:
    """One member, with its existence and size resolved now rather than at execute time."""
    exists = path.exists()
    return PlannedFile(
        path=path,
        size_bytes=_stat_size(path) if exists else 0,
        exists=exists,
        role=role,
        reason=reason,
    )


def _refuse(reason: RefusalReason, path: Path, message: str) -> DeletionRefusedError:
    return DeletionRefusedError(reason, path, message)


def _check_reserved(path: Path, reserved: Collection[Path]) -> None:
    """Refuse the names a data root holds that are not artifacts.

    Runs before anything looks at content, and walks every ancestor rather than only the
    leaf: the dangerous path is not ``.qecgen-partial-abc`` itself but the ordinary-looking
    ``.qecgen-partial-abc/dataset.h5`` inside it, which is a live run's staged output.
    """
    for part in (path, *path.parents):
        name = part.name
        if name.startswith(PARTIAL_PREFIX):
            if scratch_is_live(part):
                raise _refuse(
                    RefusalReason.STAGING_LIVE,
                    path,
                    f"{name} is a staging directory a run is writing into right now. "
                    "Cancel that run before deleting anything under it; until it commits, "
                    "these bytes are not a dataset and removing them destroys its output.",
                )
            raise _refuse(
                RefusalReason.STAGING_ORPHAN,
                path,
                f"{name} is staging left behind by a run that died. "
                "`qecgen.run.sweep_partials` removes those and `qecgen ui` calls it at "
                "startup -- that is the right tool, because a staging directory's contents "
                "are mid-write by definition, so recycling them buys nothing recoverable "
                "and spends recycle-bin capacity that would otherwise hold a real file.",
            )
        if name.startswith(DISPLACED_PREFIX):
            raise _refuse(
                RefusalReason.DISPLACED_SALVAGE,
                path,
                f"{name} holds the previous version of a dataset, salvaged when a write "
                "failed. This tool has already lost data at that path once, so it will not "
                "delete the evidence for you. Look at what is in there, then remove it "
                "yourself if you are sure.",
            )
    if path.name == LOCK_NAME:
        raise _refuse(
            RefusalReason.RESERVED_NAME,
            path,
            f"{LOCK_NAME} is the advisory lock that tells `sweep_partials` whether a "
            "staging directory is still live. Deleting a live one makes the next cleanup "
            "remove a running job's staged output.",
        )
    for protected in reserved:
        if path == protected or protected in path.parents:
            raise _refuse(
                RefusalReason.PROTECTED_ROOT,
                path,
                f"{path.name} is inside {protected.name}, where the UI keeps its run "
                "records. Delete the run from the Runs page instead, which also forgets it "
                "in memory.",
            )
    if _RUN_RECORD_NAME.match(path.name) and path.parent.name == "runs" and _is_run_record(path):
        raise _refuse(
            RefusalReason.RUN_RECORD,
            path,
            f"{path.name} is a qecgen run record. Delete the run from the Runs page, which "
            "also forgets it in memory; removing the file alone leaves a running UI showing "
            "a run that is no longer on disk.",
        )


def _is_run_record(path: Path) -> bool:
    """True when the JSON at ``path`` identifies itself as the run its filename names.

    The third leg of the run-record test, and the one that makes it provable rather than a
    path convention: a user's own ``runs/`` directory holding a twelve-hex-character JSON
    is not caught unless the file's own ``id`` agrees with its name.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("id") == path.stem


def _dataset_stems(directory: Path) -> set[str]:
    """Extension-free names of the dataset files directly inside ``directory``."""
    stems: set[str] = set()
    for child in directory.iterdir():
        if not child.is_file():
            continue
        extension = match_extension(child)
        if extension is not None:
            exporter_extension = get_exporter(extension).extension
            stems.add(child.name[: -len(exporter_extension)])
    return stems


def _is_drift_directory(directory: Path) -> bool:
    """True when ``directory`` holds what :func:`qecgen.environments.drift_dataset_names` names.

    ``train`` plus at least one ``test_*``, which is exactly the pair that function produces
    and the pair whose separation makes a drift study meaningless.
    """
    try:
        stems = _dataset_stems(directory)
    except OSError:
        return False
    return _DRIFT_TRAIN in stems and any(stem.startswith(_DRIFT_TEST) for stem in stems)


def _contains_link(directory: Path) -> Path | None:
    """The first symlink or junction under ``directory``, if any."""
    for child in directory.rglob("*"):
        if child.is_symlink():
            return child
    return None


def _is_sweep_table(path: Path) -> bool:
    """True when a ``.csv`` beside a summary is a sweep's results table, not a dataset.

    Provable in both directions rather than heuristic, and this is the one place the
    ambiguity has to be resolved: ``.csv`` is both a dataset extension and what ``qecgen
    sweep`` writes its results to. The magic line either is there or it is not --
    :class:`~qecgen.exporters.base.NotAQecgenDatasetError` exists for exactly this
    distinction, and the CSV manifest reader stops at the first non-comment line, so the
    probe is bounded. A dataset that merely happens to share a stem with a sweep therefore
    stays a dataset.
    """
    try:
        read_manifest(path)
    except NotAQecgenDatasetError:
        return True
    except Exception:
        # Unreadable for some other reason: a corrupt or half-written qecgen CSV, which is
        # a dataset that needs cleaning up, not a sweep table.
        return False
    return False


def _sweep_plan(requested: Path, summary: Path) -> DeletionPlan:
    """The triple, from its summary."""
    stem = summary.name[: -len(SWEEP_SUMMARY_SUFFIX)]
    results = summary.with_name(f"{stem}.csv")
    plot = summary.with_name(f"{stem}{SWEEP_PLOT_SUFFIX}")
    files = [
        _planned(
            summary,
            "summary",
            "the sidecar the Sweeps page finds this sweep by; removed first so a partial "
            "failure cannot leave a sweep listed with nothing to draw",
        )
    ]
    # The results table joins only if it is not itself a dataset. A dataset that happens to
    # share a stem with a sweep must not be swept up by it, and the same probe decides both
    # directions.
    if not results.exists() or _is_sweep_table(results):
        files.append(_planned(results, "results table", "the numbers this sweep collected"))
    files.append(_planned(plot, "plot", "the artifact of record, written beside the table"))
    return DeletionPlan(
        kind=TargetKind.SWEEP,
        requested=requested,
        anchor=summary,
        files=tuple(files),
        reason=(
            "A sweep is three files sharing one stem. The Sweeps page keys its listing on "
            "the .threshold.json sidecar, so removing only the table leaves a sweep the "
            "browser still lists and cannot draw."
        ),
    )


def _dataset_plan(requested: Path, path: Path, format_name: str) -> DeletionPlan:
    """A dataset and every sidecar its format declares."""
    exporter = get_exporter(format_name)
    files = [_planned(path, "dataset", f"the {format_name} dataset you named")]
    for companion in exporter.companions(path):
        files.append(
            _planned(
                companion,
                f"{companion.name.split('.')[-2]} sidecar",
                f"written beside the table by {format_name}; the table is not a readable "
                "dataset without it",
            )
        )
    return DeletionPlan(
        kind=TargetKind.DATASET,
        requested=requested,
        anchor=path,
        files=tuple(files),
        reason=(
            f"{format_name} writes {len(files)} file(s) as one set."
            if len(files) > 1
            else f"{format_name} writes one file."
        ),
    )


def _drift_plan(requested: Path, directory: Path) -> DeletionPlan:
    """A drift study, as the one unit it is.

    The directory itself is the member handed to the recycle bin: one call, one bin entry,
    and one restore that means something. ``generate_drift`` commits the whole set through a
    single two-phase :func:`~qecgen.run.staged` move precisely so a mixed old/new set cannot
    exist, and deleting members one at a time would manufacture exactly that.
    """
    contents = sorted(child for child in directory.rglob("*") if child.is_file())
    total = sum(_stat_size(child) for child in contents)
    member = PlannedFile(
        path=directory,
        size_bytes=total,
        exists=True,
        role="drift study",
        reason=(
            f"{len(contents)} file(s) in this directory. A drift study's training file and "
            "its test files only mean anything together, so they go as one."
        ),
    )
    return DeletionPlan(
        kind=TargetKind.DRIFT_SET,
        requested=requested,
        anchor=directory,
        files=(member,),
        reason=(
            "A drift study is a directory whose members only mean anything together: the "
            "training environment is what the test files are held out from. `generate_drift` "
            "commits them in one move so a half-set cannot exist, and this removes them the "
            "same way."
        ),
    )


def _companion_owner(path: Path) -> Path | None:
    """The dataset ``path`` is a sidecar of, if it is one.

    Answered by scanning the parent and asking each dataset what it writes, rather than by a
    second protocol member reversing the derivation. One ``iterdir`` is cheaper than a
    registry-wide surface that could disagree with the forward one.
    """
    try:
        siblings = sorted(path.parent.iterdir())
    except OSError:
        return None
    for sibling in siblings:
        if not sibling.is_file() or sibling == path:
            continue
        format_name = match_extension(sibling)
        if format_name is None:
            continue
        if path in get_exporter(format_name).companions(sibling):
            return sibling
    return None


def plan_deletion(path: Path, *, reserved: Collection[Path] = ()) -> DeletionPlan:
    """Name every file removing ``path`` would take, without removing anything.

    ``reserved`` is directories the *caller* declares off limits, and the UI passes its
    ``runs_dir``. This module cannot name that directory itself: ``WebSettings.create`` only
    *defaults* it to ``<root>/runs``, so a hardcoded ``"runs"`` here would be a second source
    of truth that starts lying the moment ``--runs-dir`` is passed.

    Raises:
        DeletionRefusedError: for a path this tool will not remove. The reason is on the
            exception, so a front end can choose a status code without reading the message.
    """
    path = Path(path)
    reserved_paths = [Path(entry) for entry in reserved]
    _check_reserved(path, reserved_paths)

    if not path.exists():
        raise _refuse(RefusalReason.NOT_FOUND, path, f"nothing to delete at {path.name}")

    if path.is_dir():
        link = _contains_link(path)
        if link is not None:
            raise _refuse(
                RefusalReason.CONTAINS_LINK,
                path,
                f"{path.name} contains a link ({link.name}). A recursive delete through one "
                "has no bounded blast radius, and this tool never writes one.",
            )
        for child in path.rglob("*"):
            _check_reserved(child, reserved_paths)
        if _is_drift_directory(path):
            return _drift_plan(path, path)
        raise _refuse(
            RefusalReason.UNRECOGNISED_DIRECTORY,
            path,
            f"{path.name} is not a drift study, and this tool only removes directories it "
            "wrote. Name a file inside it, or remove the directory yourself.",
        )

    # A drift member: the set is what has meaning, so widen to the directory. Reached before
    # the dataset branch because `train.h5` is a perfectly readable dataset on its own, and
    # deleting it alone is what leaves the half-set `generate_drift` exists to prevent.
    parent = path.parent
    if _is_drift_directory(parent):
        return _drift_plan(path, parent)

    name = path.name
    if name.endswith(SWEEP_SUMMARY_SUFFIX):
        return _sweep_plan(path, path)
    if name.endswith(SWEEP_PLOT_SUFFIX):
        summary = path.with_name(f"{name[: -len(SWEEP_PLOT_SUFFIX)]}{SWEEP_SUMMARY_SUFFIX}")
        if summary.exists():
            return _sweep_plan(path, summary)
    format_name = match_extension(path)
    if format_name == "csv":
        summary = path.with_name(f"{name[: -len('.csv')]}{SWEEP_SUMMARY_SUFFIX}")
        if summary.exists() and _is_sweep_table(path):
            return _sweep_plan(path, summary)

    if format_name is not None:
        try:
            read_manifest(path)
        except NotAQecgenDatasetError:
            # Intact, just not ours: a `score` correction .npz, a foreign .parquet. It has no
            # companions, and deriving some from the extension would delete a user's
            # unrelated file.
            return _plain_plan(path, "a file this tool did not write; nothing goes with it")
        except Exception:
            # Corrupt or half-written, which is exactly the case that must take its sidecars
            # with it. `companions` is pure name derivation, so it is safe on a file that
            # will not parse.
            return _dataset_plan(path, path, format_name)
        return _dataset_plan(path, path, format_name)

    owner = _companion_owner(path)
    if owner is not None:
        raise _refuse(
            RefusalReason.COMPANION_OF_DATASET,
            path,
            f"{path.name} is a sidecar of {owner.name}, and for ml_csv the manifest sidecar "
            "is what proves the table is a qecgen dataset at all. Removing it alone leaves a "
            f"real dataset that the Datasets page reports as somebody else's file. Delete "
            f"{owner.name} instead -- that removes the whole set.",
        )
    return _plain_plan(path, "one file, with nothing else on disk depending on it")


def _plain_plan(path: Path, reason: str) -> DeletionPlan:
    return DeletionPlan(
        kind=TargetKind.PLAIN_FILE,
        requested=path,
        anchor=path,
        files=(_planned(path, "file", reason),),
        reason=reason,
    )


def _classify(exc: BaseException) -> Outcome:
    """Which refusal this is, from the exception the backend raised.

    Ordered so the specific trash failure wins over the generic permission one:
    ``TrashPermissionError`` subclasses ``PermissionError``, and the two call for opposite
    user actions -- close the file, versus this volume has no trash at all.
    """
    if type(exc).__name__ == "TrashPermissionError":
        return Outcome.NO_TRASH
    if isinstance(exc, PermissionError):
        return Outcome.LOCKED
    if isinstance(exc, OSError):
        # send2trash's Windows fallback raises WindowsError(shell_code, message, path) with a
        # *shell* result code in the errno slot, so Python's errno-to-subclass mapping cannot
        # be relied on to have produced PermissionError above.
        if getattr(exc, "winerror", None) in {5, 32, 33}:
            return Outcome.LOCKED
        return Outcome.FAILED
    return Outcome.FAILED


def _remove_one(entry: PlannedFile) -> Disposition:
    """Attempt one member and report what is observably true afterwards."""
    if not os.path.lexists(entry.path):
        return Disposition(entry.path, Outcome.VANISHED, 0)
    try:
        send_to_trash(entry.path)
    except Exception as exc:
        # `Exception`, not `BaseException`: a Ctrl-C mid-delete must propagate rather than be
        # filed as this file's outcome, which would report a run the user interrupted as a
        # deletion that failed on its own.
        return Disposition(
            entry.path, _classify(exc), entry.size_bytes, f"{type(exc).__name__}: {exc}"
        )
    if os.path.lexists(entry.path):
        # The call returned and the file is still there. Backends differ in what they raise,
        # so success is checked rather than inferred -- reporting REMOVED here would be a
        # claim about something that plainly did not happen.
        return Disposition(
            entry.path,
            Outcome.FAILED,
            entry.size_bytes,
            "the recycle request returned but the path is still there",
        )
    return Disposition(entry.path, Outcome.REMOVED, entry.size_bytes)


def execute(plan: DeletionPlan) -> DeletionReport:
    """Carry out ``plan``, reporting one outcome per file.

    **The file every listing keys on goes first, and a failure there skips the rest.** Once
    the anchor is gone the set is invisible to every listing, so a partial failure can only
    leave inert bytes. The reverse leaves the key alive and the payload gone: an ``ml_csv``
    manifest with no table is a Datasets row claiming a dataset nothing can open, and a
    ``.threshold.json`` with no results table is a Sweeps row whose detail route 404s on a
    sweep the listing shows. And if the anchor itself fails, nothing else may be attempted --
    removing the companions of a file that is *still on disk* is what demotes a real dataset
    to *not a qecgen dataset*, which is the one outcome ranked worse than orphaned bytes,
    because it is false rather than merely untidy.

    One call per file rather than one call for the list. ``send2trash``'s Windows backend
    builds a single ``SHFileOperationW`` from a multi-path string and returns **one** code
    for the whole operation with no per-path attribution -- and it is not a transaction, so
    it can stop partway. Reporting "failed" for files that are gone, or "removed" for files
    that are not, is the failure this module exists to avoid. A drift study is the exception
    and is one call by construction: the member is the directory.

    Never raises for a file it could not remove; read :attr:`DeletionReport.complete`.
    """
    dispositions: list[Disposition] = []
    aborted = False
    for index, entry in enumerate(plan.files):
        if not entry.exists:
            dispositions.append(Disposition(entry.path, Outcome.ALREADY_MISSING, 0))
            continue
        if aborted:
            dispositions.append(Disposition(entry.path, Outcome.SKIPPED, entry.size_bytes))
            continue
        result = _remove_one(entry)
        dispositions.append(result)
        if index == 0 and result.outcome not in _GONE:
            aborted = True
    return DeletionReport(plan=plan, dispositions=tuple(dispositions))
