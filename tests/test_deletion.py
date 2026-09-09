"""The deletion core: what travels together, what is refused, and what is claimed.

Every test here is named for the wrong behaviour it makes impossible. The two failure
families are (1) removing part of a set, which leaves a file describing something untrue,
and (2) claiming an outcome the code did not observe.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qecgen import deletion
from qecgen.dataset import StructureLevel
from qecgen.deletion import (
    RECYCLE_CAVEAT,
    DeletionRefusedError,
    Outcome,
    RefusalReason,
    TargetKind,
    execute,
    plan_deletion,
)
from qecgen.environments import build_single_environment
from qecgen.exporters import get_exporter
from qecgen.run import DISPLACED_PREFIX, LOCK_NAME, PARTIAL_PREFIX, staged


@pytest.fixture
def trashed(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record what would have gone to the recycle bin, and unlink it instead.

    Named rather than autouse: a fixture that silently disables the real path everywhere is
    how the real path stops being covered at all. ``test_the_seam_is_wired_to_send2trash``
    checks the wiring without moving a file.
    """
    seen: list[Path] = []

    def fake(path: Path) -> None:
        seen.append(Path(path))
        target = Path(path)
        if target.is_dir():
            import shutil

            shutil.rmtree(target)
        else:
            target.unlink()

    monkeypatch.setattr(deletion, "send_to_trash", fake)
    return seen


def _write(path: Path, fmt: str, level: StructureLevel = StructureLevel.NONE) -> Path:
    """A real dataset of the named format at ``path``'s stem."""
    dataset = build_single_environment(distance=3, p=0.01, shots=8, seed=1, structure_level=level)
    exporter = get_exporter(fmt)
    target = path.with_name(f"{path.name}{exporter.extension}")
    exporter.write(dataset, target, level)
    return target


def _sweep(directory: Path, stem: str, *, plot: bool = True, table: bool = True) -> Path:
    """A sweep triple in the shape ``run_threshold_sweep`` commits."""
    directory.mkdir(parents=True, exist_ok=True)
    summary = directory / f"{stem}.threshold.json"
    summary.write_text(json.dumps({"decoders": [], "by_decoder": {}}), encoding="utf-8")
    if table:
        (directory / f"{stem}.csv").write_text("distance,p\n3,0.01\n", encoding="utf-8")
    if plot:
        (directory / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return summary


class TestDatasetSets:
    def test_a_plan_names_every_file_an_ml_csv_write_produced(self, tmp_path: Path) -> None:
        """The directory is the assertion, not a count.

        ``len(plan.files) == 4`` passes on a plan naming the wrong four -- a ``with_suffix``
        slip producing ``d.manifest.json`` instead of ``d.ml.manifest.json`` would satisfy it
        while leaving every real sidecar on disk.
        """
        directory = tmp_path / "full"
        directory.mkdir()
        table = _write(directory / "d", "ml_csv", StructureLevel.FULL)
        plan = plan_deletion(table)
        assert plan.kind is TargetKind.DATASET
        assert {entry.path for entry in plan.present} == set(directory.iterdir())

    def test_absent_sidecars_are_reported_rather_than_hidden(self, tmp_path: Path) -> None:
        """`missing` is what makes a half-deleted set visible on a retry.

        Asserting only `present` passes against an implementation that filters absent names
        away entirely, and then a second attempt after a partial failure cannot show which
        half already went.
        """
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.NONE)
        plan = plan_deletion(table)
        assert [entry.path.name for entry in plan.present] == ["d.ml.csv", "d.ml.manifest.json"]
        assert {entry.path.name for entry in plan.missing} == {
            "d.ml.structure.json",
            "d.ml.provenance.json",
        }

    def test_the_dataset_file_is_the_first_thing_removed(self, tmp_path: Path) -> None:
        """Anchor-first is the ordering, and membership alone does not check it.

        Manifest-first is the ordering that demotes a real dataset to *not a qecgen dataset*
        when the second removal fails, and a set-equality assertion passes on it.
        """
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.FULL)
        plan = plan_deletion(table)
        assert plan.anchor == table
        assert plan.files[0].path == table

    def test_a_foreign_file_with_a_dataset_extension_is_planned_alone(self, tmp_path: Path) -> None:
        """Inventing companions from the extension would delete a user's unrelated file.

        Using `.npz` for this would pass trivially -- its `companions` is `()` either way.
        `ml_csv` is the one extension where the invention would actually remove something.
        """
        stray = tmp_path / "notours.ml.csv"
        stray.write_text("a,b\n1,2\n", encoding="utf-8")
        (tmp_path / "notours.ml.manifest.json").write_text("{}", encoding="utf-8")
        plan = plan_deletion(stray)
        assert plan.kind is TargetKind.PLAIN_FILE
        assert [entry.path for entry in plan.files] == [stray]

    def test_a_corrupt_dataset_still_takes_its_sidecars(self, tmp_path: Path) -> None:
        """The half-written file is exactly the one a delete is aimed at.

        `companions` is pure name derivation, so it works on a file that will not parse; an
        implementation that planned only what it could read would strand the sidecars of
        every dataset a dead worker left behind.
        """
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.NONE)
        table.write_text("truncated garbage\n", encoding="utf-8")
        plan = plan_deletion(table)
        assert plan.kind is TargetKind.DATASET
        assert (tmp_path / "d.ml.manifest.json") in {e.path for e in plan.present}


class TestSweepTriples:
    @pytest.mark.parametrize("suffix", [".threshold.json", ".csv", ".png"])
    def test_a_sweep_is_reachable_from_every_one_of_its_three_files(
        self, tmp_path: Path, suffix: str
    ) -> None:
        """Testing only from the sidecar misses the path a user actually clicks.

        The Datasets page lists the results table, because `.csv` is a dataset extension;
        that is the row a Delete button sits on.
        """
        _sweep(tmp_path, "s")
        plan = plan_deletion(tmp_path / f"s{suffix}")
        assert plan.kind is TargetKind.SWEEP
        assert {entry.path.name for entry in plan.present} == {
            "s.threshold.json",
            "s.csv",
            "s.png",
        }

    def test_the_summary_is_removed_before_the_numbers(self, tmp_path: Path) -> None:
        """The reverse leaves a sweep the listing shows and `sweep_detail` 404s on."""
        _sweep(tmp_path, "s")
        plan = plan_deletion(tmp_path / "s.csv")
        assert plan.files[0].path.name == "s.threshold.json"

    def test_a_sweep_results_table_is_not_planned_as_a_dataset(self, tmp_path: Path) -> None:
        """`len(present) == 3` passes on a plan that invented sidecars for the .csv."""
        _sweep(tmp_path, "s")
        plan = plan_deletion(tmp_path / "s.threshold.json")
        assert plan.kind is TargetKind.SWEEP
        assert not any(entry.path.name.endswith(".manifest.json") for entry in plan.files)

    def test_a_missing_member_is_named_not_fatal(self, tmp_path: Path) -> None:
        _sweep(tmp_path, "s", plot=False)
        plan = plan_deletion(tmp_path / "s.threshold.json")
        assert [entry.path.name for entry in plan.missing] == ["s.png"]
        assert len(plan.present) == 2

    def test_a_real_csv_dataset_beside_a_same_stem_summary_stays_a_dataset(
        self, tmp_path: Path
    ) -> None:
        """Both directions, because it is one probe used twice.

        Checking only the dataset side leaves the summary-side plan eating a real dataset;
        checking only the summary side leaves a dataset planned as a sweep.
        """
        dataset = _write(tmp_path / "s", "csv")
        summary = tmp_path / "s.threshold.json"
        summary.write_text(json.dumps({"decoders": []}), encoding="utf-8")

        from_dataset = plan_deletion(dataset)
        assert from_dataset.kind is TargetKind.DATASET
        assert [entry.path for entry in from_dataset.files] == [dataset]

        from_summary = plan_deletion(summary)
        assert from_summary.kind is TargetKind.SWEEP
        assert dataset not in {entry.path for entry in from_summary.present}


class TestDriftSets:
    def test_a_drift_directory_is_planned_as_one_unit(self, tmp_path: Path) -> None:
        """Per-file planning passes a naive all-files-named check and still produces the
        mixed old/new half-set `generate_drift` exists to prevent."""
        directory = tmp_path / "drift"
        directory.mkdir()
        _write(directory / "train", "hdf5")
        _write(directory / "test_0.002", "hdf5")
        plan = plan_deletion(directory)
        assert plan.kind is TargetKind.DRIFT_SET
        assert [entry.path for entry in plan.files] == [directory]
        assert plan.total_bytes > 0

    def test_deleting_one_drift_member_takes_the_whole_study(self, tmp_path: Path) -> None:
        """A drift member is a perfectly readable dataset on its own, so the dataset branch
        would happily plan it alone -- and a study missing its training environment is
        exactly the half-set the condition labelling depends on."""
        directory = tmp_path / "drift"
        directory.mkdir()
        train = _write(directory / "train", "hdf5")
        _write(directory / "test_0.002", "hdf5")
        plan = plan_deletion(train)
        assert plan.kind is TargetKind.DRIFT_SET
        assert plan.anchor == directory

    def test_an_unrecognised_directory_is_refused(self, tmp_path: Path) -> None:
        """Recursive deletion of an arbitrary directory from a web UI is the most dangerous
        operation available here; the drift set is the only directory qecgen writes."""
        directory = tmp_path / "mine"
        directory.mkdir()
        (directory / "notes.txt").write_text("keep", encoding="utf-8")
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(directory)
        assert excinfo.value.reason is RefusalReason.UNRECOGNISED_DIRECTORY
        assert (directory / "notes.txt").is_file()

    def test_a_drift_study_goes_in_one_call(self, tmp_path: Path, trashed: list[Path]) -> None:
        """One call is one recycle-bin entry, which is the only restore that means anything
        for a set whose members are meaningless apart."""
        directory = tmp_path / "drift"
        directory.mkdir()
        _write(directory / "train", "hdf5")
        _write(directory / "test_0.002", "hdf5")
        report = execute(plan_deletion(directory))
        assert report.complete
        assert trashed == [directory]
        assert not directory.exists()


class TestRefusals:
    def test_a_live_staging_directory_is_refused_by_reason(self, tmp_path: Path) -> None:
        """`pytest.raises(ValueError)` alone passes on a NOT_FOUND refusal -- i.e. on an
        implementation that never probed the lock at all."""
        destination = tmp_path / "data"
        with staged(destination) as staging:
            victim = staging.scratch / "mid.h5"
            victim.write_text("half-written", encoding="utf-8")
            with pytest.raises(DeletionRefusedError) as excinfo:
                plan_deletion(victim)
            assert excinfo.value.reason is RefusalReason.STAGING_LIVE
            assert victim.is_file()

    def test_an_orphan_staging_directory_names_sweep_partials(self, tmp_path: Path) -> None:
        """The whole value of splitting the two staging reasons is that they tell the user
        different things to do; without the message assertion the split is decorative."""
        orphan = tmp_path / f"{PARTIAL_PREFIX}deadbeef"
        orphan.mkdir()
        (orphan / LOCK_NAME).write_bytes(b"")
        stray = orphan / "x.h5"
        stray.write_text("x", encoding="utf-8")
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(stray)
        assert excinfo.value.reason is RefusalReason.STAGING_ORPHAN
        assert "sweep_partials" in str(excinfo.value)

    def test_a_displaced_salvage_directory_is_refused_and_untouched(self, tmp_path: Path) -> None:
        """Raising is not enough; the assertion that carries the weight is that the only
        surviving copy of an overwritten dataset did not move."""
        salvage = tmp_path / f"{DISPLACED_PREFIX}abc123"
        salvage.mkdir()
        only_copy = salvage / "previous.h5"
        only_copy.write_text("the only copy", encoding="utf-8")
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(only_copy)
        assert excinfo.value.reason is RefusalReason.DISPLACED_SALVAGE
        assert only_copy.read_text(encoding="utf-8") == "the only copy"

    def test_the_lock_file_is_refused_by_name(self, tmp_path: Path) -> None:
        directory = tmp_path / "plain"
        directory.mkdir()
        lock = directory / LOCK_NAME
        lock.write_bytes(b"")
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(lock)
        assert excinfo.value.reason is RefusalReason.RESERVED_NAME

    def test_the_runs_directory_is_refused_only_when_declared(self, tmp_path: Path) -> None:
        """The second half proves the core did not hardcode "runs".

        A hardcode passes the first assertion and starts lying the moment `--runs-dir` names
        somewhere else.
        """
        runs = tmp_path / "history"
        runs.mkdir()
        record = runs / "aabbccddeeff.json"
        record.write_text(json.dumps({"id": "aabbccddeeff"}), encoding="utf-8")
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(record, reserved=(runs,))
        assert excinfo.value.reason is RefusalReason.PROTECTED_ROOT
        assert plan_deletion(record).kind is TargetKind.PLAIN_FILE

    def test_a_run_record_is_refused_by_its_own_contents(self, tmp_path: Path) -> None:
        """The structural signal, which is what closes the CLI hole: the CLI passes no
        `reserved` set, so without the file's own `id` agreeing with its name there would be
        nothing to refuse on."""
        runs = tmp_path / "runs"
        runs.mkdir()
        record = runs / "aabbccddeeff.json"
        record.write_text(json.dumps({"id": "aabbccddeeff", "mode": "generate"}), encoding="utf-8")
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(record)
        assert excinfo.value.reason is RefusalReason.RUN_RECORD
        assert "Runs page" in str(excinfo.value)

    def test_a_users_own_runs_directory_is_not_caught_by_the_shape(self, tmp_path: Path) -> None:
        """The name pattern alone would refuse a user's unrelated file; the `id` check is
        what keeps the refusal provable."""
        runs = tmp_path / "runs"
        runs.mkdir()
        theirs = runs / "aabbccddeeff.json"
        theirs.write_text(json.dumps({"something": "else"}), encoding="utf-8")
        assert plan_deletion(theirs).kind is TargetKind.PLAIN_FILE

    def test_deleting_a_manifest_sidecar_alone_is_refused_and_points_at_the_table(
        self, tmp_path: Path
    ) -> None:
        """The silent-demotion hazard. Without the message naming the table the refusal is a
        dead end -- the user is holding the one file whose removal turns a real dataset into
        somebody else's CSV, and nothing tells them what to delete instead."""
        _write(tmp_path / "d", "ml_csv", StructureLevel.NONE)
        sidecar = tmp_path / "d.ml.manifest.json"
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(sidecar)
        assert excinfo.value.reason is RefusalReason.COMPANION_OF_DATASET
        assert "d.ml.csv" in str(excinfo.value)

    def test_a_missing_path_is_refused_as_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DeletionRefusedError) as excinfo:
            plan_deletion(tmp_path / "nope.h5")
        assert excinfo.value.reason is RefusalReason.NOT_FOUND


class TestExecution:
    def test_a_file_that_vanished_between_plan_and_execute_is_not_reported_as_removed(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        """A confirmation dialog can sit open for minutes, so this window is real.

        Reporting VANISHED as REMOVED would claim this code did something it did not do.
        """
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.NONE)
        plan = plan_deletion(table)
        (tmp_path / "d.ml.manifest.json").unlink()
        report = execute(plan)
        by_name = {d.path.name: d.outcome for d in report.dispositions}
        assert by_name["d.ml.csv"] is Outcome.REMOVED
        assert by_name["d.ml.manifest.json"] is Outcome.VANISHED

    def test_a_missing_path_is_never_handed_to_the_operating_system(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        """Asserting ALREADY_MISSING passes on an implementation that calls and swallows --
        and that one raises on real Windows, where `get_short_path_name` rejects a path that
        does not exist with a backend-dependent exception."""
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.NONE)
        report = execute(plan_deletion(table))
        assert all(entry.name != "d.ml.structure.json" for entry in trashed)
        outcomes = {d.path.name: d.outcome for d in report.dispositions}
        assert outcomes["d.ml.structure.json"] is Outcome.ALREADY_MISSING

    def test_removed_is_never_reported_for_a_path_that_is_still_there(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The general anti-false-claim invariant.

        A backend that returns success without moving anything is exactly what the
        post-call `lexists` check exists for, and no per-case test covers it.
        """
        monkeypatch.setattr(deletion, "send_to_trash", lambda path: None)
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.NONE)
        report = execute(plan_deletion(table))
        for entry in report.dispositions:
            if entry.outcome is Outcome.REMOVED:
                assert not os.path.lexists(entry.path)
        assert not report.complete
        assert table.is_file()

    def test_a_failed_anchor_skips_the_rest_of_the_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asserting only that the anchor failed passes while the sidecars were deleted
        anyway -- which is the state that turns a real dataset into a `not_a_dataset` row."""
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.FULL)
        before = set(tmp_path.iterdir())

        def refuse(path: Path) -> None:
            raise PermissionError(13, "in use", str(path))

        monkeypatch.setattr(deletion, "send_to_trash", refuse)
        report = execute(plan_deletion(table))
        outcomes = [d.outcome for d in report.dispositions]
        assert outcomes[0] is Outcome.LOCKED
        assert set(outcomes[1:]) == {Outcome.SKIPPED}
        assert set(tmp_path.iterdir()) == before

    def test_a_locked_sidecar_does_not_stop_the_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the anchor is gone the set is invisible to every listing, so the remaining
        members are inert bytes and are still worth attempting."""
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.FULL)
        plan = plan_deletion(table)
        stubborn = tmp_path / "d.ml.structure.json"

        def selective(path: Path) -> None:
            if Path(path) == stubborn:
                raise PermissionError(13, "in use", str(path))
            Path(path).unlink()

        monkeypatch.setattr(deletion, "send_to_trash", selective)
        report = execute(plan)
        outcomes = {d.path.name: d.outcome for d in report.dispositions}
        assert outcomes["d.ml.csv"] is Outcome.REMOVED
        assert outcomes["d.ml.structure.json"] is Outcome.LOCKED
        assert outcomes["d.ml.provenance.json"] is Outcome.REMOVED

    def test_a_volume_with_no_trash_is_its_own_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NO_TRASH and LOCKED call for opposite user actions, and TrashPermissionError
        subclasses PermissionError -- so an isinstance chain in the wrong order tells a user
        on a network share to go close a file."""
        from send2trash import TrashPermissionError

        def no_bin(path: Path) -> None:
            raise TrashPermissionError(str(path))

        monkeypatch.setattr(deletion, "send_to_trash", no_bin)
        table = _write(tmp_path / "d", "hdf5")
        report = execute(plan_deletion(table))
        assert report.dispositions[0].outcome is Outcome.NO_TRASH
        assert table.is_file()

    def test_total_bytes_counts_only_what_is_present(self, tmp_path: Path) -> None:
        """Really asserts that a plan is constructible over a half-deleted set, which is the
        property a retry depends on."""
        table = _write(tmp_path / "d", "ml_csv", StructureLevel.NONE)
        plan = plan_deletion(table)
        expected = table.stat().st_size + (tmp_path / "d.ml.manifest.json").stat().st_size
        assert plan.total_bytes == expected


class TestHonesty:
    def test_the_recycle_caveat_travels_with_every_report(
        self, tmp_path: Path, trashed: list[Path]
    ) -> None:
        """A caveat attached only on success is absent exactly where a reader looks hardest.

        Asserted on the *report* as well as the plan, because the report is what a front end
        renders after the fact -- the moment a user is deciding whether the file is gone for
        good, which is precisely what this sentence answers.
        """
        table = _write(tmp_path / "d", "hdf5")
        plan = plan_deletion(table)
        assert plan.to_json_dict()["caveat"] == RECYCLE_CAVEAT
        report = execute(plan)
        assert report.complete
        assert report.to_json_dict()["caveat"] == RECYCLE_CAVEAT

    def test_no_outcome_claims_the_recycle_bin(self) -> None:
        """The one assertion that keeps the honesty decision from being reverted by a
        well-meaning rename to `Outcome.RECYCLED`.

        Windows deletes an oversize file outright and reports success either way, so no value
        this module produces may assert where a file ended up.
        """
        for member in Outcome:
            assert "recycle" not in member.value.lower()
            assert member.value != "recycled"

    def test_the_caveat_states_permanent_deletion(self) -> None:
        assert "permanently" in RECYCLE_CAVEAT
        assert "recoverable" in RECYCLE_CAVEAT

    def test_the_seam_is_wired_to_send2trash(self) -> None:
        """Checks the wiring without moving a file.

        Every other test in this module patches `send_to_trash`; without this one the real
        backend could be unreferenced and the whole suite would still pass.
        """
        import inspect

        source = inspect.getsource(deletion.send_to_trash)
        assert "from send2trash import send2trash" in source
        assert "send2trash(" in source

    def test_nothing_imports_the_trash_seam_by_name(self) -> None:
        """One seam, or the seam is not a seam.

        ``from qecgen.deletion import send_to_trash`` binds the function into the importing
        module, and ``monkeypatch.setattr(deletion, "send_to_trash", ...)`` then does not
        reach it. A test believing it had disabled real deletion would move the developer's
        files into their actual recycle bin -- which is what ``ui/jobs.py`` did before it was
        changed to call through the module.
        """
        package = Path(deletion.__file__).parent
        offenders: list[str] = []
        for path in package.rglob("*.py"):
            if path.name == "deletion.py":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped.startswith("from ") or "#" in stripped.split("import")[0]:
                    continue
                _, _, tail = stripped.partition(" import ")
                if "send_to_trash" in tail:
                    offenders.append(path.relative_to(package).as_posix())
                    break
        assert offenders == []

    def test_deletion_support_reports_availability(self) -> None:
        support = deletion.deletion_support()
        assert support["available"] is True
        assert support["destination"] == "recycle bin"
        assert support["problem"] is None


@pytest.mark.skipif(
    sys.platform != "win32",
    reason=(
        "only Windows refuses to remove a file another handle has open. On POSIX the unlink "
        "succeeds and the inode survives until the last close, so there is no refusal to "
        "observe -- the difference is the behaviour under test, not a gap in it."
    ),
)
def test_a_dataset_file_held_open_leaves_its_sidecars_alone(tmp_path: Path) -> None:
    """A real held handle, not a monkeypatched exception.

    The repo already simulates this failure by patching `os.replace` to raise; that proves
    the classification but not that it ever fires. CPython's `open` requests
    FILE_SHARE_READ|FILE_SHARE_WRITE and not FILE_SHARE_DELETE, so a live handle in this very
    process is enough to make the shell refuse -- the same Windows failure mode `run._commit`
    is two-phase for.

    Asserting only the outcome would pass while the sidecars were deleted anyway, which is
    the state that demotes a real dataset to somebody else's CSV.
    """
    table = _write(tmp_path / "d", "ml_csv", StructureLevel.FULL)
    before = set(tmp_path.iterdir())
    plan = plan_deletion(table)
    with table.open("rb"):
        report = execute(plan)
    assert report.dispositions[0].outcome in {Outcome.LOCKED, Outcome.FAILED}
    assert not report.complete
    assert set(tmp_path.iterdir()) == before


def test_merge_collapses_overlapping_plans(tmp_path: Path) -> None:
    """A sweep run reports all three files as artifacts, so planning each yields the same
    triple three times. Without the merge a confirmation would list nine rows for three
    files, and the executor would attempt each one three times."""
    from qecgen.deletion import DeletionPlan

    _sweep(tmp_path, "s")
    plans = [
        plan_deletion(tmp_path / f"s{suffix}") for suffix in (".csv", ".png", ".threshold.json")
    ]
    merged = DeletionPlan.merge(plans)
    assert merged is not None
    assert len(merged.files) == 3
    assert DeletionPlan.merge([]) is None


def test_arrays_are_not_touched_by_planning(tmp_path: Path) -> None:
    """Planning reads a manifest at most; it must never materialise a dataset.

    A planner that called `read()` to identify a file would load every shot of a
    multi-gigabyte dataset to answer "what goes with this", on a request whose whole purpose
    is to render a confirmation quickly.
    """
    table = _write(tmp_path / "big", "npz")
    with np.load(table, allow_pickle=False) as handle:
        expected: dict[str, Any] = {name: handle[name].shape for name in handle.files}
    plan_deletion(table)
    with np.load(table, allow_pickle=False) as handle:
        assert {name: handle[name].shape for name in handle.files} == expected
