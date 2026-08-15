"""The shared run layer: routing, naming, and the staged-write guarantee.

The staging tests are the important ones. Every exporter writes in place and truncates
its target at the first byte, so "a run failed" and "the file that was already there is
gone" used to be the same event.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qecgen.dataset import DriftCondition, StructureLevel
from qecgen.environments import (
    DriftAxis,
    build_single_environment,
    drift_dataset_names,
    stream_single_environment,
)
from qecgen.exporters import EXPORTERS, infer_format
from qecgen.run import (
    PARTIAL_PREFIX,
    DriftSpec,
    GenerateSpec,
    MultiEnvSpec,
    RunCancelledError,
    should_stream,
    staged,
    sweep_partials,
    total_shots,
)
from qecgen.run import run as run_spec
from qecgen.validate import validate_dataset


def _generate(out: Path, **overrides: Any) -> GenerateSpec:
    base: dict[str, Any] = {
        "distance": 3,
        "p": 0.01,
        "shots": 200,
        "seed": 1,
        "out": out,
        "chunk_size": 100,
    }
    base.update(overrides)
    return GenerateSpec(**base)


class TestInferFormat:
    def test_every_registered_extension_round_trips(self) -> None:
        # Driven off the registry so a new exporter is covered without editing this test.
        for name, exporter in EXPORTERS.items():
            assert infer_format(Path(f"x{exporter.extension}")) == name

    def test_unknown_suffix_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="cannot infer format"):
            infer_format(Path("dataset.pickle"))

    def test_raises_value_error_not_a_cli_error(self) -> None:
        # The registry must be usable from a front end that has never heard of typer.
        with pytest.raises(ValueError):
            infer_format(Path("dataset"))


class TestShouldStream:
    def test_only_hdf5_above_one_chunk(self) -> None:
        assert should_stream("hdf5", 1001, 1000) is True
        assert should_stream("hdf5", 1000, 1000) is False
        assert should_stream("hdf5", 999, 1000) is False

    def test_no_other_format_streams(self) -> None:
        for name, exporter in EXPORTERS.items():
            if not exporter.streaming:
                assert should_stream(name, 10_000, 100) is False


class TestTotalShots:
    def test_counts_every_file_for_drift(self) -> None:
        spec = DriftSpec(
            distance=3,
            train_p=0.005,
            test_values=(0.007, 0.01),
            shots=100,
            seed=0,
            condition=DriftCondition.FROZEN_PRIOR,
            out=Path("d"),
        )
        assert total_shots(spec) == 300

    def test_counts_every_environment_for_multi(self) -> None:
        spec = MultiEnvSpec(
            distance=3,
            axis_values=(0.005, 0.01, 0.02),
            shots_per_env=50,
            seed=0,
            out=Path("m.h5"),
        )
        assert total_shots(spec) == 150


class TestDriftNames:
    def test_names_match_the_returned_order(self) -> None:
        assert drift_dataset_names([0.007, 0.01]) == ["train", "test_0.007", "test_0.01"]

    def test_collision_is_refused(self) -> None:
        # Two DISTINCT floats that both print "test_0.01" under %g's six significant
        # digits. The earlier values 0.01 and 0.010 are the *same* float, which pinned
        # duplicate-value rejection rather than the %g name collision this locks down.
        assert 0.010000001 != 0.010000002
        with pytest.raises(ValueError, match="collide"):
            drift_dataset_names([0.010000001, 0.010000002])

    def test_refused_before_any_sampling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import qecgen.environments as environments
        from qecgen.sampling import iter_chunks as real_iter_chunks

        calls = {"n": 0}

        def counting(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            return real_iter_chunks(*args, **kwargs)  # type: ignore[arg-type]

        # Counting sampler calls asserts the "before sampling" property directly: a
        # large shot count that "returns promptly" does not, because the test still
        # passes if the check regressed to run after the build — just slowly.
        monkeypatch.setattr(environments, "iter_chunks", counting)
        spec = DriftSpec(
            distance=3,
            train_p=0.005,
            test_values=(0.010000001, 0.010000002),
            shots=100,
            seed=0,
            condition=DriftCondition.FROZEN_PRIOR,
            out=tmp_path / "drift",
        )
        with pytest.raises(ValueError, match="collide"):
            run_spec(spec)
        assert calls["n"] == 0, "the collision check must run before any sampling"
        assert not (tmp_path / "drift").exists() or list((tmp_path / "drift").iterdir()) == []


class TestStaging:
    def test_commits_under_the_name_the_exporter_chose(self, tmp_path: Path) -> None:
        # NPZ appends .npz to a path that lacks it, so the committed name is not always
        # the requested one -- reporting the requested path would name a missing file.
        written = run_spec(_generate(tmp_path / "b", fmt="npz"))
        assert written[0].path.name == "b.npz"
        assert written[0].path.is_file()

    def test_failed_write_leaves_the_target_absent(self, tmp_path: Path) -> None:
        target = tmp_path / "never.h5"
        with pytest.raises(RuntimeError, match="planted"), staged(tmp_path) as staging:
            (staging.scratch / target.name).write_bytes(b"partial")
            raise RuntimeError("planted failure")
        assert not target.exists()
        assert sweep_partials(tmp_path) == []

    def test_failed_write_does_not_destroy_the_previous_file(self, tmp_path: Path) -> None:
        target = tmp_path / "good.h5"
        run_spec(_generate(target, seed=7))
        before = target.read_bytes()

        with pytest.raises(RuntimeError), staged(tmp_path) as staging:
            (staging.scratch / target.name).write_bytes(b"junk")
            raise RuntimeError("planted failure")

        assert target.read_bytes() == before

    def test_cancelled_stream_leaves_nothing_at_the_target(self, tmp_path: Path) -> None:
        # The streaming writer opens the destination directly, so before staging this
        # left a manifest-less .h5 sitting under the user's chosen name.
        target = tmp_path / "cancelled.h5"
        calls = {"n": 0}

        def cancel_after_one(_shots: int) -> None:
            calls["n"] += 1
            if calls["n"] > 1:
                raise RunCancelledError("stop")

        spec = _generate(target, shots=1000, chunk_size=100)
        assert should_stream(spec.fmt, spec.shots, spec.chunk_size)
        with pytest.raises(RunCancelledError):
            run_spec(spec, progress=cancel_after_one)

        assert not target.exists()
        assert list(tmp_path.glob(f"{PARTIAL_PREFIX}*")) == []

    def test_cancelled_drift_commits_none_of_the_set(self, tmp_path: Path) -> None:
        # A train file without its test siblings is a trap, not a partial result.
        out = tmp_path / "drift"
        calls = {"n": 0}

        def cancel_late(_shots: int) -> None:
            calls["n"] += 1
            if calls["n"] > 3:
                raise RunCancelledError("stop")

        spec = DriftSpec(
            distance=3,
            train_p=0.005,
            test_values=(0.007, 0.01),
            shots=200,
            seed=3,
            condition=DriftCondition.FROZEN_PRIOR,
            out=out,
            chunk_size=100,
        )
        with pytest.raises(RunCancelledError):
            run_spec(spec, progress=cancel_late)
        assert not out.exists() or list(out.iterdir()) == []

    def test_sweep_partials_reports_what_it_removed(self, tmp_path: Path) -> None:
        orphan = tmp_path / f"{PARTIAL_PREFIX}deadbeef"
        orphan.mkdir()
        (orphan / "dataset.h5").write_bytes(b"x")
        removed = sweep_partials(tmp_path)
        assert removed == [orphan]
        assert not orphan.exists()


class TestRunProducesValidData:
    @pytest.mark.parametrize("fmt", sorted(EXPORTERS))
    def test_every_format_validates(self, tmp_path: Path, fmt: str) -> None:
        exporter = EXPORTERS[fmt]
        level = StructureLevel.NONE
        written = run_spec(
            _generate(tmp_path / f"d{exporter.extension}", fmt=fmt, structure_level=level)
        )
        dataset = exporter.read(written[0].path)
        report = validate_dataset(dataset)
        assert report.ok, [c.name for c in report.failures]

    def test_streaming_and_materialising_agree_on_content_hash(self, tmp_path: Path) -> None:
        # The routing decision is only safe to make automatically if both paths produce
        # the same bytes for the same seed and chunk size.
        streamed = stream_single_environment(
            path=tmp_path / "streamed.h5", distance=3, p=0.01, shots=400, seed=11, chunk_size=100
        )
        materialised = build_single_environment(
            distance=3, p=0.01, shots=400, seed=11, chunk_size=100
        )
        assert streamed.content_hash == materialised.meta.content_hash

    def test_drift_files_carry_their_own_manifest(self, tmp_path: Path) -> None:
        spec = DriftSpec(
            distance=3,
            train_p=0.005,
            test_values=(0.007, 0.01),
            shots=100,
            seed=4,
            condition=DriftCondition.FROZEN_PRIOR,
            out=tmp_path / "drift",
            chunk_size=100,
        )
        written = run_spec(spec)
        # Staging commits in sorted order, which is not build order; pairing positionally
        # would stamp each file with a neighbour's manifest.
        by_name = {file.path.name: file for file in written}
        assert by_name["train.h5"].drift_condition is DriftCondition.ORACLE_CALIBRATED
        assert by_name["test_0.007.h5"].drift_condition is DriftCondition.FROZEN_PRIOR
        assert by_name["test_0.01.h5"].drift_condition is DriftCondition.FROZEN_PRIOR


class TestProgressHooks:
    def test_increments_sum_to_total_for_every_mode(self, tmp_path: Path) -> None:
        for spec in (
            _generate(tmp_path / "a.h5", shots=400, chunk_size=100),
            _generate(tmp_path / "b.npz", fmt="npz", shots=300, chunk_size=100),
            MultiEnvSpec(
                distance=3,
                axis_values=(0.005, 0.01),
                shots_per_env=150,
                seed=2,
                out=tmp_path / "m.h5",
                chunk_size=50,
            ),
            DriftSpec(
                distance=3,
                train_p=0.005,
                test_values=(0.007,),
                shots=100,
                seed=3,
                condition=DriftCondition.ORACLE_CALIBRATED,
                out=tmp_path / "d",
                chunk_size=50,
            ),
        ):
            seen: list[int] = []
            run_spec(spec, progress=seen.append)
            assert sum(seen) == total_shots(spec), spec
            assert len(seen) > 1, "progress must arrive in steps, not one jump at the end"

    def test_phase_switches_to_writing_on_the_materialising_path(self, tmp_path: Path) -> None:
        # Sampling is the only phase with progress; concatenation, hashing and gzip come
        # after, and a bar that fills and then sits still reads as a hang.
        phases: list[str] = []
        run_spec(_generate(tmp_path / "p.npz", fmt="npz"), on_phase=phases.append)
        assert phases == ["sampling", "writing"]

    def test_multi_env_reports_across_all_environments(self, tmp_path: Path) -> None:
        spec = MultiEnvSpec(
            distance=3,
            axis_values=(0.005, 0.01, 0.02),
            shots_per_env=100,
            seed=5,
            out=tmp_path / "m.h5",
            chunk_size=50,
            axis=DriftAxis.P,
        )
        seen: list[int] = []
        run_spec(spec, progress=seen.append)
        assert sum(seen) == 300
        assert len(seen) == 6
