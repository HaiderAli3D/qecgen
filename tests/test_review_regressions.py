"""Regression tests for defects found by external code review.

Each test is named for the finding it locks down and asserts that the **old** behaviour
is now impossible. The original suite passed while every one of these was live, which is
the point: these cover the surrounding surface (manifest text, dtypes, None fields,
degenerate inputs) rather than only the parts that were already thought about carefully.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import stim

from qecgen.circuits import ChannelVector, NoiseModel, apply_xz_bias, build_circuit
from qecgen.dataset import (
    DatasetMeta,
    DriftCondition,
    StreamingContentHasher,
    StructureLevel,
    content_hash,
)
from qecgen.environments import (
    DriftAxis,
    build_drift_environments,
    build_environment,
    build_multi_environment,
    build_single_environment,
    stream_single_environment,
)
from qecgen.exporters import EXPORTERS, StreamingHDF5Writer, get_exporter
from qecgen.qa import (
    LogicalErrorEstimate,
    clopper_pearson,
    crossing_from_intervals,
    detection_event_rate,
    estimate_crossing,
)
from qecgen.validate import validate_dataset, validate_dem_structure


class TestC1InvalidRates:
    """NaN rates used to build, sample all-zeros, and pass validation."""

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.5])
    def test_channel_vector_rejects_non_probabilities(self, bad: float) -> None:
        with pytest.raises(ValueError):
            ChannelVector(after_clifford_depolarization=bad)

    def test_nan_rate_rejected_before_any_sampling(self) -> None:
        with pytest.raises(ValueError):
            build_multi_environment(
                distance=3, error_rates=[math.nan], shots_per_env=20, seed=1, chunk_size=20
            )

    def test_nan_axis_value_rejected(self) -> None:
        """xz_bias never reaches a channel, so ChannelVector alone would not catch it."""
        with pytest.raises(ValueError, match="finite"):
            build_environment(0, 3, 0.01, DriftAxis.XZ_BIAS, math.nan, 10)

    def test_negative_bias_rejected(self) -> None:
        with pytest.raises(ValueError, match="eta must be > 0"):
            build_environment(0, 3, 0.01, DriftAxis.XZ_BIAS, -1.0, 10)

    def test_manifest_json_is_standards_compliant(self) -> None:
        """A bare NaN token is not valid JSON and strict parsers reject it."""
        dataset = build_single_environment(distance=3, p=0.01, shots=20, seed=1, chunk_size=20)
        text = dataset.meta.to_json()
        assert "NaN" not in text
        json.loads(text)


class TestC2ContractBWidth:
    """Contract B labels used to ship with n_mechanisms=None and validate clean."""

    def test_n_mechanisms_set_without_structure(self) -> None:
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=40,
            seed=1,
            chunk_size=40,
            emit_mechanisms=True,
            structure_level=StructureLevel.NONE,
        )
        assert dataset.meta.n_mechanisms is not None
        assert dataset.mechanisms is not None
        assert dataset.mechanisms.shape[1] == math.ceil(dataset.meta.n_mechanisms / 8)
        assert validate_dataset(dataset).ok

    def test_validator_fails_when_count_is_missing(self) -> None:
        dataset = build_single_environment(
            distance=3, p=0.01, shots=40, seed=1, chunk_size=40, emit_mechanisms=True
        )
        broken = dataclasses.replace(
            dataset, meta=dataclasses.replace(dataset.meta, n_mechanisms=None)
        )
        failed = {r.name for r in validate_dataset(broken, check_hash=False).failures}
        assert "mechanisms.n_mechanisms_declared" in failed

    def test_jsonl_does_not_silently_drop_labels(self, tmp_path: Path) -> None:
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=20,
            seed=1,
            chunk_size=20,
            emit_mechanisms=True,
            structure_level=StructureLevel.NONE,
        )
        exporter = get_exporter("jsonl")
        path = tmp_path / "m.jsonl"
        exporter.write(dataset, path, StructureLevel.NONE)
        restored = exporter.read(path)
        assert restored.mechanisms is not None
        assert dataset.mechanisms is not None
        assert np.array_equal(restored.mechanisms, dataset.mechanisms)


class TestC3ValidatorGaps:
    """The validator certified structurally unrelated and malformed data."""

    def test_detects_mismatched_structure(self) -> None:
        big = build_single_environment(
            distance=5,
            p=0.01,
            shots=20,
            seed=1,
            chunk_size=20,
            structure_level=StructureLevel.DEM,
        )
        small = build_single_environment(
            distance=3,
            p=0.01,
            shots=20,
            seed=1,
            chunk_size=20,
            structure_level=StructureLevel.DEM,
        )
        frankenstein = dataclasses.replace(big, structure=small.structure)
        failed = {r.name for r in validate_dataset(frankenstein).failures}
        assert "structure.detectors_match_manifest" in failed

    def test_detects_float_observables(self) -> None:
        dataset = build_single_environment(distance=3, p=0.01, shots=20, seed=1, chunk_size=20)
        broken = dataclasses.replace(dataset, observables=dataset.observables.astype(np.float64))
        failed = {r.name for r in validate_dataset(broken, check_hash=False).failures}
        assert "observables.dtype" in failed

    def test_missing_hash_is_a_failure_not_a_skip(self) -> None:
        dataset = build_single_environment(distance=3, p=0.01, shots=20, seed=1, chunk_size=20)
        broken = dataclasses.replace(
            dataset, meta=dataclasses.replace(dataset.meta, content_hash=None)
        )
        report = validate_dataset(broken)
        assert not report.ok
        assert "content_hash" in {r.name for r in report.failures}


class TestC4FrozenPriorLeak:
    """The finding that most directly damages the generalisation experiment."""

    @staticmethod
    def _frozen_pair() -> tuple[object, object]:
        train, test = build_drift_environments(
            distance=3,
            train_p=0.005,
            test_values=[0.02],
            shots=40,
            seed=1,
            condition=DriftCondition.FROZEN_PRIOR,
            chunk_size=40,
            structure_level=StructureLevel.FULL,
        )
        return train, test

    def test_manifest_never_carries_dem_text(self) -> None:
        _, test = self._frozen_pair()
        payload = test.meta.to_json()  # type: ignore[attr-defined]
        for line in test.meta.environments[0].dem.splitlines():  # type: ignore[attr-defined]
            if line.startswith("error"):
                assert line not in payload

    def test_written_file_manifest_is_leak_free(self, tmp_path: Path) -> None:
        """Scan the serialised manifest for any substring of the test DEM.

        This is the check the original suite lacked: it compared priors numerically and
        never looked at the text.
        """
        _, test = self._frozen_pair()
        path = tmp_path / "test.h5"
        get_exporter("hdf5").write(test, path, StructureLevel.FULL)  # type: ignore[arg-type]

        with h5py.File(path, "r") as handle:
            manifest = str(handle.attrs["manifest"])
            assert "provenance" in handle

        dem_lines = [
            line
            for line in test.meta.environments[0].dem.splitlines()  # type: ignore[attr-defined]
            if line.startswith("error")
        ]
        assert dem_lines, "expected the test DEM to contain error instructions"
        for line in dem_lines:
            assert line not in manifest

    def test_provenance_is_still_recoverable_for_audit(self, tmp_path: Path) -> None:
        _, test = self._frozen_pair()
        path = tmp_path / "test.h5"
        get_exporter("hdf5").write(test, path, StructureLevel.FULL)  # type: ignore[arg-type]
        with h5py.File(path, "r") as handle:
            provenance = json.loads(str(handle["provenance"].attrs["environments"]))
        assert provenance["environments"][0]["dem"].strip()
        assert "must not read" in provenance["warning"]

    def test_structure_dem_sha_identifies_the_source(self) -> None:
        train, test = self._frozen_pair()
        assert test.meta.structure_dem_sha == train.meta.structure_dem_sha  # type: ignore[attr-defined]


class TestC5MechanismColumns:
    def test_frozen_prior_with_incompatible_mechanisms_is_refused(self) -> None:
        with pytest.raises(ValueError, match="column k would denote"):
            build_drift_environments(
                distance=3,
                train_p=0.0,
                test_values=[0.02],
                shots=20,
                seed=1,
                condition=DriftCondition.FROZEN_PRIOR,
                chunk_size=20,
                emit_mechanisms=True,
                structure_level=StructureLevel.DEM,
            )

    def test_mechanism_source_environment_is_recorded(self) -> None:
        dataset = build_single_environment(
            distance=3, p=0.01, shots=20, seed=1, chunk_size=20, emit_mechanisms=True
        )
        assert dataset.meta.mechanism_source_environment_id == 0


class TestH1BiasScope:
    def test_bias_rewrites_gate_noise_too(self) -> None:
        """The docs claimed data noise only; a gate-noise-only circuit disproves that."""
        import stim

        gate_only = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.01,
        )
        assert "DEPOLARIZE1" in str(gate_only)
        biased = apply_xz_bias(gate_only, 9.0)
        assert "PAULI_CHANNEL_1" in str(biased)

    def test_two_qubit_noise_stays_symmetric(self) -> None:
        circuit, _ = build_circuit(3, 0.01)
        assert "DEPOLARIZE2" in str(apply_xz_bias(circuit, 9.0))

    def test_bias_scope_recorded_in_manifest(self) -> None:
        dataset = build_multi_environment(
            distance=3,
            error_rates=[0.5, 4.0],
            shots_per_env=20,
            seed=1,
            chunk_size=20,
            axis=DriftAxis.XZ_BIAS,
            base_p=0.008,
        )
        assert dataset.meta.bias_scope is not None
        assert "depolarize2_unbiased" in dataset.meta.bias_scope


class TestH3StructureLevelAgreement:
    @pytest.mark.parametrize("name", sorted(EXPORTERS))
    def test_write_refuses_disagreeing_level(self, name: str, tmp_path: Path) -> None:
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=20,
            seed=1,
            chunk_size=20,
            structure_level=StructureLevel.DEM,
        )
        exporter = get_exporter(name)
        with pytest.raises(ValueError, match=r"disagrees with meta\.structure_level"):
            exporter.write(dataset, tmp_path / f"x{exporter.extension}", StructureLevel.NONE)

    @pytest.mark.parametrize(
        "name", sorted(n for n, e in EXPORTERS.items() if not e.structure_round_trip)
    )
    def test_non_round_tripping_formats_downgrade_the_recorded_level(
        self, name: str, tmp_path: Path
    ) -> None:
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=20,
            seed=1,
            chunk_size=20,
            structure_level=StructureLevel.DEM,
        )
        exporter = get_exporter(name)
        assert not exporter.structure_round_trip
        path = tmp_path / f"d{exporter.extension}"
        exporter.write(dataset, path, StructureLevel.DEM)
        assert exporter.read(path).meta.structure_level is StructureLevel.NONE


class TestH4CoordsOnlyValidates:
    @pytest.mark.parametrize(
        "name", sorted(n for n, e in EXPORTERS.items() if e.structure_round_trip)
    )
    def test_coords_only_round_trip_validates(self, name: str, tmp_path: Path) -> None:
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=20,
            seed=1,
            chunk_size=20,
            structure_level=StructureLevel.COORDS,
        )
        exporter = get_exporter(name)
        path = tmp_path / f"c{exporter.extension}"
        exporter.write(dataset, path, StructureLevel.COORDS)
        report = validate_dataset(exporter.read(path))
        assert report.ok, str(report)


class TestH5StreamingColumns:
    def test_column_added_mid_stream_is_refused(self, tmp_path: Path) -> None:
        """Otherwise every earlier row gets zero-filled, fabricating labels."""
        writer = StreamingHDF5Writer(tmp_path / "s.h5", 24, 1)
        writer.append(np.zeros((4, 3), np.uint8), np.zeros((4, 1), np.uint8))
        with pytest.raises(ValueError, match="zero-fill"):
            writer.append(
                np.zeros((4, 3), np.uint8),
                np.zeros((4, 1), np.uint8),
                mechanisms=np.zeros((4, 5), np.uint8),
            )
        writer.abort()

    def test_column_dropped_mid_stream_is_refused(self, tmp_path: Path) -> None:
        writer = StreamingHDF5Writer(tmp_path / "s2.h5", 24, 1)
        writer.append(
            np.zeros((4, 3), np.uint8),
            np.zeros((4, 1), np.uint8),
            mechanisms=np.zeros((4, 5), np.uint8),
        )
        with pytest.raises(ValueError, match="differs from the set fixed"):
            writer.append(np.zeros((4, 3), np.uint8), np.zeros((4, 1), np.uint8))
        writer.abort()

    def test_width_change_is_refused(self, tmp_path: Path) -> None:
        writer = StreamingHDF5Writer(tmp_path / "s3.h5", 24, 1)
        writer.append(np.zeros((4, 3), np.uint8), np.zeros((4, 1), np.uint8))
        with pytest.raises(ValueError, match="width"):
            writer.append(np.zeros((4, 7), np.uint8), np.zeros((4, 1), np.uint8))
        writer.abort()


class TestH6StreamingCLI:
    def test_streamed_file_matches_materialised_exactly(self, tmp_path: Path) -> None:
        """Both routes must agree, or the streaming path is a different dataset."""
        kwargs: dict[str, Any] = {
            "distance": 3,
            "p": 0.008,
            "shots": 500,
            "seed": 17,
            "chunk_size": 100,
        }
        materialised = build_single_environment(**kwargs)

        path = tmp_path / "streamed.h5"
        meta = stream_single_environment(path=path, **kwargs)

        restored = get_exporter("hdf5").read(path)
        assert np.array_equal(restored.detectors, materialised.detectors)
        assert np.array_equal(restored.observables, materialised.observables)
        assert meta.content_hash == materialised.meta.content_hash
        assert validate_dataset(restored).ok

    def test_streaming_hasher_matches_materialised_hash(self) -> None:
        rng = np.random.default_rng(0)
        det = rng.integers(0, 256, size=(50, 3), dtype=np.uint8)
        obs = rng.integers(0, 256, size=(50, 1), dtype=np.uint8)

        hasher = StreamingContentHasher()
        for start in range(0, 50, 10):
            hasher.update(det[start : start + 10], obs[start : start + 10])

        assert hasher.hexdigest(50, 24, 1) == content_hash(det, obs)


class TestH7H8LabelIntegrity:
    def test_not_applicable_drift_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must state its condition"):
            build_drift_environments(
                distance=3,
                train_p=0.005,
                test_values=[0.02],
                shots=20,
                seed=1,
                condition=DriftCondition.NOT_APPLICABLE,
                chunk_size=20,
            )

    def test_code_capacity_with_measurement_ratio_is_refused(self) -> None:
        with pytest.raises(ValueError, match="CODE_CAPACITY"):
            build_environment(
                0,
                3,
                0.01,
                DriftAxis.MEASUREMENT_RATIO,
                1.0,
                10,
                noise_model=NoiseModel.CODE_CAPACITY,
            )


class TestH9HashNaming:
    def test_algorithm_is_named_accurately(self) -> None:
        dataset = build_single_environment(distance=3, p=0.01, shots=20, seed=1, chunk_size=20)
        assert dataset.meta.content_hash_algorithm == "blake2b-256"
        assert "content_sha256" not in dataset.meta.to_json()

    def test_algorithm_is_externally_reproducible(self) -> None:
        """An external checker must be able to reproduce the advertised digest."""
        import hashlib

        dataset = build_single_environment(distance=3, p=0.01, shots=20, seed=1, chunk_size=20)
        outer = hashlib.blake2b(digest_size=32)
        for name, array in (
            ("detectors", dataset.detectors),
            ("observables", dataset.observables),
            ("environment_ids", None),
            ("mechanisms", None),
        ):
            outer.update(name.encode("utf-8"))
            if array is None:
                outer.update(b"\x00none")
                continue
            contiguous = np.ascontiguousarray(array)
            outer.update(str(contiguous.dtype).encode("utf-8"))
            outer.update(str(contiguous.shape).encode("utf-8"))
            outer.update(hashlib.blake2b(contiguous.tobytes(), digest_size=32).digest())
        assert outer.hexdigest() == dataset.meta.content_hash


class TestH10StrictParsing:
    def test_rotated_string_false_is_refused(self) -> None:
        """bool('false') is True, so a truthy string would flip the code layout."""
        dataset = build_single_environment(distance=3, p=0.01, shots=20, seed=1, chunk_size=20)
        payload = dataset.meta.to_json_dict()
        payload["rotated"] = "false"
        with pytest.raises(ValueError, match="must be a JSON boolean"):
            DatasetMeta.from_json_dict(payload)

    def test_jsonl_rejects_non_binary_characters(self) -> None:
        from qecgen.exporters.jsonl import _string_to_bits

        assert _string_to_bits("0101").tolist() == [False, True, False, True]
        with pytest.raises(ValueError, match="non-0/1"):
            _string_to_bits("01x1")

    def test_npz_suffix_is_normalised(self, tmp_path: Path) -> None:
        """numpy would append .npz silently, so the reported path would be wrong."""
        dataset = build_single_environment(distance=3, p=0.01, shots=20, seed=1, chunk_size=20)
        get_exporter("npz").write(dataset, tmp_path / "out", StructureLevel.NONE)
        assert (tmp_path / "out.npz").exists()


class TestOtherFindings:
    def test_estimate_crossing_matches_on_p_not_position(self) -> None:
        def est(d: int, p: float, errors: int) -> LogicalErrorEstimate:
            return LogicalErrorEstimate(d, p, 3, clopper_pearson(errors, 1000), 0.1)

        # Disjoint grids: there is no shared p, so no crossing can be claimed.
        mismatched = {
            3: [est(3, 0.001, 10), est(3, 0.02, 300)],
            5: [est(5, 0.05, 400), est(5, 0.06, 500)],
        }
        assert estimate_crossing(mismatched) is None

        # Shared grid where d=5 overtakes d=3 at the second point.
        aligned = {
            3: [est(3, 0.001, 100), est(3, 0.02, 200)],
            5: [est(5, 0.001, 10), est(5, 0.02, 400)],
        }
        assert estimate_crossing(aligned) == 0.02

    def test_zero_shots_does_not_crash(self) -> None:
        dataset = build_single_environment(distance=3, p=0.01, shots=0, seed=1, chunk_size=10)
        assert dataset.n_shots == 0
        assert dataset.detectors.shape == (0, 3)

    def test_zero_shot_multi_environment(self) -> None:
        dataset = build_multi_environment(
            distance=3, error_rates=[0.01], shots_per_env=0, seed=1, chunk_size=10
        )
        assert dataset.n_shots == 0

    def test_multi_env_rejects_incomparable_mechanism_indexing(self) -> None:
        """Matching counts are not enough; the target topology must agree too."""
        with pytest.raises(ValueError, match="mechanism"):
            build_multi_environment(
                distance=3,
                error_rates=[0.0, 0.01],
                shots_per_env=20,
                seed=1,
                chunk_size=20,
                emit_mechanisms=True,
            )


class TestW1GitCommitUnboundedSubprocess:
    """`DatasetMeta` could hang forever inside its own constructor.

    `git_commit` used `subprocess.run(capture_output=True, timeout=10)`. On a timeout
    that kills the child and then joins the pipe reader threads — and git spawns helpers
    that inherit the pipe handles, so those threads wait on an EOF that never arrives and
    the timeout stops bounding anything. Caught with py-spy on a generation run frozen
    at 1000 of 5000 shots, the whole stack sitting in `git_commit` under
    `DatasetMeta.__init__`. It affected `qecgen generate` too, not just the web UI.
    """

    def test_no_pipes_are_used_so_the_timeout_can_bound_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qecgen import dataset as dataset_module

        dataset_module._git_commit_cached.cache_clear()
        seen: dict[str, Any] = {}
        real = subprocess.run

        def capture(*args: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return real(*args, **kwargs)

        # qecgen.dataset resolves `subprocess.run` on the shared module object at call
        # time, so patching it here is what the generator will actually call.
        monkeypatch.setattr(subprocess, "run", capture)
        dataset_module.git_commit()
        dataset_module._git_commit_cached.cache_clear()

        assert "capture_output" not in seen, (
            "capture_output creates pipe reader threads that are joined after a timeout "
            "kill; a git grandchild holding the handle then blocks that join forever"
        )
        assert seen["stdout"] is not subprocess.PIPE
        assert seen["stderr"] is subprocess.DEVNULL
        assert seen["stdin"] is subprocess.DEVNULL, "git must never wait on a prompt"
        assert seen["timeout"] == 10

    def test_resolved_once_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A drift study builds one manifest per file; that was one git spawn each."""
        from qecgen import dataset as dataset_module

        dataset_module._git_commit_cached.cache_clear()
        calls: list[str | None] = []
        real = subprocess.run

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs.get("cwd"))
            return real(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", counting)
        first = dataset_module.git_commit()
        second = dataset_module.git_commit()
        assert first == second
        assert len(calls) == 1, f"expected one git call, got {len(calls)}"
        dataset_module._git_commit_cached.cache_clear()


class TestW2StagedWrites:
    """Front ends wrote straight to the path the user would later read.

    Every exporter truncates its target on open, and `StreamingHDF5Writer` opens the
    destination directly, so an interrupted run destroyed whatever was already there and
    left a file named exactly like a finished dataset.
    """

    def test_an_interrupted_write_does_not_destroy_the_existing_file(self, tmp_path: Path) -> None:
        from qecgen.run import GenerateSpec, staged
        from qecgen.run import run as run_spec

        target = tmp_path / "keep.h5"
        run_spec(GenerateSpec(distance=3, p=0.01, shots=100, seed=1, out=target, chunk_size=50))
        before = target.read_bytes()

        with pytest.raises(RuntimeError), staged(tmp_path) as staging:
            (staging.scratch / target.name).write_bytes(b"truncated")
            raise RuntimeError("interrupted")

        assert target.read_bytes() == before

    def test_a_cancelled_stream_leaves_no_file_at_the_target(self, tmp_path: Path) -> None:
        from qecgen.run import GenerateSpec, RunCancelledError
        from qecgen.run import run as run_spec

        target = tmp_path / "gone.h5"
        seen = {"n": 0}

        def stop(_shots: int) -> None:
            seen["n"] += 1
            if seen["n"] > 1:
                raise RunCancelledError("stop")

        with pytest.raises(RunCancelledError):
            run_spec(
                GenerateSpec(distance=3, p=0.01, shots=500, seed=1, out=target, chunk_size=100),
                progress=stop,
            )
        assert not target.exists()


class TestW3ProgressWasCosmetic:
    """The CLI filled its bar after the work, not during it.

    `cli.py` opened a progress bar, called `build_single_environment` as one opaque
    blocking call, then jumped the bar to 100%. `multi-env` and `drift` had no bar at all.
    """

    def test_materialising_builders_report_per_chunk(self) -> None:
        seen: list[int] = []
        build_single_environment(
            distance=3, p=0.01, shots=400, seed=1, chunk_size=100, progress=seen.append
        )
        assert seen == [100, 100, 100, 100]

    def test_drift_reports_across_every_file(self) -> None:
        seen: list[int] = []
        build_drift_environments(
            distance=3,
            train_p=0.005,
            test_values=[0.007],
            shots=100,
            seed=1,
            condition=DriftCondition.ORACLE_CALIBRATED,
            chunk_size=50,
            progress=seen.append,
        )
        assert sum(seen) == 200, "two files of 100 shots each"


class TestW4DriftCollisionCheckedTooLate:
    """The `%g` filename collision check ran after every dataset was already built."""

    def test_the_check_is_available_without_a_cli(self) -> None:
        from qecgen.environments import drift_dataset_names

        # Distinct floats that %g-collide; 0.01 vs 0.010 are the SAME float and pinned
        # duplicate rejection rather than the printed-name collision.
        with pytest.raises(ValueError, match="collide"):
            drift_dataset_names([0.010000001, 0.010000002])

    def test_it_refuses_before_sampling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qecgen import environments
        from qecgen.dataset import DriftCondition as Condition
        from qecgen.run import DriftSpec
        from qecgen.run import run as run_spec
        from qecgen.sampling import iter_chunks as real_iter_chunks

        calls = {"n": 0}

        def counting(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            return real_iter_chunks(*args, **kwargs)  # type: ignore[arg-type]

        # Counting sampler invocations asserts the property itself; a huge shot count
        # that "returns promptly" still passes if the check regressed to after the
        # build — just minutes later.
        monkeypatch.setattr(environments, "iter_chunks", counting)
        with pytest.raises(ValueError, match="collide"):
            run_spec(
                DriftSpec(
                    distance=3,
                    train_p=0.005,
                    test_values=(0.010000001, 0.010000002),
                    shots=100,
                    seed=0,
                    condition=Condition.FROZEN_PRIOR,
                    out=tmp_path / "d",
                )
            )
        assert calls["n"] == 0, "the collision check must run before any sampling"


class TestFullRepoReviewStagedCommit:
    """staged() committed with a bare os.replace loop.

    A destination file locked mid-commit left the earlier files committed while the
    cleanup then destroyed the rest of the staged set — the mixed old/new drift state
    ``generate_drift`` promises cannot exist.
    """

    def test_a_failed_commit_restores_the_previous_complete_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qecgen import run as run_mod

        dest = tmp_path / "d"
        dest.mkdir()
        (dest / "a.txt").write_text("old-a")
        (dest / "b.txt").write_text("old-b")

        real_replace = os.replace

        def failing(src: str | Path, dst: str | Path) -> None:
            commit_direction = Path(src).parent.name.startswith(run_mod.PARTIAL_PREFIX)
            if commit_direction and Path(dst).name == "b.txt":
                raise PermissionError("simulated: destination handle open")
            real_replace(src, dst)

        # Patching the os module itself patches the reference run.py resolves at call
        # time; run_mod.os is the same object but mypy rejects the unexported access.
        monkeypatch.setattr(os, "replace", failing)

        with pytest.raises(PermissionError), run_mod.staged(dest) as staging:
            (staging.scratch / "a.txt").write_text("new-a")
            (staging.scratch / "b.txt").write_text("new-b")

        assert (dest / "a.txt").read_text() == "old-a"
        assert (dest / "b.txt").read_text() == "old-b"
        assert list(dest.glob(f"{run_mod.PARTIAL_PREFIX}*")) == []
        assert list(dest.glob(f"{run_mod.DISPLACED_PREFIX}*")) == []

    def test_sweep_partials_skips_a_live_staging_directory(self, tmp_path: Path) -> None:
        from qecgen.run import staged, sweep_partials

        with staged(tmp_path / "data") as staging:
            (staging.scratch / "mid.h5").write_text("half-written")
            assert sweep_partials(tmp_path) == []
            assert staging.scratch.exists()
        assert sweep_partials(tmp_path) == []

    def test_sweep_partials_still_removes_orphans(self, tmp_path: Path) -> None:
        from qecgen.run import PARTIAL_PREFIX, sweep_partials

        orphan = tmp_path / "data" / f"{PARTIAL_PREFIX}deadbeef"
        orphan.mkdir(parents=True)
        (orphan / ".qecgen-lock").write_bytes(b"")  # creator died, so the lock is free
        assert sweep_partials(tmp_path) == [orphan]
        assert not orphan.exists()


class TestFullRepoReviewCrossingRule:
    """The crossing rule compared point estimates with ``>=``.

    Two statistically indistinguishable points -- including two zero-error points in a
    quick sweep -- tied on their point estimates, and the tie was reported as a threshold
    crossing at the lowest sampled p, contradicting the qa module's own "never point
    estimates" rule.
    """

    def test_two_zero_error_points_are_not_a_crossing(self) -> None:
        zero = clopper_pearson(0, 10_000)
        tied = {3: {0.001: zero}, 7: {0.001: zero}}
        assert crossing_from_intervals(tied) is None

    def test_a_demonstrated_reversal_still_reports(self) -> None:
        bracket = {
            3: {0.001: clopper_pearson(2_000, 100_000), 0.02: clopper_pearson(5_000, 100_000)},
            7: {0.001: clopper_pearson(500, 100_000), 0.02: clopper_pearson(9_000, 100_000)},
        }
        assert crossing_from_intervals(bracket) == 0.02

    def test_overlapping_intervals_mid_grid_claim_nothing(self) -> None:
        """The reversal must be demonstrated, not tied into existence."""
        overlap_mid = {
            3: {
                0.001: clopper_pearson(2_000, 100_000),
                0.01: clopper_pearson(100, 1_000),
                0.02: clopper_pearson(5_000, 100_000),
            },
            7: {
                0.001: clopper_pearson(500, 100_000),
                0.01: clopper_pearson(110, 1_000),
                0.02: clopper_pearson(9_000, 100_000),
            },
        }
        assert crossing_from_intervals(overlap_mid) == 0.02

    def test_reversal_without_prior_outperformance_is_not_a_crossing(self) -> None:
        """A decoder that is above threshold across the whole sampled range never showed
        the ordering, so there is nothing whose reversal could be bracketed."""
        above = {
            3: {0.005: clopper_pearson(500, 100_000)},
            7: {0.005: clopper_pearson(2_000, 100_000)},
        }
        assert crossing_from_intervals(above) is None


class TestFullRepoReviewCorrectionSemantics:
    """rec[-k] resolution, record counting and input validation in correction.py."""

    def test_mid_circuit_observable_include_raises_not_misattributes(self) -> None:
        """stim resolves rec[-k] at the instruction's position; the extractor resolved
        against the end-of-circuit total, silently mapping this observable (qubit 0's
        early record) onto qubit 2."""
        from qecgen.correction import extract_logical_operators

        circuit = stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            QUBIT_COORDS(2, 0) 2
            R 0 1 2
            M 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            X 1
            M 1 2
            """
        )
        with pytest.raises(ValueError, match="not in the final data layer"):
            extract_logical_operators(circuit)

    def test_every_record_producing_instruction_is_counted(self) -> None:
        """MPP before the final layer shifted every exported measurement_record by one
        under the old M/MR name-allowlist counting."""
        from qecgen.correction import build_correction_schema

        circuit = stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            QUBIT_COORDS(2, 0) 2
            MPP X0*X1
            M 0 1 2
            """
        )
        schema = build_correction_schema(circuit)
        assert [q.measurement_record for q in schema.data_qubits] == [1, 2, 3]

    def test_mpp_final_layer_gets_its_dedicated_error(self) -> None:
        """The docstring promised this diagnostic; the old in-layer check was
        unreachable because the layer walk breaks at MPP."""
        from qecgen.correction import build_correction_schema

        with pytest.raises(ValueError, match="MPP"):
            build_correction_schema(stim.Circuit("M 0\nMPP X1*X2"))

    def test_score_corrections_rejects_wrong_width_observables(self) -> None:
        """Passing the detectors array as observables scored 0.98 instead of raising."""
        from qecgen.correction import extract_logical_operators, score_corrections

        circuit, _ = build_circuit(3, 0.01)
        operators = extract_logical_operators(circuit)
        width = operators.schema.packed_width
        cx = np.zeros((5, width), dtype=np.uint8)
        cz = np.zeros((5, width), dtype=np.uint8)
        with pytest.raises(ValueError, match="observables has shape"):
            score_corrections(cx, cz, np.zeros((5, 3), dtype=np.uint8), operators)
        with pytest.raises(ValueError, match="packed uint8"):
            score_corrections(cx, cz, np.zeros((5, 1), dtype=bool), operators)

    def test_unpack_bits_rejects_wrong_width_input(self) -> None:
        """numpy's count= zero-fills past the end of an under-wide array (and reads
        uninitialized memory at width 0) instead of raising."""
        from qecgen.sampling import unpack_bits

        with pytest.raises(ValueError, match="expected"):
            unpack_bits(np.zeros((4, 2), dtype=np.uint8), 24)
        with pytest.raises(ValueError, match="expected"):
            unpack_bits(np.zeros((4, 0), dtype=np.uint8), 1)


class TestFullRepoReviewZeroShotStreaming:
    """Zero shots broke the streaming path three ways: digest, file, Contract B."""

    def test_streamed_zero_shot_file_reads_back_and_digests_match(self, tmp_path: Path) -> None:
        path = tmp_path / "zero.h5"
        meta = stream_single_environment(path=path, distance=3, p=0.01, shots=0, seed=1)
        built = build_single_environment(distance=3, p=0.01, shots=0, seed=1)
        assert meta.content_hash == built.meta.content_hash
        restored = get_exporter("hdf5").read(path)
        assert restored.detectors.shape == (0, 3)
        assert validate_dataset(restored).ok, str(validate_dataset(restored))

    def test_zero_shot_contract_b_still_carries_a_mechanisms_array(self, tmp_path: Path) -> None:
        built = build_single_environment(distance=3, p=0.01, shots=0, seed=1, emit_mechanisms=True)
        assert built.mechanisms is not None
        assert built.mechanisms.shape[0] == 0
        assert validate_dataset(built).ok, str(validate_dataset(built))
        streamed = stream_single_environment(
            path=tmp_path / "zero_mech.h5",
            distance=3,
            p=0.01,
            shots=0,
            seed=1,
            emit_mechanisms=True,
        )
        assert streamed.content_hash == built.meta.content_hash
        restored = get_exporter("hdf5").read(tmp_path / "zero_mech.h5")
        assert restored.mechanisms is not None


class TestFullRepoReviewManifestConsistency:
    def test_structure_source_is_none_at_level_none_on_every_path(self, tmp_path: Path) -> None:
        """_materialise stamped a source id even when no structure was exported, and
        only on some code paths, so identical runs disagreed by routing."""
        single = build_single_environment(distance=3, p=0.01, shots=10, seed=1, chunk_size=10)
        assert single.meta.structure_source_environment_id is None
        multi = build_multi_environment(
            distance=3, error_rates=[0.01], shots_per_env=10, seed=1, chunk_size=10
        )
        assert multi.meta.structure_source_environment_id is None
        drift = build_drift_environments(
            distance=3,
            train_p=0.005,
            test_values=[0.01],
            shots=10,
            seed=1,
            condition=DriftCondition.ORACLE_CALIBRATED,
            chunk_size=10,
            structure_level=StructureLevel.NONE,
        )
        assert all(d.meta.structure_source_environment_id is None for d in drift)
        streamed = stream_single_environment(
            path=tmp_path / "s.h5", distance=3, p=0.01, shots=10, seed=1, chunk_size=10
        )
        assert streamed.structure_source_environment_id is None

    def test_manifest_channels_must_be_complete(self) -> None:
        """ChannelVector(**partial) fills absent keys with 0.0, so a corrupt manifest
        read back as (partially) noiseless instead of failing."""
        dataset = build_single_environment(distance=3, p=0.01, shots=10, seed=1, chunk_size=10)
        payload = dataset.meta.to_json_dict()
        del payload["environments"][0]["channels"]["after_reset_flip_probability"]
        with pytest.raises(ValueError, match="missing"):
            DatasetMeta.from_json_dict(payload)

    def test_frozen_prior_mechanism_topology_mismatch_is_refused(self) -> None:
        """Equal counts in a different enumeration order would permute every label
        column against the shipped training structure; the old guard compared counts
        only."""
        from qecgen.environments import _assert_frozen_mechanisms_compatible

        train = stim.DetectorErrorModel("error(0.1) D0\nerror(0.1) D1 L0")
        reordered = stim.DetectorErrorModel("error(0.1) D1 L0\nerror(0.1) D0")
        with pytest.raises(ValueError, match="enumerate their mechanisms differently"):
            _assert_frozen_mechanisms_compatible(train, reordered)
        # Priors may differ freely -- that is what an environment is.
        _assert_frozen_mechanisms_compatible(
            train, stim.DetectorErrorModel("error(0.2) D0\nerror(0.2) D1 L0")
        )

    def test_unbiased_point_fails_closed(self) -> None:
        from qecgen.environments import unbiased_point

        with pytest.raises(ValueError, match="no unbiased point"):
            unbiased_point(DriftAxis.P)


class TestFullRepoReviewValidatorBlindSpots:
    def test_single_environment_shot_mismatch_fails_validation(self) -> None:
        dataset = build_single_environment(distance=3, p=0.01, shots=10, seed=1, chunk_size=10)
        doctored_env = dataclasses.replace(dataset.meta.environments[0], shots=99)
        doctored = dataclasses.replace(
            dataset, meta=dataclasses.replace(dataset.meta, environments=(doctored_env,))
        )
        report = validate_dataset(doctored, check_hash=False)
        assert any(
            r.name == "environment.shots_match_arrays" and not r.passed for r in report.results
        )

    def test_structure_mechanism_count_mismatch_fails_validation(self) -> None:
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=10,
            seed=1,
            chunk_size=10,
            emit_mechanisms=True,
            structure_level=StructureLevel.DEM,
        )
        doctored = dataclasses.replace(
            dataset, meta=dataclasses.replace(dataset.meta, n_mechanisms=7)
        )
        report = validate_dataset(doctored, check_hash=False)
        assert any(
            r.name == "structure.mechanisms_match_manifest" and not r.passed for r in report.results
        )

    def test_empty_components_beside_mechanisms_fails_validation(self) -> None:
        """The component checks were gated on `if structure.components:`, which skipped
        every_mechanism_has_a_component in the one state it was written for."""
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=10,
            seed=1,
            chunk_size=10,
            structure_level=StructureLevel.DEM,
        )
        assert dataset.structure is not None
        gutted = dataclasses.replace(dataset.structure, components=())
        report = validate_dem_structure(gutted)
        assert any(
            r.name == "dem.every_mechanism_has_a_component" and not r.passed for r in report.results
        )


class TestFullRepoReviewExporterHonesty:
    def test_exporters_refuse_a_manifest_that_overclaims_structure(self, tmp_path: Path) -> None:
        """HDF5 and NPZ silently wrote no structure when the dataset carried none while
        the manifest claimed a level; JSONL already refused."""
        dataset = build_single_environment(
            distance=3,
            p=0.01,
            shots=10,
            seed=1,
            chunk_size=10,
            structure_level=StructureLevel.DEM,
        )
        stripped = dataclasses.replace(dataset, structure=None)
        for name in ("hdf5", "npz", "jsonl"):
            with pytest.raises(ValueError, match="carries no"):
                get_exporter(name).write(stripped, tmp_path / f"x_{name}", StructureLevel.DEM)

    def test_jsonl_read_rejects_wrong_width_bit_strings(self, tmp_path: Path) -> None:
        """Uniformly short rows repacked with fabricated zero trailing bits whenever
        the byte count still matched; over-long rows were silently truncated."""
        dataset = build_single_environment(distance=3, p=0.01, shots=3, seed=1, chunk_size=3)
        path = tmp_path / "d.jsonl"
        get_exporter("jsonl").write(dataset, path, StructureLevel.NONE)
        doctored_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if isinstance(obj.get("detectors"), str):
                obj["detectors"] = obj["detectors"][:-1]
                doctored_lines.append(json.dumps(obj))
            else:
                doctored_lines.append(line)
        path.write_text("\n".join(doctored_lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="refusing to zero-fill"):
            get_exporter("jsonl").read(path)


def test_detection_event_rate_zero_shots_is_zero_not_a_crash() -> None:
    """`min(shots, 50_000)` handed chunk_size=0 to iter_chunks, which raises."""
    circuit, _ = build_circuit(3, 0.01)
    assert detection_event_rate(circuit, shots=0, seed=1) == 0.0


def test_full_suite_still_produces_valid_files(tmp_path: Path) -> None:
    """End-to-end guard: every format writes something the validator accepts."""
    dataset = build_multi_environment(
        distance=3,
        error_rates=[0.005, 0.01],
        shots_per_env=40,
        seed=3,
        chunk_size=40,
        structure_level=StructureLevel.DEM,
    )
    for name, exporter in sorted(EXPORTERS.items()):
        path = tmp_path / f"all_{name}{exporter.extension}"
        exporter.write(dataset, path, StructureLevel.DEM)
        restored = exporter.read(path)
        assert validate_dataset(restored).ok, f"{name}: {validate_dataset(restored)}"
