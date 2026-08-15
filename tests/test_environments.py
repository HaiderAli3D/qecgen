"""Multi-environment pooling, shuffling and the drift conditions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qecgen.circuits import NoiseModel
from qecgen.dataset import Contract, DriftCondition, InMemoryDataset, StructureLevel
from qecgen.environments import (
    AXIS_BUILDERS,
    DriftAxis,
    _validate_axis_value,
    build_drift_environments,
    build_environment,
    build_multi_environment,
    build_single_environment,
    derive_seeds,
    unbiased_point,
)
from qecgen.validate import validate_dataset

RATES = [0.004, 0.009, 0.013]


def _pooled(**overrides: Any) -> InMemoryDataset:
    """Build the standard pooled fixture, with per-test overrides."""
    params: dict[str, Any] = {
        "distance": 3,
        "error_rates": RATES,
        "shots_per_env": 150,
        "seed": 2024,
        "chunk_size": 64,
    }
    params.update(overrides)
    return build_multi_environment(**params)


def _row_keys(ds: InMemoryDataset) -> list[tuple[bytes, ...]]:
    """Full-row identity, so a shuffle that detaches labels from data is detectable."""
    keys: list[tuple[bytes, ...]] = []
    for i in range(ds.n_shots):
        row = [bytes(ds.detectors[i]), bytes(ds.observables[i])]
        if ds.environment_ids is not None:
            row.append(str(int(ds.environment_ids[i])).encode())
        if ds.mechanisms is not None:
            row.append(bytes(ds.mechanisms[i]))
        keys.append(tuple(row))
    return keys


class TestShuffle:
    def test_shuffle_preserves_per_shot_correspondence(self) -> None:
        """The whole row must move together, or labels detach from data."""
        ordered = _pooled(shuffle=False, emit_mechanisms=True)
        shuffled = _pooled(shuffle=True, emit_mechanisms=True)
        assert sorted(_row_keys(ordered)) == sorted(_row_keys(shuffled))

    def test_shuffle_actually_reorders(self) -> None:
        ordered = _pooled(shuffle=False)
        shuffled = _pooled(shuffle=True)
        assert _row_keys(ordered) != _row_keys(shuffled)

    def test_shuffle_is_reproducible(self) -> None:
        a = _pooled(shuffle=True)
        b = _pooled(shuffle=True)
        assert a.meta.content_hash == b.meta.content_hash
        assert a.environment_ids is not None
        assert b.environment_ids is not None
        assert np.array_equal(a.environment_ids, b.environment_ids)

    def test_environment_counts_survive_shuffling(self) -> None:
        dataset = _pooled(shuffle=True)
        assert dataset.environment_ids is not None
        _, counts = np.unique(dataset.environment_ids, return_counts=True)
        assert set(counts) == {150}

    def test_shuffle_seed_is_recorded(self) -> None:
        dataset = _pooled(shuffle=True)
        assert dataset.meta.shuffle_seed is not None
        assert _pooled(shuffle=False).meta.shuffle_seed is None


class TestManifestShape:
    def test_no_top_level_p_or_circuit(self) -> None:
        """p, circuit and dem are per-environment properties only."""
        dataset = _pooled()
        payload = dataset.meta.to_json_dict()
        assert "p" not in payload
        assert "circuit" not in payload
        assert "dem" not in payload
        assert len(payload["environments"]) == len(RATES)

    def test_single_environment_still_uses_a_list(self) -> None:
        dataset = build_single_environment(distance=3, p=0.01, shots=64, seed=1, chunk_size=64)
        assert len(dataset.meta.environments) == 1
        assert dataset.environment_ids is None

    def test_each_environment_stores_its_channel_vector(self) -> None:
        dataset = _pooled()
        for env, rate in zip(dataset.meta.environments, RATES, strict=True):
            assert env.p == rate
            assert env.channels.after_clifford_depolarization == rate

    def test_environments_carry_circuit_and_dem_text(self) -> None:
        dataset = _pooled()
        for env in dataset.meta.environments:
            assert env.circuit.strip()
            assert env.dem.strip()


class TestStackingGuards:
    def test_rejects_environments_with_different_detector_counts(self) -> None:
        """Different rounds change n_detectors and must fail loudly."""
        with pytest.raises(ValueError, match="detector and observable counts"):
            from qecgen.environments import _assert_stackable

            builds = [
                build_environment(0, 3, 0.01, DriftAxis.P, 0.01, 10, rounds=3),
                build_environment(1, 3, 0.01, DriftAxis.P, 0.01, 10, rounds=5),
            ]
            _assert_stackable(builds)

    def test_non_p_axis_requires_base_p(self) -> None:
        with pytest.raises(ValueError, match="requires base_p"):
            build_multi_environment(
                distance=3,
                error_rates=[1.0, 2.0],
                shots_per_env=10,
                seed=1,
                axis=DriftAxis.MEASUREMENT_RATIO,
            )

    def test_empty_rates_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            build_multi_environment(distance=3, error_rates=[], shots_per_env=10, seed=1)


class TestAxes:
    def test_registry_covers_every_axis(self) -> None:
        """All three per-axis extension points, not just the builder registry.

        CLAUDE.md names three things a new axis must provide: a builder, a domain check
        in `_validate_axis_value` and an unbiased point. Checking only AXIS_BUILDERS let
        the other two fall through — previously to a silent 0.0 unbiased point and an
        unchecked value domain; both now fail closed, and this probes each so the
        failure surfaces here rather than in the first drift run on the new axis.
        """
        assert set(AXIS_BUILDERS) == set(DriftAxis)
        probe_values = {
            DriftAxis.P: 0.005,
            DriftAxis.XZ_BIAS: 0.5,
            DriftAxis.MEASUREMENT_RATIO: 1.0,
        }
        assert set(probe_values) == set(DriftAxis), "add the new axis to this probe map"
        for axis, value in probe_values.items():
            _validate_axis_value(axis, 0.005, value)
            if axis is not DriftAxis.P:
                assert math.isfinite(unbiased_point(axis))

    def test_measurement_ratio_scales_only_measurement(self) -> None:
        build = build_environment(
            0,
            3,
            0.01,
            DriftAxis.MEASUREMENT_RATIO,
            5.0,
            10,
            noise_model=NoiseModel.STIM_UNIFORM_CIRCUIT_LEVEL,
        )
        channels = build.spec.channels
        assert channels.before_measure_flip_probability == pytest.approx(0.05)
        assert channels.after_clifford_depolarization == pytest.approx(0.01)

    def test_xz_bias_rewrites_the_circuit(self) -> None:
        build = build_environment(0, 3, 0.01, DriftAxis.XZ_BIAS, 12.0, 10)
        assert "PAULI_CHANNEL_1" in build.spec.circuit
        assert "DEPOLARIZE1" not in build.spec.circuit

    def test_xz_bias_environments_pool(self) -> None:
        dataset = build_multi_environment(
            distance=3,
            error_rates=[0.5, 4.0, 16.0],
            shots_per_env=60,
            seed=9,
            chunk_size=60,
            axis=DriftAxis.XZ_BIAS,
            base_p=0.008,
        )
        assert dataset.n_shots == 180
        assert [e.axis for e in dataset.meta.environments] == ["xz_bias"] * 3
        assert [e.axis_value for e in dataset.meta.environments] == [0.5, 4.0, 16.0]
        assert validate_dataset(dataset).ok


class TestDrift:
    def test_frozen_prior_ships_training_structure(self) -> None:
        datasets = build_drift_environments(
            distance=3,
            train_p=0.005,
            test_values=[0.010, 0.014],
            shots=64,
            seed=1,
            condition=DriftCondition.FROZEN_PRIOR,
            chunk_size=64,
        )
        train, *tests = datasets
        assert train.structure is not None
        for test in tests:
            assert test.structure is not None
            assert test.meta.drift_condition is DriftCondition.FROZEN_PRIOR
            assert test.meta.structure_source_environment_id == 0
            assert np.allclose(test.structure.priors, train.structure.priors)

    def test_oracle_calibrated_ships_own_structure(self) -> None:
        datasets = build_drift_environments(
            distance=3,
            train_p=0.005,
            test_values=[0.014],
            shots=64,
            seed=1,
            condition=DriftCondition.ORACLE_CALIBRATED,
            chunk_size=64,
        )
        train, test = datasets
        assert train.structure is not None
        assert test.structure is not None
        assert test.meta.drift_condition is DriftCondition.ORACLE_CALIBRATED
        assert test.meta.structure_source_environment_id == 1
        assert not np.allclose(test.structure.priors, train.structure.priors)

    def test_conditions_differ_only_in_structure_not_shots(self) -> None:
        """The samples are identical; only the exported prior changes."""
        kwargs: dict[str, Any] = {
            "distance": 3,
            "train_p": 0.005,
            "test_values": [0.014],
            "shots": 64,
            "seed": 4,
            "chunk_size": 64,
        }
        frozen = build_drift_environments(condition=DriftCondition.FROZEN_PRIOR, **kwargs)
        oracle = build_drift_environments(condition=DriftCondition.ORACLE_CALIBRATED, **kwargs)
        assert np.array_equal(frozen[1].detectors, oracle[1].detectors)
        assert np.array_equal(frozen[1].observables, oracle[1].observables)


class TestSeeds:
    def test_derive_seeds_is_deterministic(self) -> None:
        assert derive_seeds(123, 4) == derive_seeds(123, 4)

    def test_derive_seeds_are_distinct(self) -> None:
        seeds = derive_seeds(123, 8)
        assert len(set(seeds)) == 8

    def test_different_master_seed_gives_different_children(self) -> None:
        assert derive_seeds(1, 4) != derive_seeds(2, 4)


def test_contract_is_recorded_from_emit_mechanisms() -> None:
    plain = _pooled()
    labelled = _pooled(emit_mechanisms=True)
    assert plain.meta.contract is Contract.LOGICAL_FRAME
    assert plain.mechanisms is None
    assert labelled.meta.contract is Contract.DEM_MECHANISM
    assert labelled.mechanisms is not None


def test_pooled_dataset_validates() -> None:
    dataset = _pooled(structure_level=StructureLevel.DEM)
    report = validate_dataset(dataset)
    assert report.ok, str(report)
